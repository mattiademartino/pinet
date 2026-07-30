"""Contextual-bandit environment around the parametric QP.

The environment is a one-step (bandit) MDP:

- the *context* is the right-hand side ``x`` of the equality constraints, drawn
  from the dataset;
- the *action* is a point ``y`` in the full decision space, produced by the
  policy without any constraint-awareness;
- the environment projects the action onto the feasible set and returns the
  reward ``-L(P(y))``.

The projection lives entirely inside the environment and is wrapped in
``jax.lax.stop_gradient``, so no gradient can flow through it by construction:
the agent only ever sees a scalar reward. This is what makes the comparison with
the self-supervised pinet training meaningful -- the agent has access to neither
the objective gradient nor the projection Jacobian.
"""

from typing import Callable

import jax
import jax.numpy as jnp

from pinet import (
    AffineInequalityConstraint,
    EqualityConstraint,
    EqualityConstraintsSpecification,
    EquilibrationParams,
    Project,
    ProjectionInstance,
)


class QPBanditEnv:
    """One-step environment that projects the action and returns ``-L(P(y))``."""

    def __init__(
        self,
        A: jnp.ndarray,
        G: jnp.ndarray,
        h: jnp.ndarray,
        X: jnp.ndarray,
        batched_objective: Callable[[jnp.ndarray], jnp.ndarray],
        batched_loss: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        hyperparameters: dict,
    ) -> None:
        """Initialize the environment.

        Args:
            A (jnp.ndarray): Equality constraint matrix, shape (1, n_eq, dim).
            G (jnp.ndarray): Inequality constraint matrix, shape (1, n_ineq, dim).
            h (jnp.ndarray): Inequality upper bounds, shape (1, n_ineq, 1).
            X (jnp.ndarray): Dataset of right-hand sides, shape (N, n_eq, 1). Only
                used to instantiate the equality constraint; the actual right-hand
                side is passed per call.
            batched_objective (Callable): Batched objective, used for reporting.
            batched_loss (Callable): Batched training loss, used for the reward.
                It coincides with the objective when no penalty is configured, so
                the agent optimises exactly the pinet training loss.
            hyperparameters (dict): Configuration of the projection layer. The same
                file used by the pinet baseline, so that the forward projection is
                bit-identical.
        """
        self.A = A
        self.G = G
        self.h = h
        self.dim = A.shape[2]
        self.n_eq = A.shape[1]
        self.batched_objective = batched_objective
        self.batched_loss = batched_loss

        eq_constraint = EqualityConstraint(A=A, b=X, method=None, var_b=True)
        ineq_constraint = AffineInequalityConstraint(
            C=G, ub=h, lb=-jnp.inf * jnp.ones_like(h)
        )
        self.projection_layer = Project(
            eq_constraint=eq_constraint,
            ineq_constraint=ineq_constraint,
            unroll=False,
            equilibration_params=EquilibrationParams(**hyperparameters["equilibrate"]),
        )
        self._sigma = hyperparameters["sigma"]
        self._omega = hyperparameters["omega"]
        self._kw = {
            "n_iter_bwd": hyperparameters["n_iter_bwd"],
            "fpi": hyperparameters["fpi"],
        }
        self.n_iter = hyperparameters["n_iter_train"]
        self.n_iter_eval = hyperparameters["n_iter_test"]

        self.project = jax.jit(lambda y, b: self._project(y, b, self.n_iter))
        self.project_eval = jax.jit(lambda y, b: self._project(y, b, self.n_iter_eval))
        self.step = jax.jit(lambda y, b: self._step(y, b, self.project))
        self.step_eval = jax.jit(lambda y, b: self._step(y, b, self.project_eval))

    def _project_raw(self, y: jnp.ndarray, b: jnp.ndarray, n_iter: int) -> jnp.ndarray:
        """Project a batch of points, keeping the layer's implicit VJP.

        Args:
            y (jnp.ndarray): Points, shape (batch, dim).
            b (jnp.ndarray): Contexts, shape (batch, n_eq, 1).
            n_iter (int): Number of Douglas-Rachford iterations.

        Returns:
            jnp.ndarray: Projected points, shape (batch, dim).
        """
        inp = ProjectionInstance(
            x=y[..., None], eq=EqualityConstraintsSpecification(b=b)
        )
        return self.projection_layer.call(
            yraw=inp,
            sigma=self._sigma,
            omega=self._omega,
            n_iter=n_iter,
            **self._kw,
        )[0].x[..., 0]

    def _project(self, y: jnp.ndarray, b: jnp.ndarray, n_iter: int) -> jnp.ndarray:
        """Project a batch of actions, detaching them from the autodiff graph.

        Args:
            y (jnp.ndarray): Actions, shape (batch, dim).
            b (jnp.ndarray): Contexts, shape (batch, n_eq, 1).
            n_iter (int): Number of Douglas-Rachford iterations.

        Returns:
            jnp.ndarray: Projected actions, shape (batch, dim), detached.
        """
        # The environment is a black box: nothing differentiates through it.
        return jax.lax.stop_gradient(self._project_raw(y, b, n_iter))

    def differentiable_projection(self) -> Callable:
        """Return the projection *with* its implicit-function-theorem gradient.

        This deliberately bypasses the black-box contract of the environment. Two
        kinds of arm call it: the pinet arms, which are not reinforcement learning
        and differentiate the loss through the projection exactly as the baseline
        does; and ``sac_proj``, whose critic is defined on the feasible set, so
        its actor has to reach the critic through the projection. The other RL
        agents never call this, and see the environment as a black box.

        Returns:
            Callable: ``project(y, b)`` differentiable in ``y``, using the same
                iteration budget as the environment step.
        """
        return jax.jit(lambda y, b: self._project_raw(y, b, self.n_iter))

    def _step(self, y: jnp.ndarray, b: jnp.ndarray, project: Callable) -> tuple:
        """Take one environment step.

        Args:
            y (jnp.ndarray): Actions, shape (batch, dim).
            b (jnp.ndarray): Contexts, shape (batch, n_eq, 1).
            project (Callable): Projection to apply to the actions.

        Returns:
            tuple: ``(reward, info)`` with the reward of shape (batch,) and a dict
                holding the projected action, the objective and the constraint
                violations.
        """
        y_proj = project(y, b)
        reward = -self.batched_loss(y_proj, b).ravel()
        return reward, {
            "y_proj": y_proj,
            "objective": self.batched_objective(y_proj).ravel(),
            "eq_cv": self.eq_cv(y_proj, b),
            "ineq_cv": self.ineq_cv(y_proj),
        }

    def eq_cv(self, y: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Maximum equality-constraint violation per instance.

        Args:
            y (jnp.ndarray): Points, shape (batch, dim).
            b (jnp.ndarray): Contexts, shape (batch, n_eq, 1).

        Returns:
            jnp.ndarray: Violations, shape (batch,).
        """
        batch = y.shape[0]
        residual = self.A[0].reshape(1, self.n_eq, self.dim) @ y.reshape(
            batch, self.dim, 1
        )
        return jnp.max(jnp.abs(residual - b), axis=1).ravel()

    def ineq_cv(self, y: jnp.ndarray) -> jnp.ndarray:
        """Maximum inequality-constraint violation per instance.

        Args:
            y (jnp.ndarray): Points, shape (batch, dim).

        Returns:
            jnp.ndarray: Violations, shape (batch,).
        """
        batch = y.shape[0]
        residual = self.G[0].reshape(1, self.G.shape[1], self.dim) @ y.reshape(
            batch, self.dim, 1
        )
        return jnp.max(jnp.maximum(residual - self.h, 0), axis=1).ravel()
