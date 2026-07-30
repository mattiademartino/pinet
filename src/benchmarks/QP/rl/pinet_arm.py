"""The pinet arms: same loss, true gradient, deterministic or sampled action.

These two arms are *not* reinforcement learning. They minimise the loss directly
and differentiate through the projection with its implicit-function-theorem VJP,
exactly as the pinet baseline does. They are run through the same driver as the
RL agents only so that every curve in the comparison comes out of the same
evaluation protocol, on the same budget axes.

- ``pinet``: the network outputs a point, which is projected. This is the usual
  baseline, re-expressed as one iteration per sampled batch of contexts.
- ``pinet_stochastic``: the network outputs a mean *and* a standard deviation,
  an action is sampled, and the loss is evaluated at the projection of that
  sample. The gradient is pathwise (reparameterisation), so it flows through the
  projection Jacobian.

The second arm is the bridge between the two worlds: it optimises the same
Gaussian-smoothed objective as the RL agents,

``J_sigma(theta) = E_x E_eps [ L(P(mu(x) + sigma(x) * eps)) ]``

but estimates its gradient pathwise instead of with the score function. Comparing
it with SAC and REINFORCE isolates the estimator from the objective, and
comparing it with ``pinet`` isolates the effect of the sampling itself.

Nothing pushes the standard deviation up: with no entropy bonus the pathwise
gradient is free to shrink it, and on a convex objective it generally will --
whether it does, and how fast, is one of the things the curves show.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import optax

from .common import periodic_evaluation
from .policy import GaussianPolicy, gaussian_entropy


def _build(env, config: dict, key: jax.Array) -> tuple:
    """Set up the network, the optimiser and the differentiable projection.

    Args:
        env (QPBanditEnv): The environment, used for its projection and loss.
        config (dict): Merged configuration.
        key (jax.Array): Random key for the initialisation.

    Returns:
        tuple: ``(policy, params, tx, opt_state, project)``.
    """
    activation = getattr(jax.nn, config["activation"])
    policy = GaussianPolicy(
        dim=env.dim,
        features_list=config["features_list"],
        activation=activation,
        state_dependent_std=config["state_dependent_std"],
        log_std_init=config["log_std_init"],
        log_std_min=config["log_std_min"],
        log_std_max=config["log_std_max"],
    )
    params = {"policy": policy.init(key, jnp.zeros((2, env.n_eq)))["params"]}
    tx = optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.adam(config["learning_rate"]),
    )
    return policy, params, tx, tx.init(params), env.differentiable_projection()


def _run(
    key: jax.Array,
    env,
    sampler: Callable,
    evaluate: Callable,
    config: dict,
    history,
    stochastic: bool,
) -> dict:
    """Train one pinet arm.

    Args:
        key (jax.Array): Random key.
        env (QPBanditEnv): The environment.
        sampler (Callable): Context sampler.
        evaluate (Callable): Deterministic evaluation of the parameters.
        config (dict): Merged configuration.
        history (History): Bookkeeping object, updated in place.
        stochastic (bool): Whether to sample the action before projecting.

    Returns:
        dict: The trained parameters.
    """
    key, key_init = jax.random.split(key)
    policy, params, tx, opt_state, project = _build(env, config, key_init)
    tag = "[pinet_stochastic]" if stochastic else "[pinet]"
    ent_coef = config.get("ent_coef", 0.0)

    @jax.jit
    def update(params: dict, opt_state, key: jax.Array, x, b) -> tuple:
        """One optimisation step on the projected loss."""

        def loss_fn(p):
            mean, log_std = policy.apply({"params": p["policy"]}, x)
            if stochastic:
                # Reparameterised sample: the gradient reaches both the mean and
                # the scale through the projection.
                noise = jax.random.normal(key, mean.shape)
                y = mean + jnp.exp(log_std) * noise
            else:
                y = mean
            y_proj = project(y, b)
            loss = env.batched_loss(y_proj, b).mean()
            if stochastic and ent_coef:
                loss = loss - ent_coef * gaussian_entropy(log_std).mean()
            return loss, (y_proj, log_std)

        (loss, (y_proj, log_std)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params
        )
        updates, opt_state = tx.update(grads, opt_state, params)
        return (
            optax.apply_updates(params, updates),
            opt_state,
            loss,
            optax.global_norm(grads),
            env.ineq_cv(y_proj).mean(),
            log_std.mean(),
        )

    for iteration in range(config["n_iterations"]):
        history.start_timer()
        key, key_context, key_action = jax.random.split(key, 3)
        x, b = sampler(key_context, config["context_batch"])
        params, opt_state, loss, grad_norm, ineq_cv, log_std = update(
            params, opt_state, key_action, x, b
        )
        jax.block_until_ready(params["policy"])
        history.stop_timer()
        history.env_samples += config["context_batch"]
        history.gradient_steps += 1

        if iteration % config["log_every"] == 0:
            history.log_iteration(
                iteration=iteration,
                env_samples=history.env_samples,
                train_time=history.train_time,
                # The reward of the RL arms is the negative of this loss, so the
                # two are directly comparable on the same axis.
                reward_mean=-loss,
                loss=loss,
                grad_norm=grad_norm,
                ineq_cv_mean=ineq_cv,
                sampling_log_std=log_std,
            )
        periodic_evaluation(history, evaluate, params, iteration, config, tag)
    return params


def train_deterministic(
    key: jax.Array, env, sampler: Callable, evaluate: Callable, config: dict, history
) -> dict:
    """Train the deterministic pinet arm.

    Args:
        key (jax.Array): Random key.
        env (QPBanditEnv): The environment.
        sampler (Callable): Context sampler.
        evaluate (Callable): Deterministic evaluation of the parameters.
        config (dict): Merged configuration.
        history (History): Bookkeeping object, updated in place.

    Returns:
        dict: The trained parameters.
    """
    return _run(key, env, sampler, evaluate, config, history, stochastic=False)


def train_stochastic(
    key: jax.Array, env, sampler: Callable, evaluate: Callable, config: dict, history
) -> dict:
    """Train the pinet arm with a sampled action.

    Args:
        key (jax.Array): Random key.
        env (QPBanditEnv): The environment.
        sampler (Callable): Context sampler.
        evaluate (Callable): Deterministic evaluation of the parameters.
        config (dict): Merged configuration.
        history (History): Bookkeeping object, updated in place.

    Returns:
        dict: The trained parameters.
    """
    return _run(key, env, sampler, evaluate, config, history, stochastic=True)
