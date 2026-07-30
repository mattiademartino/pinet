"""Shared utilities for the RL baselines: contexts, evaluation and bookkeeping.

Evaluation deliberately mirrors ``run_grad_ablation.evaluate`` so that the
numbers land in the same units as the pinet baselines: relative suboptimality
against the dataset optimum, and equality/inequality violations after the
test-time projection.
"""

import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


def split_arrays(loader) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Materialise a whole split from a loader that yields it in one batch.

    Args:
        loader: Data loader over one split.

    Returns:
        tuple: ``(X, objectives)`` with the contexts of shape (N, n_eq, 1) and the
            optimal objective values of shape (N,).
    """
    for X, objectives in loader:
        pass
    return X, jnp.asarray(objectives).ravel()


def make_context_sampler(X_train: jnp.ndarray) -> Callable:
    """Build an i.i.d. sampler of contexts from the training set.

    The bandit formulation draws a fresh context per interaction, so contexts are
    sampled with replacement rather than iterated in epochs.

    Args:
        X_train (jnp.ndarray): Training contexts, shape (N, n_eq, 1).

    Returns:
        Callable: ``sample(key, batch) -> (x, b)`` with ``x`` of shape
            (batch, n_eq) and ``b`` of shape (batch, n_eq, 1).
    """
    n_train = X_train.shape[0]

    def sample(key: jax.Array, batch: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Draw a batch of contexts."""
        idx = jax.random.randint(key, (batch,), 0, n_train)
        b = X_train[idx]
        return b[:, :, 0], b

    return sample


def make_evaluator(env, policy, X_eval: jnp.ndarray, obj_eval: jnp.ndarray) -> Callable:
    """Build the deterministic evaluation of a policy on a fixed split.

    The evaluated action is the mean of the policy, projected with the test-time
    iteration budget -- exactly what the pinet baseline reports.

    Args:
        env (QPBanditEnv): The environment.
        policy (GaussianPolicy): The policy module.
        X_eval (jnp.ndarray): Contexts of the split, shape (N, n_eq, 1).
        obj_eval (jnp.ndarray): Optimal objectives of the split, shape (N,).

    Returns:
        Callable: ``evaluate(params) -> dict``, where ``params`` is the full
            parameter dict of the agent, holding the policy under ``"policy"``.
    """

    @jax.jit
    def evaluate(params: dict) -> dict:
        """Evaluate the deterministic policy."""
        mean, log_std = policy.apply({"params": params["policy"]}, X_eval[:, :, 0])
        _, info = env.step_eval(mean, X_eval)
        rs = (info["objective"] - obj_eval) / jnp.abs(obj_eval)
        return {
            "rs": rs,
            "objective": info["objective"],
            "eq_cv": info["eq_cv"],
            "ineq_cv": info["ineq_cv"],
            "log_std": log_std.mean(),
            "action_norm": jnp.linalg.norm(mean, axis=1).mean(),
        }

    def evaluate_numpy(params: dict) -> dict:
        """Evaluate and summarise into scalars plus the raw suboptimality."""
        out = jax.block_until_ready(evaluate(params))
        return {
            "rs_mean": float(jnp.mean(out["rs"])),
            "rs_median": float(jnp.median(out["rs"])),
            "objective_mean": float(jnp.mean(out["objective"])),
            "eq_cv_max": float(jnp.max(out["eq_cv"])),
            "ineq_cv_mean": float(jnp.mean(out["ineq_cv"])),
            "ineq_cv_max": float(jnp.max(out["ineq_cv"])),
            "log_std_mean": float(out["log_std"]),
            "action_norm_mean": float(out["action_norm"]),
            "rs": np.asarray(out["rs"]),
            "ineq_cv": np.asarray(out["ineq_cv"]),
            "eq_cv": np.asarray(out["eq_cv"]),
        }

    return evaluate_numpy


def periodic_evaluation(
    history, evaluate: Callable, params: dict, iteration: int, config: dict, tag: str
) -> None:
    """Evaluate, record and print, if the iteration is an evaluation point.

    Evaluation happens off the clock: the timer is stopped by the caller before
    this function runs, so the reported training time is comparable with the
    pinet baselines.

    Args:
        history (History): Bookkeeping object, updated in place.
        evaluate (Callable): Deterministic evaluation of the parameters.
        params (dict): Current parameters.
        iteration (int): Current iteration index.
        config (dict): Configuration, read for ``eval_every`` and ``n_iterations``.
        tag (str): Prefix for the printed line.
    """
    last = iteration == config["n_iterations"] - 1
    if iteration % config["eval_every"] and not last:
        return
    metrics = evaluate(params)
    history.log_evaluation(
        iteration=iteration,
        env_samples=history.env_samples,
        train_time=history.train_time,
        **metrics,
    )
    print(
        f"{tag} it {iteration + 1}/{config['n_iterations']} "
        f"samples={history.env_samples} t={history.train_time:.0f}s "
        f"rs={metrics['rs_mean']:.4f} "
        f"obj={metrics['objective_mean']:.4f} "
        f"ineqcv={metrics['ineq_cv_max']:.2e} "
        f"log_std={metrics['log_std_mean']:.3f}",
        flush=True,
    )


class History:
    """Collects per-iteration scalars and the evaluation curve."""

    def __init__(self) -> None:
        """Initialize the empty history."""
        self.iterations: dict[str, list] = {}
        self.evaluations: dict[str, list] = {}
        self.env_samples = 0
        self.gradient_steps = 0
        self.train_time = 0.0
        self._start = None

    def start_timer(self) -> None:
        """Start timing a training section, excluding evaluation."""
        self._start = time.time()

    def stop_timer(self) -> float:
        """Stop timing and accumulate the elapsed training time.

        Returns:
            float: Seconds elapsed since :meth:`start_timer`.
        """
        elapsed = time.time() - self._start
        self.train_time += elapsed
        return elapsed

    def log_iteration(self, **values) -> None:
        """Append one row of training scalars.

        Args:
            **values: Named scalars to record.
        """
        for name, value in values.items():
            self.iterations.setdefault(name, []).append(float(value))

    def log_evaluation(self, **values) -> None:
        """Append one row of evaluation scalars.

        Args:
            **values: Named scalars to record; arrays are skipped.
        """
        for name, value in values.items():
            if np.ndim(value) == 0:
                self.evaluations.setdefault(name, []).append(float(value))

    def as_payload(self) -> dict:
        """Return the history as a flat dict of arrays, ready for ``savez``.

        Returns:
            dict: Arrays keyed by ``"iter/<name>"`` and ``"eval/<name>"``.
        """
        payload = {f"iter/{k}": np.array(v) for k, v in self.iterations.items()}
        payload.update({f"eval/{k}": np.array(v) for k, v in self.evaluations.items()})
        payload["train_time"] = np.array(self.train_time)
        payload["env_samples"] = np.array(self.env_samples)
        payload["gradient_steps"] = np.array(self.gradient_steps)
        return payload
