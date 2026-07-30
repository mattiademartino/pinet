"""REINFORCE with a learned state-value baseline, on the one-step QP bandit.

With a single-step episode the policy gradient reduces to the score-function
estimator

``grad J = E[ (r - V(x)) * grad log pi(y | x) ]``

where ``r = -L(P(y))`` is returned by the environment. Nothing differentiates
through the projection, which is why this arm is a clean measurement of the cost
of giving up first-order information about the objective.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import optax

from .common import periodic_evaluation
from .policy import (
    GaussianPolicy,
    ValueNetwork,
    gaussian_entropy,
    gaussian_log_prob,
    gaussian_sample,
)

# Guard for the normalisation of the advantages.
EPS = 1e-8


def train(
    key: jax.Array,
    env,
    sampler: Callable,
    evaluate: Callable,
    config: dict,
    history,
) -> dict:
    """Train a policy with REINFORCE.

    Args:
        key (jax.Array): Random key.
        env (QPBanditEnv): The environment.
        sampler (Callable): Context sampler, see ``common.make_context_sampler``.
        evaluate (Callable): Deterministic evaluation of the policy parameters.
        config (dict): Merged RL configuration (shared keys plus the
            ``reinforce`` section).
        history (History): Bookkeeping object, updated in place.

    Returns:
        dict: The trained parameters.
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
    value = ValueNetwork(features_list=config["features_list"], activation=activation)

    key, key_policy, key_value = jax.random.split(key, 3)
    x_example = jnp.zeros((2, env.n_eq))
    params = {
        "policy": policy.init(key_policy, x_example)["params"],
        "value": value.init(key_value, x_example)["params"],
    }
    tx = optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.adam(config["learning_rate"]),
    )
    opt_state = tx.init(params)

    batch = config["context_batch"]
    ent_coef = config["ent_coef"]
    value_coef = config["value_coef"]

    @jax.jit
    def act(params: dict, key: jax.Array, x: jnp.ndarray) -> tuple:
        """Sample an action for every context."""
        mean, log_std = policy.apply({"params": params["policy"]}, x)
        actions = gaussian_sample(key, mean, log_std)
        return actions, gaussian_log_prob(actions, mean, log_std)

    @jax.jit
    def baseline(params: dict, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the state-value baseline."""
        return value.apply({"params": params["value"]}, x)

    def policy_loss(params: dict, x, actions, advantages) -> jnp.ndarray:
        """Score-function surrogate whose gradient is the policy gradient."""
        mean, log_std = policy.apply({"params": params["policy"]}, x)
        log_prob = gaussian_log_prob(actions, mean, log_std)
        entropy = gaussian_entropy(log_std)
        return -(log_prob * advantages).mean() - ent_coef * entropy.mean()

    @jax.jit
    def update(params, opt_state, x, actions, advantages, returns) -> tuple:
        """One optimisation step on the policy and the baseline."""

        def loss_fn(p):
            pg_loss = policy_loss(p, x, actions, advantages)
            values = value.apply({"params": p["value"]}, x)
            v_loss = jnp.mean((values - returns) ** 2)
            return pg_loss + value_coef * v_loss, (pg_loss, v_loss)

        (loss, (pg_loss, v_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params
        )
        updates, opt_state = tx.update(grads, opt_state, params)
        return (
            optax.apply_updates(params, updates),
            opt_state,
            loss,
            pg_loss,
            v_loss,
            optax.global_norm(grads),
        )

    @jax.jit
    def gradient_noise(params, x, actions, advantages) -> jnp.ndarray:
        """Cosine between the policy gradients of the two batch halves.

        A cheap proxy for the signal-to-noise ratio of the estimator: with an
        unbiased low-variance gradient the two halves agree, while pure noise
        gives a cosine around zero.
        """
        half = x.shape[0] // 2
        grad_fn = jax.grad(policy_loss)
        g1 = grad_fn(params, x[:half], actions[:half], advantages[:half])
        g2 = grad_fn(params, x[half:], actions[half:], advantages[half:])
        flat1 = jnp.concatenate(
            [leaf.ravel() for leaf in jax.tree_util.tree_leaves(g1["policy"])]
        )
        flat2 = jnp.concatenate(
            [leaf.ravel() for leaf in jax.tree_util.tree_leaves(g2["policy"])]
        )
        return jnp.sum(flat1 * flat2) / (
            jnp.linalg.norm(flat1) * jnp.linalg.norm(flat2) + EPS
        )

    for iteration in range(config["n_iterations"]):
        history.start_timer()
        key, key_context, key_action = jax.random.split(key, 3)
        x, b = sampler(key_context, batch)
        actions, _ = act(params, key_action, x)
        rewards, info = env.step(actions, b)

        values = baseline(params, x)
        advantages = rewards - values
        if config["normalize_advantage"]:
            advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)
        params_before = params
        params, opt_state, loss, pg_loss, v_loss, grad_norm = update(
            params, opt_state, x, actions, advantages, rewards
        )
        jax.block_until_ready(params["policy"])
        history.stop_timer()
        history.env_samples += batch
        history.gradient_steps += 1

        if iteration % config["log_every"] == 0:
            noise = (
                gradient_noise(params_before, x, actions, advantages)
                if config["grad_noise_probe"]
                else jnp.nan
            )
            explained = 1.0 - jnp.var(rewards - values) / (jnp.var(rewards) + EPS)
            history.log_iteration(
                iteration=iteration,
                env_samples=history.env_samples,
                train_time=history.train_time,
                reward_mean=rewards.mean(),
                reward_std=rewards.std(),
                loss=loss,
                policy_loss=pg_loss,
                value_loss=v_loss,
                grad_norm=grad_norm,
                explained_variance=explained,
                grad_cosine=noise,
                ineq_cv_mean=info["ineq_cv"].mean(),
            )
        periodic_evaluation(history, evaluate, params, iteration, config, "[reinforce]")
    return params
