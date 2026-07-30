"""PPO on the one-step QP bandit.

With a single-step episode there is no bootstrapping and no discounting: the
advantage is simply ``r - V(x)`` and GAE degenerates to that same quantity. What
PPO still contributes over REINFORCE is the reuse of each collected batch for
several clipped minibatch updates, which is exactly the axis of interest here --
the environment interaction (a projection over the whole batch) is by far the
most expensive part of an iteration.

As for REINFORCE, the policy is defined over the raw action, so the importance
ratio is exact: no density is ever evaluated at a projected point.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .common import periodic_evaluation
from .policy import (
    GaussianPolicy,
    ValueNetwork,
    gaussian_entropy,
    gaussian_log_prob,
    gaussian_sample,
)

EPS = 1e-8


def train(
    key: jax.Array,
    env,
    sampler: Callable,
    evaluate: Callable,
    config: dict,
    history,
) -> dict:
    """Train a policy with PPO.

    Args:
        key (jax.Array): Random key.
        env (QPBanditEnv): The environment.
        sampler (Callable): Context sampler.
        evaluate (Callable): Deterministic evaluation of the policy parameters.
        config (dict): Merged RL configuration (shared keys plus the ``ppo``
            section).
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
    minibatch = min(config["minibatch_size"], batch)
    n_minibatches = batch // minibatch
    clip_eps = config["clip_eps"]
    ent_coef = config["ent_coef"]
    value_coef = config["value_coef"]

    @jax.jit
    def act(params: dict, key: jax.Array, x: jnp.ndarray) -> tuple:
        """Sample actions and record their log-probability under the behaviour."""
        mean, log_std = policy.apply({"params": params["policy"]}, x)
        actions = gaussian_sample(key, mean, log_std)
        return actions, gaussian_log_prob(actions, mean, log_std)

    @jax.jit
    def baseline(params: dict, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the state-value baseline."""
        return value.apply({"params": params["value"]}, x)

    @jax.jit
    def update(params, opt_state, x, actions, log_prob_old, advantages, returns):
        """One clipped minibatch update."""

        def loss_fn(p):
            mean, log_std = policy.apply({"params": p["policy"]}, x)
            log_prob = gaussian_log_prob(actions, mean, log_std)
            ratio = jnp.exp(log_prob - log_prob_old)
            unclipped = ratio * advantages
            clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            pg_loss = -jnp.minimum(unclipped, clipped).mean()
            entropy = gaussian_entropy(log_std).mean()
            values = value.apply({"params": p["value"]}, x)
            v_loss = jnp.mean((values - returns) ** 2)
            loss = pg_loss + value_coef * v_loss - ent_coef * entropy
            # Approximate KL of Schulman, unbiased and non-negative.
            log_ratio = log_prob - log_prob_old
            approx_kl = jnp.mean(jnp.exp(log_ratio) - 1 - log_ratio)
            clip_fraction = jnp.mean(jnp.abs(ratio - 1) > clip_eps)
            return loss, (pg_loss, v_loss, entropy, approx_kl, clip_fraction)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    for iteration in range(config["n_iterations"]):
        history.start_timer()
        key, key_context, key_action = jax.random.split(key, 3)
        x, b = sampler(key_context, batch)
        actions, log_prob_old = act(params, key_action, x)
        rewards, info = env.step(actions, b)

        values = baseline(params, x)
        advantages = rewards - values
        if config["normalize_advantage"]:
            advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

        stop = False
        for epoch in range(config["n_epochs"]):
            key, key_perm = jax.random.split(key)
            perm = np.asarray(jax.random.permutation(key_perm, batch))
            for start in range(0, n_minibatches * minibatch, minibatch):
                idx = perm[start : start + minibatch]
                params, opt_state, loss, aux = update(
                    params,
                    opt_state,
                    x[idx],
                    actions[idx],
                    log_prob_old[idx],
                    advantages[idx],
                    rewards[idx],
                )
                history.gradient_steps += 1
            _, _, _, approx_kl, _ = aux
            if config["target_kl"] > 0 and float(approx_kl) > config["target_kl"]:
                stop = True
                break
        jax.block_until_ready(params["policy"])
        history.stop_timer()
        history.env_samples += batch

        if iteration % config["log_every"] == 0:
            pg_loss, v_loss, entropy, approx_kl, clip_fraction = aux
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
                entropy=entropy,
                approx_kl=approx_kl,
                clip_fraction=clip_fraction,
                early_stop_epoch=epoch if stop else config["n_epochs"],
                explained_variance=explained,
                ineq_cv_mean=info["ineq_cv"].mean(),
            )
        periodic_evaluation(history, evaluate, params, iteration, config, "[ppo]")
    return params
