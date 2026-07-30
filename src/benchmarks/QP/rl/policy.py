"""Networks and distributions for the RL baselines.

The policy trunk mirrors the pinet MLP (same ``features_list`` and activation),
so the only difference between the RL agents and the self-supervised baseline is
how the parameters are updated, not the capacity of the network.

Actions are unbounded: the policy is a diagonal Gaussian over the whole decision
space, with no squashing. Feasibility is never the policy's concern, since the
environment projects.
"""

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

# Constant term of the diagonal Gaussian log-density, per dimension.
_LOG_2PI = 1.8378770664093453


class GaussianPolicy(nn.Module):
    """Diagonal Gaussian policy over unbounded actions."""

    dim: int
    features_list: Sequence[int]
    activation: Callable = nn.relu
    state_dependent_std: bool = False
    log_std_init: float = -1.6
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return the mean and log standard deviation of the action distribution.

        Args:
            x (jnp.ndarray): Contexts, shape (batch, n_eq).

        Returns:
            tuple: ``(mean, log_std)``, both of shape (batch, dim).
        """
        h = x
        for features in self.features_list:
            h = self.activation(nn.Dense(features)(h))
        mean = nn.Dense(self.dim)(h)
        if self.state_dependent_std:
            log_std = nn.Dense(self.dim)(h)
        else:
            log_std = self.param(
                "log_std",
                lambda _: jnp.full((self.dim,), self.log_std_init),
            )
            log_std = jnp.broadcast_to(log_std, mean.shape)
        return mean, jnp.clip(log_std, self.log_std_min, self.log_std_max)


class ValueNetwork(nn.Module):
    """State-value baseline ``V(x)``."""

    features_list: Sequence[int]
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the baseline.

        Args:
            x (jnp.ndarray): Contexts, shape (batch, n_eq).

        Returns:
            jnp.ndarray: Values, shape (batch,).
        """
        h = x
        for features in self.features_list:
            h = self.activation(nn.Dense(features)(h))
        return nn.Dense(1)(h).ravel()


class TwinQNetwork(nn.Module):
    """Two independent action-value heads ``Q(x, y)``, as in SAC."""

    features_list: Sequence[int]
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> tuple:
        """Evaluate both critics.

        Args:
            x (jnp.ndarray): Contexts, shape (batch, n_eq).
            y (jnp.ndarray): Actions, shape (batch, dim).

        Returns:
            tuple: ``(q1, q2)``, both of shape (batch,).
        """
        inp = jnp.concatenate([x, y], axis=-1)
        outputs = []
        for _ in range(2):
            h = inp
            for features in self.features_list:
                h = self.activation(nn.Dense(features)(h))
            outputs.append(nn.Dense(1)(h).ravel())
        return outputs[0], outputs[1]


def gaussian_sample(
    key: jax.Array, mean: jnp.ndarray, log_std: jnp.ndarray
) -> jnp.ndarray:
    """Sample from a diagonal Gaussian.

    Args:
        key (jax.Array): Random key.
        mean (jnp.ndarray): Means, shape (batch, dim).
        log_std (jnp.ndarray): Log standard deviations, shape (batch, dim).

    Returns:
        jnp.ndarray: Samples, shape (batch, dim).
    """
    return mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)


def gaussian_log_prob(
    actions: jnp.ndarray, mean: jnp.ndarray, log_std: jnp.ndarray
) -> jnp.ndarray:
    """Log-density of a diagonal Gaussian, summed over the action dimensions.

    Args:
        actions (jnp.ndarray): Actions, shape (batch, dim).
        mean (jnp.ndarray): Means, shape (batch, dim).
        log_std (jnp.ndarray): Log standard deviations, shape (batch, dim).

    Returns:
        jnp.ndarray: Log-probabilities, shape (batch,).
    """
    normalized = (actions - mean) / jnp.exp(log_std)
    return -0.5 * jnp.sum(normalized**2 + _LOG_2PI + 2 * log_std, axis=-1)


def gaussian_entropy(log_std: jnp.ndarray) -> jnp.ndarray:
    """Entropy of a diagonal Gaussian, summed over the action dimensions.

    Args:
        log_std (jnp.ndarray): Log standard deviations, shape (batch, dim).

    Returns:
        jnp.ndarray: Entropies, shape (batch,).
    """
    return jnp.sum(log_std + 0.5 * (_LOG_2PI + 1.0), axis=-1)
