"""SAC on the one-step QP bandit, in two variants.

Two structural simplifications follow from the episode being a single step, and
both are exact rather than approximations:

- every transition is terminal, so the critic target is the reward itself. There
  is no bootstrapping, hence no discount factor and no target network -- the
  critic is a regression of ``r`` onto ``(x, y)``.
- the actor uses the reparameterised, pathwise gradient
  ``grad E[alpha * log pi - Q(x, a)]``.

The two variants differ in *which action the critic is a function of*, and are
selected by the ``critic_on_projection`` flag:

``sac`` (``False``)
    The critic sees the raw action, the one the policy emitted. It therefore has
    to represent ``r(x, y) = -L(P(y))`` over the whole of ``R^dim``. That
    function is constant along every fibre of the projection: all the raw
    actions that project to the same feasible point share one value, so the
    landscape is a union of plateaus whose shape the critic has to discover from
    data. The actor gradient flows through the critic only; the projection is
    never differentiated.

``sac_proj`` (``True``)
    The critic only ever sees *projected* actions, both when it is trained and
    when the actor queries it. It therefore learns ``-L`` on the feasible set
    alone -- a smooth function, and exactly the one the reward is. The plateaus
    are no longer learned, they are structural: the actor reaches the critic
    through the projection, so its gradient is ``J_P^T grad_y Q``, which vanishes
    along precisely the directions the projection collapses. The critic is never
    evaluated off the feasible set, where it would be extrapolating.

The price of ``sac_proj`` is that the actor gradient has to traverse the
projection, so this arm uses ``J_P`` -- through the implicit function theorem,
as the pinet baseline does -- while still not using ``grad L``: the objective
reaches it only through the learned critic. On the "what does the agent know"
axis it therefore sits between the RL agents and the pinet arms, and the
``proj_grad_cos`` / ``proj_grad_ratio`` diagnostics report how much of the
critic's gradient the projection removes.

Actions are unbounded, so there is no tanh squashing and no log-det correction:
the Gaussian log-density is exact. This is unaffected by the projection, because
the entropy term is always evaluated on the raw Gaussian sample -- the density of
the projected action is degenerate (it puts mass on the faces of the feasible
set) and admits no change of variables.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .common import periodic_evaluation
from .policy import GaussianPolicy, TwinQNetwork, gaussian_log_prob, gaussian_sample

EPS = 1e-8
# Instances used by the projection-flatness probe of the ``sac_proj`` variant.
N_PROBE_INSTANCES = 256


class ReplayBuffer:
    """Flat replay buffer of ``(context, action, reward)`` triples."""

    def __init__(self, capacity: int, n_eq: int, dim: int) -> None:
        """Initialize the buffer.

        Args:
            capacity (int): Maximum number of stored transitions.
            n_eq (int): Dimension of the context.
            dim (int): Dimension of the action.
        """
        self.capacity = capacity
        self.x = np.zeros((capacity, n_eq))
        self.y = np.zeros((capacity, dim))
        self.r = np.zeros(capacity)
        self.size = 0
        self.pointer = 0

    def add(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> None:
        """Insert a batch of transitions, overwriting the oldest ones.

        Args:
            x (np.ndarray): Contexts, shape (batch, n_eq).
            y (np.ndarray): Actions the critic is trained on, shape (batch, dim).
                Raw actions for ``sac``, projected ones for ``sac_proj``.
            r (np.ndarray): Rewards, shape (batch,).
        """
        batch = x.shape[0]
        idx = (self.pointer + np.arange(batch)) % self.capacity
        self.x[idx] = x
        self.y[idx] = y
        self.r[idx] = r
        self.pointer = int((self.pointer + batch) % self.capacity)
        self.size = int(min(self.size + batch, self.capacity))

    def sample(self, rng: np.random.Generator, batch: int) -> tuple:
        """Draw a minibatch uniformly at random.

        Args:
            rng (np.random.Generator): Random generator.
            batch (int): Minibatch size.

        Returns:
            tuple: ``(x, y, r)`` as arrays.
        """
        idx = rng.integers(0, self.size, size=batch)
        return self.x[idx], self.y[idx], self.r[idx]


def train(
    key: jax.Array,
    env,
    sampler: Callable,
    evaluate: Callable,
    config: dict,
    history,
) -> dict:
    """Train a policy with SAC.

    Args:
        key (jax.Array): Random key.
        env (QPBanditEnv): The environment.
        sampler (Callable): Context sampler.
        evaluate (Callable): Deterministic evaluation of the policy parameters.
        config (dict): Merged RL configuration (shared keys plus the ``sac`` or
            ``sac_proj`` section). ``critic_on_projection`` selects the variant.
        history (History): Bookkeeping object, updated in place.

    Returns:
        dict: The trained parameters.
    """
    critic_on_projection = bool(config.get("critic_on_projection", False))
    # Restricting the critic to the feasible set is only half of the variant: the
    # actor then has to reach it *through* the projection, which is also what
    # exposes the flat directions to the policy gradient. Hence the differentiable
    # projection, with the same iteration budget as the environment step.
    project = env.differentiable_projection() if critic_on_projection else None

    activation = getattr(jax.nn, config["activation"])
    # SAC needs a state-dependent scale: the entropy term acts per context.
    policy = GaussianPolicy(
        dim=env.dim,
        features_list=config["features_list"],
        activation=activation,
        state_dependent_std=True,
        log_std_init=config["log_std_init"],
        log_std_min=config["log_std_min"],
        log_std_max=config["log_std_max"],
    )
    critic = TwinQNetwork(features_list=config["features_list"], activation=activation)

    key, key_policy, key_critic = jax.random.split(key, 3)
    x_example = jnp.zeros((2, env.n_eq))
    y_example = jnp.zeros((2, env.dim))
    params = {
        "policy": policy.init(key_policy, x_example)["params"],
        "critic": critic.init(key_critic, x_example, y_example)["params"],
    }
    log_alpha = jnp.array(np.log(config["init_alpha"]))
    target_entropy = -config["target_entropy_scale"] * env.dim

    tx_actor = optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.adam(config["learning_rate"]),
    )
    tx_critic = optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.adam(config["critic_learning_rate"]),
    )
    tx_alpha = optax.adam(config["alpha_learning_rate"])
    opt_actor = tx_actor.init(params["policy"])
    opt_critic = tx_critic.init(params["critic"])
    opt_alpha = tx_alpha.init(log_alpha)

    buffer = ReplayBuffer(config["buffer_capacity"], env.n_eq, env.dim)
    rng = np.random.default_rng(config["seed"])
    batch = config["context_batch"]

    @jax.jit
    def act(params: dict, key: jax.Array, x: jnp.ndarray) -> jnp.ndarray:
        """Sample an action from the current policy."""
        mean, log_std = policy.apply({"params": params["policy"]}, x)
        return gaussian_sample(key, mean, log_std)

    @jax.jit
    def act_warmup(key: jax.Array, x: jnp.ndarray) -> jnp.ndarray:
        """Sample an uninformed action, used to seed the buffer."""
        return config["warmup_std"] * jax.random.normal(key, (x.shape[0], env.dim))

    @jax.jit
    def update(
        params: dict,
        log_alpha: jnp.ndarray,
        opt_actor,
        opt_critic,
        opt_alpha,
        key: jax.Array,
        x: jnp.ndarray,
        y: jnp.ndarray,
        r: jnp.ndarray,
    ) -> tuple:
        """One SAC update of critic, actor and temperature."""
        alpha = jnp.exp(log_alpha)

        def critic_loss_fn(critic_params):
            # Terminal transition: the target is the reward, exactly.
            q1, q2 = critic.apply({"params": critic_params}, x, y)
            loss = jnp.mean((q1 - r) ** 2) + jnp.mean((q2 - r) ** 2)
            corr = jnp.corrcoef(q1, r)[0, 1]
            return loss, corr

        (critic_loss, q_corr), critic_grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(params["critic"])
        critic_updates, opt_critic = tx_critic.update(
            critic_grads, opt_critic, params["critic"]
        )
        critic_params = optax.apply_updates(params["critic"], critic_updates)

        def actor_loss_fn(policy_params):
            mean, log_std = policy.apply({"params": policy_params}, x)
            actions = gaussian_sample(key, mean, log_std)
            # The entropy term stays on the raw Gaussian: the projected action has
            # no density to speak of.
            log_prob = gaussian_log_prob(actions, mean, log_std)
            # Query the critic where it was trained. For sac_proj that is the
            # feasible set, so the action is projected first and the gradient
            # comes back as J_P^T grad_y Q -- zero along the collapsed directions.
            evaluated = (
                project(actions, x[..., None]) if critic_on_projection else actions
            )
            q1, q2 = critic.apply({"params": critic_params}, x, evaluated)
            q = jnp.minimum(q1, q2)
            return jnp.mean(alpha * log_prob - q), log_prob

        (actor_loss, log_prob), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(params["policy"])
        actor_updates, opt_actor = tx_actor.update(
            actor_grads, opt_actor, params["policy"]
        )
        policy_params = optax.apply_updates(params["policy"], actor_updates)

        def alpha_loss_fn(log_alpha):
            return -jnp.mean(log_alpha * (log_prob + target_entropy))

        alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(log_alpha)
        alpha_updates, opt_alpha = tx_alpha.update(alpha_grads, opt_alpha, log_alpha)
        log_alpha = optax.apply_updates(log_alpha, alpha_updates)

        return (
            {"policy": policy_params, "critic": critic_params},
            log_alpha,
            opt_actor,
            opt_critic,
            opt_alpha,
            {
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "alpha_loss": alpha_loss,
                "alpha": alpha,
                "log_prob": jnp.mean(log_prob),
                "q_reward_corr": q_corr,
            },
        )

    @jax.jit
    def flatness_probe(params: dict, key: jax.Array, x: jnp.ndarray) -> tuple:
        """Measure how much of the critic's gradient the projection removes.

        The actor of ``sac_proj`` receives ``J_P^T v`` where ``v = grad_y Q`` is
        taken at the projected action. Comparing the two isolates the flat
        directions: a cosine near zero, or a norm ratio near zero, means the
        projection is discarding most of what the critic asks for.

        Args:
            params (dict): Current parameters.
            key (jax.Array): Random key for the action sample.
            x (jnp.ndarray): Contexts, shape (batch, n_eq).

        Returns:
            tuple: Mean cosine and mean norm ratio between ``J_P^T v`` and ``v``.
        """
        mean, log_std = policy.apply({"params": params["policy"]}, x)
        actions = gaussian_sample(key, mean, log_std)
        y_proj, vjp_fn = jax.vjp(lambda a: project(a, x[..., None]), actions)

        def q_min(y: jnp.ndarray) -> jnp.ndarray:
            """Sum of the pessimistic critic over the batch."""
            q1, q2 = critic.apply({"params": params["critic"]}, x, y)
            return jnp.sum(jnp.minimum(q1, q2))

        v = jax.grad(q_min)(y_proj)
        jtv = vjp_fn(v)[0]
        norm_v = jnp.linalg.norm(v, axis=1)
        norm_jtv = jnp.linalg.norm(jtv, axis=1)
        cos = jnp.sum(v * jtv, axis=1) / (norm_v * norm_jtv + EPS)
        return jnp.mean(cos), jnp.mean(norm_jtv / (norm_v + EPS))

    metrics = {}
    for iteration in range(config["n_iterations"]):
        history.start_timer()
        key, key_context, key_action = jax.random.split(key, 3)
        x, b = sampler(key_context, batch)
        if iteration < config["warmup_iterations"]:
            actions = act_warmup(key_action, x)
        else:
            actions = act(params, key_action, x)
        rewards, info = env.step(actions, b)
        # sac_proj stores the feasible point the reward was actually earned at, so
        # the critic never sees an action off the feasible set.
        stored = info["y_proj"] if critic_on_projection else actions
        buffer.add(np.asarray(x), np.asarray(stored), np.asarray(rewards))

        for _ in range(config["updates_per_iteration"]):
            key, key_update = jax.random.split(key, 2)
            x_b, y_b, r_b = buffer.sample(rng, config["batch_size"])
            (
                params,
                log_alpha,
                opt_actor,
                opt_critic,
                opt_alpha,
                metrics,
            ) = update(
                params,
                log_alpha,
                opt_actor,
                opt_critic,
                opt_alpha,
                key_update,
                jnp.asarray(x_b),
                jnp.asarray(y_b),
                jnp.asarray(r_b),
            )
            history.gradient_steps += 1
        jax.block_until_ready(params["policy"])
        history.stop_timer()
        history.env_samples += batch

        if iteration % config["log_every"] == 0:
            # Measured off the clock, like the evaluation, and on a small slice:
            # the probe costs one extra projection and its backward pass.
            probe = {}
            if critic_on_projection:
                key, key_probe = jax.random.split(key, 2)
                cos, ratio = flatness_probe(params, key_probe, x[:N_PROBE_INSTANCES])
                probe = {
                    "proj_grad_cos": float(cos),
                    "proj_grad_ratio": float(ratio),
                }
            history.log_iteration(
                iteration=iteration,
                env_samples=history.env_samples,
                train_time=history.train_time,
                reward_mean=rewards.mean(),
                reward_std=rewards.std(),
                buffer_size=buffer.size,
                ineq_cv_mean=info["ineq_cv"].mean(),
                **probe,
                **{name: float(value) for name, value in metrics.items()},
            )
        # Both variants run this file, so the tag names the arm, not the module.
        periodic_evaluation(
            history, evaluate, params, iteration, config, f"[{config['algo']}]"
        )
    return params
