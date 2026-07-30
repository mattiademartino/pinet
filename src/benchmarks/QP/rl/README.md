# RL baselines on the QP benchmark

This directory casts the parametric QP as a **one-step contextual bandit** and
solves it with REINFORCE, PPO and SAC, so that the reinforcement-learning route
can be compared with the self-supervised pinet training on identical problems.

## Formulation

| | |
|---|---|
| context | `x`, the right-hand side of the equality constraints, drawn i.i.d. from the training split |
| action | `y ∈ R^d`, a point of the decision space, produced without any constraint awareness |
| reward | `r = -L(P(y))`, where `P` is the pinet projection and `L` the training loss |
| episode | one step; every transition is terminal |

The episode is a single step, so this is a bandit rather than a sequential
problem. That is deliberate: it makes the RL arms directly comparable with the
pinet baseline, which solves the same amortisation problem in one shot.

## The projection is part of the environment

`P` is applied **inside** `QPBanditEnv` and wrapped in `jax.lax.stop_gradient`.
Two consequences, both intended:

1. **The agent never differentiates the projection.** It sees a scalar reward,
   nothing else. Compared with the pinet baseline it gives up *both* the
   objective gradient and the projection Jacobian, which is exactly the quantity
   this comparison is meant to price.
2. **The policy density stays well defined.** `P` is not injective and maps sets
   of positive measure onto the faces of the polytope, so the law of a projected
   action is singular and `log pi(P(y))` does not exist. Because the policy is
   defined over the *raw* action, the PPO importance ratio and the SAC entropy
   term are exact, with no surrogate involved. This is why the projection must
   not be moved into the actor unless one is willing to approximate the entropy.

Feasibility is unaffected: the reported solution is always `P(y)`, so equality
constraints hold to machine precision and inequalities to the accuracy of the
truncated Douglas-Rachford iteration, exactly as for pinet.

## The six arms

Four of them are RL agents, which learn the objective only through the reward.
The other two differentiate the loss through the projection and are there to
bracket the comparison; they are run through the same driver so that every curve
comes from the same evaluation protocol and the same budget axes.

| arm | action | gradient | uses `grad L` | uses `J_P` |
|---|---|---|---|---|
| `pinet` | deterministic | analytic | yes | yes |
| `pinet_stochastic` | sampled, network outputs mean *and* std | pathwise (reparameterisation) | yes | yes |
| `reinforce` | sampled | score function | no | no |
| `ppo` | sampled | score function + clipped importance ratio | no | no |
| `sac` | sampled | pathwise through a *learned* critic | no | no |
| `sac_proj` | sampled | pathwise through a critic defined on the feasible set | no | **yes** |

`pinet_stochastic` is the bridge on the *estimator* axis; `sac_proj` is the
bridge on the *knowledge* axis. It is the only arm that uses `J_P` without using
`grad L`: the objective still reaches it only through a learned critic, but the
critic is restricted to the feasible set, so the actor has to reach it through
the projection.

`pinet_stochastic` is the bridge: it optimises the same Gaussian-smoothed
objective as the RL agents, `E_eps[L(P(mu + sigma * eps))]`, but estimates its
gradient pathwise rather than by scoring samples. Against `pinet` it isolates
the effect of sampling; against SAC and REINFORCE it isolates the estimator.
Nothing pushes its standard deviation up, so whether it collapses to the
deterministic arm is an empirical question the curves answer.

## What each algorithm reduces to in a one-step episode

Both simplifications below are exact, not approximations.

- **REINFORCE** — the policy gradient is the score-function estimator
  `E[(r - V(x)) grad log pi(y|x)]`, with `V(x)` a learned baseline.
- **PPO** — no bootstrapping and no discounting, so GAE degenerates to
  `r - V(x)`. What PPO still adds over REINFORCE is the reuse of each collected
  batch for several clipped minibatch updates: relevant here, since the
  environment step (a projection over the whole batch) dominates the cost of an
  iteration. Early stopping on the approximate KL is enabled by default.
- **SAC** — every transition is terminal, so the critic target is the reward
  itself: no discount, no target network, and the critic is a plain regression of
  `r` onto `(x, y)`. The actor keeps its reparameterised pathwise gradient, which
  flows through the *critic*, never through the environment. Actions are
  unbounded, hence no tanh squashing and no log-det correction.
- **SAC on projected actions** (`sac_proj`) — the same, except for *where the
  critic lives*. See below.

The SAC critic has to learn over `R^(n_eq + d)` a function that the pinet
baseline is handed in closed form; the `q_reward_corr` diagnostic tracks how well
it does.

## Why `sac_proj` exists

The reward is `r = -L(P(y))`, so as a function of the raw action it is **constant
along every fibre of the projection**: all the actions that project to the same
feasible point earn the same reward. The value landscape a plain SAC critic has
to fit over `R^d` is therefore a union of plateaus, and their shape is decided by
the geometry of the polytope — something the critic can only discover from data,
spending capacity on a structure that is known in advance.

`sac_proj` removes that burden by construction:

- **training** — the buffer stores `P(y)`, the feasible point where the reward was
  actually earned, instead of the raw action. The critic then regresses `r` onto
  the feasible set only, where the target is simply `-L`: smooth, and exactly the
  objective.
- **use** — the actor queries the critic at `P(y)` too, never off the feasible
  set where the critic would be extrapolating. Its gradient therefore comes back
  as `J_P^T grad_y Q`, which is **exactly zero along the directions the projection
  collapses**. The flatness is no longer learned: it is imposed.

The trade-off is explicit and must be reported as such: the actor gradient now
traverses the projection, so this arm uses `J_P` — via the implicit function
theorem, like the pinet baseline — and is no longer a black-box agent. It is also
more expensive, since every actor update runs a projection and its backward pass
on top of the environment step.

## Running

```bash
# Sanity check: a small convex QP where the agents are expected to converge.
python -m src.benchmarks.QP.generate_small_QP
python -m src.benchmarks.QP.run_rl --id dc3_simple_small --algo reinforce --seed 0 \
    --total_env_samples 256000 --context_batch 512

# Full benchmark.
python -m src.benchmarks.QP.run_rl --id dc3_simple_1 --algo ppo --seed 0
./src/benchmarks/QP/run_rl_batch.sh          # 3 agents x 2 datasets x 5 seeds

# Figures and summary table.
python -m src.benchmarks.QP.parse_rl
```

`--config` selects the projection configuration and defaults to
`benchmark_small_autotune`, the same file used by the pinet baseline, so the
environment is bit-identical across arms. `--rl_config` selects the agent
hyperparameters (`configs/rl_default.yaml`); its per-algorithm sections are
merged over the shared ones. The budget is expressed in environment samples
(`total_env_samples`) and the iteration count is derived from it, so all agents
consume the same number of interactions.

## Comparing fairly

The pinet reference is obtained from the `ift` arm of `run_grad_ablation.py`,
which trains the same MLP on the same loss. Two caveats when reading the table:

- **Sample budget and update budget are different axes.** The RL agents consume
  millions of environment interactions; pinet performs one update per minibatch
  and converges in a few hundred. Report both `learning.png` panels -- against
  samples and against wall-clock time -- and state which one a claim refers to.
- **Check that the reference has converged.** On a small dataset,
  `n_epochs x (dataset / batch_size)` can be a very small number of updates: on
  the 1600-instance sanity problem the default configuration performs only 50.
  Raise `--n_epochs` until the reference plateaus before comparing against it.

## Diagnostics worth watching

| quantity | where | what it tells you |
|---|---|---|
| `grad_cosine` | REINFORCE | cosine between the policy gradients of the two halves of the batch, a cheap signal-to-noise proxy. Near zero means the estimator is dominated by noise. |
| `explained_variance` | REINFORCE, PPO | quality of the baseline `V(x)`; low values mean the advantage is mostly reward noise. |
| `approx_kl`, `clip_fraction` | PPO | how far each batch of updates moves the policy; drives the early stop. |
| `q_reward_corr` | SAC, `sac_proj` | correlation between the critic and the true reward, i.e. how well the critic has learned the objective it is replacing. Expected to rise faster for `sac_proj`, whose regression target is smooth. |
| `proj_grad_cos`, `proj_grad_ratio` | `sac_proj` | cosine and norm ratio between `J_P^T v` and `v`, where `v = grad_y Q` at the projected action. They measure the fraction of the critic's demand that the projection discards as infeasible. Both are computed off the clock, every `log_every` iterations, on 256 instances. Since `J_P` is an orthogonal projector the two coincide, which doubles as a correctness check of the layer. |
| `log_std_mean` | all | collapse or explosion of the exploration scale. |
