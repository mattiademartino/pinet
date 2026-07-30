"""Run an RL baseline on the QP contextual bandit.

The problem is cast as a one-step bandit: the context is the right-hand side of
the equality constraints, the action is a point of the decision space, and the
environment returns ``-L(P(y))`` after projecting. The projection lives in the
environment and is never differentiated, so the agent has access to neither the
objective gradient nor the projection Jacobian -- unlike the pinet baseline,
which uses both.

Run with, e.g.::

    python -m src.benchmarks.QP.run_rl --id dc3_simple_1 --algo reinforce --seed 0
"""

import argparse
import pathlib
import time

import jax
import numpy as np

from benchmarks.QP.load_QP import load_data
from benchmarks.QP.rl import pinet_arm, ppo, reinforce, sac
from benchmarks.QP.rl.common import (
    History,
    make_context_sampler,
    make_evaluator,
    split_arrays,
)
from benchmarks.QP.rl.env import QPBanditEnv
from benchmarks.QP.rl.policy import GaussianPolicy
from src.tools.utils import load_configuration

jax.config.update("jax_enable_x64", True)

ALGORITHMS = {
    "reinforce": reinforce.train,
    "ppo": ppo.train,
    "sac": sac.train,
    # Same file as "sac": its configuration section sets critic_on_projection, so
    # the critic is trained and queried only on feasible, projected actions.
    "sac_proj": sac.train,
    # Not reinforcement learning: these differentiate through the projection and
    # are run here only to share the evaluation protocol and the budget axes.
    "pinet": pinet_arm.train_deterministic,
    "pinet_stochastic": pinet_arm.train_stochastic,
}


def merge_config(rl_config: dict, algo: str, seed: int, overrides: dict) -> dict:
    """Flatten the shared and algorithm-specific sections of the configuration.

    Args:
        rl_config (dict): Configuration as loaded from the yaml file.
        algo (str): Algorithm name, selecting the section to merge in.
        seed (int): Seed, recorded in the configuration for the replay buffer.
        overrides (dict): Command-line overrides, applied last; ``None`` values
            are ignored.

    Returns:
        dict: The merged configuration, with ``n_iterations`` derived from the
            sample budget.
    """
    config = {k: v for k, v in rl_config.items() if k not in ALGORITHMS}
    config.update(rl_config.get(algo, {}))
    config.update({k: v for k, v in overrides.items() if v is not None})
    config["seed"] = seed
    config["algo"] = algo
    if not config.get("n_iterations"):
        config["n_iterations"] = int(
            np.ceil(config["total_env_samples"] / config["context_batch"])
        )
    return config


def main(
    dataset_id: str,
    config_name: str,
    rl_config_name: str,
    algo: str,
    seed: int,
    overrides: dict,
    out_dir: pathlib.Path,
) -> None:
    """Train one RL agent and store its history and final test metrics.

    Args:
        dataset_id (str): Dataset identifier, see the ``ids`` folder.
        config_name (str): Configuration of the projection/environment.
        rl_config_name (str): Configuration of the RL hyperparameters.
        algo (str): One of ``reinforce``, ``ppo``, ``sac``.
        seed (int): Training seed.
        overrides (dict): Command-line overrides of the RL configuration.
        out_dir (pathlib.Path): Where to write the results.
    """
    here = pathlib.Path(__file__).parent.resolve()
    dataset = load_configuration(here / "ids" / (dataset_id + ".yaml"))
    hyperparameters = load_configuration(
        here.parent / "configs" / (config_name + ".yaml")
    )
    rl_config = load_configuration(here.parent / "configs" / (rl_config_name + ".yaml"))
    config = merge_config(rl_config, algo, seed, overrides)

    key = jax.random.PRNGKey(seed)
    loader_key, key = jax.random.split(key, 2)
    (
        A,
        G,
        h,
        X,
        batched_objective,
        train_loader,
        valid_loader,
        test_loader,
        batched_loss,
    ) = load_data(
        use_DC3_dataset=dataset["use_DC3_dataset"],
        use_convex=dataset["use_convex"],
        problem_seed=dataset["problem_seed"],
        problem_var=dataset["problem_var"],
        problem_nineq=dataset["problem_nineq"],
        problem_neq=dataset["problem_neq"],
        problem_examples=dataset["problem_examples"],
        rng_key=loader_key,
        batch_size=hyperparameters.get("batch_size", 2048),
        use_jax_loader=True,
        penalty=hyperparameters.get("penalty", 0.0),
    )

    start_setup = time.time()
    env = QPBanditEnv(
        A=A,
        G=G,
        h=h,
        X=X,
        batched_objective=batched_objective,
        batched_loss=batched_loss,
        hyperparameters=hyperparameters,
    )
    setup_time = time.time() - start_setup

    X_train = train_loader.dataset.X
    sampler = make_context_sampler(X_train)
    X_valid, obj_valid = split_arrays(valid_loader)
    X_test, obj_test = split_arrays(test_loader)

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
    evaluate_valid = make_evaluator(env, policy, X_valid, obj_valid)
    evaluate_test = make_evaluator(env, policy, X_test, obj_test)

    print(
        f"[{algo} seed={seed}] {dataset_id}: dim={env.dim} n_eq={env.n_eq} "
        f"iterations={config['n_iterations']} context_batch={config['context_batch']} "
        f"env_samples={config['n_iterations'] * config['context_batch']}",
        flush=True,
    )

    history = History()
    key, train_key = jax.random.split(key)
    params = ALGORITHMS[algo](
        key=train_key,
        env=env,
        sampler=sampler,
        evaluate=evaluate_valid,
        config=config,
        history=history,
    )

    test = evaluate_test(params)
    print(
        f"[{algo} seed={seed}] TEST rs={test['rs_mean']:.5f} "
        f"objective={test['objective_mean']:.5f} "
        f"eqcv_max={test['eq_cv_max']:.3e} ineqcv_max={test['ineq_cv_max']:.3e} "
        f"train_time={history.train_time:.1f}s",
        flush=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{dataset_id}_{algo}_seed{seed}.npz"
    payload = history.as_payload()
    payload.update(
        {
            "algo": algo,
            "dataset_id": dataset_id,
            "config": config_name,
            "rl_config": rl_config_name,
            "seed": seed,
            "setup_time": setup_time,
            "test_rs": test["rs"],
            "test_ineq_cv": test["ineq_cv"],
            "test_eq_cv": test["eq_cv"],
            "test_rs_mean": test["rs_mean"],
            "test_objective_mean": test["objective_mean"],
            "test_eq_cv_max": test["eq_cv_max"],
            "test_ineq_cv_mean": test["ineq_cv_mean"],
            "test_ineq_cv_max": test["ineq_cv_max"],
            "n_iterations": config["n_iterations"],
            "context_batch": config["context_batch"],
        }
    )
    np.savez_compressed(out_file, **payload)
    print(f"[{algo} seed={seed}] wrote {out_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RL baselines on the QP contextual bandit."
    )
    parser.add_argument("--id", type=str, required=True, help="Dataset identifier.")
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_small_autotune",
        help="Configuration of the projection, shared with the pinet baseline.",
    )
    parser.add_argument(
        "--rl_config",
        type=str,
        default="rl_default",
        help="Configuration of the RL hyperparameters.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=sorted(ALGORITHMS),
        help="Which agent to train.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Training seed.")
    parser.add_argument(
        "--n_iterations", type=int, default=None, help="Override the iteration count."
    )
    parser.add_argument(
        "--total_env_samples",
        type=int,
        default=None,
        help="Override the environment-sample budget.",
    )
    parser.add_argument(
        "--context_batch", type=int, default=None, help="Override the context batch."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=None, help="Override the learning rate."
    )
    parser.add_argument(
        "--log_std_init",
        type=float,
        default=None,
        help="Override the initial log standard deviation of the policy.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(pathlib.Path(__file__).parent / "results" / "rl"),
        help="Output directory.",
    )
    args = parser.parse_args()

    main(
        dataset_id=args.id,
        config_name=args.config,
        rl_config_name=args.rl_config,
        algo=args.algo,
        seed=args.seed,
        overrides={
            "n_iterations": args.n_iterations,
            "total_env_samples": args.total_env_samples,
            "context_batch": args.context_batch,
            "learning_rate": args.learning_rate,
            "log_std_init": args.log_std_init,
        },
        out_dir=pathlib.Path(args.out),
    )
