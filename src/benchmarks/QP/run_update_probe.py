"""One-step counterfactual: does the wrong gradient change the projected output?

Along a reference training trajectory, at every step we branch: starting from the
*same* parameters and the *same* optimiser state, we apply one update with the
true (implicit) projection gradient and one with the straight-through gradient.
The two resulting networks are then evaluated on a fixed probe batch, before and
after the projection.

The comparison is run with two optimisers:

- ``adam``: what actually trains the model. Its per-coordinate normalisation
  absorbs most of the gradient magnitude discrepancy.
- ``sgd``: a plain ``-lr * g`` step, which isolates the raw gradient direction.

Run with, e.g.::

    python -m src.benchmarks.QP.run_update_probe --id dc3_simple_1 \
        --trajectory ift --seed 0
"""

import argparse
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch

from benchmarks.model import HardConstrainedMLP
from benchmarks.QP.load_QP import load_data
from benchmarks.QP.run_grad_ablation import build_projections
from src.tools.utils import load_configuration

jax.config.update("jax_enable_x64", True)

N_PROBE_INSTANCES = 256
EPS = 1e-30
# Per-instance distributions are stored in full only at these epochs.
CHECKPOINT_EPOCHS = (0, 4, 9, 24, 49)


def make_output_fn(model_projected, model_raw):
    """Build a function returning the pre- and post-projection outputs.

    Args:
        model_projected: MLP whose test-time forward includes the projection.
        model_raw: The same MLP with ``raw_test=True``, i.e. no projection.

    Returns:
        Callable: ``outputs(params, x, b) -> (y_raw, y)``.
    """

    def outputs(params: dict, x: jnp.ndarray, b: jnp.ndarray) -> tuple:
        """Evaluate the network before and after the projection."""
        y_raw = model_raw.apply({"params": params}, x=x, b=b, test=True)
        y = model_projected.apply({"params": params}, x=x, b=b, test=True)
        return y_raw, y

    return outputs


def divergence(a: jnp.ndarray, b: jnp.ndarray, base: jnp.ndarray) -> dict:
    """Compare two updated outputs against the pre-update one.

    Args:
        a (jnp.ndarray): Output after the true-gradient update, shape (B, d).
        b (jnp.ndarray): Output after the straight-through update, shape (B, d).
        base (jnp.ndarray): Output before the update, shape (B, d).

    Returns:
        dict: Per-instance distance between the two updates, the size of each
            update, their ratio, and the cosine between the two displacements.
    """
    delta_a = a - base
    delta_b = b - base
    step_a = jnp.linalg.norm(delta_a, axis=1)
    step_b = jnp.linalg.norm(delta_b, axis=1)
    dist = jnp.linalg.norm(a - b, axis=1)
    cos = jnp.sum(delta_a * delta_b, axis=1) / (step_a * step_b + EPS)
    return {
        "dist": dist,
        "step_ift": step_a,
        "step_st": step_b,
        "relative": dist / (step_a + EPS),
        "cos": cos,
    }


def make_quality_fn(batched_objective, A, G, h):
    """Build a function measuring objective and constraint violation.

    Args:
        batched_objective: Batched objective function.
        A (jnp.ndarray): Equality constraint matrix.
        G (jnp.ndarray): Inequality constraint matrix.
        h (jnp.ndarray): Inequality bounds.

    Returns:
        Callable: ``quality(y, b) -> (objective, eq_cv, ineq_cv)`` per instance.
    """

    def quality(y: jnp.ndarray, b: jnp.ndarray) -> tuple:
        """Evaluate objective and violations of a batch of candidate solutions."""
        batch = y.shape[0]
        obj = batched_objective(y).reshape(batch)
        eq_cv = jnp.max(
            jnp.abs(
                A[0].reshape(1, A.shape[1], A.shape[2]) @ y.reshape(batch, -1, 1) - b
            ),
            axis=1,
        ).reshape(batch)
        ineq_cv = jnp.max(
            jnp.maximum(
                G[0].reshape(1, G.shape[1], G.shape[2]) @ y.reshape(batch, -1, 1) - h, 0
            ),
            axis=1,
        ).reshape(batch)
        return obj, eq_cv, ineq_cv

    return quality


def main(
    dataset_id: str,
    config: str,
    trajectory: str,
    seed: int,
    n_epochs: int,
    out_dir: pathlib.Path,
) -> None:
    """Run the one-step counterfactual probe along one trajectory.

    Args:
        dataset_id (str): Dataset identifier.
        config (str): Hyperparameter configuration name.
        trajectory (str): Which gradient drives the reference trajectory.
        seed (int): Training seed.
        n_epochs (int): Override for the number of epochs, if > 0.
        out_dir (pathlib.Path): Output directory.
    """
    here = pathlib.Path(__file__).parent.resolve()
    dataset = load_configuration(here / "ids" / (dataset_id + ".yaml"))
    hyperparameters = load_configuration(here.parent / "configs" / (config + ".yaml"))
    if n_epochs > 0:
        hyperparameters["n_epochs"] = n_epochs

    torch.manual_seed(seed)
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
        _,
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

    project_ift, project_st, project_test = build_projections(
        hyperparameters, A, X, G, h
    )
    common = {
        "project_test": project_test,
        "dim": A.shape[2],
        "features_list": hyperparameters["features_list"],
        "activation": getattr(jax.nn, hyperparameters["activation"]),
    }
    model_ift = HardConstrainedMLP(project=project_ift, **common)
    model_st = HardConstrainedMLP(project=project_st, **common)
    model_raw = HardConstrainedMLP(project=project_ift, raw_test=True, **common)

    params = model_ift.init(key, x=X[:2, :, 0], b=X[:2], test=False)["params"]

    def make_loss(model):
        """Build the training loss for a given model."""

        def loss_fn(p: dict, x: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
            preds = model.apply({"params": p}, x=x, b=b, test=False)
            return batched_loss(preds, b).mean()

        return loss_fn

    grad_ift = jax.jit(jax.value_and_grad(make_loss(model_ift)))
    grad_st = jax.jit(jax.value_and_grad(make_loss(model_st)))

    learning_rate = hyperparameters["learning_rate"]
    tx = optax.adam(learning_rate)
    opt_state = tx.init(params)

    outputs = make_output_fn(model_ift, model_raw)
    quality = make_quality_fn(batched_objective, A, G, h)

    @jax.jit
    def counterfactual(
        p: dict,
        state,
        g_ift: dict,
        g_st: dict,
        x_probe: jnp.ndarray,
        b_probe: jnp.ndarray,
    ) -> dict:
        """Apply both updates from the same state and compare the outputs."""
        # Adam: both branches start from the *same* optimiser state.
        upd_ift, _ = tx.update(g_ift, state, p)
        upd_st, _ = tx.update(g_st, state, p)
        params_adam = {
            "ift": optax.apply_updates(p, upd_ift),
            "st": optax.apply_updates(p, upd_st),
        }
        # Plain gradient step, isolating the raw gradient direction.
        params_sgd = {
            "ift": jax.tree.map(lambda a, b: a - learning_rate * b, p, g_ift),
            "st": jax.tree.map(lambda a, b: a - learning_rate * b, p, g_st),
        }

        raw_base, proj_base = outputs(p, x_probe, b_probe)
        obj_base, _, ineqcv_base = quality(proj_base, b_probe)

        out = {}
        for optimiser, branch in (("adam", params_adam), ("sgd", params_sgd)):
            raw_ift, proj_ift = outputs(branch["ift"], x_probe, b_probe)
            raw_st, proj_st = outputs(branch["st"], x_probe, b_probe)
            raw_stats = divergence(raw_ift, raw_st, raw_base)
            proj_stats = divergence(proj_ift, proj_st, proj_base)
            obj_ift, _, ineq_ift = quality(proj_ift, b_probe)
            obj_st, _, ineq_st = quality(proj_st, b_probe)
            out[optimiser] = {
                "raw_dist": raw_stats["dist"],
                "raw_step_ift": raw_stats["step_ift"],
                "raw_relative": raw_stats["relative"],
                "raw_cos": raw_stats["cos"],
                "proj_dist": proj_stats["dist"],
                "proj_step_ift": proj_stats["step_ift"],
                "proj_step_st": proj_stats["step_st"],
                "proj_relative": proj_stats["relative"],
                "proj_cos": proj_stats["cos"],
                # How much of the parameter-update difference survives the layer.
                "contraction": proj_stats["dist"] / (raw_stats["dist"] + EPS),
                "dobj_ift": obj_ift - obj_base,
                "dobj_st": obj_st - obj_base,
                "dineqcv_ift": ineq_ift - ineqcv_base,
                "dineqcv_st": ineq_st - ineqcv_base,
                "norm_y": jnp.linalg.norm(proj_base, axis=1),
            }
        # Parameter-space divergence, for reference.
        for optimiser, branch in (("adam", params_adam), ("sgd", params_sgd)):
            flat_ift = jnp.concatenate(
                [leaf.ravel() for leaf in jax.tree_util.tree_leaves(branch["ift"])]
            )
            flat_st = jnp.concatenate(
                [leaf.ravel() for leaf in jax.tree_util.tree_leaves(branch["st"])]
            )
            flat_base = jnp.concatenate(
                [leaf.ravel() for leaf in jax.tree_util.tree_leaves(p)]
            )
            out[optimiser]["param_dist"] = jnp.linalg.norm(flat_ift - flat_st)
            out[optimiser]["param_step"] = jnp.linalg.norm(flat_ift - flat_base)
        return out

    @jax.jit
    def update(p: dict, state, grads: dict) -> tuple:
        """Apply the reference optimiser step."""
        updates, state = tx.update(grads, state, p)
        return optax.apply_updates(p, updates), state

    for X_valid, _ in valid_loader:
        pass
    probe_batch = X_valid[:N_PROBE_INSTANCES]
    x_probe, b_probe = probe_batch[:, :, 0], probe_batch

    grad_reference = grad_ift if trajectory == "ift" else grad_st

    records, checkpoints = [], {}
    start = time.time()
    step_index = 0
    for epoch in range(hyperparameters["n_epochs"]):
        for X_batch, _ in train_loader:
            x_in, b_in = X_batch[:, :, 0], X_batch
            _, g_ift = grad_ift(params, x_in, b_in)
            _, g_st = grad_st(params, x_in, b_in)

            measured = jax.block_until_ready(
                counterfactual(params, opt_state, g_ift, g_st, x_probe, b_probe)
            )
            row = {"epoch": epoch, "step": step_index}
            for optimiser, values in measured.items():
                for name, value in values.items():
                    array = np.asarray(value)
                    if array.ndim == 0:
                        row[f"{optimiser}/{name}"] = float(array)
                    else:
                        row[f"{optimiser}/{name}_median"] = float(np.median(array))
                        row[f"{optimiser}/{name}_p10"] = float(np.percentile(array, 10))
                        row[f"{optimiser}/{name}_p90"] = float(np.percentile(array, 90))
            records.append(row)

            # Follow the reference trajectory.
            loss, g_reference = grad_reference(params, x_in, b_in)
            params, opt_state = update(params, opt_state, g_reference)
            step_index += 1

        if epoch in CHECKPOINT_EPOCHS:
            checkpoints[epoch] = {
                f"{optimiser}/{name}": np.asarray(value)
                for optimiser, values in measured.items()
                for name, value in values.items()
                if np.asarray(value).ndim > 0
            }
        print(
            f"[probe {trajectory} seed={seed}] epoch {epoch + 1}"
            f"/{hyperparameters['n_epochs']} loss={float(loss):.5f} "
            f"adam proj/raw={records[-1]['adam/contraction_median']:.4f} "
            f"proj_rel={records[-1]['adam/proj_relative_median']:.4f} "
            f"cos={records[-1]['adam/proj_cos_median']:.4f} "
            f"({time.time() - start:.0f}s)",
            flush=True,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    columns = sorted(records[0].keys())
    table = np.array([[row[name] for name in columns] for row in records])
    payload = {
        "columns": np.array(columns),
        "table": table,
        "dataset_id": dataset_id,
        "trajectory": trajectory,
        "seed": seed,
        "config": config,
    }
    for epoch, values in checkpoints.items():
        for name, array in values.items():
            payload[f"checkpoint/{epoch}/{name}"] = array
    out_file = out_dir / f"{dataset_id}_{trajectory}_seed{seed}.npz"
    np.savez_compressed(out_file, **payload)
    print(f"[probe {trajectory} seed={seed}] wrote {out_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-step counterfactual probe of the projection gradient."
    )
    parser.add_argument("--id", type=str, required=True, help="Dataset identifier.")
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_small_autotune",
        help="Hyperparameter configuration.",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default="ift",
        choices=["ift", "st"],
        help="Which gradient drives the reference trajectory.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Training seed.")
    parser.add_argument(
        "--n_epochs", type=int, default=0, help="Override the number of epochs."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(pathlib.Path(__file__).parent / "results" / "update_probe"),
        help="Output directory.",
    )
    args = parser.parse_args()

    main(
        dataset_id=args.id,
        config=args.config,
        trajectory=args.trajectory,
        seed=args.seed,
        n_epochs=args.n_epochs,
        out_dir=pathlib.Path(args.out),
    )
