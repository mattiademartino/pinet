"""Compare training with the true projection gradient against a straight-through one.

Two arms share a bit-identical forward pass and differ only in the backward:

- ``ift``: the projection is differentiated with the implicit function theorem,
  i.e. the custom VJP of :class:`pinet.Project` (the default behaviour).
- ``st``: straight-through. The Jacobian of the projection is replaced by the
  identity, so ``dL/dyraw == dL/dy``. The loss is still evaluated at the projected
  (hence feasible) point; only the backward is altered.

At every optimisation step both gradients are evaluated at the *same* parameters,
so their angular and magnitude discrepancy can be measured while each arm follows
its own trajectory. Only the arm's own gradient is timed and used for the update;
the other one is computed off the clock, purely as a measurement.

Run with, e.g.::

    python -m src.benchmarks.QP.run_grad_ablation --id dc3_simple_1 \
        --config benchmark_small_autotune --arm ift --seed 0
"""

import argparse
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from flax.training import train_state

from benchmarks.model import HardConstrainedMLP
from benchmarks.QP.load_QP import load_data
from pinet import (
    AffineInequalityConstraint,
    EqualityConstraint,
    EqualityConstraintsSpecification,
    EquilibrationParams,
    Project,
    ProjectionInstance,
)
from src.tools.utils import load_configuration

jax.config.update("jax_enable_x64", True)

# Number of validation instances used for the per-instance geometry probe.
N_GEOMETRY_INSTANCES = 256
# Guard against division by zero when a gradient is identically zero.
EPS = 1e-300


def build_projections(
    hyperparameters: dict,
    A: jnp.ndarray,
    X: jnp.ndarray,
    G: jnp.ndarray,
    h: jnp.ndarray,
) -> tuple:
    """Build the differentiable and straight-through projection callables.

    Both callables evaluate exactly the same Douglas-Rachford iteration, so their
    forward outputs are bit-identical. They differ only in what the backward pass
    sees.

    Args:
        hyperparameters (dict): Hyperparameters for the projection layer.
        A (jnp.ndarray): Coefficient matrix of the equality constraint.
        X (jnp.ndarray): Right-hand sides of the equality constraint (the dataset).
        G (jnp.ndarray): Coefficient matrix of the inequality constraint.
        h (jnp.ndarray): Upper bounds of the inequality constraint.

    Returns:
        tuple: ``(project_ift, project_st, project_test)`` where the first is the
            projection with the implicit-function-theorem VJP, the second is its
            straight-through counterpart, and the third is the test-time
            projection (never differentiated).
    """
    eq_constraint = EqualityConstraint(A=A, b=X, method=None, var_b=True)
    ineq_constraint = AffineInequalityConstraint(
        C=G, ub=h, lb=-jnp.inf * jnp.ones_like(h)
    )
    projection_layer = Project(
        eq_constraint=eq_constraint,
        ineq_constraint=ineq_constraint,
        unroll=False,
        equilibration_params=EquilibrationParams(**hyperparameters["equilibrate"]),
    )

    kw = {
        "n_iter_bwd": hyperparameters["n_iter_bwd"],
        "fpi": hyperparameters["fpi"],
    }

    def raw_project(x: jnp.ndarray, b: jnp.ndarray, n_iter: int) -> jnp.ndarray:
        """Run the projection layer on a batch of points."""
        inp = ProjectionInstance(
            x=x[..., None], eq=EqualityConstraintsSpecification(b=b)
        )
        return projection_layer.call(
            yraw=inp,
            sigma=hyperparameters["sigma"],
            omega=hyperparameters["omega"],
            n_iter=n_iter,
            **kw,
        )[0].x[..., 0]

    def project_ift(x: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Project with the true (implicit) gradient."""
        return raw_project(x, b, hyperparameters["n_iter_train"])

    def project_st(x: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Project, but let the backward treat the projection as the identity.

        The input of the projection is detached, so the whole Douglas-Rachford
        subgraph carries no differentiable dependence and its VJP is never built.
        The added term evaluates to exactly zero, so the forward output is
        bit-identical to ``raw_project(x, b)`` while the backward sees the
        identity.
        """
        x_const = jax.lax.stop_gradient(x)
        y_const = jax.lax.stop_gradient(
            raw_project(x_const, b, hyperparameters["n_iter_train"])
        )
        return y_const + (x - x_const)

    def project_test(x: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Project at test time."""
        return raw_project(x, b, hyperparameters["n_iter_test"])

    return project_ift, project_st, project_test


def _pair_stats(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Compare two flat vectors.

    Args:
        a (jnp.ndarray): Reference vector (the IFT gradient).
        b (jnp.ndarray): Vector to compare against the reference (straight-through).

    Returns:
        jnp.ndarray: ``[cosine, ||b||/||a||, ||b - a||/||a||, ||a||, ||b||]``.
    """
    na = jnp.linalg.norm(a)
    nb = jnp.linalg.norm(b)
    cos = jnp.sum(a * b) / (na * nb + EPS)
    return jnp.stack(
        [cos, nb / (na + EPS), jnp.linalg.norm(b - a) / (na + EPS), na, nb]
    )


def make_grad_comparison(params: dict) -> tuple:
    """Build a jitted function comparing two gradient pytrees.

    Args:
        params (dict): A parameter pytree, used only to read off the leaf names.

    Returns:
        tuple: ``(compare, names)`` where ``compare(g_ift, g_st)`` returns a dict
            mapping ``"global"`` and each parameter tensor name to the statistics
            of :func:`_pair_stats`.
    """
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(params)
    names = ["/".join(str(k.key) for k in path) for path, _ in leaves_with_path]

    @jax.jit
    def compare(g_ift: dict, g_st: dict) -> dict:
        """Compare the two gradients globally and per parameter tensor."""
        flat_ift = [leaf.ravel() for leaf in jax.tree_util.tree_leaves(g_ift)]
        flat_st = [leaf.ravel() for leaf in jax.tree_util.tree_leaves(g_st)]
        out = {
            "global": _pair_stats(jnp.concatenate(flat_ift), jnp.concatenate(flat_st))
        }
        for name, a, b in zip(names, flat_ift, flat_st):
            out[name] = _pair_stats(a, b)
        return out

    return compare, names


def make_geometry_probe(model_raw, project_ift, batched_objective):
    """Build a jitted probe of the projection Jacobian at the projection interface.

    This isolates the object the straight-through estimator approximates: it
    replaces ``J_P^T`` by the identity, so we measure how ``J_P^T v`` compares to
    ``v``, per instance, where ``v`` is the cotangent of the objective at the
    projected point.

    Args:
        model_raw: The MLP evaluated without the projection (``raw_test=True``).
        project_ift: The differentiable projection.
        batched_objective: Batched objective function.

    Returns:
        Callable: ``probe(params, x, b)`` returning per-instance cosine, norm
            ratio, ``||v||`` and ``||J_P^T v||``.
    """

    @jax.jit
    def probe(params: dict, x: jnp.ndarray, b: jnp.ndarray) -> tuple:
        """Measure the per-instance action of the projection Jacobian."""
        yraw = model_raw.apply({"params": params}, x=x, b=b, test=True)
        y, vjp_fn = jax.vjp(lambda yy: project_ift(yy, b), yraw)
        v = jax.grad(lambda z: batched_objective(z).sum())(y)
        jtv = vjp_fn(v)[0]
        norm_v = jnp.linalg.norm(v, axis=1)
        norm_jtv = jnp.linalg.norm(jtv, axis=1)
        cos = jnp.sum(v * jtv, axis=1) / (norm_v * norm_jtv + EPS)
        return cos, norm_jtv / (norm_v + EPS), norm_v, norm_jtv

    return probe


def evaluate(
    apply_test,
    params: dict,
    loader,
    batched_objective,
    A: jnp.ndarray,
    G: jnp.ndarray,
    h: jnp.ndarray,
) -> dict:
    """Evaluate a model on a loader that yields the whole split in one batch.

    Mirrors the metrics of ``run_QP.evaluate_hcnn`` so the numbers are comparable.

    Args:
        apply_test: Callable ``(params, X) -> predictions`` at test time.
        params (dict): Model parameters.
        loader: Data loader for the split.
        batched_objective: Batched objective function.
        A (jnp.ndarray): Equality constraint matrix.
        G (jnp.ndarray): Inequality constraint matrix.
        h (jnp.ndarray): Inequality constraint bounds.

    Returns:
        dict: Relative suboptimality, equality and inequality violations.
    """
    for X, obj in loader:
        pass
    predictions = apply_test(params, X)
    hcnn_obj = batched_objective(predictions)
    rs = (hcnn_obj - obj) / jnp.abs(obj)
    eq_cv = jnp.max(
        jnp.abs(
            A[0].reshape(1, A.shape[1], A.shape[2])
            @ predictions.reshape(X.shape[0], A.shape[2], 1)
            - X
        ),
        axis=1,
    )
    ineq_cv = jnp.max(
        jnp.maximum(
            G[0].reshape(1, G.shape[1], G.shape[2])
            @ predictions.reshape(X.shape[0], G.shape[2], 1)
            - h,
            0,
        ),
        axis=1,
    )
    return {
        "rs": np.asarray(rs).ravel(),
        "objective": np.asarray(hcnn_obj).ravel(),
        "optimal_objective": np.asarray(obj).ravel(),
        "eq_cv": np.asarray(eq_cv).ravel(),
        "ineq_cv": np.asarray(ineq_cv).ravel(),
    }


def main(
    dataset_id: str,
    config: str,
    arm: str,
    seed: int,
    n_epochs: int,
    batch_size: int,
    out_dir: pathlib.Path,
) -> None:
    """Run one arm of the ablation.

    Args:
        dataset_id (str): Dataset identifier, see the ``ids`` folder.
        config (str): Hyperparameter configuration name.
        arm (str): Either ``"ift"`` or ``"st"``; selects which gradient updates.
        seed (int): Seed for training.
        n_epochs (int): Number of epochs, overriding the configuration if > 0.
        batch_size (int): Batch size, overriding the configuration if > 0. Small
            datasets need it to be lowered, otherwise an epoch is a single
            update and the arms are compared over too few steps.
        out_dir (pathlib.Path): Where to write the results.
    """
    here = pathlib.Path(__file__).parent.resolve()
    dataset = load_configuration(here / "ids" / (dataset_id + ".yaml"))
    hyperparameters = load_configuration(here.parent / "configs" / (config + ".yaml"))
    if n_epochs > 0:
        hyperparameters["n_epochs"] = n_epochs
    if batch_size > 0:
        hyperparameters["batch_size"] = batch_size

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
    project_ift, project_st, project_test = build_projections(
        hyperparameters, A, X, G, h
    )
    setup_time = time.time() - start_setup

    activation = getattr(jax.nn, hyperparameters["activation"])
    common = {
        "project_test": project_test,
        "dim": A.shape[2],
        "features_list": hyperparameters["features_list"],
        "activation": activation,
    }
    model_ift = HardConstrainedMLP(project=project_ift, **common)
    model_st = HardConstrainedMLP(project=project_st, **common)
    # Same parameters, but the forward stops before the projection. Used by the
    # geometry probe to recover the pre-projection output of the network.
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
    grad_own, grad_other = (grad_ift, grad_st) if arm == "ift" else (grad_st, grad_ift)

    @jax.jit
    def apply_test(p: dict, x_batch: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the model at test time."""
        return model_ift.apply({"params": p}, x=x_batch[:, :, 0], b=x_batch, test=True)

    compare, leaf_names = make_grad_comparison(params)
    probe = make_geometry_probe(model_raw, project_ift, batched_objective)

    tx = optax.adam(hyperparameters["learning_rate"])
    state = train_state.TrainState.create(
        apply_fn=model_ift.apply, params=params, tx=tx
    )

    @jax.jit
    def update(st, grads):
        """Apply one optimiser step."""
        return st.apply_gradients(grads=grads)

    # Fixed probe batch, identical across arms and epochs.
    for X_valid, _ in valid_loader:
        pass
    probe_batch = X_valid[:N_GEOMETRY_INSTANCES]

    # Compilation happens on the first call; time it separately.
    start_compile = time.time()
    _ = jax.block_until_ready(grad_own(state.params, X[:2, :, 0], X[:2]))
    _ = jax.block_until_ready(grad_other(state.params, X[:2, :, 0], X[:2]))
    compilation_time = time.time() - start_compile

    step_records = {name: [] for name in ["global"] + leaf_names}
    step_loss, step_time = [], []
    epoch_rows, geom_cos, geom_ratio = [], [], []
    cumulative_train_time = 0.0

    for epoch in range(hyperparameters["n_epochs"]):
        epoch_losses, epoch_sizes = [], []
        epoch_time = 0.0
        for X_batch, _ in train_loader:
            x_in, b_in = X_batch[:, :, 0], X_batch
            params_before = state.params

            # --- timed: the arm's own gradient and the parameter update ---
            start_step = time.time()
            loss, g_own = grad_own(params_before, x_in, b_in)
            state = update(state, g_own)
            jax.block_until_ready((loss, state.params))
            elapsed = time.time() - start_step
            # --------------------------------------------------------------

            # Measurement only: the other gradient, at the *same* parameters.
            _, g_other = jax.block_until_ready(grad_other(params_before, x_in, b_in))
            g_pair = (g_own, g_other) if arm == "ift" else (g_other, g_own)
            stats = compare(*g_pair)
            for name, value in stats.items():
                step_records[name].append(np.asarray(value))

            epoch_time += elapsed
            step_time.append(elapsed)
            step_loss.append(float(loss))
            epoch_losses.append(float(loss))
            epoch_sizes.append(X_batch.shape[0])

        cumulative_train_time += epoch_time
        valid = evaluate(
            apply_test, state.params, valid_loader, batched_objective, A, G, h
        )
        cos, ratio, _, _ = probe(state.params, probe_batch[:, :, 0], probe_batch)
        geom_cos.append(np.asarray(cos))
        geom_ratio.append(np.asarray(ratio))
        epoch_rows.append(
            [
                float(
                    sum(le * bs for le, bs in zip(epoch_losses, epoch_sizes))
                    / sum(epoch_sizes)
                ),
                float(np.mean(valid["rs"])),
                float(np.max(valid["eq_cv"])),
                float(np.mean(valid["ineq_cv"])),
                float(np.max(valid["ineq_cv"])),
                epoch_time,
                cumulative_train_time,
            ]
        )
        print(
            f"[{arm} seed={seed}] epoch {epoch + 1}/{hyperparameters['n_epochs']} "
            f"loss={epoch_rows[-1][0]:.6f} rs={epoch_rows[-1][1]:.6f} "
            f"ineqcv={epoch_rows[-1][3]:.3e} t={epoch_time:.2f}s "
            f"cos={step_records['global'][-1][0]:.4f} "
            f"ratio={step_records['global'][-1][1]:.4f}",
            flush=True,
        )

    test = evaluate(apply_test, state.params, test_loader, batched_objective, A, G, h)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{dataset_id}_{arm}_seed{seed}.npz"
    np.savez_compressed(
        out_file,
        arm=arm,
        dataset_id=dataset_id,
        config=config,
        seed=seed,
        leaf_names=np.array(leaf_names),
        # stats columns: [cosine, ||g_st||/||g_ift||, rel_err, ||g_ift||, ||g_st||]
        **{f"stats/{name}": np.stack(values) for name, values in step_records.items()},
        step_loss=np.array(step_loss),
        step_time=np.array(step_time),
        # epoch columns: [train_loss, valid_rs, valid_eqcv_max,
        #                 valid_ineqcv_mean, valid_ineqcv_max, epoch_t, cum_t]
        epochs=np.array(epoch_rows),
        geometry_cos=np.stack(geom_cos),
        geometry_ratio=np.stack(geom_ratio),
        test_rs=test["rs"],
        test_eq_cv=test["eq_cv"],
        test_ineq_cv=test["ineq_cv"],
        test_objective=test["objective"],
        test_optimal_objective=test["optimal_objective"],
        setup_time=setup_time,
        compilation_time=compilation_time,
        training_time=cumulative_train_time,
    )
    print(f"[{arm} seed={seed}] wrote {out_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Projection-gradient ablation: IFT vs straight-through."
    )
    parser.add_argument("--id", type=str, required=True, help="Dataset identifier.")
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_small_autotune",
        help="Hyperparameter configuration.",
    )
    parser.add_argument(
        "--arm",
        type=str,
        required=True,
        choices=["ift", "st"],
        help="Which gradient drives the update.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Training seed.")
    parser.add_argument(
        "--n_epochs", type=int, default=0, help="Override the number of epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=0, help="Override the batch size."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(pathlib.Path(__file__).parent / "results" / "grad_ablation"),
        help="Output directory.",
    )
    args = parser.parse_args()

    main(
        dataset_id=args.id,
        config=args.config,
        arm=args.arm,
        seed=args.seed,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        out_dir=pathlib.Path(args.out),
    )
