"""Generate a small convex QP in the DC3 dataset format.

The instance follows the same recipe as ``generate_QP.py`` -- diagonal ``Q``,
random ``A`` and ``G``, and ``h`` chosen so that every right-hand side in
``[-1, 1]`` admits a feasible point -- but at a size where an RL agent can be
expected to converge. It is meant as a sanity check: if an agent fails here, the
problem is the implementation, not the dimensionality.

The output is written as the train/valid/test triple that ``DC3_dataset_setup``
expects, so it can be used through the usual ``ids`` mechanism.

Run with::

    python -m src.benchmarks.QP.generate_small_QP
"""

import argparse
import os

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)


def solve_instances(
    Q: np.ndarray,
    p: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    """Solve every instance of the parametric QP with OSQP.

    Args:
        Q (np.ndarray): Quadratic term, shape (dim, dim).
        p (np.ndarray): Linear term, shape (dim,).
        A (np.ndarray): Equality matrix, shape (n_eq, dim).
        X (np.ndarray): Right-hand sides, shape (N, n_eq, 1).
        G (np.ndarray): Inequality matrix, shape (n_ineq, dim).
        h (np.ndarray): Inequality bounds, shape (n_ineq,).

    Returns:
        np.ndarray: Optimal solutions, shape (N, dim, 1).
    """
    dim = Q.shape[0]
    solutions = np.zeros((X.shape[0], dim, 1))
    y = cp.Variable(dim)
    rhs = cp.Parameter(A.shape[0])
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(y, Q) + p @ y),
        [A @ y == rhs, G @ y <= h],
    )
    for index in tqdm(range(X.shape[0])):
        rhs.value = X[index, :, 0]
        problem.solve(solver=cp.OSQP)
        solutions[index, :, 0] = y.value
    return solutions


def main(
    seed: int, dim: int, n_ineq: int, n_eq: int, n_examples: int, out_dir: str
) -> None:
    """Generate and save the dataset.

    Args:
        seed (int): Seed of the problem generation.
        dim (int): Number of decision variables.
        n_ineq (int): Number of inequality constraints.
        n_eq (int): Number of equality constraints.
        n_examples (int): Total number of instances, split 80/10/10.
        out_dir (str): Directory where the three files are written.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), 5)
    Q = jnp.expand_dims(
        jnp.diag(jax.random.uniform(keys[0], shape=(dim,), minval=0.0, maxval=1.0)),
        axis=0,
    )
    p = jax.random.uniform(keys[1], shape=(1, dim, 1), minval=0.0, maxval=1.0)
    A = jax.random.normal(keys[2], shape=(1, n_eq, dim))
    X = jax.random.uniform(
        keys[3], shape=(n_examples, n_eq, 1), minval=-1.0, maxval=1.0
    )
    G = jax.random.normal(keys[4], shape=(1, n_ineq, dim))
    h = jnp.expand_dims(jnp.sum(jnp.abs(G @ jnp.linalg.pinv(A[0])), axis=1), axis=2)

    Ystar = solve_instances(
        np.asarray(Q[0]),
        np.asarray(p[0, :, 0]),
        np.asarray(A[0]),
        np.asarray(X),
        np.asarray(G[0]),
        np.asarray(h[0, :, 0]),
    )

    n_valid = n_examples // 10
    n_test = n_examples // 10
    n_train = n_examples - n_valid - n_test
    splits = {
        "train": slice(0, n_train),
        "valid": slice(n_train, n_train + n_valid),
        "test": slice(n_train + n_valid, n_examples),
    }
    os.makedirs(out_dir, exist_ok=True)
    stem = (
        f"dc3_random_simple_dataset_var{dim}_ineq{n_ineq}" f"_eq{n_eq}_ex{n_examples}"
    )
    for name, index in splits.items():
        path = os.path.join(out_dir, f"{stem}{name}.npz")
        np.savez(
            path,
            Q=np.asarray(Q),
            p=np.asarray(p),
            A=np.asarray(A),
            G=np.asarray(G),
            h=np.asarray(h),
            X=np.asarray(X[index]),
            Ystar=Ystar[index],
        )
        print(f"wrote {path} ({X[index].shape[0]} instances)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a small convex QP.")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed.")
    parser.add_argument("--dim", type=int, default=10, help="Decision variables.")
    parser.add_argument("--n_ineq", type=int, default=5, help="Inequalities.")
    parser.add_argument("--n_eq", type=int, default=5, help="Equalities.")
    parser.add_argument(
        "--n_examples", type=int, default=2000, help="Number of instances."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "datasets"),
        help="Output directory.",
    )
    args = parser.parse_args()
    main(
        seed=args.seed,
        dim=args.dim,
        n_ineq=args.n_ineq,
        n_eq=args.n_eq,
        n_examples=args.n_examples,
        out_dir=args.out,
    )
