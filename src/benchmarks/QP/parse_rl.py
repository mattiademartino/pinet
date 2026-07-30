"""Aggregate the RL runs and compare them with the pinet baseline.

Reads the ``.npz`` files written by ``run_rl.py`` (and, when available, the
``ift`` arm of ``run_grad_ablation.py`` as the self-supervised reference) and
produces the figures and the summary table. It never reruns training.

Run with::

    python -m src.benchmarks.QP.parse_rl --out src/benchmarks/QP/results/rl
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.QP.parse_grad_ablation import (
    DATASET_LABEL,
    INK_SECONDARY,
    add_row_label,
    band,
    style,
)

# Ordered so that the two "true gradient" arms come first, in blue, and the
# score-function / learned-critic agents follow. The two SAC variants share a
# hue, the lighter one being the critic restricted to the feasible set.
ALGO_COLOR = {
    "pinet": "#2a78d6",
    "pinet_stochastic": "#7db9f2",
    "reinforce": "#eb6834",
    "ppo": "#12896b",
    "sac": "#8b5cf6",
    "sac_proj": "#c084fc",
}
ALGO_LABEL = {
    "pinet": "pinet (deterministic, true gradient)",
    "pinet_stochastic": "pinet stochastic (sampled action, true gradient)",
    "reinforce": "REINFORCE",
    "ppo": "PPO",
    "sac": "SAC",
    "sac_proj": "SAC (critic on projected actions)",
}
REFERENCE_COLOR = "#2a78d6"
REFERENCE_LABEL = "pinet (self-supervised, IFT)"
DATASET_LABEL = dict(DATASET_LABEL, dc3_simple_small="Convex QP (d = 10)")
# Thresholds of run_QP.py: an instance counts as solved below both.
RS_THRESHOLD = 5e-2
CV_THRESHOLD = 1e-3


def load_runs(out_dir: pathlib.Path) -> dict:
    """Load every RL run, grouped by dataset and algorithm.

    Args:
        out_dir (pathlib.Path): Directory holding the ``.npz`` result files.

    Returns:
        dict: Mapping ``(dataset_id, algo) -> list`` of runs, ordered by seed.
    """
    runs = {}
    for path in sorted(out_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        if "algo" not in data:
            continue
        runs.setdefault((str(data["dataset_id"]), str(data["algo"])), []).append(data)
    for key in runs:
        runs[key].sort(key=lambda d: int(d["seed"]))
    return runs


def load_reference(reference_dir: pathlib.Path) -> dict:
    """Load the pinet baseline, if its results are available.

    Args:
        reference_dir (pathlib.Path): Directory with the ``run_grad_ablation``
            results.

    Returns:
        dict: Mapping ``dataset_id -> dict`` with the mean test suboptimality,
            the maximum violation and the training time of the ``ift`` arm.
    """
    reference = {}
    if not reference_dir.is_dir():
        return reference
    grouped = {}
    for path in sorted(reference_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        if "arm" not in data or str(data["arm"]) != "ift":
            continue
        grouped.setdefault(str(data["dataset_id"]), []).append(data)
    for dataset, items in grouped.items():
        reference[dataset] = {
            "rs": float(np.mean([np.mean(d["test_rs"]) for d in items])),
            "rs_std": float(np.std([np.mean(d["test_rs"]) for d in items])),
            "ineq_cv_max": float(np.max([np.max(d["test_ineq_cv"]) for d in items])),
            "train_time": float(np.mean([float(d["training_time"]) for d in items])),
            "n_seeds": len(items),
        }
    return reference


def curve(runs: list, x_key: str, y_key: str) -> tuple:
    """Stack one evaluation curve across seeds.

    Args:
        runs (list): Runs of one (dataset, algorithm) pair.
        x_key (str): Name of the x-axis array inside the npz.
        y_key (str): Name of the y-axis array inside the npz.

    Returns:
        tuple: ``(x, values)`` with ``values`` of shape (n_seeds, n_points).
    """
    length = min(len(np.asarray(r[y_key])) for r in runs)
    values = np.stack([np.asarray(r[y_key])[:length] for r in runs])
    return np.asarray(runs[0][x_key])[:length], values


def figure_learning(
    runs: dict, datasets: list, reference: dict, out: pathlib.Path
) -> None:
    """Show how every arm converges, on the three budget axes plus feasibility.

    Args:
        runs (dict): Loaded runs.
        datasets (list): Datasets to show, one row each.
        reference (dict): The pinet baseline, drawn as a horizontal line only if
            no ``pinet`` arm was run.
        out (pathlib.Path): Directory where the figure is written.
    """
    panels = [
        (
            "eval/env_samples",
            "eval/rs_mean",
            "environment samples",
            "relative suboptimality",
            True,
            True,
        ),
        (
            "eval/train_time",
            "eval/rs_mean",
            "training time [s]",
            "relative suboptimality",
            False,
            True,
        ),
        (
            "iter/env_samples",
            "iter/reward_mean",
            "environment samples",
            "mean reward  ( = -loss )",
            True,
            False,
        ),
        (
            "eval/env_samples",
            "eval/ineq_cv_max",
            "environment samples",
            "max inequality violation",
            True,
            True,
        ),
    ]
    fig, axes = plt.subplots(
        len(datasets), len(panels), figsize=(17, 3.4 * len(datasets)), squeeze=False
    )
    for row, dataset in enumerate(datasets):
        for column, (x_key, y_key, xlabel, ylabel, logx, logy) in enumerate(panels):
            ax = axes[row][column]
            for algo in ALGO_LABEL:
                key = (dataset, algo)
                if key not in runs or y_key not in runs[key][0]:
                    continue
                x, values = curve(runs[key], x_key, y_key)
                if logy:
                    values = np.abs(values)
                band(ax, x, values, ALGO_COLOR[algo], ALGO_LABEL[algo], log=logy)
            if ylabel.startswith("relative"):
                ax.axhline(
                    RS_THRESHOLD,
                    color=INK_SECONDARY,
                    linestyle=":",
                    linewidth=1.0,
                    label=f"solved threshold ({RS_THRESHOLD:g})",
                )
                if dataset in reference and (dataset, "pinet") not in runs:
                    ax.axhline(
                        reference[dataset]["rs"],
                        color=REFERENCE_COLOR,
                        linestyle="--",
                        linewidth=1.5,
                        label=REFERENCE_LABEL,
                    )
            if ylabel.startswith("max inequality"):
                ax.axhline(
                    CV_THRESHOLD,
                    color=INK_SECONDARY,
                    linestyle=":",
                    linewidth=1.0,
                    label=f"feasibility tolerance ({CV_THRESHOLD:g})",
                )
            if logx:
                ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.6)
        add_row_label(fig, row, len(datasets), DATASET_LABEL.get(dataset, dataset))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 6),
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Convergence of every arm on the QP: same environment, same evaluation",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0.03, 0.03, 1, 0.96])
    fig.savefig(out / "learning.png", bbox_inches="tight")
    plt.close(fig)


def figure_diagnostics(runs: dict, datasets: list, out: pathlib.Path) -> None:
    """Plot the estimator diagnostics that explain the learning curves.

    Args:
        runs (dict): Loaded runs.
        datasets (list): Datasets to show, one row each.
        out (pathlib.Path): Directory where the figure is written.
    """
    panels = [
        ("iter/grad_cosine", "half-batch gradient cosine", False),
        ("iter/explained_variance", "baseline explained variance", False),
        ("eval/log_std_mean", "mean policy log-std", False),
        ("eval/ineq_cv_max", "max inequality violation", True),
    ]
    fig, axes = plt.subplots(
        len(datasets), len(panels), figsize=(15, 3.2 * len(datasets)), squeeze=False
    )
    for row, dataset in enumerate(datasets):
        for column, (key, ylabel, logy) in enumerate(panels):
            ax = axes[row][column]
            x_key = "eval/env_samples" if key.startswith("eval") else "iter/env_samples"
            for algo in ALGO_LABEL:
                run_key = (dataset, algo)
                if run_key not in runs or key not in runs[run_key][0]:
                    continue
                values = np.asarray(runs[run_key][0][key])
                if not values.size or np.all(np.isnan(values)):
                    continue
                x, stacked = curve(runs[run_key], x_key, key)
                band(ax, x, stacked, ALGO_COLOR[algo], ALGO_LABEL[algo], log=logy)
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("environment samples")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.6)
            if key == "iter/grad_cosine":
                ax.axhline(0.0, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
            if row == 0 and column == 0:
                ax.legend(fontsize=8)
        add_row_label(fig, row, len(datasets), DATASET_LABEL.get(dataset, dataset))
    fig.suptitle(
        "Why the curves look the way they do: estimator diagnostics",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0.03, 0, 1, 0.96])
    fig.savefig(out / "diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def summarise(runs: dict, datasets: list, reference: dict, out: pathlib.Path) -> str:
    """Build the markdown summary table.

    Args:
        runs (dict): Loaded runs.
        datasets (list): Datasets to include.
        reference (dict): The pinet baseline.
        out (pathlib.Path): Directory where ``summary.md`` is written.

    Returns:
        str: The markdown table.
    """
    lines = [
        f"Solved = relative suboptimality < {RS_THRESHOLD:g} and inequality "
        f"violation < {CV_THRESHOLD:g}, per test instance.\n",
        "| dataset | method | solved | test RS | test CV (max) | train time [s] "
        "| env samples | grad steps |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for dataset in datasets:
        label = DATASET_LABEL.get(dataset, dataset)
        if dataset in reference and (dataset, "pinet") not in runs:
            ref = reference[dataset]
            lines.append(
                f"| {label} | {REFERENCE_LABEL} | — | "
                f"{ref['rs']:.4f} ± {ref['rs_std']:.4f} | "
                f"{ref['ineq_cv_max']:.2e} | {ref['train_time']:.1f} | — | — |"
            )
        for algo in ALGO_LABEL:
            key = (dataset, algo)
            if key not in runs:
                continue
            items = runs[key]
            rs = np.array([float(d["test_rs_mean"]) for d in items])
            cv = np.max([float(d["test_ineq_cv_max"]) for d in items])
            time = np.mean([float(d["train_time"]) for d in items])
            samples = int(np.mean([int(d["env_samples"]) for d in items]))
            steps = int(np.mean([int(d["gradient_steps"]) for d in items]))
            solved = "—"
            if all("test_ineq_cv" in d for d in items):
                flags = np.concatenate(
                    [
                        (np.asarray(d["test_rs"]) < RS_THRESHOLD)
                        & (np.asarray(d["test_ineq_cv"]) < CV_THRESHOLD)
                        for d in items
                    ]
                )
                solved = f"{100 * flags.mean():.1f} %"
            lines.append(
                f"| {label} | {ALGO_LABEL[algo]} | {solved} "
                f"| {rs.mean():.4f} ± {rs.std():.4f} "
                f"| {cv:.2e} | {time:.1f} | {samples} | {steps} |"
            )
    table = "\n".join(lines) + "\n"
    (out.parent / "summary.md").write_text(table)
    return table


def main() -> None:
    """Parse the RL results and write figures and summary."""
    parser = argparse.ArgumentParser(description="Aggregate the RL runs.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(pathlib.Path(__file__).parent / "results" / "rl"),
        help="Directory with the RL results.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=str(pathlib.Path(__file__).parent / "results" / "grad_ablation"),
        help="Directory with the pinet baseline results.",
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    runs = load_runs(out_dir)
    if not runs:
        raise SystemExit(f"No RL results found in {out_dir}")
    reference = load_reference(pathlib.Path(args.reference))
    datasets = []
    for dataset, _ in runs:
        if dataset not in datasets:
            datasets.append(dataset)
    datasets.sort(
        key=lambda d: list(DATASET_LABEL).index(d) if d in DATASET_LABEL else 99
    )

    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    style()
    figure_learning(runs, datasets, reference, figures)
    figure_diagnostics(runs, datasets, figures)
    print(summarise(runs, datasets, reference, figures))
    print(f"figures written to {figures}")


if __name__ == "__main__":
    main()
