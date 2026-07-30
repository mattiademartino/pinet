"""Aggregate and plot the one-step counterfactual probe.

Answers, quantitatively: after two different updates from the same parameters,
how different are the *projected* outputs of the two resulting networks?
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Categorical slots: violet/aqua for the optimisers, blue/orange for the updates
# (the latter matching the arm colours used elsewhere in this study).
OPT_COLOR = {"adam": "#4a3aa7", "sgd": "#1baf7a"}
OPT_LABEL = {"adam": "Adam", "sgd": "SGD"}
UPDATE_COLOR = {"ift": "#2a78d6", "st": "#eb6834"}
UPDATE_LABEL = {"ift": "true-gradient update", "st": "straight-through update"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e4e0"
SURFACE = "#fcfcfb"
DATASET_LABEL = {
    "dc3_simple_1": "Convex QP",
    "dc3_nonconvex_1": "Non-convex QP",
}
TRAJECTORY_LABEL = {
    "ift": "true-gradient trajectory",
    "st": "straight-through trajectory",
}


def style() -> None:
    """Apply the shared figure style."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": INK_SECONDARY,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "font.size": 9,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.dpi": 140,
        }
    )


def load_runs(out_dir: pathlib.Path) -> dict:
    """Load probe runs grouped by dataset and trajectory.

    Args:
        out_dir (pathlib.Path): Directory with the ``.npz`` probe results.

    Returns:
        dict: Mapping ``(dataset_id, trajectory) -> list`` of runs.
    """
    runs = {}
    for path in sorted(out_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        key = (str(data["dataset_id"]), str(data["trajectory"]))
        runs.setdefault(key, []).append(data)
    for key in runs:
        runs[key].sort(key=lambda d: int(d["seed"]))
    return runs


def column(run, name: str) -> np.ndarray:
    """Read one named column out of a probe run.

    Args:
        run: Loaded npz handle.
        name (str): Column name.

    Returns:
        np.ndarray: The column values, one per step.
    """
    columns = [str(c) for c in run["columns"]]
    return np.asarray(run["table"])[:, columns.index(name)]


def band(ax, values: np.ndarray, color: str, label: str, log: bool) -> None:
    """Plot the across-seed mean of a per-step quantity with a spread band."""
    steps = np.arange(1, values.shape[1] + 1)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    lower = mean - std
    if log:
        positive = values[values > 0]
        lower = np.maximum(lower, positive.min() * 0.5 if positive.size else 1e-12)
    ax.plot(steps, mean, color=color, linewidth=1.8, label=label, zorder=3)
    ax.fill_between(
        steps, lower, mean + std, color=color, alpha=0.16, linewidth=0, zorder=2
    )
    # Relief rule: direct label at the line end, so identity never rests on colour.
    ax.annotate(
        label,
        xy=(steps[-1], mean[-1]),
        xytext=(4, 0),
        textcoords="offset points",
        color=color,
        fontsize=8,
        fontweight="bold",
        va="center",
    )


def figure_divergence(runs: dict, trajectory: str, out: pathlib.Path) -> None:
    """Plot how the two updates differ, before and after the projection."""
    datasets = [d for d in DATASET_LABEL if (d, trajectory) in runs]
    if not datasets:
        return
    panels = [
        ("contraction", r"$\|\Delta y\| \, / \, \|\Delta \hat{y}_{raw}\|$", True),
        ("proj_relative", r"$\|\Delta y\| \, / \, \|y_{ift} - y_k\|$", True),
        ("proj_cos", r"$\cos(y_{ift}-y_k,\ y_{st}-y_k)$", False),
    ]
    fig, axes = plt.subplots(
        len(datasets), 3, figsize=(12, 3.2 * len(datasets)), squeeze=False
    )
    for row, dataset in enumerate(datasets):
        group = runs[(dataset, trajectory)]
        for col, (metric, title, log) in enumerate(panels):
            ax = axes[row][col]
            for optimiser in ("adam", "sgd"):
                values = np.stack(
                    [column(r, f"{optimiser}/{metric}_median") for r in group]
                )
                band(ax, values, OPT_COLOR[optimiser], OPT_LABEL[optimiser], log)
            if log:
                ax.set_yscale("log")
            else:
                ax.set_ylim(-1.05, 1.05)
                ax.axhline(0.0, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
            if metric == "proj_relative":
                ax.axhline(1.0, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
            ax.set_xlabel("training step")
            ax.set_title(title if row == 0 else "")
            ax.grid(True, axis="y", zorder=0)
            ax.set_axisbelow(True)
            ax.margins(x=0.14)
            if col == 0:
                ax.set_ylabel(DATASET_LABEL.get(dataset, dataset), color=INK)
    fig.suptitle(
        "Effect of the two updates on the network output - "
        f"{TRAJECTORY_LABEL[trajectory]}",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def figure_quality(runs: dict, trajectory: str, out: pathlib.Path) -> None:
    """Plot what each update does to the objective and to feasibility."""
    datasets = [d for d in DATASET_LABEL if (d, trajectory) in runs]
    if not datasets:
        return
    fig, axes = plt.subplots(
        len(datasets), 2, figsize=(9, 3.2 * len(datasets)), squeeze=False
    )
    for row, dataset in enumerate(datasets):
        group = runs[(dataset, trajectory)]
        for col, (metric, title) in enumerate(
            [
                ("dobj", "Change in objective (one update, Adam)"),
                ("dineqcv", "Change in constraint violation"),
            ]
        ):
            ax = axes[row][col]
            for update in ("ift", "st"):
                values = np.stack(
                    [column(r, f"adam/{metric}_{update}_median") for r in group]
                )
                band(ax, values, UPDATE_COLOR[update], UPDATE_LABEL[update], False)
            ax.axhline(0.0, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
            ax.set_yscale("symlog", linthresh=1e-6)
            ax.set_xlabel("training step")
            ax.set_title(title if row == 0 else "")
            ax.grid(True, axis="y", zorder=0)
            ax.set_axisbelow(True)
            ax.margins(x=0.30)
            if col == 0:
                ax.set_ylabel(DATASET_LABEL.get(dataset, dataset), color=INK)
    fig.suptitle(
        f"Output quality after a single update - {TRAJECTORY_LABEL[trajectory]}",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def summarise(runs: dict, out: pathlib.Path) -> str:
    """Build the probe summary table.

    Args:
        runs (dict): Loaded probe runs.
        out (pathlib.Path): Destination markdown file.

    Returns:
        str: The rendered table.
    """
    lines = [
        "| dataset | trajectory | opt | phase | contraction proj/raw | "
        "relative divergence | cosine |",
        "|---|---|---|---|---|---|---|",
    ]
    for (dataset, trajectory), group in sorted(runs.items()):
        n_steps = np.asarray(group[0]["table"]).shape[0]
        phases = {
            "start (first 20 steps)": slice(0, 20),
            "end (last 20 steps)": slice(n_steps - 20, n_steps),
        }
        for optimiser in ("adam", "sgd"):
            for phase_name, window in phases.items():
                contraction = np.concatenate(
                    [
                        column(r, f"{optimiser}/contraction_median")[window]
                        for r in group
                    ]
                )
                relative = np.concatenate(
                    [
                        column(r, f"{optimiser}/proj_relative_median")[window]
                        for r in group
                    ]
                )
                cosine = np.concatenate(
                    [column(r, f"{optimiser}/proj_cos_median")[window] for r in group]
                )
                lines.append(
                    f"| {DATASET_LABEL.get(dataset, dataset)} | {trajectory} "
                    f"| {OPT_LABEL[optimiser]} | {phase_name} "
                    f"| {np.median(contraction):.3f} "
                    f"| {np.median(relative):.2f} "
                    f"| {np.median(cosine):+.3f} |"
                )
    table = "\n".join(lines)
    out.write_text(table + "\n")
    return table


def main() -> None:
    """Parse the probe results and write figures and summary."""
    parser = argparse.ArgumentParser(description="Parse the one-step update probe.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(pathlib.Path(__file__).parent / "results" / "update_probe"),
        help="Directory with the probe results.",
    )
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out)
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    style()
    runs = load_runs(out_dir)
    if not runs:
        raise SystemExit(f"No probe results found in {out_dir}")

    for trajectory in ("ift", "st"):
        figure_divergence(runs, trajectory, figures / f"divergence_{trajectory}.png")
        figure_quality(runs, trajectory, figures / f"quality_{trajectory}.png")

    print(summarise(runs, out_dir / "summary.md"))
    print(f"\nFigures written to {figures}")


if __name__ == "__main__":
    main()
