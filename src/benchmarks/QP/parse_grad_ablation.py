"""Aggregate and plot the projection-gradient ablation.

Reads the per-run ``.npz`` files written by ``run_grad_ablation.py`` and produces
the figures and the summary table comparing the true (implicit) projection
gradient against the straight-through one.
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Categorical slots 1 and 2 of the reference palette (validated for light mode).
ARM_COLOR = {"ift": "#2a78d6", "st": "#eb6834"}
ARM_LABEL = {
    "ift": "IFT update (implicit gradient)",
    "st": "ST update (Jacobian = identity)",
}
ARM_SHORT_LABEL = {"ift": "IFT trajectory", "st": "ST trajectory"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e4e0"
SURFACE = "#fcfcfb"
DATASET_LABEL = {
    "dc3_simple_small": "Convex QP (d = 10)",
    "dc3_simple_1": "Convex QP (d = 100)",
    "dc3_nonconvex_1": "Non-convex QP (d = 100)",
}
# One line style per dataset, so the datasets stay distinguishable in the panels
# where colour already encodes the arm.
DATASET_LINESTYLE = ["-", "--", ":"]
# Column layout of the per-step statistics arrays.
COS, RATIO, RELERR, NORM_IFT, NORM_ST = range(5)
# Column layout of the per-epoch array.
(
    TRAIN_LOSS,
    VALID_RS,
    VALID_EQCV,
    VALID_INEQCV_MEAN,
    VALID_INEQCV_MAX,
    EPOCH_T,
    CUM_T,
) = range(7)


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
            "axes.titlesize": 11,
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


def add_row_label(fig, row: int, n_rows: int, dataset: str) -> None:
    """Add a readable dataset label outside one row of a grid of panels."""
    y = 1 - (row + 0.5) / n_rows
    fig.text(
        0.012,
        y,
        DATASET_LABEL.get(dataset, dataset),
        rotation=90,
        va="center",
        ha="left",
        color=INK,
        fontweight="bold",
        fontsize=9,
    )


def load_runs(out_dir: pathlib.Path) -> dict:
    """Load every run, grouped by dataset and arm.

    Args:
        out_dir (pathlib.Path): Directory holding the ``.npz`` result files.

    Returns:
        dict: Mapping ``(dataset_id, arm) -> list`` of loaded npz handles,
            ordered by seed.
    """
    runs = {}
    for path in sorted(out_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        runs.setdefault((str(data["dataset_id"]), str(data["arm"])), []).append(data)
    for key in runs:
        runs[key].sort(key=lambda d: int(d["seed"]))
    return runs


def band(ax, x, values: np.ndarray, color: str, label: str, log: bool = False) -> None:
    """Plot the mean over seeds with a standard-deviation band.

    Args:
        ax: Matplotlib axes.
        x: Shared x coordinates.
        values (np.ndarray): Array of shape ``(n_seeds, n_points)``.
        color (str): Line colour.
        label (str): Series label.
        log (bool): Whether the y axis is logarithmic.
    """
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    lower = mean - std
    if log:
        lower = np.maximum(lower, np.min(values[values > 0]) * 0.5)
    ax.plot(x, mean, color=color, linewidth=1.8, label=label, zorder=3)
    ax.fill_between(
        x, lower, mean + std, color=color, alpha=0.16, linewidth=0, zorder=2
    )


def stack(runs: list, key: str, column: int = None) -> np.ndarray:
    """Stack one quantity across seeds.

    Args:
        runs (list): Runs for one (dataset, arm) pair.
        key (str): Array name inside the npz.
        column (int): Column to select, if the array is 2D.

    Returns:
        np.ndarray: Array of shape ``(n_seeds, n_points)``.
    """
    if column is None:
        return np.stack([np.asarray(r[key]) for r in runs])
    return np.stack([np.asarray(r[key])[:, column] for r in runs])


def figure_training(runs: dict, datasets: list, out: pathlib.Path) -> None:
    """Show how the optimization outcome evolves for the two update rules."""
    fig, axes = plt.subplots(
        len(datasets), 3, figsize=(12, 3.2 * len(datasets)), squeeze=False
    )
    panels = [
        ("Mean training loss", TRAIN_LOSS, False, "loss"),
        (
            "Relative distance from the optimum (validation)",
            VALID_RS,
            True,
            "|relative suboptimality|",
        ),
        (
            "Maximum constraint violation (validation)",
            VALID_INEQCV_MAX,
            True,
            "maximum violation",
        ),
    ]
    for row, dataset in enumerate(datasets):
        add_row_label(fig, row, len(datasets), dataset)
        for col, (title, column, log, ylabel) in enumerate(panels):
            ax = axes[row][col]
            for arm in ("ift", "st"):
                if (dataset, arm) not in runs:
                    continue
                values = stack(runs[(dataset, arm)], "epochs", column)
                epochs = np.arange(1, values.shape[1] + 1)
                if log:
                    values = np.abs(values)
                band(ax, epochs, values, ARM_COLOR[arm], ARM_LABEL[arm], log)
            if log:
                ax.set_yscale("log")
            elif column == TRAIN_LOSS:
                # The straight-through arm can blow up by six orders of
                # magnitude; symlog keeps the converging arm readable too.
                ax.set_yscale("symlog", linthresh=10)
            ax.grid(True, axis="y", zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(title if row == 0 else "")
            if row == 0 and col == 0:
                ax.legend(loc="upper right")
    fig.suptitle(
        "Effect of the backward rule on training",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.5,
        0.012,
        "Line = mean over 5 seeds; band = ±1 standard deviation.  "
        "The last two columns use a logarithmic scale: lower is better.",
        ha="center",
        color=INK_SECONDARY,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.035, 0.045, 1, 0.96))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def figure_walltime(runs: dict, datasets: list, out: pathlib.Path) -> None:
    """Compare solution quality per elapsed training time and update cost."""
    fig, axes = plt.subplots(
        1, len(datasets) + 1, figsize=(4.0 * (len(datasets) + 1), 3.4), squeeze=False
    )
    for col, dataset in enumerate(datasets):
        ax = axes[0][col]
        for arm in ("ift", "st"):
            if (dataset, arm) not in runs:
                continue
            group = runs[(dataset, arm)]
            rs = np.abs(stack(group, "epochs", VALID_RS))
            time_axis = stack(group, "epochs", CUM_T).mean(axis=0)
            band(ax, time_axis, rs, ARM_COLOR[arm], ARM_LABEL[arm], log=True)
        ax.set_yscale("log")
        ax.set_xlabel("cumulative training time [s]")
        ax.set_ylabel("|relative suboptimality|" if col == 0 else "")
        ax.set_title(DATASET_LABEL.get(dataset, dataset))
        ax.grid(True, axis="y", zorder=0)
        ax.set_axisbelow(True)
        if col == 0:
            ax.legend(loc="upper right")

    ax = axes[0][-1]
    positions, labels, colors = [], [], []
    data = []
    for index, dataset in enumerate(datasets):
        for offset, arm in enumerate(("ift", "st")):
            if (dataset, arm) not in runs:
                continue
            steps = stack(runs[(dataset, arm)], "step_time").ravel() * 1e3
            data.append(steps)
            positions.append(index * 2.6 + offset)
            labels.append({"ift": "IFT", "st": "ST"}[arm])
            colors.append(ARM_COLOR[arm])
    box = ax.boxplot(
        data, positions=positions, widths=0.7, patch_artist=True, showfliers=False
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.4)
    for element in ("medians", "whiskers", "caps"):
        for line, color in zip(
            box[element],
            np.repeat(colors, 2 if element != "medians" else 1),
        ):
            line.set_color(color)
            line.set_linewidth(1.6)
    # Direct labels: identity must never rest on colour alone.
    for position, values, color, name in zip(positions, data, colors, labels):
        ax.annotate(
            name,
            xy=(position, np.percentile(values, 75)),
            xytext=(0, 6),
            textcoords="offset points",
            color=color,
            fontsize=8,
            fontweight="bold",
            ha="center",
        )
    ax.set_xticks([index * 2.6 + 0.5 for index in range(len(datasets))])
    ax.set_xticklabels(
        [DATASET_LABEL.get(d, d).split(" (")[0] for d in datasets], fontsize=8
    )
    ax.set_ylabel("time per update [ms]")
    ax.set_title("Computational cost of one update")
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    fig.suptitle(
        "Speed: solution quality over time and cost of each update",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.5,
        0.012,
        "Curves: mean ± standard deviation over 5 seeds (lower is better).  "
        "Boxplot: 25th–75th percentiles, inner line = median; outliers are hidden.",
        ha="center",
        color=INK_SECONDARY,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def figure_gradients(runs: dict, datasets: list, out: pathlib.Path) -> None:
    """Explain how the IFT and ST gradients disagree at identical parameters."""
    fig, axes = plt.subplots(
        len(datasets), 3, figsize=(12, 3.2 * len(datasets)), squeeze=False
    )
    panels = [
        ("Directional agreement", COS, False, r"cosine $\cos(g_{IFT}, g_{ST})$"),
        (
            "Magnitude agreement",
            RATIO,
            True,
            r"ratio $\|g_{ST}\| / \|g_{IFT}\|$",
        ),
        ("Norms of both gradients", None, True, "gradient norm"),
    ]
    for row, dataset in enumerate(datasets):
        add_row_label(fig, row, len(datasets), dataset)
        for col, (title, column, log, ylabel) in enumerate(panels):
            ax = axes[row][col]
            for arm in ("ift", "st"):
                if (dataset, arm) not in runs:
                    continue
                group = runs[(dataset, arm)]
                steps = np.arange(1, np.asarray(group[0]["stats/global"]).shape[0] + 1)
                if column is not None:
                    values = stack(group, "stats/global", column)
                    band(
                        ax,
                        steps,
                        values,
                        ARM_COLOR[arm],
                        ARM_SHORT_LABEL[arm],
                        log,
                    )
                else:
                    for kind, style_kw in (
                        (NORM_IFT, {"linestyle": "-"}),
                        (NORM_ST, {"linestyle": "--"}),
                    ):
                        values = stack(group, "stats/global", kind).mean(axis=0)
                        ax.plot(
                            steps,
                            values,
                            color=ARM_COLOR[arm],
                            linewidth=1.6,
                            **style_kw,
                        )
            if log:
                ax.set_yscale("log")
            if col == 0:
                ax.axhline(1.0, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
                ax.set_ylim(-0.15, 1.05)
            if col == 1:
                ax.axhline(1.0, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
            ax.set_xlabel("training step")
            ax.set_ylabel(ylabel)
            ax.set_title(title if row == 0 else "")
            ax.grid(True, axis="y", zorder=0)
            ax.set_axisbelow(True)
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=8)
            if row == 0 and col == 2:
                ax.plot(
                    [], [], color=INK_SECONDARY, linestyle="-", label=r"$\|g_{ift}\|$"
                )
                ax.plot(
                    [], [], color=INK_SECONDARY, linestyle="--", label=r"$\|g_{st}\|$"
                )
                ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "How different are the IFT and straight-through gradients?\n"
        "Both are measured at the same parameters before the update.",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.5,
        0.012,
        "For cosine and ratio, the dotted line at 1 indicates perfect agreement.  "
        "Line/band = mean ± standard deviation over 5 seeds; colors = update trajectory.",
        ha="center",
        color=INK_SECONDARY,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.035, 0.045, 1, 0.96))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _moving_mean(values: np.ndarray, window: int = 25) -> np.ndarray:
    """Return a centred moving average, preserving the length of ``values``."""
    if window < 1:
        raise ValueError("window must be positive")
    if values.size < window:
        window = values.size
    kernel = np.full(window, 1.0 / window)
    # ``same`` pads with zeros, which biases the ends of the curve.  Pad with
    # edge values instead, so the smoothing has the same interpretation there.
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def figure_analysis(runs: dict, datasets: list, out: pathlib.Path) -> None:
    """Recreate the detailed per-update gradient diagnostic requested for the ablation.

    The result files retain scalar comparisons, rather than full gradient
    vectors.  All five diagnostics below can nevertheless be reconstructed
    exactly from ``cos(g_IFT, g_ST)``, ``||g_ST|| / ||g_IFT||``, and the stored
    relative error.  Each colour denotes the trajectory at which the pair was
    measured; faint lines are the mean over seeds at each update and solid
    lines are a 25-update moving average.
    """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 14))
    grid = GridSpec(4, 2, figure=fig, height_ratios=(1, 1, 1, 0.9))
    axes = [fig.add_subplot(grid[row, col]) for row in range(3) for col in range(2)]
    validation_ax = fig.add_subplot(grid[3, :])

    def pair_values(group: list) -> dict[str, np.ndarray]:
        stats = np.stack([np.asarray(run["stats/global"]) for run in group])
        cosine = np.clip(stats[:, :, COS], -1.0, 1.0)
        ratio = stats[:, :, RATIO]
        relative_error = stats[:, :, RELERR]
        return {
            # 2||a-b|| / (||a||+||b||), with a=g_IFT and b=g_ST.
            "symmetric": 2 * relative_error / (1 + ratio),
            "relative": relative_error,
            "angle": np.degrees(np.arccos(cosine)),
            # The unnormalised parallel component, retained separately because
            # it makes changes in gradient scale visible.
            "dot": stats[:, :, NORM_IFT] ** 2 * (1 - ratio * cosine),
            # g_IFT^T(g_IFT-g_ST) / ||g_IFT||^2 = 1-ratio*cosine.
            "parallel": 1 - ratio * cosine,
            "loss": np.stack([np.asarray(run["step_loss"]) for run in group]),
        }

    panels = [
        ("Symmetric relative difference", "symmetric relative difference", "symmetric"),
        (
            r"$\|g_{IFT} - g_{ST}\|_2 / \|g_{IFT}\|_2$",
            "relative difference",
            "relative",
        ),
        ("Angle between gradients", "angle (degrees)", "angle"),
        (
            r"$g_{IFT}^T(g_{IFT}-g_{ST})$",
            "dot product",
            "dot",
        ),
        (
            r"$g_{IFT}^T(g_{IFT}-g_{ST}) / \|g_{IFT}\|_2^2$",
            "parallel difference",
            "parallel",
        ),
        ("Training loss", "training loss", "loss"),
    ]

    for index, (title, ylabel, key) in enumerate(panels):
        ax = axes[index]
        for dataset_index, dataset in enumerate(datasets):
            linestyle = DATASET_LINESTYLE[dataset_index % len(DATASET_LINESTYLE)]
            for arm in ("ift", "st"):
                group = runs.get((dataset, arm))
                if not group:
                    continue
                values = pair_values(group)[key].mean(axis=0)
                steps = np.arange(1, values.size + 1)
                color = ARM_COLOR[arm]
                prefix = f"{DATASET_LABEL.get(dataset, dataset).split(' (')[0]} — "
                ax.plot(
                    steps,
                    values,
                    color=color,
                    alpha=0.14,
                    linewidth=0.9,
                    linestyle=linestyle,
                )
                ax.plot(
                    steps,
                    _moving_mean(values),
                    color=color,
                    linewidth=1.8,
                    linestyle=linestyle,
                    label=prefix + ARM_SHORT_LABEL[arm],
                )
        ax.set_title(title)
        ax.set_xlabel("training update")
        ax.set_ylabel(ylabel)
        ax.grid(True, zorder=0)
        ax.set_axisbelow(True)
        if index == 0:
            ax.legend(loc="upper left", fontsize=7, ncols=2)
        if key in {"dot", "parallel", "loss"}:
            ax.set_yscale("symlog", linthresh=0.1)

    for dataset_index, dataset in enumerate(datasets):
        linestyle = DATASET_LINESTYLE[dataset_index % len(DATASET_LINESTYLE)]
        for arm in ("ift", "st"):
            group = runs.get((dataset, arm))
            if not group:
                continue
            values = np.abs(stack(group, "epochs", VALID_RS)).mean(axis=0)
            epochs = np.arange(1, values.size + 1)
            color = ARM_COLOR[arm]
            validation_ax.plot(
                epochs,
                values,
                color=color,
                alpha=0.14,
                linewidth=0.9,
                linestyle=linestyle,
            )
            validation_ax.plot(
                epochs,
                _moving_mean(values, window=5),
                color=color,
                linewidth=1.8,
                linestyle=linestyle,
                label=(
                    DATASET_LABEL.get(dataset, dataset).split(" (")[0]
                    + " — "
                    + ARM_SHORT_LABEL[arm]
                ),
            )
    validation_ax.set_yscale("log")
    validation_ax.set_title("Validation relative suboptimality")
    validation_ax.set_xlabel("training epoch")
    validation_ax.set_ylabel("|relative suboptimality|")
    validation_ax.grid(True, zorder=0)
    validation_ax.set_axisbelow(True)
    validation_ax.legend(loc="upper right", fontsize=7, ncols=2)

    fig.suptitle(
        "IFT vs straight-through: detailed gradient analysis",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def figure_per_layer(runs: dict, dataset: str, out: pathlib.Path) -> None:
    """Localize the directional gradient mismatch in every parameter tensor."""
    key = (dataset, "ift")
    if key not in runs:
        return
    names = [str(n) for n in runs[key][0]["leaf_names"]]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), squeeze=False)
    for index, name in enumerate(names):
        ax = axes[index // 3][index % 3]
        for arm in ("ift", "st"):
            if (dataset, arm) not in runs:
                continue
            group = runs[(dataset, arm)]
            values = stack(group, f"stats/{name}", COS)
            steps = np.arange(1, values.shape[1] + 1)
            band(ax, steps, values, ARM_COLOR[arm], ARM_SHORT_LABEL[arm])
        ax.axhline(1.0, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
        ax.set_ylim(-0.35, 1.05)
        ax.set_title(name)
        ax.grid(True, axis="y", zorder=0)
        ax.set_axisbelow(True)
        if index // 3 == 1:
            ax.set_xlabel("training step")
        if index % 3 == 0:
            ax.set_ylabel(r"cosine $\cos(g_{IFT}, g_{ST})$")
        if index == 0:
            ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "Where does the directional gradient mismatch originate?\n"
        "Cosine computed separately for each parameter tensor — "
        f"{DATASET_LABEL.get(dataset, dataset)}",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.5,
        0.012,
        "The dotted line at 1 indicates parallel gradients.  "
        "Line/band = mean ± standard deviation over 5 seeds.",
        ha="center",
        color=INK_SECONDARY,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def figure_jacobian(runs: dict, datasets: list, out: pathlib.Path) -> None:
    """Show the action of the projection Jacobian that ST replaces by identity."""
    fig, axes = plt.subplots(
        len(datasets), 3, figsize=(12, 3.2 * len(datasets)), squeeze=False
    )
    for row, dataset in enumerate(datasets):
        add_row_label(fig, row, len(datasets), dataset)
        for col, (key, title, ylabel) in enumerate(
            [
                (
                    "geometry_cos",
                    r"Direction preserved by $J_P^\top$",
                    r"median $\cos(J_P^\top v, v)$",
                ),
                (
                    "geometry_ratio",
                    r"Magnitude preserved by $J_P^\top$",
                    r"median $\|J_P^\top v\| / \|v\|$",
                ),
            ]
        ):
            ax = axes[row][col]
            for arm in ("ift", "st"):
                if (dataset, arm) not in runs:
                    continue
                group = runs[(dataset, arm)]
                values = np.stack([np.asarray(r[key]) for r in group])
                median = np.median(values, axis=(0, 2))
                low = np.percentile(values, 10, axis=(0, 2))
                high = np.percentile(values, 90, axis=(0, 2))
                epochs = np.arange(1, median.shape[0] + 1)
                ax.plot(
                    epochs,
                    median,
                    color=ARM_COLOR[arm],
                    linewidth=1.8,
                    label=ARM_SHORT_LABEL[arm],
                )
                ax.fill_between(
                    epochs, low, high, color=ARM_COLOR[arm], alpha=0.16, linewidth=0
                )
            if col == 0:
                ax.set_ylim(-0.1, 1.02)
            else:
                ax.set_ylim(0, 1.02)
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(title if row == 0 else "")
            ax.grid(True, axis="y", zorder=0)
            ax.set_axisbelow(True)
            if col == 0:
                if row == 0:
                    ax.legend(loc="lower left", fontsize=8)

        # Third panel: the projector signature, cosine against norm ratio.
        ax = axes[row][2]
        for arm in ("ift", "st"):
            if (dataset, arm) not in runs:
                continue
            group = runs[(dataset, arm)]
            cos = np.concatenate([np.asarray(r["geometry_cos"]).ravel() for r in group])
            ratio = np.concatenate(
                [np.asarray(r["geometry_ratio"]).ravel() for r in group]
            )
            ax.scatter(ratio, cos, s=4, color=ARM_COLOR[arm], alpha=0.25, linewidths=0)
        x_limits = [0, 1.02]
        y_limits = [-0.6, 1.02]
        ax.plot(x_limits, x_limits, color=INK_SECONDARY, linewidth=1.0, linestyle=":")
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_xlabel(r"ratio $\|J_P^\top v\| / \|v\|$")
        ax.set_ylabel(r"cosine $\cos(J_P^\top v, v)$")
        ax.set_title("Orthogonal-projector signature" if row == 0 else "")
        ax.grid(True, zorder=0)
        ax.set_axisbelow(True)
    fig.suptitle(
        "How much does the projection Jacobian differ from the identity?\n"
        "Straight-through replaces $J_P^\top$ with the identity in the backward pass.",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.5,
        0.012,
        "First two columns: line = median over instances and seeds, "
        "band = 10th–90th percentile.  "
        "On the right, every point is one instance: on the diagonal "
        "cosine = ratio, as for an orthogonal projector.",
        ha="center",
        color=INK_SECONDARY,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.035, 0.045, 1, 0.96))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def summarise(runs: dict, datasets: list, out: pathlib.Path) -> str:
    """Build the summary table and write it to disk.

    Args:
        runs (dict): Loaded runs.
        datasets (list): Dataset identifiers, in display order.
        out (pathlib.Path): Destination markdown file.

    Returns:
        str: The rendered table.
    """
    lines = [
        "| dataset | arm | test RS | test CV (max) | test CV (mean) | "
        "train time [s] | t/step [ms] | final cosine | final norm ratio |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for dataset in datasets:
        for arm in ("ift", "st"):
            if (dataset, arm) not in runs:
                continue
            group = runs[(dataset, arm)]
            rs = np.array([np.mean(r["test_rs"]) for r in group])
            cv = np.array(
                [np.max(np.maximum(r["test_eq_cv"], r["test_ineq_cv"])) for r in group]
            )
            cv_mean = np.array(
                [np.mean(np.maximum(r["test_eq_cv"], r["test_ineq_cv"])) for r in group]
            )
            train_time = np.array([float(r["training_time"]) for r in group])
            step_time = (
                np.concatenate([np.asarray(r["step_time"]) for r in group]) * 1e3
            )
            tail_cos = np.concatenate(
                [np.asarray(r["stats/global"])[-20:, COS] for r in group]
            )
            tail_ratio = np.concatenate(
                [np.asarray(r["stats/global"])[-20:, RATIO] for r in group]
            )
            lines.append(
                f"| {DATASET_LABEL.get(dataset, dataset)} | {ARM_LABEL[arm]} "
                f"| {rs.mean():.4f} ± {rs.std():.4f} "
                f"| {cv.mean():.2e} | {cv_mean.mean():.2e} "
                f"| {train_time.mean():.1f} ± {train_time.std():.1f} "
                f"| {np.median(step_time):.0f} "
                f"| {tail_cos.mean():.3f} ± {tail_cos.std():.3f} "
                f"| {tail_ratio.mean():.2f} ± {tail_ratio.std():.2f} |"
            )
    table = "\n".join(lines)

    # Projector signature. For the *exact* projection onto a polyhedron the
    # Jacobian is an orthogonal projector, which forces cos == ratio. The code
    # runs a truncated Douglas-Rachford, so the identity only holds to the extent
    # that the iteration has converged: the residual below measures exactly that.
    extra = [
        "",
        "",
        "Projector signature - per-instance |cos - ratio| "
        "(zero iff the Jacobian is an orthogonal projector):",
        "",
        "| dataset | arm | median | p99 | max |",
        "|---|---|---|---|---|",
    ]
    for dataset in datasets:
        for arm in ("ift", "st"):
            if (dataset, arm) not in runs:
                continue
            residual = np.concatenate(
                [
                    np.abs(
                        np.asarray(r["geometry_cos"]) - np.asarray(r["geometry_ratio"])
                    ).ravel()
                    for r in runs[(dataset, arm)]
                ]
            )
            extra.append(
                f"| {DATASET_LABEL.get(dataset, dataset)} | {ARM_LABEL[arm]} "
                f"| {np.median(residual):.2e} | {np.percentile(residual, 99):.2e} "
                f"| {residual.max():.2e} |"
            )
    extra = "\n".join(extra) + "\n"
    out.write_text(table + extra)
    return table + extra


def main() -> None:
    """Parse the ablation results and write figures and summary."""
    parser = argparse.ArgumentParser(description="Parse the gradient ablation.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(pathlib.Path(__file__).parent / "results" / "grad_ablation"),
        help="Directory with the run results.",
    )
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out)
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    style()
    runs = load_runs(out_dir)
    if not runs:
        raise SystemExit(f"No results found in {out_dir}")
    datasets = [d for d in DATASET_LABEL if any(k[0] == d for k in runs)]

    figure_training(runs, datasets, figures / "training.png")
    figure_walltime(runs, datasets, figures / "walltime.png")
    figure_gradients(runs, datasets, figures / "gradients.png")
    figure_analysis(runs, datasets, figures / "analysis.png")
    figure_jacobian(runs, datasets, figures / "jacobian.png")
    for dataset in datasets:
        figure_per_layer(runs, dataset, figures / f"per_layer_{dataset}.png")

    print(summarise(runs, datasets, out_dir / "summary.md"))
    print(f"\nFigures written to {figures}")


if __name__ == "__main__":
    main()
