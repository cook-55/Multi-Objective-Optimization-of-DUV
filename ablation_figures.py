"""Paper figures for the controlled comparisons in ablation.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


GROUPS = ("A", "B", "C", "D")
GROUP_NAMES = ("Full / DNN", "Full / random", "Binary / DNN", "Binary / random")
GROUP_COLORS = ("#0072B2", "#D55E00", "#0072B2", "#D55E00")
GROUP_POSITIONS = (0.0, 1.0, 2.35, 3.35)
INK = "#20252B"
PAIR_LINE = "#BEC5CC"
METRICS = (
    ("comprehensive_cost", "Composite score"),
    ("avg_T", "Mean transmittance (%)"),
    ("avg_R", "Mean reflectance (%)"),
    ("avg_stress", "Mean film stress (MPa)"),
)
OVERVIEW_METRICS = METRICS[:3] + (
    ("std_R", "Radial reflectance SD (pp)"),
    METRICS[3],
    ("tmm_evals", "Online exact TMM evaluations (×10³)"),
)
CONTRASTS = (
    ("A", "B", "Full: screening", "#0072B2"),
    ("C", "D", "Binary: screening", "#0072B2"),
    ("A", "C", "DNN: material library", "#00866A"),
    ("B", "D", "Random: material library", "#00866A"),
)
PAPER_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "text.color": INK,
    "text.usetex": False,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.7,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": None,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _new_figure(width, height):
    fig = Figure(figsize=(width, height), constrained_layout=True)
    FigureCanvasAgg(fig)
    return fig


def _style_axis(ax, ylabel, panel=None):
    ax.set_ylabel(ylabel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E0E5EA", linewidth=0.6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.margins(y=0.14)
    if panel is not None:
        ax.set_title(panel, loc="left", fontsize=10, fontweight="bold", pad=5)


def _save_pair(fig, output_dir, stem):
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    return [pdf, png]


def _metric_figure(runs, transmission_target):
    fig = _new_figure(7.25, 6.95)
    grid = fig.add_gridspec(4, 2, height_ratios=[0.20, 1, 1, 1])
    legend_ax = fig.add_subplot(grid[0, :])
    legend_ax.set_axis_off()
    legend_ax.legend(
        handles=[
            Line2D([], [], color=PAIR_LINE, linewidth=1.0, label="Paired seed"),
            Line2D([], [], color=INK, marker="o", markersize=3.8, linestyle="none",
                   alpha=0.55, label="Individual run"),
            Line2D([], [], color=INK, marker="D", markerfacecolor="white",
                   linewidth=1.4, markersize=4.1, label="Median and IQR"),
        ],
        loc="center", ncol=3, frameon=False,
    )
    labels = [group + "\n" + name.replace(" / ", "\n")
              for group, name in zip(GROUPS, GROUP_NAMES)]
    for i, (column, ylabel) in enumerate(OVERVIEW_METRICS):
        ax = fig.add_subplot(grid[1 + i // 2, i % 2])
        _style_axis(ax, ylabel, "abcdef"[i])
        paired = runs.pivot(index="seed", columns="group", values=column)
        values = paired.reindex(columns=GROUPS).sort_index().to_numpy(dtype=float)
        if column == "tmm_evals":
            values /= 1000.0
        offsets = np.linspace(-0.11, 0.11, len(values)) if len(values) > 1 else np.zeros(len(values))

        for row, offset in zip(values, offsets):
            for j in (0, 2):
                if np.isfinite(row[j:j + 2]).all():
                    ax.plot(np.asarray(GROUP_POSITIONS[j:j + 2]) + offset,
                            row[j:j + 2], color=PAIR_LINE, linewidth=0.65, zorder=2)

        for j, (position, color) in enumerate(zip(GROUP_POSITIONS, GROUP_COLORS)):
            finite = np.isfinite(values[:, j])
            column_values = values[finite, j]
            if column_values.size == 0:
                continue
            ax.scatter(position + offsets[finite], column_values, s=15,
                       color=color, alpha=0.62, edgecolors="none", zorder=3)
            q1, median, q3 = np.quantile(column_values, (0.25, 0.5, 0.75))
            ax.vlines(position, q1, q3, color=color, linewidth=1.8, zorder=4)
            ax.scatter(position, median, s=26, marker="D", facecolor="white",
                       edgecolor=color, linewidth=1.25, zorder=5)

        ax.set_xticks(GROUP_POSITIONS)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.45, 3.80)
        if column == "avg_T" and transmission_target is not None:
            ax.axhline(transmission_target, color=INK, linestyle=(0, (3, 2)),
                       linewidth=0.8, label=f"{transmission_target:g}% soft target")
            ax.legend(loc="lower left", frameon=False, fontsize=7)
    return fig


def _paired_figure(runs):
    fig = _new_figure(7.25, 5.25)
    grid = fig.add_gridspec(3, 2, height_ratios=[0.18, 1, 1])
    key = fig.add_subplot(grid[0, :])
    key.set_axis_off()
    key.legend(handles=[
        Line2D([], [], color=INK, marker="o", linestyle="none", markersize=4,
               alpha=0.6, label="Paired seed"),
        Line2D([], [], color=INK, marker="D", markerfacecolor="white",
               linewidth=1.5, markersize=4, label="Median and IQR"),
        Line2D([], [], color="#797F87", linestyle="--", linewidth=0.8,
               label="Zero difference"),
    ], loc="center", ncol=3, frameon=False)
    metrics = (
        ("initial_cost", "Initial composite score difference"),
        ("comprehensive_cost", "Final composite score difference"),
        ("avg_T", "Mean transmittance difference (pp)"),
        ("std_R", "Radial reflectance SD difference (pp)"),
    )
    positions = np.arange(len(CONTRASTS))[::-1]
    for i, (column, xlabel) in enumerate(metrics):
        ax = fig.add_subplot(grid[1 + i // 2, i % 2])
        _style_axis(ax, "", "abcd"[i])
        ax.grid(False, axis="y")
        ax.axvline(0, color="#797F87", linestyle="--", linewidth=0.8, zorder=1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.set_xlabel(xlabel + "\n(left group minus right group)")
        paired = runs.pivot(index="seed", columns="group", values=column).sort_index()
        for y, (left, right, _, color) in zip(positions, CONTRASTS):
            values = (paired[left] - paired[right]).dropna().to_numpy()
            if not values.size:
                continue
            offsets = np.linspace(-0.15, 0.15, values.size) if values.size > 1 else np.zeros(1)
            ax.scatter(values, y + offsets, s=15, color=color, alpha=0.65,
                       edgecolors="none", zorder=3)
            q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
            ax.hlines(y, q1, q3, color=color, linewidth=1.8, zorder=4)
            ax.scatter(median, y, s=25, marker="D", facecolor="white",
                       edgecolor=color, linewidth=1.2, zorder=5)
        ax.set_yticks(positions)
        ax.set_yticklabels([f"{left} − {right}\n{label}" for left, right, label, _ in CONTRASTS])
        ax.set_ylim(-0.45, positions[0] + 0.45)
        ax.margins(x=0.10)
    return fig


def _convergence_figure(runs, traces, top_k):
    if traces.empty:
        return None

    histories = {group: [] for group in GROUPS}
    caps = runs.set_index(["group", "seed"])["tmm_evals"]
    for key, trace in traces.groupby(["group", "seed"], sort=False):
        group, _ = key
        if group not in histories or key not in caps.index:
            continue
        evals = trace["tmm_evals"].to_numpy(dtype=float)
        costs = trace["best_cost"].to_numpy(dtype=float)
        cap = caps.loc[key]
        if not np.isfinite(cap) or cap < 1:
            continue
        valid = np.isfinite(evals) & np.isfinite(costs) & (evals >= 1) & (evals <= cap)
        evals, costs = evals[valid], costs[valid]
        if evals.size:
            order = np.argsort(evals, kind="stable")
            histories[group].append((evals[order], np.minimum.accumulate(costs[order])))

    series = [history for group_histories in histories.values() for history in group_histories]
    if not series:
        return None
    if len(series) != len(runs):
        raise ValueError("Each run needs a valid TMM trace for the common-budget figure.")

    # Compare the same runs at every point and stop at the earliest observed end.
    first = max(top_k, int(max(evals[0] for evals, _ in series)))
    last = int(min(evals[-1] for evals, _ in series))
    if first > last:
        return None
    events = [
        evals[(evals >= first) & (evals <= last)] for evals, _ in series
    ]
    budget_grid = np.unique(np.concatenate(([first, last], *events)))
    quantiles = {}
    for group, group_histories in histories.items():
        if not group_histories:
            continue
        trajectories = np.stack([
            costs[np.searchsorted(evals, budget_grid, side="right") - 1]
            for evals, costs in group_histories
        ])
        quantiles[group] = np.quantile(trajectories, (0.25, 0.5, 0.75), axis=0)

    fig = _new_figure(7.25, 3.8)
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    for i, ax in enumerate(axes):
        title = ("Full material library", "Binary material library")[i]
        _style_axis(ax, "Best observed composite score" if i == 0 else "",
                    f"{'ab'[i]}  {title}")
        ax.set_xlabel("Online exact TMM evaluations")
    for j, group in enumerate(GROUPS):
        if group not in quantiles:
            continue
        ax = axes[j // 2]
        q1, median, q3 = quantiles[group]
        color = GROUP_COLORS[j]
        ax.fill_between(budget_grid, q1, q3, step="post", color=color,
                        alpha=0.12, linewidth=0)
        ax.step(budget_grid, median, where="post", color=color, linewidth=1.55,
                linestyle="-" if j % 2 == 0 else "--",
                marker="o" if first == last else None,
                label=f"{group}  {GROUP_NAMES[j].split(' / ')[1]} (n={len(histories[group])})")

    if last >= 10 * first:
        axes[0].set_xscale("log")
    else:
        axes[0].xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    if first < last:
        axes[0].set_xlim(first, last)
    if all(np.min(values) >= 0 for values in quantiles.values()):
        axes[0].set_ylim(bottom=0)
    for ax in axes:
        ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"Median and IQR | Common observed range: {first:,}–{last:,} evaluations",
                 fontsize=8.5)
    return fig


def _material_stack_figure(stack_runs):
    fig = _new_figure(7.25, 5.25)
    grid = fig.add_gridspec(3, 2, height_ratios=[0.18, 1, 1])
    legend_ax = fig.add_subplot(grid[0, :])
    legend_ax.set_axis_off()
    variants = ("both", "yf3_only", "alf3_only", "neither")
    labels = ("Both", r"YF$_3$ only", r"AlF$_3$ only", "Neither")
    stages = ("fixed", "refined")
    colors = ("#5F6C78", "#0072B2")
    legend_ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color=colors[0],
                   markerfacecolor="white", label="Fixed thickness"),
            Line2D([], [], marker="s", linestyle="none", color=colors[1],
                   label="Refined thickness"),
        ],
        loc="center", ncol=2, frameon=False,
    )
    for i, (column, ylabel) in enumerate(METRICS):
        ax = fig.add_subplot(grid[1 + i // 2, i % 2])
        _style_axis(ax, ylabel, "abcd"[i])
        values = stack_runs.pivot(index="variant", columns="stage", values=column)
        values = values.reindex(index=variants, columns=stages).to_numpy(dtype=float)
        for j, row in enumerate(values):
            x = np.array([j - 0.11, j + 0.11])
            if np.isfinite(row).all():
                ax.plot(x, row, color=PAIR_LINE, linewidth=1.0, zorder=2)
            if np.isfinite(row[0]):
                ax.scatter(x[0], row[0], s=31, facecolor="white", edgecolor=colors[0], zorder=3)
            if np.isfinite(row[1]):
                ax.scatter(x[1], row[1], s=29, color=colors[1], marker="s", zorder=3)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.45, 3.45)
    return fig


def save_ablation_figures(
    runs: pd.DataFrame,
    traces: pd.DataFrame,
    output_dir: Path,
    stack_runs: pd.DataFrame | None = None,
    *,
    settings: dict | None = None,
) -> list[Path]:
    """Save vector PDF and 600 dpi PNG figures from completed experiment tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = settings or {}
    paths = []
    with mpl.rc_context(PAPER_STYLE):
        if not runs.empty:
            main = _metric_figure(runs, settings.get("transmission_target_percent"))
            paths.extend(_save_pair(main, output_dir, "ablation_metrics"))
            paired = _paired_figure(runs)
            paths.extend(_save_pair(paired, output_dir, "ablation_paired_effects"))
            convergence = _convergence_figure(runs, traces, settings.get("top_k", 1))
            if convergence is not None:
                paths.extend(_save_pair(convergence, output_dir, "ablation_convergence"))
        if stack_runs is not None and not stack_runs.empty:
            material_stack = _material_stack_figure(stack_runs)
            paths.extend(_save_pair(material_stack, output_dir, "ablation_material_stack"))
    return paths


def main():
    parser = argparse.ArgumentParser(description="Redraw ablation figures from completed CSVs.")
    parser.add_argument("result_dir", type=Path, help="Directory containing ablation_runs.csv.")
    parser.add_argument("--output-dir", type=Path, help="Default: RESULT_DIR/publication_figures.")
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    runs = pd.read_csv(result_dir / "ablation_runs.csv")
    traces = pd.read_csv(result_dir / "ablation_trace.csv")
    settings_path = result_dir / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.is_file() else {}
    stack_path = result_dir / "material_stack_comparison.csv"
    stack = pd.read_csv(stack_path) if stack_path.is_file() else None
    output_dir = args.output_dir or result_dir / "publication_figures"
    for path in save_ablation_figures(runs, traces, output_dir, stack, settings=settings):
        print(path)


if __name__ == "__main__":
    main()
