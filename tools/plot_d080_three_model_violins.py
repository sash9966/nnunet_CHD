#!/usr/bin/env python3
"""Plot complete, paired Dataset080 results for D071, D090 and D091.

Requires numpy, scipy and matplotlib. The input directory must contain
dice_D071.csv, dice_D090.csv and dice_D091.csv, with columns Patient_ID, WH,
LV-BP, RV-BP, LA, RA, Myo, Ao, PA and Macro7. These must be real results from
the matched native-input, five-fold evaluation, not substituted baselines.

Example:
    python tools/plot_d080_three_model_violins.py --analysis-dir /path/to/analysis

PNG, editable SVG and vector PDF are written beside the CSVs by default.
All patient scores, including zero, contribute to every plotted summary.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde


MODELS = ("D071", "D090", "D091")
METRICS = ("WH", "LV-BP", "RV-BP", "LA", "RA", "Myo", "Ao", "PA")
NUMERIC_COLUMNS = METRICS + ("Macro7",)
CASE_IDS = (
    "BAF004", "CHIPS001", "CHIPS002", "CHIPS005",
    "CHIPS006", "CHIPS007", "CHIPS010", "CHIPS016",
)
PANEL_TITLES = (
    "Whole-heart union", "LV blood pool", "RV blood pool", "Left atrium",
    "Right atrium", "Myocardium", "Aorta", "Pulmonary artery",
)
COLORS = {"D071": "#b4b0c9", "D090": "#66c2a5", "D091": "#fc8d62"}
EDGES = {"D071": "#756f91", "D090": "#3c8d76", "D091": "#bc6548"}
MODEL_LABELS = {
    "D071": "D071 | ImageCHD only",
    "D090": "D090 | first adaptation",
    "D091": "D091 | second adaptation",
}
INK = "#243842"
MUTED = "#61737d"
MEAN_COLOR = "#bd4145"
JITTER = np.array([-0.080, 0.024, 0.084, -0.030, 0.055, -0.057, 0.005, -0.004])

Table = Dict[str, Dict[str, float]]


def load_results(analysis_dir: Path) -> Tuple[Dict[str, Table], List[str]]:
    """Reject incomplete cohorts and invalid values before creating output files."""
    tables: Dict[str, Table] = {}
    required = {"Patient_ID", *NUMERIC_COLUMNS}
    for model in MODELS:
        path = analysis_dir / ("dice_" + model + ".csv")
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            if len(fields) != len(set(fields)) or set(fields) != required:
                raise ValueError(
                    f"{path}: expected exactly the columns "
                    + ", ".join(("Patient_ID",) + NUMERIC_COLUMNS)
                )
            table: Table = {}
            for line_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"{path}:{line_number}: malformed CSV row")
                case_id = row["Patient_ID"].strip()
                if case_id in table:
                    raise ValueError(f"{path}:{line_number}: duplicate case {case_id!r}")
                values = {}
                for metric in NUMERIC_COLUMNS:
                    try:
                        value = float(row[metric])
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"{path}:{line_number}: {metric} is not numeric"
                        ) from error
                    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                        raise ValueError(
                            f"{path}:{line_number}: {metric} must be finite and within [0, 1]"
                        )
                    values[metric] = value
                macro = sum(values[metric] for metric in METRICS[1:]) / 7.0
                if not math.isclose(values["Macro7"], macro, rel_tol=0.0, abs_tol=1e-6):
                    raise ValueError(
                        f"{path}:{line_number}: Macro7 is not the mean of the seven structures"
                    )
                table[case_id] = values
            if set(table) != set(CASE_IDS) or len(table) != 8:
                missing = sorted(set(CASE_IDS) - set(table))
                extra = sorted(set(table) - set(CASE_IDS))
                raise ValueError(
                    f"{path}: expected the eight Dataset080 cases; "
                    f"missing={missing}, unexpected={extra}, rows={len(table)}"
                )
            tables[model] = table
    return tables, list(CASE_IDS)


def draw_violin(ax, values: np.ndarray, position: float, model: str) -> None:
    """Scott-bandwidth KDE, truncated at observed extrema; no fabricated tails."""
    low, high = float(values.min()), float(values.max())
    if high - low > 1e-10:
        grid = np.linspace(low, high, 256)
        density = gaussian_kde(values, bw_method="scott")(grid)
        half_width = 0.29 * density / density.max()
        ax.fill_betweenx(
            grid, position - half_width, position + half_width,
            facecolor=COLORS[model], edgecolor=EDGES[model],
            alpha=0.78, linewidth=0.85, zorder=2,
        )
    else:
        # A constant sample has no estimable density. Draw the observed level.
        ax.plot(
            [position - 0.24, position + 0.24], [low, low],
            color=COLORS[model], linewidth=4, solid_capstyle="butt",
            zorder=2, clip_on=False,
        )

    ax.scatter(
        position + JITTER, values, s=11, color=INK, alpha=0.80,
        edgecolors="white", linewidths=0.32, zorder=4, clip_on=False,
    )
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    for quantile in (q1, q3):
        ax.plot(
            [position - 0.12, position + 0.12], [quantile, quantile],
            color="#202326", linewidth=0.75, zorder=5, clip_on=False,
        )
    ax.plot(
        [position - 0.16, position + 0.16], [median, median],
        color="#202326", linewidth=1.45, zorder=5, clip_on=False,
    )
    ax.plot(
        [position - 0.23, position + 0.23], [values.mean(), values.mean()],
        color=MEAN_COLOR, linewidth=1.15, linestyle=(0, (3, 1.5)),
        zorder=6, clip_on=False,
    )


def create_figure(
    tables: Dict[str, Table], case_ids: List[str], draft_label: Optional[str] = None
):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": MUTED,
        "axes.edgecolor": "#b7c5cc",
        "axes.linewidth": 0.65,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 5.55), sharey=True)
    fig.subplots_adjust(left=0.072, right=0.986, top=0.795, bottom=0.235,
                        hspace=0.51, wspace=0.28)
    title = draft_label or "Clinical CHD segmentation after iterative adaptation"
    fig.text(0.072, 0.964, title, fontsize=11.4, fontweight="bold", va="top",
             color=MEAN_COLOR if draft_label else INK)
    fig.text(
        0.072, 0.919,
        "Dataset080 | n = 8 patients | five-fold ensembles | native-input inference",
        fontsize=8.1, color=MUTED, va="top",
    )
    model_handles = [
        Patch(facecolor=COLORS[model], edgecolor=EDGES[model], label=MODEL_LABELS[model])
        for model in MODELS
    ]
    fig.legend(
        handles=model_handles, loc="upper left", bbox_to_anchor=(0.061, 0.884),
        ncol=3, frameon=False, fontsize=7.8, handlelength=1.3,
        columnspacing=1.6, handletextpad=0.6,
    )

    for panel, (ax, metric, title) in enumerate(zip(axes.flat, METRICS, PANEL_TITLES)):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e6ecef", linewidth=0.60)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for position, model in enumerate(MODELS):
            values = np.array([tables[model][case][metric] for case in case_ids])
            draw_violin(ax, values, float(position), model)
        ax.set_xlim(-0.53, 2.53)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
        ax.set_xticks(range(3))
        ax.set_xticklabels(MODELS, fontsize=7.2)
        ax.tick_params(axis="both", length=0, pad=4, labelsize=7.2)
        ax.set_title(title, fontsize=8.3, fontweight="bold", loc="left", pad=9)
        ax.text(-0.035, 1.105, chr(ord("a") + panel), transform=ax.transAxes,
                fontsize=8.5, fontweight="bold", va="bottom", ha="right")
        if panel % 4 == 0:
            ax.set_ylabel("Dice coefficient", labelpad=5)

    statistical_handles = [
        Line2D([], [], marker="o", linestyle="none", color=INK, markersize=3.2,
               label="Individual patient"),
        Line2D([], [], color="#202326", linewidth=1.45, label="Median"),
        Line2D([], [], color="#202326", linewidth=0.75, label="25th / 75th percentile"),
        Line2D([], [], color=MEAN_COLOR, linewidth=1.15, linestyle=(0, (3, 1.5)),
               label="Mean"),
    ]
    fig.legend(
        handles=statistical_handles, loc="upper left", bbox_to_anchor=(0.061, 0.161),
        ncol=4, frameon=False, fontsize=7.3, handlelength=1.5,
        columnspacing=1.5, handletextpad=0.55,
    )
    macro_values = [
        np.mean([tables[model][case]["Macro7"] for case in case_ids]) for model in MODELS
    ]
    fig.text(
        0.072, 0.088,
        "Mean seven-structure Dice: "
        + "   |   ".join(f"{model}  {value:.3f}" for model, value in zip(MODELS, macro_values)),
        fontsize=7.7, fontweight="bold", color=INK,
    )
    fig.text(
        0.072, 0.048,
        "All eight patient scores retained. Whole-heart Dice uses the union of the seven structures.",
        fontsize=7.1, color=MUTED,
    )
    return fig


def save_figure(fig, out_prefix: Path, draft_label: Optional[str] = None) -> List[Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (".png", ".svg", ".pdf"):
        path = Path(str(out_prefix) + suffix)
        kwargs = {"dpi": 400} if suffix == ".png" else {}
        if suffix == ".pdf":
            kwargs["metadata"] = {
                "Title": draft_label or "Dataset080: D071, D090 and D091 segmentation",
                "Subject": "Eight paired patients; complete Dice distributions; native five-fold evaluation",
            }
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument(
        "--out-prefix", type=Path,
        help="Output path without extension (default: ANALYSIS_DIR/figure2_d080_three_model_violins)",
    )
    parser.add_argument(
        "--draft-label", help="Prominent draft banner for synthetic layout checks; omit for actual results"
    )
    args = parser.parse_args()
    try:
        tables, case_ids = load_results(args.analysis_dir)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(2, f"ERROR: {error}\n")
    prefix = args.out_prefix or args.analysis_dir / "figure2_d080_three_model_violins"
    fig = create_figure(tables, case_ids, args.draft_label)
    for path in save_figure(fig, prefix, args.draft_label):
        print(path)


if __name__ == "__main__":
    main()
