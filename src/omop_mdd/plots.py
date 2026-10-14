"""Static figures for the synthetic demonstration."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

# Categorical slots 1-2 of the reference palette; text stays in neutral ink.
MODEL_COLORS = {"logistic_regression": "#2a78d6", "lightgbm": "#eb6834"}
MODEL_LABELS = {"logistic_regression": "Logistic regression", "lightgbm": "LightGBM"}
INK = "#1f1f1e"
MUTED = "#6b6a63"
GRID = "#e4e3dc"
SYNTHETIC_NOTE = "Synthetic data — illustrates the pipeline, not a study result"


def _style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def roc_figure(y_test: pd.Series, probs: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=150)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(
            fpr, tpr, color=MODEL_COLORS[name], linewidth=2, label=MODEL_LABELS[name]
        )
    ax.set_xlabel("False positive rate", color=INK)
    ax.set_ylabel("True positive rate", color=INK)
    ax.set_title("Held-out ROC", color=INK, loc="left", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    _style(ax)
    ax.legend(frameon=False, loc="lower right", labelcolor=INK)
    fig.text(0.01, 0.005, SYNTHETIC_NOTE, color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def confusion_figure(metrics: dict[str, dict], path: Path) -> None:
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(4.2 * len(metrics), 3.8), dpi=150
    )
    axes = np.atleast_1d(axes)
    for ax, (name, m) in zip(axes, metrics.items()):
        cm = m["confusion_matrix"]
        grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(grid, cmap="Blues", vmin=0, vmax=max(grid.max(), 1))
        for (i, j), value in np.ndenumerate(grid):
            dark = value > grid.max() * 0.6
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=12,
                color="white" if dark else INK,
            )
        ax.set_xticks([0, 1], ["Comparison", "Case"])
        ax.set_yticks([0, 1], ["Comparison", "Case"])
        ax.set_xlabel("Predicted", color=INK)
        ax.set_ylabel("Actual", color=INK)
        ax.set_title(
            f"{MODEL_LABELS[name]} (threshold {m['threshold']:.2f})",
            color=INK,
            loc="left",
            fontsize=10,
        )
        ax.tick_params(colors=MUTED, labelcolor=INK)
    fig.text(0.01, 0.005, SYNTHETIC_NOTE, color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def permutation_figure(importances: dict[str, pd.DataFrame], path: Path) -> None:
    """Permutation importance (drop in test AUROC) for both models, one panel each."""
    fig, axes = plt.subplots(
        1, len(importances), figsize=(5.0 * len(importances), 6.0), dpi=150, sharex=True
    )
    axes = np.atleast_1d(axes)
    for ax, (name, imp) in zip(axes, importances.items()):
        imp = imp.sort_values("permutation_auroc_drop_mean")
        ax.barh(
            imp["feature"],
            imp["permutation_auroc_drop_mean"],
            xerr=imp["permutation_auroc_drop_sd"],
            color=MODEL_COLORS[name],
            height=0.6,
            error_kw={"ecolor": MUTED, "elinewidth": 1},
        )
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set_title(MODEL_LABELS[name], color=INK, loc="left", fontsize=10)
        ax.set_xlabel("Mean drop in test AUROC when permuted", color=INK)
        _style(ax)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Permutation importance (held-out set)", color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, SYNTHETIC_NOTE, color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def coefficient_figure(imp: pd.DataFrame, path: Path) -> None:
    """Standardized logistic-regression coefficients (log-odds per 1 SD)."""
    imp = imp.sort_values("coefficient_std")
    fig, ax = plt.subplots(figsize=(5.4, 6.0), dpi=150)
    ax.barh(
        imp["feature"],
        imp["coefficient_std"],
        color=MODEL_COLORS["logistic_regression"],
    )
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("Coefficient (log-odds per 1 SD)", color=INK)
    ax.set_title("Logistic regression coefficients", color=INK, loc="left", fontsize=10)
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.text(0.01, 0.005, SYNTHETIC_NOTE, color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)
