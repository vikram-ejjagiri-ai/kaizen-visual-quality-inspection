"""Figures: one consistent style, colorblind-safe colors."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from src.config import FIGURES_DIR

GOOD = "#2a78d6"       # blue
DEFECT = "#eb6834"     # orange
OTHER = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.frameon": False, "font.size": 10, "lines.linewidth": 2,
        "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name)
    plt.close(fig)


def overlay(image, heatmap, vmin, vmax):
    """Blend a heatmap (jet colormap) over an RGB image."""
    norm = np.clip((heatmap - vmin) / (vmax - vmin + 1e-8), 0, 1)
    colored = plt.get_cmap("jet")(norm)[..., :3]
    return (0.55 * image / 255.0 + 0.45 * colored).clip(0, 1)


def plot_score_distribution(scores, labels, threshold, title, name):
    scores, labels = np.asarray(scores), np.asarray(labels)
    bins = np.linspace(scores.min(), scores.max(), 35)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(scores[labels == 0], bins=bins, color=GOOD, alpha=0.75, label="Good parts")
    ax.hist(scores[labels == 1], bins=bins, color=DEFECT, alpha=0.75, label="Defective parts")
    ax.axvline(threshold, color=INK, linestyle="--", linewidth=1.5)
    ax.text(threshold, ax.get_ylim()[1] * 0.95, "  decision threshold", color=INK2, va="top")
    ax.set(xlabel="Anomaly score (higher = more unusual)", ylabel="Number of images", title=title)
    ax.legend(loc="upper right")
    save(fig, name)


def plot_roc(labels, curves, name):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    colors = [DEFECT, MUTED, OTHER]
    for (label, scores, auc), color in zip(curves, colors):
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr, tpr, color=color, label=f"{label} (AUROC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=GRID, linestyle=":")
    ax.set(xlabel="Good parts wrongly rejected (false reject rate)",
           ylabel="Defects caught (detection rate)", title="Image-level ROC (full test set)")
    ax.legend(loc="lower right")
    save(fig, name)


def plot_cost_curve(thresholds, costs, chosen, name):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(thresholds, costs, color=GOOD)
    ax.axvline(chosen, color=DEFECT, linestyle="--", linewidth=1.5)
    ax.set(xlabel="Decision threshold on the anomaly score", ylabel="Business cost",
           title="Choosing the threshold on the calibration half")
    save(fig, name)


def plot_examples(rows, images, maps, masks, vmin, vmax, title, name):
    """Grid: one column per example; rows = image, ground truth, heatmap overlay."""
    n = len(rows)
    if n == 0:
        return
    fig, axes = plt.subplots(3, n, figsize=(2.6 * n, 8.2), squeeze=False)
    for j, (row, img, amap, mask) in enumerate(zip(rows, images, maps, masks)):
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"{row['defect_type']}\nscore {row['score']:.2f}", fontsize=9)
        axes[1, j].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[2, j].imshow(overlay(img, amap, vmin, vmax))
        for i in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].grid(False)
    axes[0, 0].set_ylabel("Image", fontsize=10)
    axes[1, 0].set_ylabel("True defect (mask)", fontsize=10)
    axes[2, 0].set_ylabel("AI heatmap", fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, name)


def plot_detection_by_type(table, name):
    data = table.sort_values("detection_rate")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(data) + 1.5))
    bars = ax.barh(data["defect_type"], data["detection_rate"] * 100, color=DEFECT, height=0.6)
    ax.bar_label(bars, labels=[f"{c}/{n}" for c, n in zip(data["caught"], data["count"])],
                 padding=4, fontsize=9, color=INK2)
    ax.set(xlim=(0, 110), xlabel="Defects caught [%] (evaluation half)",
           title="Detection rate per defect type")
    ax.grid(axis="y", visible=False)
    save(fig, name)
