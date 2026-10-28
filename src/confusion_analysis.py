"""OOF confusion analysis - which syndromes get confused with which.

Macro-F1 alone just says the model is wrong ~25% of the time. The matrix
shows where, and you can cross-check against the t-SNE clusters.

Reads oof_predictions_{metric}.csv from knn_classification. Each row is an
image scored by a model that never saw that subject's images, so it's a real
held-out matrix.
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import metrics as m

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_oof(output_dir: Path, metric: str) -> pd.DataFrame:
    path = output_dir / f"oof_predictions_{metric}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run knn_classification.py (or main.py) first."
        )
    return pd.read_csv(path, dtype={"syndrome_id": str, "predicted_syndrome": str})


def normalise_rows(cm: np.ndarray) -> np.ndarray:
    """Row-normalise (per-class recall).

    Raw counts are dominated by the biggest class under ~3.3x imbalance.
    """
    totals = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.divide(cm, totals, out=np.zeros_like(cm, dtype=float), where=totals > 0)
    return out


def plot_confusion(cm_norm: np.ndarray, labels, metric: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted syndrome_id")
    ax.set_ylabel("True syndrome_id")
    ax.set_title(f"Out-of-fold confusion matrix, row-normalised - {metric} distance")

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = cm_norm[i, j]
            if value >= 0.005:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if value > 0.55 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction of true class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix plot to %s", out_path)


def confused_pairs(cm: np.ndarray, cm_norm: np.ndarray, labels) -> pd.DataFrame:
    """Rank off-diagonal cells, most-confused first."""
    rows = []
    for i, true_lab in enumerate(labels):
        for j, pred_lab in enumerate(labels):
            if i == j:
                continue
            if cm[i, j] == 0:
                continue
            rows.append({
                "true_syndrome": true_lab,
                "predicted_syndrome": pred_lab,
                "n_errors": int(cm[i, j]),
                "fraction_of_true_class": float(cm_norm[i, j]),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("fraction_of_true_class", ascending=False).reset_index(drop=True)


def per_class_report(cm: np.ndarray, labels, y_true, y_pred) -> pd.DataFrame:
    stats = m.precision_recall_f1_per_class(y_true, y_pred, labels)
    return pd.DataFrame([
        {
            "syndrome_id": lab,
            "support": stats[lab]["support"],
            "precision": stats[lab]["precision"],
            "recall": stats[lab]["recall"],
            "f1": stats[lab]["f1"],
        }
        for lab in labels
    ]).sort_values("f1", ascending=False).reset_index(drop=True)


def run(output_dir: Path, metric: str):
    df = load_oof(output_dir, metric)
    y_true = df["syndrome_id"].to_numpy()
    y_pred = df["predicted_syndrome"].to_numpy()
    labels = sorted(set(y_true) | set(y_pred))

    cm = m.confusion_matrix(y_true, y_pred, labels)
    cm_norm = normalise_rows(cm)

    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        output_dir / f"confusion_matrix_{metric}.csv")
    plot_confusion(cm_norm, labels, metric, output_dir / f"confusion_matrix_{metric}.png")

    pairs = confused_pairs(cm, cm_norm, labels)
    pairs.to_csv(output_dir / f"confused_pairs_{metric}.csv", index=False)

    per_class = per_class_report(cm, labels, y_true, y_pred)
    per_class.to_csv(output_dir / f"per_class_metrics_{metric}.csv", index=False)

    if not pairs.empty:
        logger.info("Top confused pairs (%s):\n%s", metric,
                    pairs.head(5).to_string(index=False))
    logger.info("Weakest classes by F1 (%s):\n%s", metric,
                per_class.tail(3).to_string(index=False))
    return cm, pairs, per_class


def parse_args():
    p = argparse.ArgumentParser(description="Out-of-fold confusion analysis.")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--metric", type=str, default="cosine",
                   choices=["cosine", "euclidean"])
    return p.parse_args()


def main():
    args = parse_args()
    run(Path(args.output_dir), args.metric)


if __name__ == "__main__":
    main()
