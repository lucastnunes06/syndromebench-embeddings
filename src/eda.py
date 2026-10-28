"""Counts, imbalance ratio and the images-per-syndrome bar chart."""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_embeddings(npz_path: Path):
    data = np.load(npz_path, allow_pickle=False)
    return data["X"], data["y"], data["subject_id"], data["image_id"]


def summarize(y: np.ndarray, subject_id: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"syndrome_id": y, "subject_id": subject_id})
    counts = df.groupby("syndrome_id").agg(
        n_images=("syndrome_id", "size"),
        n_subjects=("subject_id", "nunique"),
    ).sort_values("n_images", ascending=False)
    counts["images_per_subject"] = counts["n_images"] / counts["n_subjects"]
    return counts


def print_report(counts: pd.DataFrame, X: np.ndarray, subject_id: np.ndarray = None):
    n_syndromes = len(counts)
    n_images = counts["n_images"].sum()
    if subject_id is not None:
        n_subjects = int(pd.Series(subject_id).nunique())
    else:
        n_subjects = int(counts["n_subjects"].sum())

    logger.info("=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info("Total images:              %d", n_images)
    logger.info("Total distinct syndromes:  %d", n_syndromes)
    logger.info("Total distinct subjects:   %d", n_subjects)
    logger.info("Embedding dimensionality:  %d", X.shape[1])
    logger.info("-" * 60)
    logger.info("Images per syndrome: min=%d  max=%d  mean=%.2f  median=%.1f  std=%.2f",
                counts["n_images"].min(), counts["n_images"].max(),
                counts["n_images"].mean(), counts["n_images"].median(),
                counts["n_images"].std())
    imbalance_ratio = counts["n_images"].max() / counts["n_images"].min()
    logger.info("Imbalance ratio (max/min syndrome count): %.2f", imbalance_ratio)
    logger.info("-" * 60)
    logger.info("Top 5 most represented syndromes:\n%s", counts.head(5))
    logger.info("Bottom 5 least represented syndromes:\n%s", counts.tail(5))


def plot_distribution(counts: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(counts)), counts["n_images"].values, color="#4C72B0")
    ax.set_xlabel("syndrome_id (sorted by frequency)")
    ax.set_ylabel("Number of images")
    ax.set_title("Images per Syndrome ID")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([str(i) for i in counts.index], rotation=45, ha="right", fontsize=8)
    for i, v in enumerate(counts["n_images"].values):
        ax.text(i, v + 2, str(int(v)), ha="center", fontsize=8)
    fig.tight_layout()
    out_path = output_dir / "eda_images_per_syndrome.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved distribution plot to %s", out_path)


def parse_args():
    p = argparse.ArgumentParser(description="EDA on embeddings.")
    p.add_argument("--input", type=str, default="outputs/embeddings.npz")
    p.add_argument("--output_dir", type=str, default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, subject_id, _ = load_embeddings(Path(args.input))
    counts = summarize(y, subject_id)
    print_report(counts, X, subject_id)

    counts.to_csv(output_dir / "eda_syndrome_counts.csv")
    plot_distribution(counts, output_dir)


if __name__ == "__main__":
    main()
