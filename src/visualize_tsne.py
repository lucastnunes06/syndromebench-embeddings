"""2D t-SNE of the 320-d embeddings, coloured by syndrome_id."""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_tsne(X: np.ndarray, perplexity: float = 30.0, random_state: int = 42) -> np.ndarray:
    """Standardise then project with t-SNE.

    Perplexity gets capped if n_samples is small so sklearn doesn't complain.
    """
    X_scaled = StandardScaler().fit_transform(X)
    safe_perplexity = min(perplexity, max(5.0, (len(X) - 1) / 3))
    logger.info("Running t-SNE (n=%d, perplexity=%.1f)", len(X), safe_perplexity)
    tsne = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    return tsne.fit_transform(X_scaled)


def plot_embedding(coords: np.ndarray, labels: np.ndarray, output_path: Path):
    unique_labels = sorted(set(labels))
    n_labels = len(unique_labels)
    cmap = plt.get_cmap("tab20" if n_labels <= 20 else "nipy_spectral")

    fig, ax = plt.subplots(figsize=(11, 9))
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=18, alpha=0.75,
            color=cmap(i / max(n_labels - 1, 1)),
            label=str(lab),
        )
    ax.set_title("t-SNE Projection of Syndrome Embeddings (coloured by syndrome_id)")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")

    if n_labels <= 30:
        ax.legend(
            title="syndrome_id", bbox_to_anchor=(1.02, 1), loc="upper left",
            fontsize=7, ncol=1 if n_labels <= 20 else 2,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved t-SNE plot to %s", output_path)


def parse_args():
    p = argparse.ArgumentParser(description="t-SNE visualisation of embeddings.")
    p.add_argument("--input", type=str, default="outputs/embeddings.npz")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--perplexity", type=float, default=30.0)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.input, allow_pickle=False)
    X, y = data["X"], data["y"]

    coords = run_tsne(X, perplexity=args.perplexity)
    np.save(output_dir / "tsne_coords.npy", coords)

    plot_embedding(coords, y, output_dir / "tsne_by_syndrome.png")


if __name__ == "__main__":
    main()
