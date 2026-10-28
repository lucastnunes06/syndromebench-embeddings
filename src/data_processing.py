"""Loads the hierarchical pickle (syndrome -> subject -> image -> 320-d
embedding) and flattens it into embeddings.npz + flattened_data.pkl.
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

EXPECTED_DIM = 320


def load_pickle(path: Path) -> dict:
    """Load the raw hierarchical pickle.

    If numpy versions don't match we try to give a clearer error than the
    default ModuleNotFoundError.
    """
    # Only accept a pickle from a source you trust: pickle.load may execute code.
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        logger.error("Input file not found: %s", path)
        sys.exit(1)
    except (ModuleNotFoundError, ImportError) as e:
        # message varies depending on which numpy wrote/read the file
        # (numpy.core._multiarray_umath, numpy._core, etc.)
        if "numpy" in str(e).lower():
            raise RuntimeError(
                f"Could not unpickle {path}: the file was written by a different "
                f"numpy version (underlying error: {e}). Try "
                "'pip install -U numpy' and retry. If the pickle is older, "
                "install the version that wrote it instead."
            ) from e
        raise
    if not isinstance(data, dict):
        raise TypeError(f"Expected top-level dict, got {type(data)}")
    return data


def flatten_data(raw: dict) -> pd.DataFrame:
    """Flatten syndrome/subject/image hierarchy into a long DataFrame.

    Columns: syndrome_id, subject_id, image_id, embedding (list of 320 floats).
    Drops rows with bad embeddings (wrong dim, NaNs, non-numeric) and logs them.
    """
    records = []
    n_bad = 0
    n_total = 0

    for syndrome_id, subjects in raw.items():
        if not isinstance(subjects, dict):
            logger.warning("Skipping syndrome %s: unexpected structure", syndrome_id)
            continue
        for subject_id, images in subjects.items():
            if not isinstance(images, dict):
                logger.warning(
                    "Skipping subject %s under syndrome %s: unexpected structure",
                    subject_id, syndrome_id,
                )
                continue
            for image_id, embedding in images.items():
                n_total += 1
                try:
                    vec = np.asarray(embedding, dtype=np.float64)
                except (TypeError, ValueError):
                    n_bad += 1
                    continue

                if vec.shape != (EXPECTED_DIM,):
                    n_bad += 1
                    logger.debug(
                        "Bad embedding shape %s for image %s (expected (%d,))",
                        vec.shape, image_id, EXPECTED_DIM,
                    )
                    continue

                if not np.isfinite(vec).all():
                    n_bad += 1
                    logger.debug("Non-finite values in embedding for image %s", image_id)
                    continue

                records.append(
                    {
                        "syndrome_id": syndrome_id,
                        "subject_id": subject_id,
                        "image_id": image_id,
                        "embedding": vec,
                    }
                )

    logger.info("Parsed %d total records; dropped %d malformed (%.2f%%)",
                n_total, n_bad, 100 * n_bad / max(n_total, 1))

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No valid records found after flattening/cleaning.")

    # exact duplicate (subject_id, image_id) pairs
    dup_mask = df.duplicated(subset=["subject_id", "image_id"], keep="first")
    if dup_mask.any():
        logger.warning("Dropping %d duplicate (subject_id, image_id) rows", dup_mask.sum())
        df = df[~dup_mask].reset_index(drop=True)

    return df


def save_flattened(df: pd.DataFrame, output_dir: Path) -> None:
    """Write pickle (quick reload) and compressed npz (embeddings + labels)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_pickle(output_dir / "flattened_data.pkl")

    X = np.stack(df["embedding"].values)
    # Force fixed-width Unicode arrays, not pandas object arrays: safe np.load
    # can then use allow_pickle=False for the processed NPZ.
    y = np.asarray([str(value) for value in df["syndrome_id"]], dtype=str)
    subjects = np.asarray([str(value) for value in df["subject_id"]], dtype=str)
    image_ids = np.asarray([str(value) for value in df["image_id"]], dtype=str)

    np.savez_compressed(
        output_dir / "embeddings.npz",
        X=X, y=y, subject_id=subjects, image_id=image_ids,
    )
    logger.info("Saved flattened_data.pkl and embeddings.npz to %s", output_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Load and flatten syndrome embedding data.")
    p.add_argument("--input", type=str, default="mini_gm_public_v0.1.p",
                    help="Path to the raw pickle file.")
    p.add_argument("--output_dir", type=str, default="outputs",
                    help="Where to write the processed files.")
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    logger.info("Loading raw pickle from %s", input_path)
    raw = load_pickle(input_path)

    logger.info("Flattening hierarchical structure")
    df = flatten_data(raw)

    logger.info(
        "Flattened dataset: %d images, %d syndromes, %d subjects",
        len(df), df["syndrome_id"].nunique(), df["subject_id"].nunique(),
    )

    save_flattened(df, output_dir)


if __name__ == "__main__":
    main()
