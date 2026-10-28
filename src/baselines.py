"""Reference models on the exact same outer folds as the KNN.

Without something to compare against, a macro-F1 of ~0.73 is hard to interpret.
Three simple baselines:

- Stratified dummy - random guessing that respects class priors (floor).
- Nearest centroid - one prototype per syndrome. If KNN barely beats this,
  the local neighbourhood isn't adding much.
- Multinomial logistic regression - linear boundary in embedding space.
  If this matches or beats KNN, classes are roughly linearly separable.

Folds use the same StratifiedGroupKFold + random_state as knn_classification,
so numbers are comparable fold by fold.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

import metrics as m
from knn_classification import OUTER_FOLDS, _safe_group_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_baselines(random_state: int):
    """Return {name: pipeline}. Preprocessing is inside each pipeline so it
    only ever sees the training fold."""
    return {
        "Stratified dummy": Pipeline([
            ("clf", DummyClassifier(strategy="stratified", random_state=random_state)),
        ]),
        "Nearest centroid (L2)": Pipeline([
            ("preprocess", Normalizer(norm="l2")),
            ("clf", NearestCentroid()),
        ]),
        "Logistic regression": Pipeline([
            ("preprocess", StandardScaler()),
            # multinomial is default for multiclass in recent sklearn;
            # multi_class= arg was removed in 1.7+
            ("clf", LogisticRegression(max_iter=2000, random_state=random_state)),
        ]),
    }


def evaluate_baselines(X, y, groups, random_state=42):
    classes = np.array(sorted(np.unique(y)))
    splits = _safe_group_splits(X, y, groups, OUTER_FOLDS, random_state)
    rows = []

    for name, model in build_baselines(random_state).items():
        f1s, accs = [], []
        for train_idx, test_idx in splits:
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            y_test = y[test_idx]
            f1s.append(m.macro_f1(y_test, pred, list(classes)))
            accs.append(float(np.mean(pred == y_test)))
        rows.append({
            "model": name,
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s, ddof=1)),
            "top1_accuracy_mean": float(np.mean(accs)),
            "top1_accuracy_std": float(np.std(accs, ddof=1)),
        })
        logger.info("%-24s macro-F1 = %.4f ± %.4f", name, rows[-1]["macro_f1_mean"],
                    rows[-1]["macro_f1_std"])

    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Baseline models on the same outer folds.")
    p.add_argument("--input", type=str, default="outputs/embeddings.npz")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.input, allow_pickle=False)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(str)
    groups = data["subject_id"].astype(str)

    df = evaluate_baselines(X, y, groups, args.random_state)
    df.to_csv(output_dir / "baseline_comparison.csv", index=False)
    logger.info("Saved baseline comparison to %s", output_dir / "baseline_comparison.csv")


if __name__ == "__main__":
    main()
