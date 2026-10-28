"""Runs the full pipeline end-to-end:
  1. Data processing
  2. EDA
  3. t-SNE
  4. Nested subject-grouped KNN (cosine + euclidean)
  5. Baselines on the same folds
  6. OOF confusion analysis

Everything lands in --output_dir. Step 4 also writes run_manifest.json so the
report generator can stay consistent with the numbers.

Usage:
    python main.py --input ../mini_gm_public_v0.1.p --output_dir ../outputs
"""

import argparse
import logging
from pathlib import Path

import numpy as np

import baselines
import confusion_analysis
import data_processing
import eda
import knn_classification
import visualize_tsne

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

N_STEPS = 6


def parse_args():
    p = argparse.ArgumentParser(description="Full syndrome classification pipeline.")
    p.add_argument("--input", type=str, default="mini_gm_public_v0.1.p")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--k_max", type=int, default=knn_classification.DEFAULT_K_MAX,
                   help="Upper bound for the k grid (warns if ceiling is hit).")
    p.add_argument("--scaling", choices=["auto", "standard", "l2", "none"], default="auto")
    p.add_argument("--no_tune_weights", action="store_true")
    p.add_argument("--tie_handling",
                   choices=["fractional", "optimistic", "pessimistic"],
                   default="fractional")
    p.add_argument("--skip_processing", action="store_true",
                   help="Reuse existing embeddings.npz.")
    p.add_argument("--skip_baselines", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "embeddings.npz"

    if not args.skip_processing or not embeddings_path.exists():
        logger.info("STEP 1/%d: Data processing", N_STEPS)
        raw = data_processing.load_pickle(Path(args.input))
        df = data_processing.flatten_data(raw)
        data_processing.save_flattened(df, output_dir)
    else:
        logger.info("STEP 1/%d: Skipped (using existing %s)", N_STEPS, embeddings_path)

    logger.info("STEP 2/%d: Exploratory data analysis", N_STEPS)
    X, y, subject_id, _ = eda.load_embeddings(embeddings_path)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(str)
    subject_id = np.asarray(subject_id).astype(str)

    counts = eda.summarize(y, subject_id)
    eda.print_report(counts, X, subject_id)
    counts.to_csv(output_dir / "eda_syndrome_counts.csv")
    eda.plot_distribution(counts, output_dir)

    logger.info("STEP 3/%d: t-SNE visualisation", N_STEPS)
    coords = visualize_tsne.run_tsne(X, perplexity=args.perplexity,
                                     random_state=args.random_state)
    np.save(output_dir / "tsne_coords.npy", coords)
    visualize_tsne.plot_embedding(coords, y, output_dir / "tsne_by_syndrome.png")

    logger.info("STEP 4/%d: Leakage-safe K-NN classification and evaluation", N_STEPS)
    knn_classification.check_subject_syndrome_consistency(y, subject_id)
    common = dict(random_state=args.random_state, k_max=args.k_max,
                  scaling=args.scaling, tune_weights=not args.no_tune_weights,
                  tie_handling=args.tie_handling)
    res_cos = knn_classification.evaluate_metric(
        X, y, subject_id, "cosine", output_dir, **common)
    res_euc = knn_classification.evaluate_metric(
        X, y, subject_id, "euclidean", output_dir, **common)

    knn_classification.plot_roc_comparison(res_cos, res_euc, output_dir)
    summary = knn_classification.build_summary_table(res_cos, res_euc)
    summary.to_csv(output_dir / "performance_summary.csv", index=False)
    knn_classification.write_run_manifest(output_dir, [res_cos, res_euc], vars(args))

    if args.skip_baselines:
        logger.info("STEP 5/%d: Skipped (--skip_baselines)", N_STEPS)
    else:
        logger.info("STEP 5/%d: Baseline models on the same outer folds", N_STEPS)
        baseline_df = baselines.evaluate_baselines(X, y, subject_id, args.random_state)
        baseline_df.to_csv(output_dir / "baseline_comparison.csv", index=False)

    logger.info("STEP 6/%d: Out-of-fold confusion analysis", N_STEPS)
    for metric in knn_classification.DISTANCE_METRICS:
        confusion_analysis.run(output_dir, metric)

    logger.info("Pipeline complete. All artefacts saved to %s", output_dir.resolve())
    logger.info("\n%s", summary.to_string(index=False))
    logger.info("Next: python generate_report.py --output_dir %s", output_dir)


if __name__ == "__main__":
    main()
