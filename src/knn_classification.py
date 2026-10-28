"""Subject-grouped nested KNN for syndrome_id from 320-d embeddings.

Main points:
- Same subject always stays in the same fold.
- Preprocessing fitted only on the training fold.
- k and weights picked by inner 9-fold CV, scores from outer 10-fold.
- Metrics come from metrics.py (hand-rolled, tested against sklearn).

Scaling (default --scaling auto):
  cosine    -> L2 row norm
  euclidean -> StandardScaler

Reason for the split: cosine cares about angle. StandardScaler moves the
origin, so the angle after it isn't the angle of the original vectors. L2
keeps direction and just kills magnitude. --scaling standard forces
StandardScaler on both if you want the old behaviour (comparison then
mixes metric + preprocessing).

Note: on already L2-normalised vectors, euclidean and cosine pick the same
neighbours (monotone relationship).
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

import metrics as m

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_K_MAX = 15
OUTER_FOLDS = 10
INNER_FOLDS = 9
DISTANCE_METRICS = ["cosine", "euclidean"]
WEIGHT_OPTIONS = ["uniform", "distance"]
TOP_K_VALUES = (1, 3, 5)


# Integrity checks

def check_subject_syndrome_consistency(y: np.ndarray, groups: np.ndarray) -> None:
    """Each subject should map to exactly one syndrome.

    If not, the hierarchy is broken and the group CV guarantee is meaningless.
    Fail loud rather than train on garbage.
    """
    counts = pd.DataFrame({"subject": groups, "syndrome": y}) \
        .groupby("subject")["syndrome"].nunique()
    inconsistent = counts[counts > 1]
    if not inconsistent.empty:
        raise ValueError(
            f"Found {len(inconsistent)} subjects associated with multiple syndromes. "
            "Please inspect the dataset before classification."
        )


# Model construction

def resolve_scaling(metric: str, scaling: str) -> str:
    """Map the --scaling policy to a concrete preprocessing step name."""
    if scaling == "auto":
        return "l2" if metric == "cosine" else "standard"
    return scaling


def make_model(k: int, metric: str, weights: str, scaling: str) -> Pipeline:
    """Create a leakage-safe preprocessing + K-NN pipeline."""
    mode = resolve_scaling(metric, scaling)
    if mode == "standard":
        pre = StandardScaler()
    elif mode == "l2":
        pre = Normalizer(norm="l2")
    elif mode == "none":
        pre = "passthrough"
    else:
        raise ValueError(f"Unknown scaling mode: {mode}")

    return Pipeline([
        ("preprocess", pre),
        ("knn", KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)),
    ])


def _safe_group_splits(X, y, groups, n_splits, random_state):
    """Return group-aware stratified folds, with a clear error if impossible."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=random_state)
    try:
        splits = list(splitter.split(X, y, groups))
    except ValueError as exc:
        raise ValueError(
            f"Could not create {n_splits}-fold StratifiedGroupKFold. "
            "The dataset may not contain enough distinct subjects per class."
        ) from exc
    if len(splits) != n_splits:
        raise ValueError(f"Expected {n_splits} folds, got {len(splits)}.")
    return splits


# Inner-CV hyper-parameter selection

def select_hyperparams_inner_cv(X, y, groups, metric, train_indices,
                                random_state, k_max, scaling, tune_weights):
    """Pick (k, weights) using only the outer-training subjects.

    Preprocessing is refit inside every inner fold.

    Tie-break: higher mean inner-CV macro-F1, then smaller k, then
    uniform before distance. mergesort needed because pandas default
    quicksort isn't stable.
    """
    X_outer, y_outer, g_outer = X[train_indices], y[train_indices], groups[train_indices]
    inner_splits = _safe_group_splits(X_outer, y_outer, g_outer, INNER_FOLDS, random_state)

    weight_grid = WEIGHT_OPTIONS if tune_weights else ["uniform"]
    labels = np.array(sorted(np.unique(y_outer)))

    rows = []
    for weights in weight_grid:
        for k in range(1, k_max + 1):
            fold_f1, fold_acc = [], []
            for inner_train, inner_val in inner_splits:
                model = make_model(k, metric, weights, scaling)
                model.fit(X_outer[inner_train], y_outer[inner_train])
                y_pred = model.predict(X_outer[inner_val])
                y_val = y_outer[inner_val]
                # Score only against classes actually present in this
                # validation fold; otherwise absent classes contribute a
                # spurious F1 of 0 and bias macro-F1 downward.
                present = [lab for lab in labels if np.any(y_val == lab)]
                fold_f1.append(m.macro_f1(y_val, y_pred, present))
                fold_acc.append(float(np.mean(y_pred == y_val)))

            rows.append({
                "k": k,
                "weights": weights,
                "mean_macro_f1": float(np.mean(fold_f1)),
                "std_macro_f1": float(np.std(fold_f1, ddof=1)),
                "mean_accuracy": float(np.mean(fold_acc)),
                "std_accuracy": float(np.std(fold_acc, ddof=1)),
            })

    result = pd.DataFrame(rows)
    ranked = result.sort_values(
        ["mean_macro_f1", "k", "weights"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    best = ranked.iloc[0]
    return int(best["k"]), str(best["weights"]), result


# Outer-CV evaluation

def _fold_probability_matrix(model, X_test, global_classes):
    """Map model.predict_proba columns onto one global class ordering."""
    local_proba = model.predict_proba(X_test)
    full = np.zeros((len(X_test), len(global_classes)), dtype=float)
    class_to_idx = {c: i for i, c in enumerate(global_classes)}
    for j, cls in enumerate(model.classes_):
        full[:, class_to_idx[cls]] = local_proba[:, j]
    return full


def evaluate_metric(X, y, groups, metric, output_dir, random_state=42,
                    k_max=DEFAULT_K_MAX, scaling="auto", tune_weights=True,
                    tie_handling="fractional"):
    """Nested, subject-grouped 10-fold evaluation for one distance metric."""
    output_dir = Path(output_dir)
    scaling_mode = resolve_scaling(metric, scaling)
    logger.info("=" * 74)
    logger.info("Distance metric: %s | preprocessing: %s", metric, scaling_mode)
    logger.info("Outer CV: %d folds | Inner CV: %d folds | k=1..%d | weights=%s",
                OUTER_FOLDS, INNER_FOLDS, k_max,
                "{uniform,distance}" if tune_weights else "uniform")
    logger.info("=" * 74)

    classes = np.array(sorted(np.unique(y)))
    outer_splits = _safe_group_splits(X, y, groups, OUTER_FOLDS, random_state)

    oof_proba = np.zeros((len(y), len(classes)), dtype=float)
    oof_pred = np.empty(len(y), dtype=object)
    oof_fold = np.full(len(y), -1, dtype=int)
    selected_ks, selected_weights = [], []
    fold_rows, fold_rocs, inner_tables = [], [], []

    for fold_id, (train_idx, test_idx) in enumerate(outer_splits):
        overlap = set(groups[train_idx]).intersection(set(groups[test_idx]))
        if overlap:
            raise RuntimeError(
                f"Subject leakage detected in outer fold {fold_id}: "
                f"{len(overlap)} overlapping groups."
            )

        best_k, best_weights, k_table = select_hyperparams_inner_cv(
            X, y, groups, metric, train_idx,
            random_state=random_state + fold_id + 1,
            k_max=k_max, scaling=scaling, tune_weights=tune_weights,
        )
        selected_ks.append(best_k)
        selected_weights.append(best_weights)
        k_table = k_table.copy()
        k_table["outer_fold"] = fold_id
        k_table["metric"] = metric
        inner_tables.append(k_table)

        model = make_model(best_k, metric, best_weights, scaling)
        model.fit(X[train_idx], y[train_idx])
        proba = _fold_probability_matrix(model, X[test_idx], classes)
        pred = classes[np.argmax(proba, axis=1)]

        oof_proba[test_idx] = proba
        oof_pred[test_idx] = pred
        oof_fold[test_idx] = fold_id

        y_test = y[test_idx]
        labels = list(classes)
        roc = m.roc_auc_one_vs_rest(y_test, classes, proba)
        fold_rocs.append(roc)

        row = {
            "fold": fold_id,
            "n_train_images": len(train_idx),
            "n_test_images": len(test_idx),
            "n_train_subjects": len(np.unique(groups[train_idx])),
            "n_test_subjects": len(np.unique(groups[test_idx])),
            "optimal_k": best_k,
            "optimal_weights": best_weights,
            "macro_f1": m.macro_f1(y_test, pred, labels),
            "weighted_f1": m.weighted_f1(y_test, pred, labels),
            "macro_auc": m.macro_auc(roc),
        }
        for kk in TOP_K_VALUES:
            kk_eff = min(kk, len(classes))
            row[f"top{kk}_accuracy"] = m.top_k_accuracy(
                y_test, classes, proba, kk_eff, tie_handling=tie_handling)
            lo, hi = m.top_k_accuracy_bounds(y_test, classes, proba, kk_eff)
            row[f"top{kk}_accuracy_lo"] = lo
            row[f"top{kk}_accuracy_hi"] = hi
            row[f"top{kk}_tied_fraction"] = m.tied_boundary_fraction(
                y_test, classes, proba, kk_eff)
        fold_rows.append(row)

        logger.info(
            "[%s] fold %2d/%d | k=%2d w=%-8s | F1=%.4f | Top-1=%.4f | AUC=%.4f | test subj=%d",
            metric, fold_id + 1, OUTER_FOLDS, best_k, best_weights,
            row["macro_f1"], row["top1_accuracy"], row["macro_auc"], row["n_test_subjects"],
        )

    if np.any(oof_fold < 0):
        raise RuntimeError("Some samples did not receive an outer out-of-fold prediction.")

    fold_df = pd.DataFrame(fold_rows)
    inner_df = pd.concat(inner_tables, ignore_index=True)

    # --- k-grid saturation check -------------------------------------------
    n_at_ceiling = int(np.sum(np.array(selected_ks) == k_max))
    saturated = n_at_ceiling > 0
    if saturated:
        logger.warning(
            "[%s] k hit the ceiling (k=%d) in %d/%d outer folds. The inner-CV "
            "optimum may lie outside the search grid; re-run with a larger "
            "--k_max before describing the selected k as optimal.",
            metric, k_max, n_at_ceiling, OUTER_FOLDS,
        )

    # --- Pooled out-of-fold metrics ----------------------------------------
    labels = list(classes)
    pooled_roc = m.roc_auc_one_vs_rest(y, classes, oof_proba)
    pooled = {
        "macro_f1_oof": m.macro_f1(y, oof_pred, labels),
        "weighted_f1_oof": m.weighted_f1(y, oof_pred, labels),
        "macro_auc_oof": m.macro_auc(pooled_roc),
    }
    for kk in TOP_K_VALUES:
        kk_eff = min(kk, len(classes))
        pooled[f"top{kk}_accuracy_oof"] = m.top_k_accuracy(
            y, classes, oof_proba, kk_eff, tie_handling=tie_handling)
        lo, hi = m.top_k_accuracy_bounds(y, classes, oof_proba, kk_eff)
        pooled[f"top{kk}_accuracy_oof_lo"] = lo
        pooled[f"top{kk}_accuracy_oof_hi"] = hi
        pooled[f"top{kk}_tied_fraction_oof"] = m.tied_boundary_fraction(
            y, classes, oof_proba, kk_eff)

    # --- Fold-average metrics (the primary CV estimates) -------------------
    mean_metrics = {}
    numeric_cols = ["macro_f1", "weighted_f1", "macro_auc"] + \
                   [f"top{kk}_accuracy" for kk in TOP_K_VALUES]
    for col in numeric_cols:
        mean_metrics[col] = float(fold_df[col].mean())
        mean_metrics[f"{col}_std"] = float(fold_df[col].std(ddof=1))

    # --- Persist artefacts --------------------------------------------------
    fold_df.to_csv(output_dir / f"fold_metrics_{metric}.csv", index=False)
    inner_df.to_csv(output_dir / f"nested_k_selection_{metric}.csv", index=False)

    k_counts = pd.Series(selected_ks).value_counts().sort_index()
    k_summary = pd.DataFrame({"k": k_counts.index, "outer_fold_count": k_counts.values})
    k_summary["selection_frequency"] = k_summary["outer_fold_count"] / OUTER_FOLDS
    k_summary.to_csv(output_dir / f"optimal_k_frequency_{metric}.csv", index=False)

    # Out-of-fold predictions AND probabilities: every image scored exactly
    # once by a model that never saw its subject. Probabilities are saved so
    # AUC and Top-k can be re-derived and audited without re-running the CV.
    oof_df = pd.DataFrame({
        "syndrome_id": y,
        "subject_id": groups,
        "predicted_syndrome": oof_pred,
        "outer_fold": oof_fold,
    })
    for j, cls in enumerate(classes):
        oof_df[f"proba_{cls}"] = oof_proba[:, j]
    oof_df.to_csv(output_dir / f"oof_predictions_{metric}.csv", index=False)
    np.savez_compressed(
        output_dir / f"oof_probabilities_{metric}.npz",
        proba=oof_proba, classes=classes.astype(str),
        y=y.astype(str), fold=oof_fold,
    )

    # Frequency ties are common; record them instead of hiding them behind
    # idxmax, which silently returns the smallest tied k.
    modal_count = int(k_counts.max())
    modal_ks = [int(kk) for kk in k_counts[k_counts == modal_count].index]

    _plot_k_selection(inner_df, metric, k_max, output_dir)

    return {
        "metric": metric,
        "scaling": scaling_mode,
        "modal_k": modal_ks[0],
        "modal_k_ties": modal_ks,
        "selected_ks": selected_ks,
        "selected_weights": selected_weights,
        "k_max": k_max,
        "k_grid_saturated": saturated,
        "n_folds_at_k_ceiling": n_at_ceiling,
        "mean_metrics": mean_metrics,
        "pooled_metrics": pooled,
        "fold_metrics": fold_df,
        "fold_rocs": fold_rocs,
        "oof_proba": oof_proba,
        "oof_pred": oof_pred,
        "classes": classes,
    }


def _plot_k_selection(inner_df, metric, k_max, output_dir):
    """Inner-CV macro-F1 vs k, averaged across outer folds, per weighting."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for weights, sub in inner_df.groupby("weights"):
        grouped = sub.groupby("k")["mean_macro_f1"].agg(["mean", "std"])
        ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"],
                    marker="o", capsize=3, label=f"weights={weights}")
    ax.axvline(k_max, linestyle=":", color="grey", lw=1)
    ax.annotate("search ceiling", xy=(k_max, ax.get_ylim()[0]),
                xytext=(-6, 8), textcoords="offset points",
                rotation=90, fontsize=7, color="grey", ha="right")
    ax.set_xlabel("k (number of neighbours)")
    ax.set_ylabel("Inner-CV macro-F1")
    ax.set_title(f"Nested k Selection - {metric.capitalize()} Distance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"k_selection_{metric}.png", dpi=150)
    plt.close(fig)


# Reporting helpers

def plot_roc_comparison(results_cosine, results_euclidean, output_dir):
    """Fold-averaged macro ROC curves with fold-to-fold variability."""
    fig, ax = plt.subplots(figsize=(8, 7))
    grid = np.linspace(0, 1, 200)

    for res, label in [(results_cosine, "Cosine"), (results_euclidean, "Euclidean")]:
        fold_tprs = np.asarray([
            m.mean_roc_curve(roc, n_points=len(grid))[1] for roc in res["fold_rocs"]
        ])
        mean_tpr = fold_tprs.mean(axis=0)
        std_tpr = fold_tprs.std(axis=0, ddof=1)
        mm = res["mean_metrics"]
        ax.plot(grid, mean_tpr, lw=2,
                label=f"{label} (macro-AUC={mm['macro_auc']:.3f} ± {mm['macro_auc_std']:.3f})")
        ax.fill_between(grid, np.maximum(0, mean_tpr - std_tpr),
                        np.minimum(1, mean_tpr + std_tpr), alpha=0.15)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=1, color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("10-Fold Subject-Grouped Macro-Averaged ROC: Cosine vs Euclidean")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path = Path(output_dir) / "roc_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC comparison plot to %s", out_path)


def build_summary_table(results_cosine, results_euclidean):
    """Summary table. Column names state what the number actually is.

    'Modal k' is the k most frequently chosen by the inner CV -- it is not a
    validated optimum, and calling it 'best k' would overstate the evidence.
    """
    rows = []
    for res in (results_cosine, results_euclidean):
        mm, pooled = res["mean_metrics"], res["pooled_metrics"]
        rows.append({
            "Distance Metric": res["metric"].capitalize(),
            "Preprocessing": res["scaling"],
            "Modal k": res["modal_k"],
            "k ceiling hit": f"{res['n_folds_at_k_ceiling']}/{OUTER_FOLDS}",
            "Macro F1 (mean ± std)": f"{mm['macro_f1']:.4f} ± {mm['macro_f1_std']:.4f}",
            "Weighted F1 (mean ± std)": f"{mm['weighted_f1']:.4f} ± {mm['weighted_f1_std']:.4f}",
            "Top-1 (mean ± std)": f"{mm['top1_accuracy']:.4f} ± {mm['top1_accuracy_std']:.4f}",
            "Top-3 (mean ± std)": f"{mm['top3_accuracy']:.4f} ± {mm['top3_accuracy_std']:.4f}",
            "Top-5 (mean ± std)": f"{mm['top5_accuracy']:.4f} ± {mm['top5_accuracy_std']:.4f}",
            "Macro AUC (mean ± std)": f"{mm['macro_auc']:.4f} ± {mm['macro_auc_std']:.4f}",
            "OOF Macro F1": round(pooled["macro_f1_oof"], 4),
            "OOF Macro AUC": round(pooled["macro_auc_oof"], 4),
            "OOF Top-5 [lo, hi]": f"[{pooled['top5_accuracy_oof_lo']:.4f}, "
                                  f"{pooled['top5_accuracy_oof_hi']:.4f}]",
            "OOF Top-5 tied frac.": round(pooled["top5_tied_fraction_oof"], 4),
        })
    return pd.DataFrame(rows)


def write_run_manifest(output_dir, results, args_dict):
    """Record how the artefacts in output_dir were produced.

    generate_report.py reads this file; if it is missing, the report says so
    rather than silently describing stale numbers with fresh prose.
    """
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": args_dict,
        "metrics": {
            res["metric"]: {
                "scaling": res["scaling"],
                "modal_k": res["modal_k"],
                "modal_k_ties": res["modal_k_ties"],
                "selected_ks": res["selected_ks"],
                "selected_weights": res["selected_weights"],
                "k_max": res["k_max"],
                "k_grid_saturated": res["k_grid_saturated"],
                "n_folds_at_k_ceiling": res["n_folds_at_k_ceiling"],
                "pooled_metrics": {k: float(v) for k, v in res["pooled_metrics"].items()},
                "mean_metrics": {k: float(v) for k, v in res["mean_metrics"].items()},
            }
            for res in results
        },
    }
    path = Path(output_dir) / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote run manifest to %s", path)
    return path


# CLI

def parse_args():
    p = argparse.ArgumentParser(
        description="Leakage-safe K-NN classification of syndrome embeddings.")
    p.add_argument("--input", type=str, default="outputs/embeddings.npz")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--k_max", type=int, default=DEFAULT_K_MAX,
                   help="Upper bound of the k search grid. A warning is emitted "
                        "if the selected k reaches this ceiling.")
    p.add_argument("--scaling", choices=["auto", "standard", "l2", "none"], default="auto",
                   help="auto: L2 for cosine, StandardScaler for euclidean.")
    p.add_argument("--no_tune_weights", action="store_true",
                   help="Fix weights='uniform' instead of tuning it in the inner CV.")
    p.add_argument("--tie_handling", choices=list(m.VALID_TIE_HANDLING),
                   default="fractional",
                   help="How Top-k credits classes tied at the k-th position.")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.input, allow_pickle=False)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(str)
    groups = data["subject_id"].astype(str)

    check_subject_syndrome_consistency(y, groups)

    common = dict(random_state=args.random_state, k_max=args.k_max,
                  scaling=args.scaling, tune_weights=not args.no_tune_weights,
                  tie_handling=args.tie_handling)
    res_cos = evaluate_metric(X, y, groups, "cosine", output_dir, **common)
    res_euc = evaluate_metric(X, y, groups, "euclidean", output_dir, **common)

    plot_roc_comparison(res_cos, res_euc, output_dir)

    summary = build_summary_table(res_cos, res_euc)
    summary.to_csv(output_dir / "performance_summary.csv", index=False)
    write_run_manifest(output_dir, [res_cos, res_euc], vars(args))
    logger.info("Final performance summary:\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
