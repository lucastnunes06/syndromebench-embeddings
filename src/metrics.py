"""Hand-rolled metrics: macro/weighted F1, Top-k accuracy, one-vs-rest ROC/AUC
(trapezoid rule) and confusion matrix.

Kept independent of sklearn.metrics on purpose so the behaviour is visible.
Only basic numpy. Everything is checked against sklearn in the tests.

About Top-k and ties
--------------------
Uniform-weight KNN only produces probs that are multiples of 1/k and zeros
everything outside the neighbourhood. With 10 classes and k around 12 most
classes sit at 0. argsort breaks ties by column index, so Top-3/5 end up
depending on alphabetical label order. Quick simulation moved Top-5 by up
to ~4.5 pp just by permuting columns.

top_k_accuracy handles ties explicitly:

  n_above = classes strictly above the true one
  n_tied  = classes tied with the true one (including it)

  n_above >= k          → 0
  n_above + n_tied <= k → 1
  otherwise             → true class is inside the block that cuts Top-k

tie_handling options:
  fractional  → (k - n_above) / n_tied   (default)
  optimistic  → 1.0
  pessimistic → 0.0

top_k_accuracy_bounds returns the interval. When the gap is wide, Top-k
isn't really well-defined for that model/k; distance weighting mostly
removes the issue.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

VALID_TIE_HANDLING = ("fractional", "optimistic", "pessimistic")


# ---------------------------------------------------------------------------
# Confusion counts and F1
# ---------------------------------------------------------------------------

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray,
                     labels: Sequence) -> Dict[object, Dict[str, int]]:
    """Per-class TP / FP / FN / TN counts (one-vs-rest)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    counts = {}
    for lab in labels:
        tp = int(np.sum((y_true == lab) & (y_pred == lab)))
        fp = int(np.sum((y_true != lab) & (y_pred == lab)))
        fn = int(np.sum((y_true == lab) & (y_pred != lab)))
        tn = int(np.sum((y_true != lab) & (y_pred != lab)))
        counts[lab] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return counts


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     labels: Sequence) -> np.ndarray:
    """Full multiclass confusion matrix, rows = true, cols = predicted.

    Implemented manually (no sklearn) to stay consistent with the rest of
    this module. Row i / column j counts samples whose true label is
    labels[i] and whose predicted label is labels[j].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    index = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            cm[index[t], index[p]] += 1
    return cm


def precision_recall_f1_per_class(y_true: np.ndarray, y_pred: np.ndarray,
                                  labels: Sequence) -> Dict[object, Dict[str, float]]:
    counts = confusion_counts(y_true, y_pred, labels)
    out = {}
    for lab, c in counts.items():
        denom_p = c["tp"] + c["fp"]
        denom_r = c["tp"] + c["fn"]
        precision = c["tp"] / denom_p if denom_p > 0 else 0.0
        recall = c["tp"] / denom_r if denom_r > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        out[lab] = {"precision": precision, "recall": recall, "f1": f1,
                    "support": denom_r}
    return out


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence) -> float:
    """Unweighted mean of per-class F1 scores."""
    per_class = precision_recall_f1_per_class(y_true, y_pred, labels)
    f1s = [v["f1"] for v in per_class.values()]
    return float(np.mean(f1s)) if f1s else 0.0


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence) -> float:
    """Support-weighted mean of per-class F1 scores (robust to class imbalance)."""
    per_class = precision_recall_f1_per_class(y_true, y_pred, labels)
    total_support = sum(v["support"] for v in per_class.values())
    if total_support == 0:
        return 0.0
    return float(sum(v["f1"] * v["support"] for v in per_class.values()) / total_support)


# ---------------------------------------------------------------------------
# Top-k accuracy (tie aware)
# ---------------------------------------------------------------------------

def _top_k_per_sample(y_true: np.ndarray, class_order: np.ndarray,
                      proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (n_above, n_tied, is_true_present) arrays, one entry per sample.

    n_above -- classes scored strictly above the true class
    n_tied  -- classes scored exactly equal to the true class (incl. itself)
    """
    class_order = np.asarray(class_order)
    y_true = np.asarray(y_true)

    index = {lab: j for j, lab in enumerate(class_order)}
    true_cols = np.array([index.get(lab, -1) for lab in y_true])
    present = true_cols >= 0

    n = len(y_true)
    safe_cols = np.where(present, true_cols, 0)
    true_scores = proba[np.arange(n), safe_cols]

    n_above = np.sum(proba > true_scores[:, None], axis=1)
    n_tied = np.sum(proba == true_scores[:, None], axis=1)
    return n_above, n_tied, present


def top_k_accuracy(y_true: np.ndarray, class_order: np.ndarray,
                   proba: np.ndarray, k: int,
                   tie_handling: str = "fractional") -> float:
    """Tie-aware Top-k accuracy.

    Parameters
    ----------
    y_true : (n_samples,) true labels
    class_order : (n_classes,) label for each column of `proba`
    proba : (n_samples, n_classes) predicted scores (higher = more likely)
    k : number of top predictions to consider
    tie_handling : "fractional" (default), "optimistic" or "pessimistic".
        See the module docstring; ties are common and consequential for K-NN.
    """
    if tie_handling not in VALID_TIE_HANDLING:
        raise ValueError(f"tie_handling must be one of {VALID_TIE_HANDLING}")

    n = len(y_true)
    if n == 0:
        return 0.0
    k = int(min(k, proba.shape[1]))

    n_above, n_tied, present = _top_k_per_sample(y_true, class_order, proba)

    credit = np.zeros(n, dtype=float)
    clear_hit = (n_above + n_tied) <= k
    straddling = (n_above < k) & ~clear_hit

    credit[clear_hit] = 1.0
    if tie_handling == "optimistic":
        credit[straddling] = 1.0
    elif tie_handling == "fractional":
        credit[straddling] = (k - n_above[straddling]) / n_tied[straddling]
    # "pessimistic" leaves them at 0.0

    credit[~present] = 0.0
    return float(np.mean(credit))


def top_k_accuracy_bounds(y_true: np.ndarray, class_order: np.ndarray,
                          proba: np.ndarray, k: int) -> Tuple[float, float]:
    """(pessimistic, optimistic) bounds on Top-k accuracy given ties.

    A wide interval means the metric is dominated by arbitrary tie-breaking
    rather than by the model's ranking ability.
    """
    lo = top_k_accuracy(y_true, class_order, proba, k, tie_handling="pessimistic")
    hi = top_k_accuracy(y_true, class_order, proba, k, tie_handling="optimistic")
    return lo, hi


def tied_boundary_fraction(y_true: np.ndarray, class_order: np.ndarray,
                           proba: np.ndarray, k: int) -> float:
    """Fraction of samples whose Top-k verdict is decided by a tie.

    This is the diagnostic that justifies (or refutes) reporting Top-k at
    all for a given model. Values near 1.0 mean Top-k is not identifiable.
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    k = int(min(k, proba.shape[1]))
    n_above, n_tied, _ = _top_k_per_sample(y_true, class_order, proba)
    straddling = (n_above < k) & ((n_above + n_tied) > k)
    return float(np.mean(straddling))


# ---------------------------------------------------------------------------
# ROC / AUC
# ---------------------------------------------------------------------------

def _roc_curve_binary(y_true_bin: np.ndarray,
                      scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Manual ROC curve computation for a single binary problem.

    Sorts samples by descending score and sweeps the decision threshold
    across every distinct score value, computing TPR/FPR at each cut point.
    Tied scores are collapsed into a single threshold, which is what makes
    the resulting AUC equal to the Mann-Whitney statistic even when the
    classifier emits few distinct probability values.

    Returns (fpr, tpr, thresholds), each starting at (0,0) and ending at (1,1).
    """
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true_bin[order]
    scores_sorted = scores[order]

    n_pos = int(np.sum(y_true_bin == 1))
    n_neg = int(np.sum(y_true_bin == 0))
    if n_pos == 0 or n_neg == 0:
        # Degenerate case: a proper ROC curve is undefined.
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)

    # Keep only the last index of every run of equal scores.
    distinct_idx = np.where(np.diff(scores_sorted, append=np.nan) != 0)[0]

    tpr = tps[distinct_idx] / n_pos
    fpr = fps[distinct_idx] / n_neg
    thresh = scores_sorted[distinct_idx]

    fpr = np.concatenate(([0.0], fpr, [1.0]))
    tpr = np.concatenate(([0.0], tpr, [1.0]))
    thresh = np.concatenate(([np.inf], thresh, [-np.inf]))
    return fpr, tpr, thresh


def _trapezoidal_area(y: np.ndarray, x: np.ndarray) -> float:
    """Manual trapezoidal-rule integration.

    Avoids numpy.trapz, which was removed in numpy 2.0 in favour of
    numpy.trapezoid; doing it by hand keeps the module working on both.
    """
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


def _auc_trapezoidal(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Area under a curve given (x, y) points, via the trapezoidal rule."""
    order = np.argsort(fpr, kind="mergesort")
    return _trapezoidal_area(tpr[order], fpr[order])


def roc_auc_one_vs_rest(y_true: np.ndarray, class_order: np.ndarray,
                        proba: np.ndarray) -> Dict[object, Dict[str, np.ndarray]]:
    """One-vs-rest ROC curve + AUC for every class.

    Returns {label: {"fpr": ..., "tpr": ..., "auc": float}}.
    Classes absent from `y_true` get auc = nan and are skipped by macro_auc.
    """
    results = {}
    y_true = np.asarray(y_true)
    for j, lab in enumerate(class_order):
        y_bin = (y_true == lab).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            results[lab] = {"fpr": np.array([0.0, 1.0]),
                            "tpr": np.array([0.0, 1.0]),
                            "auc": float("nan")}
            continue
        fpr, tpr, _ = _roc_curve_binary(y_bin, proba[:, j])
        results[lab] = {"fpr": fpr, "tpr": tpr, "auc": _auc_trapezoidal(fpr, tpr)}
    return results


def macro_auc(roc_results: Dict[object, Dict[str, np.ndarray]]) -> float:
    """Unweighted mean AUC across classes (macro-average one-vs-rest AUC)."""
    aucs = [v["auc"] for v in roc_results.values() if not np.isnan(v["auc"])]
    return float(np.mean(aucs)) if aucs else float("nan")


def mean_roc_curve(roc_results: Dict[object, Dict[str, np.ndarray]],
                   n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Macro-average ROC curve: interpolate each class onto a common FPR grid.

    Classes with an undefined ROC (absent from the fold) are skipped so they
    do not drag the average toward the chance diagonal.
    """
    grid = np.linspace(0, 1, n_points)
    interp_tprs = [np.interp(grid, v["fpr"], v["tpr"])
                   for v in roc_results.values() if not np.isnan(v["auc"])]
    if not interp_tprs:
        return grid, grid.copy()
    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
    return grid, mean_tpr


if __name__ == "__main__":
    # Lightweight self-test; the real checks live in tests/test_metrics.py.
    rng = np.random.default_rng(0)
    labels = np.array(["A", "B", "C"])
    y_true = rng.choice(labels, size=200)
    proba = rng.random((200, 3))
    proba /= proba.sum(axis=1, keepdims=True)
    y_pred = labels[np.argmax(proba, axis=1)]

    print("Macro F1:   ", macro_f1(y_true, y_pred, list(labels)))
    print("Weighted F1:", weighted_f1(y_true, y_pred, list(labels)))
    print("Top-1 acc:  ", top_k_accuracy(y_true, labels, proba, k=1))
    print("Top-2 acc:  ", top_k_accuracy(y_true, labels, proba, k=2))
    print("Macro AUC:  ", macro_auc(roc_auc_one_vs_rest(y_true, labels, proba)))
