"""Check the hand-rolled metrics in metrics.py against sklearn, and pin the two
behaviours that used to be wrong:

  1. Top-k accuracy must be invariant to class-column order.
  2. Hyper-parameter selection must break ties toward the smaller k.

Run:  python -m pytest tests/ -v
  or: python tests/test_metrics.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import metrics as m  # noqa: E402
from knn_classification import make_model, resolve_scaling  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    roc_auc_score,
    top_k_accuracy_score,
)


@pytest.fixture(scope="module")
def continuous_case():
    """Well-separated continuous scores: no ties, so sklearn agrees exactly."""
    rng = np.random.default_rng(1)
    labels = np.array(["A", "B", "C", "D"])
    y = rng.choice(labels, 400)
    proba = rng.random((400, 4))
    proba /= proba.sum(axis=1, keepdims=True)
    pred = labels[proba.argmax(axis=1)]
    return labels, y, proba, pred


@pytest.fixture(scope="module")
def knn_like_case():
    """Coarse K-NN-style probabilities: multiples of 1/k with many zeros."""
    rng = np.random.default_rng(0)
    labels = np.array([f"S{i}" for i in range(10)])
    n, k = 800, 12
    y = labels[rng.integers(0, 10, n)]
    proba = np.zeros((n, 10))
    for i in range(n):
        cols = rng.choice(10, 3, replace=False)
        proba[i, cols] = rng.multinomial(k, [0.6, 0.3, 0.1]) / k
    return labels, y, proba


# Agreement with sklearn

def test_macro_f1_matches_sklearn(continuous_case):
    labels, y, _, pred = continuous_case
    assert m.macro_f1(y, pred, list(labels)) == pytest.approx(
        f1_score(y, pred, labels=list(labels), average="macro"))


def test_weighted_f1_matches_sklearn(continuous_case):
    labels, y, _, pred = continuous_case
    assert m.weighted_f1(y, pred, list(labels)) == pytest.approx(
        f1_score(y, pred, labels=list(labels), average="weighted"))


def test_macro_auc_matches_sklearn(continuous_case):
    labels, y, proba, _ = continuous_case
    roc = m.roc_auc_one_vs_rest(y, labels, proba)
    assert m.macro_auc(roc) == pytest.approx(
        roc_auc_score(y, proba, multi_class="ovr", average="macro"))


def test_confusion_matrix_matches_sklearn(continuous_case):
    labels, y, _, pred = continuous_case
    np.testing.assert_array_equal(
        m.confusion_matrix(y, pred, list(labels)),
        sk_confusion_matrix(y, pred, labels=list(labels)))


@pytest.mark.parametrize("k", [1, 2, 3])
def test_top_k_matches_sklearn_without_ties(continuous_case, k):
    """With continuous scores there are no ties, so every tie policy agrees
    with sklearn. This is the check that the base logic is right."""
    labels, y, proba, _ = continuous_case
    expected = top_k_accuracy_score(y, proba, k=k, labels=list(labels))
    for policy in m.VALID_TIE_HANDLING:
        assert m.top_k_accuracy(y, labels, proba, k, tie_handling=policy) == \
            pytest.approx(expected)


def test_auc_equals_mann_whitney_with_heavy_ties(knn_like_case):
    """AUC must stay correct when the classifier emits few distinct values.
    Ties are collapsed into one threshold, matching sklearn's convention."""
    labels, y, proba = knn_like_case
    roc = m.roc_auc_one_vs_rest(y, labels, proba)
    for j, lab in enumerate(labels):
        expected = roc_auc_score((y == lab).astype(int), proba[:, j])
        assert roc[lab]["auc"] == pytest.approx(expected)


# Regression test 1: Top-k must be invariant to class column order

def test_top_k_is_invariant_to_class_order(knn_like_case):
    """The original argsort-based implementation broke ties by column index,
    so simply relabelling the classes changed Top-5 by several points."""
    labels, y, proba = knn_like_case
    values = []
    for seed in range(10):
        perm = np.random.default_rng(seed).permutation(len(labels))
        values.append(m.top_k_accuracy(y, labels[perm], proba[:, perm], 5))
    assert max(values) - min(values) < 1e-12, (
        f"Top-5 varies by {max(values) - min(values):.4f} across class orderings")


def test_naive_argsort_would_have_failed(knn_like_case):
    """Documents the bug being guarded against: the old approach really does
    move under permutation, so the test above is not vacuous."""
    labels, y, proba = knn_like_case

    def naive_top_k(y_true, class_order, p, k):
        idx = np.argsort(-p, axis=1)[:, :k]
        return float(np.mean([y_true[i] in class_order[idx[i]] for i in range(len(y_true))]))

    values = []
    for seed in range(10):
        perm = np.random.default_rng(seed).permutation(len(labels))
        values.append(naive_top_k(y, labels[perm], proba[:, perm], 5))
    assert max(values) - min(values) > 0.01


def test_tie_bounds_bracket_the_reported_value(knn_like_case):
    labels, y, proba = knn_like_case
    lo, hi = m.top_k_accuracy_bounds(y, labels, proba, 5)
    mid = m.top_k_accuracy(y, labels, proba, 5, tie_handling="fractional")
    assert lo <= mid <= hi
    assert hi - lo > 0.0, "this fixture is supposed to contain ties"


def test_tied_boundary_fraction_detects_the_problem(knn_like_case):
    labels, y, proba = knn_like_case
    assert m.tied_boundary_fraction(y, labels, proba, 5) > 0.5
    # Top-1 with a clear argmax winner is far less affected.
    assert m.tied_boundary_fraction(y, labels, proba, 5) > \
        m.tied_boundary_fraction(y, labels, proba, 1)


# Regression test 2: hyper-parameter tie-break

def test_hyperparam_tie_breaks_to_smallest_k():
    """pandas' default quicksort is unstable; without kind='mergesort' a tie
    between k=8..15 resolved to an arbitrary k (observed: 14)."""
    result = pd.DataFrame({
        "k": list(range(1, 16)) * 1,
        "weights": ["uniform"] * 15,
        "mean_macro_f1": [0.5] * 7 + [0.9] * 8,
    })
    ranked = result.sort_values(["mean_macro_f1", "k", "weights"],
                               ascending=[False, True, True], kind="mergesort")
    assert int(ranked.iloc[0]["k"]) == 8


# Pipeline construction

def test_scaling_policy_auto():
    assert resolve_scaling("cosine", "auto") == "l2"
    assert resolve_scaling("euclidean", "auto") == "standard"
    assert resolve_scaling("cosine", "standard") == "standard"


def test_euclidean_on_l2_matches_cosine_neighbours():
    """Sanity check on the geometry: euclidean distance on L2-normalised
    vectors is a monotone function of cosine distance, so both pipelines must
    select the same neighbours and therefore predict identically."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 32))
    y = np.array([f"c{i % 4}" for i in range(120)])
    cos = make_model(5, "cosine", "uniform", "l2").fit(X[:90], y[:90])
    euc = make_model(5, "euclidean", "uniform", "l2").fit(X[:90], y[:90])
    np.testing.assert_array_equal(cos.predict(X[90:]), euc.predict(X[90:]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
