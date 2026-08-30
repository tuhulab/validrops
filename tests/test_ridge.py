import numpy as np
from sklearn.datasets import make_classification

from validrops.tl._ridge import logistic_ridge_1se, roc_best_threshold


def test_1se_is_more_regularised_than_the_minimum():
    X, y = make_classification(n_samples=300, n_features=20, random_state=0)
    model = logistic_ridge_1se(X, y, nfolds=5, random_state=0, n_alphas=40)
    assert model.C_1se <= model.C_min + 1e-12


def test_predicts_probabilities_in_range():
    X, y = make_classification(n_samples=200, n_features=10, random_state=1)
    model = logistic_ridge_1se(X, y, nfolds=5, random_state=0, n_alphas=40)
    probs = model.predict_proba(X)[:, 1]
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_is_deterministic_for_a_seed():
    X, y = make_classification(n_samples=200, n_features=10, random_state=2)
    a = logistic_ridge_1se(X, y, nfolds=5, random_state=3, n_alphas=40).predict_proba(X)[:, 1]
    b = logistic_ridge_1se(X, y, nfolds=5, random_state=3, n_alphas=40).predict_proba(X)[:, 1]
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_sample_weights_shift_the_fit():
    X, y = make_classification(n_samples=200, n_features=5, random_state=4)
    w = np.where(y == 1, 10.0, 0.1)
    unweighted = logistic_ridge_1se(X, y, nfolds=5, random_state=0, n_alphas=40).predict_proba(X)[:, 1]
    weighted = logistic_ridge_1se(X, y, sample_weight=w, nfolds=5, random_state=0, n_alphas=40).predict_proba(X)[:, 1]
    assert weighted.mean() > unweighted.mean()


def test_roc_best_threshold_on_perfect_separation():
    labels = np.array(["live", "live", "dead", "dead"])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    threshold, specificity = roc_best_threshold(labels, scores)
    assert specificity == 1.0
    assert 0.2 <= threshold <= 0.8


def test_roc_best_threshold_on_random_scores_is_finite():
    rng = np.random.default_rng(0)
    labels = rng.choice(["live", "dead"], size=200)
    threshold, specificity = roc_best_threshold(labels, rng.random(200))
    assert np.isfinite(threshold)
    assert 0.0 <= specificity <= 1.0
