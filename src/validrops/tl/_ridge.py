"""Logistic ridge with glmnet's lambda.1se rule, and pROC-compatible thresholds."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold


class _Fitted(LogisticRegression):
    """LogisticRegression carrying the two selected penalties."""

    C_min: float
    C_1se: float


def logistic_ridge_1se(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    nfolds: int = 5,
    n_alphas: int = 100,
    random_state: int = 0,
) -> _Fitted:
    """L2 logistic regression at glmnet's ``lambda.1se``.

    ``cv.glmnet(..., s = "lambda.1se")`` picks the strongest regularisation
    whose cross-validated deviance is within one standard error of the
    minimum. sklearn has no equivalent, so it is implemented here.

    Parameters
    ----------
    X
        Design matrix.
    y
        Binary labels.
    sample_weight
        Per-observation weights.
    nfolds
        Cross-validation folds.
    n_alphas
        Points on the penalty path.
    random_state
        Seed for the fold split.

    Returns
    -------
    A fitted model with ``C_min`` and ``C_1se`` attached.
    """
    grid = np.logspace(-4, 4, n_alphas)
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError(f"expected two classes, got {classes.size}")
    binary = (y == classes[1]).astype(int)

    splitter = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=random_state)
    deviance = np.zeros((nfolds, grid.size))

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, binary)):
        weights = None if sample_weight is None else sample_weight[train_idx]
        for j, c in enumerate(grid):
            model = LogisticRegression(C=c, max_iter=1000)
            model.fit(X[train_idx], binary[train_idx], sample_weight=weights)
            probs = np.clip(model.predict_proba(X[test_idx])[:, 1], 1e-10, 1 - 1e-10)
            actual = binary[test_idx]
            deviance[fold, j] = -2 * np.sum(actual * np.log(probs) + (1 - actual) * np.log(1 - probs))

    mean = deviance.mean(axis=0)
    stderr = deviance.std(axis=0, ddof=1) / np.sqrt(nfolds)
    best = int(np.argmin(mean))
    within = np.flatnonzero(mean <= mean[best] + stderr[best])
    # smaller C means stronger regularisation
    chosen = int(within[np.argmin(grid[within])])

    final = _Fitted(C=grid[chosen], max_iter=1000)
    final.fit(X, binary, sample_weight=sample_weight)
    final.C_min = float(grid[best])
    final.C_1se = float(grid[chosen])
    return final


def roc_best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Youden-optimal threshold, matching ``pROC::coords(roc, "best")``.

    Ties are broken toward the lowest threshold at the highest specificity,
    which is what ``label_dead.R:295-299`` falls back to.

    Parameters
    ----------
    labels
        ``"live"``/``"dead"`` per observation; ``"dead"`` is the positive class.
    scores
        Predicted probability of being dead.

    Returns
    -------
    ``(threshold, specificity)``.
    """
    positive = (np.asarray(labels) == "dead").astype(int)
    fpr, tpr, thresholds = roc_curve(positive, np.asarray(scores, dtype=np.float64))
    youden = tpr - fpr
    best = np.flatnonzero(youden == youden.max())
    specificity = 1.0 - fpr[best]
    at_max_spec = best[specificity == specificity.max()]
    chosen = at_max_spec[np.argmin(thresholds[at_max_spec])]
    return float(thresholds[chosen]), float(1.0 - fpr[chosen])
