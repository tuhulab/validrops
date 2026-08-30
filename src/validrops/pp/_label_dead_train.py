"""Stage 4 consensus training. Ports the stochastic half of ``label_dead.R:155-479``."""

import logging

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from joblib import Parallel, delayed
from scipy.sparse.linalg import svds
from scipy.stats import kendalltau
from sklearn.metrics import roc_curve

from validrops._constants import (
    DEAD_CELL_CONSENSUS,
    DEAD_CELL_RUNS,
    DEAD_COR_MAX,
    DEAD_COR_MIN,
    DEAD_COR_STEPS,
    DEAD_EPOCHS,
    DEAD_FAIL_WEIGHT,
    DEAD_FEATURE_TRY,
    DEAD_MAX_LIVE,
    DEAD_MIN_DEAD,
    DEAD_NFOLDS,
    DEAD_NPCS,
    DEAD_NREP,
    DEAD_NREP_COR,
)
from validrops.tl._ridge import logistic_ridge_1se, roc_best_threshold

logger = logging.getLogger(__name__)

# glmnet's default penalty path is 100 points; logistic_ridge_1se is called
# from the cor-threshold search AND the training loop (10 runs x up to 20
# epochs x 10 replicates), so the path is shortened to 30 points to keep the
# stage tractable (~3x fewer fits).
NB_ALPHAS = 30


def consensus(runs: np.ndarray, n_min: int = DEAD_CELL_CONSENSUS) -> np.ndarray:
    """Combine per-run labels. Ports ``label_dead.R:459``.

    Parameters
    ----------
    runs
        ``(n_barcodes, n_runs)`` array of ``"live"``/``"dead"``.
    n_min
        Runs that must agree for a confident call.

    Returns
    -------
    ``"live"``, ``"dead"`` or ``"uncertain"`` per barcode.
    """
    n_live = (runs == "live").sum(axis=1)
    n_dead = (runs == "dead").sum(axis=1)
    return np.where(n_live >= n_min, "live", np.where(n_dead >= n_min, "dead", "uncertain"))


def escalate_flag(flag: str, uncertain_fraction: float) -> str:
    """Downgrade the run flag when too many barcodes are uncertain.

    Ports the uncertain-fraction checks at ``label_dead.R:467-478``: a medium
    share escalates Success to Caution, a high share forces Failed.
    """
    if flag == "Failed":
        return "Failed"
    if uncertain_fraction >= 0.025:
        return "Failed"
    if uncertain_fraction >= 0.0125:
        return "Caution"
    return flag


def _embed(adata: AnnData, barcodes, npcs: int, random_state: int) -> np.ndarray:
    """Per-cell scaled embedding of raw counts. Ports ``label_dead.R:163-179``.

    Two quirks preserved, both from the R source:
    - ``colMeans`` on R's genes x cells matrix is a **per-cell** mean across
      genes (:func:`~validrops.tl.expression_metrics` scales per gene instead;
      the two genuinely differ and Stage 4's is reproduced here).
    - ``label_dead.R:166-167`` computes variable features and never uses them;
      the SVD runs on all non-zero genes. That computation is omitted.

    R also returns ``svd$u`` without multiplying by the singular values, so no
    ``* s`` here either. ``npcs`` is clipped to ``min(shape) - 1`` so ARPACK
    never asks for more vectors than observations.
    """
    counts = sp.csr_matrix(adata[barcodes].X, dtype=np.float64)
    counts = counts[:, np.asarray(counts.sum(axis=0, dtype=np.float64)).ravel() > 0]

    # per-cell mean and sample sd across genes, computed sparsely to avoid a
    # second dense copy of the ~33k-gene matrix
    n_genes = counts.shape[1]
    sums = np.asarray(counts.sum(axis=1, dtype=np.float64)).ravel()
    means = sums / n_genes
    sq = counts.copy()
    sq.data **= 2
    mean_sq = np.asarray(sq.sum(axis=1, dtype=np.float64)).ravel() / n_genes
    sds = np.sqrt((mean_sq - means**2) * (n_genes / (n_genes - 1)))
    sds[sds == 0] = 1.0

    scaled = np.asarray(counts.todense(), dtype=np.float64)
    scaled -= means[:, None]
    scaled /= sds[:, None]

    npcs = min(npcs, min(scaled.shape) - 1)
    u, s, _ = svds(scaled, k=npcs, random_state=random_state)
    order = np.argsort(-s)
    return u[:, order]


def _resample(metrics_qc, labels, probs, rng, min_dead, max_live):
    """Weighted resampling with replacement. Ports ``label_dead.R:386-389``."""

    def draw(qc_value, label_value, size_fn, weights):
        idx = np.flatnonzero((metrics_qc == qc_value) & (labels == label_value))
        if idx.size == 0:
            return idx
        size = size_fn(idx.size)
        w = weights[idx]
        total = w.sum()
        p = w / total if total > 0 else None
        return rng.choice(idx, size=size, replace=True, p=p)

    dead_fail = draw("fail", "dead", lambda n: max(min_dead, n), probs)
    dead_pass = draw("pass", "dead", lambda n: max(min_dead, n), probs)
    live_fail = draw("fail", "live", lambda n: min(max_live, n), np.abs(probs - 1))
    live_pass = draw("pass", "live", lambda n: min(max_live, n), np.abs(probs - 1))
    return np.concatenate([dead_fail, live_fail]), np.concatenate([dead_pass, live_pass])


def _train_rep(X, labels, qc, probs, fail_weight, nfolds, min_dead, max_live, rng, model_seed):
    """One replicate: resample, jitter, fit a ridge, predict P(dead) for all cells.

    Ports the body of the innermost ``for (rep in 1:nrep)`` loop,
    ``label_dead.R:388-413``. Returns ``None`` when the resample degenerates to
    a single class, which R cannot even represent (empty quadrant or one-class
    sample would error).
    """
    fail_idx, pass_idx = _resample(qc, labels, probs, rng, min_dead, max_live)
    idx = np.concatenate([fail_idx, pass_idx])
    if idx.size == 0 or np.unique(labels[idx]).size < 2:
        return None
    sample_X = X[idx].copy()
    sample_y = labels[idx]
    weights = np.ones(idx.size)
    weights[: fail_idx.size] = fail_weight  # label_dead.R:381-382
    for d in range(sample_X.shape[1]):  # jitter, label_dead.R:401
        amount = np.std(sample_X[:, d], ddof=1) / 5
        sample_X[:, d] += rng.uniform(-amount, amount, size=idx.size)
    order = rng.permutation(idx.size)
    # encode explicitly so column 1 of predict_proba is always P(dead);
    # passing the strings would make "live" the positive class, since
    # np.unique sorts "dead" before "live"
    binary = (sample_y == "dead").astype(int)
    model = logistic_ridge_1se(
        sample_X[order],
        binary[order],
        sample_weight=weights[order],
        nfolds=nfolds,
        n_alphas=NB_ALPHAS,
        random_state=model_seed,
    )
    return model.predict_proba(X)[:, 1]


def _cor_taus(embedding, labels) -> np.ndarray:
    """Kendall correlation of each PC with the dead/live label, ``label_dead.R:245-246``."""
    taus = np.array(
        [kendalltau(embedding[:, d], (labels == "dead").astype(float)).statistic for d in range(embedding.shape[1])]
    )
    return np.nan_to_num(taus)


def _search_cor_threshold(
    embedding,
    labels,
    qc,
    rng,
    *,
    cor_min=DEAD_COR_MIN,
    cor_max=DEAD_COR_MAX,
    cor_steps=DEAD_COR_STEPS,
    nrep_cor=DEAD_NREP_COR,
    fail_weight=DEAD_FAIL_WEIGHT,
    nfolds=DEAD_NFOLDS,
    min_dead=DEAD_MIN_DEAD,
    max_live=DEAD_MAX_LIVE,
    feature_try=DEAD_FEATURE_TRY,
    model_seed=0,
):
    """Search the PC-correlation threshold. Ports ``label_dead.R:239-352``.

    Each candidate threshold is scored by training ``nrep_cor`` ridge
    replicates on the resampled cells whose PCs clear the threshold and
    reading a Youden ROC cut off the QC-passing cells. The final choice is
    R's nested prioritisation of predicted-dead count, specificity and the
    false-negative balance.

    Returns
    -------
    The chosen ``cor_threshold``.
    """
    probs0 = np.where(labels == "live", 0.0, 1.0)
    taus = _cor_taus(embedding, labels)
    # cor.coef is recomputed per candidate in R but never changes (labels are
    # constant during the search), so computing once is behaviour-identical.
    expanded = taus**2

    ratio = cor_max / cor_min

    def evaluate(lo, hi):
        sequence = 2.0 ** np.linspace(np.log2(lo), np.log2(hi), cor_steps)
        rows = []
        for candidate in sequence:
            selected = np.flatnonzero(expanded >= candidate)
            if selected.size < 2:  # R:246 requires at least 2 correlated dims
                continue
            X = embedding[:, selected]
            columns = [
                _train_rep(X, labels, qc, probs0, fail_weight, nfolds, min_dead, max_live, rng, model_seed + rep)
                for rep in range(nrep_cor)
            ]
            if any(c is None for c in columns):
                continue
            prob = np.median(np.column_stack(columns), axis=1)
            passing = qc == "pass"
            cut, specificity = roc_best_threshold(labels[passing], prob[passing])
            prediction = np.where(prob > cut, "dead", "live")
            n_pred_dead = int(np.sum(prediction == "dead"))
            missed_dead = int(np.sum((prediction == "live") & (labels == "dead")))
            live_live = int(np.sum((prediction == "live") & (labels == "live")))
            n_live = int(np.sum(labels == "live"))
            rows.append([candidate, specificity, n_pred_dead, missed_dead, live_live, n_live])
        return rows

    rows = evaluate(cor_min, cor_max)
    retry = 0
    # R:328-331 -- shrink the range by `ratio` while no threshold yields
    # (predicted dead > 0 & specificity >= 0.99)
    while not any(r[2] > 0 and r[1] >= 0.99 for r in rows) and retry < feature_try:
        retry += 1
        cor_max, cor_min = cor_min, cor_min / ratio
        rows = evaluate(cor_min, cor_max)
    if not rows:
        return float(np.sqrt(cor_min * cor_max))

    def pick(rows_):
        sub = [r for r in rows_ if r[2] > 0]  # stats[,3] > 0
        if not sub:
            return rows_[int(np.argmax([r[1] for r in rows_]))][0]
        sub_spec = [r for r in sub if r[1] >= 0.99]
        if not sub_spec:
            return sub[int(np.argmax([r[1] for r in sub]))][0]
        sub_bal = [r for r in sub_spec if r[4] / r[5] <= 0.5]  # live_live / n_live
        if not sub_bal:
            return sub_spec[int(np.argmax([r[4] / r[5] for r in sub_spec]))][0]
        sub_ratio = [r for r in sub_bal if r[2] / r[5] >= 2]  # n_pred_dead / n_live
        if not sub_ratio:
            return sub_bal[int(np.argmax([r[2] / r[5] for r in sub_bal]))][0]
        # which.min(missed_dead)
        return sub_ratio[int(np.argmin([r[3] for r in sub_ratio]))][0]

    return float(pick(rows))


def _one_run(embedding, labels, qc, fail_weight, cor_threshold, epochs, nrep, nfolds, min_dead, max_live, seed):
    """One independent optimisation run. Ports the ``bplapply`` body, ``label_dead.R:200-448``.

    Notes
    -----
    ``label_dead.R:225-236`` builds a score- and QC-derived weight vector, but
    every replicate then overwrites ``weights`` wholesale at ``R:276-277`` with
    ``rep(1, ...)`` and ``fail_weight`` for the QC-failing block. The initial
    vector is therefore dead code in R, so it is not computed here -- only the
    per-replicate weights that actually reach the model.
    """
    rng = np.random.default_rng(seed)
    labels = labels.copy()
    probs = np.where(labels == "live", 0.0, 1.0)

    if cor_threshold is None:
        cor_threshold = _search_cor_threshold(
            embedding,
            labels,
            qc,
            rng,
            fail_weight=fail_weight,
            nfolds=nfolds,
            min_dead=min_dead,
            max_live=max_live,
            model_seed=seed,
        )

    spec_old = 0.0
    balance_old = 0.0
    relabel_old = labels.size
    trigger = False

    for _ in range(epochs):
        taus = _cor_taus(embedding, labels)
        if trigger:
            cor_threshold += DEAD_COR_MIN  # escalation, label_dead.R:373
        selected = np.flatnonzero(taus**2 >= cor_threshold)
        if selected.size == 0:
            break
        X = embedding[:, selected]

        columns = [
            _train_rep(X, labels, qc, probs, fail_weight, nfolds, min_dead, max_live, rng, seed + rep)
            for rep in range(nrep)
        ]
        if any(c is None for c in columns):
            return labels, "Failed"
        new_prob = np.median(np.column_stack(columns), axis=1)
        passing = qc == "pass"
        cut, specificity = roc_best_threshold(labels[passing], new_prob[passing])
        prediction = np.where(new_prob > cut, "dead", "live")

        disagree = prediction != labels
        relabel = int(disagree.sum())
        balance = np.sum((prediction == "dead") & (labels == "live")) / relabel if relabel else 0.0

        if relabel <= labels.size * 0.002 or relabel >= relabel_old * 2:
            break
        if (spec_old > specificity or balance_old >= balance * 1.5) and not trigger:
            trigger = True
        else:
            balance_old = balance
            spec_old = specificity
            probs = new_prob
            labels = prediction
            if min(np.sum(labels == "dead"), np.sum(labels == "live")) == 0:
                return labels, "Failed"
        relabel_old = relabel

    return labels, "Success"


def train_labels(
    adata,
    mask,
    score,
    labels,
    qc,
    threshold,
    flag,
    *,
    rep=DEAD_CELL_RUNS,
    n_min=DEAD_CELL_CONSENSUS,
    npcs=DEAD_NPCS,
    epochs=DEAD_EPOCHS,
    nrep=DEAD_NREP,
    nfolds=DEAD_NFOLDS,
    fail_weight=DEAD_FAIL_WEIGHT,
    cor_threshold=None,
    cor_min=DEAD_COR_MIN,
    cor_max=DEAD_COR_MAX,
    min_dead=DEAD_MIN_DEAD,
    max_live=DEAD_MAX_LIVE,
    n_jobs=1,
    random_state=0,
):
    """Run ``rep`` independent optimisations and take the consensus.

    ``score`` and ``threshold`` are accepted so the call site mirrors R's, but
    are unused: see the note in :func:`_one_run` about R's dead weight vector.
    ``cor_threshold=None`` triggers the full search (``label_dead.R:239-352``)
    inside each run; a number skips the search, as R does.

    Returns
    -------
    ``(labels, flag)``.
    """
    barcodes = adata.obs_names[mask]
    embedding = _embed(adata, barcodes, npcs, random_state)

    # label_dead.R:181-191 -- one ridge pass on ALL PCs before training, which
    # relabels QC-passing soft-dead cells the model is confident are live
    # (P(dead) <= the lowest threshold with specificity >= 0.99). The modified
    # labels seed every run of the consensus.
    binary = (labels == "dead").astype(int)
    initial = logistic_ridge_1se(
        embedding,
        binary,
        nfolds=nfolds,
        n_alphas=NB_ALPHAS,
        random_state=random_state,
    )
    initial_prob = initial.predict_proba(embedding)[:, 1]
    passing = qc == "pass"
    fpr, tpr, thresholds = roc_curve(binary[passing], initial_prob[passing])
    spec = 1.0 - fpr
    enough = np.flatnonzero(spec >= 0.99)
    if enough.size:
        # lowest threshold still meeting 99% specificity; relabel only fires
        # when its sensitivity is > 0, exactly like R's coords lookup
        at = enough[np.argmin(thresholds[enough])]
        threshold99 = thresholds[at]
        if tpr[at] > 0:
            relabel = np.flatnonzero(passing & (labels == "dead") & (initial_prob <= threshold99))
            if relabel.size:
                logger.info("Initial ridge pass relabelled %d soft-dead cells to live", relabel.size)
                labels[relabel] = "live"

    results = Parallel(n_jobs=n_jobs)(
        delayed(_one_run)(
            embedding,
            labels,
            qc,
            fail_weight,
            cor_threshold,
            epochs,
            nrep,
            nfolds,
            min_dead,
            max_live,
            random_state + run,
        )
        for run in range(rep)
    )

    runs = np.column_stack([r[0] for r in results])
    flags = [r[1] for r in results]
    final = consensus(runs, n_min=n_min)

    # label_dead.R:462-478 -- merge run flags (Failed > Caution > Success),
    # then apply the uncertain-fraction escalation
    if "Failed" in flags:
        flag = "Failed"
    elif "Caution" in flags:
        flag = "Caution"
    passing = qc == "pass"
    uncertain_fraction = float(np.mean(final[passing] == "uncertain")) if passing.any() else 0.0
    flag = escalate_flag(flag, uncertain_fraction)

    logger.info(
        "Step 6: %d dead, %d uncertain (flag=%s)", int(np.sum(final == "dead")), int(np.sum(final == "uncertain")), flag
    )
    return final, flag
