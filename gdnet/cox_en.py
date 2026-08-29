"""Cox-Elastic-Net risk estimation + evaluation (paper Sections 2.1.4, 2.1.6, 2.2).

Pipeline position: the low-dimensional embeddings produced by the contrastive
GCN encoder ("multi-omics meta-features") are fed into an elastic-net
regularised Cox proportional-hazards model (Eq. 6-7). The linear predictor
beta^T x is the patient's RISK SCORE; splitting patients at the median risk
gives the high-/low-risk subgroups used later for feature selection.

Evaluation (paper Section 2.2):
  * Harrell's concordance index (C-index, Eq. 9)   -- higher = better ranking
  * |log10(p)| of the log-rank test between the median-split risk groups

Implementation notes:
  * We use `lifelines.CoxPHFitter(penalizer, l1_ratio)` which implements the
    elastic-net penalty  lambda * ( l1_ratio*|b|_1 + (1-l1_ratio)/2*|b|_2^2 ).
    The paper does not publish lambda / l1_ratio (they are in unavailable
    supplementary material); the defaults below (penalizer=0.1, l1_ratio=0.5)
    are a balanced choice.  <-- TUNE THESE if you want to squeeze out more CI.
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold


def fit_cox_en(embeddings: np.ndarray, time: np.ndarray, status: np.ndarray,
               penalizer: float = 0.1, l1_ratio: float = 0.5) -> CoxPHFitter:
    """Fit the Cox-EN model on (n_patients, n_dims) embeddings."""
    df = pd.DataFrame(embeddings, columns=[f"z{i}" for i in range(embeddings.shape[1])])
    df["time"] = time
    df["status"] = status
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(df, duration_col="time", event_col="status")
    return cph


def risk_scores(cph: CoxPHFitter, embeddings: np.ndarray) -> np.ndarray:
    """Linear predictor beta^T x  (log partial hazard) = the risk score."""
    df = pd.DataFrame(embeddings, columns=[f"z{i}" for i in range(embeddings.shape[1])])
    return cph.predict_partial_hazard(df).to_numpy().ravel()


def evaluate(risk: np.ndarray, time: np.ndarray, status: np.ndarray) -> dict:
    """C-index (Eq. 9) + log-rank p between median-split risk groups."""
    # concordance_index expects HIGHER prediction = LONGER survival,
    # so we negate the risk (higher risk = shorter survival).
    ci = concordance_index(time, -risk, status)

    high = risk >= np.median(risk)  # high-risk group
    lr = logrank_test(time[high], time[~high],
                      event_observed_A=status[high], event_observed_B=status[~high])
    p = max(lr.p_value, 1e-300)  # avoid log10(0)
    return {"c_index": ci, "logrank_p": lr.p_value, "abs_log10_p": abs(np.log10(p))}


def cross_validate(embeddings: np.ndarray, time: np.ndarray, status: np.ndarray,
                   n_splits: int = 5, penalizer: float = 0.1, l1_ratio: float = 0.5,
                   seed: int = 42) -> pd.DataFrame:
    """5-fold cross-validation exactly as in paper Section 2.1.6 / Table 1.

    For every fold: fit Cox-EN on 4/5 of the patients, score the held-out 1/5.
    Returns a DataFrame with one row per fold (c_index, |log10 p|).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(kf.split(embeddings)):
        cph = fit_cox_en(embeddings[tr], time[tr], status[tr],
                         penalizer=penalizer, l1_ratio=l1_ratio)
        r = risk_scores(cph, embeddings[te])
        m = evaluate(r, time[te], status[te])
        m["fold"] = fold
        rows.append(m)
    res = pd.DataFrame(rows).set_index("fold")
    print(f"5-fold CV  C-index = {res.c_index.mean():.3f} (+/- {res.c_index.std():.3f})   "
          f"|log10(p)| = {res.abs_log10_p.mean():.2f}")
    return res


def predict_risk_groups(embeddings: np.ndarray, time: np.ndarray, status: np.ndarray,
                        penalizer: float = 0.1, l1_ratio: float = 0.5):
    """Fit Cox-EN on ALL patients and return (risk_scores, high_risk_bool).

    These median-split HIGH/LOW risk labels are the prediction target of the
    XGBoost feature-selection module (paper Section 2.1.5) and the groups
    compared in the differential-expression analysis (Section 2.3).
    """
    cph = fit_cox_en(embeddings, time, status, penalizer, l1_ratio)
    risk = risk_scores(cph, embeddings)
    return risk, risk >= np.median(risk)
