"""Informative feature selection (paper Sections 2.1.5 and 2.3, Fig. 1C).

Two complementary feature sets are computed from the ORIGINAL (pre-embedding)
multimodal features, using the HIGH/LOW risk labels predicted by GD-Net:

  F_XGB : top-200 features by importance of an XGBoost classifier trained to
          predict the risk label (paper: "lightweight XGBoost module",
          tree depth searched in 2..8).
  F_DE  : differentially expressed molecules between the two risk groups with
          |log2(fold change)| > 1.6 and BH-adjusted p < 0.05.
          NOTE: the paper uses DESeq2 (R) on raw counts for the mRNA modality.
          Here we use a Welch t-test + BH-FDR on the (log-scale) matrix so the
          whole pipeline stays in Python. Thresholds are identical. If you
          want the exact paper setup, export the raw counts + risk labels and
          run DESeq2 in R (see README).

  IFMs      = F_XGB  UNION  F_DE   (global informative molecules)
  key-IFMs  = F_XGB  INTERSECT F_DE (key informative molecules -- the
              biomarker candidates, e.g. Table 2 of the paper)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GridSearchCV
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier

from .utils import modality_of


def xgboost_importance(X: np.ndarray, feature_names: list, high_risk: np.ndarray,
                       top_n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Train the interpretable XGBoost module and rank features by importance.

    Args:
        X:             (n_patients, n_features) ORIGINAL fused matrix.
        feature_names: names matching X's columns.
        high_risk:     boolean risk label per patient (from cox_en.predict_risk_groups).
        top_n:         paper keeps the top 200 features (Section 2.3).

    Returns DataFrame ['feature', 'importance', 'modality'] sorted by importance.
    """
    # Depth searched 2..8 like the paper (ref. Yang's study [15]).
    # 3-fold CV on the risk label; small n_estimators keeps it "lightweight".
    grid = GridSearchCV(
        XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=seed,
                      eval_metric="logloss", n_jobs=-1),
        param_grid={"max_depth": [2, 3, 4, 5, 6, 7, 8]},
        cv=3, scoring="roc_auc",
    )
    grid.fit(X, high_risk.astype(int))
    model = grid.best_estimator_
    print(f"XGBoost best depth = {grid.best_params_['max_depth']} "
          f"(CV AUC = {grid.best_score_:.3f})")

    imp = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
    imp["modality"] = imp["feature"].map(modality_of)
    return imp.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def differential_analysis(X: np.ndarray, feature_names: list, high_risk: np.ndarray,
                          lfc_threshold: float = 1.6, p_threshold: float = 0.05) -> pd.DataFrame:
    """Differential analysis between risk groups (Python stand-in for DESeq2).

    Selection criteria (identical to the paper, Section 2.3):
        |log2(fold change)| > 1.6  AND  BH-adjusted p < 0.05

    IMPORTANT: X is on log2 scale after preprocessing, so the fold change is
    computed on 2**X (i.e. back-transformed means). For methylation (beta
    values, not log) the "fold change" is a mean-ratio -- interpret with care.
    """
    hi, lo = X[high_risk], X[~high_risk]

    # Welch t-test per feature (vectorised)
    t, p = stats.ttest_ind(hi, lo, axis=0, equal_var=False)
    p = np.nan_to_num(p, nan=1.0)
    p_adj = multipletests(p, method="fdr_bh")[1]

    # log2 fold change of back-transformed group means
    mean_hi = np.log2(np.power(2.0, hi).mean(axis=0) + 1e-9)
    mean_lo = np.log2(np.power(2.0, lo).mean(axis=0) + 1e-9)
    lfc = mean_hi - mean_lo

    df = pd.DataFrame({"feature": feature_names, "log2FC": lfc,
                       "p_value": p, "p_adj": p_adj})
    df["modality"] = df["feature"].map(modality_of)
    de = df[(df.log2FC.abs() > lfc_threshold) & (df.p_adj < p_threshold)]
    return de.sort_values("p_adj").reset_index(drop=True)


def informative_features(f_xgb: pd.DataFrame, f_de: pd.DataFrame) -> dict:
    """Combine the two feature sets (paper Section 2.3).

    Returns dict with:
        'ifms'     : union  (global informative molecules)
        'key_ifms' : intersection (key informative molecules / biomarkers)
    """
    set_xgb, set_de = set(f_xgb.feature), set(f_de.feature)
    ifms = sorted(set_xgb | set_de)
    key = sorted(set_xgb & set_de)

    # per-modality counts, like the paper reports ("319 genes, 15 miRNAs, ...")
    def counts(feats):
        s = pd.Series([modality_of(f) for f in feats])
        return s.value_counts().to_dict() if len(s) else {}

    print(f"IFMs (union): {len(ifms)}  {counts(ifms)}")
    print(f"key-IFMs (intersection): {len(key)}  {counts(key)}")
    return {"ifms": ifms, "key_ifms": key}
