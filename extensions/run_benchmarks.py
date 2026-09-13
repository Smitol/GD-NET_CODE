#!/usr/bin/env python
"""Benchmark comparison harness: paper Table 1, Figure 2A/2B and Figure S1.

Runs 5-fold CV of the six comparison methods (extensions/benchmarks.py) on
the SAME folds, computes per-fold C-index and AUC, adds the GD-Net column
(from embeddings produced by run_pipeline.py), and writes:

    <out>/benchmark_cindex.csv     per-fold C-index, all methods (-> Table 1)
    <out>/benchmark_auc.csv        per-fold AUC, all methods (-> Fig S1)
    <out>/benchmark_summary.csv    mean +/- std + paired t-test vs GD-Net
    <out>/benchmark_boxplot.png    Fig 2A-style boxplot
    <out>/roc_curves.png           Fig S1-style ROC curves (last fold)

AUC DEFINITION (the paper does not specify the exact estimator): we use the
standard binary formulation at the median follow-up horizon tau:
    positive = died before tau; negative = observed alive at tau;
    patients censored before tau are excluded (they are unlabelable).
This is the common "ROC at fixed horizon" used in survival benchmarking.

Example (after running run_pipeline.py so embeddings.csv exists):
    python -m extensions.run_benchmarks \
        --input_h5ad_path data/paad_demo/paad.h5ad \
        --surv_path       data/paad_demo/paad_surv.csv \
        --gdnet_embeddings results/paad_demo/embeddings.csv \
        --out results/paad_demo/benchmarks

Runtime note: 6 methods x 5 folds on 13k features takes ~10-30 min on CPU.
Use --methods to run a subset (comma-separated), e.g. --methods PCA-Cox,eXGBC.
"""

import argparse
import os

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold

from extensions.benchmarks import METHODS
from gdnet.cox_en import fit_cox_en, risk_scores
from gdnet.utils import ensure_dir, load_h5ad, load_survival, seed_everything


def horizon_labels(time, status, tau):
    """Binary event-by-tau labels; returns (labels, evaluable_mask)."""
    dead_by_tau = (time <= tau) & (status > 0)
    alive_at_tau = time > tau
    evaluable = dead_by_tau | alive_at_tau       # censored before tau -> excluded
    return dead_by_tau.astype(int), evaluable


def main():
    ap = argparse.ArgumentParser(description="GD-Net benchmark comparisons (Table 1)")
    ap.add_argument("--input_h5ad_path", required=True)
    ap.add_argument("--surv_path", required=True)
    ap.add_argument("--gdnet_embeddings", default=None,
                    help="embeddings.csv from run_pipeline.py; adds the GD-Net column")
    ap.add_argument("--out", default="results/benchmarks")
    ap.add_argument("--methods", default=None,
                    help="comma-separated subset, e.g. 'PCA-Cox,eXGBC' (default: all six)")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.out)

    # ---------------------------------------------------------------- load
    X, var_names, obs_names = load_h5ad(args.input_h5ad_path)
    surv = load_survival(args.surv_path)
    keep = [i for i, s in enumerate(obs_names) if s in surv.index]
    X, obs_names = X[keep], [obs_names[i] for i in keep]
    surv = surv.loc[obs_names]
    time, status = surv["time"].to_numpy(float), surv["status"].to_numpy(float)
    print(f"{X.shape[0]} patients x {X.shape[1]} features, {int(status.sum())} events")

    emb = None
    if args.gdnet_embeddings and os.path.exists(args.gdnet_embeddings):
        emb = (pd.read_csv(args.gdnet_embeddings, index_col=0)
               .reindex(obs_names).to_numpy(np.float32))
        print(f"GD-Net embeddings loaded: {emb.shape}")

    methods = dict(METHODS)
    if args.methods:
        wanted = [m.strip() for m in args.methods.split(",")]
        methods = {k: v for k, v in methods.items() if k in wanted}

    tau = float(np.median(time))                 # AUC horizon (see docstring)

    # -------------------------------------------------- identical CV folds
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    folds = list(kf.split(X))

    ci_rows, auc_rows, roc_store = [], [], {}
    all_names = list(methods) + (["GD-Net"] if emb is not None else [])

    for name in all_names:
        for fold, (tr, te) in enumerate(folds):
            if name == "GD-Net":                 # Cox-EN on contrastive embeddings
                cph = fit_cox_en(emb[tr], time[tr], status[tr])
                risk = risk_scores(cph, emb[te])
            else:
                risk = methods[name](X[tr], time[tr], status[tr], X[te],
                                     seed=args.seed)

            ci = concordance_index(time[te], -risk, status[te])
            ci_rows.append({"method": name, "fold": fold, "c_index": ci})

            y, ok = horizon_labels(time[te], status[te], tau)
            if ok.sum() > 5 and len(np.unique(y[ok])) == 2:
                auc = roc_auc_score(y[ok], risk[ok])
                auc_rows.append({"method": name, "fold": fold, "auc": auc})
                roc_store[name] = roc_curve(y[ok], risk[ok])   # keep last fold
            print(f"  {name:>10s}  fold {fold}  CI={ci:.3f}")

    ci_df = pd.DataFrame(ci_rows)
    auc_df = pd.DataFrame(auc_rows)
    ci_df.to_csv(os.path.join(args.out, "benchmark_cindex.csv"), index=False)
    auc_df.to_csv(os.path.join(args.out, "benchmark_auc.csv"), index=False)

    # ------------------------------- summary + t-test vs GD-Net (Table 1 row)
    rows = []
    gd = ci_df[ci_df.method == "GD-Net"].c_index.values if emb is not None else None
    for name in all_names:
        v = ci_df[ci_df.method == name].c_index.values
        a = auc_df[auc_df.method == name].auc.values
        row = {"method": name,
               "c_index_mean": v.mean(), "c_index_std": v.std(ddof=1),
               "auc_mean": a.mean() if len(a) else np.nan}
        if gd is not None and name != "GD-Net":
            row["t_test_p_vs_gdnet"] = stats.ttest_rel(v, gd).pvalue
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out, "benchmark_summary.csv"), index=False)
    print("\n", summary.to_string(index=False))

    # ------------------------------------------------------------- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Fig 2A style boxplot of per-fold C-index
        fig, ax = plt.subplots(figsize=(8, 5))
        data = [ci_df[ci_df.method == n].c_index.values for n in all_names]
        try:                                   # matplotlib >= 3.9
            ax.boxplot(data, tick_labels=all_names)
        except TypeError:                      # older matplotlib
            ax.boxplot(data, labels=all_names)
        ax.set_ylabel("C-index (5-fold CV)")
        ax.set_title("Benchmark comparison (paper Fig 2A)")
        plt.xticks(rotation=30)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "benchmark_boxplot.png"), dpi=150)

        # Fig S1 style ROC curves
        fig, ax = plt.subplots(figsize=(6, 6))
        for name, (fpr, tpr, _) in roc_store.items():
            a = auc_df[auc_df.method == name].auc.mean()
            ax.plot(fpr, tpr, label=f"{name} (AUC={a:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC at median follow-up ({tau:.0f} days) -- paper Fig S1")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "roc_curves.png"), dpi=150)
        print(f"figures saved to {args.out}/")
    except Exception as e:
        print(f"plotting skipped: {e}")


if __name__ == "__main__":
    main()
