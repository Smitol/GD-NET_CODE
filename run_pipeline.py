#!/usr/bin/env python
"""GD-Net Stage 2: full downstream pipeline for ONE cancer dataset.

Takes the same inputs as main.py (fused h5ad + KEGG edges + survival) and runs
EVERYTHING end-to-end:

  1. train the contrastive GCN encoder      (paper Fig. 1B, via main.py functions)
  2. Cox-Elastic-Net risk model, 5-fold CV  (paper Table 1: C-index +/- std)
  3. median-split high/low risk groups      (paper Fig. 1D)
  4. XGBoost top-200 feature importance     (paper Fig. 1C, Section 2.1.5)
  5. differential analysis between groups   (paper Section 2.3)
  6. IFMs / key-IFMs feature sets           (paper Section 2.3, Table 2)
  7. Kaplan-Meier plot of the risk groups   (paper Fig. 3E)

Outputs land in <out>/:
    embeddings.csv, cv_results.csv, risk_groups.csv,
    xgboost_top200.csv, differential_features.csv,
    ifms.txt, key_ifms.txt, km_curve.png

Example (bundled demo data, ~2 min on CPU):
    python run_pipeline.py \
        --input_h5ad_path data/paad_demo/paad.h5ad \
        --input_edge_path data/paad_demo/paad_edges.csv \
        --surv_path       data/paad_demo/paad_surv.csv \
        --epochs 100 --out results/paad_demo

For your own data (after preprocessing, see README):
    python run_pipeline.py \
        --input_h5ad_path data/processed/LIHC.h5ad \
        --input_edge_path data/processed/LIHC_edges.csv \
        --surv_path       data/processed/LIHC_surv.csv \
        --out results/LIHC
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

# reuse the training code from main.py
from main import extract_embeddings, train_encoder
from gdnet.cox_en import cross_validate, predict_risk_groups
from gdnet.feature_selection import (differential_analysis, informative_features,
                                     xgboost_importance)
from gdnet.utils import (ensure_dir, load_edges, load_h5ad, load_survival,
                         seed_everything)


def get_args():
    p = argparse.ArgumentParser(description="GD-Net end-to-end pipeline")
    p.add_argument("--input_h5ad_path", required=True)
    p.add_argument("--input_edge_path", required=True)
    p.add_argument("--surv_path", required=True,
                   help="survival csv (transposed authors' format or tidy sample,time,status)")
    p.add_argument("--out", default="results/run")
    # training hyper-parameters (see main.py for docs)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("-b", "--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--wd", type=float, default=1e-6)
    p.add_argument("--cos", action="store_true", default=True)
    p.add_argument("--low_dim", type=int, default=200)
    p.add_argument("--moco_r", type=int, default=512)
    p.add_argument("--moco_m", type=float, default=0.999)
    p.add_argument("--temperature", type=float, default=0.2)
    # Cox-EN hyper-parameters -- TUNE if CI is low on your cancer
    p.add_argument("--penalizer", type=float, default=0.1)
    p.add_argument("--l1_ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--skip_train", action="store_true",
                   help="reuse <out>/embeddings.csv from a previous run "
                        "(handy when only tuning the downstream steps)")
    return p.parse_args()


def main():
    args = get_args()
    seed_everything(args.seed)
    ensure_dir(args.out)
    device = f"cuda:{args.gpu}" if (args.gpu is not None and torch.cuda.is_available()) else "cpu"

    # ------------------------------------------------------------------ load
    X_raw, var_names, obs_names = load_h5ad(args.input_h5ad_path)
    surv = load_survival(args.surv_path)

    # align survival with the sample order of the matrix
    missing = [s for s in obs_names if s not in surv.index]
    if missing:
        print(f"WARNING: {len(missing)} samples have no survival info -> dropped "
              f"(e.g. {missing[:3]})")
        keep = [i for i, s in enumerate(obs_names) if s in surv.index]
        X_raw = X_raw[keep]
        obs_names = [obs_names[i] for i in keep]
    surv = surv.loc[obs_names]
    time_arr = surv["time"].to_numpy(float)
    status_arr = surv["status"].to_numpy(float)
    print(f"{X_raw.shape[0]} patients x {X_raw.shape[1]} features "
          f"({int(status_arr.sum())} events)")

    # ------------------------------------------------- 1. encoder + embeddings
    emb_path = os.path.join(args.out, "embeddings.csv")
    if args.skip_train and os.path.exists(emb_path):
        print("reusing existing embeddings")
        emb = pd.read_csv(emb_path, index_col=0).loc[obs_names].to_numpy(np.float32)
    else:
        edge_index = load_edges(args.input_edge_path)
        # z-score features (see main.py for why)
        mu, sd = X_raw.mean(0, keepdims=True), X_raw.std(0, keepdims=True)
        Xz = (X_raw - mu) / np.maximum(sd, 1e-8)
        model = train_encoder(Xz, edge_index, args, device)
        emb = extract_embeddings(model, Xz, device)
        pd.DataFrame(emb, index=obs_names,
                     columns=[f"z{i}" for i in range(emb.shape[1])]).to_csv(emb_path)

    # --------------------------------------------------- 2. Cox-EN, 5-fold CV
    print("\n--- Cox-EN 5-fold cross-validation (paper Table 1) ---")
    cv = cross_validate(emb, time_arr, status_arr, n_splits=5,
                        penalizer=args.penalizer, l1_ratio=args.l1_ratio,
                        seed=args.seed)
    cv.to_csv(os.path.join(args.out, "cv_results.csv"))

    # ------------------------------------------------------ 3. risk subgroups
    risk, high_risk = predict_risk_groups(emb, time_arr, status_arr,
                                          args.penalizer, args.l1_ratio)
    pd.DataFrame({"sample": obs_names, "risk_score": risk,
                  "group": np.where(high_risk, "high", "low"),
                  "time": time_arr, "status": status_arr}) \
        .to_csv(os.path.join(args.out, "risk_groups.csv"), index=False)
    print(f"risk groups: {int(high_risk.sum())} high / {int((~high_risk).sum())} low")

    # --------------------------------------- 4. XGBoost feature importance
    print("\n--- XGBoost informative-feature selection (paper Sec 2.1.5) ---")
    f_xgb = xgboost_importance(X_raw, var_names, high_risk, top_n=200, seed=args.seed)
    f_xgb.to_csv(os.path.join(args.out, "xgboost_top200.csv"), index=False)

    # --------------------------------------------- 5. differential analysis
    print("\n--- differential analysis between risk groups (paper Sec 2.3) ---")
    f_de = differential_analysis(X_raw, var_names, high_risk)
    f_de.to_csv(os.path.join(args.out, "differential_features.csv"), index=False)
    print(f"{len(f_de)} differential features (|log2FC|>1.6, adj-p<0.05)")

    # ------------------------------------------------- 6. IFMs and key-IFMs
    sets = informative_features(f_xgb, f_de)
    for name in ("ifms", "key_ifms"):
        with open(os.path.join(args.out, f"{name}.txt"), "w") as fh:
            fh.write("\n".join(sets[name]))

    # --------------------------------------------------- 7. KM curve (Fig 3E)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test

        fig, ax = plt.subplots(figsize=(6, 5))
        for mask, label, color in [(high_risk, "high risk", "#d62728"),
                                   (~high_risk, "low risk", "#1f77b4")]:
            KaplanMeierFitter().fit(time_arr[mask], status_arr[mask], label=label) \
                .plot_survival_function(ax=ax, color=color)
        p = logrank_test(time_arr[high_risk], time_arr[~high_risk],
                         event_observed_A=status_arr[high_risk],
                         event_observed_B=status_arr[~high_risk]).p_value
        ax.set_title(f"GD-Net risk groups (log-rank p = {p:.2e})")
        ax.set_xlabel("time (days)")
        ax.set_ylabel("survival probability")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "km_curve.png"), dpi=150)
        print(f"KM curve saved (log-rank p = {p:.2e})")
    except Exception as e:  # plotting must never kill the pipeline
        print(f"KM plot skipped: {e}", file=sys.stderr)

    print(f"\nAll outputs in {args.out}/")


if __name__ == "__main__":
    main()
