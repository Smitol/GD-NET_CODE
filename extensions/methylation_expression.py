#!/usr/bin/env python
"""Methylation-vs-expression analysis: paper Figure S7 (Section 3.4).

(NOTE: this analysis was NOT in the user's missing list -- it is in the paper:
 "most genes exhibited lower methylation levels in the high-risk group ...
  an inverse relationship between DNA methylation and gene expression in
  liver cancer prognosis", e.g. ECT2, SFN, S100P.)

For every gene that has BOTH a '_geneExp' and a '_methylation' feature in
the fused matrix, compares the high- vs low-risk group means of both
modalities and reports genes with the paper's inverse pattern
(methylation DOWN + expression UP in high risk, or vice versa).

Example:
    python -m extensions.methylation_expression \
        --input_h5ad_path data/processed/LIHC.h5ad \
        --risk_groups results/LIHC/risk_groups.csv \
        --out results/LIHC/meth_expr
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from gdnet.utils import ensure_dir, load_h5ad


def main():
    ap = argparse.ArgumentParser(description="methylation vs expression (Fig S7)")
    ap.add_argument("--input_h5ad_path", required=True)
    ap.add_argument("--risk_groups", required=True)
    ap.add_argument("--p_threshold", type=float, default=0.05,
                    help="expression-difference significance cut (paper: 0.05)")
    ap.add_argument("--out", default="results/meth_expr")
    args = ap.parse_args()

    ensure_dir(args.out)
    X, var_names, obs_names = load_h5ad(args.input_h5ad_path)
    rg = pd.read_csv(args.risk_groups).set_index("sample").reindex(obs_names)
    hi = (rg["group"] == "high").to_numpy()

    idx = {v: i for i, v in enumerate(var_names)}
    genes = sorted({v[:-len("_geneExp")] for v in var_names if v.endswith("_geneExp")}
                   & {v[:-len("_methylation")] for v in var_names
                      if v.endswith("_methylation")})
    print(f"{len(genes)} genes have both expression and methylation features")

    rows = []
    for g in genes:
        e = X[:, idx[f"{g}_geneExp"]]
        m = X[:, idx[f"{g}_methylation"]]
        # expression difference between risk groups (paper: p < 0.05)
        _, p_e = stats.ttest_ind(e[hi], e[~hi], equal_var=False)
        d_expr = e[hi].mean() - e[~hi].mean()
        d_meth = m[hi].mean() - m[~hi].mean()
        rows.append({"gene": g, "delta_expression": d_expr, "expr_p": p_e,
                     "delta_methylation": d_meth,
                     "inverse": (d_expr * d_meth) < 0})
    df = pd.DataFrame(rows)
    sig = df[df.expr_p < args.p_threshold].copy()
    n_inv = int(sig.inverse.sum())
    n_lowmeth = int((sig.delta_methylation < 0).sum())
    print(f"{len(sig)} genes with significant expression difference; "
          f"{n_inv} show the inverse methylation-expression pattern; "
          f"{n_lowmeth} are hypo-methylated in the high-risk group "
          f"(paper: 'most genes exhibited lower methylation in high-risk')")
    df.to_csv(os.path.join(args.out, "meth_expr_all.csv"), index=False)
    sig.sort_values("expr_p").to_csv(
        os.path.join(args.out, "meth_expr_significant.csv"), index=False)

    try:  # Fig S7-style scatter: delta methylation vs delta expression
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(df.delta_methylation, df.delta_expression, s=6, c="grey",
                   alpha=0.35, label="all shared genes")
        ax.scatter(sig.delta_methylation, sig.delta_expression, s=10,
                   c=np.where(sig.inverse, "#d62728", "#4c72b0"), alpha=0.8)
        for _, r in sig.reindex(sig.expr_p.nsmallest(8).index).iterrows():
            ax.annotate(r.gene, (r.delta_methylation, r.delta_expression),
                        fontsize=7)
        ax.axhline(0, lw=0.6, c="k")
        ax.axvline(0, lw=0.6, c="k")
        ax.set_xlabel("Δ methylation (high − low risk)")
        ax.set_ylabel("Δ expression (high − low risk)")
        ax.set_title("Methylation vs expression between risk groups (Fig S7)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "meth_expr_scatter.png"), dpi=150)
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
