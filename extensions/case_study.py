#!/usr/bin/env python
"""Full case-study orchestrator: paper Section 3.4 (LIHC) and the LUAD case
study mentioned in the Discussion (Fig S8).

The paper runs the SAME battery of downstream analyses for a cancer:
    1. key-IFM table with DE stats + XGBoost importances   (paper Table 2)
    2. PCoA + PERMANOVA + t-SNE + ARI clustering           (Fig 2D/2E, 3D)
    3. methylation-vs-expression analysis                  (Fig S7)
    4. GO/KEGG + STRING + ENCORI enrichment                (Fig 3F/3G) [internet]

This script chains those extension modules for ONE cancer, given the outputs
of run_pipeline.py. Works identically for LIHC, LUAD (paper Fig S8) or any
other cancer -- just point --results at that cancer's pipeline output.

Example (after run_pipeline.py finished for LUAD):
    python -m extensions.case_study \
        --input_h5ad_path data/processed/LUAD.h5ad \
        --results results/LUAD \
        --out results/LUAD/case_study
Use --skip_enrichment when offline.
"""

import argparse
import os
import subprocess
import sys

import pandas as pd

from gdnet.utils import ensure_dir


def run(mod, *cli):
    """Run an extension module as a subprocess so one failure doesn't kill all."""
    cmd = [sys.executable, "-m", mod, *cli]
    print(f"\n>>> {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"WARNING: {mod} exited with {r.returncode} -- continuing")


def key_ifm_table(results_dir, out_dir):
    """Paper Table 2: key genes with |logFC|, AveExp, adjusted p, XGB importance."""
    xgb = pd.read_csv(os.path.join(results_dir, "xgboost_top200.csv"))
    de = pd.read_csv(os.path.join(results_dir, "differential_features.csv"))
    key = xgb.merge(de, on="feature", suffixes=("_xgb", "_de"))
    if key.empty:
        print("key-IFM table: intersection is empty (see key_ifms.txt) -- "
              "writing the top XGBoost features with their DE stats instead")
        key = xgb.merge(de, on="feature", how="left")
    cols = {"feature": "feature", "log2FC": "logFC", "p_adj": "adj_p",
            "importance": "xgb_importance"}
    key = key.rename(columns=cols)[[c for c in
        ("feature", "logFC", "adj_p", "xgb_importance") if c in key.rename(columns=cols)]]
    key.to_csv(os.path.join(out_dir, "key_ifm_table.csv"), index=False)
    print(f"key-IFM table ({len(key)} rows) -> key_ifm_table.csv (paper Table 2)")


def main():
    ap = argparse.ArgumentParser(description="full case study (paper Sec 3.4 / Fig S8)")
    ap.add_argument("--input_h5ad_path", required=True)
    ap.add_argument("--results", required=True,
                    help="run_pipeline.py output dir of this cancer")
    ap.add_argument("--out", default=None,
                    help="default: <results>/case_study")
    ap.add_argument("--skip_enrichment", action="store_true",
                    help="skip the web-API enrichment steps (offline mode)")
    args = ap.parse_args()

    out = args.out or os.path.join(args.results, "case_study")
    ensure_dir(out)
    rg = os.path.join(args.results, "risk_groups.csv")
    xgb = os.path.join(args.results, "xgboost_top200.csv")
    key = os.path.join(args.results, "key_ifms.txt")
    ifms = os.path.join(args.results, "ifms.txt")

    # 1. Table-2-style key-IFM table
    key_ifm_table(args.results, out)

    # 2. ordination on the XGBoost features (Fig 2D) ------------------------
    run("extensions.ordination",
        "--input_h5ad_path", args.input_h5ad_path,
        "--risk_groups", rg, "--features", xgb,
        "--out", os.path.join(out, "ordination"))

    # 3. methylation vs expression (Fig S7) ---------------------------------
    run("extensions.methylation_expression",
        "--input_h5ad_path", args.input_h5ad_path,
        "--risk_groups", rg, "--out", os.path.join(out, "meth_expr"))

    # 4. enrichment + networks (Fig 3F/3G) -- needs internet ---------------
    if not args.skip_enrichment:
        genes_file = key if os.path.getsize(key) > 0 else ifms
        run("extensions.enrichment", "--genes", genes_file,
            "--out", os.path.join(out, "enrichment"))
    else:
        print("enrichment skipped (--skip_enrichment)")

    print(f"\ncase study complete -> {out}/")


if __name__ == "__main__":
    main()
