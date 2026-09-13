#!/usr/bin/env python
"""Real DESeq2 differential-expression analysis (paper Sections 2.3-2.4).

The paper: "The DESeq2 was used for differential expression analysis to
detect the differentially expressed molecules set F_DE" with thresholds
|log2FC| > 1.6 and adjusted p < 0.05 (LIHC: 579 DEGs, Fig 3A).

The core repo substitutes a Welch t-test (documented in its README). This
module provides the REAL DESeq2 route, two ways:

  1. PYTHON (this script): pydeseq2, a faithful Python port of DESeq2.
     Needs raw INTEGER COUNTS (not the log2 FPKM values used elsewhere!) --
     download e.g. TCGA-LIHC.star_counts.tsv from the same Xena GDC hub.
  2. R (exact paper setup): extensions/r_scripts/run_deseq2.R

Install:  pip install pydeseq2

Example:
    python -m extensions.deseq2_analysis \
        --counts_tsv data/TCGA-LIHC.star_counts.tsv \
        --risk_groups results/LIHC/risk_groups.csv \
        --out results/LIHC/deseq2

Outputs: deseq2_results.csv (all genes) + deseq2_degs.csv (paper thresholds)
+ volcano.png (Fig 3A style).
"""

import argparse
import os

import numpy as np
import pandas as pd

from gdnet.utils import ensure_dir


def load_counts(path: str) -> pd.DataFrame:
    """Load a Xena-style counts TSV (rows = genes, cols = samples).

    Xena 'star_counts' files are log2(count+1) -- we detect that (non-integer
    values) and back-transform to raw integers, which DESeq2 requires.
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    if not np.allclose(df.values[:200], np.round(df.values[:200])):
        print("counts look log2(x+1)-transformed -> back-transforming to integers")
        df = (2 ** df - 1).round().astype(int)
    return df


def main():
    ap = argparse.ArgumentParser(description="DESeq2 DE analysis via pydeseq2 (Sec 2.4)")
    ap.add_argument("--counts_tsv", required=True,
                    help="raw mRNA counts TSV (genes x samples), e.g. Xena star_counts")
    ap.add_argument("--risk_groups", required=True,
                    help="risk_groups.csv from run_pipeline.py")
    ap.add_argument("--lfc", type=float, default=1.6, help="|log2FC| threshold (paper: 1.6)")
    ap.add_argument("--padj", type=float, default=0.05, help="adjusted-p threshold (paper: 0.05)")
    ap.add_argument("--min_count", type=int, default=10,
                    help="drop genes with total counts below this (standard DESeq2 practice)")
    ap.add_argument("--out", default="results/deseq2")
    args = ap.parse_args()

    ensure_dir(args.out)

    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError:
        raise SystemExit(
            "pydeseq2 is not installed. Either:\n"
            "  pip install pydeseq2\n"
            "or use the exact-paper R route:\n"
            "  Rscript extensions/r_scripts/run_deseq2.R <counts.tsv> <risk_groups.csv> <outdir>")

    counts = load_counts(args.counts_tsv)
    rg = pd.read_csv(args.risk_groups).set_index("sample")

    # sample IDs: risk_groups uses 15-char TCGA barcodes; counts columns may be
    # longer -- truncate for matching (adjust if your IDs differ).
    counts.columns = [c[:15] for c in counts.columns]
    common = [s for s in rg.index if s in counts.columns]
    print(f"{len(common)} samples shared between counts and risk groups")
    if len(common) < 10:
        raise SystemExit("too few shared samples -- check that sample IDs match")

    cts = counts[common].T                                  # samples x genes
    cts = cts.loc[:, cts.sum(0) >= args.min_count]          # filter low counts
    meta = pd.DataFrame({"condition": rg.loc[common, "group"].values}, index=common)

    # ---------------- DESeq2: size factors, dispersion, Wald test ----------
    dds = DeseqDataSet(counts=cts, metadata=meta, design="~condition", quiet=True)
    dds.deseq2()
    # contrast high vs low risk (log2FC > 0 = up-regulated in HIGH risk)
    stat = DeseqStats(dds, contrast=["condition", "high", "low"], quiet=True)
    stat.summary()
    res = stat.results_df.rename_axis("feature").reset_index()
    res.to_csv(os.path.join(args.out, "deseq2_results.csv"), index=False)

    degs = res[(res.log2FoldChange.abs() > args.lfc) & (res.padj < args.padj)]
    degs = degs.sort_values("padj")
    degs.to_csv(os.path.join(args.out, "deseq2_degs.csv"), index=False)
    n_up = int((degs.log2FoldChange > 0).sum())
    print(f"{len(degs)} DEGs (paper LIHC: 579) -- {n_up} up / {len(degs) - n_up} down "
          f"in the high-risk group")

    try:  # volcano plot (Fig 3A style)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        p = -np.log10(res.padj.clip(lower=1e-300))
        sig = (res.log2FoldChange.abs() > args.lfc) & (res.padj < args.padj)
        up = sig & (res.log2FoldChange > 0)
        ax.scatter(res.log2FoldChange[~sig], p[~sig], s=4, c="grey", alpha=0.4)
        ax.scatter(res.log2FoldChange[sig & ~up], p[sig & ~up], s=6, c="#1f77b4",
                   label="down in high-risk")
        ax.scatter(res.log2FoldChange[up], p[up], s=6, c="#d62728",
                   label="up in high-risk")
        ax.axvline(args.lfc, ls="--", lw=0.6, c="k")
        ax.axvline(-args.lfc, ls="--", lw=0.6, c="k")
        ax.set_xlabel("log2 fold change (high vs low risk)")
        ax.set_ylabel("-log10 adjusted p")
        ax.set_title("DESeq2 volcano (paper Fig 3A)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "volcano.png"), dpi=150)
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
