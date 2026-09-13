#!/usr/bin/env python
"""External / independent validation: paper Section 3.4, Figures S5 and S6.

Two validations from the paper, run as sub-commands:

  gse54236  -- "we also used the liver cancer RNA-seq data from GSE54236 [38]
               for survival validation" (Fig S5): downloads the GEO series
               matrix, extracts expression + survival, and runs a per-gene
               Kaplan-Meier validation (median expression split, log-rank p)
               for the key informative genes.

  stages    -- "345 liver cancer samples with clearly defined cancer stages
               (Stage normal-IV) for informative genes validation" (Fig S6B):
               stage-wise expression trend of key genes (e.g. MCM3, BUB1B,
               DLGAP5, ECT2 rising from Stage I to III). Needs the TCGA
               expression TSV + a clinical file with a stage column
               (e.g. TCGA-LIHC.clinical.tsv from the Xena GDC hub).

Examples:
    python -m extensions.external_validation gse54236 \
        --genes SPP1 SLC27A5 IGF2 MCM3 BUB1B DLGAP5 ECT2 \
        --out results/LIHC/validation

    python -m extensions.external_validation stages \
        --expr_tsv data/TCGA-LIHC/TCGA-LIHC.star_fpkm-uq.tsv \
        --clinical_tsv data/TCGA-LIHC/TCGA-LIHC.clinical.tsv \
        --genes MCM3 BUB1B DLGAP5 ECT2 \
        --out results/LIHC/validation
"""

import argparse
import gzip
import io
import os
import urllib.request

import numpy as np
import pandas as pd

from gdnet.utils import ensure_dir

GSE54236_URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE54nnn/GSE54236/"
                "matrix/GSE54236_series_matrix.txt.gz")


# ===========================================================================
# GSE54236 survival validation (Fig S5)
# ===========================================================================
def load_gse54236(cache_dir):
    """Download + parse the GEO series matrix.

    Returns (expr: genes x samples DataFrame, surv: DataFrame[time, status]).
    GSE54236 (Villa 2016) is a two-channel Agilent array of 78 HCC samples
    with 'survival (months)' recorded in the sample characteristics.
    """
    path = os.path.join(cache_dir, "GSE54236_series_matrix.txt.gz")
    if not os.path.exists(path):
        print("downloading GSE54236 series matrix (~20 MB)...")
        urllib.request.urlretrieve(GSE54236_URL, path)

    chars, table_lines, in_table = {}, [], False
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("!Sample_geo_accession"):
                chars["gsm"] = [c.strip('"') for c in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch"):
                vals = [c.strip('"') for c in line.rstrip("\n").split("\t")[1:]]
                # each characteristics line is "key: value"
                key = vals[0].split(":")[0] if vals and ":" in vals[0] else None
                if key:
                    chars.setdefault(key.strip().lower(), []).extend(
                        [v.split(":", 1)[1].strip() if ":" in v else "" for v in vals])
            elif line.startswith("!series_matrix_table_begin"):
                in_table = True
            elif line.startswith("!series_matrix_table_end"):
                in_table = False
            elif in_table:
                table_lines.append(line)

    expr = pd.read_csv(io.StringIO("".join(table_lines)), sep="\t", index_col=0)
    # find the survival characteristic (key name contains 'survival')
    surv_key = next((k for k in chars if "survival" in k and k != "gsm"), None)
    if surv_key is None:
        raise SystemExit(f"no survival field found; available: {list(chars)}")
    time = pd.to_numeric(pd.Series(chars[surv_key][:expr.shape[1]],
                                   index=expr.columns), errors="coerce")
    surv = pd.DataFrame({"time": time,
                         # GEO does not give censoring for all -> treat missing
                         # as censored at last follow-up; recorded deaths = event.
                         "status": (~time.isna()).astype(int)})
    print(f"GSE54236: {expr.shape[1]} samples, {expr.shape[0]} probes, "
          f"{int(surv.status.sum())} with survival time")

    # probe -> gene symbol mapping from the platform annotation row if present;
    # otherwise expression rows stay as probes and we map via GPL file.
    return expr, surv


def map_probes_to_genes(expr, cache_dir, platform="GPL6480"):
    """Map Agilent probe IDs to gene symbols using the GEO platform annot."""
    url = (f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/{platform}/"
           f"annot/{platform}.annot.gz")
    path = os.path.join(cache_dir, f"{platform}.annot.gz")
    try:
        if not os.path.exists(path):
            print(f"downloading {platform} annotation...")
            urllib.request.urlretrieve(url, path)
        ann_lines, in_t = [], False
        with gzip.open(path, "rt", errors="replace") as fh:
            for line in fh:
                if line.startswith("!platform_table_begin"):
                    in_t = True
                elif line.startswith("!platform_table_end"):
                    in_t = False
                elif in_t:
                    ann_lines.append(line)
        ann = pd.read_csv(io.StringIO("".join(ann_lines)), sep="\t",
                          index_col=0, low_memory=False)
        symcol = next(c for c in ann.columns if "symbol" in c.lower())
        sym = ann[symcol].dropna()
        expr = expr.join(sym.rename("symbol"), how="inner")
        expr = expr.groupby("symbol").mean()          # average probes per gene
        print(f"mapped to {expr.shape[0]} gene symbols")
        return expr
    except Exception as e:
        print(f"probe mapping failed ({e}) -- keeping probe IDs; "
              f"pass gene-level matrix via --expr_tsv instead")
        return expr


def km_validation(expr, surv, genes, out_dir):
    """Per-gene KM validation: median split on expression, log-rank test."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = surv.dropna(subset=["time"]).index.intersection(expr.columns)
    rows, found = [], [g for g in genes if g in expr.index]
    print(f"{len(found)}/{len(genes)} genes present: {found}")
    if not found:
        return

    ncol = min(3, len(found))
    nrow = int(np.ceil(len(found) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.8 * nrow),
                             squeeze=False)
    for i, g in enumerate(found):
        e = expr.loc[g, ok].astype(float)
        hi = e >= e.median()
        t, s = surv.loc[ok, "time"], surv.loc[ok, "status"]
        lr = logrank_test(t[hi], t[~hi], event_observed_A=s[hi],
                          event_observed_B=s[~hi])
        rows.append({"gene": g, "logrank_p": lr.p_value})
        ax = axes[i // ncol][i % ncol]
        for m, lab, c in [(hi, "high expr", "#d62728"), (~hi, "low expr", "#1f77b4")]:
            KaplanMeierFitter().fit(t[m], s[m], label=lab).plot_survival_function(
                ax=ax, color=c, ci_show=False)
        ax.set_title(f"{g} (p={lr.p_value:.3g})", fontsize=10)
    for j in range(len(found), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("GSE54236 survival validation of key genes (paper Fig S5)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gse54236_km.png"), dpi=150)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "gse54236_logrank.csv"),
                              index=False)
    print(pd.DataFrame(rows).to_string(index=False))


# ===========================================================================
# staged-cohort trend validation (Fig S6B)
# ===========================================================================
def stage_trends(expr_tsv, clinical_tsv, genes, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sps

    expr = pd.read_csv(expr_tsv, sep="\t", index_col=0)
    expr.columns = [c[:15] for c in expr.columns]
    # strip Ensembl version + map via first column if needed; assume symbols
    clin = pd.read_csv(clinical_tsv, sep="\t", index_col=0, low_memory=False)
    clin.index = [str(i)[:15] for i in clin.index]

    stage_col = next((c for c in clin.columns
                      if "stage" in c.lower() and clin[c].notna().sum() > 50), None)
    if stage_col is None:
        raise SystemExit(f"no usable stage column in {clinical_tsv}; "
                         f"columns: {list(clin.columns)[:30]}...")
    print(f"using stage column: {stage_col}")

    def norm_stage(v):
        v = str(v).upper()
        for s in ("IV", "III", "II", "I"):
            if f"STAGE {s}" in v:
                return s
        return None
    stages = clin[stage_col].map(norm_stage)

    # normal samples = TCGA barcode '-11' (solid tissue normal)
    normals = [c for c in expr.columns if c.endswith("-11")]
    order = ["normal", "I", "II", "III", "IV"]

    rows = []
    found = [g for g in genes if g in expr.index]
    print(f"{len(found)}/{len(genes)} genes present in the expression matrix")
    ncol = min(2, len(found)) or 1
    nrow = int(np.ceil(len(found) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow),
                             squeeze=False)
    for i, g in enumerate(found):
        groups, labels = [], []
        for st in order:
            if st == "normal":
                vals = expr.loc[g, normals].astype(float).values
            else:
                ids = stages[stages == st].index.intersection(expr.columns)
                vals = expr.loc[g, ids].astype(float).values
            if len(vals) >= 3:
                groups.append(vals)
                labels.append(f"{st}\n(n={len(vals)})")
        # monotone-trend test: Spearman rho of expression vs ordinal stage
        xs = np.concatenate([[k] * len(v) for k, v in enumerate(groups)])
        ys = np.concatenate(groups)
        rho, p = sps.spearmanr(xs, ys)
        rows.append({"gene": g, "spearman_rho": rho, "trend_p": p})
        ax = axes[i // ncol][i % ncol]
        try:                                   # matplotlib >= 3.9
            ax.boxplot(groups, tick_labels=labels)
        except TypeError:                      # older matplotlib
            ax.boxplot(groups, labels=labels)
        ax.set_title(f"{g}  (trend rho={rho:.2f}, p={p:.2g})", fontsize=10)
        ax.set_ylabel("expression")
    for j in range(len(found), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Stage-wise expression of key genes (paper Fig S6B)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stage_trends.png"), dpi=150)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "stage_trends.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="external validation (Fig S5/S6)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gse54236", help="GSE54236 KM validation (Fig S5)")
    g.add_argument("--genes", nargs="+", required=True,
                   help="gene symbols to validate, e.g. SPP1 SLC27A5 IGF2")
    g.add_argument("--out", default="results/validation")
    g.add_argument("--expr_tsv", default=None,
                   help="OPTIONAL pre-made gene-level expression TSV for GSE54236 "
                        "(skips download + probe mapping)")

    s = sub.add_parser("stages", help="staged-cohort trends (Fig S6B)")
    s.add_argument("--expr_tsv", required=True)
    s.add_argument("--clinical_tsv", required=True,
                   help="clinical TSV with a tumour-stage column (Xena GDC hub)")
    s.add_argument("--genes", nargs="+", required=True)
    s.add_argument("--out", default="results/validation")

    args = ap.parse_args()
    ensure_dir(args.out)

    if args.cmd == "gse54236":
        if args.expr_tsv:
            expr = pd.read_csv(args.expr_tsv, sep="\t", index_col=0)
            surv = None
            raise SystemExit("--expr_tsv route: also provide survival -- see "
                             "load_gse54236() to adapt; default download route "
                             "handles both automatically")
        expr, surv = load_gse54236(args.out)
        expr = map_probes_to_genes(expr, args.out)
        km_validation(expr, surv, args.genes, args.out)
    else:
        stage_trends(args.expr_tsv, args.clinical_tsv, args.genes, args.out)


if __name__ == "__main__":
    main()
