#!/usr/bin/env python
"""External / independent validation: paper Section 3.4, Figures S5 and S6.

Two validations from the paper, run as sub-commands:

  gse54236  -- Fig S5: "The prediction results of additional datasets
               GSE54236. The Kaplan-Meier survival curves show the clinical
               relevance of the two predicted groups. The statistical p-values
               were determined by the two-tailed log-rank sum test."

               GSE54236 is a single-platform mRNA microarray (Agilent GPL6480),
               so the multi-omics TCGA encoder cannot be applied to it
               directly. Instead, the GD-Net risk groups are transferred
               through the key informative genes:
                 1. on TCGA, an L2-regularised logistic regression learns the
                    GD-Net high/low-risk label (risk_groups.csv) from the
                    expression of the key genes (<gene>_geneExp features);
                 2. each cohort is z-scored per gene within itself (this removes
                    the RNA-seq vs microarray scale difference);
                 3. the model scores the GSE54236 tumours and they are split at
                    the median score into predicted high/low-risk groups;
                 4. KM curves + two-tailed log-rank test of the two groups.
               A per-gene KM analysis (median-expression split) is also
               written as a supplementary output.

  stages    -- "345 liver cancer samples with clearly defined cancer stages
               (Stage normal-IV) for informative genes validation" (Fig S6B):
               stage-wise expression trend of key genes (e.g. MCM3, BUB1B,
               DLGAP5, ECT2 rising from Stage I to III). Needs the TCGA
               expression TSV + a clinical file with a stage column
               (e.g. TCGA-LIHC.clinical.tsv from the Xena GDC hub).

SURVIVAL STATUS IN GSE54236 (read this before interpreting Fig S5):
  The GEO record gives only "survival time(months)" for the 81 tumour arrays
  (78 patients + 3 technical replicates). It has NO death/censoring
  indicator. The script therefore needs one of:
    --surv_tsv FILE        sample,time,status table with real vital status
                           (e.g. from the original study authors), OR
    --assume_all_events    treat every patient as deceased at the recorded
                           time. That is only valid if the cohort was
                           followed until death. This is an ASSUMPTION the
                           GEO data cannot confirm, and it is recorded in
                           the outputs.
  Without either flag the script stops rather than invent a status.

Examples:
    # predicted risk groups (Fig S5) from a finished LIHC run_pipeline.py
    python -m extensions.external_validation gse54236 \
        --tcga_h5ad data/processed/LIHC.h5ad \
        --risk_groups results/LIHC/risk_groups.csv \
        --genes_file results/LIHC/key_ifms.txt \
        --assume_all_events --out results/LIHC/validation

    # per-gene KM only (no TCGA model)
    python -m extensions.external_validation gse54236 \
        --genes SPP1 SLC27A5 IGF2 MCM3 BUB1B DLGAP5 ECT2 \
        --assume_all_events --out results/LIHC/validation

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
GENE_SUFFIX = "_geneExp"


# ===========================================================================
# GSE54236 loading
# ===========================================================================
def _split_row(line):
    return [c.strip().strip('"') for c in line.rstrip("\n").split("\t")[1:]]


def load_gse54236(cache_dir):
    """Download + parse the GEO series matrix.

    Returns (expr, meta):
        expr : probes x samples DataFrame (tumour, non-replicate samples only)
        meta : per-sample DataFrame with 'title', 'tissue', 'time' (months)

    Characteristics are parsed PER SAMPLE ("key: value" cells). GEO rows are
    not aligned by key: in GSE54236 the row that holds "survival time(months)"
    for tumours holds "scanner update" for non-tumour samples.
    """
    path = os.path.join(cache_dir, "GSE54236_series_matrix.txt.gz")
    if not os.path.exists(path):
        print("downloading GSE54236 series matrix (~15 MB)...")
        urllib.request.urlretrieve(GSE54236_URL, path)

    gsm, titles, char_rows, table_lines, in_table = None, None, [], [], False
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("!Sample_geo_accession"):
                gsm = _split_row(line)
            elif line.startswith("!Sample_title"):
                titles = _split_row(line)
            elif line.startswith("!Sample_characteristics_ch"):
                char_rows.append(_split_row(line))
            elif line.startswith("!series_matrix_table_begin"):
                in_table = True
            elif line.startswith("!series_matrix_table_end"):
                in_table = False
            elif in_table:
                table_lines.append(line)

    per_sample = []
    for j, g in enumerate(gsm):
        d = {"sample": g, "title": titles[j] if titles else ""}
        for row in char_rows:
            cell = row[j] if j < len(row) else ""
            if ":" in cell:
                k, v = cell.split(":", 1)
                d[k.strip().lower()] = v.strip()
        per_sample.append(d)
    meta = pd.DataFrame(per_sample).set_index("sample")

    surv_col = next((c for c in meta.columns if c.startswith("survival")), None)
    tissue_col = next((c for c in meta.columns if c.startswith("tissue type")), None)
    if surv_col is None:
        raise SystemExit(f"no survival field in GSE54236; fields: {list(meta.columns)}")
    meta["time"] = pd.to_numeric(meta[surv_col], errors="coerce")
    meta["tissue"] = meta[tissue_col] if tissue_col else ""

    is_tumour = meta["tissue"].str.contains("tumor", case=False) & \
        ~meta["tissue"].str.contains("non", case=False)
    is_rep = meta["title"].str.contains("_rep", case=False)
    keep = meta.index[is_tumour & ~is_rep & meta["time"].notna()]

    expr = pd.read_csv(io.StringIO("".join(table_lines)), sep="\t", index_col=0)
    expr = expr[[c for c in keep if c in expr.columns]]
    print(f"GSE54236: {int(is_tumour.sum())} tumour arrays, {int((is_tumour & is_rep).sum())} "
          f"technical replicates dropped -> {expr.shape[1]} patients, "
          f"{expr.shape[0]} probes")
    return expr, meta.loc[expr.columns, ["title", "tissue", "time"]]


def map_probes_to_genes(expr, cache_dir, platform="GPL6480"):
    """Map Agilent probe IDs to gene symbols using the GEO platform annot."""
    url = (f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/{platform}/"
           f"annot/{platform}.annot.gz")
    path = os.path.join(cache_dir, f"{platform}.annot.gz")
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
    sym = sym[~sym.str.contains("///")]           # drop multi-gene probes
    expr = expr.join(sym.rename("symbol"), how="inner")
    expr = expr.groupby("symbol").mean()          # average probes per gene
    print(f"mapped to {expr.shape[0]} gene symbols")
    return expr


def build_survival(meta, surv_tsv=None, assume_all_events=False):
    """Return (surv DataFrame[time, status], description of the status source)."""
    if surv_tsv:
        s = pd.read_csv(surv_tsv, sep=None, engine="python")
        s.columns = [c.lower() for c in s.columns]
        s = s.set_index(s.columns[0])[["time", "status"]].astype(float)
        s = s.reindex(meta.index).dropna()
        print(f"survival from {surv_tsv}: {len(s)} patients, "
              f"{int(s.status.sum())} events")
        return s, f"vital status from {os.path.basename(surv_tsv)}"
    if assume_all_events:
        s = pd.DataFrame({"time": meta["time"].astype(float), "status": 1.0},
                         index=meta.index)
        print("WARNING: GSE54236 has no death/censoring field -- ASSUMING every "
              f"patient died at the recorded time ({len(s)} events). Results "
              "are only valid if the cohort was followed until death.")
        return s, "ASSUMED: all patients deceased (GEO provides no vital status)"
    raise SystemExit(
        "GSE54236 provides survival time but NO death/censoring indicator.\n"
        "Pass --surv_tsv <sample,time,status> with real vital status, or\n"
        "--assume_all_events to explicitly treat every patient as deceased.")


def read_genes(genes, genes_file):
    """Gene symbols from --genes and/or --genes_file. Feature names such as
    'ECT2_geneExp' are reduced to 'ECT2'; methylation/miRNA features are
    skipped (GSE54236 is an mRNA array)."""
    out = list(genes or [])
    if genes_file:
        with open(genes_file) as fh:
            for l in fh:
                l = l.strip().split(",")[0]
                if not l or l == "feature":
                    continue
                if l.endswith(GENE_SUFFIX):
                    out.append(l[:-len(GENE_SUFFIX)])
                elif "_" not in l and not l.lower().startswith("hsa-"):
                    out.append(l)
    return list(dict.fromkeys(out))                # de-duplicate, keep order


def _zscore_rows(df):
    """z-score each gene (row) across the samples of ONE cohort."""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, 1), axis=0)


def _km_plot(ax, t, s, groups, title):
    from lifelines import KaplanMeierFitter
    for m, lab, c in [(groups, "high risk", "#d62728"), (~groups, "low risk", "#1f77b4")]:
        KaplanMeierFitter().fit(t[m], s[m], label=f"{lab} (n={int(m.sum())})") \
            .plot_survival_function(ax=ax, color=c, ci_show=False)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("time (months)")
    ax.set_ylabel("survival probability")


# ===========================================================================
# Fig S5: GD-Net predicted risk groups on GSE54236
# ===========================================================================
def predicted_group_validation(expr, surv, tcga_h5ad, risk_groups_csv, genes,
                               out_dir, status_note, seed=42):
    """Transfer the GD-Net TCGA risk groups to GSE54236 via the key genes."""
    from lifelines.statistics import logrank_test
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from gdnet.utils import load_h5ad
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, var_names, obs_names = load_h5ad(tcga_h5ad)
    rg = pd.read_csv(risk_groups_csv).set_index("sample")
    rows = [i for i, s in enumerate(obs_names) if s in rg.index]
    y = (rg.loc[[obs_names[i] for i in rows], "group"] == "high").to_numpy(int)

    col = {v[:-len(GENE_SUFFIX)]: j for j, v in enumerate(var_names)
           if v.endswith(GENE_SUFFIX)}
    shared = [g for g in genes if g in col and g in expr.index]
    missing = [g for g in genes if g not in shared]
    print(f"{len(shared)}/{len(genes)} key genes present in BOTH cohorts"
          + (f" (missing: {missing[:15]}{' ...' if len(missing) > 15 else ''})"
             if missing else ""))
    if len(shared) < 2:
        raise SystemExit("need >= 2 shared genes to build the transfer model")

    tcga = pd.DataFrame(X[np.ix_(rows, [col[g] for g in shared])].T, index=shared)
    # z-score within each cohort; missing array values (GEO nulls) are set to
    # 0 after z-scoring, i.e. imputed with that gene's cohort mean
    tcga_z = _zscore_rows(tcga).T.fillna(0.0).to_numpy()      # patients x genes
    gse = expr.loc[shared, surv.index].astype(float)
    n_nan = int(gse.isna().sum().sum())
    if n_nan:
        print(f"GSE54236: {n_nan} missing expression values imputed with the gene mean")
    gse_z = _zscore_rows(gse).T.fillna(0.0).to_numpy()

    # L2 logistic regression (sklearn default penalty), C chosen by 5-fold AUC
    search = GridSearchCV(LogisticRegression(max_iter=5000),
                          {"C": np.logspace(-4, 4, 10)}, scoring="roc_auc",
                          cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    search.fit(tcga_z, y)
    clf, auc = search.best_estimator_, float(search.best_score_)
    print(f"TCGA transfer model: 5-fold CV AUC for the GD-Net risk label = {auc:.3f}")

    score = clf.decision_function(gse_z)
    high = score >= np.median(score)
    t, s = surv["time"].to_numpy(float), surv["status"].to_numpy(float)
    lr = logrank_test(t[high], t[~high], event_observed_A=s[high],
                      event_observed_B=s[~high])
    print(f"GSE54236 predicted groups: {int(high.sum())} high / {int((~high).sum())} low, "
          f"two-tailed log-rank p = {lr.p_value:.3g}")

    pd.DataFrame({"sample": surv.index, "risk_score": score,
                  "group": np.where(high, "high", "low"),
                  "time": t, "status": s}) \
        .to_csv(os.path.join(out_dir, "gse54236_predicted_groups.csv"), index=False)
    pd.DataFrame({"gene": shared, "coef": clf.coef_.ravel()}) \
        .sort_values("coef", key=np.abs, ascending=False) \
        .to_csv(os.path.join(out_dir, "gse54236_transfer_model_coefs.csv"), index=False)
    pd.DataFrame([{"n_patients": len(t), "n_high": int(high.sum()),
                   "n_low": int((~high).sum()), "n_genes": len(shared),
                   "tcga_cv_auc": auc, "logrank_p": lr.p_value,
                   "status_source": status_note}]) \
        .to_csv(os.path.join(out_dir, "gse54236_logrank_groups.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    _km_plot(ax, t, s, high, f"GSE54236 GD-Net predicted groups "
                             f"(log-rank p = {lr.p_value:.3g})")
    if status_note.startswith("ASSUMED"):
        ax.text(0.01, 0.01, "status assumed: all events", transform=ax.transAxes,
                fontsize=7, color="grey")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gse54236_km_groups.png"), dpi=150)


def km_validation(expr, surv, genes, out_dir, status_note):
    """Supplementary per-gene KM: median expression split, log-rank test."""
    from lifelines.statistics import logrank_test
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    found = [g for g in genes if g in expr.index]
    print(f"per-gene KM: {len(found)}/{len(genes)} genes present")
    if not found:
        return
    t, s = surv["time"], surv["status"]
    ncol = min(3, len(found))
    nrow = int(np.ceil(len(found) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.8 * nrow),
                             squeeze=False)
    rows = []
    for i, g in enumerate(found):
        e = expr.loc[g, surv.index].astype(float).dropna()   # skip GEO nulls
        tg, sg = t.loc[e.index].to_numpy(), s.loc[e.index].to_numpy()
        hi = (e >= e.median()).to_numpy()
        lr = logrank_test(tg[hi], tg[~hi], event_observed_A=sg[hi],
                          event_observed_B=sg[~hi])
        rows.append({"gene": g, "n": len(e), "logrank_p": lr.p_value,
                     "status_source": status_note})
        _km_plot(axes[i // ncol][i % ncol], tg, sg, hi, f"{g} (p={lr.p_value:.3g})")
    for j in range(len(found), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("GSE54236 per-gene survival (median expression split)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gse54236_km_per_gene.png"), dpi=150)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "gse54236_logrank_per_gene.csv"), index=False)
    print(df.drop(columns="status_source").to_string(index=False))


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

    g = sub.add_parser("gse54236", help="GSE54236 survival validation (Fig S5)")
    g.add_argument("--genes", nargs="+", default=None,
                   help="gene symbols, e.g. SPP1 SLC27A5 IGF2")
    g.add_argument("--genes_file", default=None,
                   help="key_ifms.txt / ifms.txt / any one-per-line list; "
                        "'<GENE>_geneExp' names are accepted")
    g.add_argument("--tcga_h5ad", default=None,
                   help="TCGA fused h5ad used by run_pipeline.py (enables the "
                        "predicted-group analysis of Fig S5)")
    g.add_argument("--risk_groups", default=None,
                   help="risk_groups.csv from run_pipeline.py (with --tcga_h5ad)")
    g.add_argument("--surv_tsv", default=None,
                   help="sample,time,status table with real vital status for "
                        "the GSE54236 samples (GSM IDs)")
    g.add_argument("--assume_all_events", action="store_true",
                   help="treat every GSE54236 patient as deceased (GEO has no "
                        "vital status); see module docstring")
    g.add_argument("--expr_tsv", default=None,
                   help="OPTIONAL gene-level expression TSV (genes x GSM samples) "
                        "to use instead of downloading + probe mapping")
    g.add_argument("--out", default="results/validation")
    g.add_argument("--seed", type=int, default=42)

    s = sub.add_parser("stages", help="staged-cohort trends (Fig S6B)")
    s.add_argument("--expr_tsv", required=True)
    s.add_argument("--clinical_tsv", required=True,
                   help="clinical TSV with a tumour-stage column (Xena GDC hub)")
    s.add_argument("--genes", nargs="+", required=True)
    s.add_argument("--out", default="results/validation")

    args = ap.parse_args()
    ensure_dir(args.out)

    if args.cmd == "gse54236":
        genes = read_genes(args.genes, args.genes_file)
        if not genes:
            raise SystemExit("give key genes via --genes and/or --genes_file")
        if bool(args.tcga_h5ad) != bool(args.risk_groups):
            raise SystemExit("--tcga_h5ad and --risk_groups must be given together")

        # the series matrix is always read: it carries the survival times and
        # the tumour / replicate annotation, even when --expr_tsv is supplied
        probe_expr, meta = load_gse54236(args.out)
        surv, status_note = build_survival(meta, args.surv_tsv, args.assume_all_events)
        if args.expr_tsv:
            expr = pd.read_csv(args.expr_tsv, sep="\t", index_col=0)
            print(f"expression from {args.expr_tsv}: {expr.shape}")
        else:
            expr = map_probes_to_genes(probe_expr, args.out)
        surv = surv.loc[surv.index.intersection(expr.columns)]
        if len(surv) < 10:
            raise SystemExit(f"only {len(surv)} samples have both expression and "
                             "survival -- check sample IDs (GSM accessions)")

        if args.tcga_h5ad:
            predicted_group_validation(expr, surv, args.tcga_h5ad, args.risk_groups,
                                       genes, args.out, status_note, seed=args.seed)
        else:
            print("NOTE: without --tcga_h5ad/--risk_groups only the per-gene "
                  "analysis runs; the paper's Fig S5 shows the two PREDICTED groups")
        km_validation(expr, surv, genes, args.out, status_note)
    else:
        stage_trends(args.expr_tsv, args.clinical_tsv, args.genes, args.out)


if __name__ == "__main__":
    main()
