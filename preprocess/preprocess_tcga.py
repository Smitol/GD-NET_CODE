#!/usr/bin/env python
"""Step 1 of preprocessing: turn one cancer's raw TSVs into GD-Net inputs.

This implements paper Section 2.6 ("Datasets") for the UCSC-Xena-style GDC
files you downloaded (the layout of your Drive folder "RT SEM03"):

    <data_dir>/TCGA-<CANCER>/TCGA-<CANCER>.star_fpkm-uq.tsv    mRNA   (rows: Ensembl IDs,  values: log2(fpkm-uq+1))
    <data_dir>/TCGA-<CANCER>/TCGA-<CANCER>.mirna.tsv           miRNA  (rows: miRNA IDs,    values: log2(RPM+1))
    <data_dir>/TCGA-<CANCER>/TCGA-<CANCER>.methylation450.tsv  methyl (rows: cg probes,    values: beta 0..1)
    <data_dir>/TCGA-<CANCER>/TCGA-<CANCER>.survival.tsv        columns: sample, OS.time, OS, _PATIENT

(The .tsv may also be gzipped as .tsv.gz, or sit inside a same-named
sub-folder -- both are handled automatically.)

Two ID-mapping files are needed (auto-downloaded if missing, ~25 MB total):
    gencode.v36 probemap ......... Ensembl gene id -> gene symbol   (for mRNA)
    HM450 gencode.v36 probeMap ... cg probe -> gene symbol(s)       (for methylation)

WHAT THE SCRIPT DOES (paper Section 2.6, in order):
    1. survival: keep primary-tumour samples (barcode '-01'), OS time/event
    2. mRNA:   Ensembl -> symbol, keep KEGG-network genes
    3. methyl: average beta of all CpG probes inside each gene ("we computed
               features at the gene level by taking the average of DNA
               methylation values across CpG sites located within each gene"),
               keep KEGG-network genes
    4. miRNA:  keep all miRNAs (they are not nodes of the KEGG gene graph)
    5. intersect samples present in ALL three modalities + survival
    6. drop features with >20% missing, then samples with >20% missing,
       impute remaining NaN with the feature median
    7. fuse: X = rowbind(mRNA, methylation, miRNA)  (paper Eq. 1) -- feature
       names get suffixes '_geneExp' / '_methylation' / '_miRNAExp' exactly
       like the authors' sample data
    8. build the GCN edge list: each KEGG gene-gene edge is added twice,
       once between the two '_geneExp' features and once between the two
       '_methylation' features (this is what the official dataprocess.py does)
    9. save  <out_dir>/<CANCER>.h5ad, <CANCER>_edges.csv, <CANCER>_surv.csv

Usage (repeat per cancer, or loop over all eight):
    python preprocess/preprocess_tcga.py \\
        --cancer LIHC \\
        --data_dir "path/to/RT SEM03" \\
        --kegg_edges data/kegg_edges_symbols.csv \\
        --out_dir data/processed

NOTE ON MEMORY: methylation450 files are large (400k+ rows). We stream them
in chunks and only keep probes that map to KEGG genes, so peak memory stays
low (a few GB). If you still run out of RAM, lower --chunksize.
"""

import argparse
import gzip
import os
import urllib.request

import anndata
import numpy as np
import pandas as pd

# Public URLs of the two ID-mapping files (same files for every cancer).
GENCODE_PROBEMAP_URL = ("https://gdc-hub.s3.us-east-1.amazonaws.com/download/"
                        "gencode.v36.annotation.gtf.gene.probemap")
METHYL_PROBEMAP_URL = ("https://gdc-hub.s3.us-east-1.amazonaws.com/download/"
                       "HM450.hg38.manifest.gencode.v36.probeMap")


# --------------------------------------------------------------------------- io helpers
def find_file(data_dir: str, cancer: str, kind: str) -> str:
    """Locate e.g. TCGA-LIHC.mirna.tsv, tolerating .gz and the
    'file inside a same-named folder' layout that Drive sometimes creates."""
    base = os.path.join(data_dir, f"TCGA-{cancer}", f"TCGA-{cancer}.{kind}.tsv")
    for cand in (base, base + ".gz",
                 os.path.join(base, f"TCGA-{cancer}.{kind}.tsv"),          # nested folder
                 os.path.join(base, f"TCGA-{cancer}.{kind}.tsv") + ".gz"):
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        f"could not find TCGA-{cancer}.{kind}.tsv under {data_dir} -- "
        f"adjust find_file() if your folder layout differs")


def ensure_probemap(path: str, url: str) -> str:
    """Download an ID-mapping file if it is not present yet."""
    if not os.path.isfile(path):
        print(f"downloading {os.path.basename(path)} ...")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def short_id(barcode: str) -> str:
    """'TCGA-2J-AAB1-01A' -> 'TCGA-2J-AAB1-01' (drop the vial letter so the
    same physical sample matches across modalities)."""
    return barcode[:15]


# --------------------------------------------------------------------------- loaders
def load_survival(path: str) -> pd.DataFrame:
    """-> DataFrame indexed by 15-char sample id, columns time / status."""
    df = pd.read_csv(path, sep="\t")
    df = df[df["sample"].str[13:15] == "01"]              # primary tumour only
    df["sid"] = df["sample"].map(short_id)
    df = df.drop_duplicates("sid").set_index("sid")
    out = df.rename(columns={"OS.time": "time", "OS": "status"})[["time", "status"]]
    return out.dropna()


def load_mrna(path: str, probemap: str, kegg_genes: set) -> pd.DataFrame:
    """mRNA matrix -> (samples x KEGG genes), symbols as columns.

    Xena's star_fpkm-uq.tsv is ALREADY log2(fpkm-uq + 1); no further log is
    taken. If you use raw counts instead, log2-transform them here.
    """
    pm = pd.read_csv(probemap, sep="\t")[["id", "gene"]]
    ens2sym = dict(zip(pm["id"], pm["gene"]))

    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index = [ens2sym.get(i, ens2sym.get(i.split(".")[0], "")) for i in df.index]
    df = df[df.index.isin(kegg_genes)]
    df = df.groupby(level=0).mean()                        # collapse duplicate symbols
    df.columns = [short_id(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.loc[:, [c for c in df.columns if c.endswith("-01")]]
    return df.T                                            # -> samples x genes


def load_mirna(path: str) -> pd.DataFrame:
    """miRNA matrix -> (samples x miRNAs). Values are already log2(RPM+1)."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.columns = [short_id(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.loc[:, [c for c in df.columns if c.endswith("-01")]]
    return df.T


def load_methylation(path: str, probemap: str, kegg_genes: set,
                     chunksize: int = 20000) -> pd.DataFrame:
    """Methylation -> (samples x KEGG genes) = mean beta over each gene's CpGs.

    Streamed in chunks because the raw file has ~400k probe rows. Probes
    mapping to several genes (comma-separated in the probemap) count for
    each of those genes.
    """
    pm = pd.read_csv(probemap, sep="\t")
    pm.columns = [c.lstrip("#") for c in pm.columns]
    probe_genes = {}                                       # probe -> [KEGG genes]
    for probe, genes in zip(pm["id"], pm["gene"].astype(str)):
        gl = [g for g in genes.split(",") if g in kegg_genes]
        if gl:
            probe_genes[probe] = gl

    sums, counts, columns = {}, {}, None
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for chunk in pd.read_csv(fh, sep="\t", index_col=0, chunksize=chunksize):
            if columns is None:
                columns = chunk.columns
            chunk = chunk[chunk.index.isin(probe_genes)]
            if chunk.empty:
                continue
            vals = chunk.to_numpy(dtype=float)
            for row, probe in enumerate(chunk.index):
                v = vals[row]
                m = ~np.isnan(v)
                for g in probe_genes[probe]:
                    if g in sums:
                        sums[g][m] += v[m]
                        counts[g][m] += 1
                    else:
                        s = np.zeros(len(v)); c = np.zeros(len(v))
                        s[m] = v[m]; c[m] = 1
                        sums[g], counts[g] = s, c

    genes = sorted(sums)
    mat = np.vstack([np.divide(sums[g], counts[g],
                               out=np.full_like(sums[g], np.nan),
                               where=counts[g] > 0) for g in genes])
    df = pd.DataFrame(mat, index=genes, columns=[short_id(c) for c in columns])
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.loc[:, [c for c in df.columns if c.endswith("-01")]]
    return df.T


# --------------------------------------------------------------------------- pipeline
def filter_and_impute(df: pd.DataFrame, max_missing: float = 0.2) -> pd.DataFrame:
    """Paper Section 2.6: drop features then samples with >20% missing,
    impute the rest with the feature median."""
    df = df.loc[:, df.isna().mean(axis=0) <= max_missing]
    df = df.loc[df.isna().mean(axis=1) <= max_missing, :]
    return df.fillna(df.median(axis=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cancer", required=True, help="e.g. LIHC, PAAD, LUAD ...")
    ap.add_argument("--data_dir", required=True,
                    help="folder that contains the TCGA-<CANCER> sub-folders")
    ap.add_argument("--kegg_edges", required=True,
                    help="gene-symbol edge list from build_kegg_network.py")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--gencode_probemap", default="data/probemaps/gencode.v36.probemap")
    ap.add_argument("--methyl_probemap",
                    default="data/probemaps/HM450.hg38.manifest.gencode.v36.probeMap")
    ap.add_argument("--chunksize", type=int, default=20000,
                    help="methylation streaming chunk size (lower it if RAM is tight)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ensure_probemap(args.gencode_probemap, GENCODE_PROBEMAP_URL)
    ensure_probemap(args.methyl_probemap, METHYL_PROBEMAP_URL)

    # ---- KEGG network ------------------------------------------------------
    edges = pd.read_csv(args.kegg_edges, header=None, names=["a", "b"])
    kegg_genes = set(edges["a"]) | set(edges["b"])
    print(f"KEGG network: {len(edges)} edges, {len(kegg_genes)} genes")

    # ---- load the four files ----------------------------------------------
    print("loading survival ...")
    surv = load_survival(find_file(args.data_dir, args.cancer, "survival"))
    print(f"  {len(surv)} tumour samples with survival")

    print("loading mRNA (fpkm-uq) ...")
    mrna = load_mrna(find_file(args.data_dir, args.cancer, "star_fpkm-uq"),
                     args.gencode_probemap, kegg_genes)
    print(f"  {mrna.shape[0]} samples x {mrna.shape[1]} KEGG genes")

    print("loading miRNA ...")
    mirna = load_mirna(find_file(args.data_dir, args.cancer, "mirna"))
    print(f"  {mirna.shape[0]} samples x {mirna.shape[1]} miRNAs")

    print("loading methylation450 (chunked, takes a few minutes) ...")
    meth = load_methylation(find_file(args.data_dir, args.cancer, "methylation450"),
                            args.methyl_probemap, kegg_genes, args.chunksize)
    print(f"  {meth.shape[0]} samples x {meth.shape[1]} gene-level methylation features")

    # ---- sample intersection (paper: patients must have all modalities) ----
    samples = sorted(set(mrna.index) & set(mirna.index) & set(meth.index) & set(surv.index))
    print(f"samples with all 3 modalities + survival: {len(samples)}")
    if len(samples) < 50:
        print("WARNING: very few overlapping samples -- check that your files "
              "belong to the same cancer cohort")

    mrna, mirna, meth = mrna.loc[samples], mirna.loc[samples], meth.loc[samples]
    surv = surv.loc[samples]

    # ---- missing-value filtering + imputation ------------------------------
    mrna, mirna, meth = (filter_and_impute(m) for m in (mrna, mirna, meth))
    # filtering may have dropped samples; re-intersect
    samples = sorted(set(mrna.index) & set(mirna.index) & set(meth.index))
    mrna, mirna, meth, surv = mrna.loc[samples], mirna.loc[samples], meth.loc[samples], surv.loc[samples]

    # ---- fuse (paper Eq. 1: X = rowbind(X1, X2, X3)) -----------------------
    mrna.columns = [f"{g}_geneExp" for g in mrna.columns]
    meth.columns = [f"{g}_methylation" for g in meth.columns]
    mirna.columns = [f"{m}_miRNAExp" for m in mirna.columns]
    fused = pd.concat([mrna, meth, mirna], axis=1).astype(np.float32)
    print(f"fused matrix: {fused.shape[0]} samples x {fused.shape[1]} features "
          f"({mrna.shape[1]} mRNA + {meth.shape[1]} methyl + {mirna.shape[1]} miRNA)")

    # ---- edge list on feature indices --------------------------------------
    feat_idx = {f: i for i, f in enumerate(fused.columns)}
    rows = []
    for a, b in edges.itertuples(index=False):
        for suffix in ("_geneExp", "_methylation"):        # same topology for both blocks
            ia, ib = feat_idx.get(f"{a}{suffix}"), feat_idx.get(f"{b}{suffix}")
            if ia is not None and ib is not None:
                rows.append((ia, ib))
    edge_df = pd.DataFrame(rows).drop_duplicates()
    print(f"feature-index edges: {len(edge_df)}")

    # ---- save ---------------------------------------------------------------
    h5ad_path = os.path.join(args.out_dir, f"{args.cancer}.h5ad")
    anndata.AnnData(X=fused.values,
                    obs=pd.DataFrame(index=fused.index),
                    var=pd.DataFrame(index=fused.columns)).write_h5ad(h5ad_path)
    edge_df.to_csv(os.path.join(args.out_dir, f"{args.cancer}_edges.csv"),
                   index=False, header=False)
    # survival in the transposed format used by the authors' sample data
    surv_t = pd.DataFrame([surv["time"].values, surv["status"].values],
                          index=["time", "status"], columns=samples)
    surv_t.index.name = "GeneSymbol"
    surv_t.to_csv(os.path.join(args.out_dir, f"{args.cancer}_surv.csv"))

    print(f"\nwrote:\n  {h5ad_path}\n  {args.out_dir}/{args.cancer}_edges.csv"
          f"\n  {args.out_dir}/{args.cancer}_surv.csv")
    print("\nNext step:\n  python run_pipeline.py "
          f"--input_h5ad_path {h5ad_path} "
          f"--input_edge_path {args.out_dir}/{args.cancer}_edges.csv "
          f"--surv_path {args.out_dir}/{args.cancer}_surv.csv "
          f"--out results/{args.cancer}")


if __name__ == "__main__":
    main()
