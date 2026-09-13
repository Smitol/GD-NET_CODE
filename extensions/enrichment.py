#!/usr/bin/env python
"""Biological enrichment & network analysis: paper Section 2.5, Fig 3F/3G, Table S3.

Three analyses, all via public web APIs (INTERNET REQUIRED at run time):

  1. GO + KEGG functional enrichment of the informative genes
     -- paper uses R clusterProfiler; here we call the Enrichr REST API
        (same underlying gene-set libraries). Exact-paper route:
        extensions/r_scripts/clusterProfiler_enrichment.R
  2. STRING protein-protein interaction network + PPI enrichment p-value
     (paper Fig 3G: "PPI enrichment p-value of genes is less than 1E-16")
  3. ENCORI/starBase miRNA -> target-gene interactions for the key genes
     (paper: "12 key miRNAs are correlated with 8 key informative genes")

Input = a feature/gene list from the pipeline (key_ifms.txt, ifms.txt or any
one-gene-per-line file). Feature suffixes (_geneExp etc.) are stripped
automatically; miRNA features are separated out for the ENCORI step.

Example:
    python -m extensions.enrichment \
        --genes results/LIHC/key_ifms.txt \
        --out results/LIHC/enrichment

If an API is unreachable the step is skipped with a warning (the other steps
still run) -- rerun later or use the R script.
"""

import argparse
import json
import os
import urllib.parse
import urllib.request

import pandas as pd

from gdnet.utils import ensure_dir, modality_of

ENRICHR = "https://maayanlab.cloud/Enrichr"
STRING = "https://string-db.org/api"
# ENCORI (formerly starBase) REST endpoint -- see https://rnasysu.com/encori/
ENCORI = ("https://rnasysu.com/encori/api/miRNATarget/"
          "?assembly=hg38&geneType=mRNA&miRNA=all&clipExpNum=1"
          "&degraExpNum=0&pancancerNum=0&programNum=1&program=None"
          "&target={gene}&cellType=all")


def _get(url, data=None, timeout=60):
    req = urllib.request.Request(url, data=data,
                                 headers={"User-Agent": "gdnet-extensions"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def split_gene_list(path):
    """Read a feature list; return (gene symbols, miRNA names), suffixes stripped."""
    with open(path) as fh:
        feats = [l.strip() for l in fh if l.strip()]
    genes, mirnas = [], []
    for f in feats:
        if modality_of(f) == "miRNA":
            mirnas.append(f.replace("_miRNAExp", ""))
        else:
            genes.append(f.replace("_geneExp", "").replace("_methylation", ""))
    return sorted(set(genes)), sorted(set(mirnas))


# ---------------------------------------------------------------------------
# 1. GO / KEGG enrichment via Enrichr (libraries analogous to clusterProfiler)
# ---------------------------------------------------------------------------
def enrichr(genes, out_dir,
            libraries=("GO_Biological_Process_2023",
                       "GO_Cellular_Component_2023",
                       "GO_Molecular_Function_2023",
                       "KEGG_2021_Human")):
    payload = urllib.parse.urlencode({
        "list": (None, "\n".join(genes))[1], "description": "gdnet"}).encode()
    # Enrichr expects multipart; simplest robust way: use the JSON addList route
    boundary = "----gdnetboundary"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"list\"\r\n\r\n"
            + "\n".join(genes) +
            f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"description\""
            f"\r\n\r\ngdnet\r\n--{boundary}--\r\n").encode()
    req = urllib.request.Request(f"{ENRICHR}/addList", data=body, headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "User-Agent": "gdnet-extensions"})
    with urllib.request.urlopen(req, timeout=60) as r:
        user_list_id = json.loads(r.read())["userListId"]

    frames = []
    for lib in libraries:
        raw = _get(f"{ENRICHR}/enrich?userListId={user_list_id}&backgroundType={lib}")
        rows = json.loads(raw)[lib]
        # Enrichr row: [rank, term, p, zscore, combined, genes, adj_p, ...]
        df = pd.DataFrame([{"library": lib, "term": r[1], "p_value": r[2],
                            "adj_p": r[6], "combined_score": r[4],
                            "genes": ";".join(r[5])} for r in rows])
        frames.append(df)
    res = pd.concat(frames, ignore_index=True)
    sig = res[res.adj_p < 0.05].sort_values("adj_p")     # paper: FDR < 0.05
    res.to_csv(os.path.join(out_dir, "enrichment_all.csv"), index=False)
    sig.to_csv(os.path.join(out_dir, "enrichment_significant.csv"), index=False)
    print(f"enrichment: {len(sig)} significant terms (FDR<0.05) across {len(libraries)} libraries")

    try:  # Fig 3F style bar plot: top 15 terms by adjusted p
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        top = sig.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top.term.str.slice(0, 60), -np.log10(top.adj_p), color="#4c72b0")
        ax.set_xlabel("-log10 adjusted p")
        ax.set_title("GO/KEGG enrichment of informative genes (paper Fig 3F)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "enrichment_bar.png"), dpi=150)
    except Exception as e:
        print(f"plot skipped: {e}")
    return sig


# ---------------------------------------------------------------------------
# 2. STRING PPI network + enrichment p-value (Fig 3G)
# ---------------------------------------------------------------------------
def string_ppi(genes, out_dir, species=9606, score_threshold=400):
    ids = "%0d".join(genes)
    net = _get(f"{STRING}/tsv/network?identifiers={ids}&species={species}"
               f"&required_score={score_threshold}")
    with open(os.path.join(out_dir, "string_network.tsv"), "wb") as fh:
        fh.write(net)
    edges = pd.read_csv(os.path.join(out_dir, "string_network.tsv"), sep="\t")

    ppi = _get(f"{STRING}/tsv/ppi_enrichment?identifiers={ids}&species={species}")
    ppi_df = pd.read_csv(pd.io.common.BytesIO(ppi), sep="\t")
    p = float(ppi_df["p_value"].iloc[0])
    print(f"STRING: {len(edges)} edges; PPI enrichment p = {p:.2e} "
          f"(paper Fig 3G: < 1e-16)")
    ppi_df.to_csv(os.path.join(out_dir, "string_ppi_enrichment.tsv"),
                  sep="\t", index=False)

    # node degrees (paper: node size/colour = network degree)
    if len(edges):
        deg = (pd.concat([edges.preferredName_A, edges.preferredName_B])
               .value_counts().rename_axis("gene").reset_index(name="degree"))
        deg.to_csv(os.path.join(out_dir, "string_degrees.csv"), index=False)
        print("hub genes:", ", ".join(deg.head(8).gene))
    return edges


# ---------------------------------------------------------------------------
# 3. ENCORI miRNA -> target interactions among the key genes
# ---------------------------------------------------------------------------
def encori_targets(genes, mirnas, out_dir, max_genes=50):
    rows = []
    for g in genes[:max_genes]:                      # API is slow; cap requests
        try:
            raw = _get(ENCORI.format(gene=g), timeout=30).decode()
        except Exception as e:
            print(f"  ENCORI {g}: {e}")
            continue
        for line in raw.splitlines():
            if line.startswith(("#", "miRNAid")) or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) > 3:
                rows.append({"miRNA": parts[1], "gene": g})
    df = pd.DataFrame(rows).drop_duplicates()
    df.to_csv(os.path.join(out_dir, "encori_mirna_targets.csv"), index=False)

    if len(df) and mirnas:
        # which of OUR key miRNAs regulate OUR key genes (paper: 12 miRNAs -> 8 genes)
        norm = lambda s: s.lower().replace("hsa-", "").replace("mir-", "miR-")
        ours = df[df.miRNA.str.lower().str.replace("hsa-", "")
                  .isin([m.lower().replace("hsa-", "") for m in mirnas])]
        ours.to_csv(os.path.join(out_dir, "encori_key_pairs.csv"), index=False)
        print(f"ENCORI: {df.miRNA.nunique()} miRNAs target the key genes; "
              f"{ours.miRNA.nunique()} of them are among OUR key miRNAs "
              f"({ours.gene.nunique()} genes)")
    else:
        print(f"ENCORI: {len(df)} miRNA-target pairs saved")
    return df


def main():
    ap = argparse.ArgumentParser(description="GO/KEGG + STRING + ENCORI (Fig 3F/3G)")
    ap.add_argument("--genes", required=True,
                    help="gene list file: key_ifms.txt / ifms.txt / one symbol per line")
    ap.add_argument("--out", default="results/enrichment")
    ap.add_argument("--skip", default="",
                    help="comma-separated steps to skip: enrichr,string,encori")
    args = ap.parse_args()

    ensure_dir(args.out)
    genes, mirnas = split_gene_list(args.genes)
    print(f"{len(genes)} genes, {len(mirnas)} miRNAs in input list")
    if not genes:
        raise SystemExit("no gene symbols in the input list")
    skip = set(args.skip.split(","))

    for step, fn in [("enrichr", lambda: enrichr(genes, args.out)),
                     ("string", lambda: string_ppi(genes, args.out)),
                     ("encori", lambda: encori_targets(genes, mirnas, args.out))]:
        if step in skip:
            continue
        try:
            fn()
        except Exception as e:   # API down / no internet -> continue with others
            print(f"WARNING: {step} step failed ({e}) -- skipped. "
                  f"Rerun later or use extensions/r_scripts/.")

    print(f"outputs in {args.out}/")


if __name__ == "__main__":
    main()
