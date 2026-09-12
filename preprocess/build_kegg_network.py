#!/usr/bin/env python
"""Step 0 of preprocessing: build the KEGG gene-gene network from KGML files.

The paper uses "the KEGG gene connection pathway G for each cancer ... as the
input network to the graph convolutional layer" (Section 2.1.1). This script
turns your downloaded raw KEGG pathway files into a simple 2-column edge list
of GENE SYMBOLS which preprocess_tcga.py then maps onto feature indices.

INPUTS (this matches the layout of your Drive folder "RT SEM03/KEGG"):
    --kgml_dir   folder with the KGML pathway XMLs, e.g. KEGG/KEGG_human_raw/kgml/
                 (hsa00010.xml, hsa04110.xml, ...)
    --gene_list  the KEGG human gene list from `https://rest.kegg.jp/list/hsa`,
                 e.g. KEGG/KEGG_human_raw/mappings/kegg_human_genes.txt
                 Format per line:  hsa:<ncbi_id> \\t <type> \\t <position> \\t "SYM, alias...; description"

OUTPUT:
    --out        CSV with two columns (gene_a, gene_b), one row per undirected
                 edge, gene symbols, no header. Default: kegg_edges_symbols.csv

HOW EDGES ARE EXTRACTED from each KGML pathway:
    * every <entry type="gene"> node lists one or more genes ("hsa:1029 hsa:51343 ...")
    * every <relation entry1=".." entry2=".."> between two gene entries adds
      an edge between ALL gene pairs of the two entries (KEGG groups isoforms/
      complex members into one node).
This covers PPrel (protein-protein), GErel (gene expression), ECrel (enzyme)
relations -- i.e. the "gene connection pathways" of the paper.

Usage:
    python preprocess/build_kegg_network.py \\
        --kgml_dir  "path/to/RT SEM03/KEGG/KEGG_human_raw/kgml" \\
        --gene_list "path/to/RT SEM03/KEGG/KEGG_human_raw/mappings/kegg_human_genes.txt" \\
        --out data/kegg_edges_symbols.csv
"""

import argparse
import glob
import os
import xml.etree.ElementTree as ET

import pandas as pd


def load_symbol_map(gene_list_path: str) -> dict:
    """Parse `rest.kegg.jp/list/hsa` output into {'hsa:7157': 'TP53', ...}.

    The 4th column looks like "TP53, BCC7, LFS1; tumor protein p53".
    The official gene SYMBOL is the first token before the first ',' or ';'.
    Non-coding entries without a symbol (no ';' separator) are skipped.
    """
    mapping = {}
    with open(gene_list_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            kegg_id, desc = parts[0], parts[3]
            if ";" not in desc:      # no symbol annotated -> skip
                continue
            symbol = desc.split(";")[0].split(",")[0].strip()
            if symbol:
                mapping[kegg_id] = symbol
    return mapping


def parse_kgml(path: str) -> set:
    """Extract gene-gene edges (as pairs of 'hsa:<id>') from one KGML file."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        print(f"  ! could not parse {os.path.basename(path)}, skipped")
        return set()

    # entry id -> list of hsa gene ids (only for type="gene" entries)
    entry_genes = {}
    for entry in root.findall("entry"):
        if entry.get("type") == "gene":
            genes = [g for g in entry.get("name", "").split() if g.startswith("hsa:")]
            if genes:
                entry_genes[entry.get("id")] = genes

    edges = set()
    for rel in root.findall("relation"):
        g1 = entry_genes.get(rel.get("entry1"))
        g2 = entry_genes.get(rel.get("entry2"))
        if not g1 or not g2:
            continue  # relation involves a compound/map node -> not a gene-gene edge
        for a in g1:
            for b in g2:
                if a != b:
                    edges.add((min(a, b), max(a, b)))  # undirected, deduped
    return edges


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kgml_dir", required=True)
    ap.add_argument("--gene_list", required=True)
    ap.add_argument("--out", default="data/kegg_edges_symbols.csv")
    args = ap.parse_args()

    symbol_of = load_symbol_map(args.gene_list)
    print(f"{len(symbol_of)} KEGG ids with gene symbols")

    xmls = sorted(glob.glob(os.path.join(args.kgml_dir, "*.xml")))
    if not xmls:
        raise SystemExit(f"no .xml files found in {args.kgml_dir} -- check the path!")
    print(f"parsing {len(xmls)} KGML pathway files ...")

    all_edges = set()
    for x in xmls:
        all_edges |= parse_kgml(x)
    print(f"{len(all_edges)} unique hsa-id edges")

    # map ids -> symbols; drop edges whose genes have no symbol
    sym_edges = set()
    for a, b in all_edges:
        sa, sb = symbol_of.get(a), symbol_of.get(b)
        if sa and sb and sa != sb:
            sym_edges.add((min(sa, sb), max(sa, sb)))
    print(f"{len(sym_edges)} unique gene-symbol edges "
          f"({len(set(g for e in sym_edges for g in e))} genes)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(sorted(sym_edges)).to_csv(args.out, index=False, header=False)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
