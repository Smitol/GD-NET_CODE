#!/usr/bin/env bash
# Run the complete GD-Net pipeline for all eight TCGA cancers of the paper.
#
# Usage:
#     bash run_all_cancers.sh "path/to/your/data_dir"
#
# <data_dir> must contain one TCGA-<CANCER>/ folder per cancer with the four
# Xena-style TSVs (see README section 3). Edit CANCERS below if your list
# differs from the paper's eight.
#
# Prereq (run once): the shared KEGG edge list
#     python preprocess/build_kegg_network.py --kgml_dir ... --gene_list ... \
#         --out data/kegg_edges_symbols.csv
set -euo pipefail

DATA_DIR="${1:?usage: bash run_all_cancers.sh <data_dir>}"
KEGG_EDGES="data/kegg_edges_symbols.csv"

# The eight cancers benchmarked in the paper (Table 1). EDIT to match your data.
CANCERS=(BLCA BRCA HNSC KIRC LGG LIHC LUAD PAAD)

for C in "${CANCERS[@]}"; do
    echo "=============================== $C ==============================="

    # Step 1: preprocess raw TSVs -> fused h5ad + edge list + survival
    python preprocess/preprocess_tcga.py \
        --cancer "$C" \
        --data_dir "$DATA_DIR" \
        --kegg_edges "$KEGG_EDGES" \
        --out_dir data/processed

    # Step 2: encoder training + Cox-EN CV + feature selection + KM curve
    python run_pipeline.py \
        --input_h5ad_path "data/processed/$C.h5ad" \
        --input_edge_path "data/processed/${C}_edges.csv" \
        --surv_path       "data/processed/${C}_surv.csv" \
        --out "results/$C"
done

# Summary table (mean C-index per cancer, like paper Table 1)
echo
echo "================== SUMMARY (mean 5-fold C-index) =================="
python - <<'PY'
import glob, os
import pandas as pd
rows = []
for f in sorted(glob.glob("results/*/cv_results.csv")):
    df = pd.read_csv(f, index_col=0)
    rows.append({"cancer": os.path.basename(os.path.dirname(f)),
                 "c_index": df.c_index.mean(), "std": df.c_index.std(),
                 "abs_log10_p": df.abs_log10_p.mean()})
print(pd.DataFrame(rows).to_string(index=False))
PY
