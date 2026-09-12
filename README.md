# GD-Net — reproduction of "Contrastive-learning of a graph neural network for cancer survival prediction from multi-omics data" (Lin et al., *J Cell Mol Med* 2024)

A complete, runnable re-implementation of **GD-Net**: predicting cancer
survival from fused multi-omics data (mRNA + miRNA + DNA methylation) with a
**GCN + MoCo contrastive encoder**, an **elastic-net Cox** risk model, and
**XGBoost + differential-analysis** biomarker selection.

The authors' official repo (`github.com/JackiLin/GD-Net`) ships only the
encoder and a PAAD sample dataset — the training script, Cox model, XGBoost
module and all evaluation code are missing. This repo fills in every missing
piece, is **CPU-friendly** (no GPU needed), and includes preprocessing for
UCSC-Xena-style TCGA TSV downloads (the format of your 8-cancer dataset).

> **Note on results:** exact hyper-parameters live in the paper's unavailable
> supplementary material, so numbers will be *in the ballpark* of the paper
> (e.g. PAAD C-index 0.670 ± 0.077 in Table 1), not bit-identical.

---

## Repository layout

```
main.py                          Stage 1: train the contrastive GCN encoder, export embeddings
run_pipeline.py                  Stage 1+2: EVERYTHING end-to-end for one cancer
gdnet/
  builder.py                     GCN layer + MoCo momentum-contrast model (paper Eq. 2-5)
  loader.py                      data augmentations -> positive pairs (paper Fig. 1A)
  cox_en.py                      Cox-Elastic-Net risk model + 5-fold CV (Eq. 6-9)
  feature_selection.py           XGBoost top-200 + differential analysis + IFMs (Sec 2.1.5, 2.3)
  utils.py                       loaders, seeding, modality tagging
preprocess/
  build_kegg_network.py          Step 0: KGML pathway XMLs -> gene-gene edge list
  preprocess_tcga.py             Step 1: raw TCGA TSVs -> fused .h5ad + edges + survival
data/paad_demo/                  authors' bundled PAAD data (173 patients) — works out of the box
requirements.txt
```

## The GD-Net pipeline (what each step does)

| # | Paper section | Step | Code |
|---|---|---|---|
| 0 | 2.1.1 | Build the KEGG gene-gene network | `preprocess/build_kegg_network.py` |
| 1 | 2.6 | Clean each omics TSV, gene-level methylation, sample intersection, **early fusion** `X = rowbind(mRNA, methyl, miRNA)` (Eq. 1) | `preprocess/preprocess_tcga.py` |
| 2 | 2.1.2 | Two augmented views per patient (mask / Gaussian noise / swap / crossover) = positive pairs | `gdnet/loader.py` |
| 3 | 2.1.3 | GCN over KEGG graph (Eq. 2) + MLP encoder, trained with MoCo momentum contrast + InfoNCE loss (Eq. 3-5) → 200-dim embeddings | `gdnet/builder.py`, `main.py` |
| 4 | 2.1.4 | Cox-Elastic-Net on the embeddings → risk scores → median-split high/low risk groups (Eq. 6-7) | `gdnet/cox_en.py` |
| 5 | 2.1.5 | XGBoost (depth 2–8 grid) predicts the risk label from the ORIGINAL features → top-200 importance = F_XGB | `gdnet/feature_selection.py` |
| 6 | 2.3 | Differential analysis high vs low risk (\|log2FC\|>1.6, adj-p<0.05) = F_DE; **IFMs = F_XGB ∪ F_DE**, **key-IFMs = F_XGB ∩ F_DE** | `gdnet/feature_selection.py` |
| 7 | 2.1.6, 2.2 | 5-fold CV: Harrell's C-index (Eq. 9) + log-rank \|log10 p\| + KM curves | `gdnet/cox_en.py`, `run_pipeline.py` |

---

## 1. Installation

```bash
python3 -m venv venv && source venv/bin/activate   # optional but recommended

# CPU-only torch first (skip this line if you have a CUDA GPU):
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

Tested with Python 3.11, torch 2.x, on CPU.

## 2. Quick start — bundled PAAD demo (no downloads needed)

```bash
python run_pipeline.py \
    --input_h5ad_path data/paad_demo/paad.h5ad \
    --input_edge_path data/paad_demo/paad_edges.csv \
    --surv_path       data/paad_demo/paad_surv.csv \
    --epochs 100 --out results/paad_demo
```

Runtime: a few minutes on a laptop CPU. Outputs in `results/paad_demo/`:

| file | contents |
|---|---|
| `embeddings.csv` | 200-dim contrastive embeddings per patient |
| `train_log.csv` | contrastive loss / accuracy per 10 epochs |
| `cv_results.csv` | **C-index ± std and \|log10 p\| per CV fold (paper Table 1)** |
| `risk_groups.csv` | risk score + high/low group per patient |
| `xgboost_top200.csv` | F_XGB: top-200 informative features |
| `differential_features.csv` | F_DE: differential molecules between risk groups |
| `ifms.txt`, `key_ifms.txt` | union / intersection biomarker sets (paper Table 2) |
| `km_curve.png` | Kaplan-Meier curves of the two risk groups (paper Fig. 3E) |

To only train the encoder (Stage 1) use `main.py` with the same `--input_*`
flags — this mirrors the CLI that the official README documents.

## 3. Running on YOUR 8-cancer TCGA dataset

Your Drive folder layout (UCSC-Xena GDC hub files) is what
`preprocess_tcga.py` expects:

```
<data_dir>/TCGA-<CANCER>/TCGA-<CANCER>.star_fpkm-uq.tsv     mRNA
<data_dir>/TCGA-<CANCER>/TCGA-<CANCER>.mirna.tsv            miRNA
<data_dir>/TCGA-<CANCER>/TCGA-<CANCER>.methylation450.tsv   DNA methylation
<data_dir>/TCGA-<CANCER>/TCGA-<CANCER>.survival.tsv         OS time + event
```
(`.tsv.gz` also works; files nested one folder deeper are found automatically.)

### Step 0 — build the KEGG network (once, shared by all cancers)

```bash
python preprocess/build_kegg_network.py \
    --kgml_dir  "path/to/KEGG/KEGG_human_raw/kgml" \
    --gene_list "path/to/KEGG/KEGG_human_raw/mappings/kegg_human_genes.txt" \
    --out data/kegg_edges_symbols.csv
```

If you don't have the KGML files: download them with
`https://rest.kegg.jp/list/pathway/hsa` → `https://rest.kegg.jp/get/<id>/kgml`
(one XML per pathway), and the gene list from `https://rest.kegg.jp/list/hsa`.

### Step 1 — preprocess one cancer (repeat / loop for all eight)

```bash
python preprocess/preprocess_tcga.py \
    --cancer LIHC \
    --data_dir "path/to/your/data" \
    --kegg_edges data/kegg_edges_symbols.csv \
    --out_dir data/processed
```

This writes `data/processed/LIHC.h5ad`, `LIHC_edges.csv`, `LIHC_surv.csv`.
Two small ID-mapping files (Ensembl→symbol, CpG-probe→gene) are
auto-downloaded on first use (~25 MB).

### Step 2 — run the full pipeline

```bash
python run_pipeline.py \
    --input_h5ad_path data/processed/LIHC.h5ad \
    --input_edge_path data/processed/LIHC_edges.csv \
    --surv_path       data/processed/LIHC_surv.csv \
    --out results/LIHC
```

### All 8 cancers in one go

```bash
bash run_all_cancers.sh "path/to/your/data"    # edit the CANCERS list inside
```

---

## Things you may want to tune (marked with comments in the code)

| Where | Flag / constant | Why |
|---|---|---|
| `run_pipeline.py` | `--epochs` (200), `--lr` (0.01), `--low_dim` (200) | official README defaults; fewer epochs = faster, usually fine |
| `run_pipeline.py` | `--penalizer` (0.1), `--l1_ratio` (0.5) | Cox-EN regularisation — the paper doesn't publish these; tune if C-index is low. The fit auto-escalates the penalty if it fails to converge on small cohorts. |
| `gdnet/loader.py` | `DEFAULT_AUG` dict | augmentation strengths; lower them if the contrastive loss diverges |
| `gdnet/feature_selection.py` | `lfc_threshold=1.6`, `p_threshold=0.05` | paper's differential cut-offs |
| `preprocess/preprocess_tcga.py` | missing-rate 20 %, median impute | paper Section 2.6 |

## Deviations from the paper (documented, deliberate)

1. **DESeq2 → Welch t-test + BH-FDR.** The paper runs DESeq2 (R) on raw mRNA
   counts. To keep everything in Python we use a t-test on the log2 matrix
   with the *same* thresholds. For the exact paper setup, export
   `risk_groups.csv` + raw counts and run DESeq2 in R.
2. **Vectorised GCN.** The official per-sample `GCNConv(1,1)` loop is
   replaced by one sparse mat-mul (mathematically identical, 100–1000×
   faster, no torch_geometric dependency) — see `gdnet/builder.py` header.
3. **Robust Cox fitting.** Embeddings are standardised and the elastic-net
   penalty auto-escalates on convergence failure (small-cohort necessity).
4. **Hyper-parameters** not published in the paper use the official README's
   CLI defaults where available, otherwise sensible values (all documented
   in `--help` and code comments).
