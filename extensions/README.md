# GD-Net extensions — the analyses the paper reports *around* the core pipeline

The core repo covers: data → embeddings → risk score → C-index → biomarker
lists. This package adds **everything else the paper reports**, one module
per gap. Nothing here modifies the core code — extensions only *import* it.

Run every module from the **repo root** with `python -m extensions.<module>`
(that makes the `gdnet` and `main` imports resolve). All have `--help`.

## Gap → module map

| # | Missing item (paper location) | Module | Output |
|---|---|---|---|
| 1 | 6 benchmark methods: PCA-Cox, Bridge-Cox, eXGBC, Deep_surv, DCAP, MTC (Table 1, Fig 2A/2B) | `benchmarks.py` + `run_benchmarks.py` | per-fold CI/AUC CSVs, summary with t-test vs GD-Net, Fig-2A boxplot |
| 2 | Single-modality ablation (Fig 2C) | `ablation.py` | CI per modality + bar chart |
| 3a | AUC / ROC curves (Sec 3.2, Fig S1) | inside `run_benchmarks.py` | `benchmark_auc.csv`, `roc_curves.png` |
| 3b | Runtime scaling analysis (Fig S3) | `runtime_scaling.py` | time-vs-n CSV + linear-fit plot |
| 4 | PCoA, PERMANOVA, t-SNE, KMeans+ARI w/ shuffle baseline (Fig 2D/2E, 3D) | `ordination.py` | `pcoa.png`, `tsne.png`, `ordination_stats.txt` |
| 5 | GO/KEGG enrichment, STRING PPI, ENCORI miRNA-targets (Sec 2.5, Fig 3F/3G, Table S3) | `enrichment.py` (web APIs) / `r_scripts/clusterProfiler_enrichment.R` (exact paper) | enrichment CSVs, bar plot, network TSV, degree table |
| 6 | GSE54236 validation + 345-sample staged-cohort trends (Sec 3.4, Fig S5/S6) | `external_validation.py` | per-gene KM + log-rank CSV, stage-trend boxplots |
| 7 | Real DESeq2 differential expression (Sec 2.3–2.4) | `deseq2_analysis.py` (pydeseq2) / `r_scripts/run_deseq2.R` (exact paper) | DEG tables, volcano plot |
| 8 | LUAD (or any-cancer) full case study (Discussion, Fig S8) | `case_study.py` | chains 4+5+Table-2 for one cancer |
| + | Methylation-vs-expression inverse-relation analysis (Fig S7 — bonus, not on the original list) | `methylation_expression.py` | gene table + scatter |

## Extra dependencies

Core `requirements.txt` covers most. Additionally:

```bash
pip install pydeseq2        # only for deseq2_analysis.py (optional; R route exists)
# R routes (exact paper setup): R + BiocManager::install(c("DESeq2","clusterProfiler","org.Hs.eg.db"))
```

`enrichment.py` and `external_validation.py gse54236` need **internet**
(Enrichr / STRING / ENCORI / GEO web APIs).

## Typical order for one cancer (after `run_pipeline.py` has produced `results/<C>/`)

```bash
C=LIHC
# 1. benchmarks (Table 1 / Fig 2A / Fig S1) — slowest step, ~10–30 min CPU
python -m extensions.run_benchmarks \
    --input_h5ad_path data/processed/$C.h5ad --surv_path data/processed/${C}_surv.csv \
    --gdnet_embeddings results/$C/embeddings.csv --out results/$C/benchmarks

# 2. ablation (Fig 2C) — trains the encoder 4×
python -m extensions.ablation \
    --input_h5ad_path data/processed/$C.h5ad --input_edge_path data/processed/${C}_edges.csv \
    --surv_path data/processed/${C}_surv.csv --epochs 100 --out results/$C/ablation

# 3. full case study (ordination + meth-expr + enrichment + Table 2)
python -m extensions.case_study \
    --input_h5ad_path data/processed/$C.h5ad --results results/$C

# 4. real DESeq2 (needs the raw-counts TSV, e.g. TCGA-LIHC.star_counts.tsv)
python -m extensions.deseq2_analysis \
    --counts_tsv data/TCGA-$C/TCGA-$C.star_counts.tsv \
    --risk_groups results/$C/risk_groups.csv --out results/$C/deseq2

# 5. external validation (liver cancer only, paper Fig S5/S6)
python -m extensions.external_validation gse54236 \
    --genes SPP1 SLC27A5 IGF2 EFNA3 DCAF4L2 --out results/$C/validation
python -m extensions.external_validation stages \
    --expr_tsv data/TCGA-$C/TCGA-$C.star_fpkm-uq.tsv \
    --clinical_tsv data/TCGA-$C/TCGA-$C.clinical.tsv \
    --genes MCM3 BUB1B DLGAP5 ECT2 --out results/$C/validation

# 6. runtime scaling (Fig S3)
python -m extensions.runtime_scaling \
    --input_h5ad_path data/processed/$C.h5ad --input_edge_path data/processed/${C}_edges.csv \
    --out results/$C/scaling
```

## Honesty notes (read before comparing numbers to the paper)

* **Benchmark hyper-parameters are not published** (supplementary material is
  unavailable) — each of the six methods is a faithful re-creation of its
  published *idea* with sensible defaults, `# TUNE` comments mark the knobs.
  Expect Table-1-ballpark numbers, not identical ones.
* **MTC** (Qiu 2020) is a *meta-learning* method that pre-trains on external
  pan-cancer cohorts; without those corpora we substitute within-cohort
  self-supervised pre-training + Cox fine-tuning (documented in the code).
* **AUC** definition (the paper never specifies one): event-by-median-follow-up
  binary ROC, censored-before-horizon samples excluded.
* **GSE54236** is an Agilent two-channel array — cross-platform application of
  the trained TCGA model is not meaningful; like the paper's Fig S5 we
  validate the *key genes* by expression-split survival analysis instead.
* **PERMANOVA/PCoA** are implemented from the standard definitions
  (Torgerson MDS + Anderson 2001) — results match R `vegan`/`ape` up to
  permutation noise.
