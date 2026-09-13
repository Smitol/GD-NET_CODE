"""GD-Net *extensions*: everything the paper reports AROUND the core pipeline.

The core repo (gdnet/, main.py, run_pipeline.py) covers:
    data -> embeddings -> Cox-EN risk -> C-index -> IFM biomarker lists

This package adds the missing analyses, one module per gap:

    benchmarks.py               6 comparison methods of Table 1 / Fig 2A
    run_benchmarks.py           CLI: 5-fold CV table + boxplot + ROC/AUC (Fig S1)
    ablation.py                 single-modality ablation (Fig 2C)
    ordination.py               PCoA + PERMANOVA + t-SNE + KMeans/ARI (Fig 2D/2E, 3D)
    runtime_scaling.py          samples-vs-time scaling analysis (Fig S3)
    deseq2_analysis.py          real DESeq2 DE analysis via pydeseq2 (Sec 2.4)
    enrichment.py               GO/KEGG (Enrichr), STRING PPI, ENCORI miRNA (Fig 3F/3G)
    external_validation.py      GSE54236 cohort + staged-cohort trends (Fig S5/S6)
    methylation_expression.py   methylation-vs-expression analysis (Fig S7)
    case_study.py               orchestrates the full case study (LIHC / LUAD / any)
    r_scripts/                  exact-paper R routes (DESeq2, clusterProfiler)

Each module is an independent CLI -- run with -h for usage. None of them
modify the core package; they only IMPORT from it.
"""
