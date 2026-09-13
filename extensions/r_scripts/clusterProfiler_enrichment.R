#!/usr/bin/env Rscript
# Exact-paper GO + KEGG enrichment with clusterProfiler (paper Section 2.5,
# Figure 3F: "cell cycle, complement and coagulation, fatty acid degradation"
# KEGG pathways; "chromosomal region, spindle pole" GO terms; FDR < 0.05).
#
# Usage:
#   Rscript extensions/r_scripts/clusterProfiler_enrichment.R <genes.txt> <outdir>
#
#   genes.txt   one gene per line -- use results/<CANCER>/key_ifms.txt or
#               ifms.txt from run_pipeline.py (feature suffixes are stripped)
#
# Install once:
#   BiocManager::install(c("clusterProfiler", "org.Hs.eg.db", "enrichplot"))

suppressMessages({
  library(clusterProfiler)
  library(org.Hs.eg.db)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: clusterProfiler_enrichment.R <genes.txt> <outdir>")
genes_path <- args[1]; outdir <- args[2]
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# read + strip the fused-matrix suffixes, drop miRNAs (not in org.Hs.eg.db)
raw <- readLines(genes_path)
sym <- unique(gsub("_(geneExp|methylation|miRNAExp)$", "", raw))
sym <- sym[!grepl("^hsa-", sym)]
message(length(sym), " gene symbols")

eg <- bitr(sym, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)
message(nrow(eg), " mapped to Entrez IDs")

# GO enrichment: all three ontologies, FDR (BH) < 0.05 like the paper
ego <- enrichGO(gene = eg$ENTREZID, OrgDb = org.Hs.eg.db, ont = "ALL",
                pAdjustMethod = "BH", qvalueCutoff = 0.05, readable = TRUE)
write.csv(as.data.frame(ego), file.path(outdir, "go_enrichment.csv"), row.names = FALSE)

# KEGG enrichment
ekegg <- enrichKEGG(gene = eg$ENTREZID, organism = "hsa",
                    pAdjustMethod = "BH", qvalueCutoff = 0.05)
write.csv(as.data.frame(ekegg), file.path(outdir, "kegg_enrichment.csv"), row.names = FALSE)

# Fig 3F style bar plots
tryCatch({
  library(enrichplot)
  ggplot2::ggsave(file.path(outdir, "go_barplot.png"),
                  barplot(ego, showCategory = 15), width = 9, height = 6, dpi = 150)
  ggplot2::ggsave(file.path(outdir, "kegg_barplot.png"),
                  barplot(ekegg, showCategory = 15), width = 9, height = 6, dpi = 150)
}, error = function(e) message("plots skipped: ", conditionMessage(e)))

message("done -> ", outdir)
