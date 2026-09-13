#!/usr/bin/env Rscript
# Exact-paper DESeq2 differential expression (paper Sections 2.3-2.4).
#
# The paper: DEGs between GD-Net's high/low risk groups with
# |log2(fold change)| > 1.6 and adjusted p < 0.05 (LIHC: 579 DEGs).
#
# Usage:
#   Rscript extensions/r_scripts/run_deseq2.R <counts.tsv> <risk_groups.csv> <outdir>
#
#   counts.tsv       raw mRNA counts, rows = genes, columns = TCGA sample IDs
#                    (e.g. TCGA-LIHC.star_counts.tsv from the Xena GDC hub;
#                     if values are log2(count+1) they are back-transformed)
#   risk_groups.csv  output of run_pipeline.py (columns: sample, group, ...)
#
# Install dependencies once:
#   if (!require("BiocManager")) install.packages("BiocManager")
#   BiocManager::install("DESeq2")

suppressMessages(library(DESeq2))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("usage: run_deseq2.R <counts.tsv> <risk_groups.csv> <outdir>")
counts_path <- args[1]; risk_path <- args[2]; outdir <- args[3]
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

counts <- read.delim(counts_path, row.names = 1, check.names = FALSE)
colnames(counts) <- substr(colnames(counts), 1, 15)   # 15-char TCGA barcodes

# Xena star_counts are log2(count+1): detect and back-transform to integers
if (any(counts[1:100, ] %% 1 != 0, na.rm = TRUE)) {
  message("back-transforming log2(count+1) -> raw counts")
  counts <- round(2^counts - 1)
}

risk <- read.csv(risk_path)
common <- intersect(risk$sample, colnames(counts))
message(length(common), " samples shared")
counts <- counts[, common]
condition <- factor(risk$group[match(common, risk$sample)], levels = c("low", "high"))

dds <- DESeqDataSetFromMatrix(countData = as.matrix(counts),
                              colData = data.frame(condition = condition),
                              design = ~ condition)
dds <- dds[rowSums(counts(dds)) >= 10, ]              # standard low-count filter
dds <- DESeq(dds)
res <- results(dds, contrast = c("condition", "high", "low"))
res <- as.data.frame(res)
res$feature <- rownames(res)
write.csv(res, file.path(outdir, "deseq2_results.csv"), row.names = FALSE)

# paper thresholds
degs <- subset(res, abs(log2FoldChange) > 1.6 & !is.na(padj) & padj < 0.05)
degs <- degs[order(degs$padj), ]
write.csv(degs, file.path(outdir, "deseq2_degs.csv"), row.names = FALSE)
message(nrow(degs), " DEGs (paper LIHC: 579; ",
        sum(degs$log2FoldChange > 0), " up-regulated in high risk)")
