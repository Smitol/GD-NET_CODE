"""Shared helpers: loading the preprocessed inputs, seeding, modality tagging."""

import os
import random

import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int = 42):
    """Fix all RNG seeds so runs are repeatable (results will still differ
    slightly across machines/library versions -- that is expected)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_h5ad(path: str):
    """Load the fused multi-omics AnnData file produced by preprocessing.

    Returns:
        X          : np.ndarray (n_patients, n_features)
        var_names  : list of feature names (e.g. 'TP53_geneExp', 'TP53_methylation',
                     'hsa-mir-21_miRNAExp')
        obs_names  : list of patient/sample IDs
    """
    import anndata
    adata = anndata.read_h5ad(path)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    return X.astype(np.float32), list(adata.var_names), list(adata.obs_names)


def load_edges(path: str) -> torch.Tensor:
    """Load the KEGG edge list (two integer columns = feature indices, no header)."""
    e = pd.read_csv(path, header=None).values
    return torch.as_tensor(e.T, dtype=torch.long)  # shape (2, E)


def load_survival(path: str) -> pd.DataFrame:
    """Load survival info and return a tidy DataFrame indexed by sample ID with
    columns ['time', 'status'] (status: 1 = death observed, 0 = censored).

    Two accepted formats:
      A) the authors'/our transposed format: rows = GeneSymbol/time/status,
         columns = samples (this is what preprocessing writes);
      B) a tidy TSV/CSV with columns sample,time,status (any order).
    """
    df = pd.read_csv(path, index_col=0)
    if "time" in df.index and "status" in df.index:          # format A (transposed)
        out = df.loc[["time", "status"]].T
        out.columns = ["time", "status"]
    else:                                                    # format B (tidy)
        df.columns = [c.lower() for c in df.columns]
        out = df[["time", "status"]]
    out = out.astype(float)
    # Cox models cannot use non-positive follow-up times; clip 0 -> 1 day
    out.loc[out["time"] <= 0, "time"] = 1.0
    return out


def modality_of(feature_name: str) -> str:
    """Map a fused-matrix feature name to its omics modality (by suffix).

    NOTE: if you change the suffixes in preprocessing, change them here too.
    """
    if feature_name.endswith("_geneExp"):
        return "mRNA"
    if feature_name.endswith("_methylation"):
        return "methylation"
    if "miRNA" in feature_name or feature_name.startswith("hsa-"):
        return "miRNA"
    return "unknown"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path
