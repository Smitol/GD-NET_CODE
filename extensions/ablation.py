#!/usr/bin/env python
"""Single-modality ablation experiment: paper Figure 2C (Section 3.3).

"When using each single-modality of multi-omics data as input of GD-Net ...
 the mRNA data achieved the highest average CI value of 0.64 ... the miRNA
 got the lowest one of 0.589 ... and the methylation data (average CI=0.603)
 performed better than miRNA."

For each modality (mRNA / miRNA / methylation) AND the full fusion this
script: slices the fused matrix by the feature-name suffix, remaps the KEGG
edge list onto the kept feature indices, trains the SAME GD-Net encoder, and
evaluates Cox-EN with 5-fold CV. Output: ablation_cindex.csv + bar plot.

Example:
    python -m extensions.ablation \
        --input_h5ad_path data/paad_demo/paad.h5ad \
        --input_edge_path data/paad_demo/paad_edges.csv \
        --surv_path       data/paad_demo/paad_surv.csv \
        --epochs 100 --out results/paad_demo/ablation

NOTE: this trains the encoder 4x, so it takes ~4x one run_pipeline training.
Lower --epochs for a quick look.
"""

import argparse
import os
import types

import numpy as np
import pandas as pd
import torch

from gdnet.cox_en import cross_validate
from gdnet.utils import (ensure_dir, load_edges, load_h5ad, load_survival,
                         modality_of, seed_everything)
from main import extract_embeddings, train_encoder


def slice_modality(X, var_names, edge_index, modality):
    """Subset the fused matrix (and remap edges) to ONE modality.

    modality: 'mRNA' | 'miRNA' | 'methylation' | 'fusion' (= keep everything)
    Edges whose endpoints are not both kept are dropped (miRNA features have
    no KEGG edges anyway -- the graph is gene-based).
    """
    if modality == "fusion":
        return X, list(range(len(var_names))), edge_index
    keep = [i for i, v in enumerate(var_names) if modality_of(v) == modality]
    remap = -np.ones(len(var_names), dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    e = edge_index.numpy()
    mask = (remap[e[0]] >= 0) & (remap[e[1]] >= 0)
    new_edges = torch.as_tensor(np.stack([remap[e[0][mask]], remap[e[1][mask]]]),
                                dtype=torch.long)
    return X[:, keep], keep, new_edges


def main():
    ap = argparse.ArgumentParser(description="GD-Net single-modality ablation (Fig 2C)")
    ap.add_argument("--input_h5ad_path", required=True)
    ap.add_argument("--input_edge_path", required=True)
    ap.add_argument("--surv_path", required=True)
    ap.add_argument("--out", default="results/ablation")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    # forwarded encoder hyper-parameters (same defaults as run_pipeline.py)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--low_dim", type=int, default=200)
    args = ap.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.out)

    X, var_names, obs_names = load_h5ad(args.input_h5ad_path)
    edge_index = load_edges(args.input_edge_path)
    surv = load_survival(args.surv_path)
    keep = [i for i, s in enumerate(obs_names) if s in surv.index]
    X, obs_names = X[keep], [obs_names[i] for i in keep]
    surv = surv.loc[obs_names]
    time, status = surv["time"].to_numpy(float), surv["status"].to_numpy(float)

    rows = []
    for modality in ("mRNA", "miRNA", "methylation", "fusion"):
        Xm, kept, em = slice_modality(X, var_names, edge_index, modality)
        if Xm.shape[1] == 0:
            print(f"{modality}: no features -- skipped")
            continue
        print(f"\n===== {modality}: {Xm.shape[1]} features, "
              f"{em.shape[1]} edges =====")

        # z-score then train the SAME encoder as the main pipeline.
        mu, sd = Xm.mean(0, keepdims=True), Xm.std(0, keepdims=True)
        Xz = (Xm - mu) / np.maximum(sd, 1e-8)
        # train_encoder() reads hyper-params from an argparse-like namespace:
        targs = types.SimpleNamespace(
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            momentum=0.9, wd=1e-6, cos=True, low_dim=args.low_dim,
            moco_r=512, moco_m=0.999, temperature=0.2, out=args.out)
        model = train_encoder(Xz, em, targs, "cpu")
        emb = extract_embeddings(model, Xz, "cpu")

        cv = cross_validate(emb, time, status, n_splits=5, seed=args.seed)
        rows.append({"modality": modality,
                     "c_index_mean": cv.c_index.mean(),
                     "c_index_std": cv.c_index.std(),
                     "n_features": Xm.shape[1]})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "ablation_cindex.csv"), index=False)
    print("\n", res.to_string(index=False))

    try:  # Fig 2C style bar plot
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(res.modality, res.c_index_mean, yerr=res.c_index_std,
               capsize=4, color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"])
        ax.set_ylabel("C-index (5-fold CV)")
        ax.set_title("Single-modality ablation (paper Fig 2C)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "ablation_bar.png"), dpi=150)
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
