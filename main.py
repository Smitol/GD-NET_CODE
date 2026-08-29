#!/usr/bin/env python
"""GD-Net Stage 1: train the GCN + contrastive encoder and export embeddings.

This is the `main.py` that the official repo's README references but does not
ship. It reproduces paper Section 2.1.2-2.1.3 (Fig. 1B):

    fused multi-omics matrix + KEGG feature graph
        -> two augmented views per patient (positive pairs)
        -> MoCo momentum-contrast training of the GCN+MLP encoder (InfoNCE loss)
        -> export the low-dimensional embeddings ("multi-omics meta-features")

The embeddings are then consumed by run_pipeline.py (Cox-EN risk model,
XGBoost feature selection).

Example (works out of the box with the bundled demo data):
    python main.py \
        --input_h5ad_path data/paad_demo/paad.h5ad \
        --input_edge_path data/paad_demo/paad_edges.csv \
        --epochs 100 --lr 0.01 --low_dim 200 --out results/paad_demo

For your own cancer, first run preprocess/preprocess_tcga.py to create the
h5ad + edge files, then point the two --input_* flags at them.
"""

import argparse
import math
import os
import time as time_mod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gdnet.builder import MoCo, normalized_adjacency
from gdnet.loader import MultiOmicsDataset
from gdnet.utils import ensure_dir, load_edges, load_h5ad, seed_everything


def get_args():
    p = argparse.ArgumentParser(description="GD-Net contrastive encoder training")
    # ---- inputs -----------------------------------------------------------
    p.add_argument("--input_h5ad_path", required=True,
                   help="fused multi-omics AnnData (patients x features)")
    p.add_argument("--input_edge_path", required=True,
                   help="KEGG edge list csv: two integer columns of feature indices, no header")
    p.add_argument("--out", default="results/run", help="output directory")
    # ---- optimisation (official README defaults: epochs 200, lr 0.01) -----
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("-b", "--batch_size", type=int, default=512,
                   help="if larger than the cohort, the whole cohort is one batch")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    p.add_argument("--wd", type=float, default=1e-6,
                   help="weight decay = the L2 regularisation of paper Eq. 4")
    p.add_argument("--cos", action="store_true", default=True,
                   help="cosine learning-rate schedule (official flag)")
    # ---- model (paper Section 2.1.3) --------------------------------------
    p.add_argument("--low_dim", type=int, default=200,
                   help="embedding dimension (official README: 200)")
    p.add_argument("--moco_r", type=int, default=512,
                   help="negative-key queue size (auto-shrunk to fit the batch)")
    p.add_argument("--moco_m", type=float, default=0.999,
                   help="key-encoder momentum lambda of Eq. 5")
    p.add_argument("--temperature", type=float, default=0.2,
                   help="tau of the contrastive loss Eq. 3")
    # ---- misc --------------------------------------------------------------
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=None,
                   help="GPU id; omit to run on CPU (this re-implementation is CPU-friendly)")
    return p.parse_args()


def adjust_lr(optimizer, epoch, args):
    """Cosine learning-rate decay over the full training run."""
    lr = args.lr
    if args.cos:
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    for g in optimizer.param_groups:
        g["lr"] = lr


def train_encoder(X, edge_index, args, device):
    """Contrastive training loop; returns the trained MoCo model."""
    n_samples, n_features = X.shape
    batch = min(args.batch_size, n_samples)

    # normalised KEGG adjacency (paper Eq. 2) -- computed once, reused every step
    adj_hat = normalized_adjacency(edge_index, n_features).to(device)

    model = MoCo(adj_hat, num_genes=n_features, dim=args.low_dim, r=args.moco_r,
                 m=args.moco_m, T=args.temperature, batch_size=batch,
                 device=device).to(device)

    dataset = MultiOmicsDataset(X, transform=True)
    # drop_last so every batch has the same size (keeps the MoCo queue simple);
    # with batch >= n_samples nothing is dropped.
    loader = DataLoader(dataset, batch_size=batch, shuffle=True,
                        drop_last=(n_samples % batch != 0), num_workers=0)

    criterion = nn.CrossEntropyLoss()  # InfoNCE == CE over (positive|negatives) logits
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.wd)

    log_rows = []
    for epoch in range(args.epochs):
        adjust_lr(optimizer, epoch, args)
        model.train()
        epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0
        for (views, _idx) in loader:
            im_q, im_k = views[0].to(device), views[1].to(device)
            logits, labels = model(im_q, im_k)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            # "accuracy" = fraction of samples whose positive pair scored highest
            epoch_acc += (logits.argmax(1) == labels).float().mean().item() * 100
            n_batches += 1

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d}  loss {epoch_loss / n_batches:.4f}  "
                  f"acc {epoch_acc / n_batches:.1f}%")
            log_rows.append({"epoch": epoch, "loss": epoch_loss / n_batches,
                             "accuracy": epoch_acc / n_batches})

    pd.DataFrame(log_rows).to_csv(os.path.join(args.out, "train_log.csv"), index=False)
    return model


@torch.no_grad()
def extract_embeddings(model, X, device, batch=256):
    """Run every patient (UNaugmented) through the key encoder -> embeddings."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        out.append(model(xb, is_eval=True).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    args = get_args()
    seed_everything(args.seed)
    ensure_dir(args.out)

    device = f"cuda:{args.gpu}" if (args.gpu is not None and torch.cuda.is_available()) else "cpu"
    print(f"device = {device}")

    # ---- load inputs -------------------------------------------------------
    X, var_names, obs_names = load_h5ad(args.input_h5ad_path)
    edge_index = load_edges(args.input_edge_path)
    print(f"data: {X.shape[0]} patients x {X.shape[1]} features, "
          f"{edge_index.shape[1]} KEGG edges")

    # z-score each feature across patients: contrastive learning + GCN work
    # much better on standardised inputs (raw log2 scales differ per modality).
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    X = (X - mu) / np.maximum(sd, 1e-8)

    # ---- train + export ----------------------------------------------------
    t0 = time_mod.time()
    model = train_encoder(X, edge_index, args, device)
    print(f"training took {time_mod.time() - t0:.1f}s")

    emb = extract_embeddings(model, X, device)
    emb_df = pd.DataFrame(emb, index=obs_names,
                          columns=[f"z{i}" for i in range(emb.shape[1])])
    emb_df.to_csv(os.path.join(args.out, "embeddings.csv"))
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    print(f"saved embeddings {emb.shape} -> {args.out}/embeddings.csv")


if __name__ == "__main__":
    main()
