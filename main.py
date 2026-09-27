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
        --epochs 200 --lr 0.01 --low_dim 20 --out results/paad_demo

For your own cancer, first run preprocess/preprocess_tcga.py to create the
h5ad + edge files, then point the two --input_* flags at them.

Hyper-parameters follow the paper's Supporting Information (gdnet/hparams.py).
This script has no survival data, so it cannot run the SI's 5-fold selection
of --low_dim / --lr; it trains ONE configuration (default: low_dim 20,
lr 1e-2). Use run_pipeline.py to run the 5-fold selection.
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

from gdnet import hparams as HP
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
    # ---- optimisation (SI: max 200 epochs, LR from {1e-2..1e-5}) -----------
    p.add_argument("--epochs", type=int, default=HP.EPOCHS)
    p.add_argument("-b", "--batch_size", type=int, default=HP.BATCH_SIZE,
                   help="if larger than the cohort, the whole cohort is one batch")
    p.add_argument("--lr", type=float, default=HP.DEFAULT_LR,
                   help=f"learning rate; SI grid {list(HP.LR_GRID)}")
    p.add_argument("--momentum", type=float, default=HP.SGD_MOMENTUM,
                   help="SGD optimiser momentum (not in the SI; official repo value)")
    p.add_argument("--wd", type=float, default=HP.WEIGHT_DECAY,
                   help="weight decay = the L2 regularisation of paper Eq. 4")
    p.add_argument("--cos", action="store_true", default=True,
                   help="cosine learning-rate schedule (official flag)")
    # ---- model (paper Section 2.1.3 + SI hyper-parameters) ----------------
    p.add_argument("--low_dim", type=int, default=HP.DEFAULT_LOW_DIM,
                   help=f"middle-layer / embedding size; SI grid {list(HP.LOW_DIM_GRID)}")
    p.add_argument("--hidden1", type=int, default=HP.HIDDEN1,
                   help="hidden layer 1 size (SI: 1024)")
    p.add_argument("--hidden2", type=int, default=HP.HIDDEN2,
                   help="hidden layer 2 size (SI: 128)")
    p.add_argument("--moco_r", type=int, default=HP.QUEUE_SIZE,
                   help="negative-key queue size (auto-shrunk to fit the batch)")
    p.add_argument("--moco_m", type=float, default=HP.MOCO_MOMENTUM,
                   help="momentum coefficient of the key encoder, Eq. 5 (SI: 0.99)")
    p.add_argument("--temperature", type=float, default=HP.TEMPERATURE,
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


def train_encoder(X, edge_index, args, device, log=True):
    """Contrastive training loop; returns the trained MoCo model.

    `args` needs: epochs, batch_size, lr, momentum, wd, cos, low_dim, moco_r,
    moco_m, temperature, out (hidden1/hidden2 are optional, default = SI).
    With log=True the per-10-epoch log is written to <out>/train_log.csv.
    """
    n_samples, n_features = X.shape
    batch = min(args.batch_size, n_samples)

    # normalised KEGG adjacency (paper Eq. 2) -- computed once, reused every step
    adj_hat = normalized_adjacency(edge_index, n_features).to(device)

    model = MoCo(adj_hat, num_genes=n_features, dim=args.low_dim, r=args.moco_r,
                 m=args.moco_m, T=args.temperature, batch_size=batch,
                 device=device,
                 hidden1=getattr(args, "hidden1", HP.HIDDEN1),
                 hidden2=getattr(args, "hidden2", HP.HIDDEN2)).to(device)

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
            if log:
                print(f"epoch {epoch:4d}  loss {epoch_loss / n_batches:.4f}  "
                      f"acc {epoch_acc / n_batches:.1f}%")
            log_rows.append({"epoch": epoch, "loss": epoch_loss / n_batches,
                             "accuracy": epoch_acc / n_batches})

    model.train_log_ = pd.DataFrame(log_rows)   # kept for callers (e.g. tuning)
    if log and getattr(args, "out", None):
        model.train_log_.to_csv(os.path.join(args.out, "train_log.csv"), index=False)
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
