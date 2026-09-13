#!/usr/bin/env python
"""Runtime / hardware scaling analysis: paper Figure S3 (Section 3.3).

"The linear correlation between sample numbers and computational time
 indicates that the GD-Net model can handle increasing data volumes without
 a significant rise in computational demands."

Trains the GD-Net encoder on growing random subsets of the cohort, records
wall-clock training time, fits a linear regression, and plots time-vs-n.

Example (fast: 10 epochs per size):
    python -m extensions.runtime_scaling \
        --input_h5ad_path data/paad_demo/paad.h5ad \
        --input_edge_path data/paad_demo/paad_edges.csv \
        --epochs 10 --out results/paad_demo/scaling
"""

import argparse
import os
import time as time_mod
import types

import numpy as np
import pandas as pd
from scipy import stats

from gdnet.utils import ensure_dir, load_edges, load_h5ad, seed_everything
from main import train_encoder


def main():
    ap = argparse.ArgumentParser(description="runtime scaling (paper Fig S3)")
    ap.add_argument("--input_h5ad_path", required=True)
    ap.add_argument("--input_edge_path", required=True)
    ap.add_argument("--out", default="results/scaling")
    ap.add_argument("--epochs", type=int, default=10,
                    help="epochs per size point (few are enough: time/epoch is constant)")
    ap.add_argument("--sizes", default=None,
                    help="comma-separated sample sizes; default = 5 even steps")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.out)
    rng = np.random.default_rng(args.seed)

    X, _, _ = load_h5ad(args.input_h5ad_path)
    edge_index = load_edges(args.input_edge_path)
    n = X.shape[0]
    sizes = ([int(s) for s in args.sizes.split(",")] if args.sizes
             else np.linspace(max(30, n // 5), n, 5, dtype=int).tolist())

    rows = []
    for m in sizes:
        sub = rng.choice(n, size=m, replace=False)
        Xs = X[sub]
        Xs = (Xs - Xs.mean(0)) / np.maximum(Xs.std(0), 1e-8)
        targs = types.SimpleNamespace(epochs=args.epochs, batch_size=512,
                                      lr=0.01, momentum=0.9, wd=1e-6, cos=True,
                                      low_dim=200, moco_r=512, moco_m=0.999,
                                      temperature=0.2, out=args.out)
        t0 = time_mod.time()
        train_encoder(Xs, edge_index, targs, "cpu")
        dt = time_mod.time() - t0
        rows.append({"n_samples": m, "seconds": dt,
                     "sec_per_epoch": dt / args.epochs})
        print(f"n={m:5d}  {dt:.1f}s total  {dt / args.epochs:.2f}s/epoch")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "runtime_scaling.csv"), index=False)

    # linear fit: time = a*n + b (the paper's claim is linearity, R^2 near 1)
    lr = stats.linregress(df.n_samples, df.seconds)
    print(f"linear fit: time = {lr.slope:.4f}*n + {lr.intercept:.2f}  "
          f"(R^2 = {lr.rvalue ** 2:.3f})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.scatter(df.n_samples, df.seconds, c="#4c72b0")
        xs = np.array([df.n_samples.min(), df.n_samples.max()])
        ax.plot(xs, lr.slope * xs + lr.intercept, "r--",
                label=f"linear fit (R²={lr.rvalue ** 2:.3f})")
        ax.set_xlabel("number of samples")
        ax.set_ylabel(f"training time for {args.epochs} epochs (s)")
        ax.set_title("Runtime scaling (paper Fig S3)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "runtime_scaling.png"), dpi=150)
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
