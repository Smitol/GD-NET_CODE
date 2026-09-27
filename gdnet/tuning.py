"""5-fold hyper-parameter selection from the paper's Supporting Information.

    "The node number in middle hidden layer was chosen from the set
     [10, 20, 50]. The learning rate (LR) was chosen from the range
     [1e-2, 1e-3, 1e-4, 1e-5] ... The parameters were selected based on the
     5-fold results in the experiments."

For every (low_dim, lr) combination the contrastive encoder is trained
(unsupervised, no survival labels involved), the patients are embedded, and
the embeddings are scored by the same 5-fold Cox-EN cross-validation that
produces the reported C-index (gdnet/cox_en.cross_validate, identical folds
for every candidate). The combination with the highest mean C-index wins.

Caveat: selecting on the same 5-fold C-index that is then reported is what
the SI describes, but it makes the reported number slightly optimistic
compared with a nested (outer/inner) cross-validation.
"""

import copy
import os

import numpy as np
import pandas as pd

from . import hparams as HP
from .cox_en import cross_validate
from .utils import seed_everything


def select_hyperparameters(Xz, edge_index, time, status, args, device,
                           low_dim_grid=HP.LOW_DIM_GRID, lr_grid=HP.LR_GRID,
                           n_splits=HP.N_FOLDS):
    """Grid-search low_dim x lr by mean 5-fold C-index.

    Args:
        Xz:          z-scored (patients x features) matrix fed to the encoder.
        edge_index:  KEGG edges (2, E).
        time/status: survival arrays aligned with Xz rows.
        args:        namespace with all other train_encoder() settings plus
                     penalizer, l1_ratio, seed, out.
        device:      'cpu' | 'cuda:N'.

    Returns dict with keys:
        low_dim, lr     : selected values
        model           : trained MoCo model of the selected configuration
        embeddings      : its (patients x low_dim) embeddings
        cv              : its per-fold CV DataFrame
        table           : DataFrame of every candidate's mean/std C-index
    """
    # imported here to avoid a circular import (main.py imports gdnet.*)
    from main import extract_embeddings, train_encoder

    rows, best = [], None
    n_cand = len(low_dim_grid) * len(lr_grid)
    k = 0
    for low_dim in low_dim_grid:
        for lr in lr_grid:
            k += 1
            print(f"\n[search {k}/{n_cand}] low_dim={low_dim}  lr={lr:g}")
            cand = copy.copy(args)
            cand.low_dim, cand.lr = int(low_dim), float(lr)
            # same seed for every candidate -> differences come from the
            # hyper-parameters, not from the random initialisation/augmentation
            seed_everything(args.seed)
            model = train_encoder(Xz, edge_index, cand, device, log=False)
            emb = extract_embeddings(model, Xz, device)
            try:
                cv = cross_validate(emb, time, status, n_splits=n_splits,
                                    penalizer=args.penalizer,
                                    l1_ratio=args.l1_ratio, seed=args.seed)
                ci_mean, ci_std = float(cv.c_index.mean()), float(cv.c_index.std())
            except Exception as e:           # e.g. Cox-EN never converged
                print(f"  candidate failed: {e}")
                cv, ci_mean, ci_std = None, np.nan, np.nan
            rows.append({"low_dim": int(low_dim), "lr": float(lr),
                         "c_index_mean": ci_mean, "c_index_std": ci_std})
            if cv is not None and (best is None or ci_mean > best["c_index_mean"]):
                best = {"low_dim": int(low_dim), "lr": float(lr),
                        "c_index_mean": ci_mean, "model": model,
                        "embeddings": emb, "cv": cv}

    table = pd.DataFrame(rows).sort_values("c_index_mean", ascending=False,
                                           na_position="last")
    if best is None:
        raise RuntimeError("hyper-parameter search: every candidate failed")
    print("\n--- hyper-parameter search (5-fold mean C-index) ---")
    print(table.to_string(index=False))
    print(f"selected: low_dim={best['low_dim']}  lr={best['lr']:g}  "
          f"(C-index {best['c_index_mean']:.3f})")
    if getattr(args, "out", None):
        table.to_csv(os.path.join(args.out, "hparam_search.csv"), index=False)
    best["table"] = table
    return best
