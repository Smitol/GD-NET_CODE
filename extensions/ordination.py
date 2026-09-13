#!/usr/bin/env python
"""Sample clustering & ordination: paper Figures 2D/2E and 3D (Sections 2.4, 3.3).

Implements everything the paper uses to show that GD-Net-selected features
separate the predicted risk groups:

  * PCoA  (principal coordinate analysis = classical metric MDS on a
          distance matrix) with % variance explained per axis  -> Fig 3D
  * PERMANOVA (permutational multivariate ANOVA) significance of the
          risk-group separation in the distance matrix           -> Fig 3D
  * t-SNE 2-D visualisation of the samples                        -> Fig 2D
  * KMeans(2) clustering + Adjusted Rand Index vs the predicted risk
          labels (paper LIHC: ARI = 0.56), including the paper's
          "shuffle group" baseline = random features of same count -> Fig 2D/2E

Inputs are the outputs of run_pipeline.py.

Example (whole feature set + XGBoost top-200 + key-IFMs):
    python -m extensions.ordination \
        --input_h5ad_path data/paad_demo/paad.h5ad \
        --risk_groups results/paad_demo/risk_groups.csv \
        --features   results/paad_demo/xgboost_top200.csv \
        --out results/paad_demo/ordination
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from gdnet.utils import ensure_dir, load_h5ad, modality_of, seed_everything


# ---------------------------------------------------------------------------
# PCoA: double-centre the squared distance matrix, eigendecompose.
# (classical Torgerson MDS; equivalent to R's cmdscale / ape::pcoa)
# ---------------------------------------------------------------------------
def pcoa(D: np.ndarray, n_axes: int = 2):
    """Returns (coordinates (n, n_axes), pct variance explained per axis)."""
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n          # centring matrix
    B = -0.5 * J @ (D ** 2) @ J                  # Gower's double centring
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]              # eigh returns ascending
    evals, evecs = evals[order], evecs[:, order]
    pos = np.maximum(evals, 0)                   # negative eigenvalues -> 0
    coords = evecs[:, :n_axes] * np.sqrt(pos[:n_axes])
    pct = 100 * pos / pos.sum()
    return coords, pct[:n_axes]


# ---------------------------------------------------------------------------
# PERMANOVA (Anderson 2001): pseudo-F of between/within group distances,
# p-value by permuting group labels. Same test as R vegan::adonis2.
# ---------------------------------------------------------------------------
def permanova(D: np.ndarray, groups: np.ndarray, n_perm: int = 999, seed: int = 42):
    """Returns dict(pseudo_F, p_value, n_perm)."""
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    D2 = D ** 2

    def pseudo_f(g):
        sst = D2[np.triu_indices(n, 1)].sum() / n
        ssw = 0.0
        for lvl in np.unique(g):
            idx = np.where(g == lvl)[0]
            if len(idx) > 1:
                sub = D2[np.ix_(idx, idx)]
                ssw += sub[np.triu_indices(len(idx), 1)].sum() / len(idx)
        ssa = sst - ssw
        a = len(np.unique(g))
        return (ssa / (a - 1)) / (ssw / (n - a))

    f_obs = pseudo_f(groups)
    perm_f = np.array([pseudo_f(rng.permutation(groups)) for _ in range(n_perm)])
    p = (1 + (perm_f >= f_obs).sum()) / (n_perm + 1)
    return {"pseudo_F": f_obs, "p_value": p, "n_perm": n_perm}


# ---------------------------------------------------------------------------
# clustering quality: KMeans(2) vs predicted risk labels, ARI + shuffle baseline
# ---------------------------------------------------------------------------
def clustering_ari(X_sel: np.ndarray, X_all: np.ndarray, labels: np.ndarray,
                   seed: int = 42, n_baseline: int = 10):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)

    km = KMeans(2, n_init=10, random_state=seed).fit(X_sel)
    ari = adjusted_rand_score(labels, km.labels_)

    # paper baseline ("shuffle group"): random feature subsets of same size
    base = []
    for _ in range(n_baseline):
        idx = rng.choice(X_all.shape[1], size=X_sel.shape[1], replace=False)
        kb = KMeans(2, n_init=10, random_state=seed).fit(X_all[:, idx])
        base.append(adjusted_rand_score(labels, kb.labels_))
    return ari, float(np.mean(base)), float(np.std(base))


def read_feature_list(path: str) -> list:
    """Accepts xgboost_top200.csv / differential_features.csv (has a 'feature'
    column) or a plain one-name-per-line txt (ifms.txt / key_ifms.txt)."""
    if path.endswith(".csv"):
        return pd.read_csv(path)["feature"].tolist()
    with open(path) as fh:
        return [l.strip() for l in fh if l.strip()]


def main():
    ap = argparse.ArgumentParser(description="PCoA/PERMANOVA/t-SNE/ARI (Fig 2D-E, 3D)")
    ap.add_argument("--input_h5ad_path", required=True)
    ap.add_argument("--risk_groups", required=True,
                    help="risk_groups.csv from run_pipeline.py")
    ap.add_argument("--features", required=True,
                    help="feature subset: xgboost_top200.csv | ifms.txt | key_ifms.txt")
    ap.add_argument("--metric", default="euclidean",
                    help="distance metric for PCoA/PERMANOVA "
                         "(euclidean | braycurtis | correlation ...)")
    ap.add_argument("--out", default="results/ordination")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.out)

    X, var_names, obs_names = load_h5ad(args.input_h5ad_path)
    rg = pd.read_csv(args.risk_groups).set_index("sample").reindex(obs_names)
    labels = (rg["group"] == "high").to_numpy(int)

    feats = read_feature_list(args.features)
    name_to_idx = {v: i for i, v in enumerate(var_names)}
    sel = [name_to_idx[f] for f in feats if f in name_to_idx]
    print(f"{len(sel)}/{len(feats)} selected features found in the matrix")
    X_sel = X[:, sel]
    # z-score so no single feature dominates the distances
    X_sel = (X_sel - X_sel.mean(0)) / np.maximum(X_sel.std(0), 1e-8)

    stats_lines = []

    # ---------------- per-modality + overall PCoA + PERMANOVA (Fig 3D) ----
    sel_names = [var_names[i] for i in sel]
    groups = {"all": np.arange(X_sel.shape[1])}
    for m in ("mRNA", "miRNA", "methylation"):
        idx = [j for j, f in enumerate(sel_names) if modality_of(f) == m]
        if len(idx) >= 3:
            groups[m] = np.array(idx)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4.5))
    axes = np.atleast_1d(axes)

    for ax, (gname, gidx) in zip(axes, groups.items()):
        D = squareform(pdist(X_sel[:, gidx], metric=args.metric))
        coords, pct = pcoa(D)
        pm = permanova(D, labels, seed=args.seed)
        line = (f"PCoA[{gname}] axis1={pct[0]:.2f}% axis2={pct[1]:.2f}%  "
                f"PERMANOVA F={pm['pseudo_F']:.2f} p={pm['p_value']:.4f}")
        print(line)
        stats_lines.append(line)
        for lab, color, txt in [(1, "#d62728", "high risk"), (0, "#1f77b4", "low risk")]:
            m = labels == lab
            ax.scatter(coords[m, 0], coords[m, 1], s=14, c=color, label=txt, alpha=0.75)
        ax.set_xlabel(f"PCoA1 ({pct[0]:.1f}%)")
        ax.set_ylabel(f"PCoA2 ({pct[1]:.1f}%)")
        ax.set_title(f"{gname}  (PERMANOVA p={pm['p_value']:.3g})")
        ax.legend(fontsize=8)
    fig.suptitle("PCoA of informative features (paper Fig 3D)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "pcoa.png"), dpi=150)

    # ------------------------------------------------ t-SNE plot (Fig 2D) --
    from sklearn.manifold import TSNE
    ts = TSNE(2, random_state=args.seed,
              perplexity=min(30, X.shape[0] // 4)).fit_transform(X_sel)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for lab, color, txt in [(1, "#d62728", "high risk"), (0, "#1f77b4", "low risk")]:
        m = labels == lab
        ax.scatter(ts[m, 0], ts[m, 1], s=14, c=color, label=txt, alpha=0.75)
    ax.set_title("t-SNE on selected features (paper Fig 2D)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "tsne.png"), dpi=150)

    # ------------------------------- KMeans + ARI + shuffle baseline (2D/2E)
    ari, base_mean, base_std = clustering_ari(X_sel, X, labels, seed=args.seed)
    line = (f"KMeans(2) ARI vs risk labels = {ari:.3f}   "
            f"shuffle-baseline ARI = {base_mean:.3f} (+/- {base_std:.3f})   "
            f"[paper LIHC key features: ARI = 0.56]")
    print(line)
    stats_lines.append(line)

    with open(os.path.join(args.out, "ordination_stats.txt"), "w") as fh:
        fh.write("\n".join(stats_lines) + "\n")
    print(f"outputs in {args.out}/")


if __name__ == "__main__":
    main()
