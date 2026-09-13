"""The six benchmark survival methods of paper Table 1 / Figure 2A (Sec 2.2).

Paper: "We evaluated ... by comparison with three typical methods (the
PCA-Cox model [31], eXGBC and the bridge regularisation-based Cox method
(Bridge-Cox) [32]) and three DL-based methods (Deep_surv [23], DCAP [15]
and MTC [33])."

Every method implements the same interface so the CV harness in
run_benchmarks.py can treat them identically:

    risk_test = METHODS[name](X_train, time_train, status_train, X_test, seed)

where X is the ORIGINAL fused multi-omics matrix (patients x features) and
the return value is a 1-D risk score for the test patients
(HIGHER risk = SHORTER expected survival).

IMPLEMENTATION HONESTY -- the original papers do not all ship code, and the
GD-Net paper gives no benchmark hyper-parameters (they are in unavailable
supplementary material). Each implementation below is a faithful,
*documented* re-creation of the method's published idea; expect results in
the ballpark of Table 1, not identical. Places you may want to tune are
marked with "# TUNE".
"""

import numpy as np
import torch
import torch.nn as nn

from gdnet.cox_en import fit_cox_en, risk_scores  # robust Cox from the core repo


# ===========================================================================
# shared helpers
# ===========================================================================
def _top_variance(X_tr, X_te, k):
    """Keep the k highest-variance features (computed on TRAIN only).
    High-dimensional Cox solvers need this; DL methods use larger k."""
    if X_tr.shape[1] <= k:
        return X_tr, X_te
    idx = np.argsort(X_tr.var(axis=0))[::-1][:k]
    return X_tr[:, idx], X_te[:, idx]


def _zscore(X_tr, X_te):
    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd = np.maximum(sd, 1e-8)
    return (X_tr - mu) / sd, (X_te - mu) / sd


def _cox_partial_likelihood(risk, time, status):
    """Negative Cox partial log-likelihood (Breslow ties), differentiable.

    risk:   (n,) torch tensor = beta^T x (log hazard)
    Used as the training loss of Bridge-Cox / DeepSurv / DCAP / MTC below --
    exactly the l(beta) of paper Eq. 7.
    """
    order = torch.argsort(time, descending=True)   # sort by time DESC ->
    risk, status = risk[order], status[order]      # risk set = prefix sums
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    events = status > 0
    if events.sum() == 0:
        return risk.sum() * 0.0
    return -((risk[events] - log_cumsum[events]).sum()) / events.sum()


def _train_torch_cox(model, X_tr, time_tr, status_tr, epochs, lr,
                     penalty_fn=None, seed=42):
    """Generic full-batch trainer: minimise Cox loss (+ optional penalty)."""
    torch.manual_seed(seed)
    Xt = torch.as_tensor(X_tr, dtype=torch.float32)
    tt = torch.as_tensor(time_tr, dtype=torch.float32)
    st = torch.as_tensor(status_tr, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        risk = model(Xt).squeeze(-1)
        loss = _cox_partial_likelihood(risk, tt, st)
        if penalty_fn is not None:
            loss = loss + penalty_fn(model)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def _torch_risk(model, X):
    model.eval()
    return model(torch.as_tensor(X, dtype=torch.float32)).squeeze(-1).numpy()


# ===========================================================================
# 1. PCA-Cox  [ref 31]:  PCA dimension reduction -> Cox PH
# ===========================================================================
def pca_cox(X_tr, t_tr, s_tr, X_te, seed=42, n_components=50):  # TUNE n_components
    from sklearn.decomposition import PCA
    X_tr, X_te = _zscore(*_top_variance(X_tr, X_te, 5000))
    pca = PCA(n_components=min(n_components, X_tr.shape[0] - 2),
              random_state=seed).fit(X_tr)
    Z_tr, Z_te = pca.transform(X_tr), pca.transform(X_te)
    cph = fit_cox_en(Z_tr, t_tr, s_tr, penalizer=0.1, l1_ratio=0.0)  # ridge Cox
    return risk_scores(cph, Z_te)


# ===========================================================================
# 2. Bridge-Cox [ref 32]: Cox with bridge penalty  lambda * sum |beta_j|^q,
#    0 < q < 1  -> sparser than lasso. Optimised directly with Adam on a
#    smoothed penalty (|b|^q ~ (b^2 + eps)^(q/2)), which is the standard
#    trick to make the non-convex bridge penalty differentiable.
# ===========================================================================
def bridge_cox(X_tr, t_tr, s_tr, X_te, seed=42,
               q=0.5, lam=0.01, epochs=300, lr=0.01):  # TUNE q / lam
    X_tr, X_te = _zscore(*_top_variance(X_tr, X_te, 1000))
    model = nn.Linear(X_tr.shape[1], 1, bias=False)

    def bridge_penalty(m):
        b = m.weight
        return lam * ((b ** 2 + 1e-8) ** (q / 2)).sum()

    _train_torch_cox(model, X_tr, t_tr, s_tr, epochs, lr, bridge_penalty, seed)
    return _torch_risk(model, X_te)


# ===========================================================================
# 3. eXGBC: XGBoost with the built-in Cox survival objective
#    (gradient-boosted trees on the partial likelihood; cf. ref [11]).
# ===========================================================================
def exgbc(X_tr, t_tr, s_tr, X_te, seed=42, n_estimators=200):  # TUNE
    from xgboost import XGBRegressor
    # xgboost survival:cox convention: label = time, NEGATIVE time = censored
    y = np.where(s_tr > 0, t_tr, -t_tr)
    model = XGBRegressor(objective="survival:cox", n_estimators=n_estimators,
                         max_depth=3, learning_rate=0.05, subsample=0.8,
                         random_state=seed, n_jobs=-1)
    model.fit(X_tr, y)
    return np.log(np.maximum(model.predict(X_te), 1e-30))  # HR -> log-risk


# ===========================================================================
# 4. Deep_surv [ref 23, Katzman 2018]: MLP whose output is the log-hazard,
#    trained on the Cox partial likelihood (original: 2 hidden layers,
#    dropout, weight decay).
# ===========================================================================
def deepsurv(X_tr, t_tr, s_tr, X_te, seed=42,
             hidden=(128, 64), dropout=0.3, epochs=300, lr=1e-3):  # TUNE
    X_tr, X_te = _zscore(*_top_variance(X_tr, X_te, 3000))
    layers, d = [], X_tr.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers += [nn.Linear(d, 1)]
    model = nn.Sequential(*layers)
    _train_torch_cox(model, X_tr, t_tr, s_tr, epochs, lr, seed=seed)
    return _torch_risk(model, X_te)


# ===========================================================================
# 5. DCAP [ref 15, Chai 2021]: Denoising-autoencoder Cox: a DAE compresses
#    the multi-omics matrix to a bottleneck; the bottleneck features go into
#    a (elastic-net) Cox model.
# ===========================================================================
def dcap(X_tr, t_tr, s_tr, X_te, seed=42,
         bottleneck=100, noise=0.2, epochs=200, lr=1e-3):  # TUNE
    torch.manual_seed(seed)
    X_tr, X_te = _zscore(*_top_variance(X_tr, X_te, 3000))
    d = X_tr.shape[1]
    enc = nn.Sequential(nn.Linear(d, 500), nn.ReLU(), nn.Linear(500, bottleneck))
    dec = nn.Sequential(nn.Linear(bottleneck, 500), nn.ReLU(), nn.Linear(500, d))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    Xt = torch.as_tensor(X_tr, dtype=torch.float32)
    for _ in range(epochs):                      # denoising reconstruction
        opt.zero_grad()
        noisy = Xt + noise * torch.randn_like(Xt)
        loss = nn.functional.mse_loss(dec(enc(noisy)), Xt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        Z_tr = enc(Xt).numpy()
        Z_te = enc(torch.as_tensor(X_te, dtype=torch.float32)).numpy()
    cph = fit_cox_en(Z_tr, t_tr, s_tr, penalizer=0.1, l1_ratio=0.5)  # Cox-EN
    return risk_scores(cph, Z_te)


# ===========================================================================
# 6. MTC [ref 33, Qiu 2020 "A meta-learning approach for genomic survival
#    analysis"]: pre-train a shared encoder on an auxiliary task, then
#    fine-tune a Cox head on the target cohort. Without the paper's external
#    pan-cancer pre-training corpora we approximate the two-phase recipe
#    WITHIN the cohort: phase 1 self-supervised reconstruction pre-training
#    of the encoder, phase 2 Cox fine-tuning of encoder+head.
#    (Documented approximation -- the closest single-cohort analogue.)
# ===========================================================================
def mtc(X_tr, t_tr, s_tr, X_te, seed=42,
        hidden=256, emb=64, pre_epochs=150, ft_epochs=200, lr=1e-3):  # TUNE
    torch.manual_seed(seed)
    X_tr, X_te = _zscore(*_top_variance(X_tr, X_te, 3000))
    d = X_tr.shape[1]
    encoder = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, emb))
    decoder = nn.Sequential(nn.Linear(emb, hidden), nn.ReLU(), nn.Linear(hidden, d))
    Xt = torch.as_tensor(X_tr, dtype=torch.float32)

    # ---- phase 1: pre-train encoder (meta/auxiliary task stand-in) --------
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    for _ in range(pre_epochs):
        opt.zero_grad()
        nn.functional.mse_loss(decoder(encoder(Xt)), Xt).backward()
        opt.step()

    # ---- phase 2: fine-tune encoder + Cox head on survival ---------------
    model = nn.Sequential(encoder, nn.Linear(emb, 1))
    _train_torch_cox(model, X_tr, t_tr, s_tr, ft_epochs, lr / 2, seed=seed)
    return _torch_risk(model, X_te)


# registry used by run_benchmarks.py (order = paper Table 1 column order)
METHODS = {
    "Bridge-Cox": bridge_cox,
    "PCA-Cox": pca_cox,
    "eXGBC": exgbc,
    "Deep_surv": deepsurv,
    "DCAP": dcap,
    "MTC": mtc,
}
