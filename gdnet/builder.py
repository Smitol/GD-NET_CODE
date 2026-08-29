"""GD-Net model architecture: GCN layer + MLP encoder wrapped in a MoCo-style
momentum-contrast framework (paper Sections 2.1.3, Eq. 2-5).

This file is adapted from the authors' official `pcl/builder.py`
(https://github.com/JackiLin/GD-Net) with three practical changes, each marked
with "# CHANGED":

1. CPU/GPU agnostic ........ the original hard-codes `.cuda()`; we take a
                             `device` argument so it also runs on CPU.
2. Vectorised GCN .......... the original loops over every sample and calls
                             torch_geometric's GCNConv(1, 1) once per sample,
                             which is extremely slow. Because the GCN has a
                             single input/output channel, it is mathematically
                             identical to one sparse matrix multiplication:
                                 GCNConv(1,1)(x, edge_index) == A_hat @ x * w + b
                             where A_hat = D^-1/2 (A + I) D^-1/2 is the
                             symmetrically-normalised adjacency (paper Eq. 2).
                             We precompute A_hat once and do a single sparse
                             matmul for the whole batch (100-1000x faster,
                             and removes the torch_geometric dependency).
3. Flexible queue size ..... MoCo's negative-sample queue requires
                             r % batch_size == 0. TCGA cohorts are small
                             (~150-500 patients), so we auto-shrink the queue
                             to a multiple of the batch size instead of
                             crashing on the original `assert`.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Graph convolution (paper Eq. 2):  F(H, A) = sigma( D~^-1/2 A~ D~^-1/2 H W )
# ---------------------------------------------------------------------------
def normalized_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Build the sparse normalised adjacency A_hat = D~^-1/2 (A + I) D~^-1/2.

    Args:
        edge_index: LongTensor of shape (2, E) with feature-index pairs
                    (the KEGG gene-gene network mapped onto feature indices).
        num_nodes:  total number of features (graph nodes).

    Returns:
        torch.sparse_coo_tensor of shape (num_nodes, num_nodes).
    """
    # make the graph undirected + add self loops (A~ = A + I, paper Eq. 2)
    src = torch.cat([edge_index[0], edge_index[1], torch.arange(num_nodes)])
    dst = torch.cat([edge_index[1], edge_index[0], torch.arange(num_nodes)])

    # deduplicate edges
    idx = src * num_nodes + dst
    idx, perm = torch.unique(idx, return_inverse=False), None
    src = idx // num_nodes
    dst = idx % num_nodes

    # degree of A~
    deg = torch.zeros(num_nodes).scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    vals = deg_inv_sqrt[src] * deg_inv_sqrt[dst]  # D~^-1/2 A~ D~^-1/2
    return torch.sparse_coo_tensor(
        torch.stack([src, dst]), vals, (num_nodes, num_nodes)
    ).coalesce()


class GCNLayer(nn.Module):
    """Single-channel GCN layer, equivalent to torch_geometric GCNConv(1, 1).

    Input : X of shape (batch, num_features)  -- each ROW is one patient,
            each COLUMN is one molecular feature (= one node of the KEGG graph).
    Output: same shape, after mixing every feature with its KEGG neighbours.
    """

    def __init__(self, adj_hat: torch.Tensor):
        super().__init__()
        # A_hat is fixed (depends only on KEGG) -> register as buffer, not parameter
        self.register_buffer("adj_hat", adj_hat)
        # scalar weight + bias == the 1x1 "W" of GCNConv(1, 1)
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (F, F) @ (F, B) -> (F, B), then back to (B, F)
        out = torch.sparse.mm(self.adj_hat, x.t()) * self.weight + self.bias
        return torch.relu(out.t())


# ---------------------------------------------------------------------------
# Encoder: GCN layer -> 2 fully-connected blocks -> low-dim embedding
# (paper Fig. 1B: "GCN + MLP", embeddings later fed to the Cox-EN model)
# ---------------------------------------------------------------------------
def full_block(in_features, out_features, p_drop=0.0):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.ReLU(),
        nn.Dropout(p=p_drop),
    )


class MLPEncoder(nn.Module):
    """GCN + MLP encoder (same layer sizes as the official code: F -> 1024 -> low_dim)."""

    def __init__(self, adj_hat, num_genes=10000, num_hiddens=128, p_drop=0.0):
        super().__init__()
        self.gcn = GCNLayer(adj_hat)
        self.encoder = nn.Sequential(
            full_block(num_genes, 1024, p_drop),
            full_block(1024, num_hiddens, p_drop),
        )

    def forward(self, x):
        # x: (batch, num_features) fused multi-omics matrix
        return self.encoder(self.gcn(x))


# ---------------------------------------------------------------------------
# MoCo: momentum-contrast wrapper (paper Section 2.1.3, Eq. 3-5)
#   - query encoder f_q : updated by backprop
#   - key   encoder f_k : updated by momentum  theta_k <- m*theta_k + (1-m)*theta_q  (Eq. 5)
#   - queue of negative keys for the InfoNCE contrastive loss (Eq. 3)
# ---------------------------------------------------------------------------
class MoCo(nn.Module):
    def __init__(self, adj_hat, num_genes, dim=128, r=512, m=0.999, T=0.2,
                 batch_size=None, device="cpu"):
        """
        Args:
            adj_hat:    precomputed normalised adjacency (from normalized_adjacency()).
            num_genes:  number of input features.
            dim:        embedding dimension (paper/README: --low_dim 200).
            r:          queue size (number of negative keys).
            m:          momentum for the key encoder (Eq. 5; official default 0.999).
            T:          softmax temperature of the contrastive loss (official 0.2).
            batch_size: if given, r is rounded DOWN to a multiple of it   # CHANGED
            device:     'cpu' or 'cuda'                                    # CHANGED
        """
        super().__init__()
        self.m = m
        self.T = T
        self.device = device

        # CHANGED: queue must be a multiple of the batch size for the simple
        # ring-buffer update below; auto-adjust instead of asserting.
        if batch_size is not None and r % batch_size != 0:
            r = max(batch_size, (r // batch_size) * batch_size)
        self.r = r

        self.encoder_q = MLPEncoder(adj_hat, num_genes=num_genes, num_hiddens=dim)
        self.encoder_k = MLPEncoder(adj_hat, num_genes=num_genes, num_hiddens=dim)

        # key encoder starts as a copy of the query encoder and is never
        # updated by gradients (only by momentum) -- Eq. 5
        for pq, pk in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            pk.data.copy_(pq.data)
            pk.requires_grad = False

        # queue of negative embeddings (dim x r), randomly initialised
        self.register_buffer("queue", nn.functional.normalize(torch.randn(dim, self.r), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """theta_k <- m * theta_k + (1 - m) * theta_q   (paper Eq. 5)"""
        for pq, pk in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            pk.data = pk.data * self.m + pq.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Push the newest keys into the ring-buffer queue of negatives."""
        bs = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + bs <= self.r:
            self.queue[:, ptr:ptr + bs] = keys.T
            ptr = (ptr + bs) % self.r
        else:                       # CHANGED: wrap around for uneven last batch
            first = self.r - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, :bs - first] = keys[first:].T
            ptr = bs - first
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k=None, is_eval=False):
        """
        Training : im_q, im_k are two AUGMENTED views of the same patients
                   (the "positive pair" of the paper). Returns (logits, labels)
                   for the InfoNCE / CrossEntropy contrastive loss (Eq. 3).
        Eval     : is_eval=True -> return the (normalised) key-encoder
                   embedding of im_q. These embeddings are the "multi-omics
                   meta-features" fed to the Cox-EN model.
        """
        if is_eval:
            k = self.encoder_k(im_q)
            return nn.functional.normalize(k, dim=1)

        # ---- key (positive) embedding, no gradient ----
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = nn.functional.normalize(self.encoder_k(im_k), dim=1)

        # ---- query embedding ----
        q = nn.functional.normalize(self.encoder_q(im_q), dim=1)

        # InfoNCE logits (Eq. 3):
        #   positive similarity: q . k  (same patient, different augmentation)
        #   negative similarity: q . queue (other patients / older samples)
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)          # (B, 1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])  # (B, r)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T

        # the "correct class" is always index 0 (the positive pair)
        labels = torch.zeros(logits.shape[0], dtype=torch.long,
                             device=self.device)   # CHANGED: was .cuda()

        self._dequeue_and_enqueue(k)
        return logits, labels
