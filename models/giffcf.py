"""
GiffCF – Graph Signal Diffusion Model for Collaborative Filtering (Zhu et al., 2024).

Replaces Gaussian diffusion with heat-equation diffusion on the item-item graph.
Forward: smooth interaction signal via graph filters (low-pass).
Reverse: two-stage denoiser (graph conv + transformer cross-attention).

Mac memory note:
  Dense M×M item graph is infeasible. Use top-K Jaccard sparsification (K=50).
  Build the sparse Laplacian in CPU float32, move to device in batches.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm


def build_item_graph(train_mat: sp.csr_matrix, top_k: int = 50) -> torch.Tensor:
    """
    Compute sparse item-item Jaccard similarity graph, keep top-K per item.
    Returns normalised sparse Laplacian as COO FloatTensor (n_items × n_items).
    """
    print("  [GiffCF] Building item-item Jaccard graph (top-K sparse)...")
    n_items = train_mat.shape[1]
    R = train_mat.T.tocsr().astype(np.float32)  # (n_items, n_users)

    # Compute norms for Jaccard: |A ∩ B| / |A ∪ B|
    item_norms = np.array(R.sum(axis=1)).flatten()
    rows_all, cols_all, vals_all = [], [], []

    batch = 500
    for start in tqdm(range(0, n_items, batch), desc="  Jaccard batches"):
        end = min(start + batch, n_items)
        # Intersection counts: (batch_items, all_items)
        inter = R[start:end].dot(R.T).toarray()
        norm_i = item_norms[start:end, None]          # (batch, 1)
        norm_j = item_norms[None, :]                  # (1, n_items)
        union = norm_i + norm_j - inter
        jac = np.where(union > 0, inter / union, 0.0)
        np.fill_diagonal(jac[:, start:end], 0)        # no self-loops

        # Keep top-K per row
        for i, row in enumerate(jac):
            g_idx = start + i
            top_idx = np.argpartition(row, -min(top_k, n_items-1))[-top_k:]
            top_vals = row[top_idx]
            mask = top_vals > 0
            rows_all.extend([g_idx] * mask.sum())
            cols_all.extend(top_idx[mask].tolist())
            vals_all.extend(top_vals[mask].tolist())

    # Symmetrise
    rows_np = np.array(rows_all + cols_all, dtype=np.int32)
    cols_np = np.array(cols_all + rows_all, dtype=np.int32)
    vals_np = np.array(vals_all + vals_all, dtype=np.float32)
    S = sp.coo_matrix((vals_np, (rows_np, cols_np)), shape=(n_items, n_items))
    S = sp.csr_matrix(S)

    # Normalised Laplacian: L = I - D^{-1/2} S D^{-1/2}
    deg = np.array(S.sum(axis=1)).flatten()
    d_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    D_inv = sp.diags(d_inv)
    L = sp.eye(n_items) - D_inv @ S
    L = L.tocoo().astype(np.float32)

    indices = torch.from_numpy(np.vstack([L.row, L.col]).astype(np.int64))
    values  = torch.from_numpy(L.data)
    return torch.sparse_coo_tensor(indices, values, (n_items, n_items))


class GraphConvDenoiser(nn.Module):
    """Two-stage denoiser: GCN layer + transformer cross-attention."""

    def __init__(self, n_items: int, d_model: int = 64, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.proj_in  = nn.Linear(n_items, d_model)
        self.proj_out = nn.Linear(d_model, n_items)
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout)
        )
        self.time_emb = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, d_model)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                L: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_items) noisy interaction signal
        t: (batch,) normalised time step
        L: (n_items, n_items) sparse graph Laplacian
        """
        # Graph convolution: x - alpha * L @ x^T  (smooth signal)
        alpha = 0.5
        x_smooth = x - alpha * torch.sparse.mm(L, x.T).T

        h = self.proj_in(x_smooth)                              # (batch, d)
        t_e = self.time_emb(t.unsqueeze(-1).float())            # (batch, d)
        h = h + t_e

        # Self-attention over single-token sequence (treat d as token dim)
        h_seq = h.unsqueeze(1)                                  # (batch, 1, d)
        attn_out, _ = self.attn(h_seq, h_seq, h_seq)
        h = self.norm1(h + attn_out.squeeze(1))
        h = self.norm2(h + self.ff(h))
        return self.proj_out(h)


class GiffCF(nn.Module):
    """
    Args:
        n_items:   Number of items.
        train_mat: scipy CSR (needed to build item graph; pass None after init if pre-built).
        d_model:   Transformer hidden size.
        n_heads:   Attention heads.
        t_max:     Heat diffusion max time.
        top_k_graph: Sparsity of item graph (neighbours per item).
        T_inf:     DDIM inference steps.
    """

    def __init__(self, n_items: int, train_mat: sp.csr_matrix = None,
                 d_model: int = 64, n_heads: int = 4, t_max: float = 0.5,
                 top_k_graph: int = 50, T_inf: int = 10, dropout: float = 0.1):
        super().__init__()
        self.n_items = n_items
        self.t_max = t_max
        self.T_inf = T_inf

        self.denoiser = GraphConvDenoiser(n_items, d_model, n_heads, dropout)

        if train_mat is not None:
            L = build_item_graph(train_mat, top_k=top_k_graph)
            self.register_buffer("L", L)
        else:
            self.L = None

    def _smooth(self, x: torch.Tensor, t_cont: float) -> torch.Tensor:
        """Approximate graph heat diffusion: x_t ≈ (I - t*L) x_0 (first-order)."""
        L = self.L.to(x.device)
        return x - t_cont * torch.sparse.mm(L, x.T).T

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """Training: random t, smooth, predict residual."""
        t_cont = torch.rand(x0.shape[0], device=x0.device) * self.t_max
        x_t = torch.stack([
            self._smooth(x0[i:i+1], t_cont[i].item()) for i in range(len(x0))
        ]).squeeze(1)
        t_norm = t_cont / self.t_max
        pred = self.denoiser(x_t, t_norm, self.L.to(x0.device))
        return F.mse_loss(pred, x0)

    @torch.no_grad()
    def recommend(self, x0: torch.Tensor, top_k: int = 10,
                  exclude_mask: torch.Tensor = None) -> torch.Tensor:
        device = x0.device
        L = self.L.to(device)
        # Start from smooth signal, iteratively sharpen
        t_steps = torch.linspace(self.t_max, 0, self.T_inf, device=device)
        x = self._smooth(x0, self.t_max)
        for t_val in t_steps:
            t_norm = (t_val / self.t_max).expand(x.shape[0])
            x = self.denoiser(x, t_norm, L)
        if exclude_mask is not None:
            x[exclude_mask] = -1e9
        _, top_items = x.topk(top_k, dim=-1)
        return top_items
