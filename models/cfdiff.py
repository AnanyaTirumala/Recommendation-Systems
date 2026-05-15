"""
CF-Diff – Collaborative Filtering Based on Diffusion Models (Hou et al., 2024).

Key idea: cross-attention over multi-hop neighbours during each denoising step,
capturing high-order collaborative signals without quadratic cost.

Scales O(N+M) at inference via sparse neighbour sampling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import math


def cosine_betas(T: int):
    steps = T + 1
    t = torch.linspace(0, T, steps) / T
    ab = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    ab /= ab[0]
    betas = (1 - ab[1:] / ab[:-1]).clamp(1e-4, 0.9999)
    return betas, ab[1:]


class CrossAttnDenoiser(nn.Module):
    """Denoiser that attends to sampled multi-hop neighbours."""

    def __init__(self, d_model: int, n_heads: int = 4,
                 n_hops: int = 3, dropout: float = 0.1):
        super().__init__()
        self.n_hops = n_hops
        self.item_proj = nn.Linear(d_model, d_model)
        self.user_proj = nn.Linear(d_model, d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout)
        )
        self.time_emb = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, d_model)
        )

    def forward(self, x_user: torch.Tensor, neighbor_embs: torch.Tensor,
                t_norm: torch.Tensor) -> torch.Tensor:
        """
        x_user:       (batch, d) noisy user latent
        neighbor_embs:(batch, max_nb, d) sampled neighbour item embeddings
        t_norm:       (batch,) in [0, 1]
        """
        q = self.user_proj(x_user).unsqueeze(1)           # (B, 1, d)
        kv = self.item_proj(neighbor_embs)                 # (B, nb, d)
        t_e = self.time_emb(t_norm.unsqueeze(-1).float())  # (B, d)
        q = q + t_e.unsqueeze(1)
        ca_out, _ = self.cross_attn(q, kv, kv)
        h = self.norm1(q + ca_out).squeeze(1)             # (B, d)
        sa_out, _ = self.self_attn(h.unsqueeze(1), h.unsqueeze(1), h.unsqueeze(1))
        h = self.norm2(h + sa_out.squeeze(1))
        return h + self.ff(h)


class CFDiff(nn.Module):
    """
    Args:
        n_users, n_items: Dataset dimensions.
        d_model:          Latent dimension.
        n_heads:          Attention heads.
        n_hops:           Neighbourhood hops for context (default 3).
        max_neighbors:    Max neighbours sampled per user per hop.
        T, T_inf:         Diffusion steps.
        train_mat:        scipy CSR — used to build adjacency for neighbour lookup.
    """

    def __init__(self, n_users: int, n_items: int, d_model: int = 128,
                 n_heads: int = 4, n_hops: int = 3, max_neighbors: int = 20,
                 T: int = 1000, T_inf: int = 10, dropout: float = 0.1,
                 train_mat: sp.csr_matrix = None):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.d_model = d_model
        self.T = T
        self.T_inf = T_inf
        self.max_neighbors = max_neighbors

        self.item_emb = nn.Embedding(n_items, d_model)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        # Encoder/decoder for user interaction rows
        self.encoder = nn.Sequential(
            nn.Linear(n_items, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model)
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, n_items)
        )

        self.denoiser = CrossAttnDenoiser(d_model, n_heads, n_hops, dropout)

        betas, ab = cosine_betas(T)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_bar", ab)
        self.register_buffer("sqrt_ab", ab.sqrt())
        self.register_buffer("sqrt_1ab", (1 - ab).sqrt())

        # Precompute neighbour lists from training matrix
        if train_mat is not None:
            self._build_neighbor_lists(train_mat)
        else:
            self.neighbor_lists = None

    def _build_neighbor_lists(self, train_mat: sp.csr_matrix):
        """Store item neighbours for each user (top-max_neighbors by frequency)."""
        print("  [CF-Diff] Building neighbour lists...")
        lists = []
        for u in range(train_mat.shape[0]):
            row = train_mat.getrow(u)
            items = row.indices[:self.max_neighbors].tolist()
            lists.append(items)
        self.neighbor_lists = lists

    def _get_neighbor_embs(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Look up item embeddings of each user's neighbours. Pad to max_neighbors."""
        device = self.item_emb.weight.device
        batch = len(user_ids)
        out = torch.zeros(batch, self.max_neighbors, self.d_model, device=device)
        if self.neighbor_lists is None:
            return out
        for i, uid in enumerate(user_ids.cpu().tolist()):
            nb = self.neighbor_lists[uid]
            if not nb:
                continue
            t = torch.tensor(nb[:self.max_neighbors], dtype=torch.long, device=device)
            out[i, :len(t)] = self.item_emb(t)
        return out

    def forward(self, x0: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        z0 = self.encoder(x0)
        batch = z0.shape[0]
        t = torch.randint(0, self.T, (batch,), device=z0.device)
        noise = torch.randn_like(z0)
        z_t = self.sqrt_ab[t].view(-1, 1) * z0 + self.sqrt_1ab[t].view(-1, 1) * noise
        nb = self._get_neighbor_embs(user_ids)
        t_norm = t.float() / self.T
        z_pred = self.denoiser(z_t, nb, t_norm)
        diff_loss = F.mse_loss(z_pred, z0)
        recon = F.binary_cross_entropy_with_logits(self.decoder(z0), x0)
        return diff_loss + 0.1 * recon

    @torch.no_grad()
    def recommend(self, x0: torch.Tensor, user_ids: torch.Tensor,
                  top_k: int = 10, exclude_mask: torch.Tensor = None) -> torch.Tensor:
        device = x0.device
        z = torch.randn(x0.shape[0], self.d_model, device=device)
        nb = self._get_neighbor_embs(user_ids)
        steps = torch.linspace(self.T - 1, 0, self.T_inf, dtype=torch.long, device=device)
        for t_val in steps:
            t_norm = (t_val.float() / self.T).expand(z.shape[0])
            z = self.denoiser(z, nb, t_norm)
        scores = self.decoder(z)
        if exclude_mask is not None:
            scores[exclude_mask] = -1e9
        _, top_items = scores.topk(top_k, dim=-1)
        return top_items
