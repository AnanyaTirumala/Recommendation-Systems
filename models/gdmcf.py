"""
GDMCF – Graph-based Diffusion Model for Collaborative Filtering (2025).

Two-component architecture:
  1. Graph encoder (LightGCN-style) → captures high-order collaborative signals.
  2. Diffusion denoiser (MLP) → operates in the graph-embedding latent space.

Addresses DiffRec's limitation of ignoring higher-order signals.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np


def build_norm_adj(train_mat: sp.csr_matrix) -> torch.Tensor:
    n_users, n_items = train_mat.shape
    R = train_mat.astype(np.float32)
    upper = sp.hstack([sp.csr_matrix((n_users, n_users)), R])
    lower = sp.hstack([R.T, sp.csr_matrix((n_items, n_items))])
    A = sp.vstack([upper, lower]).tocsr()
    deg = np.array(A.sum(axis=1)).flatten()
    d_inv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv = sp.diags(d_inv)
    A_norm = (D_inv @ A @ D_inv).tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack([A_norm.row, A_norm.col]).astype(np.int64))
    val = torch.from_numpy(A_norm.data)
    return torch.sparse_coo_tensor(idx, val, A_norm.shape)


def cosine_schedule(T: int):
    steps = T + 1
    t = torch.linspace(0, T, steps) / T
    ab = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    ab /= ab[0]
    betas = (1 - ab[1:] / ab[:-1]).clamp(1e-4, 0.9999)
    return betas, ab[1:]


class GraphEncoder(nn.Module):
    """LightGCN-style graph encoder, outputs user + item embeddings."""

    def __init__(self, n_users: int, n_items: int,
                 emb_dim: int, n_layers: int = 3):
        super().__init__()
        self.n_users = n_users
        self.n_layers = n_layers
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, adj: torch.Tensor):
        E = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        agg = [E]
        for _ in range(self.n_layers):
            E = torch.sparse.mm(adj, E)
            agg.append(E)
        E = torch.stack(agg).mean(0)
        return E[:self.n_users], E[self.n_users:]


class DiffusionDenoiser(nn.Module):
    def __init__(self, latent_dim: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, hidden)
        )
        layers = [nn.Linear(latent_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_e = self.time_emb(t.unsqueeze(-1).float())
        h = self.net[0](z) + t_e
        for layer in self.net[1:]:
            h = layer(h)
        return h


class GDMCF(nn.Module):
    """
    Args:
        n_users, n_items: Dataset dimensions.
        emb_dim:          Graph embedding dimension.
        latent_dim:       Diffusion latent dimension.
        gcn_layers:       LightGCN propagation layers.
        T, T_inf:         Diffusion training/inference steps.
        train_mat:        scipy CSR training matrix.
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64,
                 latent_dim: int = 128, gcn_layers: int = 3,
                 T: int = 500, T_inf: int = 10, hidden: int = 256,
                 n_layers: int = 3, dropout: float = 0.1,
                 train_mat: sp.csr_matrix = None):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.T = T
        self.T_inf = T_inf
        self.emb_dim = emb_dim

        self.encoder = GraphEncoder(n_users, n_items, emb_dim, gcn_layers)

        # Project graph user embeddings into diffusion latent space
        self.proj_in  = nn.Linear(emb_dim, latent_dim)
        self.proj_out = nn.Linear(latent_dim, emb_dim)

        self.denoiser = DiffusionDenoiser(latent_dim, hidden, n_layers, dropout)

        # Decoder: latent → item scores
        self.score_head = nn.Linear(emb_dim, n_items)

        betas, ab = cosine_schedule(T)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_bar", ab)
        self.register_buffer("sqrt_ab", ab.sqrt())
        self.register_buffer("sqrt_1ab", (1 - ab).sqrt())

        if train_mat is not None:
            adj = build_norm_adj(train_mat)
            self.register_buffer("adj", adj)
        else:
            self.adj = None

    def _graph_user_embs(self):
        return self.encoder(self.adj.to(self.encoder.user_emb.weight.device))[0]

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        u_embs = self._graph_user_embs()[user_ids]      # (B, emb_dim)
        z0 = self.proj_in(u_embs)                       # (B, latent_dim)
        batch = z0.shape[0]
        t = torch.randint(0, self.T, (batch,), device=z0.device)
        noise = torch.randn_like(z0)
        z_t = self.sqrt_ab[t].view(-1, 1) * z0 + self.sqrt_1ab[t].view(-1, 1) * noise
        t_norm = t.float() / self.T
        z_pred = self.denoiser(z_t, t_norm)
        return F.mse_loss(z_pred, z0)

    @torch.no_grad()
    def recommend(self, user_ids: torch.Tensor, top_k: int = 10,
                  exclude_mask: torch.Tensor = None) -> torch.Tensor:
        device = user_ids.device
        z = torch.randn(len(user_ids), self.proj_in.out_features, device=device)
        steps = torch.linspace(self.T - 1, 0, self.T_inf, dtype=torch.long, device=device)
        for t_val in steps:
            t_norm = (t_val.float() / self.T).expand(z.shape[0])
            z = self.denoiser(z, t_norm)
        u_emb = self.proj_out(z)
        scores = self.score_head(u_emb)
        if exclude_mask is not None:
            scores[exclude_mask] = -1e9
        _, top_items = scores.topk(top_k, dim=-1)
        return top_items
