"""
LightGCN (He et al., 2020) – GNN collaborative filtering baseline.

Uses the user-item bipartite graph. Aggregates embeddings over K layers
with pure linear neighbourhood averaging (no activation, no transformation).
Final embeddings are mean-pooled across layers. Trained with BPR loss.

Mac note: SparseTensor ops fall back to CPU on MPS; we use standard torch
sparse COO which MPS handles fine for typical graph sizes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np


def normalised_adj(train_mat: sp.csr_matrix) -> torch.Tensor:
    """
    Build the symmetric normalised adjacency for the user-item bipartite graph.
    A = [[0, R], [R^T, 0]],  A_norm = D^{-1/2} A D^{-1/2}
    Returns a sparse COO FloatTensor of shape (N+M, N+M).
    """
    n_users, n_items = train_mat.shape
    # Build full bipartite adjacency
    R = train_mat.astype(np.float32)
    upper = sp.hstack([sp.csr_matrix((n_users, n_users)), R])
    lower = sp.hstack([R.T, sp.csr_matrix((n_items, n_items))])
    A = sp.vstack([upper, lower]).tocsr()

    # D^{-1/2}
    deg = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    A_norm = A_norm.tocoo().astype(np.float32)

    indices = torch.from_numpy(
        np.vstack([A_norm.row, A_norm.col]).astype(np.int64)
    )
    values = torch.from_numpy(A_norm.data)
    size   = torch.Size(A_norm.shape)
    return torch.sparse_coo_tensor(indices, values, size)


class LightGCN(nn.Module):
    """
    Args:
        n_users:    Number of users.
        n_items:    Number of items.
        emb_dim:    Embedding dimension.
        n_layers:   Number of propagation layers (default 3).
        train_mat:  scipy CSR matrix of training interactions (needed to build graph).
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64,
                 n_layers: int = 3, train_mat: sp.csr_matrix = None):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        if train_mat is not None:
            self.register_buffer("adj", normalised_adj(train_mat))
        else:
            self.adj = None

    def _propagate(self, device) -> tuple:
        """Run K-layer LightGCN propagation. Returns final user/item embeddings."""
        adj = self.adj.to(device)
        # Stack all entity embeddings
        E = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        layer_embs = [E]
        for _ in range(self.n_layers):
            E = torch.sparse.mm(adj, E)
            layer_embs.append(E)
        E_final = torch.stack(layer_embs, dim=0).mean(dim=0)
        users_E = E_final[:self.n_users]
        items_E = E_final[self.n_users:]
        return users_E, items_E

    def forward(self, user_ids: torch.Tensor,
                pos_ids: torch.Tensor,
                neg_ids: torch.Tensor) -> torch.Tensor:
        """
        BPR loss for a batch of (user, pos_item, neg_item) triples.
        """
        device = user_ids.device
        users_E, items_E = self._propagate(device)
        u  = users_E[user_ids]
        pi = items_E[pos_ids]
        ni = items_E[neg_ids]
        pos_score = (u * pi).sum(dim=-1)
        neg_score = (u * ni).sum(dim=-1)
        loss = -F.logsigmoid(pos_score - neg_score).mean()
        # L2 regularisation on raw embeddings (not propagated)
        reg = (self.user_emb(user_ids).norm(2).pow(2) +
               self.item_emb(pos_ids).norm(2).pow(2) +
               self.item_emb(neg_ids).norm(2).pow(2)) / len(user_ids)
        return loss + 1e-4 * reg

    @torch.no_grad()
    def get_item_embeddings(self) -> torch.Tensor:
        """Return final item embeddings (used by L-DiffRec)."""
        device = self.item_emb.weight.device
        _, items_E = self._propagate(device)
        return items_E

    @torch.no_grad()
    def recommend(self, user_ids: torch.Tensor, top_k: int = 10,
                  exclude_mask: torch.Tensor = None) -> torch.Tensor:
        device = user_ids.device
        users_E, items_E = self._propagate(device)
        u = users_E[user_ids]             # (batch, emb)
        scores = u @ items_E.T            # (batch, n_items)
        if exclude_mask is not None:
            scores[exclude_mask] = -1e9
        _, top_items = scores.topk(top_k, dim=-1)
        return top_items
