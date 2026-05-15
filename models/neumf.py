"""
NeuMF – Neural Matrix Factorization (He et al., 2017).
Baseline model for the comparative study.

Architecture:
  - GMF branch: element-wise product of user/item embeddings
  - MLP branch: concatenation → dense layers
  - Fusion:     concat(GMF, MLP) → sigmoid output
"""
import torch
import torch.nn as nn


class NeuMF(nn.Module):
    """
    Neural Matrix Factorization.

    Args:
        n_users:      Number of users.
        n_items:      Number of items.
        emb_size:     Embedding dimension for GMF branch.
        mlp_layers:   Hidden sizes for MLP branch, e.g. [128, 64, 32].
        dropout:      Dropout rate on MLP layers.
    """

    def __init__(self, n_users: int, n_items: int,
                 emb_size: int = 64,
                 mlp_layers: list = None,
                 dropout: float = 0.2):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [128, 64, 32]

        self.n_users = n_users
        self.n_items = n_items

        # ── GMF embeddings ──────────────────────────────────────────────────
        self.user_emb_gmf = nn.Embedding(n_users, emb_size)
        self.item_emb_gmf = nn.Embedding(n_items, emb_size)

        # ── MLP embeddings (input size = emb_size * 2) ──────────────────────
        mlp_input = emb_size * 2
        self.user_emb_mlp = nn.Embedding(n_users, mlp_input // 2)
        self.item_emb_mlp = nn.Embedding(n_items, mlp_input // 2)

        # ── MLP layers ───────────────────────────────────────────────────────
        layers = []
        in_size = mlp_input
        for out_size in mlp_layers:
            layers += [nn.Linear(in_size, out_size), nn.ReLU(), nn.Dropout(dropout)]
            in_size = out_size
        self.mlp = nn.Sequential(*layers)

        # ── Final prediction layer ──────────────────────────────────────────
        self.predict = nn.Linear(emb_size + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for emb in [self.user_emb_gmf, self.item_emb_gmf,
                    self.user_emb_mlp, self.item_emb_mlp]:
            nn.init.normal_(emb.weight, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_ids: (batch,) long tensor
            item_ids: (batch,) long tensor
        Returns:
            scores:   (batch,) float tensor in [0, 1]
        """
        # GMF branch
        ug = self.user_emb_gmf(user_ids)
        ig = self.item_emb_gmf(item_ids)
        gmf_out = ug * ig                                  # element-wise

        # MLP branch
        um = self.user_emb_mlp(user_ids)
        im = self.item_emb_mlp(item_ids)
        mlp_out = self.mlp(torch.cat([um, im], dim=-1))

        # Fusion
        concat = torch.cat([gmf_out, mlp_out], dim=-1)
        scores = self.sigmoid(self.predict(concat)).squeeze(-1)
        return scores

    @torch.no_grad()
    def recommend(self, user_ids: torch.Tensor, top_k: int = 10,
                  exclude_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Full-catalogue recommendation for a batch of users.

        Args:
            user_ids:     (batch,) long tensor
            top_k:        Number of items to return per user.
            exclude_mask: (batch, n_items) bool tensor; True = already interacted.
        Returns:
            top_items: (batch, top_k) long tensor of item indices.
        """
        device = next(self.parameters()).device
        batch = len(user_ids)
        # Score all items for each user
        item_ids = torch.arange(self.n_items, device=device)
        # Expand: (batch, n_items)
        u_exp = user_ids.unsqueeze(1).expand(batch, self.n_items)
        i_exp = item_ids.unsqueeze(0).expand(batch, self.n_items)
        scores = self.forward(u_exp.reshape(-1), i_exp.reshape(-1))
        scores = scores.view(batch, self.n_items)

        if exclude_mask is not None:
            scores[exclude_mask] = -1e9

        _, top_items = scores.topk(top_k, dim=-1)
        return top_items
