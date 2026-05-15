"""
DiffRec and L-DiffRec (Wang et al., 2023).

DiffRec:  Gaussian diffusion directly on user interaction rows (length n_items).
LDiffRec: Diffusion in a compressed latent space (requires pre-trained item embeddings).

DDIM sampling used at inference for speed (T_inf << T_train).

Mac note: Keep batch_size <= 64 for full Amazon Books. Use T_inf=5 for fast serving.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── noise schedule ──────────────────────────────────────────────────────────

def cosine_beta_schedule(T: int, s: float = 0.008):
    """Cosine noise schedule (Nichol & Dhariwal, 2021)."""
    steps = T + 1
    t = torch.linspace(0, T, steps) / T
    alphas_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
    return betas.clamp(0.0001, 0.9999)


# ─── MLP denoiser ────────────────────────────────────────────────────────────

class DenoiseMLP(nn.Module):
    """MLP that estimates noise given noisy input and diffusion step embedding."""

    def __init__(self, in_dim: int, hidden: int = 1000, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, hidden)
        )
        layers = [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, in_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, dim) noisy input
        t: (batch,) float step values in [0, 1]
        """
        t_emb = self.time_emb(t.unsqueeze(-1).float())
        # Inject time embedding as additive bias after first linear
        h = self.net[0](x) + t_emb
        for layer in self.net[1:]:
            h = layer(h)
        return h


# ─── DiffRec ─────────────────────────────────────────────────────────────────

class DiffRec(nn.Module):
    """
    Gaussian diffusion over user interaction rows.

    Args:
        n_items:   Vocabulary size (number of items).
        T:         Number of diffusion steps during training.
        T_inf:     Number of DDIM steps at inference (default 10).
        hidden:    Hidden size of the MLP denoiser.
        n_layers:  Depth of the MLP denoiser.
    """

    def __init__(self, n_items: int, T: int = 1000, T_inf: int = 10,
                 hidden: int = 1000, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_items = n_items
        self.T = T
        self.T_inf = T_inf

        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # Register as buffers so they move with .to(device)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("sqrt_alphas_bar", alphas_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_bar", (1 - alphas_bar).sqrt())

        self.denoiser = DenoiseMLP(n_items, hidden, n_layers, dropout)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Forward diffusion: corrupt x0 at step t.
        Returns (noisy_x, noise).
        """
        noise = torch.randn_like(x0)
        sqrt_ab  = self.sqrt_alphas_bar[t].view(-1, 1)
        sqrt_1ab = self.sqrt_one_minus_alphas_bar[t].view(-1, 1)
        x_t = sqrt_ab * x0 + sqrt_1ab * noise
        return x_t, noise

    def forward(self, x0: torch.Tensor):
        """
        Training step: randomly sample t, corrupt x0, predict noise.
        Returns MSE loss.
        """
        batch = x0.shape[0]
        t = torch.randint(0, self.T, (batch,), device=x0.device)
        x_t, noise = self.q_sample(x0, t)
        t_norm = t.float() / self.T
        noise_pred = self.denoiser(x_t, t_norm)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def ddim_sample(self, x_T: torch.Tensor) -> torch.Tensor:
        """
        DDIM reverse process: T_inf steps from noise to interaction scores.
        """
        x = x_T
        steps = torch.linspace(self.T - 1, 0, self.T_inf, dtype=torch.long,
                               device=x_T.device)
        for t_val in steps:
            t = t_val.expand(x.shape[0])
            t_norm = t.float() / self.T
            noise_pred = self.denoiser(x, t_norm)
            ab = self.alphas_bar[t_val]
            x = (x - (1 - ab).sqrt() * noise_pred) / ab.sqrt()
        return x

    @torch.no_grad()
    def recommend(self, x0: torch.Tensor, top_k: int = 10,
                  exclude_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x0: (batch, n_items) training interaction rows (normalised).
        Returns (batch, top_k) top item indices.
        """
        x_T = torch.randn_like(x0)
        scores = self.ddim_sample(x_T)
        if exclude_mask is not None:
            scores[exclude_mask] = -1e9
        _, top_items = scores.topk(top_k, dim=-1)
        return top_items


# ─── L-DiffRec ───────────────────────────────────────────────────────────────

class LDiffRec(nn.Module):
    """
    Latent DiffRec: diffusion in a low-dimensional latent space.
    Requires pre-trained item embeddings (e.g. from LightGCN).

    Args:
        n_items:      Number of items.
        item_emb_dim: Dimension of pre-trained item embeddings.
        latent_dim:   Latent dimension to project into (default 64).
        T, T_inf:     Same as DiffRec.
    """

    def __init__(self, n_items: int, item_emb_dim: int = 64,
                 latent_dim: int = 64, T: int = 1000, T_inf: int = 10,
                 hidden: int = 256, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.n_items   = n_items
        self.latent_dim = latent_dim
        self.T    = T
        self.T_inf = T_inf

        # Encoder: interaction row → latent
        self.encoder = nn.Sequential(
            nn.Linear(n_items, hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim)
        )
        # Decoder: latent → interaction scores
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.GELU(),
            nn.Linear(hidden, n_items)
        )

        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("sqrt_alphas_bar", alphas_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_bar", (1 - alphas_bar).sqrt())

        self.denoiser = DenoiseMLP(latent_dim, hidden, n_layers, dropout)

    def forward(self, x0: torch.Tensor):
        """x0: (batch, n_items). Returns total loss."""
        z0 = self.encoder(x0)
        batch = z0.shape[0]
        t = torch.randint(0, self.T, (batch,), device=z0.device)
        noise = torch.randn_like(z0)
        sqrt_ab  = self.sqrt_alphas_bar[t].view(-1, 1)
        sqrt_1ab = self.sqrt_one_minus_alphas_bar[t].view(-1, 1)
        z_t = sqrt_ab * z0 + sqrt_1ab * noise
        t_norm = t.float() / self.T
        noise_pred = self.denoiser(z_t, t_norm)
        diff_loss = F.mse_loss(noise_pred, noise)
        # Reconstruction loss through decoder
        x0_hat = self.decoder(z0)
        recon_loss = F.binary_cross_entropy_with_logits(x0_hat, x0)
        return diff_loss + 0.1 * recon_loss

    @torch.no_grad()
    def recommend(self, x0: torch.Tensor, top_k: int = 10,
                  exclude_mask: torch.Tensor = None) -> torch.Tensor:
        z_T = torch.randn(x0.shape[0], self.latent_dim, device=x0.device)
        steps = torch.linspace(self.T - 1, 0, self.T_inf, dtype=torch.long,
                               device=x0.device)
        z = z_T
        for t_val in steps:
            t = t_val.expand(z.shape[0])
            t_norm = t.float() / self.T
            noise_pred = self.denoiser(z, t_norm)
            ab = self.alphas_bar[t_val]
            z = (z - (1 - ab).sqrt() * noise_pred) / ab.sqrt()
        scores = self.decoder(z)
        if exclude_mask is not None:
            scores[exclude_mask] = -1e9
        _, top_items = scores.topk(top_k, dim=-1)
        return top_items
