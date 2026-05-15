"""
Central configuration. All modules import from here.
Load .env before importing this module:
    from dotenv import load_dotenv; load_dotenv()
"""
import os
import torch
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
CHECKPOINT_DIR = ROOT / os.getenv("CHECKPOINT_DIR", "checkpoints")
DB_PATH = ROOT / os.getenv("DB_PATH", "data/rec_study.db")
METRICS_PATH = ROOT / os.getenv("METRICS_PATH", "metrics.json")

# ── Device ────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """Return best available device: MPS > CUDA > CPU."""
    requested = os.getenv("DEVICE", "auto").lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)

DEVICE = get_device()

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_SUBSET = float(os.getenv("DATA_SUBSET", "1.0"))
RATING_THRESHOLD = int(os.getenv("RATING_THRESHOLD", "4"))
MIN_INTERACTIONS = int(os.getenv("MIN_INTERACTIONS", "5"))
TOP_K = int(os.getenv("TOP_K", "10"))

# ── Serving ───────────────────────────────────────────────────────────────────
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
MAX_LOADED_MODELS = int(os.getenv("MAX_LOADED_MODELS", "3"))

# ── Model hyperparameters (defaults; overridable via CLI) ─────────────────────
MODEL_CONFIGS = {
    "neumf": {
        "embedding_dim": 64,
        "mlp_layers": [128, 64, 32],
        "dropout": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 1024,
        "epochs": 50,
        "n_negatives": 4,
    },
    "diffrec": {
        "T": 1000,
        "T_inf": 10,          # DDIM steps at inference
        "noise_schedule": "cosine",
        "noise_min": 0.001,
        "noise_max": 0.02,
        "dims": [1000, 600, 300],
        "emb_size": 10,
        "lr": 1e-3,
        "batch_size": 32,     # keep small on Mac
        "epochs": 30,
    },
    "l_diffrec": {
        "T": 500,
        "T_inf": 5,
        "noise_schedule": "cosine",
        "latent_dim": 64,     # LightGCN item emb dimension
        "dims": [300, 200, 100],
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 30,
    },
    "giffcf": {
        "n_layers": 2,
        "d_model": 64,
        "n_heads": 4,
        "t_max": 0.5,
        "top_k_graph": 50,    # sparsify item-item graph
        "lr": 1e-3,
        "batch_size": 256,
        "epochs": 20,
    },
    "cfdiff": {
        "T": 500,
        "T_inf": 10,
        "n_hops": 3,
        "n_heads": 4,
        "d_model": 128,
        "max_neighbors": 20,
        "lr": 5e-4,
        "batch_size": 128,
        "epochs": 30,
    },
    "gdmcf": {
        "gcn_layers": 3,
        "latent_dim": 128,
        "T": 500,
        "T_inf": 10,
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 25,
    },
    "lightgcn": {
        "n_layers": 3,
        "embedding_dim": 64,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 2048,
        "epochs": 200,
    },
}

# Training order matters: L-DiffRec requires LightGCN embeddings first
TRAINING_ORDER = ["lightgcn", "neumf", "diffrec", "l_diffrec", "giffcf", "cfdiff", "gdmcf"]
