"""
model_registry.py – Lazy-loading model registry with LRU eviction.

At most MAX_LOADED models kept in RAM simultaneously.
Models are loaded on first request and evicted LRU when the cap is hit.

Mac rationale: 7 models × ~700 MB each = ~5 GB — won't fit in 16 GB alongside
the OS, the sparse matrix (~500 MB), and Flask. Cap at 3.
"""
import os, pickle, time
import numpy as np
import scipy.sparse as sp
import torch

MAX_LOADED = 3
_cache: dict = {}
_lru:   list = []

_train_mat = None
_mappings  = None
_device    = None


def _get_device():
    global _device
    if _device is None:
        if torch.backends.mps.is_available():
            _device = torch.device("mps")
        else:
            _device = torch.device("cpu")
    return _device


def _load_shared_data():
    global _train_mat, _mappings
    if _train_mat is None:
        _train_mat = sp.load_npz("data/splits/train.npz")
    if _mappings is None:
        with open("data/processed/mappings.pkl", "rb") as f:
            _mappings = pickle.load(f)
    return _train_mat, _mappings


def _load_checkpoint(name: str):
    from models import MODEL_REGISTRY
    ckpt_path = os.path.join("checkpoints", f"{name}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = _get_device()
    ckpt   = torch.load(ckpt_path, map_location=device)
    cfg    = ckpt.get("config", {})

    train_mat, mappings = _load_shared_data()

    # Graph-aware models need the training matrix
    if name in ("lightgcn", "giffcf", "cfdiff", "gdmcf"):
        model = MODEL_REGISTRY[name](**cfg, train_mat=train_mat)
    else:
        model = MODEL_REGISTRY[name](**cfg)

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.float().to(device).eval()
    return model


def get_model(name: str):
    """Return the model for `name`, loading from disk if needed."""
    global _cache, _lru
    if name not in _cache:
        if len(_cache) >= MAX_LOADED:
            evict = _lru.pop(0)
            del _cache[evict]
            print(f"  [registry] Evicted '{evict}' from memory.")
        print(f"  [registry] Loading '{name}'...")
        t0 = time.time()
        _cache[name] = _load_checkpoint(name)
        print(f"  [registry] '{name}' loaded in {time.time()-t0:.1f}s")
    if name in _lru:
        _lru.remove(name)
    _lru.append(name)
    return _cache[name]


def get_train_row(user_id: int) -> np.ndarray:
    """Return the dense training interaction row for a user."""
    train_mat, _ = _load_shared_data()
    return train_mat.getrow(user_id).toarray().squeeze()


def get_mappings() -> dict:
    _, m = _load_shared_data()
    return m


def model_status() -> dict:
    return {name: (name in _cache) for name in
            ["neumf","diffrec","ldiffrec","giffcf","cfdiff","gdmcf","lightgcn"]}
