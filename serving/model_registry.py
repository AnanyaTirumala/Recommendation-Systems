"""
model_registry.py – Lazy-loading model registry with LRU eviction.

At most MAX_LOADED models kept in RAM simultaneously.
Models are loaded on first request and evicted LRU when the cap is hit.

Supports multiple datasets; checkpoint filenames follow {dataset}_{model}.pt.

Dataset data paths:
  videogames → data/processed/, data/splits/
  yelp       → data/yelp/processed/, data/yelp/splits/
"""
import os, pickle, time
import numpy as np
import scipy.sparse as sp
import torch

MAX_LOADED = 3
_cache: dict = {}   # key: "{dataset}:{model}"
_lru:   list = []   # list of "{dataset}:{model}"

_train_mat: dict = {}   # key: dataset
_mappings:  dict = {}   # key: dataset
_device    = None

DATASET_PATHS = {
    "videogames": {"processed": "data/processed",      "splits": "data/splits"},
    "yelp":       {"processed": "data/yelp/processed", "splits": "data/yelp/splits"},
}


def _get_device():
    global _device
    if _device is None:
        if torch.backends.mps.is_available():
            _device = torch.device("mps")
        else:
            _device = torch.device("cpu")
    return _device


def _load_shared_data(dataset: str = "videogames"):
    if dataset not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset}")
    paths = DATASET_PATHS[dataset]
    if dataset not in _train_mat:
        _train_mat[dataset] = sp.load_npz(os.path.join(paths["splits"], "train.npz"))
    if dataset not in _mappings:
        with open(os.path.join(paths["processed"], "mappings.pkl"), "rb") as f:
            _mappings[dataset] = pickle.load(f)
    return _train_mat[dataset], _mappings[dataset]


def _load_checkpoint(name: str, dataset: str = "videogames"):
    from models import MODEL_REGISTRY
    ckpt_path = os.path.join("checkpoints", f"{dataset}_{name}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = _get_device()
    ckpt   = torch.load(ckpt_path, map_location="cpu")
    cfg    = ckpt.get("config", {})

    train_mat, mappings = _load_shared_data(dataset)

    if name in ("lightgcn", "giffcf", "cfdiff", "gdmcf"):
        model = MODEL_REGISTRY[name](**cfg, train_mat=train_mat)
    else:
        model = MODEL_REGISTRY[name](**cfg)

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.float().to(device).eval()
    return model


def get_model(name: str, dataset: str = "videogames"):
    """Return the model for `name` and `dataset`, loading from disk if needed."""
    key = f"{dataset}:{name}"
    if key not in _cache:
        if len(_cache) >= MAX_LOADED:
            evict = _lru.pop(0)
            del _cache[evict]
            print(f"  [registry] Evicted '{evict}' from memory.")
        print(f"  [registry] Loading '{key}'...")
        t0 = time.time()
        _cache[key] = _load_checkpoint(name, dataset)
        print(f"  [registry] '{key}' loaded in {time.time()-t0:.1f}s")
    if key in _lru:
        _lru.remove(key)
    _lru.append(key)
    return _cache[key]


def get_train_row(user_id: int, dataset: str = "videogames") -> np.ndarray:
    """Return the dense training interaction row for a user."""
    train_mat, _ = _load_shared_data(dataset)
    return train_mat.getrow(user_id).toarray().squeeze()


def get_mappings(dataset: str = "videogames") -> dict:
    _, m = _load_shared_data(dataset)
    return m


def model_status(dataset: str = "videogames") -> dict:
    return {name: (f"{dataset}:{name}" in _cache) for name in
            ["neumf", "diffrec", "ldiffrec", "giffcf", "cfdiff", "gdmcf", "lightgcn"]}


def checkpoint_exists(name: str, dataset: str = "videogames") -> bool:
    return os.path.exists(os.path.join("checkpoints", f"{dataset}_{name}.pt"))
