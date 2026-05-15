"""
train_all.py – Orchestrator: trains all 7 models sequentially.

Training order matters:
  1. LightGCN  (first — L-DiffRec needs its item embeddings)
  2. NeuMF
  3. DiffRec
  4. L-DiffRec  (requires checkpoints/lightgcn_item_emb.pt)
  5. GiffCF
  6. CF-Diff
  7. GDMCF

Run:
  python training/train_all.py --device mps --subset 0.1
"""
import argparse, json, os, pickle, time
import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim
from tqdm import tqdm

from models import MODEL_REGISTRY


# ─── device ─────────────────────────────────────────────────────────────────

def get_device(requested: str = "auto"):
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─── data helpers ────────────────────────────────────────────────────────────

def load_data(processed_dir="data/processed", splits_dir="data/splits"):
    with open(os.path.join(processed_dir, "mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)
    train_mat = sp.load_npz(os.path.join(splits_dir, "train.npz"))
    val_mat   = sp.load_npz(os.path.join(splits_dir, "val.npz"))
    import pandas as pd
    train_df = pd.read_parquet(os.path.join(splits_dir, "train.parquet"))
    return mappings, train_mat, val_mat, train_df


def neg_sample_bpr(train_mat: sp.csr_matrix, user_ids: np.ndarray) -> np.ndarray:
    """Sample one random negative item per user that they haven't interacted with."""
    n_items = train_mat.shape[1]
    negs = []
    for u in user_ids:
        pos_set = set(train_mat.getrow(u).indices.tolist())
        while True:
            neg = np.random.randint(n_items)
            if neg not in pos_set:
                negs.append(neg)
                break
    return np.array(negs)


def save_checkpoint(model, optimizer, epoch, metric, config, name, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{name}.pt")
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch":                epoch,
        "best_recall@10":       metric,
        "config":               config,
    }, path)


# ─── training routines ───────────────────────────────────────────────────────

def train_neumf(mappings, train_mat, val_mat, device, epochs, batch_size,
                lr, checkpoint_dir):
    print("\n══ Training NeuMF ══")
    n_users, n_items = mappings["n_users"], mappings["n_items"]
    cfg = dict(n_users=n_users, n_items=n_items, emb_size=64,
               mlp_layers=[128, 64, 32], dropout=0.2)
    model = MODEL_REGISTRY["neumf"](**cfg).float().to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = torch.nn.BCELoss()

    # Build positive pairs
    coo = train_mat.tocoo()
    users_pos = coo.row.astype(np.int64)
    items_pos = coo.col.astype(np.int64)

    best = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        idx = np.random.permutation(len(users_pos))
        total_loss = 0.0
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start:start + batch_size]
            u  = torch.tensor(users_pos[batch_idx], dtype=torch.long, device=device)
            ip = torch.tensor(items_pos[batch_idx], dtype=torch.long, device=device)
            neg = neg_sample_bpr(train_mat, users_pos[batch_idx])
            in_ = torch.tensor(neg, dtype=torch.long, device=device)
            scores_pos = model(u, ip)
            scores_neg = model(u, in_)
            labels = torch.cat([torch.ones_like(scores_pos),
                                 torch.zeros_like(scores_neg)])
            scores  = torch.cat([scores_pos, scores_neg])
            loss = loss_fn(scores, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{epochs}  loss={total_loss/max(1,len(idx)//batch_size):.4f}")
    save_checkpoint(model, opt, epochs, best, cfg, "neumf", checkpoint_dir)
    print("  ✓ NeuMF checkpoint saved.")


def train_diffrec(mappings, train_mat, device, epochs, batch_size, lr,
                  checkpoint_dir, latent=False, item_emb_path=None):
    name = "ldiffrec" if latent else "diffrec"
    print(f"\n══ Training {'L-DiffRec' if latent else 'DiffRec'} ══")
    n_items = mappings["n_items"]

    if latent:
        item_emb_dim = 64
        cfg = dict(n_items=n_items, item_emb_dim=item_emb_dim, latent_dim=64,
                   T=1000, T_inf=10, hidden=256, n_layers=3)
    else:
        cfg = dict(n_items=n_items, T=1000, T_inf=10, hidden=1000, n_layers=4)

    model = MODEL_REGISTRY[name](**cfg).float().to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)

    train_array = torch.tensor(
        train_mat.toarray(), dtype=torch.float32)  # keep on CPU, batch to device

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(len(train_array))
        total_loss = 0.0
        for start in range(0, len(idx), batch_size):
            batch = train_array[idx[start:start + batch_size]].to(device)
            # Normalize by row sum
            row_sum = batch.sum(dim=-1, keepdim=True).clamp(min=1)
            batch = batch / row_sum
            loss = model(batch)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}/{epochs}  loss={total_loss:.4f}")

    save_checkpoint(model, opt, epochs, 0.0, cfg, name, checkpoint_dir)
    print(f"  ✓ {name} checkpoint saved.")


def train_lightgcn(mappings, train_mat, device, epochs, batch_size, lr,
                   checkpoint_dir):
    print("\n══ Training LightGCN ══")
    n_users, n_items = mappings["n_users"], mappings["n_items"]
    cfg = dict(n_users=n_users, n_items=n_items, emb_dim=64, n_layers=3)
    # Pass train_mat to build graph
    model = MODEL_REGISTRY["lightgcn"](
        **cfg, train_mat=train_mat).float().to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    coo = train_mat.tocoo()
    users_pos = coo.row.astype(np.int64)
    items_pos = coo.col.astype(np.int64)

    for epoch in range(1, epochs + 1):
        model.train()
        idx = np.random.permutation(len(users_pos))
        total_loss = 0.0
        for start in range(0, len(idx), batch_size):
            b = idx[start:start + batch_size]
            u  = torch.tensor(users_pos[b], dtype=torch.long, device=device)
            pi = torch.tensor(items_pos[b], dtype=torch.long, device=device)
            neg = neg_sample_bpr(train_mat, users_pos[b])
            ni  = torch.tensor(neg, dtype=torch.long, device=device)
            loss = model(u, pi, ni)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}/{epochs}  loss={total_loss:.4f}")

    # Save item embeddings for L-DiffRec
    with torch.no_grad():
        item_embs = model.get_item_embeddings()
        torch.save(item_embs.cpu(),
                   os.path.join(checkpoint_dir, "lightgcn_item_emb.pt"))
    save_checkpoint(model, opt, epochs, 0.0, cfg, "lightgcn", checkpoint_dir)
    print("  ✓ LightGCN checkpoint saved.")


def train_graph_model(name, model_cls, extra_cfg, mappings, train_mat,
                      device, epochs, batch_size, lr, checkpoint_dir):
    print(f"\n══ Training {name} ══")
    n_users, n_items = mappings["n_users"], mappings["n_items"]
    cfg = dict(n_users=n_users, n_items=n_items, **extra_cfg)

    model = model_cls(**cfg, train_mat=train_mat).float().to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)

    coo = train_mat.tocoo()
    users_all = coo.row.astype(np.int64)

    for epoch in range(1, epochs + 1):
        model.train()
        idx = np.random.permutation(len(users_all))
        total_loss = 0.0
        for start in range(0, len(idx), batch_size):
            b = idx[start:start + batch_size]
            u = torch.tensor(users_all[b], dtype=torch.long, device=device)
            if name == "cfdiff":
                x0 = torch.tensor(
                    train_mat[users_all[b]].toarray(),
                    dtype=torch.float32, device=device)
                row_sum = x0.sum(-1, keepdim=True).clamp(min=1)
                loss = model(x0 / row_sum, u)
            else:  # gdmcf
                loss = model(u)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}/{epochs}  loss={total_loss:.4f}")

    save_checkpoint(model, opt, epochs, 0.0, cfg, name, checkpoint_dir)
    print(f"  ✓ {name} checkpoint saved.")


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--checkpoint_dir", default="checkpoints/")
    p.add_argument("--subset", type=float, default=1.0,
                   help="Must match value used in data_loader.py")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs_neumf",    type=int, default=50)
    p.add_argument("--epochs_diffrec",  type=int, default=30)
    p.add_argument("--epochs_lightgcn", type=int, default=200)
    p.add_argument("--epochs_graph",    type=int, default=25)
    p.add_argument("--models", nargs="+",
                   default=["lightgcn","neumf","diffrec","ldiffrec",
                            "giffcf","cfdiff","gdmcf"])
    args = p.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    mappings, train_mat, val_mat, train_df = load_data()
    n_users, n_items = mappings["n_users"], mappings["n_items"]
    print(f"Dataset: {n_users:,} users × {n_items:,} items  "
          f"(density={train_mat.nnz/(n_users*n_items)*100:.4f}%)")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(args.checkpoint_dir, "training_log.jsonl")

    # Training order is fixed (LightGCN first for L-DiffRec dependency)
    ordered = ["lightgcn","neumf","diffrec","ldiffrec","giffcf","cfdiff","gdmcf"]
    to_train = [m for m in ordered if m in args.models]

    for name in to_train:
        t0 = time.time()
        if name == "lightgcn":
            train_lightgcn(mappings, train_mat, device,
                           args.epochs_lightgcn, args.batch_size, args.lr,
                           args.checkpoint_dir)
        elif name == "neumf":
            train_neumf(mappings, train_mat, val_mat, device,
                        args.epochs_neumf, args.batch_size, args.lr,
                        args.checkpoint_dir)
        elif name == "diffrec":
            train_diffrec(mappings, train_mat, device,
                          args.epochs_diffrec, min(args.batch_size, 64), args.lr,
                          args.checkpoint_dir, latent=False)
        elif name == "ldiffrec":
            emb_path = os.path.join(args.checkpoint_dir, "lightgcn_item_emb.pt")
            train_diffrec(mappings, train_mat, device,
                          args.epochs_diffrec, min(args.batch_size, 64), args.lr,
                          args.checkpoint_dir, latent=True, item_emb_path=emb_path)
        elif name == "giffcf":
            from models.giffcf import GiffCF
            train_graph_model("giffcf", GiffCF,
                              dict(d_model=64, n_heads=4, t_max=0.5,
                                   top_k_graph=50, T_inf=10),
                              mappings, train_mat, device,
                              args.epochs_graph, min(args.batch_size, 32),
                              args.lr, args.checkpoint_dir)
        elif name == "cfdiff":
            from models.cfdiff import CFDiff
            train_graph_model("cfdiff", CFDiff,
                              dict(d_model=128, n_heads=4, n_hops=3,
                                   max_neighbors=20, T=1000, T_inf=10),
                              mappings, train_mat, device,
                              args.epochs_graph, min(args.batch_size, 64),
                              args.lr, args.checkpoint_dir)
        elif name == "gdmcf":
            from models.gdmcf import GDMCF
            train_graph_model("gdmcf", GDMCF,
                              dict(emb_dim=64, latent_dim=128, gcn_layers=3,
                                   T=500, T_inf=10),
                              mappings, train_mat, device,
                              args.epochs_graph, min(args.batch_size, 64),
                              args.lr, args.checkpoint_dir)

        elapsed = time.time() - t0
        with open(log_path, "a") as f:
            f.write(json.dumps({"model": name, "elapsed_s": round(elapsed, 1)}) + "\n")

    print("\n✓ All training complete. Run evaluate.py to compute metrics.")


if __name__ == "__main__":
    main()
