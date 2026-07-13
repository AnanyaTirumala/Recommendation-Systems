"""
evaluate.py – Evaluation harness for all 7 models.

Protocol (default: sampled):
  - Leave-one-out: per user, most recent interaction = ground-truth positive
    (produced by data_loader.temporal_split).
  - "sampled": rank the held-out positive(s) against `--n_negatives` (default 99)
    randomly sampled items the user has never interacted with, i.e. a fixed
    candidate field of ~100 items. Because the candidate field is the same size
    for every user and every dataset, the metrics are directly comparable across
    datasets regardless of catalog size.
  - "full": rank against the entire item catalog (train items excluded). This is
    catalog-size sensitive, so a bigger catalog (e.g. Yelp) yields lower numbers
    even for an equally good model — not comparable across datasets.
  - Metrics: Recall@K, NDCG@K, MRR.
  - Results written to metrics_{dataset}.json.

Run:
  python training/evaluate.py --dataset videogames                 # sampled (default)
  python training/evaluate.py --dataset yelp --protocol full       # full-catalog ranking
"""
import argparse, json, os, pickle
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

DATASET_PATHS = {
    "videogames": {"processed": "data/processed",      "splits": "data/splits"},
    "yelp":       {"processed": "data/yelp/processed", "splits": "data/yelp/splits"},
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── metrics ────────────────────────────────────────────────────────────────

def recall_at_k(ranked: list, ground_truth: set, k: int) -> float:
    return float(len(set(ranked[:k]) & ground_truth) / min(len(ground_truth), k))


def ndcg_at_k(ranked: list, ground_truth: set, k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(ranked[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked: list, ground_truth: set) -> float:
    for i, item in enumerate(ranked):
        if item in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


# ─── negative sampling ──────────────────────────────────────────────────────

def sample_negatives(seen: set, n_items: int, n_neg: int, rng) -> list:
    """Sample `n_neg` item ids the user has never interacted with.

    `seen` is the set of items to avoid (train history + held-out positives).
    Oversample then filter so we still return `n_neg` ids after removing any
    that collided with `seen`.
    """
    n_available = n_items - len(seen)
    if n_available <= n_neg:
        # Tiny catalog relative to history: use every unseen item.
        return [i for i in range(n_items) if i not in seen]
    negs: set = set()
    while len(negs) < n_neg:
        draw = rng.integers(0, n_items, size=n_neg * 2)
        for it in draw:
            it = int(it)
            if it not in seen and it not in negs:
                negs.add(it)
                if len(negs) == n_neg:
                    break
    return list(negs)


# ─── evaluation loop ────────────────────────────────────────────────────────

def evaluate_model(model, model_name: str, test_df, train_mat, mappings,
                   device, top_k: int = 10, batch_size: int = 64,
                   protocol: str = "sampled", n_negatives: int = 99,
                   seed: int = 42) -> dict:
    model.eval()
    n_users = mappings["n_users"]
    n_items = mappings["n_items"]

    test_by_user = {}
    for _, row in test_df.iterrows():
        uid = int(row["user_idx"])
        iid = int(row["item_idx"])
        test_by_user.setdefault(uid, set()).add(iid)

    recall_scores, ndcg_scores, mrr_scores = [], [], []
    user_list = list(test_by_user.keys())
    rng = np.random.default_rng(seed)

    for start in tqdm(range(0, len(user_list), batch_size), desc=f"  eval {model_name}"):
        batch_users = user_list[start:start + batch_size]
        user_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)

        if protocol == "sampled":
            # Rank each user's positives against a fixed field of n_negatives
            # sampled unseen items: mask out EVERYTHING except the candidate set,
            # so topk ranks only among {positives + sampled negatives}.
            exclude = torch.ones(len(batch_users), n_items, dtype=torch.bool, device=device)
            for i, uid in enumerate(batch_users):
                gt = test_by_user[uid]
                seen = set(train_mat.getrow(uid).indices.tolist()) | gt
                negs = sample_negatives(seen, n_items, n_negatives, rng)
                candidates = list(gt) + negs
                exclude[i, candidates] = False
        else:  # "full": rank against the whole catalog, excluding train items
            exclude = torch.zeros(len(batch_users), n_items, dtype=torch.bool, device=device)
            for i, uid in enumerate(batch_users):
                train_row = train_mat.getrow(uid)
                exclude[i, train_row.indices] = True

        with torch.no_grad():
            if model_name == "cfdiff":
                top_items = model.recommend(
                    train_mat[batch_users].toarray(),
                    user_tensor, top_k=top_k, exclude_mask=exclude
                )
            elif model_name in ("diffrec", "ldiffrec"):
                x0 = torch.tensor(
                    train_mat[batch_users].toarray(), dtype=torch.float32, device=device)
                top_items = model.recommend(x0, top_k=top_k, exclude_mask=exclude)
            else:  # neumf, lightgcn, giffcf, gdmcf
                top_items = model.recommend(user_tensor, top_k=top_k, exclude_mask=exclude)

        top_items_np = top_items.cpu().numpy()
        for i, uid in enumerate(batch_users):
            gt = test_by_user[uid]
            ranked = top_items_np[i].tolist()
            recall_scores.append(recall_at_k(ranked, gt, top_k))
            ndcg_scores.append(ndcg_at_k(ranked, gt, top_k))
            mrr_scores.append(mrr(ranked, gt))

    results = {
        f"Recall@{top_k}": float(np.mean(recall_scores)),
        f"NDCG@{top_k}":   float(np.mean(ndcg_scores)),
        "MRR":             float(np.mean(mrr_scores)),
        "n_users_tested":  len(user_list),
        "protocol":        protocol,
        "n_negatives":     n_negatives if protocol == "sampled" else None,
    }
    print(f"  {model_name}: Recall@{top_k}={results[f'Recall@{top_k}']:.4f}  "
          f"NDCG@{top_k}={results[f'NDCG@{top_k}']:.4f}  MRR={results['MRR']:.4f}")
    return results


def run_evaluation(dataset: str = "videogames", checkpoint_dir: str = "checkpoints/",
                   output_path: str = None, top_k: int = 10,
                   protocol: str = "sampled", n_negatives: int = 99):
    if dataset not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset}")

    if output_path is None:
        output_path = f"metrics_{dataset}.json"

    device = get_device()
    print(f"Device: {device}  Dataset: {dataset}")
    torch.manual_seed(42)
    np.random.seed(42)

    paths = DATASET_PATHS[dataset]
    with open(os.path.join(paths["processed"], "mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)

    import pandas as pd
    test_df   = pd.read_parquet(os.path.join(paths["splits"], "test.parquet"))
    train_mat = sp.load_npz(os.path.join(paths["splits"], "train.npz"))

    from models import MODEL_REGISTRY

    model_names = ["neumf", "diffrec", "ldiffrec", "giffcf", "cfdiff", "gdmcf", "lightgcn"]
    all_metrics = {}

    for name in model_names:
        # Checkpoint filename: {dataset}_{name}.pt
        ckpt_path = os.path.join(checkpoint_dir, f"{dataset}_{name}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] {name}: checkpoint not found at {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg  = ckpt.get("config", {})
        if name in ("lightgcn", "giffcf", "cfdiff", "gdmcf"):
            cfg = {**cfg, "train_mat": train_mat}
        model = MODEL_REGISTRY[name](**cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

        # Key in JSON matches checkpoint name: {dataset}_{name}
        checkpoint_key = f"{dataset}_{name}"
        results = evaluate_model(model, name, test_df, train_mat, mappings,
                                 device, top_k=top_k,
                                 protocol=protocol, n_negatives=n_negatives)
        all_metrics[checkpoint_key] = results

    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n✓ Metrics saved to {output_path}")
    return all_metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="videogames",
                   choices=list(DATASET_PATHS.keys()))
    p.add_argument("--checkpoint_dir", default="checkpoints/")
    p.add_argument("--output", default=None,
                   help="Output path (default: metrics_{dataset}.json)")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--protocol", default="sampled", choices=["sampled", "full"],
                   help="sampled: rank against n_negatives sampled items (comparable "
                        "across datasets); full: rank against the whole catalog")
    p.add_argument("--n_negatives", type=int, default=99,
                   help="Number of sampled negatives per user (sampled protocol only)")
    args = p.parse_args()
    run_evaluation(args.dataset, args.checkpoint_dir, args.output, args.top_k,
                   protocol=args.protocol, n_negatives=args.n_negatives)
