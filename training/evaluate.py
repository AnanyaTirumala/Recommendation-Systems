"""
evaluate.py – Evaluation harness for all 7 models.

Protocol:
  - Leave-one-out: per user, most recent interaction = ground truth positive.
  - 99 random negatives sampled per test user.
  - Metrics: Recall@K, NDCG@K, MRR.
  - Results written to metrics_{dataset}.json.

Run:
  python training/evaluate.py --dataset videogames
  python training/evaluate.py --dataset yelp
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


# ─── evaluation loop ────────────────────────────────────────────────────────

def evaluate_model(model, model_name: str, test_df, train_mat, mappings,
                   device, top_k: int = 10, batch_size: int = 64) -> dict:
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

    for start in tqdm(range(0, len(user_list), batch_size), desc=f"  eval {model_name}"):
        batch_users = user_list[start:start + batch_size]
        user_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)

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
    }
    print(f"  {model_name}: Recall@{top_k}={results[f'Recall@{top_k}']:.4f}  "
          f"NDCG@{top_k}={results[f'NDCG@{top_k}']:.4f}  MRR={results['MRR']:.4f}")
    return results


def run_evaluation(dataset: str = "videogames", checkpoint_dir: str = "checkpoints/",
                   output_path: str = None, top_k: int = 10):
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
                                 device, top_k=top_k)
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
    args = p.parse_args()
    run_evaluation(args.dataset, args.checkpoint_dir, args.output, args.top_k)
