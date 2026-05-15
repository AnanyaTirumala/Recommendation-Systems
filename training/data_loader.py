"""
data_loader.py – Amazon Books 2023 preprocessing pipeline.

Uses the single Books.jsonl file from HuggingFace (McAuley-Lab/Amazon-Reviews-2023).
Interactions (user_id, item_id, rating, timestamp) and per-item avg_rating are both
extracted from this one file.

Run:
  python training/data_loader.py \
      --data data/raw/Books.jsonl \
      --output_dir data/processed \
      --subset 0.1
"""
import argparse, json, os, pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm


def load_interactions(path: str):
    """Load interactions and per-item avg_rating from a single Books.jsonl."""
    print(f"[1/6] Loading interactions from {path}")
    rows = []
    with open(path) as f:
        for line in tqdm(f, desc="  reading"):
            obj = json.loads(line)
            rows.append({
                "user_id":   obj.get("user_id") or obj.get("reviewerID"),
                "item_id":   obj.get("parent_asin") or obj.get("asin"),
                "rating":    float(obj.get("rating") or obj.get("overall", 0)),
                "timestamp": int(obj.get("timestamp") or obj.get("unixReviewTime", 0)),
            })
    df = pd.DataFrame(rows).dropna(subset=["user_id", "item_id"])
    print(f"  raw interactions: {len(df):,}")
    return df


def filter_cold_start(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    print(f"[2/6] Applying {k}-core filter")
    while True:
        before = len(df)
        ic = df["item_id"].value_counts()
        df = df[df["item_id"].isin(ic[ic >= k].index)]
        uc = df["user_id"].value_counts()
        df = df[df["user_id"].isin(uc[uc >= k].index)]
        if len(df) == before:
            break
    print(f"  after filter: {len(df):,} interactions, "
          f"{df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items")
    return df


def binarize(df: pd.DataFrame, threshold: float = 4.0) -> pd.DataFrame:
    print(f"[3/6] Binarizing at rating >= {threshold}")
    df = df[df["rating"] >= threshold].copy()
    df["label"] = 1
    df = (df.sort_values("timestamp")
            .drop_duplicates(subset=["user_id", "item_id"], keep="last"))
    print(f"  positive interactions: {len(df):,}")
    return df


def subsample(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    print(f"[*] Subsampling {frac*100:.0f}% of users")
    users = df["user_id"].drop_duplicates().sample(frac=frac, random_state=42)
    return df[df["user_id"].isin(users)].copy()


def remap_ids(df: pd.DataFrame):
    print("[4/6] Remapping to 0-based IDs")
    user_map = {u: i for i, u in enumerate(sorted(df["user_id"].unique()))}
    item_map = {it: i for i, it in enumerate(sorted(df["item_id"].unique()))}
    df = df.copy()
    df["user_idx"] = df["user_id"].map(user_map)
    df["item_idx"] = df["item_id"].map(item_map)
    return df, user_map, item_map


def temporal_split(df: pd.DataFrame, val_r: float = 0.1, test_r: float = 0.1):
    print("[5/6] Temporal per-user split")
    tr, va, te = [], [], []
    for _, g in tqdm(df.groupby("user_idx"), desc="  splitting"):
        g = g.sort_values("timestamp")
        n = len(g)
        if n < 3:
            tr.append(g); continue
        nt = max(1, int(n * test_r))
        nv = max(1, int(n * val_r))
        nt_r = n - nt - nv
        if nt_r < 1:
            tr.append(g.iloc[:-1]); te.append(g.iloc[-1:]); continue
        tr.append(g.iloc[:nt_r])
        va.append(g.iloc[nt_r:nt_r+nv])
        te.append(g.iloc[nt_r+nv:])
    train = pd.concat(tr, ignore_index=True)
    val   = pd.concat(va, ignore_index=True) if va else pd.DataFrame(columns=df.columns)
    test  = pd.concat(te, ignore_index=True) if te else pd.DataFrame(columns=df.columns)
    print(f"  train={len(train):,}  val={len(val):,}  test={len(test):,}")
    return train, val, test


def build_matrix(df: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    rows = df["user_idx"].values.astype(np.int32)
    cols = df["item_idx"].values.astype(np.int32)
    data = np.ones(len(df), dtype=np.float32)
    mat = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    mat.eliminate_zeros()
    return mat


def preprocess(data_path, output_dir,
               min_interactions=5, rating_threshold=4.0, subset=1.0):
    os.makedirs(output_dir, exist_ok=True)
    splits_dir = os.path.join(output_dir, "..", "splits")
    os.makedirs(splits_dir, exist_ok=True)

    # Load all rows; keep a copy before binarization to compute per-item avg_rating
    df_raw = load_interactions(data_path)
    avg_rating_map = df_raw.groupby("item_id")["rating"].mean().to_dict()

    df = filter_cold_start(df_raw, min_interactions)
    df = binarize(df, rating_threshold)
    if subset < 1.0:
        df = subsample(df, subset)
        df = filter_cold_start(df, min_interactions)

    df, user_map, item_map = remap_ids(df)
    n_users, n_items = len(user_map), len(item_map)

    train, val, test = temporal_split(df)
    train_mat = build_matrix(train, n_users, n_items)
    val_mat   = build_matrix(val,   n_users, n_items)
    test_mat  = build_matrix(test,  n_users, n_items)

    print("[6/6] Saving matrices and metadata")
    sp.save_npz(os.path.join(output_dir, "train_matrix.npz"), train_mat)
    sp.save_npz(os.path.join(splits_dir, "train.npz"),  train_mat)
    sp.save_npz(os.path.join(splits_dir, "val.npz"),    val_mat)
    sp.save_npz(os.path.join(splits_dir, "test.npz"),   test_mat)
    train.to_parquet(os.path.join(splits_dir, "train.parquet"), index=False)
    val.to_parquet(  os.path.join(splits_dir, "val.parquet"),   index=False)
    test.to_parquet( os.path.join(splits_dir, "test.parquet"),  index=False)

    mappings = {
        "user_map": user_map, "item_map": item_map,
        "user_inv": {v: k for k, v in user_map.items()},
        "item_inv": {v: k for k, v in item_map.items()},
        "n_users": n_users, "n_items": n_items,
    }
    with open(os.path.join(output_dir, "mappings.pkl"), "wb") as f:
        pickle.dump(mappings, f)

    # Build metadata from the review data itself (avg_rating computed from all reviews)
    meta_rows = [
        {
            "asin": asin,
            "internal_id": internal_id,
            "title": "",
            "author": "",
            "categories": "[]",
            "image_url": "",
            "avg_rating": float(avg_rating_map.get(asin, 0)),
        }
        for asin, internal_id in item_map.items()
    ]
    pd.DataFrame(meta_rows).to_parquet(
        os.path.join(output_dir, "books_meta.parquet"), index=False)

    density = train_mat.nnz / (n_users * n_items) * 100
    print(f"\n✓ Done. Users={n_users:,} Items={n_items:,} Density={density:.4f}%")
    return mappings


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to Books.jsonl (HuggingFace review file)")
    p.add_argument("--output_dir", default="data/processed")
    p.add_argument("--min_interactions", type=int, default=5)
    p.add_argument("--rating_threshold", type=float, default=4.0)
    p.add_argument("--subset", type=float, default=1.0)
    args = p.parse_args()
    preprocess(args.data, args.output_dir,
               args.min_interactions, args.rating_threshold, args.subset)
