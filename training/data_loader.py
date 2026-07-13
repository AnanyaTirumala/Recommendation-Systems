"""
data_loader.py – Preprocessing pipeline for Amazon and Yelp datasets.

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
    """Leave-one-out per-user split.

    For every user with >= 3 interactions, the most recent interaction becomes
    the single test positive, the second most recent becomes the single val
    positive, and everything else is training. Users with < 3 interactions have
    no held-out item and go entirely to train.

    This gives every test user exactly ONE ground-truth item, so Recall@K /
    NDCG@K / MRR measure the same task for every user and are directly
    comparable across datasets regardless of how active users are. (Previously
    the last 10% of each user's history was held out, so power users had dozens
    of test positives and were unfairly penalized.) The val_r / test_r args are
    kept for signature compatibility but are no longer used.
    """
    print("[5/6] Leave-one-out per-user split")
    tr, va, te = [], [], []
    for _, g in tqdm(df.groupby("user_idx"), desc="  splitting"):
        g = g.sort_values("timestamp")
        n = len(g)
        if n < 3:
            tr.append(g); continue
        tr.append(g.iloc[:-2])   # all but the last two interactions
        va.append(g.iloc[-2:-1]) # second most recent = val positive
        te.append(g.iloc[-1:])   # most recent      = test positive
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


def load_meta_jsonl(meta_path: str) -> dict:
    """Load item metadata from an Amazon-style meta_*.jsonl file.
    Returns dict mapping parent_asin -> {title, developer, categories, image_url}.
    """
    print(f"[*] Loading item metadata from {meta_path}")
    meta = {}
    with open(meta_path) as f:
        for line in tqdm(f, desc="  reading meta"):
            obj = json.loads(line)
            asin = obj.get("parent_asin") or obj.get("asin", "")
            if not asin:
                continue
            images = obj.get("images") or []
            image_url = ""
            if images:
                img = images[0]
                image_url = img.get("large") or img.get("hi_res") or img.get("thumb") or ""

            details = obj.get("details") or {}
            developer = (
                obj.get("brand")
                or details.get("Developer")
                or details.get("Publisher")
                or details.get("Brand")
                or ""
            )

            categories = obj.get("categories") or []
            if isinstance(categories, list):
                categories = json.dumps(categories)

            meta[asin] = {
                "title":      obj.get("title", ""),
                "author":     developer,
                "categories": categories,
                "image_url":  image_url,
            }
    print(f"  loaded metadata for {len(meta):,} items")
    return meta


def _pipeline(df_raw: pd.DataFrame, item_meta: dict, output_dir: str,
              min_interactions: int, rating_threshold: float, subset: float) -> dict:
    """Shared filter → split → save pipeline used by all dataset loaders."""
    splits_dir = os.path.join(output_dir, "..", "splits")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    avg_rating_map = df_raw.groupby("item_id")["rating"].mean().to_dict()

    df = binarize(df_raw, rating_threshold)
    df = filter_cold_start(df, min_interactions)
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

    meta_rows = [
        {
            "asin":        item_id,
            "internal_id": internal_id,
            "title":       item_meta.get(item_id, {}).get("title", ""),
            "author":      item_meta.get(item_id, {}).get("author", ""),
            "categories":  item_meta.get(item_id, {}).get("categories", "[]"),
            "image_url":   item_meta.get(item_id, {}).get("image_url", ""),
            "avg_rating":  float(avg_rating_map.get(item_id, 0)),
        }
        for item_id, internal_id in item_map.items()
    ]
    pd.DataFrame(meta_rows).to_parquet(
        os.path.join(output_dir, "labels_meta.parquet"), index=False)

    density = train_mat.nnz / (n_users * n_items) * 100
    print(f"\n✓ Done. Users={n_users:,} Items={n_items:,} Density={density:.4f}%")
    return mappings


def preprocess(data_path, output_dir,
               min_interactions=5, rating_threshold=4.0, subset=1.0,
               meta_path=None):
    df_raw = load_interactions(data_path)
    item_meta = load_meta_jsonl(meta_path) if meta_path else {}
    return _pipeline(df_raw, item_meta, output_dir, min_interactions, rating_threshold, subset)


def load_yelp_interactions(review_path: str):
    """Load interactions from Yelp review.json.
    Yelp schema: {user_id, business_id, stars, date}
    Maps business_id → item_id, stars → rating, date → timestamp (unix).
    """
    import datetime
    print(f"[1/6] Loading Yelp interactions from {review_path}")
    rows = []
    with open(review_path) as f:
        for line in tqdm(f, desc="  reading"):
            obj = json.loads(line)
            date_str = obj.get("date", "")
            try:
                ts = int(datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").timestamp())
            except Exception:
                ts = 0
            rows.append({
                "user_id":   obj.get("user_id", ""),
                "item_id":   obj.get("business_id", ""),
                "rating":    float(obj.get("stars", 0)),
                "timestamp": ts,
            })
    df = pd.DataFrame(rows).dropna(subset=["user_id", "item_id"])
    df = df[df["user_id"] != ""]
    df = df[df["item_id"] != ""]
    print(f"  raw interactions: {len(df):,}")
    return df


def load_yelp_meta(business_path: str) -> dict:
    """Load Yelp business.json as item metadata.
    Returns dict mapping business_id → {title, author, categories, image_url}.
    """
    print(f"[*] Loading Yelp business metadata from {business_path}")
    meta = {}
    with open(business_path) as f:
        for line in tqdm(f, desc="  reading meta"):
            obj = json.loads(line)
            bid = obj.get("business_id", "")
            if not bid:
                continue
            cats = obj.get("categories") or ""
            meta[bid] = {
                "title":      obj.get("name", ""),
                "author":     obj.get("city", ""),
                "categories": json.dumps(cats.split(", ")) if cats else "[]",
                "image_url":  "",
            }
    print(f"  loaded metadata for {len(meta):,} businesses")
    return meta


def preprocess_yelp(review_path: str, output_dir: str,
                    min_interactions: int = 5, rating_threshold: float = 4.0,
                    subset: float = 1.0, business_path: str = None):
    df_raw = load_yelp_interactions(review_path)
    item_meta = load_yelp_meta(business_path) if business_path else {}
    return _pipeline(df_raw, item_meta, output_dir, min_interactions, rating_threshold, subset)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--format", default="amazon", choices=["amazon", "yelp"],
                   help="Dataset format: amazon (HuggingFace JSONL) or yelp (review.json)")
    p.add_argument("--data", required=True,
                   help="Path to interactions file (Amazon JSONL or Yelp review.json)")
    p.add_argument("--meta", default=None,
                   help="Path to metadata file (Amazon meta_*.jsonl or Yelp business.json)")
    p.add_argument("--output_dir", default=None,
                   help="Output dir (default: data/processed for amazon, data/yelp/processed for yelp)")
    p.add_argument("--min_interactions", type=int, default=5)
    p.add_argument("--rating_threshold", type=float, default=4.0)
    p.add_argument("--subset", type=float, default=1.0)
    args = p.parse_args()

    default_output = {"yelp": "data/yelp/processed", "amazon": "data/processed"}
    output_dir = args.output_dir or default_output[args.format]

    if args.format == "yelp":
        preprocess_yelp(args.data, output_dir,
                        args.min_interactions, args.rating_threshold, args.subset,
                        business_path=args.meta)
    else:
        preprocess(args.data, output_dir,
                   args.min_interactions, args.rating_threshold, args.subset,
                   meta_path=args.meta)
