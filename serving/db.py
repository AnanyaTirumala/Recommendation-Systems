"""
db.py – SQLite database layer using Peewee ORM.

Tables: User, Book, InferenceCache.
Run directly to initialise:
  python serving/db.py --init --processed_dir data/processed
"""
import argparse, json, os, pickle, time
import pandas as pd
from peewee import (SqliteDatabase, Model, IntegerField, TextField,
                    FloatField, AutoField)

DB_PATH = os.environ.get("DB_PATH", "rec_study.db")
db = SqliteDatabase(DB_PATH, pragmas={"journal_mode": "wal", "cache_size": -64000})


class BaseModel(Model):
    class Meta:
        database = db


class User(BaseModel):
    internal_id = IntegerField(primary_key=True)
    amazon_id   = TextField(unique=True)
    n_train     = IntegerField(default=0)
    n_test      = IntegerField(default=0)


class Book(BaseModel):
    internal_id = IntegerField(primary_key=True)
    asin        = TextField(unique=True)
    title       = TextField(default="")
    author      = TextField(default="")
    categories  = TextField(default="[]")
    image_url   = TextField(default="")
    avg_rating  = FloatField(default=0.0)


class InferenceCache(BaseModel):
    id         = AutoField()
    user_id    = IntegerField()
    model_name = TextField()
    top_k      = IntegerField()
    results    = TextField()
    created_at = IntegerField()

    class Meta:
        indexes = ((("user_id", "model_name", "top_k"), True),)


def init_db(processed_dir="data/processed", splits_dir="data/splits"):
    db.connect(reuse_if_open=True)
    db.create_tables([User, Book, InferenceCache], safe=True)

    with open(os.path.join(processed_dir, "mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)

    print("Inserting users...")
    train_df = pd.read_parquet(os.path.join(splits_dir, "train.parquet"))
    test_df  = pd.read_parquet(os.path.join(splits_dir, "test.parquet"))
    train_counts = train_df.groupby("user_idx").size().to_dict()
    test_counts  = test_df.groupby("user_idx").size().to_dict()

    batch = []
    for amazon_id, internal_id in mappings["user_map"].items():
        batch.append({
            "internal_id": internal_id, "amazon_id": amazon_id,
            "n_train": train_counts.get(internal_id, 0),
            "n_test":  test_counts.get(internal_id, 0),
        })
        if len(batch) >= 5000:
            User.insert_many(batch).on_conflict_replace().execute()
            batch = []
    if batch:
        User.insert_many(batch).on_conflict_replace().execute()
    print(f"  {User.select().count()} users inserted.")

    meta_path = os.path.join(processed_dir, "books_meta.parquet")
    if os.path.exists(meta_path):
        print("Inserting books...")
        meta = pd.read_parquet(meta_path)
        batch = []
        for _, row in meta.iterrows():
            batch.append({
                "internal_id": int(row["internal_id"]), "asin": row["asin"],
                "title": row.get("title", ""), "author": row.get("author", ""),
                "categories": row.get("categories", "[]"),
                "image_url":  row.get("image_url", ""),
                "avg_rating": float(row.get("avg_rating", 0)),
            })
            if len(batch) >= 5000:
                Book.insert_many(batch).on_conflict_replace().execute()
                batch = []
        if batch:
            Book.insert_many(batch).on_conflict_replace().execute()
        print(f"  {Book.select().count()} books inserted.")

    print("DB initialised at", DB_PATH)


def get_book(internal_id):
    try:
        b = Book.get(Book.internal_id == internal_id)
        return {"internal_id": b.internal_id, "asin": b.asin,
                "title": b.title or f"Book #{internal_id}", "author": b.author,
                "categories": json.loads(b.categories or "[]"),
                "image_url": b.image_url, "avg_rating": b.avg_rating}
    except Book.DoesNotExist:
        return {"internal_id": internal_id, "asin": "", "title": f"Book #{internal_id}",
                "author": "", "categories": [], "image_url": "", "avg_rating": 0.0}


def get_cached(user_id, model_name, top_k, ttl=86400):
    try:
        c = InferenceCache.get(
            (InferenceCache.user_id == user_id) &
            (InferenceCache.model_name == model_name) &
            (InferenceCache.top_k == top_k))
        if time.time() - c.created_at < ttl:
            return json.loads(c.results)
    except InferenceCache.DoesNotExist:
        pass
    return None


def set_cache(user_id, model_name, top_k, results):
    InferenceCache.insert(
        user_id=user_id, model_name=model_name, top_k=top_k,
        results=json.dumps(results), created_at=int(time.time())
    ).on_conflict_replace().execute()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--init", action="store_true")
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--splits_dir", default="data/splits")
    args = p.parse_args()
    if args.init:
        init_db(args.processed_dir, args.splits_dir)
