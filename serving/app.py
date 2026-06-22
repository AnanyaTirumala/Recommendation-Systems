"""
app.py – Flask REST API for the comparative study app.

Endpoints:
  POST /api/recommend     { user_id, top_k, models[], dataset }
  GET  /api/metrics?dataset=videogames
  GET  /api/users/sample?dataset=videogames
  GET  /api/books/<item_id>?dataset=videogames
  GET  /api/status?dataset=videogames

Run:
  FLASK_APP=serving/app.py flask run --port 5000
  or via VSCode launch config "Flask API"
"""
import json, os, time
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

import serving.db as _db_module
from serving.db import Book, User, get_book, get_cached, set_cache
from serving import model_registry as registry

app = Flask(__name__)
CORS(app)

VALID_MODELS = ["neumf", "diffrec", "ldiffrec", "giffcf", "cfdiff", "gdmcf", "lightgcn"]
VALID_DATASETS = ["videogames", "yelp"]
METRICS_PATHS = {
    "videogames": "metrics_videogames.json",
    "yelp":       "metrics_yelp.json",
}
DB_PATHS = {
    "videogames": "rec_study_videogames.db",
    "yelp":       "rec_study_yelp.db",
}


def _connect_db(dataset: str = "videogames"):
    db = _db_module.db
    if not db.is_closed():
        db.close()
    db.init(DB_PATHS[dataset], pragmas={"journal_mode": "wal", "cache_size": -64000})
    db.connect()


@app.before_request
def before():
    # dataset comes from query param or JSON body; default to videogames
    dataset = request.args.get("dataset", "videogames")
    if request.method == "POST" and request.content_type and "json" in request.content_type:
        try:
            dataset = (request.get_json(force=True, silent=True) or {}).get("dataset", dataset)
        except Exception:
            pass
    if dataset not in VALID_DATASETS:
        dataset = "videogames"
    _connect_db(dataset)


@app.teardown_request
def after(exc):
    db = _db_module.db
    if not db.is_closed():
        db.close()


# ─── helpers ────────────────────────────────────────────────────────────────

def _run_inference(model_name: str, user_id: int, top_k: int,
                   dataset: str = "videogames") -> list:
    """Run one model and return [{rank, item_id, score, item}]."""
    cached = get_cached(user_id, f"{dataset}:{model_name}", top_k)
    if cached:
        return cached

    device = registry._get_device()
    model  = registry.get_model(model_name, dataset)
    train_row = registry.get_train_row(user_id, dataset)
    mappings  = registry.get_mappings(dataset)
    n_items   = mappings["n_items"]

    user_tensor = torch.tensor([user_id], dtype=torch.long, device=device)
    exclude = torch.zeros(1, n_items, dtype=torch.bool, device=device)
    interacted = np.nonzero(train_row)[0]
    exclude[0, interacted] = True

    with torch.no_grad():
        if model_name in ("diffrec", "ldiffrec", "giffcf"):
            x0 = torch.tensor(train_row[None, :], dtype=torch.float32, device=device)
            row_sum = x0.sum(-1, keepdim=True).clamp(min=1)
            x0 = x0 / row_sum
            top_items = model.recommend(x0, top_k=top_k, exclude_mask=exclude)
        elif model_name == "cfdiff":
            x0 = torch.tensor(train_row[None, :], dtype=torch.float32, device=device)
            row_sum = x0.sum(-1, keepdim=True).clamp(min=1)
            top_items = model.recommend(x0 / row_sum, user_tensor,
                                        top_k=top_k, exclude_mask=exclude)
        else:  # neumf, lightgcn, gdmcf
            top_items = model.recommend(user_tensor, top_k=top_k, exclude_mask=exclude)

    top_items_list = top_items[0].cpu().numpy().tolist()

    results = []
    for rank, item_id in enumerate(top_items_list, start=1):
        item = get_book(item_id)
        results.append({
            "rank":    rank,
            "item_id": item_id,
            "score":   round(1.0 - (rank - 1) / top_k, 4),
            "item":    item,
        })

    set_cache(user_id, f"{dataset}:{model_name}", top_k, results)
    return results


# ─── endpoints ──────────────────────────────────────────────────────────────

@app.post("/api/recommend")
def recommend():
    body = request.get_json(force=True)
    user_id    = body.get("user_id")
    top_k      = int(body.get("top_k", 10))
    model_list = body.get("models", VALID_MODELS)
    dataset    = body.get("dataset", "videogames")

    if user_id is None:
        return jsonify({"error": "user_id required"}), 400
    if dataset not in VALID_DATASETS:
        return jsonify({"error": f"unknown dataset: {dataset}"}), 400

    user_id = int(user_id)
    model_list = [m for m in model_list if m in VALID_MODELS]

    results = {}
    errors  = {}
    for name in model_list:
        if not registry.checkpoint_exists(name, dataset):
            errors[name] = "checkpoint not found"
            continue
        try:
            t0 = time.time()
            results[name] = {
                "recommendations": _run_inference(name, user_id, top_k, dataset),
                "latency_ms": round((time.time() - t0) * 1000, 1),
            }
        except Exception as e:
            errors[name] = str(e)

    return jsonify({"user_id": user_id, "top_k": top_k, "dataset": dataset,
                    "results": results, "errors": errors})


@app.get("/api/metrics")
def metrics():
    dataset = request.args.get("dataset", "videogames")
    path = METRICS_PATHS.get(dataset, "metrics.json")
    if not os.path.exists(path):
        return jsonify({"error": f"{path} not found — run evaluate.py first"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.get("/api/users/sample")
def sample_users():
    """Return 20 random users from the test set."""
    users = (User.select(User.internal_id, User.amazon_id, User.n_test)
                 .where(User.n_test > 0)
                 .order_by(User.internal_id)
                 .limit(200))
    sample = list(users.dicts())
    import random; random.shuffle(sample)
    return jsonify(sample[:20])


@app.get("/api/books/<int:item_id>")
def book_detail(item_id: int):
    return jsonify(get_book(item_id))


@app.get("/api/status")
def status():
    dataset = request.args.get("dataset", "videogames")
    return jsonify({
        "models": registry.model_status(dataset),
        "device": str(registry._get_device()),
        "dataset": dataset,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
