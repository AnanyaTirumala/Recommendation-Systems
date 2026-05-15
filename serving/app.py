"""
app.py – Flask REST API for the comparative study app.

Endpoints:
  POST /api/recommend     { user_id, top_k, models[] }
  GET  /api/metrics
  GET  /api/users/sample
  GET  /api/books/<item_id>
  GET  /api/status

Run:
  FLASK_APP=serving/app.py flask run --port 5000
  or via VSCode launch config "Flask API"
"""
import json, os, time
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from serving.db import db, Book, User, get_book, get_cached, set_cache
from serving import model_registry as registry

app = Flask(__name__)
CORS(app)

VALID_MODELS = ["neumf","diffrec","ldiffrec","giffcf","cfdiff","gdmcf","lightgcn"]


def _connect_db():
    if db.is_closed():
        db.connect()


@app.before_request
def before():
    _connect_db()


@app.teardown_request
def after(exc):
    if not db.is_closed():
        db.close()


# ─── helpers ────────────────────────────────────────────────────────────────

def _run_inference(model_name: str, user_id: int, top_k: int) -> list:
    """Run one model and return [{rank, item_id, score, book}]."""
    cached = get_cached(user_id, model_name, top_k)
    if cached:
        return cached

    device = registry._get_device()
    model  = registry.get_model(model_name)
    train_row = registry.get_train_row(user_id)
    mappings  = registry.get_mappings()
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

    # Score normalisation: rank-based descending
    results = []
    for rank, item_id in enumerate(top_items_list, start=1):
        book = get_book(item_id)
        results.append({
            "rank":    rank,
            "item_id": item_id,
            "score":   round(1.0 - (rank - 1) / top_k, 4),
            "book":    book,
        })

    set_cache(user_id, model_name, top_k, results)
    return results


# ─── endpoints ──────────────────────────────────────────────────────────────

@app.post("/api/recommend")
def recommend():
    body = request.get_json(force=True)
    user_id    = body.get("user_id")
    top_k      = int(body.get("top_k", 10))
    model_list = body.get("models", VALID_MODELS)

    if user_id is None:
        return jsonify({"error": "user_id required"}), 400
    user_id = int(user_id)
    model_list = [m for m in model_list if m in VALID_MODELS]

    results = {}
    errors  = {}
    for name in model_list:
        ckpt = os.path.join("checkpoints", f"{name}.pt")
        if not os.path.exists(ckpt):
            errors[name] = "checkpoint not found"
            continue
        try:
            t0 = time.time()
            results[name] = {
                "recommendations": _run_inference(name, user_id, top_k),
                "latency_ms": round((time.time() - t0) * 1000, 1),
            }
        except Exception as e:
            errors[name] = str(e)

    return jsonify({"user_id": user_id, "top_k": top_k,
                    "results": results, "errors": errors})


@app.get("/api/metrics")
def metrics():
    path = "metrics.json"
    if not os.path.exists(path):
        return jsonify({"error": "metrics.json not found — run evaluate.py first"}), 404
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
    return jsonify({
        "models": registry.model_status(),
        "device": str(registry._get_device()),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
