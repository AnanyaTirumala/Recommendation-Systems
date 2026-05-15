<<<<<<< HEAD
# Diffusion Models for E-Commerce Recommendation
### Comparative Study · Amazon Books Dataset · macOS CPU/MPS

| Model | Type | Paper |
|---|---|---|
| NeuMF | Baseline (Neural MF) | He et al., 2017 |
| DiffRec | Gaussian Diffusion CF | Wang et al., 2023 |
| L-DiffRec | Latent Diffusion CF | Wang et al., 2023 |
| GiffCF | Graph Signal Diffusion | Zhu et al., 2024 |
| CF-Diff | Cross-Attention Diffusion | Hou et al., 2024 |
| GDMCF | Graph + Diffusion | 2025 |
| LightGCN | GNN Baseline | He et al., 2020 |

---

## Quick Start (macOS)

```bash
# 1. Clone & environment
conda create -n recsys python=3.11
conda activate recsys
pip install -r requirements.txt

# 2. Download data (HuggingFace — single review file)
# https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
# → raw/review_categories/Books.jsonl
mkdir -p data/raw
# Using huggingface_hub:
#   huggingface-cli download McAuley-Lab/Amazon-Reviews-2023 \
#       raw/review_categories/Books.jsonl --repo-type dataset --local-dir data/raw
# Then move: mv data/raw/raw/review_categories/Books.jsonl data/raw/Books.jsonl

# 3. Preprocess (10% subsample for dev)
python training/data_loader.py \
    --data data/raw/Books.jsonl \
    --output_dir data/processed \
    --subset 0.1

# 4. Init database
python serving/db.py --init --processed_dir data/processed

# 5. Train all models (LightGCN first — L-DiffRec depends on it)
python training/train_all.py --device mps --subset 0.1

# 6. Evaluate
python training/evaluate.py --checkpoint_dir checkpoints/ --output metrics.json

# 7. Start Flask API
FLASK_APP=serving/app.py flask run --port 5000

# 8. Start React frontend (new terminal)
cd frontend && npm install && npm run dev
# → http://localhost:5173
```

## VSCode Launch Configs
Use **Run and Debug** (⇧⌘D) and select:
- `Flask API` — starts backend with debugger attached
- `Train All Models` — full training run
- `Preprocess Data` — runs data_loader.py
- `Evaluate Models` — runs evaluate.py

## Key Files
```
rec_study/
├── models/          # 7 model implementations
├── training/
│   ├── data_loader.py   # Amazon Books preprocessing
│   ├── train_all.py     # Training orchestrator
│   └── evaluate.py      # Recall@K, NDCG@K, MRR
├── serving/
│   ├── app.py           # Flask REST API
│   ├── model_registry.py # Lazy-load LRU cache (max 3 models)
│   └── db.py            # SQLite (users, books, inference cache)
└── frontend/src/        # React + Vite + Recharts
```

## Mac Notes
- MPS device is auto-selected on Apple Silicon
- All models cast to float32 (MPS does not support float64)
- LRU registry keeps max 3 models in RAM at once
- GiffCF uses top-50 sparse Jaccard (not dense M×M)
- L-DiffRec requires `checkpoints/lightgcn_item_emb.pt` — train LightGCN first

## Dataset Notes
- Uses the single `Books.jsonl` review file from HuggingFace (McAuley-Lab/Amazon-Reviews-2023)
- Book title/author/categories are not in the review file; `books_meta.parquet` stores only `asin`, `internal_id`, and `avg_rating` (computed from reviews)
- The UI shows `Book #<id>` for titles since metadata is not available from this file

## Common Gotchas
| Issue | Fix |
|---|---|
| MPS float64 crash | Add `.float()` before `.to(device)` everywhere |
| GiffCF OOM | top_k_graph=50 in config; use subset |
| DiffRec slow inference | Use T_inf=5 (DDIM); T=1000 is training only |
| L-DiffRec KeyError | Train LightGCN first (train_all.py order is fixed) |
| NDCG=0 for all | Check test items exist in books table + valid item_idx |

## API Reference
```
POST /api/recommend   { user_id, top_k, models[] }
GET  /api/metrics
GET  /api/users/sample
GET  /api/books/:id
GET  /api/status
```
=======
# Recommendation-Systems
>>>>>>> afeef9b801e634b40ee5446555d3d1ec70e18bbe
