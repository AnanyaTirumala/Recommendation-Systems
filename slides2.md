---
marp: true
theme: default
paginate: true
math: mathjax
style: |
  :root {
    --primary: #1c2a3a;
    --accent: #E8620D;
    --accent-light: #FEF3EC;
    --light: #F4F6F9;
    --text: #1e2533;
    --muted: #7a8799;
  }

  section {
    font-family: 'Helvetica Neue', 'Arial', sans-serif;
    background: #ffffff;
    color: var(--text);
    padding: 44px 60px;
    font-size: 17px;
  }

  section.title-slide {
    background: var(--primary);
    color: #ffffff;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
  }

  section.title-slide h1 {
    font-size: 2.0em;
    font-weight: 700;
    border: none;
    margin-bottom: 0.3em;
    color: #ffffff;
    letter-spacing: -0.01em;
  }

  section.title-slide p {
    color: #9aacbc;
    font-size: 0.9em;
    margin-top: 0.5em;
  }

  section.title-slide .subtitle {
    font-size: 1.05em;
    color: var(--accent);
    margin-top: 0.8em;
    font-weight: 600;
    letter-spacing: 0.03em;
  }

  h1 {
    font-size: 1.45em;
    color: var(--primary);
    border-bottom: 3px solid var(--accent);
    padding-bottom: 0.25em;
    margin-bottom: 0.6em;
    font-weight: 700;
    letter-spacing: -0.01em;
  }

  h2 {
    font-size: 0.92em;
    color: var(--accent);
    margin-top: 0.6em;
    margin-bottom: 0.3em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  ul {
    margin: 0.3em 0;
    padding-left: 1.4em;
    line-height: 1.75;
  }

  li {
    margin-bottom: 0.15em;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.84em;
    margin-top: 0.5em;
  }

  th {
    background: var(--primary);
    color: #ffffff;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    letter-spacing: 0.03em;
    font-size: 0.92em;
  }

  td {
    padding: 9px 14px;
    border-bottom: 1px solid #d8dde6;
  }

  tr:nth-child(even) td {
    background: var(--light);
  }

  .highlight {
    background: var(--accent-light);
    border-left: 4px solid var(--accent);
    padding: 10px 18px;
    border-radius: 0 6px 6px 0;
    margin: 0.5em 0;
    font-size: 0.88em;
    color: var(--text);
  }

  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2em;
    margin-top: 0.5em;
  }

  .pipeline {
    background: var(--light);
    border-left: 4px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 16px 24px;
    font-family: 'Courier New', monospace;
    font-size: 0.8em;
    line-height: 1.9;
    margin-top: 0.4em;
    color: var(--primary);
  }

  .badge {
    display: inline-block;
    background: var(--accent);
    color: #fff;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 0.62em;
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-weight: 700;
    vertical-align: middle;
    margin-left: 8px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  .metric-box {
    background: var(--accent-light);
    border: 2px solid var(--accent);
    border-radius: 8px;
    padding: 12px 18px;
    text-align: center;
    font-size: 0.95em;
  }

  .metric-box .value {
    font-size: 1.6em;
    font-weight: 700;
    color: var(--accent);
    display: block;
  }

  section footer {
    font-size: 0.62em;
    color: var(--muted);
    font-family: 'Helvetica Neue', Arial, sans-serif;
  }

  header {
    font-size: 0.62em;
    color: var(--muted);
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
  }
---

<!-- _class: title-slide -->

# Ecommerce Recommendation Systems June

<div class="subtitle">DiffRec Fixes · Yelp Dataset </div>

<p>Ananya Tirumala &nbsp;|&nbsp; June 2026</p>

---

# What Changed — Overview

## Four Areas of Work

<div class="two-col">

<div>

**Model Changes**
- DiffRec: 4 targeted training fixes to address near-zero metrics
- L-DiffRec: LightGCN and Diffrec
- LightGCN: code cleaned up
</div>
<div>

**Infrastructure & Data**
- Multi-dataset support: Yelp added alongside Amazon Video Games
- Checkpoint naming: `{dataset}_{model}.pt` to avoid collisions

**Frontend**
- Dataset toggle (Video Games and Yelp)
- `GameCard` → `ItemCard` (dataset-agnostic)
- Metrics dashboard handles `{dataset}_{model}` checkpoint keys

</div>
</div>

<div class="highlight">
The primary motivation was DiffRec's near-zero Recall@10 (0.00725). 4 fixes raised it to 0.0233 without changing the core architecture.
</div>

---

# DiffRec Fixes — Timestep Embedding & Normalisation <span class="badge">Fix 1 · 2</span>

## Problem: raw scalar timestep + unnormalised interaction vectors

<div class="two-col">

<div>

**Fix 1 — Sinusoidal timestep embedding**

Old: pass raw scalar `t` through `Linear(1→64)`; the gradient signal for the time dimension was weak and poorly conditioned.

New: sinusoidal frequency encoding (same family as transformer positional encoding):

$$\text{emb}(t) = \left[\sin\!\left(\frac{t \cdot \omega_k}{T}\right),\, \cos\!\left(\frac{t \cdot \omega_k}{T}\right)\right]_{k=1}^{32}$$

Then projected: `Linear(64→hidden) → SiLU → Linear(hidden→hidden)`

</div>
<div>

**Fix 2 — L2-normalise interaction rows**

User interaction vectors are binary and extremely sparse. Without normalisation the vector norm scales with the number of interactions, but Gaussian noise $\mathcal{N}(0,I)$ is unit-scale. This mismatch means diffusion adds proportionally tiny noise to active users, so the denoiser never learns a meaningful reverse mapping.

$$x_0^{\text{norm}} = \frac{x_0}{\|x_0\|_2}$$


</div>
</div>

<div class="highlight">
These two fixes address why the denoiser was not learning: the time signal was uninformative and the data scale was incompatible with the noise schedule.
</div>

---

# DiffRec Fixes — Loss & Parameterisation <span class="badge">Fix 3 · 4</span>

## Problem: uniform MSE loss 
<div class="two-col">

<div>

**Fix 3 — Density-adaptive weighted loss**

On sparse binary data, interaction positions (1s) are vastly outnumbered by zeros. Uniform MSE minimises by predicting near-zero everywhere. New loss:

$$\mathcal{L} = \mathbb{E}\!\left[w \cdot \|\hat{x}_0 - x_0^{\text{norm}}\|^2\right]$$

$$w_{ij} = 1 + x_{0,ij} \cdot \min\!\left(\frac{1}{\bar{x}_0},\, 5000\right)$$

Interaction positions receive proportionally higher weight, forcing the model to recover the signal at observed entries.

</div>
</div>

---

# DiffRec Fixes — Inference Alignment & Conditioning <span class="badge">Fix 5 · Cond</span>

## Problem: training/inference distribution mismatch 

<div class="two-col">

<div>

**Fix 4 — Normalise at inference**

Old `recommend()` used raw binary $x_0$ as context but started from pure Gaussian noise $x_T \sim \mathcal{N}(0, I)$. The model was trained on normalised $x_0^{\text{norm}}$, so inference operated outside the training distribution.

New: normalise $x_0$ identically at inference, then corrupt to the correct noise level:

$$x_T = \sqrt{\bar\alpha_{T-1}}\,x_0^{\text{norm}} + \sqrt{1-\bar\alpha_{T-1}}\,\epsilon$$

This means inference starts from a noisy version of the actual user vector, not pure noise.

</div>
</div>


---

# L-DiffRec Changes <span class="badge">Latent Diffusion</span>

## Two changes to the latent variant

<div class="two-col">

<div>

Instead of running diffusion directly on the full n_items-dimensional interaction vector (which can be 5000+ dimensions), L-DiffRec first compresses it to a small latent vector (latent_dim=64), runs diffusion there, then decodes back to item scores

</div>
<div>
Encoder: 
**Linear(n_items→hidden)→GELU→Linear(hidden→64)**

</div>
</div>

---

# Multi-Dataset Support <span class="badge">Yelp</span>

## Amazon Video Games → Video Games + Yelp

<div class="two-col">

<div>

**Data pipeline additions** (`training/data_loader.py`)

- `load_meta_jsonl(meta_path)` — reads Amazon `meta_*.jsonl` files; extracts title, brand/developer, categories, image URL per ASIN
- `load_yelp_interactions(review_path)` — reads Yelp `review.json`; maps `business_id → item_id`, parses ISO date strings to Unix timestamps
- `load_yelp_meta(business_path)` — reads `business.json`; stores business name, city (as author), category list
- `preprocess_yelp(...)` — entry point for Yelp; delegates to shared `_pipeline`
- `_pipeline(df_raw, item_meta, ...)` — unified filter → split → save flow used by both formats

Metadata is now written to `labels_meta.parquet` and item embeddings are looked up from the meta dict rather than left blank.

</div>
<div>

**Path conventions**

| Dataset | Processed | Splits |
|---------|-----------|--------|
| videogames | `data/processed/` | `data/splits/` |
| yelp | `data/yelp/processed/` | `data/yelp/splits/` |

`DATASET_PATHS` dicts in `train_all.py` and `evaluate.py` centralise these.

**Checkpoint naming**

All checkpoints are now prefixed: `{dataset}_{model}.pt`

e.g. `videogames_lightgcn.pt`, `yelp_neumf.pt`

This lets both datasets share a single `checkpoints/` directory without collision, and the evaluation metrics key (`{dataset}_{model}`) matches the checkpoint name exactly.

</div>
</div>

---

# Frontend Changes

## Dataset toggle · ItemCard · Metrics dashboard

<div class="two-col">

<div>

**Dataset toggle** (`App.jsx`)

A pill-style toggle button group lets the user switch between "Amazon Video Games" and "Yelp" without a page reload. Switching clears current results and re-fetches metrics and status for the new dataset. All API calls pass `?dataset=` as a query param.

**`GameCard` → `ItemCard`** (`ResultsGrid.jsx`, new `ItemCard.jsx`)

`GameCard` was hard-coded for book metadata (title, Developer). `ItemCard` is dataset-agnostic: renders title, item ID, developer/city field, and image URL . The component name change reflects the broader scope.

</div>
<div>

**Metrics dashboard** (`MetricsDashboard.jsx`)

Metrics JSON now uses `{dataset}_{model}` keys (e.g. `videogames_lightgcn`). The dashboard strips the dataset prefix via `bareModelId(key)` to look up colours and labels:

Bar chart and table legends show the human-readable model label (e.g. "LightGCN") rather than the raw checkpoint key.

A new **Checkpoint** column in the metrics table shows the full checkpoint filename (`videogames_lightgcn.pt`) for traceability.

</div>
</div>

---

# Serving Changes <span class="badge">Flask API</span>

## Multi-dataset routing in `serving/app.py`

<div class="two-col">

<div>

**Dynamic DB switching**

Old: a single SQLite DB was connected at startup.

New: the DB path is resolved per-request from the `dataset` query param:

```python
DB_PATHS = {
    "videogames": "rec_study_videogames.db",
    "yelp":       "rec_study_yelp.db",
}
```

`before_request` closes any open DB connection, re-initialises with the correct path, then reconnects. This lets a single Flask process serve both datasets without restart.

</div>
<div>

**Endpoint changes**

All endpoints now accept `?dataset=videogames` (default):

| Endpoint | Change |
|----------|--------|
| `POST /api/recommend` | body gains `dataset` field |
| `GET /api/metrics` | reads `metrics_{dataset}.json` |
| `GET /api/users/sample` | queries dataset-specific DB |
| `GET /api/status` | queries dataset-specific model registry |

**`METRICS_PATHS`** maps dataset → JSON file path so the `/api/metrics` handler does not need to know the naming convention:

```python
METRICS_PATHS = {
    "videogames": "metrics_videogames.json",
    "yelp":       "metrics_yelp.json",
}
```

</div>
</div>

---

# Updated Results

| Model | Dataset | Recall@10 | NDCG@10 | MRR |
|-------|---------|:---------:|:-------:|:---:|
| **LightGCN** | Video Games | **0.0625** | **0.0339** | **0.0256** |
| LightGCN | Yelp | 0.0378 | 0.0215 | 0.0227 |
| NeuMF | Video Games | 0.0328 | 0.0172 | 0.0127 |
| NeuMF | Yelp | 0.0285 | 0.0162 | 0.0178 |
| **DiffRec (fixed)** | Video Games | **0.0233** | **0.0118** | **0.0085** |
| L-DiffRec | Video Games | 0.0127 | 0.0067 | 0.0050 |
| DiffRec (old) | Video Games | 0.0073 | 0.0039 | 0.0031 |
| DiffRec | Yelp | 0.0068 | 0.0035 | 0.0032 |

## Key Observations

**DiffRec improved 3× after fixes on Video Games.** Recall@10 rose from 0.0073 → 0.0233. The five-fix package (sinusoidal embedding, L2 normalisation, weighted loss, x0-prediction, inference alignment) collectively resolved the near-zero pathology.

**DiffRec on Yelp mirrors the pre-fix Video Games baseline.** Recall@10 of 0.0068 near-zero, similar to the unfixed Video Games run — suggesting the same pathologies apply on Yelp and the same fixes are needed there.

**LightGCN generalises well across domains.** Recall@10 drops from 0.0625 (Video Games) to 0.0378 (Yelp), a ~40% reduction likely explained by Yelp's lower density and noisier implicit signals (star ratings vs. purchase events), but graph propagation remains the strongest approach on both datasets.

**L-DiffRec still trails DiffRec.** 

---
