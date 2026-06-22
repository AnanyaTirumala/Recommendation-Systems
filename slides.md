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

# E-Commerce Recommendation System Comparisons

<div class="subtitle">NeuMF · LightGCN · DiffRec</div>

<p>Ananya Tirumala &nbsp;|&nbsp; May 2026</p>

---
# Overview

## Problem Statement
Modern e-commerce platforms rely on accurate personalised recommendations to drive engagement. Traditional matrix factorisation methods plateau in quality, while newer graph and diffusion based approaches not adequetly understood in a unified setting.

## Approach
To compare neural, graph and diffusion based models on same user-item matrix
Models to be compared: NeuMF, LightGCN, DiffRec, L-DiffRec

## Key Findings

- **LightGCN** (graph propagation) achieves the strongest overall performance
Recall@10 of **0.0625**, NDCG@10 of **0.0339**
- **NeuMF** (neural matrix factorisation) provides a solid neural baseline
Recall@10 of **0.0328**
- **DiffRec** (Gaussian diffusion on interaction rows) underperforms in current configuration
- Dataset scale and sparsity are the dominant practical bottlenecks

---

# Methodology — NeuMF <span class="badge">Baseline</span>

## Neural Matrix Factorisation (He et al., 2017)

<div class="two-col">

<div>

**GMF Branch**
- Learns separate user and item embedding matrices 
- Captures linear interaction between users and items

**MLP Branch**
- Passes through fully-connected layers: [128 → 64 → 32]
- Captures non-linear interaction patterns

</div>
<div>

**Training**
- Binary cross-entropy loss (implicit feedback)
- 4 negative samples per positive interaction
- Batch size: 1 024 · Epochs: 50 · LR: 1e-3

**Key Hyperparameters**

| Setting | Value |
|---------|-------|
| Embedding dim | 64 |
| MLP layers | [128, 64, 32] |
| Negatives/pos | 4 |
| Dropout | 0.0 |

</div>
</div>

<div class="highlight">
NeuMF acts as the baseline it outperforms pure matrix factorisation by learning non-linear user–item interactions while remaining computationally inexpensive.
</div>

---

# Methodology — LightGCN <span class="badge">Graph</span>

## Light Graph Convolutional Network (He et al., 2020)

Neighbourhood aggregation on the user–item bipartite graph, with no feature transformation and no non-linear activation.

<div class="two-col">

<div>

**Graph Construction**
- Bipartite graph G = (U ∪ I, E)

**Layer Propagation (K = 3)**
$$e^{(k+1)}_u = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}} e^{(k)}_i$$

**Final Embedding**
$$e_u = \frac{1}{K+1}\sum_{k=0}^{K} e^{(k)}_u$$

</div>

<div>

**Training**
- BPR loss: $-\log\sigma(\hat{y}_{ui} - \hat{y}_{uj})$

**Key Hyperparameters**

| Setting | Value |
|---------|-------|
| Embedding dim | 64 |
| GCN layers | 3 |
| Batch size | 2 048 |
| Epochs | 200 |
| LR | 1e-3 |

</div>
</div>

<div class="highlight">
By propagating embeddings over the interaction graph, LightGCN implicitly encodes higher-order user–item relationships (e.g., users who bought the same items share similar neighbours) without any feature engineering.
</div>

---

# Methodology — DiffRec <span class="badge">Diffusion</span>

## Diffusion-Based Recommendation (Wang et al., 2023)

DiffRec applies Gaussian diffusion directly to user interaction rows, treating each user's full item-interaction vector as a data point to corrupt and reconstruct.

<div class="two-col">

<div>

**Forward Process (T = 1 000 steps)**
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t;\, \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I)$$
Noise schedule: **cosine** (Nichol & Dhariwal, 2021)

**MLP Denoiser**
- Input: corrupted interaction vector $x_t$ + timestep embedding
- Architecture: 4 layers, hidden dim = 1 000
- Predicts noise $\epsilon_\theta(x_t, t)$
- Loss: $\mathcal{L} = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$

</div>

<div>

**Inference (DDIM, T_inf = 10)**
- Accelerated reverse diffusion (100× speedup)
- Reconstructs full item score vector
- Training items masked before top-K selection

**Key Hyperparameters**

| Setting | Value |
|---------|-------|
| Denoiser dims | [1000, 600, 300] |
| Diffusion steps T | 1 000 |
| Inference steps | 10 |
| Batch size | 32 |
| Epochs | 30 |

</div>
</div>

<div class="highlight">
DiffRec models the full interaction distribution rather than point-wise scores, offering principled uncertainty handling — though it is sensitive to noise schedule and requires careful tuning on sparse data.
</div>

---


# Dataset Selection

## Why Amazon Fashion Did Not Work

Amazon Fashion has extremely sparse user–item interactions. After K-core filtering (k = 5), the majority of users and items were removed, leaving a dataset too small to train models reliably. The resulting interaction density fell below 0.01%, producing near-zero evaluation metrics.

## Why Amazon Books Did Not Work

Amazon Books is one of the largest product categories available.

- **Memory:** loading the full dense interaction matrix (|U| × |I|) for DiffRec exceeded 16 GB RAM
- **Training time:** 200-epoch LightGCN training required > 8 hours per run, blocking iteration
- **Noise:** the Books category contains significant review noise (bots, bulk reviews), degrading signal quality even after binarisation at rating ≥ 4.0

## Amazon Reviews — Video Games / Digital Products

The Amazon Reviews (Video Games / Digital Products) subset strikes the right balance:

- **Density:** denser interaction graph than Fashion
- **Scale:** manageable size after K-core filtering and fits comfortably in memory
- **Signal quality:** purchase/engagement reviews are more discriminative than book ratings
- **Benchmark precedent:** widely used in the recommender systems literature for fair comparison

---

# Evaluation Metrics

## Why these three?

| Metric | Sensitivity |
|--------|-------------|
| Recall@10 | Set-level coverage |
| NDCG@10 | Rank-weighted relevance |
| MRR | First-hit position |

Together they probe complementary failure modes: a model can achieve high Recall while burying relevant items (low NDCG), or rank one item well while missing others.


---

# Main Results

| Model | Recall@10 | NDCG@10 | MRR | Inference Speed |
|-------|:---------:|:-------:|:---:|:---------------:|
| **LightGCN** | **0.0625** | **0.0339** | **0.0256** | Fast |
| NeuMF | 0.0328 | 0.0172 | 0.0127 | Fastest |
| DiffRec | 0.00725 | 0.00393 | 0.0030 | Slow |


## Key Observations

**LightGCN dominates** across all three metrics. Higher-order graph propagation captures collaborative signals that point-wise models miss.

**NeuMF provides a respectable neural baseline.** The dual GMF+MLP design outperforms pure matrix factorisation. Its low compute cost makes it viable for real-time serving with no quality sacrifice in latency-sensitive settings.

**DiffRec has not converged in the current configuration.** Metrics near zero indicate a training pathology — likely a combination of: (1) the cosine noise schedule being poorly tuned for sparse binary vectors, (2) the 30-epoch budget being insufficient, and (3) the batch size of 32 being too small to estimate stable gradients over the full item vocabulary.

---