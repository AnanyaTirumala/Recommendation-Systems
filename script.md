# Presentation Script
## E-Commerce Recommendation System Comparisons

---

## Slide 1 — Title

"Good [morning/afternoon]. Today I'm going to walk you through a comparative study of three recommendation system models applied to e-commerce data.

The three models are NeuMF, which is our neural baseline; LightGCN, a graph-based approach; and DiffRec, which applies Gaussian diffusion to the recommendation problem.

The goal of this study isn't just to rank these models — it's to understand *why* they perform the way they do, what tradeoffs each one makes, and what that means for building real recommendation systems."

---

## Slide 2 — Overview

"Let me start by framing the problem.

E-commerce recommendation is fundamentally a personalisation problem — given a user's interaction history, predict which items they're likely to engage with next. The challenge is that the data is extremely sparse. Most users interact with a tiny fraction of the total item catalogue, so the model has to generalise well from very few signals per user.

Traditional approaches like matrix factorisation are fast and interpretable, but they treat user-item interactions as flat, ignoring the relational structure between users and items. Newer approaches — graph neural networks, diffusion models — can exploit that structure, but they come with their own tradeoffs in compute and data requirements.

Our approach was to implement all three families from scratch, train them on the same dataset under the same evaluation protocol, and measure performance on three complementary metrics.

The key findings, which I'll elaborate on throughout the talk: LightGCN came out clearly on top, with a Recall@10 of 0.0625. NeuMF held up well as a baseline. And DiffRec, in its current configuration, hasn't converged — which is itself an interesting and informative result that I'll come back to."

---

## Slide 3 — Methodology: NeuMF

"Let's start with NeuMF, our baseline model.

NeuMF stands for Neural Matrix Factorisation, and it was introduced by He et al. in 2017. The core idea is to combine two complementary ways of representing user-item affinity.

The first branch is the GMF branch — Generalised Matrix Factorisation. This is the classical approach: you learn a separate embedding vector for each user and each item, and you score a user-item pair by taking the element-wise product of their embeddings, then projecting to a scalar. This is a linear model — it can capture how much a user's tastes align with an item's properties, but it can't capture non-linear interactions.

The second branch is the MLP branch. Here you concatenate the user and item embeddings into a single vector and pass it through a series of fully-connected layers — in our case, dimensions 128, then 64, then 32. This branch learns arbitrary non-linear patterns from the joint user-item representation.

You then concatenate the outputs of both branches and project to a final relevance score. The intuition is that the GMF branch handles the structured, linear part of preference, while the MLP branch handles everything else.

Training uses binary cross-entropy loss on implicit feedback — a rating of 4 or above is a positive, everything else is a negative. We sample 4 negatives per positive to balance the training signal.

In terms of compute: batch size of 1024, 50 epochs, learning rate 1e-3. This is by far the lightest model we trained — it trains in under an hour and inference is a single forward pass."

---

## Slide 4 — Methodology: LightGCN

"LightGCN takes a fundamentally different approach. Instead of treating each user-item pair in isolation, it models the entire interaction graph.

The setup is a bipartite graph where users and items are nodes, and each interaction is an edge. The key insight from He et al. 2020 is that the most useful operation on this graph is neighbourhood aggregation — propagating embeddings from items to users and vice versa. And crucially, they showed that feature transformation and non-linear activation — the standard components of graph convolution — actually *hurt* performance in this setting. So they strip them out entirely.

The propagation rule is straightforward: at each layer, a node's new embedding is the normalised sum of its neighbours' embeddings from the previous layer. The normalisation factor is the geometric mean of the node degrees, which prevents high-degree nodes from dominating.

We run three propagation layers. After layer 0 — which is just the raw initialised embeddings — you get layer 1, layer 2, and layer 3. The final user and item embeddings are the mean across all four of these. This multi-scale aggregation means the final embedding captures both direct neighbours (layer 1) and second- and third-order relationships.

Training uses BPR loss — Bayesian Personalised Ranking — which is a pairwise loss. For each user, you sample a positive item and a negative item, and the loss pushes the score of the positive above the score of the negative. We add L2 regularisation on the raw initial embeddings.

One practical note: running sparse graph operations on Apple Silicon requires some care. Sparse matrix operations aren't fully supported on MPS, so we coalesce them on CPU and then move the result to the GPU for the rest of the forward pass.

This model also plays an important role beyond just its own predictions — the item embeddings it learns are reused as the latent prior for L-DiffRec, the latent diffusion variant."

---

## Slide 5 — Methodology: DiffRec

"DiffRec is the most conceptually novel of the three models, and it draws on the same ideas as image generation models like DDPM and DDIM — but applied to recommendation.

The key abstraction is: instead of treating a user's interaction history as a set of discrete events, treat it as a *data point* — a dense vector of length equal to the item catalogue, where each entry is 1 if the user interacted with that item and 0 otherwise. We want to learn the distribution of these interaction vectors, and then generate personalised recommendations by sampling from that distribution conditioned on the user's history.

The forward process — which only runs during training — adds Gaussian noise to the interaction vector over T=1000 steps, following a cosine noise schedule. By the end, the vector is indistinguishable from pure noise.

The model — an MLP denoiser — is trained to reverse this process: given a noisy vector at timestep t, predict the noise that was added. The MLP takes the noisy vector and a timestep embedding as input, passes through four layers with hidden dimension 1000, and outputs the predicted noise. Loss is mean-squared error between predicted and actual noise.

At inference, we don't need all 1000 steps. Using DDIM — Denoising Diffusion Implicit Models — we can run the reverse process in just 10 steps, which is a 100× speedup. The result is a reconstructed interaction score vector, and we take the top-K items after masking out items the user already interacted with during training.

The reason DiffRec is theoretically appealing is that it models the *full distribution* over items, not just point estimates. This gives it principled uncertainty handling and the ability to generate diverse recommendations. In practice though, it's very sensitive to the noise schedule and training budget — which is part of what we're working through."

---

## Slide 6 — System Architecture: NeuMF Pipeline

"Let me walk through the end-to-end system pipeline for each model, starting with NeuMF.

Raw interaction data comes in as tuples of user ID, item ID, rating, and timestamp. The data loader handles six preprocessing stages: loading, K-core filtering, binarisation, optional subsampling, ID remapping, and temporal splitting.

K-core filtering removes users and items with fewer than 5 interactions — this is standard practice to avoid cold-start users who give us too little signal. Binarisation sets anything with a rating of 4 or above to 1 and discards everything else. The temporal split uses the most recent 10% of interactions per user for validation and test — importantly, this is per-user, not a global split, so every user has representation in all three sets.

The output is train, validation, and test matrices in CSR sparse format, which are efficient to store and iterate over.

For NeuMF, the model takes batches of user-item pairs, passes them through the GMF and MLP branches, and produces scores. After training, the checkpoint is saved and registered in the model registry.

The model registry uses lazy loading with LRU eviction — we can only keep 3 models in RAM simultaneously on a 16 GB machine, so models are loaded on first request and the least recently used is evicted when we need space. The Flask API exposes a `/api/recommend` endpoint, and results are cached in SQLite to avoid recomputing for repeated queries.

NeuMF is the fastest model to serve — it's just two embedding lookups and a few matrix multiplications."

---

## Slide 7 — System Architecture: LightGCN Pipeline

"LightGCN's pipeline has one key difference from NeuMF at the data stage: we need to build the bipartite adjacency matrix explicitly.

After preprocessing, we construct the user-item adjacency matrix A and compute its symmetric normalisation, which we store as a sparse COO tensor. This normalised matrix is what gets multiplied at each graph propagation layer.

During training, we sample triplets — user, positive item, negative item — and compute BPR loss on their scores. The graph propagation runs once per batch to compute all node embeddings, then we index into those for the sampled users and items.

One architectural point worth noting: the training matrix is *not* baked into the model checkpoint. It's passed to the model at load time by the registry. This is important because serving requires access to the adjacency structure — without it, you can't compute embeddings for new queries. The registry is graph-aware and knows to pass the training matrix when it loads LightGCN or any other graph-based model.

LightGCN's item embeddings are also exported separately for use by L-DiffRec, which uses them to initialise its latent space."

---

## Slide 8 — System Architecture: DiffRec Pipeline

"DiffRec's pipeline is the most different from the other two because the model operates on full user interaction rows rather than sampled pairs.

During training, we take a batch of users, fetch their full interaction vectors — dense vectors of length n_items — and run the forward diffusion process to corrupt them. The denoiser then predicts the noise, and we compute MSE loss.

The key challenge here is scale: n_items can be in the tens of thousands. This means the interaction vectors are large, and the denoiser has to handle vectors that grow with the catalogue. On Mac hardware with limited RAM, this forced us to use a batch size of just 32.

At inference, we run the DDIM reverse process in 10 steps. Each step calls the denoiser once, so inference for a single user requires 10 forward passes through a 4-layer MLP. This is slower than NeuMF's single forward pass but acceptable for pre-computed recommendations.

After reconstruction, we apply an exclusion mask — we zero out all items the user has already seen during training — then take the top-K items by score. This is the recommendation list returned to the API."

---

## Slide 9 — Dataset Selection

"Choosing the right dataset was actually one of the harder parts of this project, and I want to be transparent about why we went through two datasets before settling on the current one.

We started with Amazon Fashion. Intuitively it seemed like a good fit for an e-commerce recommendation study — fashion is a domain where personalisation clearly matters. But the data has a fundamental sparsity problem. People don't buy fashion items frequently, and they rarely buy many different items from the same category. After K-core filtering with k=5 — which requires every user and item to have at least 5 interactions — we were removing the vast majority of the dataset. What remained was too small to train meaningful models on.

Next we tried Amazon Books. Books has the opposite problem — it's enormous. We had over 63,000 test users, but that came with serious practical issues. Loading the full dense interaction matrix for DiffRec exceeded 16 GB of RAM. Training LightGCN for 200 epochs took over 8 hours per run, which made iteration nearly impossible. And there was a data quality issue — the books category has a lot of review noise from bots and bulk reviewers, which degrades the signal quality even after binarisation.

Amazon Reviews — specifically the Video Games and Digital Products subset — solves both problems. The interaction graph is meaningfully denser than Fashion because gamers tend to buy multiple titles and leave more consistent reviews. The scale is manageable — it fits in memory and trains in a reasonable time. The signal quality is high because gaming reviews are more deliberate. And importantly, this dataset is widely used in the recommender systems literature, which means we can benchmark our results against published numbers."

---

## Slide 10 — Evaluation Metrics

"Before I show you the results, let me explain what we're measuring and why these three metrics together give a complete picture.

All three models are evaluated under the same protocol: leave-one-out with 99 random negatives. That means for each test user, we take their most recent interaction as the ground truth positive item, randomly sample 99 items they haven't interacted with, and ask the model to rank all 100. We then evaluate whether and where the positive item appears in the ranked list.

Recall at 10 asks: is the positive item anywhere in the top 10? It measures coverage — whether the model is able to surface the relevant item at all. This is the most direct measure of whether a user would find what they're looking for.

NDCG at 10 — Normalised Discounted Cumulative Gain — adds position sensitivity. An item at rank 1 contributes much more to NDCG than an item at rank 10, because in a real UI, users are far more likely to click the first result. The normalisation ensures NDCG is between 0 and 1 regardless of list length.

MRR — Mean Reciprocal Rank — focuses specifically on where the *first* relevant item appears. It's 1 divided by that rank, averaged across all users. This is most relevant for scenarios where you only really care about getting one thing right.

The reason we use all three is that they probe different failure modes. A model can rank the positive item in the top 10 — high Recall — but push it to position 9 or 10, which would give low NDCG. Conversely, a model might rank one item perfectly but fail to surface anything else. Looking at all three together tells a more complete story."

---

## Slide 11 — Main Results and Key Observations

"Here are the results.

LightGCN is the clear winner across all three metrics. Recall@10 of 0.0625, NDCG@10 of 0.0339, MRR of 0.0256. These numbers might sound small in absolute terms, but remember — we're ranking the positive item against 99 randomly sampled negatives. A Recall@10 of 0.0625 means the model finds the right item in the top 10 about 6.25% of the time across all test users. That's meaningful signal for a system operating at catalogue scale.

NeuMF comes in second with roughly half of LightGCN's performance — Recall@10 of 0.0328. This is actually a reasonable result for a model that doesn't use any graph structure. The dual GMF+MLP design does capture useful non-linear patterns, and the training cost is a fraction of LightGCN's.

DiffRec's numbers are near zero across all metrics. Recall of 0.000389, NDCG of 0.000152. I want to be careful not to interpret this as 'diffusion models don't work for recommendation,' because that's not what this shows.

What this shows is that DiffRec, in its current configuration, has not converged. There are three likely culprits. First, the cosine noise schedule, which was designed for image generation, may not be well-suited to sparse binary interaction vectors. Second, 30 epochs is almost certainly not enough — the original paper trained for substantially longer. Third, a batch size of 32 means gradient estimates are very noisy, especially when the interaction vectors are high-dimensional. We're essentially asking the model to learn a complex distribution from very small batch statistics.

The key takeaway is that the performance gap between LightGCN and DiffRec is a tuning gap, not a fundamental architectural limitation. DiffRec outperforms LightGCN on MovieLens in the original paper with proper configuration."

---

## Slide 12 — Path Forward

"So where do we go from here?

The most pressing priority is fixing DiffRec's convergence. Concretely, that means switching from cosine to linear noise schedule, which is more appropriate for discrete data; increasing the training budget to at least 100 epochs with early stopping based on validation NDCG; and experimenting with the noise scale, which controls how aggressively the forward process corrupts the interaction vector.

Second, we need to evaluate L-DiffRec. The latent diffusion variant is fully implemented — it compresses the interaction vector into a 64-dimensional latent space using an encoder, runs diffusion in that compressed space, and uses LightGCN's item embeddings as the initialisation. This is expected to outperform standard DiffRec because it operates in a much lower-dimensional space, which makes the denoising problem more tractable.

In the medium term, we want to migrate the full comparison to the Amazon Video Games dataset and run a proper hyperparameter sweep — varying diffusion steps, noise schedule, hidden dimensions, and GCN depth systematically.

Longer term, the most interesting direction is the hybrid models — GiffCF, CF-Diff, and GDMCF — which combine graph propagation with diffusion. These are all implemented in the codebase and ready to train. The hypothesis is that they'll capture the best of both worlds: LightGCN's structural inductive bias and DiffRec's generative uncertainty modelling.

On the serving side, we're also planning to move from the current Flask + SQLite setup to async serving with pre-computed recommendation caches for high-traffic users, which will bring inference latency down to single-digit milliseconds.

That's the full picture. Happy to go deeper on any of the models, the dataset decisions, or the evaluation setup — what questions do you have?"

---

*End of script.*
