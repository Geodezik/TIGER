# Generative Retrieval for Recommendations
This repo implements a full, three-stage, TIGER-style generative retrieval pipeline for large-scale recommendation. Instead of scoring all items with a classifier, we generate an item’s semantic ID (a short code) token-by-token. Generative retrieval has become a strong trend in recommender systems because it:
- scales to huge catalogs (generate a code instead of enumerating millions of IDs),
- supports controllable decoding (prefix constraints, business rules), and can unify retrieval with generation.

We use the Amazon Beauty dataset, Raw version accessable at:
> https://drive.usercontent.google.com/download?id=1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G&export=download&authuser=0

and prioritize recall@100 / recall@1000, which are standard in large-scale settings.

# Model
- Stage 1 (Embedding): encode item text into dense vectors.
- Stage 2 (Semantic IDs): learn residual quantization codebooks (RQKMeans) and assign a multi-level code to each item. Optionally append a suffix level to resolve collisions.
- Stage 3 (Recommender): a lightweight encoder–decoder Transformer (TIGER) trained to generate the next item’s semantic ID from a user’s history. We use constrained beam search (prefix trie built from the catalog codes) to generate only valid items.

# Dataset
Amazon Beauty (product metadata and user sequences).
Sequential split: per-user leave-one-out (train target = n-2; test target = n-1).
Metrics: Recall@10, @100, @1000.

# Target metrics
We aim for the following (starter) goals at time-based split LOO evaluation strategy:
Recall@10   0.020
Recall@100  0.035
Recall@1000 0.050

These metrics are usually well-correlated with AB tests, especially high-k recalls.
The reported values are expected to be achieved with this repo's pipeline and the default configs:
> 2025-11-05 04:20:21.813 | INFO     | __main__:main:68 - Test | loss 3.1594 | R@10 0.0220 | R@100 0.0560 | R@1000 0.0566

We do not implement kv-caching, we do not make an optimistic goal for the response time. Say, 200 ms per user.

We also focus on generation misses which are quite rare for the TIGER model. Say, we want less than 1% of incorrect semantic sequences in production.

Our pipeline is implemented and checked using 2xA100 40GB VRAM and 32 GB of RAM.

# Quickstart
## Data & Models (DVC)

Large files (datasets, embeddings, semantic IDs, trained checkpoints) are tracked with **DVC** and stored outside Git.

DVC remote: **Yandex Object Storage (S3-compatible)**  
Bucket: `tiger`  
Endpoint: `https://storage.yandexcloud.net`

> Credentials are not stored in the repository. To run `dvc pull`, set:
> `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.

Requirements: Python >= 3.11

```bash
git clone <REPO_URL>
cd <REPO_DIR>
python -m pip install -r requirements.txt

export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

dvc pull
dvc repro
```

The pipeline stages are:
```md
## DVC pipeline

- `extract_embeddings`: builds item text embeddings from `data/beauty/meta.json.gz`
- `build_semantics`: residual quantization (RQKMeans) -> multi-level Semantic IDs
- `train`: encoder-decoder Transformer trained to generate the next item Semantic ID
- `evaluate`: evaluates Recall@10/100/1000 and writes metrics to `outputs/metrics/beauty_metrics.json`
```

## Manual run (optional override)

1. Extract embeddings.
Command:
> python ./scripts/extract_embeddings.py --config ./configs/embedder.yaml

Saves:
> outputs/.../item_embeddings_raw.npy

> outputs/.../item_embeddings_l2norm.npy

> outputs/.../item_ids.json

> outputs/.../embedding_stats.json

2. Train semantic IDs.
Command:
> python ./scripts/train_semids.py --config ./configs/quantizer.yaml

Saves:
> outputs/.../codes_with_suffix.npy (semantic IDs per item)

> outputs/.../codes.npy, centroids_*.npy (optional)

> outputs/.../summary.json

3. Train your recommender.
Multi-GPU (example with 2 GPUs 0,1):
> CUDA_VISIBLE_DEVICES="0,1" torchrun --rdzv-endpoint=localhost:1234 --nproc_per_node 2 ./scripts/train_recommender.py --config ./configs/recommender.yaml

Saves HF-style checkpoints:
> outputs/tiger_encdec/checkpoint-*/{config.json, model.safetensors}

4. Test a saved model.
Multi-GPU (example with 2 GPUs 0,1):
> CUDA_VISIBLE_DEVICES="0,1" torchrun --rdzv-endpoint=localhost:1234 --nproc_per_node 2 ./scripts/test_recommender.py --config ./configs/recommender.yaml --model_dir ./outputs/tiger_encdec/checkpoint-600

# Testing and CI
Tests cover key parts of quantization and the model (CPU-only).
GitHub Actions runs the test suite on each push/PR (CI file in .github/workflows/ci.yml).

Example of a failed test because of a broken commit: https://github.com/Geodezik/TIGER/actions/runs/19086493357
