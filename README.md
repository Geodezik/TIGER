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
- `extract_embeddings`: builds item text embeddings from `data/beauty/meta.json.gz`
- `build_semantics`: residual quantization (RQKMeans) -> multi-level Semantic IDs
- `train`: encoder-decoder Transformer trained to generate the next item Semantic ID
- `evaluate`: evaluates Recall@10/100/1000 and writes metrics to `outputs/metrics/beauty_metrics.json`
```

## Experiment tracking (MLflow)

MLflow runs are stored locally in `./mlruns`.

Start UI:
```bash
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
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

# Docker (offline CPU inference)

## Build image
Before building, make sure model artifacts are present locally (either via `dvc pull` or copied manually).

```bash
docker build -t ml-app:v1 .
```

## Run inference

Create a text file where each line is one user history.
Format:

```text
user_id item_1 item_2 item_3 ...
```

Example (sample_input.txt):

```txt
1 1 2 3 4 5
2 6 7 8 9 10 4
3 83 109 110 111
```

```bash
mkdir -p out

docker run --rm \
  -v "$(pwd)/sample_input.txt:/app/input.txt:ro" \
  -v "$(pwd)/out:/app/out" \
  ml-app:v1 \
  --input_path /app/input.txt \
  --output_path /app/out/preds.csv \
  --topk 128
```

The container writes preds.csv with K columns.

# TorchServe (online inference)

## Build model archive (.mar)
```bash
python scripts/export_torchserve.py \
  --model_dir outputs/tiger_encdec/checkpoint-final \
  --out_dir torchserve_artifacts

cp outputs/beauty_rqkmeans/codes_with_suffix.npy torchserve_artifacts/
cp outputs/beauty_minilm/item_ids.json torchserve_artifacts/

mkdir -p model-store
torch-model-archiver \
  --model-name mymodel \
  --version 1.0 \
  --serialized-file torchserve_artifacts/model.pt \
  --handler torchserve/handler.py \
  --extra-files "torchserve_artifacts/model_config.json,torchserve_artifacts/codes_with_suffix.npy,torchserve_artifacts/item_ids.json,torchserve_artifacts/model.py,torchserve_artifacts/utils.py" \
  --export-path model-store \
  --force
```

## Build and run server + test predict
```bash
docker build -t mymodel-serve:v1 -f torchserve/Dockerfile .
docker run -d --name tiger-ts -p 8080:8080 -p 8081:8081 mymodel-serve:v1

curl -X POST "http://localhost:8080/predictions/mymodel" \
  -H "Content-Type: application/json" \
  --data-binary @torchserve/sample_input.json
```

The above command uses sample_input.json, it can look like this:
```text
{
  "topk": 4,
  "instances": [
    {"user_id": "1", "item_ids": [1, 2, 3, 4]},
    {"user_id": "2", "item_ids": [6, 7, 8]}
  ]
}
```

From curl we expect an output like this:
```text
{
  "predictions": [
    {
      "user_id": "1",
      "predictions": [
        44,
        39,
        10521,
        3288
      ]
    },
    {
      "user_id": "2",
      "predictions": [
        4313,
        5367,
        1667,
        510
      ]
    }
  ]
}
```

# Testing and CI
Tests cover key parts of quantization and the model (CPU-only).
GitHub Actions runs the test suite on each push/PR (CI file in .github/workflows/ci.yml).

Example of a failed test because of a broken commit: https://github.com/Geodezik/TIGER/actions/runs/19086493357
