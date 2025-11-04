import os
import json
import numpy as np
import torch
import torch.distributed as dist


def init_distributed():
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    # Use env settings if provided (torchrun), else fall back to file:// init for single process
    use_env = ("RANK" in os.environ) or ("WORLD_SIZE" in os.environ) or ("MASTER_ADDR" in os.environ)
    if use_env:
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        init_file = f"file://{os.path.join(tempfile.gettempdir(), 'dist_init_' + str(uuid.uuid4()))}"
        dist.init_process_group(backend=backend, init_method=init_file, rank=0, world_size=1)
        local_rank = 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

def load_embeddings_and_ids(embedding_path, ids_path):
    X = np.load(embedding_path).astype(np.float32)
    with open(ids_path, "r") as f:
        ids = json.load(f)
    if len(ids) != X.shape[0]:
        raise ValueError(f"ids length {len(ids)} != embeddings rows {X.shape[0]}")
    return X, ids

def maybe_standardize(X, stats_path=None, compute_if_missing=True):
    if stats_path and os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            st = json.load(f)
            mean = np.array(st["mean"], dtype=np.float32)
            std = np.array(st["std"], dtype=np.float32)
    elif compute_if_missing:
        mean = X.mean(axis=0, dtype=np.float32)
        std = X.std(axis=0, dtype=np.float32) + 1e-6
    else:
        return X
    return (X - mean) / (std + 1e-6)

def disambiguate_codes(codes: np.ndarray, ids, order="by_id"):
    n, m = codes.shape
    suffix = np.zeros(n, dtype=np.int64)

    groups = {}
    for idx, row in enumerate(codes):
        key = tuple(int(x) for x in row)
        groups.setdefault(key, []).append(idx)

    collisions = {k: v for k, v in groups.items() if len(v) > 1}
    for _, idxs in collisions.items():
        if order == "by_id":
            idxs_sorted = sorted(idxs, key=lambda i: str(ids[i]))
        else:
            idxs_sorted = sorted(idxs)
        for rank, ii in enumerate(idxs_sorted):
            suffix[ii] = rank

    codes_aug = np.concatenate([codes.astype(np.int64), suffix[:, None]], axis=1)

    stats = {
        "n_items": n,
        "levels_before": m,
        "levels_after": m + 1,
        "n_unique_prefixes": len(groups),
        "n_collided_prefixes": len(collisions),
        "max_collision_size": (max((len(v) for v in collisions.values())) if collisions else 1),
        "total_collided_items": sum(len(v) for v in collisions.values()),
    }
    return codes_aug, suffix, stats

def save_outputs(out_dir, ids, codes, codes_aug, suffix, rq_model, cfg, disamb_stats, rank):
    if rank != 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "codes.npy"), codes.astype(np.int64))
    delim = cfg.get("code_delimiter", "-")
    code_strs = [delim.join(map(str, row.tolist())) for row in codes]
    with open(os.path.join(out_dir, "codes_str.txt"), "w", encoding="utf-8") as f:
        for item_id, code in zip(ids, code_strs):
            f.write(f"{item_id}\t{code}\n")

    if cfg.get("append_suffix_layer", True):
        np.save(os.path.join(out_dir, "codes_with_suffix.npy"), codes_aug.astype(np.int64))

        code_strs_aug = [delim.join(map(str, row.tolist())) for row in codes_aug]
        with open(os.path.join(out_dir, "codes_str_with_suffix.txt"), "w", encoding="utf-8") as f:
            for item_id, code in zip(ids, code_strs_aug):
                f.write(f"{item_id}\t{code}\n")

        np.save(os.path.join(out_dir, "collision_suffix.npy"), suffix)

    if cfg.get("save_centroids", True):
        for i, km in enumerate(rq_model.models):
            np.save(os.path.join(out_dir, f"centroids_{i}.npy"), km.cluster_centers_.cpu().numpy())

    summary = {
        "config": cfg,
        "disambiguation": disamb_stats if cfg.get("append_suffix_layer", True) else None,
        "n_items": len(ids),
        "num_ids": cfg["num_ids"],
        "num_clusters": cfg["num_clusters"],
    }
    with open(os.path.join(out_dir, "rqkmeans_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
