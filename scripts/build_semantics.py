#!/usr/bin/env python3
import argparse
import os
import sys
import time
import yaml
from pathlib import Path

import numpy as np

import torch
import torch.distributed as dist

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from quantization.rqkmeans import RQKMeans
from quantization.utils import init_distributed, load_embeddings_and_ids, maybe_standardize, disambiguate_codes, save_outputs


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run(cfg):
    t0 = time.time()
    logger.info("Initializing distributed...")
    init_distributed()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the provided KMeans implementation (device fixed to 'cuda').")

    logger.info("Loading embeddings and ids...")
    X_np, ids = load_embeddings_and_ids(cfg["embedding_path"], cfg["ids_path"])
    logger.info(f"Embeddings shape: {X_np.shape}")

    if cfg.get("standardize", True):
        X_np = maybe_standardize(X_np, cfg.get("stats_path"), compute_if_missing=True)
        logger.info("Applied standardization.")
    if cfg.get("l2_normalize", False):
        norms = np.linalg.norm(X_np, axis=1, keepdims=True) + 1e-12
        X_np = X_np / norms
        logger.info("Applied L2 normalization.")

    X = torch.from_numpy(X_np).float()

    logger.info("Fitting RQKMeans...")
    rq = RQKMeans(
        num_clusters=cfg["num_clusters"],
        num_ids=cfg["num_ids"],
        batch_size=cfg["batch_size"],
        window_size=cfg.get("window_size", 32),
        max_iter=cfg["max_iter"],
        tol=cfg["tol"],
        verbose=cfg.get("verbose", 1),
        random_state=cfg.get("random_state", 42),
        gpu_preload=cfg.get("gpu_preload", True),
    )
    rq.fit(X)

    logger.info("Predicting codes...")
    codes_t = rq.predict(X)
    codes = codes_t.cpu().numpy().astype(np.int64)

    disamb_stats = None
    codes_aug, suffix = None, None
    if cfg.get("append_suffix_layer", True):
        codes_aug, suffix, disamb_stats = disambiguate_codes(codes, ids, order=cfg.get("suffix_order", "by_id"))
        if dist.get_rank() == 0:
            logger.info(f"Disambiguation: {disamb_stats}")

    logger.info("Saving outputs...")
    save_outputs(cfg["out_dir"], ids, codes, codes_aug, suffix, rq, cfg, disamb_stats, dist.get_rank())

    if dist.is_initialized():
        dist.destroy_process_group()

    logger.info(f"Done in {time.time() - t0:.2f}s. Saved to: {cfg['out_dir']}")


def main():
    cfg = load_config(os.path.join(REPO_ROOT, "configs/quantizer.yaml"))
    for k in ("embedding_path", "ids_path", "stats_path", "out_dir"):
        if isinstance(cfg.get(k), str):
            cfg[k] = os.path.abspath(os.path.expanduser(cfg[k]))
    os.makedirs(cfg["out_dir"], exist_ok=True)

    run(cfg)


if __name__ == "__main__":
    main()