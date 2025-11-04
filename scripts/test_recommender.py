#!/usr/bin/env python3
import argparse, os, sys, yaml
from pathlib import Path

from loguru import logger

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from generative_retrieval.model import TigerSeq2Seq
from generative_retrieval.utils import (
    load_codes_and_maps, parse_sequential_data, train_test_split,
    TigerDataset, collate_batch, PrefixIndexer, eval_loss_and_recalls,
    ddp_init, get_rank, seed_all, from_pretrained
)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser("Test recommender")
    ap.add_argument("--eval_max_batches", type=int, default=None, help="override number of batches; None=all")
    ap.add_argument("--model_dir", type=str, help="checkpoint dir (expected to contain .safetensors)")
    args = ap.parse_args()

    cfg = load_config(os.path.join(REPO_ROOT, "configs/recommender.yaml"))
    for k in ("embedding_path", "ids_path", "stats_path", "out_dir"):
        if isinstance(cfg.get(k), str):
            cfg[k] = os.path.abspath(os.path.expanduser(cfg[k]))
    if args.eval_max_batches is not None:
        cfg["eval_max_batches"] = args.eval_max_batches

    local_rank = ddp_init()
    device = torch.device(f"cuda:{local_rank}" if (local_rank >= 0 and torch.cuda.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_all(cfg.get("seed", 42) + (get_rank() if torch.distributed.is_initialized() else 0))
    codes, code2item, vocab_sizes = load_codes_and_maps(cfg["codes_path"], cfg["item_ids_path"])
    prefix_indexer = PrefixIndexer(codes)
    seqs = parse_sequential_data(cfg["sequential_data_path"])
    _, test_pairs = train_test_split(seqs, cfg["max_seq_items"])
    eval_ds = TigerDataset(test_pairs, codes)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_ds, shuffle=False, drop_last=False) if torch.distributed.is_initialized() else None
    eval_dl = torch.utils.data.DataLoader(
        eval_ds, batch_size=cfg["batch_size"], shuffle=False, sampler=eval_sampler,
        collate_fn=lambda b: collate_batch(b, vocab_sizes, add_cls=True), num_workers=2, pin_memory=True
    )

    model, saved_cfg = from_pretrained(TigerSeq2Seq, args.model_dir, device, override_cfg=None)
    if torch.distributed.is_initialized():
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None, output_device=local_rank if device.type == "cuda" else None)

    val_loss, r10, r100, r1000 = eval_loss_and_recalls(
        model=model,
        dl=eval_dl,
        device=device,
        code2item=code2item,
        vocab_sizes= saved_cfg.get("vocab_sizes", vocab_sizes),
        max_batches=cfg.get("eval_max_batches", None),
        prefix_indexer=prefix_indexer,
        cfg=cfg,
    )
    if get_rank() == 0:
        logger.info(f"Test | loss {val_loss:.4f} | R@10 {r10:.4f} | R@100 {r100:.4f} | R@1000 {r1000:.4f}")

if __name__ == "__main__":
    main()
