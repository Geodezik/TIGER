#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from embeddings_extraction.encode import run


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config(os.path.join(REPO_ROOT, "configs/embedder.yaml"))
    for k in ("embedding_path", "ids_path", "stats_path", "out_dir"):
        if isinstance(cfg.get(k), str):
            cfg[k] = os.path.abspath(os.path.expanduser(cfg[k]))

    for k in ("meta_path", "datamaps_path", "out_dir"):
        if isinstance(cfg.get(k), str):
            cfg[k] = os.path.abspath(os.path.expanduser(cfg[k]))
    os.makedirs(cfg["out_dir"], exist_ok=True)

    run(cfg)


if __name__ == "__main__":
    main()
