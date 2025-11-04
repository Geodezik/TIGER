#!/usr/bin/env python3
import os
from pathlib import Path
import sys
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from generative_retrieval.training_pipeline import run


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
        return cfg


def main():
    cfg = load_config(os.path.join(REPO_ROOT, "configs/recommender.yaml"))
    for key in ("sequential_data_path", "codes_path", "item_ids_path", "out_dir"):
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = os.path.abspath(os.path.expanduser(cfg[key]))

    os.makedirs(cfg.get("out_dir", "./outputs"), exist_ok=True)
    run(cfg)


if __name__ == "__main__":
    main()
