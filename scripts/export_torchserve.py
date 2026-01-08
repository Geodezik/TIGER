import argparse, os, sys, json
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from generative_retrieval.model import TigerSeq2Seq
from generative_retrieval.utils import from_pretrained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="e.g. outputs/tiger_encdec/checkpoint-final")
    ap.add_argument("--out_dir", required=True, help="e.g. torchserve_artifacts")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")

    model, cfg = from_pretrained(TigerSeq2Seq, args.model_dir, device=device, override_cfg=None)
    model.eval()

    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))

    model_cfg = {
        "vocab_sizes": cfg["vocab_sizes"],
        "d_model": cfg["d_model"],
        "n_heads": cfg["n_heads"],
        "n_layers_enc": cfg["n_layers_enc"],
        "n_layers_dec": cfg["n_layers_dec"],
        "dropout": cfg["dropout"],
    }
    with open(os.path.join(args.out_dir, "model_config.json"), "w") as f:
        json.dump(model_cfg, f, indent=2)

    print("Saved:", os.path.join(args.out_dir, "model.pt"))
    print("Saved:", os.path.join(args.out_dir, "model_config.json"))


if __name__ == "__main__":
    main()

