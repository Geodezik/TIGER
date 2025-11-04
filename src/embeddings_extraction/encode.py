import os
import json
from loguru import logger
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from embeddings_extraction.data import build_item_dataframe, prepare_items


def build_encoder(model_name, device):
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device)

    def maybe_prefix(texts):
        name = model_name.lower()
        if "e5" in name:
            return [f"passage: {t}" for t in texts]
        return texts

    return model, maybe_prefix, device


def encode_texts(texts, model, batch_size=256, normalize=True):
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    if normalize:
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        normed = embs / norms
    else:
        normed = embs
    return embs, normed

def run(cfg: dict):
    os.makedirs(cfg["out_dir"], exist_ok=True)

    logger.info("Loading meta...")
    meta_df = build_item_dataframe(cfg["meta_path"])

    logger.info("Preparing items...")
    ids, texts, _ = prepare_items(meta_df, cfg.get("datamaps_path"), cfg["text_template"])
    logger.info(f"Items: {len(ids)}")

    logger.info(f"Loading encoder: {cfg['model_name']}")
    model, prefix_fn, device = build_encoder(cfg["model_name"], cfg.get("device"))
    texts_prefixed = prefix_fn(texts)

    logger.info(f"Encoding on {device}...")
    raw_embs, norm_embs = encode_texts(
        texts_prefixed,
        model,
        batch_size=cfg["batch_size"],
        normalize=True
    )

    logger.info("Saving outputs...")
    np.save(os.path.join(cfg["out_dir"], "item_embeddings_raw.npy"), raw_embs)
    np.save(os.path.join(cfg["out_dir"], "item_embeddings_l2norm.npy"), norm_embs)

    with open(os.path.join(cfg["out_dir"], "item_ids.json"), "w") as f:
        json.dump(ids, f)

    if cfg.get("save_text", True):
        with open(os.path.join(cfg["out_dir"], "item_text.json"), "w") as f:
            json.dump(texts_prefixed, f)

    stats = {
        "mean": raw_embs.mean(axis=0).tolist(),
        "std": raw_embs.std(axis=0).tolist(),
        "shape": list(raw_embs.shape),
        "model_name": cfg["model_name"],
    }
    with open(os.path.join(cfg["out_dir"], "embedding_stats.json"), "w") as f:
        json.dump(stats, f)
