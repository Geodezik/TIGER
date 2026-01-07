# src/predict.py
import argparse
import csv
import os
from typing import List, Tuple
from loguru import logger

import numpy as np
import torch

from generative_retrieval.model import TigerSeq2Seq
from generative_retrieval.utils import (
    load_codes_and_maps,
    PrefixIndexer,
    constrained_beam_search,
    ranks_to_items,
    from_pretrained,
)


def parse_txt(path: str) -> Tuple[List[str], List[List[int]]]:
    user_ids, histories = [], []
    with open(path, "r") as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) < 2:
                continue
            user_id = arr[0]
            items_1based = [int(x) for x in arr[1:]]
            hist = [x - 1 for x in items_1based if x > 0]
            user_ids.append(user_id)
            histories.append(hist)
    return user_ids, histories


def parse_csv(path: str) -> Tuple[List[str], List[List[int]]]:
    user_ids, histories = [], []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            user_id = row.get("user_id", str(i))
            s = row.get("item_ids") or row.get("history") or ""
            items_1based = [int(x) for x in s.strip().split()] if s.strip() else []
            hist = [x - 1 for x in items_1based if x > 0]
            user_ids.append(user_id)
            histories.append(hist)
    return user_ids, histories


def load_inputs(path: str) -> Tuple[List[str], List[List[int]]]:
    if path.endswith(".csv"):
        return parse_csv(path)
    return parse_txt(path)


def build_encoder_batch(histories: List[List[int]], codes: np.ndarray, vocab_sizes: List[int], max_seq_items: int) -> dict:
    L = len(vocab_sizes)
    CLS = [v + 1 for v in vocab_sizes]

    seqs = []
    for hist in histories:
        hist = hist[-max_seq_items:]
        lv_tok = []
        for l in range(L):
            lv_tok.append((l, CLS[l]))
        for it in hist:
            code = codes[it]
            for l in range(L):
                lv_tok.append((l, int(code[l])))
        seqs.append(lv_tok)

    max_len = max(len(s) for s in seqs) if seqs else 0
    B = len(seqs)

    level_ids = torch.full((B, max_len), fill_value=-1, dtype=torch.long)
    token_ids = torch.zeros((B, max_len), dtype=torch.long)
    attn_mask = torch.zeros((B, max_len), dtype=torch.bool)

    for i, seq in enumerate(seqs):
        t = len(seq)
        if t == 0:
            continue
        level_ids[i, :t] = torch.tensor([x[0] for x in seq], dtype=torch.long)
        token_ids[i, :t] = torch.tensor([x[1] for x in seq], dtype=torch.long)
        attn_mask[i, :t] = True

    return {"level_ids": level_ids, "token_ids": token_ids, "attn_mask": attn_mask}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--output_path", required=True)

    ap.add_argument("--topk", type=int, default=128)
    ap.add_argument("--beam_size", type=int, default=256)
    ap.add_argument("--per_step_candidates", type=int, default=64)

    ap.add_argument("--max_seq_items", type=int, default=64)

    ap.add_argument("--model_dir", type=str, default="/app/artifacts/model_final")
    ap.add_argument("--codes_path", type=str, default="/app/artifacts/codes_with_suffix.npy")
    ap.add_argument("--item_ids_path", type=str, default="/app/artifacts/item_ids.json")

    ap.add_argument("--device", type=str, default=None, help="cpu | cuda | cuda:0")
    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    codes, code2item, vocab_sizes = load_codes_and_maps(args.codes_path, args.item_ids_path)
    prefix_indexer = PrefixIndexer(codes)
    model, saved_cfg = from_pretrained(TigerSeq2Seq, args.model_dir, device, override_cfg=None)
    model.eval()
    user_ids, histories = load_inputs(args.input_path)
    if len(histories) == 0:
        raise RuntimeError("No valid rows found in input")

    batch = build_encoder_batch(histories, codes, vocab_sizes, max_seq_items=args.max_seq_items)
    enc_level_ids = batch["level_ids"].to(device)
    enc_token_ids = batch["token_ids"].to(device)
    enc_attn_mask = batch["attn_mask"].to(device)

    core = model
    L = len(vocab_sizes)
    bos_id = max(vocab_sizes)

    with torch.no_grad():
        memory = core.encoder(enc_level_ids, enc_token_ids, enc_attn_mask)
        mem_pad_mask = ~enc_attn_mask

        preds: List[List[int]] = []
        for i in range(len(histories)):
            mem_i = memory[i:i+1]
            pad_i = mem_pad_mask[i:i+1]
            beams = constrained_beam_search(
                model=core,
                memory=mem_i,
                mem_pad_mask=pad_i,
                L=L,
                bos_id=bos_id,
                beam_size=args.beam_size,
                per_step_candidates=args.per_step_candidates,
                device=device,
                prefix_indexer=prefix_indexer,
            )
            ranked_items0 = ranks_to_items(beams, code2item, args.topk)
            ranked_items1 = [(x + 1) for x in ranked_items0 if x >= 0]
            ranked_items1 = (ranked_items1 + ["" for _ in range(args.topk)])[:args.topk]
            preds.append(ranked_items1)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["user_id"] + [f"pred_{j+1}" for j in range(args.topk)]
        w.writerow(header)
        for uid, row in zip(user_ids, preds):
            w.writerow([uid] + row)

    logger.info(f"Wrote predictions to: {args.output_path}")


if __name__ == "__main__":
    main()
