import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from model import TigerSeq2Seq
from utils import load_codes_and_maps, PrefixIndexer, constrained_beam_search, ranks_to_items


class TigerHandler:
    def initialize(self, context):
        model_dir = context.system_properties.get("model_dir")
        cfg_path = os.path.join(model_dir, "model_config.json")
        codes_path = os.path.join(model_dir, "codes_with_suffix.npy")
        item_ids_path = os.path.join(model_dir, "item_ids.json")
        weights_path = os.path.join(model_dir, "model.pt")

        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        self.vocab_sizes = cfg["vocab_sizes"]
        self.model = TigerSeq2Seq(
            vocab_sizes=self.vocab_sizes,
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers_enc=cfg["n_layers_enc"],
            n_layers_dec=cfg["n_layers_dec"],
            dropout=cfg["dropout"],
        )

        state = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        self.codes, self.code2item, _ = load_codes_and_maps(codes_path, item_ids_path)
        self.prefix_indexer = PrefixIndexer(self.codes)

        self.L = len(self.vocab_sizes)
        self.bos_id = max(self.vocab_sizes)

    def _predict(self, data, context):
        body = data[0].get("body") or data[0].get("data")
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")
        payload = json.loads(body) if isinstance(body, str) else body

        instances = payload.get("instances", payload)
        topk = int(payload.get("topk", 4))
        beam_size = int(payload.get("beam_size", 4))
        per_step_candidates = int(payload.get("per_step_candidates", 4))
        max_seq_items = int(payload.get("max_seq_items", 8))

        histories, user_ids = [], []
        for i, ex in enumerate(instances):
            uid = ex.get("user_id", str(i))
            items_1b = ex.get("item_ids", [])
            hist = [int(x) - 1 for x in items_1b if int(x) > 0]
            histories.append(hist[-max_seq_items:])
            user_ids.append(uid)

        CLS = [v + 1 for v in self.vocab_sizes]
        seqs = []
        for hist in histories:
            lv_tok = [(l, CLS[l]) for l in range(self.L)]
            for it in hist:
                code = self.codes[it]
                for l in range(self.L):
                    lv_tok.append((l, int(code[l])))
            seqs.append(lv_tok)

        max_len = max(len(s) for s in seqs) if seqs else 0
        B = len(seqs)

        level_ids = torch.full((B, max_len), -1, dtype=torch.long)
        token_ids = torch.zeros((B, max_len), dtype=torch.long)
        attn_mask = torch.zeros((B, max_len), dtype=torch.bool)

        for i, seq in enumerate(seqs):
            t = len(seq)
            if t == 0:
                continue
            level_ids[i, :t] = torch.tensor([x[0] for x in seq], dtype=torch.long)
            token_ids[i, :t] = torch.tensor([x[1] for x in seq], dtype=torch.long)
            attn_mask[i, :t] = True

        with torch.no_grad():
            memory = self.model.encoder(level_ids, token_ids, attn_mask)
            mem_pad_mask = ~attn_mask

            outputs = []
            for i in range(B):
                beams = constrained_beam_search(
                    model=self.model,
                    memory=memory[i:i+1],
                    mem_pad_mask=mem_pad_mask[i:i+1],
                    L=self.L,
                    bos_id=self.bos_id,
                    beam_size=beam_size,
                    per_step_candidates=per_step_candidates,
                    device='cpu',
                    prefix_indexer=self.prefix_indexer,
                )
                ranked0 = ranks_to_items(beams, self.code2item, topk)
                ranked1 = [x + 1 for x in ranked0 if x >= 0]
                outputs.append(ranked1[:topk])

        return user_ids, outputs

    def handle(self, data, context):
        responses = []

        for req in data:
            user_ids, preds = self._predict(data, context)
            responses.append({
                "predictions": [
                    {"user_id": uid, "predictions": pred}
                    for uid, pred in zip(user_ids, preds)
                ]
            })

        return responses

