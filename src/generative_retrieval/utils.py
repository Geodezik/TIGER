import os
import json
import hashlib
from pathlib import Path
from itertools import islice
from typing import List, Tuple, Dict

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from safetensors.torch import save_file, load_file

import torch.distributed as dist


def sha256_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def flatten_dict(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix == "" else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def ddp_init():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return local_rank
    return -1


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def seed_all(seed: int):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_codes_and_maps(codes_path: str, item_ids_path: str):
    '''
    Loading semantic IDs from rq-quantizer (make sure the first recommender stage is complete)
    '''
    codes = np.load(codes_path).astype(np.int64)
    with open(item_ids_path, "r") as f:
        ids = json.load(f)
        assert len(ids) == codes.shape[0], "Mismatch between ids and codes"

    code2item = {tuple(code.tolist()): int(item_id) for code, item_id in zip(codes, ids)}
    L = codes.shape[1]
    vocab_sizes = []
    for l in range(L):
        vmax = int(np.max(codes[:, l]))
        vocab_sizes.append(vmax + 1)

    return codes, code2item, vocab_sizes


def parse_sequential_data(path: str) -> List[List[int]]:
    '''
    Each line: user_id item_1 item_2 ...
    We convert item ids to 0-based contiguous by subtracting 1.
    '''
    
    seqs = []
    with open(path, "r") as f:
        for line in f:
            arr = [int(x) for x in line.strip().split()]
            items = []
            for j in range(1, len(arr)):
                items.append(arr[j] - 1)
            if len(items) >= 3:
                seqs.append(items)
    return seqs


def train_test_split(seqs: List[List[int]], max_seq_items: int):
    '''
    Performs LOO time-based split on all user histories
    Validation == test to simulate real-world recommenders
    '''

    train, test = [], []
    for items in seqs:
        n = len(items)
        if n < 3:
            continue

        def last_k(seq, k):
            start = max(0, len(seq) - k)
            return seq[start:]

        train_hist = items[:n - 2]
        train.append((last_k(train_hist, max_seq_items), items[n - 2]))

        start_test = max(0, n - (max_seq_items + 1))
        test_hist = items[start_test:n - 1]
        test.append((last_k(test_hist, max_seq_items), items[n - 1]))

    return train, test


def expand_history_to_codes(history_items: List[int], codes: np.ndarray) -> np.ndarray:
    '''
    Produces a sequence of (level_id, token_id) pairs, length = L * len(history)
    '''

    if len(history_items) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    L = codes.shape[1]
    seq = []
    for it in history_items:
        code = codes[it]
        for l in range(L):
            seq.append([l, int(code[l])])
    return np.array(seq, dtype=np.int64)


class TigerDataset(Dataset):
    def __init__(self, pairs: List[Tuple[List[int], int]], codes: np.ndarray):
        self.pairs = pairs
        self.codes = codes

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hist_items, target_item = self.pairs[idx]
        hist_lv_tokens = expand_history_to_codes(hist_items, self.codes)  # (T,2)
        target_code = self.codes[target_item]  # (L,)
        return {
            "hist_lv_tokens": torch.from_numpy(hist_lv_tokens),
            "target_item": torch.tensor(target_item, dtype=torch.long),
            "target_code": torch.from_numpy(target_code.astype(np.int64)),
        }

def collate_batch(batch, vocab_sizes: List[int], add_cls=True):
    # Build padded encoder inputs:
    # - level_ids: (B,T) in [0..L-1] or -1 for padding
    # - token_ids: (B,T) token indices within level vocab (PAD values can be anything when masked)
    # - attn_mask: (B,T) True for real tokens
    L = len(vocab_sizes)
    PAD = [v for v in vocab_sizes] # encoder PAD per level
    CLS = [v + 1 for v in vocab_sizes] # encoder CLS per level

    # Prepare sequences with optional CLS block (L tokens, one per level)
    seqs = []
    for ex in batch:
        seq = ex["hist_lv_tokens"].clone()  # (T,2)
        if add_cls:
            cls_tokens = torch.stack([torch.tensor([l, CLS[l]], dtype=torch.long) for l in range(L)], dim=0)
            seq = torch.cat([cls_tokens, seq], dim=0)
        seqs.append(seq)

    max_len = max(s.size(0) for s in seqs) if len(seqs) > 0 else 0
    B = len(batch)
    level_ids = torch.full((B, max_len), fill_value=-1, dtype=torch.long)
    token_ids = torch.zeros((B, max_len), dtype=torch.long)
    attn_mask = torch.zeros((B, max_len), dtype=torch.bool)

    for i, seq in enumerate(seqs):
        t = seq.size(0)
        level_ids[i, :t] = seq[:, 0]
        token_ids[i, :t] = seq[:, 1]
        attn_mask[i, :t] = True

        # Optional: set PAD token ids on padded slots (not necessary since masked)
        if t < max_len:
            # assign per-level PAD isn't straightforward in a flat tensor; we keep token_ids default zeros since masked
            pass

    return {
        "level_ids": level_ids,
        "token_ids": token_ids,
        "attn_mask": attn_mask,
        "target_code": torch.stack([ex["target_code"] for ex in batch]),
        "target_item": torch.stack([ex["target_item"] for ex in batch]),
        "PAD": PAD,
        "CLS": CLS,
    }


class PrefixIndexer:
    '''
    SID list holder with enabled caching
    Used at validation/inference
    '''

    def __init__(self, codes: np.ndarray):
        self.codes = codes
        self.N, self.L = codes.shape
        self.rows_cache = [dict() for _ in range(self.L + 1)]
        self.allowed_cache = [dict() for _ in range(self.L)]
        self.rows_cache[0][tuple()] = np.arange(self.N, dtype=np.int64)

    def get_rows(self, prefix: tuple) -> np.ndarray:
        t = len(prefix)
        if t == 0:
            return self.rows_cache[0][tuple()]
        if prefix in self.rows_cache[t]:
            return self.rows_cache[t][prefix]
        parent = prefix[:-1]
        last_tok = prefix[-1]
        parent_rows = self.get_rows(parent)
        if parent_rows.size == 0:
            rows = parent_rows
        else:
            rows = parent_rows[self.codes[parent_rows, t - 1] == last_tok]
        self.rows_cache[t][prefix] = rows
        return rows

    def get_allowed(self, t: int, prefix: tuple, device=None):
        if t >= self.L:
            return None
        cache = self.allowed_cache[t]
        if prefix in cache:
            idx = cache[prefix]
        else:
            rows = self.get_rows(prefix)
            if rows.size == 0:
                return None
            vals = np.unique(self.codes[rows, t])
            idx = torch.tensor(vals, dtype=torch.long)
            cache[prefix] = idx
        if device is not None and idx.device != device:
            return idx.to(device, non_blocking=True)
        return idx


def ce_multi_level(logits_list: List[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    loss = 0.0
    for l, logits in enumerate(logits_list):
        loss = loss + nn.functional.cross_entropy(logits, targets[:, l])
    return loss / max(1, len(logits_list))

def causal_mask(L: int, device):
    m = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
    return m

def build_decoder_inputs_for_training(target_code: torch.Tensor, bos_id: int) -> torch.Tensor:
    B, L = target_code.size()
    dec_in = torch.zeros(B, L, dtype=torch.long, device=target_code.device)
    dec_in[:, 0].fill_(bos_id)
    if L > 1:
        dec_in[:, 1:] = target_code[:, :-1]
    return dec_in


def constrained_beam_search(
    model,
    memory: torch.Tensor,
    mem_pad_mask: torch.Tensor,
    L: int,
    bos_id: int,
    beam_size: int,
    per_step_candidates: int,
    device,
    prefix_indexer,
):
    core = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    beams = [((), 0.0)]
    t = 0
    while t < L:
        cur_len = t + 1
        B = len(beams)
        dec_level = torch.arange(0, cur_len, device=device, dtype=torch.long).unsqueeze(0).repeat(B, 1)
        dec_tokens = torch.zeros(B, cur_len, dtype=torch.long, device=device)
        dec_tokens.select(1, 0).fill_(bos_id)
        if t > 0:
            i = 0
            while i < B:
                prefix = beams[i][0]
                if len(prefix) > 0:
                    row = dec_tokens.select(0, i)  # (cur_len,)
                    row.narrow(0, 1, t).copy_(torch.tensor(prefix, dtype=torch.long, device=device))  # positions 1..t
                i += 1

        tgt_mask = causal_mask(cur_len, device)
        mem_exp = memory.expand(B, memory.size(1), memory.size(2))
        mask_exp = mem_pad_mask.expand(B, mem_pad_mask.size(1))

        logits_list = core.decoder(dec_level, dec_tokens, mem_exp, mask_exp, tgt_mask)
        logits_t = logits_list[t]  # (B, Vt)
        Vt = logits_t.size(-1)
        probs = torch.softmax(logits_t, dim=-1)

        new_beams = []
        i = 0
        while i < B:
            if t == 0:
                prefix_tuple = ()
            else:
                row = dec_tokens.select(0, i)
                prev = row.narrow(0, 1, t)
                prefix_tuple = tuple(prev.tolist())

            allowed_idx = prefix_indexer.get_allowed(t, prefix_tuple, device=device)
            p_row = probs.select(0, i)

            if (allowed_idx is not None) and (allowed_idx.numel() > 0):
                idx = allowed_idx
                idx = idx[(idx >= 0) & (idx < Vt)]
                if idx.numel() > 0:
                    p_allowed = p_row.index_select(0, idx)
                    k_i = int(min(per_step_candidates, p_allowed.numel()))
                    if k_i > 0:
                        topv_i, topi_local = torch.topk(p_allowed, k=k_i, dim=-1)
                        toks_i = idx.index_select(0, topi_local)
                        j = 0
                        while j < k_i:
                            tok = int(toks_i[j].item())
                            lp = float(topv_i[j].clamp_min(1e-12).log().item())
                            new_beams.append((beams[i][0] + (tok,), beams[i][1] + lp))
                            j += 1
            else:
                k_i = int(min(per_step_candidates, Vt))
                if k_i > 0:
                    topv_i, toks_i = torch.topk(p_row, k=k_i, dim=-1)
                    j = 0
                    while j < k_i:
                        tok = int(toks_i[j].item())
                        lp = float(topv_i[j].clamp_min(1e-12).log().item())
                        new_beams.append((beams[i][0] + (tok,), beams[i][1] + lp))
                        j += 1

            i += 1

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = list(islice(new_beams, 0, beam_size))
        t += 1

    return beams


def ranks_to_items(samples: List[Tuple[Tuple[int, ...], float]], code2item: Dict[Tuple[int, ...], int], Kmax: int) -> List[int]:
    item_best: Dict[int, float] = {}
    for code, score in samples:
        item = code2item.get(code, -1)
        prev = item_best.get(item)
        if (prev is None) or (score > prev):
            item_best[item] = score
    ranked = sorted(item_best.items(), key=lambda x: x[1], reverse=True)
    ranked_items = [itm for itm, _ in ranked]
    return ranked_items[:Kmax]


def recall_at_k(ranked_items: List[int], target_item: int, k: int) -> float:
    return 1.0 if target_item in ranked_items[:k] else 0.0


def eval_loss_and_recalls(
    model,
    dl,
    device,
    code2item,
    vocab_sizes,
    max_batches=None,
    prefix_indexer=None,
    cfg=None,
):
    model.eval()
    core = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    L = len(vocab_sizes)
    bos_id = max(vocab_sizes)
    want = [10, 100, 1000]
    hits = {k: 0.0 for k in want}
    n_eval = 0
    loss_sum = 0.0
    n_examples = 0

    with torch.no_grad():
        for bi, batch in enumerate(dl):
            if (max_batches is not None) and (bi >= max_batches):
                break

            enc_level_ids = batch["level_ids"].to(device)
            enc_token_ids = batch["token_ids"].to(device)
            enc_attn_mask = batch["attn_mask"].to(device)
            target_items = batch["target_item"].cpu().tolist()
            target_code = batch["target_code"].to(device)

            memory = core.encoder(enc_level_ids, enc_token_ids, enc_attn_mask)
            mem_pad_mask = ~enc_attn_mask
            B = enc_level_ids.size(0)

            dec_in = build_decoder_inputs_for_training(target_code, bos_id)
            dec_level = torch.arange(0, L, device=device, dtype=torch.long).unsqueeze(0).repeat(B, 1)
            tgt_mask = causal_mask(L, device)
            logits_list = core.decoder(dec_level, dec_in, memory, mem_pad_mask, tgt_mask)
            batch_loss = ce_multi_level(logits_list, target_code)
            loss_sum += float(batch_loss.item()) * B
            n_examples += B

            b = 0
            while b < B:
                mem_b = memory.narrow(0, b, 1)
                pad_b = mem_pad_mask.narrow(0, b, 1)
                beams = constrained_beam_search(
                    model=model,
                    memory=mem_b,
                    mem_pad_mask=pad_b,
                    L=L,
                    bos_id=bos_id,
                    beam_size=cfg.get("beam_size", 32),
                    per_step_candidates=cfg.get("per_step_candidates", 32),
                    device=device,
                    prefix_indexer=prefix_indexer,
                )
                ranked = ranks_to_items(beams, code2item, max(want))
                ti = int(target_items[b])
                for k in want:
                    hits[k] += recall_at_k(ranked, ti, k)
                n_eval += 1
                b += 1

    vec = torch.tensor(
        [
            hits.get(10, 0.0),
            hits.get(100, 0.0),
            hits.get(1000, 0.0),
            float(n_eval),
            float(loss_sum),
            float(n_examples),
        ],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)

    denom = max(1.0, float(vec[3].item()))
    loss_denom = max(1.0, float(vec[5].item()))
    val_loss = float(vec[4].item()) / loss_denom
    r10 = float(vec[0].item()) / denom
    r100 = float(vec[1].item()) / denom
    r1000 = float(vec[2].item()) / denom
    return val_loss, r10, r100, r1000


def save_pretrained(model, save_dir: str, cfg: dict):
    """
    HF-like: saves weights + config.json in save_dir.
    Also injects vocab_sizes into config so the model is instantiable.
    """
    os.makedirs(save_dir, exist_ok=True)
    is_ddp = isinstance(model, nn.parallel.DistributedDataParallel)
    core = model.module if is_ddp else model

    export_cfg = dict(cfg)
    if getattr(core, "vocab_sizes", None) is not None:
        export_cfg["vocab_sizes"] = list(core.vocab_sizes)

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(export_cfg, f, indent=2)

    save_file(core.state_dict(), os.path.join(save_dir, "model.safetensors"))


def from_pretrained(model_cls, load_dir: str, device, override_cfg: dict = None):
    """
    HF-like: loads config.json + weights, returns instantiated model on device.
    override_cfg can override some config keys (e.g., beam_size, batch_size) if desired.
    """
    cfg_path = os.path.join(load_dir, "config.json")
    with open(cfg_path, "r") as f:
        saved_cfg = json.load(f)
    if override_cfg:
        saved_cfg = {**saved_cfg, **override_cfg}

    vocab_sizes = saved_cfg["vocab_sizes"]
    model = model_cls(
        vocab_sizes=vocab_sizes,
        d_model=saved_cfg["d_model"],
        n_heads=saved_cfg["n_heads"],
        n_layers_enc=saved_cfg["n_layers_enc"],
        n_layers_dec=saved_cfg["n_layers_dec"],
        dropout=saved_cfg["dropout"],
    ).to(device)

    weights_safe = os.path.join(load_dir, "model.safetensors")
    state = load_file(weights_safe, device="cpu")
    model.load_state_dict(state, strict=True)
    return model, saved_cfg
