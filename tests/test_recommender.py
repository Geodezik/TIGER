import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from generative_retrieval.model import (
    MultiLevelEmbedding,
    Encoder,
    Decoder,
    TigerSeq2Seq
)
from generative_retrieval.utils import (
    ce_multi_level,
    causal_mask,
    PrefixIndexer,
    constrained_beam_search,
    ranks_to_items,
    recall_at_k
)

def _make_level_and_tokens(B=2, T=6, vocab_sizes=(5, 7)):
    L = len(vocab_sizes)
    level_ids = torch.empty(B, T, dtype=torch.long)
    token_ids = torch.empty(B, T, dtype=torch.long)
    for b in range(B):
        for t in range(T):
            l = int(torch.randint(0, L, ()).item())
            v = vocab_sizes[l]
            level_ids[b, t] = l
            token_ids[b, t] = int(torch.randint(0, v, ()).item())
    attn_mask = torch.ones(B, T, dtype=torch.bool)
    return level_ids, token_ids, attn_mask

def test_encoder_shapes():
    torch.manual_seed(0)
    vocab_sizes = [5, 7]
    d_model = 16
    B, T = 2, 6

    level_ids, token_ids, attn_mask = _make_level_and_tokens(B=B, T=T, vocab_sizes=vocab_sizes)

    emb = MultiLevelEmbedding(vocab_sizes=vocab_sizes, d_model=d_model, dropout=0.0, add_pad_cls=True)
    out = emb(level_ids, token_ids)
    assert out.shape == (B, T, d_model)

    enc = Encoder(vocab_sizes=vocab_sizes, d_model=d_model, n_heads=2, n_layers=1, dropout=0.0)
    h = enc(level_ids, token_ids, attn_mask)
    assert h.shape == (B, T, d_model)


def test_decoder_shapes():
    torch.manual_seed(0)
    vocab_sizes = [4, 3, 5]
    d_model = 16
    B = 2
    Tenc = 5
    L = len(vocab_sizes)

    memory = torch.randn(B, Tenc, d_model)
    mem_pad_mask = torch.zeros(B, Tenc, dtype=torch.bool)

    dec_level_ids = torch.arange(0, L, dtype=torch.long).unsqueeze(0).repeat(B, 1)
    bos_id = max(vocab_sizes)
    dec_token_ids = torch.zeros(B, L, dtype=torch.long)
    dec_token_ids[:, 0] = bos_id
    tgt_mask = causal_mask(L, device=dec_level_ids.device)

    dec = Decoder(vocab_sizes=vocab_sizes, d_model=d_model, n_heads=2, n_layers=1, dropout=0.0)
    logits_list = dec(dec_level_ids, dec_token_ids, memory, mem_pad_mask, tgt_mask)
    assert isinstance(logits_list, list) and len(logits_list) == L
    for l, logits in enumerate(logits_list):
        assert logits.shape == (B, vocab_sizes[l])


def test_tiger_training():
    torch.manual_seed(0)
    vocab_sizes = [4, 3]
    d_model = 16
    B, Tenc = 2, 6
    L = len(vocab_sizes)
    bos_id = max(vocab_sizes)

    level_ids, token_ids, attn_mask = _make_level_and_tokens(B=B, T=Tenc, vocab_sizes=vocab_sizes)

    dec_level = torch.arange(0, L, dtype=torch.long).unsqueeze(0).repeat(B, 1)
    target_code = torch.stack([torch.randint(0, v, (B,), dtype=torch.long) for v in vocab_sizes], dim=1)
    dec_in = torch.zeros(B, L, dtype=torch.long)
    dec_in[:, 0] = bos_id
    if L > 1:
        dec_in[:, 1:] = target_code[:, :-1]
    tgt_mask = causal_mask(L, device=dec_level.device)

    model = TigerSeq2Seq(vocab_sizes=vocab_sizes, d_model=d_model, n_heads=2, n_layers_enc=1, n_layers_dec=1, dropout=0.0)
    logits_list = model(level_ids, token_ids, attn_mask, dec_level, dec_in, tgt_mask)
    loss = ce_multi_level(logits_list, target_code)
    assert loss.ndim == 0
    loss.backward()
    has_grad = any((p.grad is not None) and torch.isfinite(p.grad).all() for p in model.parameters())
    assert has_grad


def test_generation():
    codes = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int64)
    indexer = PrefixIndexer(codes)
    a0 = indexer.get_allowed(0, (), device=None)
    assert set(a0.tolist()) == {0, 1}
    a1 = indexer.get_allowed(1, (1,), device=None)
    assert set(a1.tolist()) == {0, 1}

    vocab_sizes = [2, 2]
    d_model = 8
    B, Tenc, L = 1, 3, 2
    model = TigerSeq2Seq(vocab_sizes=vocab_sizes, d_model=d_model, n_heads=2, n_layers_enc=1, n_layers_dec=1, dropout=0.0)

    level_ids = torch.zeros(B, Tenc, dtype=torch.long)
    token_ids = torch.zeros(B, Tenc, dtype=torch.long)
    attn_mask = torch.ones(B, Tenc, dtype=torch.bool)
    memory = model.encoder(level_ids, token_ids, attn_mask)
    mem_pad_mask = ~attn_mask

    beams = constrained_beam_search(
        model=model,
        memory=memory,
        mem_pad_mask=mem_pad_mask,
        L=L,
        bos_id=max(vocab_sizes),
        beam_size=2,
        per_step_candidates=2,
        device=memory.device,
        prefix_indexer=indexer,
    )
    assert isinstance(beams, list) and len(beams) <= 2
    for code, _ in beams:
        assert isinstance(code, tuple) and len(code) == L


def test_metrics():
    beams = [((1, 0), 0.3), ((0, 1), 0.2)]
    code2item = {(1, 0): 10, (0, 1): 11}
    ranked = ranks_to_items(beams, code2item, Kmax=5)
    assert ranked[:2] == [10, 11]
    assert recall_at_k(ranked, target_item=10, k=1) == 1.0
    assert recall_at_k(ranked, target_item=11, k=1) == 0.0
    assert recall_at_k(ranked, target_item=11, k=2) == 1.0
