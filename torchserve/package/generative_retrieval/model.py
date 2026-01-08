import math
import torch
import torch.nn as nn

from typing import List

import warnings
warnings.filterwarnings("ignore", message=".nested tensors is in prototype stage.")


class MultiLevelEmbedding(nn.Module):
    '''
    Embeddings for different levels of semantic IDs
    In encoder these embeddings are used to build an interleaved SID sequence
    In decoder these embedding are used for sequential generation conditioning
    '''

    def __init__(self, vocab_sizes: List[int], d_model: int, dropout: float, add_pad_cls: bool):
        super().__init__()
        self.L = len(vocab_sizes)
        extra = 2 if add_pad_cls else 0
        self.emb_levels = nn.ModuleList([nn.Embedding(v + extra, d_model) for v in vocab_sizes])
        self.level_embed = nn.Embedding(self.L, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, level_ids: torch.Tensor, token_ids: torch.Tensor):
        B, T = level_ids.size()
        D = self.emb_levels[0].embedding_dim
        out = torch.zeros(B, T, D, device=level_ids.device)

        flat_lvl = level_ids.reshape(-1)
        flat_tok = token_ids.reshape(-1)
        out2 = out.view(-1, D)

        for l in range(self.L):
            mask = (flat_lvl == l)
            if mask.any():
                tok_l = flat_tok[mask]
                emb_tok = self.emb_levels[l](tok_l)
                emb_tok = emb_tok + self.level_embed.weight[l]
                out2[mask] = emb_tok

        return self.drop(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        T = x.size(1)
        pos_emb = self.pe[:T, :].unsqueeze(0)
        return self.drop(x + pos_emb)


class Encoder(nn.Module):
    '''
    Simple transformer-based encoder aggregating sequences of semantic IDs
    '''

    def __init__(self, vocab_sizes: List[int], d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.embed = MultiLevelEmbedding(vocab_sizes, d_model, dropout, add_pad_cls=True)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, dropout=dropout)
        try:
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)
        except TypeError:
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, level_ids: torch.Tensor, token_ids: torch.Tensor, attn_mask: torch.Tensor):
        x = self.embed(level_ids, token_ids)
        x = self.pos(x)
        src_key_padding_mask = ~attn_mask
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return h


class DecoderEmbedding(nn.Module):
    def __init__(self, vocab_sizes: List[int], d_model: int, dropout: float):
        super().__init__()
        self.L = len(vocab_sizes)
        self.max_v = max(vocab_sizes)
        self.emb = nn.Embedding(self.max_v + 1, d_model)
        self.level_embed = nn.Embedding(self.L, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, level_ids: torch.Tensor, token_ids: torch.Tensor):
        tok = self.emb(token_ids)
        lvl = self.level_embed(level_ids)
        return self.drop(tok + lvl)


class Decoder(nn.Module):
    '''
    SID sequences generator with conditioning
    '''

    def __init__(self, vocab_sizes: List[int], d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.L = len(vocab_sizes)
        self.embed = DecoderEmbedding(vocab_sizes, d_model, dropout)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, dropout=dropout)
        try:
            self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers, enable_nested_tensor=False)
        except TypeError:
            self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.heads = nn.ModuleList([nn.Linear(d_model, v) for v in vocab_sizes])

    def forward(self, dec_level_ids: torch.Tensor, dec_token_ids: torch.Tensor, memory: torch.Tensor, memory_key_padding_mask: torch.Tensor, tgt_mask: torch.Tensor):
        y = self.embed(dec_level_ids, dec_token_ids)
        y = self.pos(y)
        h = self.decoder(y, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
        T = dec_level_ids.size(1)
        logits_list = []
        for l in range(T):
            h_l = h[:, l, :]
            logits_list.append(self.heads[l](h_l))
        return logits_list


class TigerSeq2Seq(nn.Module):
    '''
    Implementation of https://arxiv.org/pdf/2305.05065
    with constrained beamsearch
    '''

    def __init__(self, vocab_sizes: List[int], d_model: int, n_heads: int, n_layers_enc: int, n_layers_dec: int, dropout: float):
        super().__init__()
        self.L = len(vocab_sizes)
        self.encoder = Encoder(vocab_sizes, d_model, n_heads, n_layers_enc, dropout)
        self.decoder = Decoder(vocab_sizes, d_model, n_heads, n_layers_dec, dropout)
        self.vocab_sizes = vocab_sizes

    def forward(self, enc_level_ids, enc_token_ids, enc_attn_mask, dec_level_ids, dec_token_ids, tgt_mask):
        memory = self.encoder(enc_level_ids, enc_token_ids, enc_attn_mask)
        mem_pad_mask = ~enc_attn_mask
        logits_list = self.decoder(dec_level_ids, dec_token_ids, memory, mem_pad_mask, tgt_mask)
        return logits_list
