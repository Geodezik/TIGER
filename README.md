import argparse
import os
import gin
import torch
import wandb

from accelerate import Accelerator
from data.processed import ItemData
from data.processed import RecDataset
from data.processed import SeqData
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
from evaluate.metrics import TopKAccumulator
from modules.model import EncoderDecoderRetrievalModel
from modules.scheduler.inv_sqrt import InverseSquareRootScheduler
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import compute_debug_metrics
from modules.utils import parse_config
from huggingface_hub import login
from torch.optim import AdamW
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from tqdm import tqdm


@gin.configurable
def train(
    iterations=500000,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=0.01,
    dataset_folder="dataset/ml-1m",
    save_dir_root="out/",
    dataset=RecDataset.ML_1M,
    pretrained_rqvae_path=None,
    pretrained_decoder_path=None,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    partial_eval_every=1000,
    full_eval_every=10000,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_sim_vq=False,
    vae_n_cat_feats=18,
    vae_n_layers=3,
    decoder_embed_dim=64,
    dropout_p=0.1,
    attn_heads=8,
    attn_embed_dim=64,
    attn_layers=4,
    dataset_split="beauty",
    push_vae_to_hf=False,
    train_data_subsample=True,
    model_jagged_mode=True,
    vae_hf_model_name="edobotta/rqvae-amazon-beauty"
):  
    if dataset != RecDataset.AMAZON:
        raise Exception(f"Dataset currently not supported: {dataset}.")

    if wandb_logging:
        params = locals()

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )

    device = accelerator.device

    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        run = wandb.init(
            project="gen-retrieval-decoder-training",
            config=params
        )
    
    item_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        force_process=force_dataset_process,
        split=dataset_split
    )
    train_dataset = SeqData(
        root=dataset_folder, 
        dataset=dataset, 
        is_train=True, 
        subsample=train_data_subsample, 
        split=dataset_split
    )
    eval_dataset = SeqData(
        root=dataset_folder, 
        dataset=dataset, 
        is_train=False, 
        subsample=False, 
        split=dataset_split
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    train_dataloader = cycle(train_dataloader)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    
    train_dataloader, eval_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader
    )

    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_rqvae_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq
    )
    tokenizer = accelerator.prepare(tokenizer)
    tokenizer.precompute_corpus_ids(item_dataset)
    
    if push_vae_to_hf:
        login()
        tokenizer.rq_vae.push_to_hub(vae_hf_model_name)

    model = EncoderDecoderRetrievalModel(
        embedding_dim=decoder_embed_dim,
        attn_dim=attn_embed_dim,
        dropout=dropout_p,
        num_heads=attn_heads,
        n_layers=attn_layers,
        num_embeddings=vae_codebook_size,
        inference_verifier_fn=lambda x: tokenizer.exists_prefix(x),
        sem_id_dim=tokenizer.sem_ids_dim,
        max_pos=train_dataset.max_seq_len*tokenizer.sem_ids_dim,
        jagged_mode=model_jagged_mode
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    lr_scheduler = InverseSquareRootScheduler(
        optimizer=optimizer,
        warmup_steps=10000
    )
    
    start_iter = 0
    if pretrained_decoder_path is not None:
        checkpoint = torch.load(pretrained_decoder_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        start_iter = checkpoint["iter"] + 1

    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )

    metrics_accumulator = TopKAccumulator(ks=[1, 5, 10])
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}, Num Parameters: {num_params}")
    with tqdm(initial=start_iter, total=start_iter + iterations,
              disable=not accelerator.is_main_process) as pbar:
        for iter in range(iterations):
            model.train()
            total_loss = 0
            optimizer.zero_grad()
            for _ in range(gradient_accumulate_every):
                data = next_batch(train_dataloader, device)
                tokenized_data = tokenizer(data)

                with accelerator.autocast():
                    model_output = model(tokenized_data)
                    loss = model_output.loss / gradient_accumulate_every
                    total_loss += loss
                
                if wandb_logging and accelerator.is_main_process:
                    train_debug_metrics = compute_debug_metrics(tokenized_data, model_output)

                accelerator.backward(total_loss)
                assert model.sem_id_embedder.emb.weight.grad is not None

            pbar.set_description(f'loss: {total_loss.item():.4f}')

            accelerator.wait_for_everyone()

            optimizer.step()
            lr_scheduler.step()

            accelerator.wait_for_everyone()

            if (iter+1) % partial_eval_every == 0:
                model.eval()
                model.enable_generation = False
                for batch in eval_dataloader:
                    data = batch_to(batch, device)
                    tokenized_data = tokenizer(data)

                    with torch.no_grad():
                        model_output_eval = model(tokenized_data)

                    if wandb_logging and accelerator.is_main_process:
                        eval_debug_metrics = compute_debug_metrics(tokenized_data, model_output_eval, "eval")
                        eval_debug_metrics["eval_loss"] = model_output_eval.loss.detach().cpu().item()
                        wandb.log(eval_debug_metrics)

            if (iter+1) % full_eval_every == 0:
                model.eval()
                model.enable_generation = True
                with tqdm(eval_dataloader, desc=f'Eval {iter+1}', disable=not accelerator.is_main_process) as pbar_eval:
                    for batch in pbar_eval:
                        data = batch_to(batch, device)
                        tokenized_data = tokenizer(data)

                        generated = model.generate_next_sem_id(tokenized_data, top_k=True, temperature=1)
                        actual, top_k = tokenized_data.sem_ids_fut, generated.sem_ids

                        metrics_accumulator.accumulate(actual=actual, top_k=top_k)

                        if accelerator.is_main_process and wandb_logging:
                            wandb.log(eval_debug_metrics)
                
                eval_metrics = metrics_accumulator.reduce()
                
                print(eval_metrics)
                if accelerator.is_main_process and wandb_logging:
                    wandb.log(eval_metrics)
                
                metrics_accumulator.reset()

            if accelerator.is_main_process:
                if (iter+1) % save_model_every == 0 or iter+1 == iterations:
                    state = {
                        "iter": iter,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict()
                    }

                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)

                    torch.save(state, save_dir_root + f"checkpoint_{iter}.pt")
                
                if wandb_logging:
                    wandb.log({
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "total_loss": total_loss.cpu().item(),
                        **train_debug_metrics
                    })

            pbar.update(1)
    
    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()

import gin
import torch

from einops import rearrange
from enum import Enum
from data.schemas import TokenizedSeqBatch
from modules.embedding.id_embedder import SemIdEmbedder
from modules.embedding.id_embedder import UserIdEmbedder
from modules.normalize import RMSNorm
from modules.transformer.attention import AttentionInput
from modules.transformer.model import TransformerDecoder
from modules.transformer.model import TransformerEncoderDecoder
from modules.utils import eval_mode
from modules.utils import maybe_repeat_interleave
from modules.utils import reset_encoder_cache
from modules.utils import reset_kv_cache
from modules.utils import select_columns_per_row
from ops.triton.jagged import jagged_to_flattened_tensor
from ops.triton.jagged import padded_to_jagged_tensor
from typing import NamedTuple
from torch import nn
from torch import Tensor
from torch.nn import functional as F

# Needed to make torch.compile succeed
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')


class ModelOutput(NamedTuple):
    loss: Tensor
    logits: Tensor
    loss_d: Tensor


class GenerationOutput(NamedTuple):
    sem_ids: Tensor
    log_probas: Tensor


class EncoderDecoderRetrievalModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        attn_dim,
        dropout,
        num_heads,
        n_layers,
        num_embeddings,
        sem_id_dim,
        inference_verifier_fn,
        max_pos=2048,
        jagged_mode: bool = True,
    ) -> None:
        super().__init__()

        self.jagged_mode = jagged_mode
        self.num_embeddings = num_embeddings
        self.sem_id_dim = sem_id_dim
        self.attn_dim = attn_dim
        self.inference_verifier_fn = inference_verifier_fn
        self.enable_generation = False

        self.bos_emb = nn.Parameter(torch.rand(embedding_dim))
        self.norm = RMSNorm(embedding_dim)
        self.norm_cxt = RMSNorm(embedding_dim)
        self.do = nn.Dropout(p=0.5)

        self.sem_id_embedder = SemIdEmbedder(
            num_embeddings=num_embeddings,
            sem_ids_dim=sem_id_dim,
            embeddings_dim=embedding_dim
        )
        self.user_id_embedder = UserIdEmbedder(2000, embedding_dim)
        
        self.wpe = nn.Embedding(num_embeddings=max_pos, embedding_dim=embedding_dim)
        self.tte = nn.Embedding(num_embeddings=sem_id_dim, embedding_dim=embedding_dim)
        self.tte_fut = nn.Embedding(num_embeddings=sem_id_dim, embedding_dim=embedding_dim)

        self.transformer = TransformerEncoderDecoder(
            d_in=attn_dim,
            d_out=attn_dim,
            dropout=dropout,
            num_heads=num_heads,
            encoder_layers=n_layers // 2,
            decoder_layers=n_layers // 2
        ) if self.jagged_mode else nn.Transformer(
            d_model=attn_dim,
            nhead=num_heads,
            num_encoder_layers=n_layers // 2,
            num_decoder_layers=n_layers // 2,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True
        )

        self.in_proj = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.in_proj_context = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, num_embeddings, bias=False)
    
    def _predict(self, batch: TokenizedSeqBatch) -> AttentionInput:
        user_emb = self.user_id_embedder(batch.user_ids)
        sem_ids_emb = self.sem_id_embedder(batch)
        sem_ids_emb, sem_ids_emb_fut = sem_ids_emb.seq, sem_ids_emb.fut
        seq_lengths = batch.seq_mask.sum(axis=1)
        
        B, N, D = sem_ids_emb.shape

        pos_max = N // self.sem_id_dim
        # pos = torch.arange(pos_max, device=batch.sem_ids.device).repeat_interleave(self.sem_id_dim)
          
        pos = torch.arange(N, device=sem_ids_emb.device).unsqueeze(0)
        wpe = self.wpe(pos)

        input_embedding = torch.cat([user_emb, wpe + sem_ids_emb], axis=1)
        input_embedding_fut = self.bos_emb.repeat(B, 1, 1)
        if sem_ids_emb_fut is not None:
            tte_fut = self.tte(batch.token_type_ids_fut)
            input_embedding_fut = torch.cat([
                input_embedding_fut, 
                sem_ids_emb_fut + tte_fut
                ], axis=1
            )

        if self.jagged_mode:
            input_embedding = padded_to_jagged_tensor(input_embedding, lengths=seq_lengths+1, max_len=input_embedding.shape[1])

            seq_lengths_fut = torch.tensor(input_embedding_fut.shape[1], device=input_embedding_fut.device, dtype=torch.int64).repeat(B)
            input_embedding_fut = padded_to_jagged_tensor(input_embedding_fut, lengths=seq_lengths_fut, max_len=input_embedding_fut.shape[1])
        else:
            mem_mask = torch.cat([
                torch.ones(B, 1, dtype=torch.bool, device=batch.seq_mask.device),
                batch.seq_mask
            ], axis=1)
            f_mask = torch.zeros_like(mem_mask, dtype=torch.float32)
            f_mask[~mem_mask] = float("-inf")
        
        transformer_context = self.in_proj_context(self.do(self.norm(input_embedding)))
        transformer_input = self.in_proj(self.do(self.norm_cxt(input_embedding_fut)))
        
        if self.jagged_mode:
            transformer_output = self.transformer(x=transformer_input, context=transformer_context, padding_mask=batch.seq_mask, jagged=self.jagged_mode)
        else:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(transformer_input.shape[1])
            transformer_output = self.transformer(src=transformer_context, tgt=transformer_input, tgt_is_causal=True, tgt_mask=causal_mask, src_key_padding_mask=f_mask, memory_key_padding_mask=f_mask)

        return transformer_output

    @eval_mode
    @reset_encoder_cache
    @torch.no_grad
    def generate_next_sem_id(
        self,
        batch: TokenizedSeqBatch,
        temperature: int = 1,
        top_k: bool = True
    ) -> GenerationOutput:
        
        assert self.enable_generation, "Model generation is not enabled"

        B, N = batch.sem_ids.shape
        generated, log_probas = None, 0
        k = 64 if top_k else 1
        n_top_k_candidates = 256 if top_k else 1

        input_batch = TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=batch.sem_ids,
            sem_ids_fut=None,
            seq_mask=batch.seq_mask,
            token_type_ids=batch.token_type_ids,
            token_type_ids_fut=None
        )

        for i in range(self.sem_id_dim):
            logits = self.forward(input_batch).logits
            probas_batched = F.softmax(logits / temperature, dim=-1)
            samples_batched = torch.multinomial(probas_batched, num_samples=n_top_k_candidates)

            if generated is None:
                is_valid_prefix = self.inference_verifier_fn(samples_batched.unsqueeze(-1))
            else:
                prefix = torch.cat([generated.flatten(0,1).unsqueeze(1).repeat_interleave(n_top_k_candidates, axis=1), samples_batched.unsqueeze(-1)], axis=-1)
                is_valid_prefix = self.inference_verifier_fn(prefix).reshape(B, -1)
            
            sampled_log_probas = torch.log(torch.gather(probas_batched, 1, samples_batched)).reshape(B, -1)
            samples = samples_batched.reshape(B, -1)

            # Get top-K:
            sorted_log_probas, sorted_indices = (
                -10000*(~is_valid_prefix) +
                sampled_log_probas +
                maybe_repeat_interleave(log_probas, n_top_k_candidates, dim=1)
            ).sort(-1, descending=True)

            top_k_log_probas, top_k_indices = sorted_log_probas[:, :k], sorted_indices[:, :k]
            top_k_samples = torch.gather(samples, 1, top_k_indices)
            
            if generated is not None:
                parent_id = torch.gather(generated, 1, (top_k_indices // n_top_k_candidates).unsqueeze(2).expand(-1,-1,i))
                top_k_samples = torch.cat([parent_id, top_k_samples.unsqueeze(-1)], axis=-1)

                next_sem_ids = top_k_samples.flatten(end_dim=1)

                input_batch = TokenizedSeqBatch(
                    user_ids=input_batch.user_ids,
                    sem_ids=input_batch.sem_ids,
                    sem_ids_fut=next_sem_ids,
                    token_type_ids_fut=torch.arange(next_sem_ids.shape[1], device=next_sem_ids.device).repeat(next_sem_ids.shape[0], 1),
                    seq_mask=input_batch.seq_mask,
                    token_type_ids=input_batch.token_type_ids
                )

                generated = torch.clone(top_k_samples.detach())
                log_probas = torch.clone(top_k_log_probas.detach())
            else:
                next_sem_ids = top_k_samples.reshape(-1, 1)
                # Explode encoder cache on dim 0 to match input size B*k
                # TODO: Figure out how to avoid jagged - padded conversions 
                # (E.g. Implement repeat_interleave jagged kernel)
                if self.jagged_mode:
                    cache = torch.zeros(input_batch.sem_ids.shape[0], input_batch.sem_ids.shape[1]+1, self.attn_dim, device=input_batch.sem_ids.device)
                    cache_mask = torch.cat([torch.ones(input_batch.sem_ids.shape[0], 1, dtype=bool, device=input_batch.seq_mask.device), input_batch.seq_mask], axis=1)
                    cache[cache_mask] = self.transformer.cached_enc_output.values()
                    lengths = self.transformer.cached_enc_output.offsets().diff().repeat_interleave(k)
                    cache = cache.repeat_interleave(k, dim=0)
                    self.transformer.cached_enc_output = padded_to_jagged_tensor(cache, lengths, max_len=cache.shape[1])

                input_batch = TokenizedSeqBatch(
                    user_ids=input_batch.user_ids.repeat_interleave(k, dim=0),
                    sem_ids=input_batch.sem_ids.repeat_interleave(k, dim=0),
                    sem_ids_fut=next_sem_ids,
                    token_type_ids_fut=torch.zeros_like(next_sem_ids),
                    seq_mask=input_batch.seq_mask.repeat_interleave(k, dim=0),
                    token_type_ids=input_batch.token_type_ids.repeat_interleave(k, dim=0)
                )

                generated = top_k_samples.unsqueeze(-1)
                log_probas = torch.clone(top_k_log_probas.detach())
        
        return GenerationOutput(
            sem_ids=generated.squeeze(),
            log_probas=log_probas.squeeze()
        )
            
    @torch.compile
    def forward(self, batch: TokenizedSeqBatch) -> ModelOutput:
        seq_mask = batch.seq_mask
        B, N = seq_mask.shape

        trnsf_out = self._predict(batch)
        
        if self.training or not self.enable_generation:
            predict_out = self.out_proj(trnsf_out)
            if self.jagged_mode:
                # This works because batch.sem_ids_fut is fixed length, no padding.
                logits = rearrange(jagged_to_flattened_tensor(predict_out), "(b n) d -> b n d", b=B)[:,:-1,:].flatten(end_dim=1)
                target = batch.sem_ids_fut.flatten(end_dim=1)
                unred_loss = rearrange(F.cross_entropy(logits, target, reduction="none", ignore_index=-1), "(b n) -> b n", b=B)
                loss = unred_loss.sum(axis=1).mean()
            else:
                logits = predict_out
                out = logits[:, :-1, :].flatten(end_dim=1)
                target = batch.sem_ids_fut.flatten(end_dim=1)
                unred_loss = rearrange(F.cross_entropy(out, target, reduction="none", ignore_index=-1), "(b n) -> b n", b=B)
                loss = unred_loss.sum(axis=1).mean()
            if not self.training and self.jagged_mode:
                self.transformer.cached_enc_output = None
            loss_d = unred_loss.mean(axis=0)
        elif self.jagged_mode:
            trnsf_out = trnsf_out.contiguous()
            trnsf_out_flattened = rearrange(jagged_to_flattened_tensor(trnsf_out), "(b n) d -> b n d", b=B)[:,-1,:]
            logits = self.out_proj(trnsf_out_flattened)
            loss = None
            loss_d = None
        else:
            trnsf_out_flattened = trnsf_out[:,-1,:]
            logits = self.out_proj(trnsf_out_flattened)
            loss = None
            loss_d = None

        return ModelOutput(loss=loss, logits=logits, loss_d=loss_d)

import torch
import torch.nn.functional as F

from ops.triton.jagged import jagged_to_flattened_tensor
from ops.triton.jagged import padded_to_jagged_tensor
from torch import nn
from torch import Tensor
from torch.nested import Tensor as NestedTensor
from typing import Optional
from typing import Union

torch.backends.cuda.enable_flash_sdp(True)

AttentionInput = Union[Tensor, NestedTensor]


class KVCache(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert len(dim) == 3, "Cache only supports 3d tensors"
        self.register_buffer("k_cache", torch.zeros(*dim, requires_grad=False))
        self.register_buffer("v_cache", torch.zeros(*dim, requires_grad=False))
        self.dim = dim
        
        self._reset_limits()
        self.is_empty = True
    
    def _reset_limits(self):
        self.cache_limits = [0 for _ in self.dim]
        self.next_seq_pos = None
    
    def reset(self):
        self.k_cache.fill_(0)
        self.v_cache.fill_(0)
        
        self._reset_limits()
        self.is_empty = True
    
    @property
    def device(self):
        return self.k_cache.device
    
    @property
    def keys(self):
        B, N, D = self.cache_limits
        return self.k_cache[:B, :N, :D]
    
    @property
    def values(self):
        B, N, D = self.cache_limits
        return self.v_cache[:B, :N, :D]
    
    @property
    def seq_lengths(self):
        if self.is_empty:
            return 0
        return self.next_seq_pos
    
    @torch.no_grad
    def store(self, keys: Tensor, values: Tensor, mask: Tensor) -> None:
        B, N = mask.shape
        self.k_cache[:B, :N, :][mask] = keys.detach()[:, :]
        self.v_cache[:B, :N, :][mask] = values.detach()[:, :]

        self.cache_limits = [B, N, self.dim[-1]]
        self.next_seq_pos = mask.sum(axis=1).unsqueeze(-1)
        self.is_empty = False
    
    @torch.no_grad
    def append_column(self, keys: Tensor, values: Tensor) -> None:
        B, N, D = self.cache_limits

        row_idx = torch.arange(B, device=self.k_cache.device)
        self.k_cache[:B, :][row_idx, self.next_seq_pos] = keys.detach()[:, :]
        self.v_cache[:B, :][row_idx, self.next_seq_pos] = values.detach()[:, :]

        max_pos_appended = self.next_seq_pos.max()
        if max_pos_appended >= N:
            self.cache_limits[1] = max_pos_appended + 1
        self.next_seq_pos += 1
    
    @torch.no_grad
    @torch.compiler.disable
    def as_jagged(self):
        keys_jagged = padded_to_jagged_tensor(self.keys, lengths=self.seq_lengths.squeeze(), max_len=self.keys.shape[1])
        values_jagged = padded_to_jagged_tensor(self.values, lengths=self.seq_lengths.squeeze(), max_len=self.values.shape[1])
        return keys_jagged, values_jagged

    @torch.no_grad
    def apply(self, fn) -> None:
        B, N, D = self.cache_limits
        k_transformed, v_transformed = fn(self.k_cache[:B, :N, :D]), fn(self.v_cache[:B, :N, :D])
        next_seq_pos_transformed = fn(self.next_seq_pos)
        B, N, D = k_transformed.shape

        self.reset()
        self.k_cache[:B, :N, :D] = k_transformed
        self.v_cache[:B, :N, :D] = v_transformed
        self.next_seq_pos = next_seq_pos_transformed
        self.cache_limits = [B, N, D]
        self.is_empty = False


class Attend(nn.Module):
    def __init__(self, d_out, num_heads, head_dim, dropout):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_out = d_out
        self.dropout = dropout
    
    def jagged_forward(self, qu: NestedTensor, ke: NestedTensor, va: NestedTensor, is_causal: bool) -> NestedTensor:
        queries = qu.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        keys = ke.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        values = va.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)

        dropout_p = 0. if not self.training else 0.0

        context_vec = F.scaled_dot_product_attention(
            queries, keys, values, dropout_p=dropout_p, is_causal=is_causal)
        
        context_vec = context_vec.transpose(1, 2).flatten(-2)
        return context_vec

    def forward(self, qkv: Tensor, is_causal: bool = False) -> Tensor:
        batch_size, num_tokens, embed_dim = qkv.shape
        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout

        context_vec = F.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=is_causal)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)
        return context_vec


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        num_heads,
        cross_attn=False,
        dropout=0.0,
        qkv_bias=False,
        enable_kv_cache=False
    ) -> None:
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"
        assert not enable_kv_cache, "KV Cache currently not supported"

        self.cross_attn = cross_attn
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out
        self.enable_kv_cache = enable_kv_cache

        if self.cross_attn:
            self.q = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.kv = nn.Linear(d_in, 2 * d_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
    
        self.proj = nn.Linear(d_out, d_out, bias=False)

        self.attend = Attend(self.d_out, self.num_heads, self.head_dim, dropout=False)

        self._kv_cache = KVCache((2560, 80, 384)) if enable_kv_cache else None # (640, 800, 64) TODO: Revisit KV Cache
    
    @property
    def kv_cache(self) -> KVCache:
        return self._kv_cache

    def forward(
        self,
        x: AttentionInput,
        x_kv: Optional[AttentionInput] = None,
        padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = True,
        jagged: bool = False,
        use_cache: bool = False,
    ) -> AttentionInput:
        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        assert not self.cross_attn or x_kv is not None, "Found null x_kv in cross attn. layer"
        
        if self.cross_attn:
            queries = self.q(x)
            keys, values = self.kv(x_kv).chunk(2, dim=-1)
        else:
            queries, keys, values = self.qkv(x).chunk(3, dim=-1)
        
        if not self.training and use_cache and self.enable_kv_cache and self.kv_cache.is_empty:
            assert padding_mask is not None
            B, N = padding_mask.shape
            
            self.kv_cache.store(
                keys=jagged_to_flattened_tensor(keys), 
                values=jagged_to_flattened_tensor(values), 
                mask=padding_mask
            )
            context_vec = self.attend.jagged_forward(queries, keys, values, is_causal=is_causal)

        elif not self.training and use_cache and self.enable_kv_cache and not self.kv_cache.is_empty:
            assert padding_mask is not None
            B, N = padding_mask.shape

            keys, values = jagged_to_flattened_tensor(keys), jagged_to_flattened_tensor(values)
            
            self.kv_cache.append_column(keys=keys, values=values)
            keys, values = self.kv_cache.as_jagged()
            
            context_vec = self.attend.jagged_forward(queries, keys, values, is_causal=False)
        
        elif jagged:
            context_vec = self.attend.jagged_forward(queries, keys, values, is_causal=is_causal)

        if not jagged:
            raise Exception("Unjagged attention currently not supported.")
            # context_vec = self.attend(qkv, is_causal=is_causal)
    
        context_vec = self.proj(context_vec)
        return context_vec

from modules.encoder import MLP
from modules.normalize import RMSNorm
from modules.transformer.attention import AttentionInput
from modules.transformer.attention import MultiHeadAttention
from typing import List
from typing import Optional
from torch import nn
from torch import Tensor


class KVCacheOpsMixin:
    def reset_kv_cache(self) -> None:
        for layer in self.layers:
            layer.reset_kv_cache()
    
    def apply_to_kv_cache(self, fn) -> None:
        for layer in self.layers:
            layer.apply_to_kv_cache(fn)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool,
        mlp_hidden_dims: List[int] = [1024],
        do_cross_attn: bool = False,
        enable_kv_cache: bool = True
    ) -> None:
        super().__init__()
        
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.do_cross_attn = do_cross_attn
        self.enable_kv_cache = enable_kv_cache

        self.attention = MultiHeadAttention(
            d_in=d_in, d_out=d_out, num_heads=num_heads, cross_attn=False, dropout=dropout, qkv_bias=qkv_bias, enable_kv_cache=enable_kv_cache
        )

        self.ff = nn.Sequential(
            RMSNorm(d_out),
            MLP(
                input_dim=d_out,
                hidden_dims=mlp_hidden_dims,
                out_dim=d_out,
                dropout=dropout,
                normalize=False
            ),
            nn.Dropout(dropout)
        )

        self.attn_norm = RMSNorm(d_out)
        self.ffn_norm = RMSNorm(d_out)
        self.do = nn.Dropout(dropout)

        if self.do_cross_attn:
            self.cross_attention = MultiHeadAttention(
                d_in=d_out, d_out=d_out, num_heads=num_heads, cross_attn=True, dropout=dropout, qkv_bias=qkv_bias
            )
            self.cross_attn_norm = RMSNorm(d_out)
    
    def forward(
        self,
        x: AttentionInput,
        x_kv: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = True,
        jagged: Optional[bool] = False
    ) -> AttentionInput:
        attn_out = x + self.attention(self.attn_norm(x), padding_mask=padding_mask, is_causal=is_causal, jagged=jagged, use_cache=not self.training and self.enable_kv_cache)
        if self.do_cross_attn:
            attn_out = attn_out + self.cross_attention(
                x=self.cross_attn_norm(attn_out), x_kv=x_kv, padding_mask=padding_mask, is_causal=False, jagged=jagged, use_cache=not self.training and self.enable_kv_cache
            )
        proj_out = attn_out + self.ff(attn_out)
        return proj_out
    
    def reset_kv_cache(self):
        self.attention.kv_cache.reset()
        if self.do_cross_attn:
            self.cross_attention.kv_cache.reset()

    def apply_to_kv_cache(self, fn):
        self.attention.kv_cache.apply(fn)
        if self.do_cross_attn:
            self.cross_attention.kv_cache.apply(fn)


class TransformerDecoder(nn.Module, KVCacheOpsMixin):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        num_heads: int,
        n_layers: int,
        do_cross_attn: bool = False,
        enable_kv_cache: bool = True
    ) -> None:
        super().__init__()

        self.do_cross_attn = do_cross_attn

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_in=d_in,
                d_out=d_out,
                dropout=dropout,
                num_heads=num_heads,
                qkv_bias=False,
                do_cross_attn=self.do_cross_attn,
                enable_kv_cache=enable_kv_cache
            ) for _ in range(n_layers)
        ])

    def forward(
        self,
        x: AttentionInput,
        padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = True,
        context: Optional[Tensor] = None,
        jagged: Optional[bool] = None
    ) -> AttentionInput:
        for layer in self.layers:
            x = layer(x=x, x_kv=context, padding_mask=padding_mask, is_causal=is_causal, jagged=jagged)
        return x
    
    @property
    def seq_lengths(self) -> Tensor:
        return self.layers[0].attention.kv_cache.seq_lengths


class TransformerEncoderDecoder(nn.Module, KVCacheOpsMixin):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        num_heads: int,
        encoder_layers: int,
        decoder_layers: int,
    ) -> None:
        super().__init__()

        self.encoder = TransformerDecoder(
            d_in=d_in,
            d_out=d_out,
            dropout=dropout,
            num_heads=num_heads,
            n_layers=encoder_layers,
            do_cross_attn=False,
            enable_kv_cache=False
        )

        self.decoder = TransformerDecoder(
            d_in=d_in,
            d_out=d_out,
            dropout=dropout,
            num_heads=num_heads,
            n_layers=decoder_layers,
            do_cross_attn=True,
            enable_kv_cache=False
        )

        self.layers = [self.encoder, self.decoder]
        self.cached_enc_output = None
    
    def forward(
        self,
        x: AttentionInput,
        padding_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        jagged: Optional[bool] = None
    ) -> AttentionInput:
        if self.cached_enc_output is None:
            context = self.encoder(context, padding_mask=padding_mask, is_causal=False, context=None, jagged=jagged)
            if not self.training:
                self.cached_enc_output = context
        else:
            context = self.cached_enc_output
        out = self.decoder(x, padding_mask=None, is_causal=True, context=context, jagged=jagged)
        return out
