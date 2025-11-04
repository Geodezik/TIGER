import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from loguru import logger
from generative_retrieval.model import TigerSeq2Seq
from generative_retrieval.utils import (ddp_init, get_rank, get_world_size, seed_all, load_codes_and_maps, parse_sequential_data, train_test_split,
                   TigerDataset, collate_batch, ce_multi_level, causal_mask, build_decoder_inputs_for_training, PrefixIndexer, eval_loss_and_recalls, save_pretrained)


def train_with_periodic_val(model, opt, train_dl, eval_dl, device, scaler, grad_clip, vocab_sizes, cfg, code2item, prefix_indexer):
    model.train()
    L = len(vocab_sizes)
    bos_id = max(vocab_sizes)
    running_loss = 0.0
    steps = 0
    rank = get_rank()

    for epoch in range(1, cfg["epochs"] + 1):
        if dist.is_initialized() and hasattr(train_dl.sampler, "set_epoch"):
            train_dl.sampler.set_epoch(epoch)

        for batch in train_dl:
            model.train()
            enc_level_ids = batch["level_ids"].to(device)
            enc_token_ids = batch["token_ids"].to(device)
            enc_attn_mask = batch["attn_mask"].to(device)
            target_code = batch["target_code"].to(device)
            B = target_code.size(0)

            dec_in = build_decoder_inputs_for_training(target_code, bos_id)
            dec_level = torch.arange(0, L, device=device, dtype=torch.long).unsqueeze(0).repeat(B, 1)
            tgt_mask = causal_mask(L, device)

            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda" and cfg["fp16"])):
                    logits_list = model(enc_level_ids, enc_token_ids, enc_attn_mask, dec_level, dec_in, tgt_mask)
                    loss = ce_multi_level(logits_list, target_code)
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                logits_list = model(enc_level_ids, enc_token_ids, enc_attn_mask, dec_level, dec_in, tgt_mask)
                loss = ce_multi_level(logits_list, target_code)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            running_loss += float(loss.detach().item())
            steps += 1

            if steps % cfg.get("log_every", 100) == 0 and rank == 0:
                logger.info(f"step {steps} | avg loss {running_loss / cfg['log_every']:.4f}")
                running_loss = 0.0

            if cfg.get("validate_every", 0) and (steps % cfg["validate_every"] == 0):
                if dist.is_initialized():
                    dist.barrier()
                val_loss, r10, r100, r1000 = eval_loss_and_recalls(
                    model,
                    eval_dl,
                    device,
                    code2item,
                    vocab_sizes=vocab_sizes,
                    max_batches=cfg.get("eval_max_batches", None),
                    prefix_indexer=prefix_indexer,
                    cfg=cfg,
                )
                if get_rank() == 0:
                    logger.info(f"val@step {steps} | loss {val_loss:.4f} | R@10 {r10:.4f} | R@100 {r100:.4f} | R@1000 {r1000:.4f}")
                if dist.is_initialized():
                    dist.barrier()

            save_every = int(cfg.get("save_every", 0) or 0)
            if save_every > 0 and (steps % save_every == 0) and get_rank() == 0:
                ckpt_dir = os.path.join(cfg["out_dir"], f"checkpoint-{steps}")
                save_pretrained(model, ckpt_dir, cfg)
                logger.info(f"Saved checkpoint to {ckpt_dir}")

    return steps


def run(cfg):
    os.makedirs(cfg["out_dir"], exist_ok=True)
    local_rank = ddp_init()
    device = torch.device(f"cuda:{local_rank}" if (local_rank >= 0 and torch.cuda.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_all(cfg["seed"] + (get_rank() if dist.is_initialized() else 0))

    codes, code2item, vocab_sizes = load_codes_and_maps(cfg["codes_path"], cfg["item_ids_path"])
    prefix_indexer = PrefixIndexer(codes)
    seqs = parse_sequential_data(cfg["sequential_data_path"])
    train_pairs, test_pairs = train_test_split(seqs, cfg["max_seq_items"])

    train_ds = TigerDataset(train_pairs, codes)
    eval_ds = TigerDataset(test_pairs, codes)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True, drop_last=True) if dist.is_initialized() else None
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_ds, shuffle=False, drop_last=True) if dist.is_initialized() else None

    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=(train_sampler is None), sampler=train_sampler,
                        collate_fn=lambda b: collate_batch(b, vocab_sizes, add_cls=True), num_workers=2, pin_memory=True)
    eval_dl = DataLoader(eval_ds, batch_size=cfg["batch_size"], shuffle=False, sampler=eval_sampler,
                        collate_fn=lambda b: collate_batch(b, vocab_sizes, add_cls=True), num_workers=2, pin_memory=True)

    steps_per_epoch = len(train_dl)
    if get_rank() == 0:
        approx_vals_per_epoch = (steps_per_epoch // cfg["validate_every"]) if cfg.get("validate_every", 0) else 0
        logger.info(f"world_size={get_world_size()} | per-rank batch={cfg['batch_size']} | "
        f"steps/epoch (per rank)={steps_per_epoch} | total steps={steps_per_epoch * cfg['epochs']} | "
        f"periodic validations/epoch≈{approx_vals_per_epoch}")

    model = TigerSeq2Seq(
        vocab_sizes=vocab_sizes,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers_enc=cfg["n_layers_enc"],
        n_layers_dec=cfg["n_layers_dec"],
        dropout=cfg["dropout"],
    ).to(device)

    if dist.is_initialized():
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None, output_device=local_rank if device.type == "cuda" else None)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler('cuda', enabled=cfg["fp16"] and device.type == "cuda")

    train_with_periodic_val(model, opt, train_dl, eval_dl, device, scaler, cfg["grad_clip"], vocab_sizes, cfg, code2item, prefix_indexer)

    if dist.is_initialized():
        dist.destroy_process_group()
