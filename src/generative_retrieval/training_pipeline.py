import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

import mlflow
import mlflow.pytorch

from loguru import logger

from generative_retrieval.model import TigerSeq2Seq
from generative_retrieval.utils import sha256_file, flatten_dict
from generative_retrieval.utils import (
    ddp_init, get_rank, get_world_size, seed_all,
    load_codes_and_maps, parse_sequential_data, train_test_split,
    TigerDataset, collate_batch, ce_multi_level, causal_mask,
    build_decoder_inputs_for_training, PrefixIndexer, eval_loss_and_recalls,
    save_pretrained
)


def _safe_mlflow_params(cfg: dict) -> dict:
    flat = flatten_dict(cfg)
    safe = {}
    for k, v in flat.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            safe[k] = v
        else:
            safe[k] = str(v)
    return safe


def train_with_periodic_val(
    model, opt, train_dl, eval_dl, device, scaler, grad_clip,
    vocab_sizes, cfg, code2item, prefix_indexer,
    mlflow_enabled: bool = False
):
    model.train()
    L = len(vocab_sizes)
    bos_id = max(vocab_sizes)
    running_loss = 0.0
    steps = 0
    rank = get_rank()

    log_every = int(cfg.get("log_every", 100) or 0)
    validate_every = int(cfg.get("validate_every", 0) or 0)
    save_every = int(cfg.get("save_every", 0) or 0)

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
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and cfg.get("fp16", False))):
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

            if log_every > 0 and (steps % log_every == 0) and rank == 0:
                avg = running_loss / log_every
                logger.info(f"step {steps} | avg loss {avg:.4f}")
                if mlflow_enabled:
                    mlflow.log_metric("train_loss", avg, step=steps)
                running_loss = 0.0

            if validate_every > 0 and (steps % validate_every == 0):
                if dist.is_initialized():
                    dist.barrier()

                val_loss, r10, r100, r1000 = eval_loss_and_recalls(
                    model=model,
                    dl=eval_dl,
                    device=device,
                    code2item=code2item,
                    vocab_sizes=vocab_sizes,
                    max_batches=cfg.get("eval_max_batches", None),
                    prefix_indexer=prefix_indexer,
                    cfg=cfg,
                )

                if rank == 0:
                    logger.info(
                        f"val@step {steps} | loss {val_loss:.4f} | "
                        f"R@10 {r10:.4f} | R@100 {r100:.4f} | R@1000 {r1000:.4f}"
                    )
                    if mlflow_enabled:
                        mlflow.log_metric("val_loss", val_loss, step=steps)
                        mlflow.log_metric("R_10", r10, step=steps)
                        mlflow.log_metric("R_100", r100, step=steps)
                        mlflow.log_metric("R_1000", r1000, step=steps)

                if dist.is_initialized():
                    dist.barrier()

            if save_every > 0 and (steps % save_every == 0) and rank == 0:
                ckpt_dir = os.path.join(cfg["out_dir"], f"checkpoint-{steps}")
                save_pretrained(model, ckpt_dir, cfg)
                logger.info(f"Saved checkpoint to {ckpt_dir}")

    return steps


def run(cfg):
    os.makedirs(cfg["out_dir"], exist_ok=True)

    local_rank = ddp_init()
    device = torch.device(
        f"cuda:{local_rank}"
        if (local_rank >= 0 and torch.cuda.is_available())
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    rank = get_rank()
    seed_all(cfg["seed"] + (rank if dist.is_initialized() else 0))

    codes, code2item, vocab_sizes = load_codes_and_maps(cfg["codes_path"], cfg["item_ids_path"])
    prefix_indexer = PrefixIndexer(codes)
    seqs = parse_sequential_data(cfg["sequential_data_path"])
    train_pairs, test_pairs = train_test_split(seqs, cfg["max_seq_items"])

    train_ds = TigerDataset(train_pairs, codes)
    eval_ds = TigerDataset(test_pairs, codes)

    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True, drop_last=True)
        if dist.is_initialized() else None
    )
    eval_sampler = (
        torch.utils.data.distributed.DistributedSampler(eval_ds, shuffle=False, drop_last=True)
        if dist.is_initialized() else None
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=lambda b: collate_batch(b, vocab_sizes, add_cls=True),
        num_workers=2,
        pin_memory=True,
    )
    eval_dl = DataLoader(
        eval_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        sampler=eval_sampler,
        collate_fn=lambda b: collate_batch(b, vocab_sizes, add_cls=True),
        num_workers=2,
        pin_memory=True,
    )

    steps_per_epoch = len(train_dl)
    if rank == 0:
        approx_vals_per_epoch = (steps_per_epoch // cfg["validate_every"]) if cfg.get("validate_every", 0) else 0
        logger.info(
            f"world_size={get_world_size()} | per-rank batch={cfg['batch_size']} | "
            f"steps/epoch (per rank)={steps_per_epoch} | total steps={steps_per_epoch * cfg['epochs']} | "
            f"periodic validations/epoch≈{approx_vals_per_epoch}"
        )

    model = TigerSeq2Seq(
        vocab_sizes=vocab_sizes,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers_enc=cfg["n_layers_enc"],
        n_layers_dec=cfg["n_layers_dec"],
        dropout=cfg["dropout"],
    ).to(device)

    if dist.is_initialized():
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None
        )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.get("fp16", False) and device.type == "cuda")

    mlflow_enabled = bool(cfg.get("mlflow_enabled", True)) and (rank == 0)
    if mlflow_enabled:
        mlflow.set_experiment(cfg.get("mlflow_experiment", "tiger"))
        mlflow.pytorch.autolog(log_models=False)

        with mlflow.start_run(run_name=cfg.get("mlflow_run_name", None)):
            mlflow.log_params(_safe_mlflow_params(cfg))
            mlflow.set_tag("dvc_lock_hash", sha256_file("dvc.lock"))
            mlflow.set_tag("dvc_yaml_hash", sha256_file("dvc.yaml"))
            mlflow.set_tag("data_beauty_dvc_hash", sha256_file("data/beauty.dvc"))

            try:
                import subprocess
                git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
                mlflow.set_tag("git_commit", git_sha)
            except Exception:
                pass

            train_with_periodic_val(
                model=model,
                opt=opt,
                train_dl=train_dl,
                eval_dl=eval_dl,
                device=device,
                scaler=scaler,
                grad_clip=cfg["grad_clip"],
                vocab_sizes=vocab_sizes,
                cfg=cfg,
                code2item=code2item,
                prefix_indexer=prefix_indexer,
                mlflow_enabled=True,
            )

            final_dir = os.path.join(cfg["out_dir"], "final")
            save_pretrained(model, final_dir, cfg)
            logger.info(f"Saved final model to {final_dir}")

            mlflow.log_artifacts(final_dir, artifact_path="model_final")

            for p in ["dvc.yaml", "dvc.lock"]:
                if os.path.exists(p):
                    mlflow.log_artifact(p, artifact_path="dvc")

            for p in ["configs/embedder.yaml", "configs/quantizer.yaml", "configs/recommender.yaml"]:
                if os.path.exists(p):
                    mlflow.log_artifact(p, artifact_path="configs")

            metrics_path = "outputs/metrics/beauty_metrics.json"
            if os.path.exists(metrics_path):
                mlflow.log_artifact(metrics_path, artifact_path="metrics")

    else:
        # non-rank0 processes (or mlflow disabled): just train
        train_with_periodic_val(
            model=model,
            opt=opt,
            train_dl=train_dl,
            eval_dl=eval_dl,
            device=device,
            scaler=scaler,
            grad_clip=cfg["grad_clip"],
            vocab_sizes=vocab_sizes,
            cfg=cfg,
            code2item=code2item,
            prefix_indexer=prefix_indexer,
            mlflow_enabled=False,
        )

        if rank == 0:
            final_dir = os.path.join(cfg["out_dir"], "final")
            save_pretrained(model, final_dir, cfg)
            logger.info(f"Saved final model to {final_dir}")

    if dist.is_initialized():
        dist.destroy_process_group()
