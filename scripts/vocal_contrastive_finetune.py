#!/usr/bin/env python3
# Copyright (c) Max Planck Institute of Animal Behavior
#
# Standalone vocal-region contrastive finetune for data2vec_multi checkpoints.
#
# Training recipe (self-supervised, not classification):
#   - Student: pretrained data2vec_multi, updated by AdamW
#   - Teacher: copy of same checkpoint, frozen (used only for anchor loss)
#   - Data: Fairseq manifest (wav paths) + CSV vocal intervals
#   - Forward: extract_features -> average top-K layer embeddings (B, T, D)
#   - Loss: triplet margin on vocal vs non-vocal/cross-span frames + teacher anchor
#   - Output: Fairseq .pt checkpoint compatible with animal2vec_inference.py

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("vocal_contrastive_finetune")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import nn  # noqa: F401 — register data2vec_multi
from fairseq import checkpoint_utils
from nn.vocal_contrastive import (
    LabelConfig,
    VocalContrastiveDataset,
    average_layer_embeddings,
    compute_contrastive_loss,
    make_collate_fn,
)


def parse_args():
    p = argparse.ArgumentParser(description="Vocal contrastive finetune (triplet + teacher anchor)")
    p.add_argument("--pretrain-ckpt", required=True, help="Pretrained data2vec_multi .pt checkpoint")
    p.add_argument("--manifest-dir", required=True, help="Directory containing train_0.tsv etc.")
    p.add_argument("--train-subset", default="train_0", help="Manifest basename without .tsv")
    p.add_argument("--label-csv-dir", default=None, help="Directory with per-recording CSV labels")
    p.add_argument("--label-csv-format", default="auto")
    p.add_argument("--label-index-rate-hz", default="44100")
    p.add_argument("--exclude-classes", default="empty,unknown,silence,unk")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-updates", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-updates", type=int, default=500)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--anchor-weight", type=float, default=0.1)
    p.add_argument("--average-top-k-layers", type=int, default=12)
    p.add_argument("--save-interval-updates", type=int, default=1000)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--no-normalize", action="store_false", dest="normalize")
    p.add_argument("--freeze-feature-extractor", action="store_true")
    return p.parse_args()


def get_lr(step: int, warmup: int, max_updates: int, peak_lr: float) -> float:
    """Linear warmup, then cosine decay to zero."""
    if step < warmup:
        return peak_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(max_updates - warmup, 1)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def load_models(pretrain_ckpt: str, device: str):
    """Load student (trainable) and teacher (frozen) from the same pretrain ckpt."""
    # Fairseq models hold lazy task refs; deepcopy triggers RecursionError.
    student_models, _cfg = checkpoint_utils.load_model_ensemble([pretrain_ckpt])
    teacher_models, _ = checkpoint_utils.load_model_ensemble([pretrain_ckpt])
    student = student_models[0].to(device)
    student.train()
    teacher = teacher_models[0].to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return student, teacher


def conv_layers_from_cfg(model) -> list:
    layers = getattr(model.cfg, "conv_feature_layers", None)
    if layers is None and hasattr(model, "modality_encoders"):
        enc = model.modality_encoders.get("AUDIO")
        if enc is not None:
            layers = getattr(enc.modality_cfg, "conv_feature_layers", None)
    if isinstance(layers, str):
        return eval(layers)
    return layers


def sample_rate_from_cfg(model) -> int:
    sr = getattr(model.cfg, "sample_rate", None)
    if sr is None:
        task = getattr(model.cfg, "task", None)
        if task is not None:
            sr = getattr(task, "sample_rate", 24000)
    return int(sr or 24000)


def freeze_feature_extractor(model):
    """Optional: train only the shared transformer, not the CNN front-end."""
    enc = model.modality_encoders.get("AUDIO")
    if enc is None:
        return
    for mod in (enc.local_encoder, enc.project_features, enc.context_encoder):
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = False
    logger.info("Frozen audio feature extractor (local_encoder, project_features, context_encoder)")


def save_fairseq_checkpoint(path: str, student, pretrain_ckpt: str, num_updates: int):
    """Overwrite model weights in a Fairseq checkpoint; keep cfg/task for inference."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = torch.load(pretrain_ckpt, map_location="cpu")
    state["model"] = student.state_dict()
    extra = state.get("extra_state") or {}
    extra["num_updates"] = num_updates
    extra["vocal_contrastive_finetune"] = True
    state["extra_state"] = extra
    torch.save(state, path)
    logger.info("Saved checkpoint to %s", path)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA unavailable; using CPU")

    # --- Models: student learns, teacher stays at pretrain weights ---
    student, teacher = load_models(args.pretrain_ckpt, device)
    if args.freeze_feature_extractor:
        freeze_feature_extractor(student)

    conv_layers = conv_layers_from_cfg(student)
    sample_rate = sample_rate_from_cfg(student)
    logger.info("sample_rate=%s conv_layers=%s", sample_rate, conv_layers)

    # --- Data: manifest wavs + CSV vocal intervals -> encoder-frame triplets ---
    label_cfg = LabelConfig(
        label_csv_dir=args.label_csv_dir,
        label_csv_format=args.label_csv_format,
        label_index_rate_hz=args.label_index_rate_hz,
        sample_rate=float(sample_rate),
        exclude_classes=tuple(c.strip() for c in args.exclude_classes.split(",") if c.strip()),
    )

    manifest_path = os.path.join(args.manifest_dir, f"{args.train_subset}.tsv")
    dataset = VocalContrastiveDataset(
        manifest_path=manifest_path,
        sample_rate=sample_rate,
        conv_feature_layers=conv_layers,
        label_cfg=label_cfg,
        normalize=args.normalize,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No usable samples in {manifest_path} (check CSV labels and vocal intervals)")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=make_collate_fn(sample_rate, conv_layers),
        drop_last=True,
    )

    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Training loop: fixed number of optimizer steps (not epochs) ---
    step = 0
    data_iter = iter(loader)
    while step < args.max_updates:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        if not batch:
            continue

        source = batch["source"].to(device)
        lr = get_lr(step, args.warmup_updates, args.max_updates, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Pretrain checkpoints enable BC mixup in forward(); disable for contrastive finetune.
        mixup_saved = getattr(student.cfg, "source_mixup", -1)
        student.cfg.source_mixup = -1

        # Frame embeddings: (B, T, D) from top-K averaged transformer layers
        student_out = student.extract_features(source=source, mask=False)
        s_emb = average_layer_embeddings(student, student_out["layer_results"], args.average_top_k_layers)

        with torch.no_grad():
            teacher_out = teacher.extract_features(source=source, mask=False)
            t_emb = average_layer_embeddings(teacher, teacher_out["layer_results"], args.average_top_k_layers)

        student.cfg.source_mixup = mixup_saved

        # Triplet + teacher-anchor loss; skip step if no valid triplets in batch
        loss, stats = compute_contrastive_loss(
            s_emb,
            t_emb,
            batch["vocal_spans_enc"],
            batch["non_vocal_enc"],
            margin=args.margin,
            anchor_weight=args.anchor_weight,
            rng=rng,
        )

        if stats["valid_triplets"] == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        step += 1

        if step % args.log_interval == 0:
            # Healthy training: pos_dist < neg_dist, loss_triplet decreasing
            logger.info(
                "update=%d loss=%.4f triplet=%.4f anchor=%.4f pos=%.4f neg=%.4f triplets=%d lr=%.2e",
                step,
                float(loss.detach()),
                stats["loss_triplet"],
                stats["loss_anchor"],
                stats["pos_dist"],
                stats["neg_dist"],
                stats["valid_triplets"],
                lr,
            )

        if step % args.save_interval_updates == 0 or step == args.max_updates:
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_{step}.pt")
            save_fairseq_checkpoint(ckpt_path, student.cpu(), args.pretrain_ckpt, step)
            student.to(device)
            save_fairseq_checkpoint(
                os.path.join(args.save_dir, "checkpoint_last.pt"),
                student.cpu(),
                args.pretrain_ckpt,
                step,
            )
            student.to(device)

    logger.info("Training finished after %d updates", step)


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
