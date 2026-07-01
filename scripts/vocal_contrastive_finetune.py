#!/usr/bin/env python3
# Copyright (c) Max Planck Institute of Animal Behavior
#
# Standalone vocal-region contrastive finetune for data2vec_multi checkpoints.
#
# Training recipe (self-supervised, not classification):
#   - Student: pretrained data2vec_multi, updated by AdamW
#   - Data: Fairseq manifest (wav paths) + CSV vocal intervals
#   - Forward: extract_features -> average top-K layer embeddings (B, T, D)
#   - Loss: triplet margin on vocal vs non-vocal/cross-file frames
#   - Output: Fairseq .pt checkpoint compatible with animal2vec_inference.py

import argparse
import json
import logging
import math
import os
import sys
import time
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
    SamplingStats,
    VocalContrastiveDataset,
    average_layer_embeddings,
    compute_contrastive_loss,
    make_collate_fn,
)


def parse_args():
    p = argparse.ArgumentParser(description="Vocal contrastive finetune (triplet loss)")
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
    p.add_argument("--noise-negative-prob", type=float, default=0.5, help="P(noise same clip) vs other-file negative")
    p.add_argument("--average-top-k-layers", type=int, default=12)
    p.add_argument("--save-interval-updates", type=int, default=1000)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--no-normalize", action="store_false", dest="normalize")
    p.add_argument("--freeze-feature-extractor", action="store_true")
    p.add_argument(
        "--class-aware",
        action="store_true",
        help="Diagnostic: same-class pos / diff-class neg from other files in batch only",
    )
    return p.parse_args()


def get_lr(step: int, warmup: int, max_updates: int, peak_lr: float) -> float:
    """Linear warmup, then cosine decay to zero."""
    if step < warmup:
        return peak_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(max_updates - warmup, 1)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def load_student(pretrain_ckpt: str, device: str):
    """Load trainable student from pretrain checkpoint."""
    student_models, _cfg = checkpoint_utils.load_model_ensemble([pretrain_ckpt])
    student = student_models[0].to(device)
    student.train()
    return student


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


def _top_per_class_neg_rates(per_class: dict, top_n: int = 5) -> str:
    if not per_class:
        return ""
    ranked = sorted(per_class.items(), key=lambda kv: kv[1].get("anchors", 0), reverse=True)[:top_n]
    parts = []
    for name, bucket in ranked:
        rate = bucket.get("neg_diff_class_rate", 0.0)
        parts.append(f"{name}={rate:.2f}")
    return " top_neg_diff=" + ",".join(parts)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA unavailable; using CPU")

    student = load_student(args.pretrain_ckpt, device)
    if args.freeze_feature_extractor:
        freeze_feature_extractor(student)

    conv_layers = conv_layers_from_cfg(student)
    sample_rate = sample_rate_from_cfg(student)
    logger.info("sample_rate=%s conv_layers=%s", sample_rate, conv_layers)

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

    run_stats = SamplingStats()
    train_start = time.perf_counter()
    step_times: list = []
    stats_jsonl_path = os.path.join(args.save_dir, "training_stats.jsonl")

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

        step_t0 = time.perf_counter()
        source = batch["source"].to(device)
        lr = get_lr(step, args.warmup_updates, args.max_updates, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        mixup_saved = getattr(student.cfg, "source_mixup", -1)
        student.cfg.source_mixup = -1

        student_out = student.extract_features(source=source, mask=False)
        s_emb = average_layer_embeddings(student, student_out["layer_results"], args.average_top_k_layers)

        student.cfg.source_mixup = mixup_saved

        loss, stats = compute_contrastive_loss(
            s_emb,
            batch["vocal_spans_enc"],
            batch["non_vocal_enc"],
            margin=args.margin,
            rng=rng,
            class_aware=args.class_aware,
            vocal_span_classes=batch.get("vocal_span_classes"),
            noise_prob=args.noise_negative_prob,
            stats_accumulator=run_stats,
        )

        if stats["valid_triplets"] == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        step += 1
        step_times.append(time.perf_counter() - step_t0)

        if step % args.log_interval == 0:
            recent = step_times[-args.log_interval :]
            sec_per_update = sum(recent) / max(len(recent), 1)
            neg_diff_rate = stats.get("neg_diff_class_rate", 0.0)
            neg_same_rate = stats.get("neg_same_class_rate", 0.0)
            pos_same_rate = stats.get("pos_same_class_rate", 0.0)
            per_class_hint = _top_per_class_neg_rates(stats.get("per_class_sampling", {}))
            logger.info(
                "update=%d loss=%.4f triplet=%.4f pos=%.4f neg=%.4f triplets=%d "
                "pos_same=%.2f neg_diff=%.2f neg_same=%.2f sec/up=%.3f lr=%.2e%s",
                step,
                float(loss.detach()),
                stats["loss_triplet"],
                stats["pos_dist"],
                stats["neg_dist"],
                stats["valid_triplets"],
                pos_same_rate,
                neg_diff_rate,
                neg_same_rate,
                sec_per_update,
                lr,
                per_class_hint,
            )
            log_row = {
                "update": step,
                "loss": float(loss.detach()),
                "sec_per_update": sec_per_update,
                **{k: stats[k] for k in stats if k != "per_class_sampling"},
                "per_class_sampling": stats.get("per_class_sampling", {}),
            }
            with open(stats_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_row) + "\n")

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

    total_wall_s = time.perf_counter() - train_start
    mean_step_s = sum(step_times) / max(len(step_times), 1)
    median_step_s = float(np.median(step_times)) if step_times else 0.0

    summary = {
        "args": vars(args),
        "sampling_config": {
            "noise_prob": args.noise_negative_prob,
            "same_clip_vocal_negative": False,
            "class_aware": args.class_aware,
        },
        "runtime": {
            "total_wall_seconds": total_wall_s,
            "total_wall_hours": total_wall_s / 3600.0,
            "mean_sec_per_update": mean_step_s,
            "median_sec_per_update": median_step_s,
            "num_updates": step,
        },
        "sampling_stats": run_stats.to_dict(),
    }
    summary_path = os.path.join(args.save_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Training finished after %d updates (wall=%.1fs, mean_sec/up=%.3f, summary=%s)",
        step,
        total_wall_s,
        mean_step_s,
        summary_path,
    )


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
