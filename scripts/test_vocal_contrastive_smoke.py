#!/usr/bin/env python3
"""Smoke tests for vocal contrastive module (no GPU / checkpoint required)."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from nn.vocal_contrastive import (
    LabelConfig,
    compute_contrastive_loss,
    intervals_to_encoder_frame_lists,
    read_vocal_intervals_seconds,
)


def test_interval_mapping():
    conv = [(127, 63, 1), (512, 10, 5), (512, 3, 4), (512, 3, 3), (512, 3, 2), (512, 3, 1), (512, 2, 1), (512, 2, 1)]
    sr = 24000
    wav_len = sr * 5  # 5 s clip
    intervals = [(1.0, 2.0, 3), (3.0, 3.5, 5)]
    vocal, non_vocal, classes = intervals_to_encoder_frame_lists(intervals, wav_len, sr, conv)
    assert len(vocal) == 2, vocal
    assert len(classes) == 2
    assert all(len(v) >= 2 for v in vocal)
    assert len(non_vocal) > 0
    print("OK test_interval_mapping:", [len(v) for v in vocal], "non_vocal", len(non_vocal), "classes", classes)


def test_csv_indices():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("1000,5000,3\n")
        f.write("8000,12000,5\n")
        path = f.name
    cfg = LabelConfig(label_csv_format="indices", label_index_rate_hz="44100", sample_rate=24000)
    iv = read_vocal_intervals_seconds(path, cfg, true_dur_s=10.0, clip_dur_s=10.0)
    assert len(iv) == 2
    assert iv[0][0] < iv[0][1]
    assert iv[0][2] == 3
    assert iv[1][2] == 5
    print("OK test_csv_indices:", iv)


def test_contrastive_loss():
    b, t, d = 4, 50, 32
    s = torch.randn(b, t, d, requires_grad=True)
    t_emb = s.detach().clone()
    vocal = [[[10, 11, 12, 13]], [[5, 6, 7]], [[20, 21, 22]], [[30, 31, 32, 33]]]
    non_vocal = [list(range(0, 5)), list(range(40, 50)), list(range(0, 4)), list(range(35, 50))]
    loss, stats = compute_contrastive_loss(s, t_emb, vocal, non_vocal, margin=0.2, anchor_weight=0.1)
    assert stats["valid_triplets"] > 0
    assert loss.requires_grad
    loss.backward()
    print("OK test_contrastive_loss:", stats)


def test_class_aware_contrastive_loss():
    b, t, d = 3, 50, 32
    s = torch.randn(b, t, d, requires_grad=True)
    t_emb = s.detach().clone()
    # clip0: class 1; clip1: class 2; clip2: class 1 (same as clip0)
    vocal = [
        [[10, 11, 12, 13]],
        [[5, 6, 7, 8]],
        [[20, 21, 22, 23]],
    ]
    classes = [[1], [2], [1]]
    non_vocal = [list(range(0, 5)), list(range(40, 50)), list(range(35, 50))]
    rng = np.random.default_rng(0)
    loss, stats = compute_contrastive_loss(
        s,
        t_emb,
        vocal,
        non_vocal,
        margin=0.2,
        anchor_weight=0.1,
        rng=rng,
        class_aware=True,
        vocal_span_classes=classes,
    )
    assert stats["valid_triplets"] > 0
    assert stats.get("pos_same_class", 0) > 0
    assert stats.get("neg_diff_class", 0) > 0
    assert loss.requires_grad
    loss.backward()
    print("OK test_class_aware_contrastive_loss:", stats)


if __name__ == "__main__":
    test_interval_mapping()
    test_csv_indices()
    test_contrastive_loss()
    test_class_aware_contrastive_loss()
    print("All smoke tests passed.")
