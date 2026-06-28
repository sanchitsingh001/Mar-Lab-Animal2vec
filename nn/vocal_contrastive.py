# Copyright (c) Max Planck Institute of Animal Behavior
#
# Vocal-region contrastive finetune: dataset, CSV parsing, triplet loss.
#
# Overview (for code review):
#   1. CSV labels define vocalization time intervals; class names are only used
#      to filter out Empty/Unknown/silence rows — we do NOT train a classifier.
#   2. Intervals are mapped from wav samples -> encoder frame indices (one frame
#      per CNN+transformer time step) so triplets are sampled in embedding space.
#   3. Loss = triplet margin on cosine distance + small teacher-anchor term:
#        - every frame in each vocal span is an anchor; positive from same span
#        - negative: non-vocal frame, different span, or different file in batch
#        - anchor term: keep student embeddings close to frozen teacher (pretrain)

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

ClassKey = Union[int, str]

import numpy as np
import torch
import torch.nn.functional as F
from scipy import interpolate
from torch.utils.data import Dataset

from nn.utils import get_conv_size

logger = logging.getLogger("animal2vec.vocal_contrastive")

_SEGMENT_WINDOW_RE = re.compile(r"_(\d{5})s_(\d{5})s$")
DEFAULT_EXCLUDE_CLASSES = ("empty", "unknown", "silence", "unk")

# Minimal NIPS class table for index-based CSV rows (class id in col 3).
NIPS_CLASS_NAMES = [
    "Empty", "Aegcau_call", "Alaarv_song", "Anttri_song", "Butbut_call", "Carcan_call",
    "Carcan_song", "Carcar_call", "Carcar_song", "Cerbra_call", "Cerbra_song", "Cetcet_song",
    "Chlchl_call", "Cicatr_song", "Cicorn_song", "Cisjun_song", "Colpal_song", "Corcor_call",
    "Denmaj_call", "Denmaj_drum", "Embcir_call", "Embcir_song", "Erirub_call", "Erirub_song",
    "Fricoe_call", "Fricoe_song", "Galcri_call", "Galcri_song", "Galthe_call", "Galthe_song",
    "Gargla_call", "Hirrus_call", "Jyntor_song", "Lopcri_call", "Loxcur_call", "Lularb_song",
    "Lusmeg_call", "Lusmeg_song", "Lyrple_song", "Motcin_call", "Musstr_call", "Oriori_call",
    "Oriori_song", "Parate_call", "Parate_song", "Parcae_call", "Parcae_song", "Parmaj_call",
    "Parmaj_song", "Pasdom_call", "Pelgra_call", "Petpet_call", "Petpet_song", "Phofem_song",
    "Phycol_call", "Phycol_song", "Picpic_call", "Plaaff_song", "Plasab_song", "Poepal_call",
    "Poepal_song", "Prumod_song", "Ptehey_song", "Pyrpyr_call", "Regign_call", "Regign_song",
    "Serser_call", "Serser_song", "Siteur_call", "Siteur_song", "Strdec_song", "Strtur_song",
    "Stuvul_call", "Sylatr_call", "Sylatr_song", "Sylcan_call", "Sylcan_song", "Sylmel_call",
    "Sylmel_song", "Sylund_call", "Sylund_song", "Tetpyg_song", "Tibtom_song", "Trotro_song",
    "Turmer_call", "Turmer_song", "Turphi_call", "Turphi_song", "Unknown",
]
EMPTY_CLASS_ID = 0
UNKNOWN_CLASS_ID = len(NIPS_CLASS_NAMES) - 1


@dataclass
class LabelConfig:
    label_csv_dir: Optional[str] = None
    label_csv_format: str = "auto"
    label_index_rate_hz: str = "44100"
    sample_rate: float = 24000.0
    exclude_classes: Sequence[str] = DEFAULT_EXCLUDE_CLASSES


def _split_label_line(line: str) -> List[str]:
    if "\t" in line:
        return [p.strip() for p in line.split("\t")]
    return [p.strip() for p in line.split(",")]


def _parse_timestamp_seconds(value: str) -> Optional[float]:
    s = value.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    if ":" not in s:
        return None
    try:
        parts = s.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
    except ValueError:
        return None
    return None


def _label_from_field(field: str) -> Tuple[Optional[str], Optional[int]]:
    f = field.strip()
    if not f:
        return None, None
    try:
        cid = int(float(f))
        if 0 <= cid < len(NIPS_CLASS_NAMES):
            return NIPS_CLASS_NAMES[cid], cid
        return None, cid
    except ValueError:
        return f, None


def _normalize_class_name(name: str) -> str:
    return name.strip().lower()


def _class_key_from_label(label_name: Optional[str], class_id: Optional[int]) -> ClassKey:
    if class_id is not None:
        return class_id
    if label_name:
        return _normalize_class_name(label_name.split()[0])
    return "unknown"


def _build_exclude_sets(exclude_classes: Optional[Sequence[str]]):
    if not exclude_classes:
        return None, None
    names = {_normalize_class_name(c) for c in exclude_classes if str(c).strip()}
    ids = set()
    for i, cn in enumerate(NIPS_CLASS_NAMES):
        if _normalize_class_name(cn) in names:
            ids.add(i)
    if "empty" in names or "silence" in names:
        ids.add(EMPTY_CLASS_ID)
    if "unknown" in names or "unk" in names:
        ids.add(UNKNOWN_CLASS_ID)
    return names, ids


def _row_is_excluded(label_name, class_id, exclude_names, exclude_ids) -> bool:
    if exclude_names is None and exclude_ids is None:
        return False
    if class_id is not None and exclude_ids is not None and class_id in exclude_ids:
        return True
    if label_name and exclude_names is not None:
        base = label_name.split()[0]
        if _normalize_class_name(base) in exclude_names:
            return True
        if _normalize_class_name(label_name) in exclude_names:
            return True
    return False


def _detect_label_csv_format(lines: Sequence[str]) -> str:
    for line in lines:
        parts = _split_label_line(line.strip())
        if len(parts) < 2:
            continue
        head = " ".join(parts[:3]).lower()
        if "name" in parts[0].lower() and "start" in head:
            return "audacity"
        if len(parts) >= 3:
            _, cid = _label_from_field(parts[2])
            try:
                a = float(parts[0])
                b = float(parts[1])
            except ValueError:
                continue
            if cid is not None and b > a and (a >= 50 or b >= 50):
                return "indices"
            if cid is not None and 0 <= cid < len(NIPS_CLASS_NAMES) and b > a and b > 1:
                return "indices"
        try:
            float(parts[0])
            float(parts[1])
            return "seconds"
        except ValueError:
            continue
    return "seconds"


def _infer_index_rate_hz(raw_lines: Sequence[str], true_dur_s: float, fallback_hz: float) -> float:
    max_end = 0.0
    for line in raw_lines:
        parts = _split_label_line(line.strip())
        if len(parts) < 2:
            continue
        try:
            max_end = max(max_end, float(parts[1]))
        except ValueError:
            continue
    if max_end <= 0 or true_dur_s <= 0:
        return float(fallback_hz)
    inferred = max_end / true_dur_s
    for cand in (8000.0, 16000.0, 22050.0, 24000.0, 44100.0, 48000.0):
        if abs(inferred - cand) / max(cand, 1.0) < 0.12:
            return cand
    return float(inferred)


def resolve_index_rate_hz(
    label_index_rate_hz: str,
    sample_rate: float,
    raw_lines: Sequence[str],
    fmt: str,
    true_dur_s: float,
) -> float:
    mode = str(label_index_rate_hz).strip().lower()
    if mode in ("auto", ""):
        if fmt == "indices" and true_dur_s > 0:
            return _infer_index_rate_hz(raw_lines, true_dur_s, sample_rate)
        return float(sample_rate)
    if mode in ("sample", "sample-rate", "match-sample-rate"):
        return float(sample_rate)
    return float(mode)


def read_vocal_intervals_seconds(
    csv_path: str,
    cfg: LabelConfig,
    true_dur_s: Optional[float] = None,
    segment_offset_s: float = 0.0,
    clip_dur_s: Optional[float] = None,
) -> List[Tuple[float, float, ClassKey]]:
    """Return clip-local half-open vocal intervals [start, end, class_key) in seconds.

    Supports NIPS index CSVs, start+duration seconds, and Audacity exports.
    Rows labeled Empty/Unknown (by name or NIPS class id) are dropped.
    class_key is NIPS class_id when present, else normalized label name.
    """
    if not os.path.isfile(csv_path):
        return []

    with open(csv_path, "r") as f:
        raw_lines = [ln.strip() for ln in f if ln.strip()]

    fmt = cfg.label_csv_format
    if fmt == "auto":
        fmt = _detect_label_csv_format(raw_lines)

    index_rate_hz = resolve_index_rate_hz(
        cfg.label_index_rate_hz, cfg.sample_rate, raw_lines, fmt, true_dur_s or 0.0
    )
    exclude_names, exclude_ids = _build_exclude_sets(cfg.exclude_classes)

    intervals: List[Tuple[float, float, ClassKey]] = []
    skip_header = fmt == "audacity"

    for line in raw_lines:
        parts = _split_label_line(line)
        if len(parts) < 2:
            continue
        if skip_header and "name" in parts[0].lower() and "start" in " ".join(parts[:3]).lower():
            skip_header = False
            continue

        label_name: Optional[str] = None
        class_id: Optional[int] = None
        start_s: Optional[float] = None
        end_s: Optional[float] = None

        if fmt == "audacity":
            label_name = parts[0]
            start_s = _parse_timestamp_seconds(parts[1])
            dur_s = _parse_timestamp_seconds(parts[2]) if len(parts) > 2 else None
            if start_s is None or dur_s is None or dur_s <= 0:
                continue
            end_s = start_s + dur_s
        else:
            try:
                a = float(parts[0])
                b = float(parts[1])
            except ValueError:
                continue
            if len(parts) >= 3:
                label_name, class_id = _label_from_field(parts[2])
            if fmt == "indices":
                start_s = a / float(index_rate_hz)
                end_s = b / float(index_rate_hz)
            elif fmt == "seconds_start_end":
                start_s = a
                end_s = b
            else:
                if b <= 0:
                    continue
                start_s = a
                end_s = a + b

        if start_s is None or end_s is None or end_s <= start_s:
            continue
        if _row_is_excluded(label_name, class_id, exclude_names, exclude_ids):
            continue
        intervals.append((start_s, end_s, _class_key_from_label(label_name, class_id)))

    intervals.sort(key=lambda x: x[0])

    # Shift recording-level CSV to segment-local time.
    if segment_offset_s > 0:
        intervals = [
            (s - segment_offset_s, e - segment_offset_s, ck)
            for s, e, ck in intervals
            if e > segment_offset_s and s < (clip_dur_s or float("inf")) + segment_offset_s
        ]

    if clip_dur_s is not None:
        out = []
        for s, e, ck in intervals:
            s = max(0.0, s)
            e = min(float(clip_dur_s), e)
            if e > s:
                out.append((s, e, ck))
        intervals = out

    return intervals


def parse_segment_offset_seconds(wav_stem: str) -> Tuple[str, float]:
    """Parse foo_00000s_00005s -> (recording_stem, offset_s).

    Segment wavs share one CSV keyed by recording stem; offset shifts intervals.
    """
    m = _SEGMENT_WINDOW_RE.search(wav_stem)
    if not m:
        return wav_stem, 0.0
    rec_stem = _SEGMENT_WINDOW_RE.sub("", wav_stem)
    offset_s = int(m.group(1))
    return rec_stem, float(offset_s)


def resolve_label_csv(wav_path: Path, wav_root: Path, label_csv_dir: Optional[Path]) -> Optional[Path]:
    """Find CSV for a wav: per-file stem, then recording stem, then csv/ subdirs."""
    stem = wav_path.stem
    rec_stem, _ = parse_segment_offset_seconds(stem)
    search_dirs: List[Path] = []
    if label_csv_dir is not None:
        search_dirs.append(label_csv_dir)
    search_dirs.extend([wav_path.parent, wav_root / "csv", wav_root])

    seen = set()
    candidates = []
    for d in search_dirs:
        if d is None:
            continue
        d = Path(d)
        key = str(d.resolve())
        if key in seen:
            continue
        seen.add(key)
        for name in (f"{stem}.csv", f"{rec_stem}.csv"):
            p = d / name
            if p.is_file():
                candidates.append(p)

    return candidates[0] if candidates else None


def encoder_output_length(num_audio_samples: int, conv_feature_layers: List[Tuple[int, int, int]]) -> int:
    ft_out_size = [num_audio_samples]
    for xx in conv_feature_layers:
        ft_out_size = [get_conv_size(ft_out_size, [xx[1]], [0], [1], [xx[2]], dim=1)[0]]
    return int(np.array(ft_out_size).squeeze())


def intervals_to_encoder_frame_lists(
    intervals_s: Sequence[Tuple],
    wav_len: int,
    sample_rate: float,
    conv_feature_layers: List[Tuple[int, int, int]],
) -> Tuple[List[List[int]], List[int], List[ClassKey]]:
    """Map vocal intervals to encoder frame indices; also return non-vocal frames.

    Each vocal span needs >= 2 encoder frames so we can sample anchor+positive
    from the same span. Linear interpolation approximates the CNN downsampling.
    Accepts (start, end) or (start, end, class_key) interval tuples.
    """
    t_enc = encoder_output_length(wav_len, conv_feature_layers)
    if t_enc <= 0:
        return [], list(range(max(t_enc, 0))), []

    wav_idx = np.arange(wav_len, dtype=np.float64)
    enc_idx = np.linspace(0, wav_len, t_enc, endpoint=False)

    vocal_spans_enc: List[List[int]] = []
    vocal_span_classes: List[ClassKey] = []
    vocal_mask = np.zeros(t_enc, dtype=bool)

    for interval in intervals_s:
        start_s, end_s = interval[0], interval[1]
        class_key = interval[2] if len(interval) >= 3 else "unknown"
        s_sample = int(np.floor(start_s * sample_rate))
        e_sample = int(np.ceil(end_s * sample_rate))
        s_sample = max(0, min(s_sample, wav_len - 1))
        e_sample = max(s_sample + 1, min(e_sample, wav_len))

        span_wav = np.zeros(wav_len, dtype=np.int64)
        span_wav[s_sample:e_sample] = 1
        f = interpolate.interp1d(wav_idx, span_wav, axis=0, kind="linear", fill_value=0, bounds_error=False)
        span_enc = np.round(f(enc_idx)).astype(bool)
        frames = np.where(span_enc)[0].tolist()
        if len(frames) >= 2:
            vocal_spans_enc.append(frames)
            vocal_span_classes.append(class_key)
            vocal_mask[span_enc] = True

    non_vocal_enc = np.where(~vocal_mask)[0].tolist()
    return vocal_spans_enc, non_vocal_enc, vocal_span_classes


def load_wav_mono(path: str, target_sr: int) -> torch.Tensor:
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    feats = torch.from_numpy(wav).float()
    if sr != target_sr:
        import torchaudio

        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        feats = resampler(feats.unsqueeze(0)).squeeze(0)
    return feats


def normalize_audio(feats: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(feats, feats.shape)


class VocalContrastiveDataset(Dataset):
    """Fairseq manifest + per-recording CSV -> wav tensor + encoder-frame spans.

    Skips clips with no CSV, no vocal intervals after filtering, or missing wav.
    """

    def __init__(
        self,
        manifest_path: str,
        sample_rate: int,
        conv_feature_layers: List[Tuple[int, int, int]],
        label_cfg: LabelConfig,
        normalize: bool = True,
        max_sample_size: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.conv_feature_layers = conv_feature_layers
        self.label_cfg = label_cfg
        self.normalize = normalize
        self.max_sample_size = max_sample_size

        self.root_dir = ""
        self.items: List[dict] = []

        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            wav_root = Path(self.root_dir)
            label_dir = Path(label_cfg.label_csv_dir) if label_cfg.label_csv_dir else None
            skipped = 0

            for i, line in enumerate(f):
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                rel_path, _sz = parts[0], int(parts[1])
                wav_path = wav_root / rel_path
                if not wav_path.is_file():
                    skipped += 1
                    continue

                try:
                    import soundfile as sf

                    info = sf.info(str(wav_path))
                    clip_dur_s = info.duration
                except Exception:
                    skipped += 1
                    continue

                rec_stem, seg_offset = parse_segment_offset_seconds(wav_path.stem)
                csv_path = resolve_label_csv(wav_path, wav_root, label_dir)
                if csv_path is None:
                    skipped += 1
                    continue

                intervals = read_vocal_intervals_seconds(
                    str(csv_path),
                    label_cfg,
                    true_dur_s=clip_dur_s + seg_offset,
                    segment_offset_s=seg_offset,
                    clip_dur_s=clip_dur_s,
                )
                if not intervals:
                    skipped += 1
                    continue

                self.items.append(
                    {
                        "id": i,
                        "rel_path": rel_path,
                        "wav_path": str(wav_path),
                        "csv_path": str(csv_path),
                        "intervals_s": intervals,
                    }
                )

        logger.info("VocalContrastiveDataset: loaded %d, skipped %d", len(self.items), skipped)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        feats = load_wav_mono(item["wav_path"], self.sample_rate)
        if self.normalize:
            feats = normalize_audio(feats)
        if self.max_sample_size is not None and feats.numel() > self.max_sample_size:
            start = torch.randint(0, feats.numel() - self.max_sample_size + 1, (1,)).item()
            feats = feats[start : start + self.max_sample_size]
            # Shift interval times when random crop is applied (rare; off by default).
            intervals_s = [
                (max(0.0, s - start / self.sample_rate), max(0.0, e - start / self.sample_rate), ck)
                for s, e, ck in item["intervals_s"]
            ]
            clip_dur_s = feats.numel() / self.sample_rate
            intervals_s = [(s, min(e, clip_dur_s), ck) for s, e, ck in intervals_s if e > s]
        else:
            intervals_s = item["intervals_s"]

        wav_len = feats.numel()
        vocal_spans, non_vocal, span_classes = intervals_to_encoder_frame_lists(
            intervals_s, wav_len, self.sample_rate, self.conv_feature_layers
        )

        return {
            "id": item["id"],
            "source": feats,
            "intervals_s": intervals_s,
            "vocal_spans_enc": vocal_spans,
            "non_vocal_enc": non_vocal,
            "vocal_span_classes": span_classes,
        }


def make_collate_fn(sample_rate: int, conv_feature_layers: List[Tuple[int, int, int]]):
    """Build collate_fn that re-maps vocal spans to the cropped batch audio length.

    Clips in a batch are truncated to the shortest length; vocal spans are
    recomputed on the cropped audio so frame indices match the forward pass.
    """

    def collate_vocal_contrastive(samples):
        samples = [s for s in samples if s.get("source") is not None]
        if not samples:
            return {}

        min_len = min(s["source"].numel() for s in samples)
        sources = torch.stack([s["source"][:min_len] for s in samples])
        clip_dur_s = min_len / float(sample_rate)

        vocal_spans_enc = []
        non_vocal_enc = []
        vocal_span_classes = []
        for s in samples:
            intervals = [
                (max(0.0, st), min(en, clip_dur_s), ck)
                for st, en, ck in s["intervals_s"]
                if en > st and st < clip_dur_s
            ]
            spans, non_voc, span_cls = intervals_to_encoder_frame_lists(
                intervals, min_len, sample_rate, conv_feature_layers
            )
            vocal_spans_enc.append(spans)
            non_vocal_enc.append(non_voc)
            vocal_span_classes.append(span_cls)

        return {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "source": sources,
            "vocal_spans_enc": vocal_spans_enc,
            "non_vocal_enc": non_vocal_enc,
            "vocal_span_classes": vocal_span_classes,
        }

    return collate_vocal_contrastive


def average_layer_embeddings(model, layer_results, average_top_k_layers: int) -> torch.Tensor:
    """Return (B, T, D) frame embeddings averaged over the top-K transformer layers.

    Matches animal2vec_inference: use the last K layer outputs (pretrain vs
    finetuned checkpoints differ slightly in layer_results structure).
    """
    finetuned = hasattr(model, "w2v_encoder")
    if finetuned:
        target = [l[0] for l in layer_results[-average_top_k_layers:]]
    else:
        target = layer_results[-average_top_k_layers:]
    return (sum(target) / len(target)).float()


def sample_negative_frame(
    batch_idx: int,
    anchor_span_idx: int,
    vocal_spans_enc: List[List[List[int]]],
    non_vocal_enc: List[List[int]],
    rng: np.random.Generator,
) -> int:
    """Pick negative encoder frame; never from the same vocalization span as the anchor.

    Roughly equal random choice among:
      (0) non-vocal frame in the same clip
      (1) vocal frame from a different span in the same clip
      (2) vocal frame from a different clip in the batch
  Fallbacks ensure a negative is always found when possible.
    """
    b = batch_idx
    choice = rng.integers(0, 3)

    # Prefer non-vocal silence/background within the same recording.
    if choice == 0 and non_vocal_enc[b]:
        return int(rng.choice(non_vocal_enc[b]))

    other_span_indices = [i for i in range(len(vocal_spans_enc[b])) if i != anchor_span_idx]
    # Different vocalization in the same clip (e.g. two bird calls).
    if choice == 1 and other_span_indices:
        si = int(rng.choice(other_span_indices))
        return int(rng.choice(vocal_spans_enc[b][si]))

    # Vocal frame from a different recording in the same batch.
    other = [j for j in range(len(vocal_spans_enc)) if j != b and vocal_spans_enc[j]]
    if other:
        j = int(rng.choice(other))
        si = int(rng.integers(0, len(vocal_spans_enc[j])))
        return int(rng.choice(vocal_spans_enc[j][si]))

    if non_vocal_enc[b]:
        return int(rng.choice(non_vocal_enc[b]))
    if other_span_indices:
        si = int(rng.choice(other_span_indices))
        return int(rng.choice(vocal_spans_enc[b][si]))

    anchor_span = vocal_spans_enc[b][anchor_span_idx]
    t_enc = max(anchor_span) + 1
    candidates = [i for i in range(t_enc) if i not in set(anchor_span)]
    return int(rng.choice(candidates)) if candidates else int(rng.choice(anchor_span))


def sample_class_aware_positive(
    batch_idx: int,
    anchor_span_idx: int,
    anchor_class: ClassKey,
    vocal_spans_enc: List[List[List[int]]],
    vocal_span_classes: List[List[ClassKey]],
    rng: np.random.Generator,
    anchor_frame_idx: Optional[int] = None,
) -> Tuple[int, int, int, bool]:
    """Return (batch_idx, frame_idx, pos_span_idx, used_same_class).

    Prefer a vocal frame from a different span with the same class_key.
  Fallback: another frame from the anchor span (distinct from anchor when possible).
    """
    b = batch_idx
    anchor_span = vocal_spans_enc[b][anchor_span_idx]
    e_idx = anchor_frame_idx if anchor_frame_idx is not None else int(rng.choice(anchor_span))

    other_spans = [
        (bb, si)
        for bb in range(len(vocal_spans_enc))
        for si, cls in enumerate(vocal_span_classes[bb])
        if cls == anchor_class and vocal_spans_enc[bb][si] and (bb != b or si != anchor_span_idx)
    ]
    if other_spans:
        bb, si = other_spans[int(rng.integers(0, len(other_spans)))]
        e1_idx = int(rng.choice(vocal_spans_enc[bb][si]))
        return bb, e1_idx, si, True

    others = [i for i in anchor_span if i != e_idx]
    if others:
        return b, int(rng.choice(others)), anchor_span_idx, False
    if len(anchor_span) >= 2:
        e_idx, e1_idx = rng.choice(anchor_span, size=2, replace=False).tolist()
        return b, e1_idx, anchor_span_idx, False
    return b, e_idx, anchor_span_idx, False


def sample_class_aware_negative(
    batch_idx: int,
    anchor_span_idx: int,
    anchor_class: ClassKey,
    vocal_spans_enc: List[List[List[int]]],
    vocal_span_classes: List[List[ClassKey]],
    non_vocal_enc: List[List[int]],
    rng: np.random.Generator,
) -> Tuple[int, int, bool]:
    """Return (batch_idx, frame_idx, used_diff_class).

    Prefer a vocal frame from a span with class_key != anchor_class.
  Fallback: sample_negative_frame (non-vocal / other span / other file).
    """
    diff_spans = [
        (bb, si)
        for bb in range(len(vocal_spans_enc))
        for si, cls in enumerate(vocal_span_classes[bb])
        if cls != anchor_class and vocal_spans_enc[bb][si]
    ]
    if diff_spans:
        bb, si = diff_spans[int(rng.integers(0, len(diff_spans)))]
        return bb, int(rng.choice(vocal_spans_enc[bb][si])), True

    frame = sample_negative_frame(batch_idx, anchor_span_idx, vocal_spans_enc, non_vocal_enc, rng)
    return batch_idx, frame, False


def compute_contrastive_loss(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    vocal_spans_enc: List[List[List[int]]],
    non_vocal_enc: List[List[int]],
    margin: float = 0.2,
    anchor_weight: float = 0.1,
    rng: Optional[np.random.Generator] = None,
    class_aware: bool = False,
    vocal_span_classes: Optional[List[List[ClassKey]]] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Contrastive triplet loss on per-frame encoder embeddings.

    Goal: finetune the student model so vocal frames cluster by vocalization while
    staying close to the frozen pretrained (teacher) representations.

    Inputs
    ------
    student_emb / teacher_emb : (B, T, D)
        Frame embeddings from the trainable student and frozen teacher encoders.
        B = batch size, T = encoder time steps, D = embedding dim.
    vocal_spans_enc : List[List[List[int]]]
        Per-clip vocalization spans mapped to encoder frame indices.
        Shape conceptually: [batch_idx][vocalization_idx] -> [frame, frame, ...]
        Each inner list is one labeled vocal event (e.g. one bird call).
    non_vocal_enc : List[List[int]]
        Per-clip encoder frames that fall outside any vocal span (background/silence).
    class_aware : bool
        If True, positives prefer same species/call label; negatives prefer diff label.

    Triplet construction (matches professor pseudocode)
    ---------------------------------------------------
    for each clip b in batch:
        for each vocalization span in clip b:
            for each encoder frame e_idx in that span:
                anchor   = student_emb[b, e_idx]
                positive = another frame from the same vocalization (or same class)
                negative = frame from non-vocal / other span / other clip in batch
                collect (anchor, positive, negative)

    Loss
    ----
    1. Triplet margin: max(0, d(anchor,pos) - d(anchor,neg) + margin) on cosine distance
       -> pull anchor closer to positive, push anchor away from negative.
    2. Teacher anchor (weighted): keep student embeddings near teacher at all three frames
       -> prevents catastrophic drift away from pretraining.

    Returns scalar loss and a stats dict for logging.
    """
    rng = rng or np.random.default_rng()
    bsz, t_enc, _ = student_emb.shape

    # Collect one (anchor, positive, negative) triplet per vocal frame.
    anchors, positives, negatives = [], [], []
    # Same triplets from the frozen teacher — used only for the anchor regularizer.
    t_anchors, t_positives, t_negatives = [], [], []
    # Counters for class-aware sampling diagnostics (logged when class_aware=True).
    pos_same_class = 0
    neg_diff_class = 0

    # --- Triplet sampling: batch -> vocalization -> frame ---
    for b in range(bsz):
        spans = vocal_spans_enc[b]  # all vocalizations in clip b
        if not spans:
            continue

        for span_idx, span in enumerate(spans):
            # Need >= 2 frames so we can pick a distinct positive from the same span.
            if len(span) < 2:
                continue

            # Every frame in this vocalization becomes an anchor.
            for e_idx in span:
                if class_aware and vocal_span_classes is not None:
                    anchor_class = vocal_span_classes[b][span_idx]
                    # Positive: prefer same-class vocal frame from another span; else same span.
                    pos_b, e1_idx, _pos_si, same_cls = sample_class_aware_positive(
                        b,
                        span_idx,
                        anchor_class,
                        vocal_spans_enc,
                        vocal_span_classes,
                        rng,
                        anchor_frame_idx=e_idx,
                    )
                    # Negative: prefer different-class vocal frame; else sample_negative_frame().
                    neg_b, e2_idx, diff_cls = sample_class_aware_negative(
                        b, span_idx, anchor_class, vocal_spans_enc, vocal_span_classes, non_vocal_enc, rng
                    )
                    if same_cls:
                        pos_same_class += 1
                    if diff_cls:
                        neg_diff_class += 1
                else:
                    # Default path: positive from same vocalization, negative via 3-way sampler.
                    others = [i for i in span if i != e_idx]
                    if not others:
                        continue
                    e1_idx = int(rng.choice(others))
                    e2_idx = sample_negative_frame(b, span_idx, vocal_spans_enc, non_vocal_enc, rng)
                    pos_b, neg_b = b, b  # both from same clip unless class_aware moved them

                # Clamp indices in case span mapping and tensor length disagree slightly.
                e2_idx = min(max(0, e2_idx), t_enc - 1)
                e1_idx = min(max(0, e1_idx), t_enc - 1)
                e_idx = min(max(0, e_idx), t_enc - 1)

                # Gather student embeddings for this triplet.
                anchors.append(student_emb[b, e_idx])
                positives.append(student_emb[pos_b, e1_idx])
                negatives.append(student_emb[neg_b, e2_idx])
                # Matching teacher embeddings (anchor always from clip b; pos/neg may differ).
                t_anchors.append(teacher_emb[b, e_idx])
                t_positives.append(teacher_emb[pos_b, e1_idx])
                t_negatives.append(teacher_emb[neg_b, e2_idx])

    if not anchors:
        zero = student_emb.new_zeros(())
        return zero, {"valid_triplets": 0}

    # Stack all triplets into (N, D) tensors and L2-normalize for cosine distance.
    a = F.normalize(torch.stack(anchors), dim=-1)
    p = F.normalize(torch.stack(positives), dim=-1)
    n = F.normalize(torch.stack(negatives), dim=-1)

    # Triplet loss: want d(anchor, pos) + margin < d(anchor, neg).
    # Cosine distance = 1 - cosine_similarity; ReLU zeroes satisfied triplets.
    d_pos = 1.0 - (a * p).sum(dim=-1)
    d_neg = 1.0 - (a * n).sum(dim=-1)
    loss_triplet = F.relu(d_pos - d_neg + margin).mean()

    # Regularizer: student should not drift far from frozen teacher on anchor/pos/neg frames.
    ta = F.normalize(torch.stack(t_anchors), dim=-1)
    tp = F.normalize(torch.stack(t_positives), dim=-1)
    tn = F.normalize(torch.stack(t_negatives), dim=-1)
    loss_anchor = (
        (1.0 - (a * ta).sum(dim=-1)).mean()
        + (1.0 - (p * tp).sum(dim=-1)).mean()
        + (1.0 - (n * tn).sum(dim=-1)).mean()
    ) / 3.0

    loss = loss_triplet + anchor_weight * loss_anchor
    stats = {
        "valid_triplets": len(anchors),
        "loss_triplet": float(loss_triplet.detach().cpu()),
        "loss_anchor": float(loss_anchor.detach().cpu()),
        "pos_dist": float(d_pos.mean().detach().cpu()),
        "neg_dist": float(d_neg.mean().detach().cpu()),
    }
    if class_aware:
        stats["pos_same_class"] = pos_same_class
        stats["neg_diff_class"] = neg_diff_class
    return loss, stats
