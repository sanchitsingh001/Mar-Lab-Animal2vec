# inference_app.py
import sys
import os
import io
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Avoid OpenBLAS/MKL thread oversubscription segfaults under Streamlit.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Allow running from the UI/ directory while importing project-local modules.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Xeno / data2vec_multi pretrain and vocal-contrastive checkpoints use 24 kHz audio.
DEFAULT_SAMPLE_RATE = 24000

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import librosa
import h5py
import base64

try:
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    rand_score,
    silhouette_score,
)

try:
    import hdbscan  # type: ignore
    _HAS_HDBSCAN = True
except Exception:
    hdbscan = None
    _HAS_HDBSCAN = False

try:
    import soundfile as sf  # type: ignore
    _HAS_SOUNDFILE = True
except Exception:
    _HAS_SOUNDFILE = False
    import wave

try:
    import umap  # type: ignore

    _HAS_UMAP = True
except Exception:
    umap = None
    _HAS_UMAP = False


# =========================
# Page config + theme
# =========================
st.set_page_config(layout="wide", page_title="Animal2Vec Remote Analysis")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

with st.sidebar:
    if st.button("Toggle Theme"):
        st.session_state.dark_mode = not st.session_state.dark_mode

hide_bar_css = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container { padding-top: 1rem; }
</style>
"""

if st.session_state.dark_mode:
    st.markdown(
        hide_bar_css
        + """
        <style>
        .stApp { background-color: #0e1117; color: #fafafa; }
        [data-testid="stSidebar"] { background-color: #262730; color: #fafafa; }
        .stTextInput > div > div > input { color: #fafafa; background-color: #262730; }
        .stSelectbox > div > div > div { color: #fafafa; background-color: #262730; }
        .stButton > button { color: #ffffff; border-color: #4b4b4b; background-color: #262730; }
        .stButton > button:hover { border-color: #ff4b4b; color: #ff4b4b; }
        h1, h2, h3 { color: #fafafa !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        hide_bar_css
        + """
        <style>
        .stApp { background-color: #ffffff; color: #31333F; }
        [data-testid="stSidebar"] { background-color: #f0f2f6; color: #31333F; }
        .stTextInput > div > div > input { color: #31333F; background-color: #ffffff; }
        h1, h2, h3 { color: #31333F !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title("Animal2Vec Remote Analysis Pipeline")

tab1, tab2, tab3 = st.tabs(["Embedding Generation", "Clustering", "Visualization"])


# =========================
# Helpers
# =========================
def normalize_server_path(path: str) -> str:
    """Ensure absolute server paths; fix common 'home/ssingh/...' typo (missing leading /)."""
    p = str(path).strip()
    if p.startswith("home/"):
        p = "/" + p
    return p


def run_streaming_subprocess(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> int:
    st.code("Running command:\n" + " ".join(cmd), language="bash")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=cwd,
    )

    stdout_placeholder = st.empty()
    stderr_placeholder = st.empty()
    buf: List[str] = []

    assert process.stdout is not None
    while True:
        line = process.stdout.readline()
        if line == "" and process.poll() is not None:
            break
        if line:
            buf.append(line)
            stdout_placeholder.code("".join(buf)[-12000:], language="text")

    assert process.stderr is not None
    err = process.stderr.read()
    if err.strip():
        stderr_placeholder.error(err)

    return int(process.returncode or 0)


def run_tmux_job(
    session_name: str,
    inner_cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    log_path: Optional[str] = None,
) -> int:
    cmd_str = " ".join(inner_cmd)

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        cmd_str = f'{cmd_str} > "{log_path}" 2>&1'

    if cwd:
        cmd_str = f'cd "{cwd}" && {cmd_str}'

    bash_cmd = ["bash", "-lc", cmd_str]
    tmux_cmd = ["tmux", "new-session", "-d", "-s", session_name] + bash_cmd
    return run_streaming_subprocess(tmux_cmd, env=env, cwd=cwd)


def wav_bytes_from_float32(y: np.ndarray, sr: int) -> io.BytesIO:
    y = np.asarray(y, dtype=np.float32)
    y = y / (np.max(np.abs(y)) + 1e-9)

    bio = io.BytesIO()
    if _HAS_SOUNDFILE:
        sf.write(bio, y, sr, format="WAV")  # type: ignore
        bio.seek(0)
        return bio

    y16 = np.clip(y, -1.0, 1.0)
    y16 = (y16 * 32767.0).astype(np.int16)
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(y16.tobytes())
    bio.seek(0)
    return bio


_NIPS_SEGMENT_WAV_RE = re.compile(r"^(?P<stem>.+)_\d{5}s_\d{5}s(?P<ext>\.wav)$", re.IGNORECASE)
_SEGMENT_WINDOW_RE = re.compile(r"_\d{5}s_\d{5}s$")


def _candidate_wav_names(wav_file: str) -> List[str]:
    base = os.path.basename(str(wav_file)).strip()
    out: List[str] = []
    if base:
        out.append(base)

    m = _NIPS_SEGMENT_WAV_RE.match(base)
    if m:
        out.append(f"{m.group('stem')}{m.group('ext')}")

    stem = Path(base).stem if base else ""
    if stem:
        out.append(f"{stem}.wav")
        out.append(f"{stem}.WAV")

    seen = set()
    uniq: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _recording_group_id(wav: str) -> str:
    """
    Map a wav basename to a parent "recording" id. Segment exports like
    ``nips4b_birds_trainfile685_00000s_00005s.wav`` share one id ``nips4b_birds_trainfile685``.
    """
    base = os.path.basename(str(wav).strip())
    m = _NIPS_SEGMENT_WAV_RE.match(base)
    if m:
        return str(m.group("stem"))
    stem = Path(base).stem if base else ""
    return stem if stem else base


def resolve_audio_path(audio_root: str, wav_file: str) -> str:
    wav_file = str(wav_file)
    audio_root = str(audio_root)

    if os.path.isabs(wav_file) and os.path.exists(wav_file):
        return wav_file

    candidate = os.path.join(audio_root, os.path.basename(wav_file))
    if os.path.exists(candidate):
        return candidate

    name_candidates = _candidate_wav_names(wav_file)
    root_candidates = [
        audio_root,
        os.path.join(audio_root, "wav"),
        os.path.join(audio_root, "WAV"),
        os.path.join(audio_root, "audio"),
        os.path.join(audio_root, "Audio"),
        os.path.join(audio_root, "wav_files"),
        os.path.join(audio_root, "wavfiles"),
    ]

    for root in root_candidates:
        for nm in name_candidates:
            p = os.path.join(root, nm)
            if os.path.exists(p):
                return p

    try:
        audio_root_p = Path(audio_root)
        if audio_root_p.exists():
            for nm in name_candidates:
                matches = list(audio_root_p.rglob(nm))
                if matches:
                    return str(matches[0])
    except Exception:
        pass

    fallback_name = name_candidates[0] if name_candidates else os.path.basename(wav_file)
    return os.path.join(audio_root, fallback_name)


def safe_spectrogram(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float32)
    n = int(y.size)
    if n < 128:
        raise ValueError("Too few samples for spectrogram.")

    max_fft = min(1024, n)
    pow2 = 2 ** int(np.floor(np.log2(max_fft)))
    n_fft = int(max(128, pow2))
    hop = int(max(32, n_fft // 4))

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop, center=True)
    S = np.abs(D) + 1e-9
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    dur = n / float(sr)
    t_axis = np.linspace(0.0, max(dur, 1e-6), S_db.shape[1])
    f_axis = np.linspace(0.0, sr / 2.0, S_db.shape[0])
    return S_db, t_axis, f_axis


def load_h5_embeddings_for_clustering(emb_dir: str) -> pd.DataFrame:
    emb_dir = emb_dir.strip()
    emb_dir_p = Path(emb_dir)
    if not emb_dir_p.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {emb_dir}")

    h5_paths = sorted(emb_dir_p.glob("*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"No .h5 files found in: {emb_dir}")

    rows: List[Dict] = []
    for h5_path in h5_paths:
        try:
            with h5py.File(h5_path, "r") as f:
                if "embedding" not in f or "time" not in f:
                    continue
                emb = np.asarray(f["embedding"][:], dtype=np.float32)
                t = np.asarray(f["time"][:], dtype=np.float32)
                if emb.ndim != 2 or t.ndim != 1 or emb.shape[0] != t.shape[0]:
                    continue

                wav_name = None
                if "filename" in f:
                    try:
                        wav_name = f["filename"][()]
                        if isinstance(wav_name, (bytes, np.bytes_)):
                            wav_name = wav_name.decode("utf-8", errors="ignore")
                        wav_name = str(wav_name)
                    except Exception:
                        wav_name = None

                if not wav_name:
                    wav_name = h5_path.name.split("_embeddings_")[0]

                if len(t) >= 2:
                    dt = float(np.median(np.diff(t)))
                    dt = dt if np.isfinite(dt) and dt > 0 else 0.0
                else:
                    dt = 0.0
                dt = max(dt, 1e-3)

                for i in range(emb.shape[0]):
                    ti = float(t[i])
                    rows.append(
                        {
                            "wav": os.path.basename(wav_name),
                            "start_s": ti,
                            "end_s": ti + dt,
                            "time_s": ti,
                            "h5_path": str(h5_path),
                            "h5_name": h5_path.name,
                            "frame_idx": int(i),
                            "embedding_vec": emb[i],
                        }
                    )
        except Exception:
            continue

    if not rows:
        raise ValueError(
            "No valid (embedding,time) pairs found in .h5 files. "
            "Expected datasets: 'embedding' (T,D) and 'time' (T,)."
        )

    return pd.DataFrame(rows)


def recording_stem_from_segment_h5_name(h5_name: str) -> Optional[str]:
    """
    Example:
      nips4b_birds_trainfile686_00000s_00005s.wav_embeddings_Wav2VecCcasFinetune_NIPS4B.h5
      -> nips4b_birds_trainfile686
    """
    name = Path(h5_name).name
    if ".wav_embeddings_" not in name:
        return None

    left = name.split(".wav_embeddings_", 1)[0]
    left = Path(left).stem  # remove trailing .wav

    if not _SEGMENT_WINDOW_RE.search(left):
        return None

    return _SEGMENT_WINDOW_RE.sub("", left)


def wav_stem_from_h5_name(h5_name: str) -> str:
    """Derive the .wav stem from an embedding .h5 filename."""
    name = Path(h5_name).name
    if ".wav_embeddings_" in name:
        left = name.split(".wav_embeddings_", 1)[0]
    else:
        left = name.split("_embeddings_", 1)[0]
    return Path(left).stem


def resolve_csv_path(
    wav_csv_dir: Path,
    *,
    h5_name: str,
    wav_name: str,
    segment_h5_use_recording_csv: bool,
) -> Tuple[Optional[Path], str, str]:
    """
    Find the annotation CSV for one embedding .h5 file.

    Tries, in order:
      1. Per-file CSV — ``{wav_stem}.csv`` (full recording or segment clip)
      2. Recording CSV — ``{recording_stem}.csv`` (segment H5 → parent recording)
    """
    stem_candidates: List[str] = []
    h5_stem = wav_stem_from_h5_name(h5_name)
    if h5_stem:
        stem_candidates.append(h5_stem)
    if wav_name:
        wav_stem = Path(wav_name).stem
        if wav_stem and wav_stem not in stem_candidates:
            stem_candidates.append(wav_stem)

    for stem in stem_candidates:
        segment_csv = wav_csv_dir / f"{stem}.csv"
        if segment_csv.exists():
            return segment_csv, stem, "file_csv"

    if segment_h5_use_recording_csv or ".wav_embeddings_" in h5_name:
        rec_stem = recording_stem_from_segment_h5_name(h5_name)
        if rec_stem is not None:
            recording_csv = wav_csv_dir / f"{rec_stem}.csv"
            if recording_csv.exists():
                return recording_csv, rec_stem, "recording_csv"

    return None, (stem_candidates[0] if stem_candidates else ""), "none"


def read_label_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(csv_path, header=None)
    except Exception:
        try:
            df = pd.read_csv(csv_path, header=None, sep=r"\s+")
        except Exception:
            return None
    if df.shape[0] == 0 or df.shape[1] < 3:
        return None
    df = df.iloc[:, :3].copy()
    df.columns = ["start", "duration", "cls"]
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df["cls"] = df["cls"].astype(str).str.strip()
    df = df.dropna(subset=["start", "duration", "cls"])
    df = df[df["duration"] > 0]
    return df if len(df) > 0 else None


def filter_annotation_segments(
    df: pd.DataFrame, *, only_call: bool, max_segment_duration: Optional[float]
) -> pd.DataFrame:
    out = df.copy()
    if only_call:
        c = out["cls"].str.lower()
        out = out[c.str.contains("call", na=False) & ~c.str.contains("unknown", na=False)]
    if max_segment_duration is not None:
        out = out[out["duration"] <= float(max_segment_duration)]
    return out


def assign_frames_to_segments(
    time_sec: np.ndarray,
    segments: pd.DataFrame,
    recording_id: str,
) -> pd.DataFrame:
    n = len(time_sec)
    out = pd.DataFrame(
        {
            "time_s": time_sec.astype(np.float32),
            "class_name": ["Unknown"] * n,
            "segment_start_s": np.full(n, np.nan, dtype=np.float32),
            "segment_end_s": np.full(n, np.nan, dtype=np.float32),
            "segment_uid": [None] * n,
        }
    )

    for seg_idx, row in segments.reset_index(drop=True).iterrows():
        start_s = float(row["start"])
        end_s = float(row["start"]) + float(row["duration"])
        cls = str(row["cls"]).strip()

        left = int(np.searchsorted(time_sec, start_s, side="left"))
        right = int(np.searchsorted(time_sec, end_s, side="right"))

        left = max(0, min(left, n))
        right = max(left, min(right, n))
        if right <= left:
            continue

        seg_uid = f"{recording_id}__{start_s:.5f}__{end_s:.5f}__{seg_idx}"

        current = out.iloc[left:right]["class_name"].to_numpy()
        mask = current == "Unknown"
        idx = np.arange(left, right)[mask]

        if len(idx) == 0:
            continue

        out.loc[idx, "class_name"] = cls
        out.loc[idx, "segment_start_s"] = start_s
        out.loc[idx, "segment_end_s"] = end_s
        out.loc[idx, "segment_uid"] = seg_uid

    return out


def filter_by_min_label_count(df: pd.DataFrame, min_label_count: int) -> pd.DataFrame:
    vals, counts = np.unique(df["class_name"].astype(str).to_numpy(), return_counts=True)
    keep = set(vals[counts >= int(min_label_count)].tolist())
    out = df[df["class_name"].astype(str).isin(keep)].copy()
    return out


def subsample_max_per_label(df: pd.DataFrame, max_samples_per_label: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(random_state))
    keep_idx: List[int] = []
    y = df["class_name"].astype(str).to_numpy()
    for lab in np.unique(y):
        idx = np.where(y == lab)[0]
        if len(idx) <= int(max_samples_per_label):
            keep_idx.extend(idx.tolist())
        else:
            keep_idx.extend(rng.choice(idx, size=int(max_samples_per_label), replace=False).tolist())
    keep_idx = np.array(sorted(keep_idx), dtype=np.int64)
    return df.iloc[keep_idx].copy()


def maybe_add_labels_from_csv(
    df_frames: pd.DataFrame,
    *,
    wav_csv_dir: str,
    segment_h5_use_recording_csv: bool,
    ignore_test_files: bool,
    only_call: bool,
    max_segment_duration: Optional[float],
) -> pd.DataFrame:
    wav_csv_dir_p = Path(wav_csv_dir)
    if not wav_csv_dir_p.exists():
        raise FileNotFoundError(f"WAV/CSV directory not found: {wav_csv_dir}")

    out_parts: List[pd.DataFrame] = []
    debug_rows: List[Dict] = []

    for _, g in df_frames.groupby("h5_path", sort=False):
        g = g.sort_values("frame_idx")
        h5_name = str(g["h5_name"].iloc[0])
        wav_name = str(g["wav"].iloc[0]).strip()

        csv_path, recording_id, match_mode = resolve_csv_path(
            wav_csv_dir_p,
            h5_name=h5_name,
            wav_name=wav_name,
            segment_h5_use_recording_csv=segment_h5_use_recording_csv,
        )

        if ignore_test_files and (
            "testfile" in recording_id.lower() or "testfile" in wav_name.lower()
        ):
            debug_rows.append(
                {
                    "h5_name": h5_name,
                    "mode": match_mode,
                    "reason": "skipped test file",
                    "csv_path": str(csv_path) if csv_path is not None else None,
                }
            )
            continue

        if csv_path is None or not csv_path.exists():
            debug_rows.append(
                {
                    "h5_name": h5_name,
                    "mode": match_mode,
                    "reason": "csv not found",
                    "csv_path": str(wav_csv_dir_p / f"{recording_id}.csv") if recording_id else None,
                }
            )
            continue

        df_seg = read_label_csv(csv_path)
        if df_seg is None:
            debug_rows.append(
                {
                    "h5_name": h5_name,
                    "mode": match_mode,
                    "reason": "csv exists but could not be parsed",
                    "csv_path": str(csv_path),
                }
            )
            continue

        before = len(df_seg)
        df_seg = filter_annotation_segments(
            df_seg,
            only_call=only_call,
            max_segment_duration=max_segment_duration,
        )
        if len(df_seg) == 0:
            debug_rows.append(
                {
                    "h5_name": h5_name,
                    "mode": match_mode,
                    "reason": f"all csv rows removed by filters (before={before})",
                    "csv_path": str(csv_path),
                }
            )
            continue

        t = g["time_s"].to_numpy(dtype=np.float32)
        frame_seg_df = assign_frames_to_segments(
            t,
            df_seg,
            recording_id=recording_id,
        )

        g2 = g.copy().reset_index(drop=True)
        g2["class_name"] = frame_seg_df["class_name"].astype(str)
        g2["segment_start_s"] = frame_seg_df["segment_start_s"]
        g2["segment_end_s"] = frame_seg_df["segment_end_s"]
        g2["segment_uid"] = frame_seg_df["segment_uid"]

        labeled_count = int((g2["class_name"].astype(str).str.lower() != "unknown").sum())
        if labeled_count == 0:
            debug_rows.append(
                {
                    "h5_name": h5_name,
                    "mode": match_mode,
                    "reason": "csv matched but no frame times fell into any segment",
                    "csv_path": str(csv_path),
                }
            )
            continue

        debug_rows.append(
            {
                "h5_name": h5_name,
                "mode": match_mode,
                "reason": f"success: labeled {labeled_count} frames",
                "csv_path": str(csv_path),
            }
        )
        out_parts.append(g2)

    debug_df = pd.DataFrame(debug_rows)
    if len(debug_df) > 0:
        st.write("CSV labeling debug summary")
        st.dataframe(debug_df, use_container_width=True)

    if not out_parts:
        raise ValueError("No frames could be labeled from CSVs. Check the debug table above.")

    return pd.concat(out_parts, axis=0, ignore_index=True)


def cluster_embeddings_df(
    df_frames: pd.DataFrame,
    algorithm: str,
    random_state: int,
    k: int = 60,
    min_cluster_size: int = 10,
    min_samples: Optional[int] = None,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
) -> pd.DataFrame:
    X = np.stack(df_frames["embedding_vec"].to_numpy())

    if algorithm.lower() in {"kmeans", "k-means", "k_means"}:
        if k <= 1 or k > X.shape[0]:
            raise ValueError(f"Invalid k={k} for n_frames={X.shape[0]}")
        km = MiniBatchKMeans(
            n_clusters=int(k),
            init="k-means++",
            random_state=int(random_state),
            n_init=10,
            batch_size=128,
        )
        labels = km.fit_predict(X).astype(int)
        extra = {"kmeans_inertia": float(km.inertia_)}
    elif algorithm.lower() in {"hdbscan", "hdb"}:
        if not _HAS_HDBSCAN or hdbscan is None:
            raise ImportError(
                "HDBSCAN is not installed in this environment. "
                "Install it with `pip install hdbscan`."
            )
        ms = None if (min_samples is None or int(min_samples) <= 0) else int(min_samples)
        cl = hdbscan.HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=ms,
            metric=str(metric),
            cluster_selection_method=str(cluster_selection_method),
        )
        labels = cl.fit_predict(X).astype(int)
        extra = {"hdbscan_noise_frac": float(np.mean(labels == -1))}
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    out = df_frames.copy()
    out["cluster_id"] = labels
    if "class_name" not in out.columns:
        out["class_name"] = "Unknown"

    out.attrs["cluster_metrics"] = extra

    cols = [
        "cluster_id",
        "wav",
        "start_s",
        "end_s",
        "class_name",
        "h5_path",
        "frame_idx",
        "time_s",
        "embedding_vec",
    ]
    for extra_col in ["segment_start_s", "segment_end_s", "segment_uid"]:
        if extra_col in out.columns:
            cols.append(extra_col)

    return out[[c for c in cols if c in out.columns]]


def summarize_clustering(
    df_results: pd.DataFrame,
    *,
    X_for_silhouette: Optional[np.ndarray] = None,
    max_silhouette_n: int = 5000,
    random_state: int = 42,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    labels = df_results["cluster_id"].to_numpy()
    out["n_points"] = float(len(labels))
    out["n_clusters_including_noise"] = float(len(np.unique(labels)))
    out["n_noise"] = float(np.sum(labels == -1))

    if X_for_silhouette is not None and len(labels) >= 3:
        keep = labels != -1
        Xk = X_for_silhouette[keep]
        yk = labels[keep]
        if len(np.unique(yk)) >= 2 and Xk.shape[0] >= 10:
            if Xk.shape[0] > max_silhouette_n:
                rng = np.random.default_rng(int(random_state))
                idx = rng.choice(Xk.shape[0], size=max_silhouette_n, replace=False)
                Xk, yk = Xk[idx], yk[idx]
            out["silhouette"] = float(silhouette_score(Xk, yk))

    if "class_name" in df_results.columns:
        y_true = df_results["class_name"].astype(str)
        if not bool((y_true.str.lower() == "unknown").all()):
            yt = y_true.to_numpy()
            out["ri_vs_class_name"] = float(rand_score(yt, labels))
            out["nmi_vs_class_name"] = float(normalized_mutual_info_score(yt, labels))
            out["ami_vs_class_name"] = float(adjusted_mutual_info_score(yt, labels))
            out["ari_vs_class_name"] = float(adjusted_rand_score(yt, labels))

    return out


def _format_kmeans_params(*, k: int, random_state: int) -> str:
    return (
        f"k={k}, random_state={random_state}, init=k-means++, "
        f"n_init=10, batch_size=128 (MiniBatchKMeans)"
    )


def _format_hdbscan_params(
    *,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    cluster_selection_method: str,
) -> str:
    ms = "None" if int(min_samples) <= 0 else str(int(min_samples))
    return (
        f"min_cluster_size={min_cluster_size}, min_samples={ms}, "
        f"metric={metric}, cluster_selection_method={cluster_selection_method}"
    )


def build_professor_summary_table(
    *,
    algorithm: str,
    diag: Dict[str, float],
    df_results: pd.DataFrame,
    random_state: int,
    k_value: Optional[int] = None,
    min_cluster_size: Optional[int] = None,
    min_samples: Optional[int] = None,
    metric: Optional[str] = None,
    cluster_selection_method: Optional[str] = None,
    use_csv_labels: bool = False,
    label_filters: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Two-column table (Field / Value) for pasting into a spreadsheet."""
    algo = str(algorithm).strip()
    is_kmeans = algo.lower().replace("-", "") == "kmeans"

    kmeans_str = "—"
    hdbscan_str = "—"
    if is_kmeans and k_value is not None:
        kmeans_str = _format_kmeans_params(k=int(k_value), random_state=int(random_state))
        params_summary = kmeans_str
    elif min_cluster_size is not None:
        hdbscan_str = _format_hdbscan_params(
            min_cluster_size=int(min_cluster_size),
            min_samples=int(min_samples or 0),
            metric=str(metric or "euclidean"),
            cluster_selection_method=str(cluster_selection_method or "eom"),
        )
        params_summary = hdbscan_str
    else:
        params_summary = "—"

    n_examples = int(diag.get("n_points", len(df_results)))
    n_clusters = int(diag.get("n_clusters_including_noise", 0))

    if "class_name" in df_results.columns:
        gt = df_results["class_name"].astype(str)
        gt_labeled = gt[gt.str.lower() != "unknown"]
        n_categories: object = int(gt_labeled.nunique()) if len(gt_labeled) > 0 else "—"
    else:
        n_categories = "—"

    def _metric_val(key: str) -> str:
        val = diag.get(key)
        if val is None:
            return "—"
        return f"{float(val):.6f}"

    rows: List[Tuple[str, str]] = [
        ("Algorithm", algo),
        ("Parameters", params_summary),
        ("k-means", kmeans_str),
        ("hdbscan", hdbscan_str),
        ("", ""),
        ("Number of examples clustered", str(n_examples)),
        ("Number of categories (ground truth)", str(n_categories)),
        ("Number of clusters (predicted)", str(n_clusters)),
        ("Noise points (HDBSCAN -1)", str(int(diag.get("n_noise", 0)))),
        ("", ""),
        ("Rand index (RI)", _metric_val("ri_vs_class_name")),
        ("Adjusted Rand index (ARI)", _metric_val("ari_vs_class_name")),
        ("Normalized mutual information (NMI)", _metric_val("nmi_vs_class_name")),
        ("Adjusted mutual information (AMI)", _metric_val("ami_vs_class_name")),
        ("Silhouette", _metric_val("silhouette")),
    ]

    if use_csv_labels and label_filters:
        rows.extend([("", ""), ("Label filters", "")])
        for k, v in label_filters.items():
            rows.append((k, v))

    return pd.DataFrame(rows, columns=["Field", "Value"])


def _professor_summary_tsv(summary_df: pd.DataFrame) -> str:
    lines = ["Field\tValue"]
    for _, row in summary_df.iterrows():
        field = str(row["Field"]).replace("\t", " ")
        value = str(row["Value"]).replace("\t", " ")
        lines.append(f"{field}\t{value}")
    return "\n".join(lines)


def _dataframe_has_usable_embeddings(df: pd.DataFrame) -> bool:
    if "embedding_vec" not in df.columns:
        return False
    s = df["embedding_vec"].dropna()
    if len(s) == 0:
        return False
    v0 = s.iloc[0]
    return isinstance(v0, (np.ndarray, list, tuple))


def _cluster_summary_table(df: pd.DataFrame, *, include_noise: bool) -> pd.DataFrame:
    d = df if include_noise else df.loc[df["cluster_id"].to_numpy() != -1].copy()
    rows: List[Dict] = []
    for cid, sub in d.groupby("cluster_id", sort=True):
        n = int(len(sub))
        nwav = int(sub["wav"].nunique())
        rec_id = sub["wav"].map(_recording_group_id)
        nrec = int(rec_id.nunique())
        vc_w = sub["wav"].astype(str).value_counts()
        dom_wav = str(vc_w.index[0]) if n else "—"
        wav_pur = float(vc_w.iloc[0] / n) if n else 0.0
        vc_r = rec_id.value_counts()
        dom_rec = str(vc_r.index[0]) if n else "—"
        rec_pur = float(vc_r.iloc[0] / n) if n else 0.0
        if "class_name" in sub.columns:
            vc = sub["class_name"].astype(str).value_counts()
            dom = str(vc.index[0])
            pur = float(vc.iloc[0] / n) if n else 0.0
        else:
            dom, pur = "—", float("nan")
        rows.append(
            {
                "cluster_id": int(cid),
                "n_frames": n,
                "n_wav": nwav,
                "n_recordings": nrec,
                "dominant_wav": dom_wav,
                "wav_purity": wav_pur,
                "dominant_recording": dom_rec,
                "recording_purity": rec_pur,
                "dominant_class": dom,
                "class_purity": pur,
            }
        )
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values("n_frames", ascending=False).reset_index(drop=True)


def _cluster_centroid_matrix(
    df: pd.DataFrame, *, include_noise: bool
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """One pass: return (centroids (K,D), counts (K,), cluster ids sorted)."""
    work = df if include_noise else df.loc[df["cluster_id"].to_numpy() != -1].copy()
    sums: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    dim: Optional[int] = None
    for t in work.itertuples(index=False):
        cid = int(t.cluster_id)
        v = np.asarray(t.embedding_vec, dtype=np.float64)
        if dim is None:
            dim = int(v.shape[0])
        if cid not in sums:
            sums[cid] = np.zeros(dim, dtype=np.float64)
            counts[cid] = 0
        sums[cid] += v
        counts[cid] += 1
    ids = sorted(sums.keys())
    if not ids:
        return np.zeros((0, 0), dtype=np.float64), np.zeros((0,), dtype=np.int64), []
    C = np.stack([sums[i] / float(counts[i]) for i in ids])
    cnt = np.array([counts[i] for i in ids], dtype=np.int64)
    return C, cnt, ids


def _small_pie_data_uri(labels: List[str], counts: List[int]) -> Optional[str]:
    """
    Return a small PNG pie as a data URI for Streamlit ImageColumn.
    Falls back to None if matplotlib isn't available.
    """
    if not _HAS_MPL or plt is None:
        return None
    if not labels or not counts or int(sum(counts)) <= 0:
        return None

    try:
        fig = plt.figure(figsize=(1.0, 1.0), dpi=140)
        ax = fig.add_subplot(111)
        ax.pie(
            counts,
            labels=None,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.85, edgecolor="white", linewidth=0.4),
        )
        ax.set_aspect("equal")
        ax.set_axis_off()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.0, transparent=True)
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None


def _cluster_pie_for_distribution(
    df_all: pd.DataFrame,
    cluster_id: int,
    *,
    basis: str = "auto",
    top_k: int = 5,
) -> Tuple[Optional[str], str]:
    """
    Create a small pie chart for one cluster and return (data_uri, basis_label).
    basis:
      - "auto": prefer `class_name` (if any non-Unknown), else `wav`
      - "class_name": `class_name` distribution (drops Unknown if any non-Unknown else uses all)
      - "wav": exact wav basename distribution
      - "recording_id": parent recording id distribution (segment windows merged)
      - "h5_path": source h5 file distribution
    """
    sub = df_all[df_all["cluster_id"] == int(cluster_id)].copy()
    if len(sub) == 0:
        return None, "—"

    s: Optional[pd.Series] = None
    basis_label = str(basis)

    if basis == "auto":
        if "class_name" in sub.columns:
            cls = sub["class_name"].astype(str)
            non_unknown = cls.str.lower().to_numpy() != "unknown"
            if bool(non_unknown.any()):
                s = cls[non_unknown]
                basis_label = "class_name"
        if s is None:
            s = sub["wav"].astype(str)
            basis_label = "wav"
    elif basis == "class_name":
        if "class_name" not in sub.columns:
            return None, "class_name (missing)"
        cls = sub["class_name"].astype(str)
        non_unknown = cls.str.lower().to_numpy() != "unknown"
        s = cls[non_unknown] if bool(non_unknown.any()) else cls
        basis_label = "class_name"
    elif basis == "wav":
        s = sub["wav"].astype(str)
        basis_label = "wav"
    elif basis == "recording_id":
        s = sub["wav"].map(_recording_group_id).astype(str)
        basis_label = "recording_id"
    elif basis == "h5_path":
        if "h5_path" not in sub.columns:
            return None, "h5_path (missing)"
        s = sub["h5_path"].astype(str)
        basis_label = "h5_path"
    else:
        return None, f"{basis} (unknown)"

    vc = s.value_counts()
    if len(vc) == 0:
        return None, basis_label

    head = vc.head(int(max(1, top_k)))
    other = int(vc.iloc[int(max(1, top_k)) :].sum()) if len(vc) > int(max(1, top_k)) else 0
    labels = head.index.astype(str).tolist()
    counts = head.to_numpy(dtype=int).tolist()
    if other > 0:
        labels.append("Other")
        counts.append(int(other))

    return _small_pie_data_uri(labels, counts), basis_label


# =========================
# Tab 1: Embedding Generation
# =========================
with tab1:
    st.header("Generate Embeddings")
    st.caption("Runs animal2vec_inference.py on the server to produce .h5 embeddings and/or CSV outputs.")

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            inference_script = st.text_input(
                "Path to inference script",
                value=str(_REPO_ROOT / "animal2vec_inference.py"),
            )
            wav_dir = st.text_input("WAV Directory", value="/home/ssingh/data/Datasets/Nips")
        with col2:
            checkpoint = st.text_input(
                "Model Checkpoint",
                value="/home/ssingh/data/Results/animal2vec_runs/checkpoints/checkpoint_last.pt",
            )
            output_dir = st.text_input(
                "Output Directory",
                value="/home/ssingh/data/Results/animal2vec_runs/embeddings/",
            )

        device = st.selectbox("Device", ["cuda", "cpu"], index=0)
        sample_rate = st.number_input(
            "Sample rate (Hz)",
            min_value=8000,
            max_value=48000,
            value=DEFAULT_SAMPLE_RATE,
            step=1000,
            help="Must match the checkpoint (24000 for Xeno pretrain and vocal-contrastive finetunes).",
        )

        write_embeddings = st.checkbox("Write embeddings (.h5)", value=True)
        write_other_predictions = st.checkbox("Write other predictions (CSV)", value=False)
        write_non_focal = st.checkbox("Write non-focal predictions (CSV)", value=False)
        additional_args = st.text_input(
            "Additional Arguments (optional)",
            value='--unique-values "[]" --overwrite-previous-preds True --average_top_k_layers 12',
            help="Extra flags for animal2vec_inference.py. Sample rate is set by the field above.",
        )

        st.markdown("---")
        st.subheader("Background execution (tmux)")
        use_tmux = st.checkbox("Run in tmux (recommended for long jobs)", value=True)
        tmux_session = st.text_input("tmux session name", value="a2v_embed")
        log_dir = st.text_input("Log directory", value=os.path.join(output_dir, "logs"))
        write_log = st.checkbox("Write logs to file", value=True)

        if st.button("Generate", type="primary"):
            wav_dir = normalize_server_path(wav_dir)
            checkpoint = normalize_server_path(checkpoint)
            output_dir = normalize_server_path(output_dir)
            log_dir = normalize_server_path(log_dir)
            if not output_dir.startswith("/"):
                st.error("Output Directory must be an absolute path (start with /home/ssingh/...).")
                st.stop()

            os.makedirs(output_dir, exist_ok=True)

            inner_cmd = [
                "python",
                inference_script,
                "--path",
                wav_dir,
                "--model-path",
                checkpoint,
                "--out-path",
                output_dir,
                "--device",
                device,
                "--write-embeddings",
                "True" if write_embeddings else "False",
                "--write-other-predictions",
                "True" if write_other_predictions else "False",
                "--write-non-focal",
                "True" if write_non_focal else "False",
            ]

            if additional_args.strip():
                inner_cmd.extend(additional_args.split())

            # Always pass sample rate last so it overrides any duplicate in additional_args.
            inner_cmd.extend(["--sample-rate", str(int(sample_rate))])

            env = os.environ.copy()
            env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

            if use_tmux:
                ts = int(pd.Timestamp.now().timestamp())
                session_name = f"{tmux_session}_{ts}"

                log_path = None
                if write_log:
                    log_path = os.path.join(log_dir, f"{session_name}.log")

                rc = run_tmux_job(
                    session_name=session_name,
                    inner_cmd=inner_cmd,
                    env=env,
                    log_path=log_path,
                )

                if rc == 0:
                    st.success(f"Started tmux session: {session_name}")
                    st.code(f"tmux attach -t {session_name}", language="bash")
                    if log_path:
                        st.caption("Log file:")
                        st.code(log_path, language="text")
                        st.caption("To follow logs:")
                        st.code(f'tail -f "{log_path}"', language="bash")
                else:
                    st.error(f"Failed to start tmux job (exit code {rc}).")
            else:
                rc = run_streaming_subprocess(inner_cmd, env=env, cwd=str(_REPO_ROOT))
                if rc == 0:
                    st.success("Done.")
                else:
                    st.error(f"Failed (exit code {rc}).")


# =========================
# Tab 2: Clustering
# =========================
with tab2:
    st.header("Cluster Embeddings")
    st.caption(
        "Loads frame-wise embeddings from .h5 files (datasets: embedding + time) and clusters them "
        "directly in the app (K-Means or HDBSCAN). Results are fed into Tab 3 visualization."
    )

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            emb_input_dir = st.text_input(
                "Embeddings Directory",
                value="/home/ssingh/data/Results/animal2vec_runs/embeddings",
                key="cluster_emb_dir",
            )
            wav_csv_input_dir = st.text_input(
                "WAV/CSV Directory (labels + audio base)",
                value="/home/ssingh/data/Datasets/Nips",
                key="cluster_wav_dir",
            )
        with col2:
            algorithm = st.selectbox("Clustering Algorithm", ["HDBSCAN", "K-Means"], index=0)
            random_state = st.number_input("Random seed", min_value=0, value=42, step=1)
            if algorithm == "K-Means":
                k_value = st.number_input("Number of clusters (k)", min_value=2, value=86, step=1)
            else:
                min_cluster_size = st.number_input("Min cluster size", min_value=2, value=10, step=1)
                min_samples = st.number_input(
                    "Min samples (0 = None/default)",
                    min_value=0,
                    value=0,
                    step=1,
                    help="0 means 'None' (HDBSCAN default behavior). Higher values make clustering more conservative (more noise, fewer clusters).",
                )
                cluster_selection_method = st.selectbox(
                    "Cluster selection method",
                    ["eom", "leaf"],
                    index=0,
                    help="eom = fewer, stabler clusters; leaf = more clusters.",
                )
                metric = st.selectbox(
                    "Distance metric",
                    ["cosine", "euclidean"],
                    index=0,
                    help="If your script normalizes embeddings, cosine is a common choice.",
                )

        with st.expander("Optional label filters (from CSV annotations)", expanded=False):
            st.caption(
                "These match `scripts/frame_level_clustering.py`. Enable only if you have matching CSVs. "
                "For segment-export H5 names, the app now auto-detects the recording CSV mapping."
            )
            use_csv_labels = st.checkbox("Use CSV labels (compute ARI/NMI/AMI)", value=False)
            segment_h5_use_recording_csv = st.checkbox(
                "Segment H5 → recording CSV mapping (NIPS segment exports)",
                value=True,
                help="Recommended for names like *_00000s_00005s.wav_embeddings_*.h5. The app also auto-detects this format.",
            )
            ignore_test_files = st.checkbox("Ignore test files", value=True)
            only_call = st.checkbox("--only-call (keep 'call' segments, drop 'unknown')", value=False)
            max_segment_duration = st.number_input(
                "--max-segment-duration (s, 0 = off)",
                min_value=0.0,
                value=0.0,
                step=0.5,
            )
            filter_noise = st.checkbox("--filter-noise (drop Unknown frames)", value=True)
            min_label_count = st.number_input(
                "--min-label-count (0 = off)",
                min_value=0,
                value=0,
                step=1,
            )
            max_samples_per_label = st.number_input(
                "--max-samples-per-label (0 = off)",
                min_value=0,
                value=0,
                step=100,
            )

        if st.button("Run Clustering", type="primary"):
            emb_input_dir = normalize_server_path(emb_input_dir.strip())
            wav_csv_input_dir = normalize_server_path(wav_csv_input_dir.strip())
            try:
                with st.spinner("Loading .h5 embeddings..."):
                    df_frames = load_h5_embeddings_for_clustering(emb_input_dir)
                st.success(f"Loaded {len(df_frames):,} frames from {df_frames['h5_path'].nunique()} .h5 files.")
            except Exception as e:
                st.error(f"Failed to load embeddings: {e}")
                st.stop()

            if use_csv_labels:
                try:
                    with st.spinner("Loading CSV annotations + building per-frame labels..."):
                        df_frames = maybe_add_labels_from_csv(
                            df_frames,
                            wav_csv_dir=wav_csv_input_dir,
                            segment_h5_use_recording_csv=bool(segment_h5_use_recording_csv),
                            ignore_test_files=bool(ignore_test_files),
                            only_call=bool(only_call),
                            max_segment_duration=None if float(max_segment_duration) <= 0 else float(max_segment_duration),
                        )
                    st.success(
                        f"Labeled {len(df_frames):,} frames from CSVs "
                        f"({df_frames['h5_path'].nunique()} .h5 files with labels)."
                    )
                except Exception as e:
                    st.error(f"Failed to apply CSV labels: {e}")
                    st.stop()

                if filter_noise:
                    m = df_frames["class_name"].astype(str).str.lower().to_numpy() != "unknown"
                    df_frames = df_frames.loc[m].copy()
                    st.info(f"After --filter-noise: {len(df_frames):,} frames")
                if int(min_label_count) > 0:
                    before = len(df_frames)
                    df_frames = filter_by_min_label_count(df_frames, int(min_label_count))
                    st.info(f"After --min-label-count: {len(df_frames):,} frames (was {before:,})")
                if int(max_samples_per_label) > 0:
                    before = len(df_frames)
                    df_frames = subsample_max_per_label(df_frames, int(max_samples_per_label), int(random_state))
                    st.info(f"After --max-samples-per-label: {len(df_frames):,} frames (was {before:,})")

                if len(df_frames) == 0:
                    st.error("No frames left after label filters.")
                    st.stop()

            try:
                with st.spinner(f"Clustering with {algorithm}..."):
                    X_all = np.stack(df_frames["embedding_vec"].to_numpy())
                    if algorithm == "K-Means":
                        df_results = cluster_embeddings_df(
                            df_frames,
                            algorithm="kmeans",
                            random_state=int(random_state),
                            k=int(k_value),
                        )
                    else:
                        df_results = cluster_embeddings_df(
                            df_frames,
                            algorithm="hdbscan",
                            random_state=int(random_state),
                            min_cluster_size=int(min_cluster_size),
                            min_samples=int(min_samples),
                            metric=str(metric),
                            cluster_selection_method=str(cluster_selection_method),
                        )
            except Exception as e:
                st.error(f"Clustering failed: {e}")
                st.stop()

            with st.container(border=True):
                st.subheader("Clustering diagnostics")
                diag = summarize_clustering(
                    df_results,
                    X_for_silhouette=X_all,
                    random_state=int(random_state),
                )
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Points", f"{int(diag.get('n_points', 0)):,.0f}")
                with c2:
                    n_cl = int(diag.get("n_clusters_including_noise", 0))
                    st.metric("Unique labels", f"{n_cl}")
                with c3:
                    st.metric("Noise points (-1)", f"{int(diag.get('n_noise', 0)):,.0f}")
                with c4:
                    sil = diag.get("silhouette", None)
                    st.metric("Silhouette (subsample)", "—" if sil is None else f"{sil:.4f}")

                if "nmi_vs_class_name" in diag:
                    st.caption("Frame-level scores vs `class_name` ground truth.")
                    st.write(
                        {
                            "RI_vs_class_name": round(float(diag.get("ri_vs_class_name", 0.0)), 6),
                            "ARI_vs_class_name": round(float(diag.get("ari_vs_class_name", 0.0)), 6),
                            "NMI_vs_class_name": round(float(diag["nmi_vs_class_name"]), 6),
                            "AMI_vs_class_name": round(float(diag["ami_vs_class_name"]), 6),
                        }
                    )
                else:
                    st.caption(
                        "ARI/NMI vs ground-truth requires per-frame labels (e.g. from CSV segments). "
                        "Current run clustered embeddings only."
                    )

            algo_tag = "hdbscan" if algorithm == "HDBSCAN" else "kmeans"

            label_filter_notes: Dict[str, str] = {}
            if use_csv_labels:
                label_filter_notes = {
                    "use_csv_labels": "yes",
                    "filter_noise": "yes" if filter_noise else "no",
                    "ignore_test_files": "yes" if ignore_test_files else "no",
                    "only_call": "yes" if only_call else "no",
                    "max_segment_duration_s": str(max_segment_duration),
                    "min_label_count": str(min_label_count),
                    "max_samples_per_label": str(max_samples_per_label),
                }

            summary_df = build_professor_summary_table(
                algorithm=algorithm,
                diag=diag,
                df_results=df_results,
                random_state=int(random_state),
                k_value=int(k_value) if algorithm == "K-Means" else None,
                min_cluster_size=int(min_cluster_size) if algorithm == "HDBSCAN" else None,
                min_samples=int(min_samples) if algorithm == "HDBSCAN" else None,
                metric=str(metric) if algorithm == "HDBSCAN" else None,
                cluster_selection_method=str(cluster_selection_method) if algorithm == "HDBSCAN" else None,
                use_csv_labels=bool(use_csv_labels),
                label_filters=label_filter_notes if use_csv_labels else None,
            )

            with st.container(border=True):
                st.subheader("Copy for comparison table")
                st.caption(
                    "Paste into your professor's spreadsheet: select the table below, or copy from the "
                    "tab-separated box (Field in column A, Value in column B)."
                )
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                st.text_area(
                    "Tab-separated (Excel / Google Sheets paste)",
                    value=_professor_summary_tsv(summary_df),
                    height=320,
                    key=f"professor_summary_tsv_{algo_tag}_{int(random_state)}",
                )

            output_csv = f"cluster_results_{algo_tag}.csv"
            try:
                df_results.drop(columns=["embedding_vec"], errors="ignore").to_csv(output_csv, index=False)
            except Exception:
                pass

            st.session_state["cluster_results"] = df_results

            if "audio_folder" not in st.session_state:
                st.session_state["audio_folder"] = wav_csv_input_dir

            st.success(f"Loaded {len(df_results):,} clustered frames into session.")
            st.caption(f"Output CSV (server-local): {output_csv}")


# =========================
# Tab 3: Visualization
# =========================
with tab3:
    st.header("Visualize Clusters")

    if "cluster_results" not in st.session_state:
        st.info("No clustering results loaded. Run Tab 2 or load a CSV here (server path or upload).")

        default_remote_csv = "/home/ssingh/data/Results/animal2vec_runs/cluster_results_hdbscan.csv"
        remote_csv_path = st.text_input(
            "Load Cluster CSV from server path",
            value=default_remote_csv,
            key="remote_csv_path",
        )

        colA, colB = st.columns([1, 2])
        with colA:
            if st.button("Load from path"):
                if os.path.exists(remote_csv_path):
                    st.session_state["cluster_results"] = pd.read_csv(remote_csv_path)
                    st.success(f"Loaded {len(st.session_state['cluster_results'])} rows.")
                    if "audio_folder" not in st.session_state:
                        st.session_state["audio_folder"] = "/home/ssingh/data/Datasets/Nips"
                else:
                    st.error(f"File not found on server: {remote_csv_path}")

        st.markdown("---")
        uploaded_file = st.file_uploader("Upload Cluster CSV (from your local machine)", type="csv")
        if uploaded_file:
            st.session_state["cluster_results"] = pd.read_csv(uploaded_file)
            st.success("CSV uploaded and loaded.")
            if "audio_folder" not in st.session_state:
                st.session_state["audio_folder"] = "/home/ssingh/data/Datasets/Nips"

    if "cluster_results" in st.session_state:
        df = st.session_state["cluster_results"]

        required_cols = {"cluster_id", "wav", "start_s", "end_s"}
        missing = required_cols - set(df.columns)
        if missing:
            st.error(f"CSV missing required columns: {sorted(missing)}")
            st.stop()

        if "audio_folder" not in st.session_state:
            st.session_state["audio_folder"] = "/home/ssingh/data/Datasets/Nips"
        audio_folder = st.text_input("Audio Folder Path for Playback", key="audio_folder")

        with st.expander("Global cluster overview (sizes, class mix, UMAP)", expanded=True):
            with st.expander("How to read every chart (plain language)", expanded=True):
                st.markdown(
                    r"""
### Before you look at any plot

- The model **did not** use filenames or species labels to form clusters. It only grouped **similar embedding vectors**.
- Every row in your results is **one short time slice (frame)** from one **`.wav`** (`wav` column). Labels (`class_name`) are only for **checking** clusters afterward.

**Suggested order:** bar chart → file-mix dots → summary table / inspect → heatmaps → UMAP (optional) → spectrograms at the bottom.

---

### 1) Bar chart — “Top clusters by frame count”

- **X-axis:** `cluster_id` (the ID the algorithm gave each pile of frames).
- **Y-axis:** how many **frames** ended up in that cluster (taller = bigger cluster).
- **How to read:** identifies **which cluster IDs matter by size**. It does **not** say *why* they group or *which species* they are.

---

### 2) Scatter — “Wav-file diversity vs. single-file dominance”

- **Each dot = one entire cluster** (not one frame).
- **X-axis (`n_wav`):** number of **different `.wav` filenames** that contributed frames to that cluster.
- **Y-axis (`wav_purity`):** between 0 and 1. **1.0** ≈ “almost every frame in this cluster comes from the **same** `.wav` file.” **Lower** ≈ frames are **split across** more files.
- **Dot size:** bigger = more frames in that cluster (`n_frames`).
- **Color:** same information as Y (brighter ≈ more “single-file” cluster).
- **How to read:**  
  - **Left + high on the chart:** cluster is **dominated by one (or few) files** — often “this recording’s acoustic fingerprint.”  
  - **Far to the right:** **many different files** land in the same cluster — more like “embedding says these frames are similar **across** recordings.”  
- **Hover** the dot: see `cluster_id`, `dominant_wav`, counts.

---

### 3) Scatter — “Recording-level diversity (segment windows merged)”

- Same logic as (2), but **X / Y use parent recordings**: short clips like `…_00000s_00005s.wav` are **folded into one parent** id so you don’t double-count the same underlying recording.
- **How to read:** “Is this cluster mostly **one session/recording** or **spread across many**?” Use this when your data has many small segment files per recording.

---

### 4) Cluster summary table (+ progress bars)

- **One row = one cluster.**
- **`n_frames`:** size of the cluster.  
- **`n_wav` / `wav_purity`:** file mixing (see scatter (2)).  
- **`n_recordings` / `recording_purity`:** same after merging segments (see (3)).  
- **`dominant_class` / `class_purity`:** if you had CSV labels — does this cluster **agree with one species/call type** or is it **mixed** labels?  
- **Progress bars:** quick visual of purities (full bar ≈ very “pure”).

---

### 5) “Inspect one cluster” tables

- Pick a **cluster_id**; you get **counts and %** of frames per **exact `.wav`** and per **parent recording**.
- **How to read:** the **ground truth for filenames** for that cluster — who actually contributed frames.

---

### 6) Class heatmap — “Class mixture within top clusters”

- **Each row = one cluster** (only the **largest** clusters by frame count, controlled by the slider).
- **Each column = one label** (`class_name` from CSV).
- **Cell color:** **row-normalized** — for that cluster, what **fraction** of its frames have that label. **Each row sums to 1** across columns.
- **Dark vs bright (Viridis):** **brighter** = **larger share** of that cluster has that class; **darker** ≈ almost no frames with that class.
- **How to read:** “Does this cluster **line up with one biology label** or **mix labels**?” Only meaningful if labels are trustworthy.

---

### 7) Recording heatmap — “Recording mixture…”

- Same as (6), but columns are **parent recording ids** instead of species classes.
- **How to read:** “Is this cluster **anchored on a few recordings** or **spread**?” Complements the scatter (3).

---

### 8) Frame UMAP (if shown)

- **Each dot = one frame** (subsampled). **X/Y are not time** — they are a **2D squeeze** of high-dimensional embeddings for visualization only.
- **Color** (dropdown): color by `cluster_id`, `class_name`, or `wav` to see if dots **separate** or **overlap**.
- **How to read:** only for **rough** shape (“blobs”, overlap). **Messy is normal.** Don’t over-interpret exact distances.

---

### 9) Centroid UMAP (if shown)

- **Each dot = one cluster** (position = UMAP of the **mean embedding** of that cluster).
- **Size** ≈ cluster size; **color** = dominant class (if available).
- **Hover:** includes `n_wav`, `wav_purity`, `dominant_wav`.
- **How to read:** which clusters are **near each other in embedding space** (coarse), not a statistical test.

---

### 10) Spectrograms further down (per selected cluster)

- **How to read:** “**What does this cluster sound like?**” You hear/see **examples**; use together with **`wav`** and **`class_name`** on each tile.

---

### Tiny glossary

- **Frame:** one embedding vector at one time in one file.  
- **Cluster:** a group of frames the algorithm thinks are similar.  
- **Purity:** how much one label or one file “wins” inside that group (0 = perfectly mixed, 1 = all the same).
                    """
                )
            st.caption(
                "Cluster heatmaps and size bars work from any loaded results. "
                "UMAP needs `umap-learn` and in-memory `embedding_vec` from Tab 2 "
                "(CSV export omits embeddings to keep files small)."
            )
            n_clust_global = int(df["cluster_id"].nunique())
            hm_max = max(1, min(120, n_clust_global))
            hm_default = min(40, hm_max)
            g1, g2, g3 = st.columns(3)
            with g1:
                g_include_noise = st.checkbox(
                    "Include HDBSCAN noise (-1) in overview",
                    value=True,
                    key="global_include_noise",
                )
            with g2:
                heatmap_top_k = st.slider(
                    "Top clusters (by frame count) for bars / heatmap",
                    1,
                    hm_max,
                    hm_default,
                    key="global_heatmap_top_k",
                )
            with g3:
                umap_max_pts = st.slider(
                    "Max points for frame UMAP",
                    1000,
                    50000,
                    15000,
                    500,
                    key="global_umap_max_pts",
                    help="Random subsample for speed; centroids still use all frames.",
                )

            g4, g5 = st.columns(2)
            with g4:
                umap_metric = st.selectbox("UMAP metric", ["cosine", "euclidean"], index=0, key="global_umap_metric")
            with g5:
                umap_color_by = st.selectbox(
                    "Color frame UMAP by",
                    ["cluster_id", "class_name", "wav"],
                    index=0,
                    key="global_umap_color",
                )

            g6, g7 = st.columns(2)
            with g6:
                rec_hm_top = st.slider(
                    "Recording heatmap: top N parent recordings (global frame count)",
                    5,
                    60,
                    15,
                    key="global_rec_hm_top",
                    help="Segment exports share one parent id (strip _00000s_00005s-style suffix).",
                )
            with g7:
                st.caption(
                    "Lower **n_wav** / high **wav_purity** ⇒ cluster is driven by one file; "
                    "high **n_wav** ⇒ mixed files."
                )

            work_global = df if g_include_noise else df.loc[df["cluster_id"].to_numpy() != -1].copy()
            if len(work_global) == 0:
                st.warning("No rows after excluding noise (-1).")
            else:
                vc = work_global["cluster_id"].value_counts()
                vc_top = vc.head(int(heatmap_top_k))
                vc_top_df = pd.DataFrame(
                    {"cluster_id": vc_top.index.to_numpy(), "count": vc_top.to_numpy(dtype=np.int64)}
                )
                fig_sz = px.bar(
                    vc_top_df,
                    x="cluster_id",
                    y="count",
                    text="count",
                    title=f"Top {len(vc_top)} clusters by frame count",
                )
                fig_sz.update_layout(xaxis_title="cluster_id", yaxis_title="frames", xaxis_tickangle=-45)
                st.plotly_chart(fig_sz, use_container_width=True)

                summ = _cluster_summary_table(df, include_noise=g_include_noise)
                if len(summ) > 0:
                    _mix_tmpl = "plotly_dark" if st.session_state.get("dark_mode", True) else "plotly_white"
                    st.subheader("File mix per cluster (visual)")
                    st.caption(
                        "**Each point = one cluster.** **X** = how many different `.wav` files appear. "
                        "**Y** = `wav_purity` (fraction of frames from the single most common wav). "
                        "**Size** ∝ cluster size (`n_frames`). **Bright color** = high purity (one file dominates). "
                        "**Upper-left** (small x, high y): mostly one or few wavs. **Toward the right** (large x): more files mixed into the same cluster."
                    )
                    smix = summ.copy()
                    smix["marker_n"] = np.sqrt(smix["n_frames"].astype(float).clip(lower=1.0))
                    _scatter_meta_cols = [
                        "cluster_id", "n_frames", "dominant_class", "class_purity",
                        "n_wav", "wav_purity", "n_recordings", "recording_purity",
                        "dominant_wav", "dominant_recording",
                    ]
                    fig_mix = px.scatter(
                        smix,
                        x="n_wav",
                        y="wav_purity",
                        size="marker_n",
                        color="wav_purity",
                        color_continuous_scale="Plasma",
                        hover_name="cluster_id",
                        hover_data=[
                            "n_frames",
                            "dominant_wav",
                            "n_recordings",
                            "recording_purity",
                            "dominant_recording",
                        ],
                        custom_data=_scatter_meta_cols,
                        labels={
                            "n_wav": "Distinct .wav files",
                            "wav_purity": "Wav purity (1 = single file dominates)",
                        },
                        title="Wav-file diversity vs. single-file dominance",
                    )
                    fig_mix.update_layout(
                        template=_mix_tmpl,
                        height=420,
                        xaxis=dict(rangemode="tozero"),
                        yaxis=dict(range=[-0.05, 1.05]),
                        showlegend=True,
                        dragmode="select",
                    )
                    fig_mix.update_traces(marker=dict(line=dict(width=0.6, color="rgba(255,255,255,0.35)"), opacity=0.9))
                    _scatter_event = st.plotly_chart(
                        fig_mix, use_container_width=True,
                        on_select="rerun", key="scatter_sel_wav",
                    )

                    _sel_pts: list = []
                    if _scatter_event is not None:
                        _sel_obj = getattr(_scatter_event, "selection", None)
                        if _sel_obj is not None:
                            _sel_pts = getattr(_sel_obj, "points", []) or []

                    if _sel_pts:
                        _sel_rows: List[Dict] = []
                        for _pt in _sel_pts:
                            _cd = (
                                _pt.get("customdata")
                                if isinstance(_pt, dict)
                                else getattr(_pt, "customdata", None)
                            )
                            if _cd is not None and len(_cd) >= len(_scatter_meta_cols):
                                _sel_rows.append(dict(zip(_scatter_meta_cols, _cd)))
                        if _sel_rows:
                            _sel_clust_df = pd.DataFrame(_sel_rows)
                            for _ic in ("cluster_id", "n_frames", "n_wav", "n_recordings"):
                                if _ic in _sel_clust_df.columns:
                                    _sel_clust_df[_ic] = (
                                        pd.to_numeric(_sel_clust_df[_ic], errors="coerce")
                                        .fillna(0)
                                        .astype(int)
                                    )
                            for _fc in ("class_purity", "wav_purity", "recording_purity"):
                                if _fc in _sel_clust_df.columns:
                                    _sel_clust_df[_fc] = (
                                        pd.to_numeric(_sel_clust_df[_fc], errors="coerce")
                                        .astype(float)
                                        .round(3)
                                    )
                            _sel_clust_df = _sel_clust_df.drop_duplicates(subset=["cluster_id"])
                            if _HAS_MPL:
                                pies_class: List[Optional[str]] = []
                                pies_wav: List[Optional[str]] = []
                                for _cid in _sel_clust_df["cluster_id"].tolist():
                                    _uri_class, _ = _cluster_pie_for_distribution(
                                        work_global,
                                        int(_cid),
                                        basis="class_name",
                                        top_k=5,
                                    )
                                    _uri_wav, _ = _cluster_pie_for_distribution(
                                        work_global,
                                        int(_cid),
                                        basis="wav",
                                        top_k=5,
                                    )
                                    pies_class.append(_uri_class)
                                    pies_wav.append(_uri_wav)
                                _sel_clust_df["pie_class"] = pies_class
                                _sel_clust_df["pie_wav"] = pies_wav
                            st.caption(f"**{len(_sel_clust_df)}** cluster(s) selected via box / lasso.")
                            _sel_col_cfg = {
                                "wav_purity": st.column_config.ProgressColumn(
                                    "wav purity", format="%.2f", min_value=0.0, max_value=1.0,
                                ),
                                "recording_purity": st.column_config.ProgressColumn(
                                    "rec. purity", format="%.2f", min_value=0.0, max_value=1.0,
                                ),
                                "class_purity": st.column_config.ProgressColumn(
                                    "class purity", format="%.2f", min_value=0.0, max_value=1.0,
                                ),
                            }
                            if "pie_class" in _sel_clust_df.columns:
                                _sel_col_cfg["pie_class"] = st.column_config.ImageColumn(
                                    "class pie",
                                    help="Within-cluster label distribution (`class_name`).",
                                    width="small",
                                )
                            if "pie_wav" in _sel_clust_df.columns:
                                _sel_col_cfg["pie_wav"] = st.column_config.ImageColumn(
                                    "wav pie",
                                    help="Within-cluster wav-file distribution (`wav`).",
                                    width="small",
                                )
                            st.dataframe(
                                _sel_clust_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config=_sel_col_cfg,
                            )
                            _sel_clust_ids = _sel_clust_df["cluster_id"].tolist()
                            _scatter_pick = st.selectbox(
                                "Inspect one selected cluster",
                                _sel_clust_ids,
                                key="scatter_inspect_pick",
                            )
                            if _scatter_pick is not None:
                                st.session_state["_scatter_inspect_cluster"] = int(_scatter_pick)
                        else:
                            st.caption(
                                "Use box or lasso select on the scatter plot to inspect overlapping clusters."
                            )
                            st.session_state.pop("_scatter_inspect_cluster", None)
                    else:
                        st.caption(
                            "Use box or lasso select on the scatter plot to inspect overlapping clusters."
                        )
                        st.session_state.pop("_scatter_inspect_cluster", None)

                    fig_mix_r = px.scatter(
                        smix,
                        x="n_recordings",
                        y="recording_purity",
                        size="marker_n",
                        color="recording_purity",
                        color_continuous_scale="Plasma",
                        hover_name="cluster_id",
                        hover_data=[
                            "n_frames",
                            "dominant_recording",
                            "n_wav",
                            "wav_purity",
                            "dominant_wav",
                        ],
                        custom_data=_scatter_meta_cols,
                        labels={
                            "n_recordings": "Parent recordings (merged segments)",
                            "recording_purity": "Recording purity (1 = one recording dominates)",
                        },
                        title="Recording-level diversity (segment windows merged)",
                    )
                    fig_mix_r.update_layout(
                        template=_mix_tmpl,
                        height=420,
                        xaxis=dict(rangemode="tozero"),
                        yaxis=dict(range=[-0.05, 1.05]),
                        dragmode="select",
                    )
                    fig_mix_r.update_traces(marker=dict(line=dict(width=0.6, color="rgba(255,255,255,0.35)"), opacity=0.9))
                    _scatter_event_r = st.plotly_chart(
                        fig_mix_r, use_container_width=True,
                        on_select="rerun", key="scatter_sel_rec",
                    )

                    _sel_pts_r: list = []
                    if _scatter_event_r is not None:
                        _sel_obj_r = getattr(_scatter_event_r, "selection", None)
                        if _sel_obj_r is not None:
                            _sel_pts_r = getattr(_sel_obj_r, "points", []) or []

                    if _sel_pts_r:
                        _sel_rows_r: List[Dict] = []
                        for _pt_r in _sel_pts_r:
                            _cd_r = (
                                _pt_r.get("customdata")
                                if isinstance(_pt_r, dict)
                                else getattr(_pt_r, "customdata", None)
                            )
                            if _cd_r is not None and len(_cd_r) >= len(_scatter_meta_cols):
                                _sel_rows_r.append(dict(zip(_scatter_meta_cols, _cd_r)))
                        if _sel_rows_r:
                            _sel_clust_df_r = pd.DataFrame(_sel_rows_r)
                            for _ic in ("cluster_id", "n_frames", "n_wav", "n_recordings"):
                                if _ic in _sel_clust_df_r.columns:
                                    _sel_clust_df_r[_ic] = (
                                        pd.to_numeric(_sel_clust_df_r[_ic], errors="coerce")
                                        .fillna(0)
                                        .astype(int)
                                    )
                            for _fc in ("class_purity", "wav_purity", "recording_purity"):
                                if _fc in _sel_clust_df_r.columns:
                                    _sel_clust_df_r[_fc] = (
                                        pd.to_numeric(_sel_clust_df_r[_fc], errors="coerce")
                                        .astype(float)
                                        .round(3)
                                    )
                            _sel_clust_df_r = _sel_clust_df_r.drop_duplicates(subset=["cluster_id"])
                            if _HAS_MPL:
                                pies_class_r: List[Optional[str]] = []
                                pies_wav_r: List[Optional[str]] = []
                                for _cid in _sel_clust_df_r["cluster_id"].tolist():
                                    _uri_class, _ = _cluster_pie_for_distribution(
                                        work_global,
                                        int(_cid),
                                        basis="class_name",
                                        top_k=5,
                                    )
                                    _uri_wav, _ = _cluster_pie_for_distribution(
                                        work_global,
                                        int(_cid),
                                        basis="wav",
                                        top_k=5,
                                    )
                                    pies_class_r.append(_uri_class)
                                    pies_wav_r.append(_uri_wav)
                                _sel_clust_df_r["pie_class"] = pies_class_r
                                _sel_clust_df_r["pie_wav"] = pies_wav_r
                            st.caption(f"**{len(_sel_clust_df_r)}** cluster(s) selected via box / lasso.")
                            _sel_col_cfg_r = {
                                "wav_purity": st.column_config.ProgressColumn(
                                    "wav purity", format="%.2f", min_value=0.0, max_value=1.0,
                                ),
                                "recording_purity": st.column_config.ProgressColumn(
                                    "rec. purity", format="%.2f", min_value=0.0, max_value=1.0,
                                ),
                                "class_purity": st.column_config.ProgressColumn(
                                    "class purity", format="%.2f", min_value=0.0, max_value=1.0,
                                ),
                            }
                            if "pie_class" in _sel_clust_df_r.columns:
                                _sel_col_cfg_r["pie_class"] = st.column_config.ImageColumn(
                                    "class pie",
                                    help="Within-cluster label distribution (`class_name`).",
                                    width="small",
                                )
                            if "pie_wav" in _sel_clust_df_r.columns:
                                _sel_col_cfg_r["pie_wav"] = st.column_config.ImageColumn(
                                    "wav pie",
                                    help="Within-cluster wav-file distribution (`wav`).",
                                    width="small",
                                )
                            st.dataframe(
                                _sel_clust_df_r,
                                use_container_width=True,
                                hide_index=True,
                                column_config=_sel_col_cfg_r,
                            )
                            _sel_clust_ids_r = _sel_clust_df_r["cluster_id"].tolist()
                            _scatter_pick_r = st.selectbox(
                                "Inspect one selected cluster",
                                _sel_clust_ids_r,
                                key="scatter_inspect_pick_rec",
                            )
                            if _scatter_pick_r is not None:
                                st.session_state["_scatter_inspect_cluster_rec"] = int(_scatter_pick_r)
                        else:
                            st.caption(
                                "Use box or lasso select on the recording scatter to inspect overlapping clusters."
                            )
                            st.session_state.pop("_scatter_inspect_cluster_rec", None)
                    else:
                        st.caption(
                            "Use box or lasso select on the recording scatter to inspect overlapping clusters."
                        )
                        st.session_state.pop("_scatter_inspect_cluster_rec", None)

                    st.subheader("Cluster summary (sortable)")
                    summ_show = summ.head(min(300, len(summ))).copy()
                    for _col in ("wav_purity", "recording_purity", "class_purity"):
                        if _col in summ_show.columns:
                            summ_show[_col] = summ_show[_col].astype(float).round(3)
                    _col_cfg: Dict = {}
                    if "wav_purity" in summ_show.columns:
                        _col_cfg["wav_purity"] = st.column_config.ProgressColumn(
                            "wav purity",
                            help="1.0 = almost all frames from one .wav file",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        )
                    if "recording_purity" in summ_show.columns:
                        _col_cfg["recording_purity"] = st.column_config.ProgressColumn(
                            "rec. purity",
                            help="1.0 = almost all frames from one parent recording (segments merged)",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        )
                    if "class_purity" in summ_show.columns:
                        _col_cfg["class_purity"] = st.column_config.ProgressColumn(
                            "class purity",
                            help="1.0 = almost all labeled frames share one class (needs CSV labels)",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        )
                    if "n_wav" in summ_show.columns:
                        _col_cfg["n_wav"] = st.column_config.NumberColumn(
                            "n_wav",
                            help="Distinct .wav basenames in this cluster",
                            format="%d",
                        )
                    if "n_recordings" in summ_show.columns:
                        _col_cfg["n_recordings"] = st.column_config.NumberColumn(
                            "n_rec",
                            help="Distinct parent recordings (segment windows folded)",
                            format="%d",
                        )
                    _df_kw: Dict = dict(use_container_width=True, hide_index=True)
                    if _col_cfg:
                        _df_kw["column_config"] = _col_cfg
                    st.dataframe(summ_show, **_df_kw)

                with st.expander("Inspect one cluster: which `.wav` files & recordings?", expanded=True):
                    if len(summ) == 0 or len(work_global) == 0:
                        st.warning("No data to inspect.")
                    else:
                        opt_ids = summ.sort_values("n_frames", ascending=False)["cluster_id"].tolist()

                        _scatter_cid = (
                            st.session_state.get("_scatter_inspect_cluster")
                            or st.session_state.get("_scatter_inspect_cluster_rec")
                        )
                        if _scatter_cid is not None and int(_scatter_cid) in opt_ids:
                            st.session_state["global_inspect_cluster_id"] = int(_scatter_cid)

                        def _fmt_cl_opt(cid: int) -> str:
                            r = summ.loc[summ["cluster_id"] == int(cid)].iloc[0]
                            return (
                                f"{cid}  |  {int(r['n_frames'])} frames  |  "
                                f"{int(r['n_wav'])} wavs  |  {int(r['n_recordings'])} recordings  |  "
                                f"wav_purity={float(r['wav_purity']):.2f}"
                            )

                        pick_c = st.selectbox(
                            "Choose cluster",
                            opt_ids,
                            format_func=_fmt_cl_opt,
                            key="global_inspect_cluster_id",
                        )
                        sub_i = work_global[work_global["cluster_id"] == int(pick_c)].copy()
                        st.caption(
                            f"**{len(sub_i)}** frames in cluster **{pick_c}**. "
                            "Tables show where those frames came from."
                        )
                        wv = sub_i["wav"].astype(str).value_counts().reset_index()
                        wv.columns = ["wav", "frames"]
                        wv["pct_of_cluster"] = (100.0 * wv["frames"] / max(len(sub_i), 1)).round(1)
                        st.markdown("**By wav filename** (exact file)")
                        st.dataframe(wv.head(40), use_container_width=True, hide_index=True)

                        sub_i["_recording_id"] = sub_i["wav"].map(_recording_group_id)
                        rv = sub_i["_recording_id"].astype(str).value_counts().reset_index()
                        rv.columns = ["recording_id", "frames"]
                        rv["pct_of_cluster"] = (100.0 * rv["frames"] / max(len(sub_i), 1)).round(1)
                        st.markdown(
                            "**By parent recording** (segment windows merged; same id = same underlying recording)"
                        )
                        st.dataframe(rv.head(40), use_container_width=True, hide_index=True)

                df_hm = work_global[work_global["cluster_id"].isin(vc_top.index)].copy()
                has_lbl = "class_name" in df_hm.columns and bool(
                    (df_hm["class_name"].astype(str).str.lower() != "unknown").any()
                )
                if has_lbl and len(df_hm) > 0:
                    ct = pd.crosstab(df_hm["cluster_id"], df_hm["class_name"], normalize="index")
                    ct = ct.reindex(vc_top.index, fill_value=0.0)
                    z = ct.to_numpy(dtype=np.float64)
                    # Prefix y labels so Plotly never treats cluster_id as a numeric axis (avoids huge empty gaps).
                    y_labels = [f"c{i}" for i in ct.index.tolist()]
                    x_labels = [str(c) for c in ct.columns.tolist()]
                    rids = ct.index.to_numpy()
                    cols_a = ct.columns.to_numpy()
                    hovertext = [
                        [
                            f"cluster_id={rids[i]}<br>class={cols_a[j]!s}<br>P(class|cluster)={z[i, j]:.4f}"
                            for j in range(z.shape[1])
                        ]
                        for i in range(z.shape[0])
                    ]
                    _hm_template = "plotly_dark" if st.session_state.get("dark_mode", True) else "plotly_white"
                    fig_hm = go.Figure(
                        data=go.Heatmap(
                            z=z,
                            x=x_labels,
                            y=y_labels,
                            hovertext=hovertext,
                            # Low P -> dark; high P -> bright. Avoid "Blues" (0=white on dark Streamlit).
                            colorscale="Viridis",
                            zmin=0.0,
                            zmax=1.0,
                            colorbar=dict(title="P(class|cluster)", tickformat=".2f"),
                            hovertemplate="%{hovertext}<extra></extra>",
                        )
                    )
                    nrows = len(ct.index)
                    fig_hm.update_layout(
                        template=_hm_template,
                        title="Class mixture within top clusters (row-normalized)",
                        xaxis_title="class_name",
                        yaxis_title="cluster_id",
                        height=int(max(400, min(1000, 18 * nrows + 120))),
                        margin=dict(l=88, r=24, t=48, b=max(140, min(360, 6 * len(x_labels) + 80))),
                        yaxis=dict(type="category", categoryorder="array", categoryarray=y_labels, autorange="reversed"),
                        xaxis=dict(type="category", tickangle=-55, tickfont=dict(size=9)),
                    )
                    st.plotly_chart(fig_hm, use_container_width=True)
                elif len(df_hm) > 0:
                    st.info("No non-Unknown labels in selection; class heatmap skipped.")

                # Recording mixture (parent ids): same-file vs mixed clusters
                df_rec_hm = work_global[work_global["cluster_id"].isin(vc_top.index)].copy()
                if len(df_rec_hm) > 0:
                    rec_counts_all = df_rec_hm["wav"].map(_recording_group_id).value_counts()
                    top_recs = rec_counts_all.head(int(rec_hm_top)).index.tolist()
                    df_rec_hm["recording_id"] = df_rec_hm["wav"].map(_recording_group_id)
                    df_rec_hm = df_rec_hm[df_rec_hm["recording_id"].isin(top_recs)]
                    if len(df_rec_hm) > 0 and len(top_recs) > 0:
                        ct_rec = pd.crosstab(
                            df_rec_hm["cluster_id"],
                            df_rec_hm["recording_id"],
                            normalize="index",
                        )
                        ct_rec = ct_rec.reindex(vc_top.index, fill_value=0.0)
                        ct_rec = ct_rec.reindex(columns=top_recs, fill_value=0.0)
                        zr = ct_rec.to_numpy(dtype=np.float64)
                        y_labels_r = [f"c{i}" for i in ct_rec.index.tolist()]
                        x_labels_r = [str(c) for c in ct_rec.columns.tolist()]
                        rids_r = ct_rec.index.to_numpy()
                        cols_r = ct_rec.columns.to_numpy()
                        hover_r = [
                            [
                                f"cluster_id={rids_r[i]}<br>recording={cols_r[j]!s}<br>P(rec|cluster)={zr[i, j]:.4f}"
                                for j in range(zr.shape[1])
                            ]
                            for i in range(zr.shape[0])
                        ]
                        _hm_tmpl_r = "plotly_dark" if st.session_state.get("dark_mode", True) else "plotly_white"
                        fig_hm_r = go.Figure(
                            data=go.Heatmap(
                                z=zr,
                                x=x_labels_r,
                                y=y_labels_r,
                                hovertext=hover_r,
                                colorscale="Viridis",
                                zmin=0.0,
                                zmax=1.0,
                                colorbar=dict(title="P(rec|cluster)", tickformat=".2f"),
                                hovertemplate="%{hovertext}<extra></extra>",
                            )
                        )
                        nr_r = len(ct_rec.index)
                        fig_hm_r.update_layout(
                            template=_hm_tmpl_r,
                            title="Recording mixture within top clusters (row-normalized; segment windows merged)",
                            xaxis_title="parent recording id",
                            yaxis_title="cluster_id",
                            height=int(max(400, min(1000, 18 * nr_r + 120))),
                            margin=dict(
                                l=88,
                                r=24,
                                t=48,
                                b=max(140, min(400, 7 * len(x_labels_r) + 80)),
                            ),
                            yaxis=dict(
                                type="category",
                                categoryorder="array",
                                categoryarray=y_labels_r,
                                autorange="reversed",
                            ),
                            xaxis=dict(type="category", tickangle=-55, tickfont=dict(size=8)),
                        )
                        st.plotly_chart(fig_hm_r, use_container_width=True)

                if not _HAS_UMAP:
                    st.warning("Install `umap-learn` for UMAP plots: `pip install umap-learn`")
                elif not _dataframe_has_usable_embeddings(work_global):
                    st.info(
                        "Frame/centroid UMAP skipped (no `embedding_vec` column). "
                        "Re-run clustering in Tab 2 to keep embeddings in session."
                    )
                else:
                    assert umap is not None
                    n_take = min(int(umap_max_pts), len(work_global))
                    sampled = (
                        work_global.sample(n=n_take, random_state=42)
                        if len(work_global) > n_take
                        else work_global
                    )
                    Xs = np.stack(
                        [np.asarray(v, dtype=np.float32) for v in sampled["embedding_vec"].to_numpy()]
                    )
                    if Xs.shape[0] < 5:
                        st.info(
                            f"Frame UMAP skipped: need at least 5 sampled points (have {Xs.shape[0]}). "
                            "Load more frames or relax filters."
                        )
                    else:
                        nn_pts = int(min(15, max(2, Xs.shape[0] - 1)))
                        red = umap.UMAP(
                            n_components=2,
                            random_state=42,
                            n_neighbors=nn_pts,
                            min_dist=0.1,
                            metric=str(umap_metric),
                        )
                        xy = red.fit_transform(Xs)
                        plot_df = sampled.reset_index(drop=True).copy()
                        plot_df["umap_x"] = xy[:, 0]
                        plot_df["umap_y"] = xy[:, 1]
                        if umap_color_by == "cluster_id":
                            plot_df["__color__"] = plot_df["cluster_id"].astype(str)
                        elif umap_color_by == "class_name":
                            plot_df["__color__"] = plot_df["class_name"].astype(str)
                        else:
                            plot_df["__color__"] = plot_df["wav"].astype(str)
                        hover_cols = ["cluster_id", "class_name", "wav", "time_s"]
                        hover_cols = [c for c in hover_cols if c in plot_df.columns]
                        fig_u = px.scatter(
                            plot_df,
                            x="umap_x",
                            y="umap_y",
                            color="__color__",
                            hover_data=hover_cols,
                            title=f"Frame-level UMAP (n={len(plot_df)}, metric={umap_metric})",
                        )
                        fig_u.update_layout(legend_title_text=umap_color_by)
                        st.plotly_chart(fig_u, use_container_width=True)

                    C, cnts, cids = _cluster_centroid_matrix(work_global, include_noise=g_include_noise)
                    if C.shape[0] < 2:
                        st.caption("Centroid UMAP needs at least two clusters in the current filter.")
                    else:
                        nn_c = int(min(15, max(2, C.shape[0] - 1)))
                        red_c = umap.UMAP(
                            n_components=2,
                            random_state=42,
                            n_neighbors=nn_c,
                            min_dist=0.1,
                            metric=str(umap_metric),
                        )
                        Z = red_c.fit_transform(C)
                        cent_df = pd.DataFrame(
                            {
                                "umap_x": Z[:, 0],
                                "umap_y": Z[:, 1],
                                "cluster_id": cids,
                                "n_frames": cnts.astype(int),
                            }
                        )
                        if len(summ) > 0:
                            m = summ.set_index("cluster_id")
                            cent_df["dominant_class"] = [
                                str(m.loc[int(cid), "dominant_class"])
                                if int(cid) in m.index
                                else "—"
                                for cid in cent_df["cluster_id"]
                            ]
                            cent_df["class_purity"] = [
                                float(m.loc[int(cid), "class_purity"])
                                if int(cid) in m.index
                                else float("nan")
                                for cid in cent_df["cluster_id"]
                            ]
                            cent_df["dominant_wav"] = [
                                str(m.loc[int(cid), "dominant_wav"])
                                if int(cid) in m.index
                                else "—"
                                for cid in cent_df["cluster_id"]
                            ]
                            cent_df["n_wav"] = [
                                int(m.loc[int(cid), "n_wav"])
                                if int(cid) in m.index
                                else 0
                                for cid in cent_df["cluster_id"]
                            ]
                            cent_df["wav_purity"] = [
                                float(m.loc[int(cid), "wav_purity"])
                                if int(cid) in m.index
                                else float("nan")
                                for cid in cent_df["cluster_id"]
                            ]
                        else:
                            cent_df["dominant_class"] = "—"
                            cent_df["class_purity"] = float("nan")
                            cent_df["dominant_wav"] = "—"
                            cent_df["n_wav"] = 0
                            cent_df["wav_purity"] = float("nan")
                        max_sz = float(cent_df["n_frames"].max()) if len(cent_df) else 1.0
                        cent_df["marker_size"] = 8.0 + 22.0 * (cent_df["n_frames"] / max(max_sz, 1.0))
                        fig_c = px.scatter(
                            cent_df,
                            x="umap_x",
                            y="umap_y",
                            size="marker_size",
                            color="dominant_class",
                            hover_data=[
                                "cluster_id",
                                "n_frames",
                                "n_wav",
                                "wav_purity",
                                "dominant_wav",
                                "class_purity",
                            ],
                            title=f"Cluster-centroid UMAP (K={len(cent_df)}, metric={umap_metric})",
                        )
                        fig_c.update_layout(legend_title_text="dominant_class")
                        st.plotly_chart(fig_c, use_container_width=True)

        with st.container(border=True):
            colX, colY, colZ = st.columns(3)
            with colX:
                context_pad = st.slider("Context padding (s)", 0.0, 2.0, 0.5, 0.05)
            with colY:
                num_samples_top = st.slider("Number of items to show", 1, 20, 6, key="num_samples_top")
            with colZ:
                stable_examples = st.checkbox("Stable examples (same on rerun)", value=True)

        unique_clusters = sorted(df["cluster_id"].unique())
        selected_cluster = st.selectbox("Select Cluster", unique_clusters)

        cluster_data = df[df["cluster_id"] == selected_cluster].copy()
        st.subheader(f"Cluster {selected_cluster} – {len(cluster_data)} frames")

        if "viz_mode" not in st.session_state:
            st.session_state["viz_mode"] = "Frame view"

        colm1, colm2 = st.columns([1, 1])
        with colm1:
            if st.button(
                "Switch to Segment View" if st.session_state["viz_mode"] == "Frame view" else "Switch to Frame View"
            ):
                st.session_state["viz_mode"] = (
                    "Segment view" if st.session_state["viz_mode"] == "Frame view" else "Frame view"
                )
        with colm2:
            st.caption(f"Current mode: **{st.session_state['viz_mode']}**")

        view_mode = st.session_state["viz_mode"]

        if "class_name" in cluster_data.columns:
            class_choices = sorted(cluster_data["class_name"].dropna().astype(str).unique().tolist())
            selected_classes = st.multiselect(
                "Filter by class (leave empty = all classes)",
                options=class_choices,
                default=[],
            )
            if selected_classes:
                cluster_data = cluster_data[cluster_data["class_name"].isin(selected_classes)]
                st.caption(f"Showing {len(cluster_data)} frames after class filter")
        else:
            st.info("No `class_name` column found in the CSV, class filter not available.")
            selected_classes = []

        if len(cluster_data) == 0:
            st.warning("No items to show for this cluster / filter combination.")
            st.stop()

        if "sample_seed" not in st.session_state:
            st.session_state["sample_seed"] = 0

        if view_mode == "Segment view":
            needed = {"segment_uid", "segment_start_s", "segment_end_s"}
            if not needed.issubset(cluster_data.columns):
                st.error(
                    "Segment view requires segment columns in the clustering results. "
                    "Run Tab 2 with CSV labels enabled."
                )
                st.stop()

            segment_data = cluster_data.dropna(subset=["segment_uid", "segment_start_s", "segment_end_s"]).copy()
            if len(segment_data) == 0:
                st.warning("No labeled segments available for this cluster.")
                st.stop()

            segment_rows = (
                segment_data.groupby("segment_uid", as_index=False)
                .agg(
                    {
                        "wav": "first",
                        "class_name": "first",
                        "segment_start_s": "first",
                        "segment_end_s": "first",
                    }
                )
            )

            st.caption(
                "Each spectrogram below shows one unique annotation segment. "
                "The shaded region is the segment, and cyan lines are clustered frame times inside that segment."
            )

            max_show = min(50, len(segment_rows))
            default_n = min(int(num_samples_top), max_show)
            default_n = max(1, default_n)
            num_segments = st.slider("Number of segments to show", 1, max_show, default_n, key="num_segments")

            col_a, col_b = st.columns([1, 1])
            with col_a:
                if st.button("Resample segments"):
                    st.session_state["sample_seed"] = int(np.random.randint(0, 2**31 - 1))
            with col_b:
                st.caption("Each item below is one unique segment, not one frame.")

            n_show = min(int(num_segments), len(segment_rows))
            if stable_examples:
                seed = int(st.session_state.get("sample_seed", 0))
                shown_segments = segment_rows.sample(n_show, random_state=seed)
            else:
                shown_segments = segment_rows.sample(n_show)

            if "class_name" in segment_rows.columns:
                counts = segment_rows["class_name"].astype(str).value_counts().reset_index()
                counts.columns = ["class_name", "count"]
                fig = px.bar(
                    counts,
                    x="class_name",
                    y="count",
                    text="count",
                    title=f"Segment-class distribution in Cluster {selected_cluster}",
                )
                fig.update_layout(xaxis_title="Class label", yaxis_title="Number of segments", xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Segment Spectrograms & Audio")

            cols = st.columns(3)

            for idx, (_, seg_row) in enumerate(shown_segments.iterrows()):
                with cols[idx % 3]:
                    wav_file = str(seg_row["wav"])
                    class_label = str(seg_row["class_name"]) if "class_name" in seg_row.index else "Unknown"
                    seg_start = float(seg_row["segment_start_s"])
                    seg_end = float(seg_row["segment_end_s"])
                    seg_uid = seg_row["segment_uid"]
                    seg_dur = max(0.0, seg_end - seg_start)

                    frames_in_segment = segment_data[segment_data["segment_uid"] == seg_uid].copy()
                    frame_times = np.sort(frames_in_segment["time_s"].to_numpy(dtype=np.float32))

                    ctx_start = max(0.0, seg_start - context_pad)
                    ctx_end = seg_end + context_pad
                    ctx_dur = max(0.001, ctx_end - ctx_start)

                    name_candidates = _candidate_wav_names(wav_file)
                    display_name = (
                        name_candidates[1]
                        if len(name_candidates) >= 2
                        else (name_candidates[0] if name_candidates else wav_file)
                    )

                    st.write(f"**{display_name}**")
                    st.markdown(f"**Class:** `{class_label}`")
                    st.caption(f"Segment: {seg_start:.3f}s – {seg_end:.3f}s (dur={seg_dur:.3f}s)")
                    st.caption(f"Cluster frames in this segment: {len(frame_times)}")

                    full_wav_path = resolve_audio_path(audio_folder, wav_file)
                    if not os.path.exists(full_wav_path):
                        st.warning("File not found.")
                        st.caption(full_wav_path)
                        continue

                    try:
                        y, sr = librosa.load(
                            full_wav_path,
                            sr=None,
                            offset=float(ctx_start),
                            duration=float(ctx_dur),
                            mono=True,
                        )
                        if y is None or y.size == 0:
                            st.warning("Empty audio segment.")
                            continue
                    except Exception as e:
                        st.error(f"Audio load error: {e}")
                        continue

                    try:
                        S_db, t_axis, f_axis = safe_spectrogram(y, sr)

                        fig = go.Figure(
                            data=go.Heatmap(
                                z=S_db,
                                x=t_axis,
                                y=f_axis,
                                colorscale="Magma",
                                showscale=False,
                            )
                        )

                        fig.add_vrect(
                            x0=seg_start - ctx_start,
                            x1=seg_end - ctx_start,
                            fillcolor="rgba(255, 75, 75, 0.20)",
                            line_width=0,
                        )

                        for ft in frame_times:
                            rel_t = float(ft - ctx_start)
                            fig.add_vline(
                                x=rel_t,
                                line_width=2,
                                line_dash="solid",
                                line_color="rgba(180, 140, 255, 0.55)",
                            )

                        fig.update_layout(
                            height=260,
                            margin=dict(l=0, r=0, t=0, b=0),
                            plot_bgcolor="rgb(20, 0, 30)",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.info(f"No spectrogram: {e}")

                    try:
                        wav_bytes = wav_bytes_from_float32(y, sr)
                        st.audio(wav_bytes, format="audio/wav")
                    except Exception as e:
                        st.error(f"Audio render error: {e}")

        else:
            st.caption("Frame view: each item below is a clustered frame with a local context window around it.")

            max_show = min(50, len(cluster_data))
            default_n = min(int(num_samples_top), max_show)
            default_n = max(1, default_n)
            num_samples = st.slider("Number of frames to show", 1, max_show, default_n, key="num_samples")

            col_a, col_b = st.columns([1, 1])
            with col_a:
                if st.button("Resample examples"):
                    st.session_state["sample_seed"] = int(np.random.randint(0, 2**31 - 1))
            with col_b:
                st.caption("Click to redraw sampled frames.")

            n_show = min(int(num_samples), len(cluster_data))
            if stable_examples:
                seed = int(st.session_state.get("sample_seed", 0))
                samples = cluster_data.sample(n_show, random_state=seed)
            else:
                samples = cluster_data.sample(n_show)

            if "class_name" in cluster_data.columns:
                counts = cluster_data["class_name"].value_counts().reset_index()
                counts.columns = ["class_name", "count"]
                fig = px.bar(
                    counts,
                    x="class_name",
                    y="count",
                    text="count",
                    title=f"Class distribution in Cluster {selected_cluster}",
                )
                fig.update_layout(xaxis_title="Class label", yaxis_title="Number of frames", xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No `class_name` column found in the CSV, so class distribution can’t be shown.")

            st.markdown("### Spectrograms & Audio (frame-centered examples)")

            cols = st.columns(3)

            for idx, (_, row) in enumerate(samples.iterrows()):
                with cols[idx % 3]:
                    wav_file = str(row["wav"])
                    start_s = float(row["start_s"])
                    end_s = float(row["end_s"])
                    dur = max(0.0, end_s - start_s)
                    class_label = row["class_name"] if "class_name" in row.index else "Unknown"

                    mid = 0.5 * (start_s + end_s)
                    ctx_start = max(0.0, mid - context_pad)
                    ctx_end = max(ctx_start, mid + context_pad)
                    ctx_start = min(ctx_start, start_s)
                    ctx_end = max(ctx_end, end_s)
                    ctx_dur = max(0.0, ctx_end - ctx_start)

                    name_candidates = _candidate_wav_names(wav_file)
                    display_name = (
                        name_candidates[1]
                        if len(name_candidates) >= 2
                        else (name_candidates[0] if name_candidates else wav_file)
                    )

                    st.write(f"**{display_name}**")
                    st.markdown(f"**Class:** `{class_label}`")
                    st.caption(f"Frame window: {start_s:.3f}s – {end_s:.3f}s (dur={dur:.3f}s)")
                    st.caption(f"Context: {ctx_start:.3f}s – {ctx_end:.3f}s (dur={ctx_dur:.3f}s)")

                    full_wav_path = resolve_audio_path(audio_folder, wav_file)
                    if not os.path.exists(full_wav_path):
                        st.warning("File not found.")
                        st.caption(full_wav_path)
                        continue

                    try:
                        y, sr = librosa.load(
                            full_wav_path,
                            sr=None,
                            offset=float(ctx_start),
                            duration=float(max(0.001, ctx_dur)),
                            mono=True,
                        )
                        if y is None or y.size == 0:
                            st.warning("Empty audio segment.")
                            continue
                    except Exception as e:
                        st.error(f"Audio load error: {e}")
                        continue

                    try:
                        S_db, t_axis, f_axis = safe_spectrogram(y, sr)

                        ev_x0 = max(0.0, start_s - ctx_start)
                        ev_x1 = max(ev_x0, end_s - ctx_start)

                        fig = go.Figure(
                            data=go.Heatmap(
                                z=S_db,
                                x=t_axis,
                                y=f_axis,
                                colorscale="Magma",
                                showscale=False,
                            )
                        )

                        fig.add_vrect(
                            x0=ev_x0,
                            x1=ev_x1,
                            fillcolor="rgba(255, 75, 75, 0.25)",
                            line_width=0,
                        )

                        fig.update_layout(
                            height=260,
                            margin=dict(l=0, r=0, t=0, b=0),
                            plot_bgcolor="rgb(20, 0, 30)",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.info(f"No spectrogram (segment too short / STFT issue): {e}")

                    try:
                        wav_bytes = wav_bytes_from_float32(y, sr)
                        st.audio(wav_bytes, format="audio/wav")
                    except Exception as e:
                        st.error(f"Audio render error: {e}")

        if not _HAS_SOUNDFILE:
            st.caption("Note: 'soundfile' not installed; using wave fallback for audio rendering.")
