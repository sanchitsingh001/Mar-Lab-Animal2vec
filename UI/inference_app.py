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

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import librosa
import h5py

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
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
def run_streaming_subprocess(cmd: List[str], env: Optional[Dict[str, str]] = None) -> int:
    st.code("Running command:\n" + " ".join(cmd), language="bash")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
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

    bash_cmd = ["bash", "-lc", cmd_str]
    tmux_cmd = ["tmux", "new-session", "-d", "-s", session_name] + bash_cmd
    return run_streaming_subprocess(tmux_cmd, env=env)


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

        # Auto-detect segment-export H5 naming so it still works even if the checkbox is forgotten.
        auto_segment_mode = bool(segment_h5_use_recording_csv) or (".wav_embeddings_" in h5_name)

        if auto_segment_mode:
            rec_stem = recording_stem_from_segment_h5_name(h5_name)
            if rec_stem is None:
                debug_rows.append(
                    {
                        "h5_name": h5_name,
                        "mode": "segment_h5->recording_csv",
                        "reason": "could not derive recording stem from h5 name",
                        "csv_path": None,
                    }
                )
                continue

            if ignore_test_files and "testfile" in rec_stem.lower():
                debug_rows.append(
                    {
                        "h5_name": h5_name,
                        "mode": "segment_h5->recording_csv",
                        "reason": "skipped test file",
                        "csv_path": None,
                    }
                )
                continue

            csv_path = wav_csv_dir_p / f"{rec_stem}.csv"
            recording_id = rec_stem
        else:
            wav_name = str(g["wav"].iloc[0]).strip()
            stem = Path(wav_name).stem

            if ignore_test_files and "testfile" in stem.lower():
                debug_rows.append(
                    {
                        "h5_name": h5_name,
                        "mode": "direct_stem_csv",
                        "reason": "skipped test file",
                        "csv_path": None,
                    }
                )
                continue

            csv_path = wav_csv_dir_p / f"{stem}.csv"
            recording_id = stem

        if not csv_path.exists():
            debug_rows.append(
                {
                    "h5_name": h5_name,
                    "mode": "segment_h5->recording_csv" if auto_segment_mode else "direct_stem_csv",
                    "reason": "csv not found",
                    "csv_path": str(csv_path),
                }
            )
            continue

        df_seg = read_label_csv(csv_path)
        if df_seg is None:
            debug_rows.append(
                {
                    "h5_name": h5_name,
                    "mode": "segment_h5->recording_csv" if auto_segment_mode else "direct_stem_csv",
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
                    "mode": "segment_h5->recording_csv" if auto_segment_mode else "direct_stem_csv",
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
                    "mode": "segment_h5->recording_csv" if auto_segment_mode else "direct_stem_csv",
                    "reason": "csv matched but no frame times fell into any segment",
                    "csv_path": str(csv_path),
                }
            )
            continue

        debug_rows.append(
            {
                "h5_name": h5_name,
                "mode": "segment_h5->recording_csv" if auto_segment_mode else "direct_stem_csv",
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

    cols = ["cluster_id", "wav", "start_s", "end_s", "class_name", "h5_path", "frame_idx", "time_s"]
    for extra_col in ["segment_start_s", "segment_end_s", "segment_uid"]:
        if extra_col in out.columns:
            cols.append(extra_col)

    return out[cols]


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
            out["nmi_vs_class_name"] = float(normalized_mutual_info_score(y_true.to_numpy(), labels))
            out["ami_vs_class_name"] = float(adjusted_mutual_info_score(y_true.to_numpy(), labels))
            out["ari_vs_class_name"] = float(adjusted_rand_score(y_true.to_numpy(), labels))

    return out


# =========================
# Tab 1: Embedding Generation
# =========================
with tab1:
    st.header("Generate Embeddings")
    st.caption("Runs animal2vec_inference.py on the server to produce .h5 embeddings and/or CSV outputs.")

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            inference_script = st.text_input("Path to inference script", value="../animal2vec_inference.py")
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

        write_embeddings = st.checkbox("Write embeddings (.h5)", value=True)
        write_other_predictions = st.checkbox("Write other predictions (CSV)", value=False)
        write_non_focal = st.checkbox("Write non-focal predictions (CSV)", value=False)
        additional_args = st.text_input("Additional Arguments (optional)", value="")

        st.markdown("---")
        st.subheader("Background execution (tmux)")
        use_tmux = st.checkbox("Run in tmux (recommended for long jobs)", value=True)
        tmux_session = st.text_input("tmux session name", value="a2v_embed")
        log_dir = st.text_input("Log directory", value=os.path.join(output_dir, "logs"))
        write_log = st.checkbox("Write logs to file", value=True)

        if st.button("Generate", type="primary"):
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
                rc = run_streaming_subprocess(inner_cmd, env=env)
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
                    st.caption("Mutual information computed vs `class_name` column (if present).")
                    st.write(
                        {
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
            output_csv = f"cluster_results_{algo_tag}.csv"
            try:
                df_results.to_csv(output_csv, index=False)
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
                                line_color="cyan",
                            )

                        fig.update_layout(height=260, margin=dict(l=0, r=0, t=0, b=0))
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
                            )
                        )

                        fig.add_vrect(
                            x0=ev_x0,
                            x1=ev_x1,
                            fillcolor="rgba(255, 75, 75, 0.25)",
                            line_width=0,
                        )

                        fig.update_layout(height=260, margin=dict(l=0, r=0, t=0, b=0))
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
