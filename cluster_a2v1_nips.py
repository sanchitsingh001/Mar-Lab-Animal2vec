#!/usr/bin/env python3
"""
cluster_a2v1_nips.py (updated)

Extract labeled segments from (wav,csv) pairs, mean-pool their embeddings from .h5 files,
OPTIONALLY filter out class 0 (Empty/background),
OPTIONALLY denoise with a denoising autoencoder (DAE),
then cluster with either MiniBatchKMeans or HDBSCAN.

This version matches the previous researcher's "remove target 0 + denoise + kmeans(k=60)" pipeline.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# KMeans used in the notebook
from sklearn.cluster import MiniBatchKMeans

# Your unified clustering API (still used for HDBSCAN, optional for KMeans)
from clustering import run_clustering


# ---- Put your 87-class list here (string -> python list) ----
CLASS_NAMES = [
    "Empty","Aegcau_call","Alaarv_song","Anttri_song","Butbut_call","Carcan_call","Carcan_song",
    "Carcar_call","Carcar_song","Cerbra_call","Cerbra_song","Cetcet_song","Chlchl_call","Cicatr_song",
    "Cicorn_song","Cisjun_song","Colpal_song","Corcor_call","Denmaj_call","Denmaj_drum","Embcir_call",
    "Embcir_song","Erirub_call","Erirub_song","Fricoe_call","Fricoe_song","Galcri_call","Galcri_song",
    "Galthe_call","Galthe_song","Gargla_call","Hirrus_call","Jyntor_song","Lopcri_call","Loxcur_call",
    "Lularb_song","Lusmeg_call","Lusmeg_song","Lyrple_song","Motcin_call","Musstr_call","Oriori_call",
    "Oriori_song","Parate_call","Parate_song","Parcae_call","Parcae_song","Parmaj_call","Parmaj_song",
    "Pasdom_call","Pelgra_call","Petpet_call","Petpet_song","Phofem_song","Phycol_call","Phycol_song",
    "Picpic_call","Plaaff_song","Plasab_song","Poepal_call","Poepal_song","Prumod_song","Ptehey_song",
    "Pyrpyr_call","Regign_call","Regign_song","Serser_call","Serser_song","Siteur_call","Siteur_song",
    "Strdec_song","Strtur_song","Stuvul_call","Sylatr_call","Sylatr_song","Sylcan_call","Sylcan_song",
    "Sylmel_call","Sylmel_song","Sylund_call","Sylund_song","Tetpyg_song","Tibtom_song","Trotro_song",
    "Turmer_call","Turmer_song","Turphi_call","Turphi_song","Unknown"
]


def read_label_csv(csv_path: Path):
    """
    Expected CSV rows: start_idx,end_idx,class_id
    Returns None if file is empty/unreadable.
    """
    from pandas.errors import EmptyDataError

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None

    try:
        df = pd.read_csv(csv_path, header=None)
    except EmptyDataError:
        return None
    except Exception:
        try:
            df = pd.read_csv(csv_path, header=None, sep=r"\s+")
        except EmptyDataError:
            return None

    if df is None or df.shape[0] == 0 or df.shape[1] < 3:
        return None

    df = df.iloc[:, :3]
    df.columns = ["start", "end", "cls"]

    for c in ["start", "end", "cls"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()

    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["cls"] = df["cls"].astype(int)

    df = df[df["end"] > df["start"]]
    if len(df) == 0:
        return None
    return df


def pool_segment(emb_TxC: np.ndarray, start_i: int, end_i: int):
    """Mean-pool along time (T)"""
    seg = emb_TxC[start_i:end_i]
    return seg.mean(axis=0)


def load_h5_embedding_and_time(h5_path: Path):
    """
    Robustly load embedding (T,C) and time (T,) from an .h5 file.
    Supports different dataset key names and skips files that don't match.
    """
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())

        emb_key_candidates = [
            "embedding", "embeddings", "embed", "features", "feat",
            "audio_embedding", "audio_embeddings", "repr", "representation"
        ]
        time_key_candidates = ["time", "times", "timestamp", "timestamps", "t"]

        emb = None
        t = None

        for k in emb_key_candidates:
            if k in f and getattr(f[k], "ndim", None) == 2:
                emb = f[k][:]
                break

        for k in time_key_candidates:
            if k in f and getattr(f[k], "ndim", None) == 1:
                t = f[k][:]
                break

        if emb is None:
            for k in keys:
                obj = f[k]
                if hasattr(obj, "ndim") and obj.ndim == 2:
                    emb = obj[:]
                    break

        if emb is None:
            raise KeyError(f"No 2D embedding dataset found. keys={keys}")

        T = emb.shape[0]

        if t is None:
            for k in keys:
                obj = f[k]
                if hasattr(obj, "ndim") and obj.ndim == 1 and len(obj) == T:
                    t = obj[:]
                    break

        if t is None:
            t = np.arange(T, dtype=np.float32)

        return emb, t, keys


def dae_denoise_numpy(X: np.ndarray,
                      bottleneck: int = 64,
                      epochs: int = 15,
                      batch_size: int = 512,
                      noise_std: float = 0.05,
                      lr: float = 1e-3,
                      seed: int = 42):
    """
    Denoising autoencoder that matches the notebook:
      input_dim=768 -> 256 -> bottleneck -> 256 -> 768
    trained with Gaussian noise added to inputs.
    Returns denoised reconstructions as numpy array.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as e:
        raise RuntimeError("PyTorch is required for --denoise. Install torch.") from e

    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = X.shape[1]
    if input_dim != 768:
        # still works, but warn
        print(f"[WARN] DAE was tuned for 768-D, but got {input_dim}-D. Proceeding anyway.")

    X_t = torch.tensor(X, dtype=torch.float32)
    X_noisy = X_t + noise_std * torch.randn_like(X_t)

    dataset = TensorDataset(X_noisy, X_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    class DAE(nn.Module):
        def __init__(self, input_dim: int, bottleneck: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, bottleneck),
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck, 256),
                nn.ReLU(),
                nn.Linear(256, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    dae = DAE(input_dim=input_dim, bottleneck=bottleneck)
    opt = torch.optim.Adam(dae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    dae.train()
    for ep in range(epochs):
        last_loss = None
        for xb_noisy, xb_clean in loader:
            recon = dae(xb_noisy)
            loss = loss_fn(recon, xb_clean)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
        print(f"[DAE] epoch {ep+1}/{epochs} loss={last_loss:.4f}")

    dae.eval()
    with torch.no_grad():
        denoised = dae(X_t).cpu().numpy()
    return denoised


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--wav_csv_dir", required=True, type=str,
                    help="Dir containing matching .wav and .csv (NIPS annotations).")
    ap.add_argument("--emb_dir", required=True, type=str,
                    help="Dir containing a2v1 .h5 embedding files.")
    ap.add_argument("--index_rate_hz", type=float, default=8000.0,
                    help="Interpret CSV start/end indices as: idx / index_rate_hz = seconds. "
                         "If your CSV indices are samples at 44.1kHz, set 44100. If already 8k, keep 8000.")
    ap.add_argument("--min_dur_s", type=float, default=0.02,
                    help="Drop labeled segments shorter than this duration (seconds).")
    ap.add_argument("--max_segments", type=int, default=0,
                    help="Optional cap for debugging (0 = no cap).")

    # --- Key replication knob from the notebook ---
    ap.add_argument("--skip_class0", action="store_true",
                    help="Drop class_id==0 (Empty/background). This is critical to match the notebook results.")
    ap.add_argument("--skip_unknown", action="store_true",
                help="Drop class_id == last CLASS_NAMES entry (often 'Unknown').")
    # --- DAE denoising (matches notebook defaults) ---
    ap.add_argument("--denoise", action="store_true",
                    help="Apply a denoising autoencoder before clustering (matches notebook pipeline).")
    ap.add_argument("--dae_bottleneck", type=int, default=64)
    ap.add_argument("--dae_epochs", type=int, default=15)
    ap.add_argument("--dae_batch_size", type=int, default=512)
    ap.add_argument("--dae_noise_std", type=float, default=0.05)
    ap.add_argument("--dae_lr", type=float, default=1e-3)

    # Algorithm choice
    ap.add_argument("--cluster_algo", type=str, default="kmeans",
                    choices=["hdbscan", "kmeans"],
                    help="Which clustering algorithm to use.")

    # KMeans params (note: notebook used MiniBatchKMeans)
    ap.add_argument("--k", type=int, default=60,
                    help="KMeans number of clusters (only used if --cluster_algo kmeans).")
    ap.add_argument("--kmeans_batch_size", type=int, default=128)
    ap.add_argument("--kmeans_n_init", type=int, default=10)

    # HDBSCAN params
    ap.add_argument("--min_cluster_size", type=int, default=10)
    ap.add_argument("--min_samples", type=int, default=0)
    ap.add_argument("--metric", type=str, default="cosine")
    ap.add_argument("--cluster_selection_method", type=str, default="eom", choices=["eom", "leaf"])

    ap.add_argument("--out_csv", type=str, default="a2v1_nips_segment_vectors.csv",
                    help="Where to save pooled vectors + metadata (CSV).")

    args = ap.parse_args()

    wav_csv_dir = Path(args.wav_csv_dir)
    emb_dir = Path(args.emb_dir)

    emb_files = sorted(emb_dir.glob("*.h5"))
    if not emb_files:
        raise SystemExit(f"No .h5 files found in {emb_dir}")

    pooled = []
    y_true = []
    meta = []

    emb_map = {}
    for p in emb_files:
        name = p.name
        if ".wav_embeddings_" in name:
            wavname = name.split(".wav_embeddings_")[0] + ".wav"
        else:
            wavname = name.split("_embeddings_")[0]
        emb_map[wavname] = p

    wavs = sorted(wav_csv_dir.glob("*.wav")) + sorted(wav_csv_dir.glob("*.WAV"))
    if not wavs:
        raise SystemExit(f"No wavs found in {wav_csv_dir}")

    seg_count = 0
    skipped_h5 = 0
    skipped_class0 = 0

    for wav_path in tqdm(wavs, desc="Extracting labeled segments"):
        csv_path = wav_path.with_suffix(".csv")
        if not csv_path.exists():
            continue

        emb_path = emb_map.get(wav_path.name, None)
        if emb_path is None or not emb_path.exists():
            continue

        try:
            emb, t, _keys = load_h5_embedding_and_time(emb_path)
        except Exception as e:
            skipped_h5 += 1
            print(f"[WARN] Skipping {emb_path.name} (bad h5 schema): {e}")
            continue

        df = read_label_csv(csv_path)
        if df is None:
            continue

        for _, row in df.iterrows():
            cls = int(row["cls"])
            if cls < 0 or cls >= len(CLASS_NAMES):
                continue

            if args.skip_class0 and cls == 0:
                skipped_class0 += 1
                continue
            if args.skip_unknown and cls == (len(CLASS_NAMES) - 1):
                continue


            start_s = float(row["start"]) / args.index_rate_hz
            end_s = float(row["end"]) / args.index_rate_hz
            dur_s = end_s - start_s
            if dur_s < args.min_dur_s:
                continue

            start_i = int(np.searchsorted(t, start_s, side="left"))
            end_i = int(np.searchsorted(t, end_s, side="right"))

            start_i = max(0, min(start_i, len(t) - 1))
            end_i = max(start_i + 1, min(end_i, len(t)))

            v = pool_segment(emb, start_i, end_i)

            pooled.append(v)
            y_true.append(cls)
            meta.append({
                "wav": wav_path.name,
                "csv": csv_path.name,
                "class_id": cls,
                "class_name": CLASS_NAMES[cls],
                "start_s": start_s,
                "end_s": end_s,
                "dur_s": dur_s,
                "emb_file": emb_path.name,
                "emb_start_i": start_i,
                "emb_end_i": end_i,
            })

            seg_count += 1
            if args.max_segments and seg_count >= args.max_segments:
                break

        if args.max_segments and seg_count >= args.max_segments:
            break

    if not pooled:
        raise SystemExit(
            "No segments extracted. Common causes: wrong --index_rate_hz, CSV parse mismatch, or all H5s skipped."
        )

    X = np.stack(pooled, axis=0)
    y_true = np.array(y_true, dtype=int)

    print(f"Extracted {X.shape[0]} segments with vector dim {X.shape[1]}")
    if args.skip_class0:
        print(f"[FILTER] skipped class0 segments: {skipped_class0}")
    if skipped_h5:
        print(f"[WARN] Skipped {skipped_h5} .h5 files due to missing/unknown embedding keys.")

    # --- DAE denoising (notebook-style) ---
    X_for_cluster = X
    if args.denoise:
        print("[DAE] training denoising autoencoder...")
        X_for_cluster = dae_denoise_numpy(
            X,
            bottleneck=int(args.dae_bottleneck),
            epochs=int(args.dae_epochs),
            batch_size=int(args.dae_batch_size),
            noise_std=float(args.dae_noise_std),
            lr=float(args.dae_lr),
            seed=42
        )
        print("[DAE] done. clustering denoised embeddings.")

    # ------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------
    algo = args.cluster_algo.lower().strip()

    if algo == "kmeans":
        # Match notebook: MiniBatchKMeans in original 768-D space (after denoise if enabled)
        kmeans = MiniBatchKMeans(
            n_clusters=int(args.k),
            init="k-means++",
            random_state=42,
            n_init=int(args.kmeans_n_init),
            batch_size=int(args.kmeans_batch_size),
        )
        y_pred = kmeans.fit_predict(X_for_cluster)
        info = {"k": int(args.k), "kmeans": "MiniBatchKMeans"}
        n_noise = 0
    else:
        # HDBSCAN path uses your unified API
        min_samples = None if int(args.min_samples) == 0 else int(args.min_samples)
        y_pred, info = run_clustering(
            X_for_cluster,
            algo="hdbscan",
            l2_normalize=True,
            min_cluster_size=int(args.min_cluster_size),
            min_samples=min_samples,
            metric=str(args.metric),
            cluster_selection_method=str(args.cluster_selection_method),
        )
        n_noise = int((y_pred == -1).sum())

    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    counts = pd.Series(y_pred).value_counts()

    print(f"[{algo.upper()}] n_clusters={n_clusters}, noise={n_noise}/{len(y_pred)} ({n_noise/len(y_pred):.1%})")
    print("Top cluster sizes:\n", counts.head(10))

    # Metrics
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    sil = None
    try:
        mask = (y_pred != -1)
        if mask.sum() > 10 and len(np.unique(y_pred[mask])) > 1 and mask.sum() <= 50000:
            sil = silhouette_score(X_for_cluster[mask], y_pred[mask])
    except Exception:
        sil = None

    print(f"ARI: {ari:.4f}")
    print(f"NMI: {nmi:.4f}")
    if sil is not None:
        print(f"Silhouette (non-noise): {sil:.4f}")

    # Save output
    out_csv = Path(args.out_csv)
    df_meta = pd.DataFrame(meta)
    vec_df = pd.DataFrame(X_for_cluster, columns=[f"v{i}" for i in range(X_for_cluster.shape[1])])

    df_out = pd.concat(
        [
            df_meta,
            pd.Series(y_pred, name="cluster_id"),
            pd.Series([algo] * len(y_pred), name="cluster_algo"),
        ],
        axis=1,
    )
    df_out = pd.concat([df_out, vec_df], axis=1)
    df_out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
