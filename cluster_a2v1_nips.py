import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import hdbscan
# from sklearn.cluster import KMeans  # keep available if you want to switch back
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize  # recommended for embedding clustering
from sklearn.cluster import KMeans

# ---- Put your 87-class list here (string -> python list) ----
CLASS_NAMES = ['Empty','Aegcau_call','Alaarv_song','Anttri_song','Butbut_call','Carcan_call','Carcan_song','Carcar_call','Carcar_song','Cerbra_call','Cerbra_song','Cetcet_song','Chlchl_call','Cicatr_song','Cicorn_song','Cisjun_song','Colpal_song','Corcor_call','Denmaj_call','Denmaj_drum','Embcir_call','Embcir_song','Erirub_call','Erirub_song','Fricoe_call','Fricoe_song','Galcri_call','Galcri_song','Galthe_call','Galthe_song','Gargla_call','Hirrus_call','Jyntor_song','Lopcri_call','Loxcur_call','Lularb_song','Lusmeg_call','Lusmeg_song','Lyrple_song','Motcin_call','Musstr_call','Oriori_call','Oriori_song','Parate_call','Parate_song','Parcae_call','Parcae_song','Parmaj_call','Parmaj_song','Pasdom_call','Pelgra_call','Petpet_call','Petpet_song','Phofem_song','Phycol_call','Phycol_song','Picpic_call','Plaaff_song','Plasab_song','Poepal_call','Poepal_song','Prumod_song','Ptehey_song','Pyrpyr_call','Regign_call','Regign_song','Serser_call','Serser_song','Siteur_call','Siteur_song','Strdec_song','Strtur_song','Stuvul_call','Sylatr_call','Sylatr_song','Sylcan_call','Sylcan_song','Sylmel_call','Sylmel_song','Sylund_call','Sylund_song','Tetpyg_song','Tibtom_song','Trotro_song','Turmer_call','Turmer_song','Turphi_call','Turphi_song','Unknown']


def read_label_csv(csv_path: Path):
    """
    Expected CSV rows: start_idx,end_idx,class_id
    Returns None if file is empty/unreadable.
    """
    from pandas.errors import EmptyDataError

    if not csv_path.exists():
        return None
    if csv_path.stat().st_size == 0:
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

    # Coerce to int; drop rows that can't parse
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
    # Mean-pool along time
    seg = emb_TxC[start_i:end_i]
    return seg.mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_csv_dir", required=True, type=str,
                    help="Dir containing matching .wav and .csv (NIPS annotations).")
    ap.add_argument("--emb_dir", required=True, type=str,
                    help="Dir containing a2v1 .h5 embedding files.")
    ap.add_argument("--index_rate_hz", type=float, default=8000.0,
                    help="How to interpret start/end indices in CSV: idx / index_rate_hz = seconds. "
                         "If your CSV indices are samples at 44.1kHz, set 44100. If already 8k, keep 8000.")
    ap.add_argument("--min_dur_s", type=float, default=0.02,
                    help="Drop labeled segments shorter than this duration (seconds).")
    ap.add_argument("--max_segments", type=int, default=0,
                    help="Optional cap for debugging (0 = no cap).")

    # Keep k for compatibility (not used by HDBSCAN)
    ap.add_argument("--k", type=int, default=60,
                    help="(Unused with HDBSCAN) Kept for compatibility with older KMeans runs.")

    # HDBSCAN params
    ap.add_argument("--min_cluster_size", type=int, default=10,
                    help="HDBSCAN min_cluster_size (try 5, 8, 10, 20).")
    ap.add_argument("--min_samples", type=int, default=1,
                    help="HDBSCAN min_samples (1 is permissive; try 1, 2, 5, 10).")
    ap.add_argument("--metric", type=str, default="cosine",
                    help="HDBSCAN distance metric; for embeddings, 'cosine' is often best.")

    ap.add_argument("--out_csv", type=str, default="a2v1_nips_segment_vectors.csv",
                    help="Where to save pooled vectors + metadata (CSV).")
    args = ap.parse_args()

    wav_csv_dir = Path(args.wav_csv_dir)
    emb_dir = Path(args.emb_dir)

    # Find all embedding files
    emb_files = sorted(emb_dir.glob("*.h5"))
    if not emb_files:
        raise SystemExit(f"No .h5 files found in {emb_dir}")

    pooled = []
    y_true = []
    meta = []

    # Build a mapping from wavstem.wav -> embedding file
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
    for wav_path in tqdm(wavs, desc="Extracting labeled segments"):
        csv_path = wav_path.with_suffix(".csv")
        if not csv_path.exists():
            continue

        emb_path = emb_map.get(wav_path.name, None)
        if emb_path is None or not emb_path.exists():
            continue

        # Load embeddings + time
        with h5py.File(emb_path, "r") as f:
            emb = f["embedding"][:]   # (T, C)
            t = f["time"][:]          # (T,) seconds

        df = read_label_csv(csv_path)
        if df is None:
            continue

        for _, row in df.iterrows():
            cls = int(row["cls"])
            if cls < 0 or cls >= len(CLASS_NAMES):
                continue

            start_s = float(row["start"]) / args.index_rate_hz
            end_s = float(row["end"]) / args.index_rate_hz
            dur_s = end_s - start_s
            if dur_s < args.min_dur_s:
                continue

            # Map seconds -> embedding indices using the time vector
            start_i = int(np.searchsorted(t, start_s, side="left"))
            end_i = int(np.searchsorted(t, end_s, side="right"))

            # Clamp
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
        raise SystemExit("No segments extracted. Most common causes: wrong --index_rate_hz or csv parse mismatch.")

    X = np.stack(pooled, axis=0)
    y_true = np.array(y_true, dtype=int)
    print(f"Extracted {X.shape[0]} segments with vector dim {X.shape[1]}")

    # Recommended: normalize embeddings before distance-based clustering
    X = normalize(X)

    # --------------------------
    # KMeans clustering (COMMENTED OUT)
    # --------------------------
    k = min(args.k, X.shape[0])
    if k < 2:
        raise SystemExit("Not enough segments to cluster.")
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    y_pred = km.fit_predict(X)

    # --------------------------
    # HDBSCAN clustering (ACTIVE)
    # --------------------------
#    clusterer = hdbscan.HDBSCAN(
#        min_cluster_size=args.min_cluster_size,
#        min_samples=args.min_samples,
#        metric=args.metric
#    )
#    y_pred = clusterer.fit_predict(X)  # noise labeled as -1
#
    # Diagnostics
    n_noise = int((y_pred == -1).sum())
    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    counts = pd.Series(y_pred).value_counts()
    print(f"HDBSCAN clusters: {n_clusters}, noise points: {n_noise}/{len(y_pred)} ({n_noise/len(y_pred):.1%})")
    print("Top cluster sizes:\n", counts.head(10))

    # Metrics
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    sil = None
    try:
        # Silhouette: ignore noise and require >= 2 clusters among non-noise points
        mask = (y_pred != -1)
        if mask.sum() > 10 and len(np.unique(y_pred[mask])) > 1 and mask.sum() <= 50000:
            sil = silhouette_score(X[mask], y_pred[mask])
    except Exception:
        sil = None

    print(f"ARI: {ari:.4f}")
    print(f"NMI: {nmi:.4f}")
    if sil is not None:
        print(f"Silhouette (non-noise): {sil:.4f}")

    # Save per-segment output
    out_csv = Path(args.out_csv)
    df_meta = pd.DataFrame(meta)
    vec_df = pd.DataFrame(X, columns=[f"v{i}" for i in range(X.shape[1])])

    df_out = pd.concat([df_meta, pd.Series(y_pred, name="cluster_id"), vec_df], axis=1)
    df_out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv.resolve()}")


if __name__ == "__main__":
    main()

