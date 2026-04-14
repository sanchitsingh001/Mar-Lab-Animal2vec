#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import h5py


def iter_wavs(root: Path, recursive: bool) -> Iterator[Path]:
    pats = ("*.wav", "*.WAV")
    if recursive:
        for p in pats:
            yield from root.rglob(p)
    else:
        for p in pats:
            yield from root.glob(p)


def safe_stem_from_relpath(rel: Path) -> str:
    return "_".join(rel.with_suffix("").parts)


def resample_audio(x: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    if src_sr == tgt_sr:
        return x
    g = math.gcd(src_sr, tgt_sr)
    up = tgt_sr // g
    down = src_sr // g
    y = np.vstack([resample_poly(x[:, c], up=up, down=down) for c in range(x.shape[1])]).T
    return y


def write_dummy_h5(path: Path) -> None:
    """Create a minimal valid HDF5 file with an empty 'labels' dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with h5py.File(str(path), "w") as f:
        f.create_dataset("labels", data=np.zeros((0,), dtype=np.int16))


def main():
    ap = argparse.ArgumentParser(
        description="Prepare Animal2Vec-style dataset: wav/ lbl/ manifest/ from raw wavs."
    )
    ap.add_argument("--input", required=True, help="Input directory containing raw wavs")
    ap.add_argument("--out-root", required=True, help="Where to create the processed dataset folder")
    ap.add_argument("--name", default=None, help="Dataset folder name (default: input dir name)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    ap.add_argument("--chunk-seconds", type=float, default=6.0, help="Chunk duration (default: 6.0)")
    ap.add_argument("--target-sr", type=int, default=32000, help="Resample to this SR (default: 32000)")
    ap.add_argument("--target-hours", type=float, default=None, help="Stop after this many total hours")
    ap.add_argument("--mono", action="store_true", help="Convert to mono (average channels)")
    ap.add_argument("--manifest-name", default="pretrain", help="Manifest subset name (default: pretrain)")
    ap.add_argument("--log-every-files", type=int, default=1, help="Log every N input files (default: 1)")
    ap.add_argument("--log-every-chunks", type=int, default=0,
                    help="If >0, log every N chunks globally (WARNING: huge logs for big runs)")
    args = ap.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Not a directory: {input_dir}")

    out_root = Path(args.out_root).expanduser().resolve()
    dataset_name = args.name or input_dir.name
    dataset_root = out_root / dataset_name

    wav_dir = dataset_root / "wav"
    lbl_dir = dataset_root / "lbl"
    manifest_dir = dataset_root / "manifest"
    wav_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / f"{args.manifest_name}.tsv"

    wav_files = sorted(iter_wavs(input_dir, args.recursive))
    if not wav_files:
        print("No wav files found.")
        return

    target_seconds: Optional[int] = None
    if args.target_hours is not None:
        target_seconds = int(round(args.target_hours * 3600))

    tgt_frames = int(round(args.chunk_seconds * args.target_sr))

    entries: List[Tuple[str, int]] = []
    total_written_seconds = 0
    global_id = 0
    skipped_files = 0

    for file_idx, wav_path in enumerate(wav_files, start=1):
        if target_seconds is not None and total_written_seconds >= target_seconds:
            break

        try:
            info = sf.info(str(wav_path))
            src_sr = info.samplerate
            frames_per_chunk = int(round(args.chunk_seconds * src_sr))

            if info.frames < 1 or frames_per_chunk <= 0:
                skipped_files += 1
                continue

            rel = wav_path.relative_to(input_dir)
            base = safe_stem_from_relpath(rel)

            chunks_written_for_file = 0

            with sf.SoundFile(str(wav_path), mode="r") as f:
                chunk_idx = 0
                while True:
                    if target_seconds is not None and total_written_seconds >= target_seconds:
                        break

                    data = f.read(frames_per_chunk, dtype="float32", always_2d=True)
                    if data.shape[0] == 0:
                        break

                    # pad last chunk to full length
                    if data.shape[0] < frames_per_chunk:
                        pad = frames_per_chunk - data.shape[0]
                        data = np.pad(data, ((0, pad), (0, 0)), mode="constant")

                    if args.mono and data.shape[1] > 1:
                        data = data.mean(axis=1, keepdims=True)

                    data_rs = resample_audio(data, src_sr, args.target_sr)

                    # enforce exact length post-resample
                    if data_rs.shape[0] != tgt_frames:
                        if data_rs.shape[0] > tgt_frames:
                            data_rs = data_rs[:tgt_frames]
                        else:
                            pad = tgt_frames - data_rs.shape[0]
                            data_rs = np.pad(data_rs, ((0, pad), (0, 0)), mode="constant")

                    out_wav_name = f"{base}_chunk_{chunk_idx:06d}_id_{global_id:09d}_{args.target_sr//1000}k.wav"
                    out_wav_path = wav_dir / out_wav_name
                    sf.write(str(out_wav_path), data_rs, args.target_sr, subtype="PCM_16")

                    out_h5_path = lbl_dir / f"{out_wav_path.stem}.h5"
                    write_dummy_h5(out_h5_path)

                    rel_wav = Path("wav") / out_wav_name
                    entries.append((rel_wav.as_posix(), tgt_frames))

                    total_written_seconds += int(round(args.chunk_seconds))
                    global_id += 1
                    chunk_idx += 1
                    chunks_written_for_file += 1

                    if args.log_every_chunks and (global_id % args.log_every_chunks == 0):
                        print(f"[chunks] global={global_id} hours={total_written_seconds/3600:.2f}")

                    # stop at EOF
                    if f.tell() >= info.frames:
                        break

            if args.log_every_files and (file_idx % args.log_every_files == 0):
                print(
                    f"[file {file_idx}/{len(wav_files)}] {rel} | src_sr={src_sr} | chunks={chunks_written_for_file} "
                    f"| total_hours={total_written_seconds/3600:.2f}"
                )

        except Exception as e:
            skipped_files += 1
            print(f"Skipping {wav_path} because {repr(e)}")
            continue

    # Write manifest (root path first line)
    with manifest_path.open("w", encoding="utf-8") as mf:
        mf.write(str(dataset_root) + "\n")
        for p, nframes in entries:
            mf.write(f"{p}\t{nframes}\n")

    print("\nDone.")
    print(f"Dataset root: {dataset_root}")
    print(f"Audio dir:    {wav_dir}")
    print(f"Label dir:    {lbl_dir}")
    print(f"Manifest:     {manifest_path}")
    print(f"Chunks:       {len(entries)}")
    print(f"Hours:        {total_written_seconds/3600:.3f}")
    print(f"Skipped files:{skipped_files}")


if __name__ == "__main__":
    main()

