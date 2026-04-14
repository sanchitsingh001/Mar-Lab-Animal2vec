#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

CHUNK_SECONDS_DEFAULT = 6
TARGET_SR_DEFAULT = 32000
TARGET_HOURS_DEFAULT = 700


def resample_audio(x: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    """x: float32 array (n_samples, n_channels)."""
    if src_sr == tgt_sr:
        return x
    g = math.gcd(src_sr, tgt_sr)
    up = tgt_sr // g
    down = src_sr // g
    # resample per-channel
    y = np.vstack([resample_poly(x[:, c], up=up, down=down) for c in range(x.shape[1])]).T
    return y


def iter_wavs(root: Path, recursive: bool):
    pats = ("*.wav", "*.WAV")
    if recursive:
        for p in pats:
            yield from root.rglob(p)
    else:
        for p in pats:
            yield from root.glob(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=str, help="Directory containing wav files")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    ap.add_argument("--out-root", type=str, default=".", help="Output root (default: .)")
    ap.add_argument("--chunk-seconds", type=int, default=CHUNK_SECONDS_DEFAULT, help="Chunk length in seconds")
    ap.add_argument("--target-sr", type=int, default=TARGET_SR_DEFAULT, help="Target sample rate (Hz)")
    ap.add_argument("--target-hours", type=float, default=TARGET_HOURS_DEFAULT, help="Stop after this many hours")
    ap.add_argument("--mono", action="store_true", help="Convert to mono (average channels) before writing")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Not a directory: {input_dir}")

    out_root = Path(args.out_root).expanduser().resolve()
    out_dir = out_root / "processed" / input_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    target_seconds = int(args.target_hours * 3600)
    written_seconds = 0
    global_id = 0
    skipped_files = 0

    wavs = sorted(iter_wavs(input_dir, args.recursive))
    if not wavs:
        print("No .wav files found.")
        return

    for wav_path in wavs:
        if written_seconds >= target_seconds:
            break

        try:
            info = sf.info(str(wav_path))
            src_sr = info.samplerate
            frames_per_chunk = int(args.chunk_seconds * src_sr)

            # safe stem to avoid collisions across subfolders
            rel = wav_path.relative_to(input_dir)
            safe_stem = "_".join(rel.with_suffix("").parts)

            with sf.SoundFile(str(wav_path), mode="r") as f:
                chunk_idx = 0
                while written_seconds < target_seconds:
                    data = f.read(frames_per_chunk, dtype="float32", always_2d=True)
                    if data.shape[0] == 0:
                        break

                    # pad last chunk to exactly chunk length (so all outputs are same duration)
                    if data.shape[0] < frames_per_chunk:
                        pad = frames_per_chunk - data.shape[0]
                        data = np.pad(data, ((0, pad), (0, 0)), mode="constant")

                    if args.mono and data.shape[1] > 1:
                        data = data.mean(axis=1, keepdims=True)

                    data_rs = resample_audio(data, src_sr, args.target_sr)

                    out_path = out_dir / f"{safe_stem}_chunk_{chunk_idx:06d}_id_{global_id:09d}_{args.target_sr//1000}k.wav"
                    sf.write(str(out_path), data_rs, args.target_sr, subtype="PCM_16")

                    written_seconds += args.chunk_seconds
                    global_id += 1
                    chunk_idx += 1

                    # if we hit EOF originally (short read), stop for this file
                    if f.tell() >= info.frames:
                        break

        except Exception as e:
            skipped_files += 1
            print(f"Skipping {wav_path} because {repr(e)}")
            continue

    print(f"Done. Output: {out_dir}")
    print(f"Wrote ~{written_seconds/3600:.2f} hours ({global_id} chunks of {args.chunk_seconds}s) at {args.target_sr} Hz.")
    if skipped_files:
        print(f"Skipped {skipped_files} file(s).")


if __name__ == "__main__":
    main()

