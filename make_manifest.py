#!/usr/bin/env python3
import argparse
from pathlib import Path

import soundfile as sf


def iter_wavs(root: Path, recursive: bool):
    pats = ("*.wav", "*.WAV")
    if recursive:
        for p in pats:
            yield from root.rglob(p)
    else:
        for p in pats:
            yield from root.glob(p)


def main():
    ap = argparse.ArgumentParser(description="Create a TSV manifest: <relpath>\\t<num_samples>")
    ap.add_argument("wav_dir", help="Directory that contains wav files (or subfolders with wavs)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    ap.add_argument("--out", default="pretrain.tsv", help="Output manifest filename (default: pretrain.tsv)")
    ap.add_argument(
        "--base",
        default=None,
        help="Base directory to make paths relative to. Default: directory containing the manifest.",
    )
    ap.add_argument("--sort", action="store_true", help="Sort entries by path")
    args = ap.parse_args()

    wav_dir = Path(args.wav_dir).expanduser().resolve()
    if not wav_dir.is_dir():
        raise SystemExit(f"Not a directory: {wav_dir}")

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_dir = Path(args.base).expanduser().resolve() if args.base else out_path.parent

    wavs = list(iter_wavs(wav_dir, args.recursive))
    if args.sort:
        wavs.sort()

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for wav in wavs:
            try:
                info = sf.info(str(wav))
                num_samples = info.frames  # per-channel frame count (what most manifests want)
                rel = wav.relative_to(base_dir)
                f.write(f"{rel}\t{num_samples}\n")
                n += 1
            except Exception as e:
                print(f"Skipping {wav} because {repr(e)}")

    print(f"Wrote {n} entries to {out_path}")
    print(f"Paths are relative to: {base_dir}")


if __name__ == "__main__":
    main()

