import numpy as np
import soundfile as sf
from pathlib import Path

INPUT_DIR = Path("/host_root/cache/a/ssingh/Datasets/Chunked-Xeno/wav_24k/")
OUTPUT_DIR = Path("/host_root/cache/a/ssingh/Datasets/Chunked-Xeno/5s_wav_24k/")
CHUNK_SECONDS = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

chunk_id = 0

for wav_path in INPUT_DIR.rglob("*.wav"):
    try:
        data, sr = sf.read(str(wav_path), always_2d=True)  # shape: (n_samples, n_channels)
        frames_per_chunk = CHUNK_SECONDS * sr

        # full chunks only
        num_chunks = data.shape[0] // frames_per_chunk
        if num_chunks == 0:
            continue

        # avoid filename collisions if same stem exists in subfolders
        rel = wav_path.relative_to(INPUT_DIR)
        safe_stem = "_".join(rel.with_suffix("").parts)

        for i in range(num_chunks):
            start = i * frames_per_chunk
            end = start + frames_per_chunk
            chunk = data[start:end]

            out_path = OUTPUT_DIR / f"{safe_stem}_chunk_{i:06d}.wav"

            # Write as 16-bit PCM (common for ML/audio pipelines)
            sf.write(str(out_path), chunk, sr, subtype="PCM_16")
            chunk_id += 1

    except Exception as e:
        print("Skipping", wav_path, "because", repr(e))
        continue

print("Done. Total chunks:", chunk_id)

