import random
import wave
from pathlib import Path

INPUT_DIR = Path("/host_root/cache/a/ssingh/Datasets/Xeno-canto/")
OUTPUT_DIR = Path("/host_root/cache/a/ssingh/Datasets/Chunked-Xeno/")

CHUNK_SECONDS = 10
TARGET_HOURS = 30

TARGET_SECONDS = TARGET_HOURS * 3600

# Just list file paths (cheap)
wav_files = list(INPUT_DIR.rglob("*.wav"))
random.shuffle(wav_files)

total_seconds = 0
chunk_id = 0

while total_seconds < TARGET_SECONDS:
    wav_path = random.choice(wav_files)

    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            nframes = wf.getnframes()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()

            frames_per_chunk = CHUNK_SECONDS * sr
            if nframes < frames_per_chunk:
                continue

            max_start = nframes - frames_per_chunk
            start = random.randint(0, max_start)

            wf.setpos(start)
            audio = wf.readframes(frames_per_chunk)

        out_path = OUTPUT_DIR / f"chunk_{chunk_id:07d}.wav"
        with wave.open(str(out_path), "wb") as out:
            out.setnchannels(channels)
            out.setsampwidth(sampwidth)
            out.setframerate(sr)
            out.writeframes(audio)

        chunk_id += 1
        total_seconds += CHUNK_SECONDS

    except Exception:
        continue

print(f"Done: {total_seconds/3600:.2f} hours, {chunk_id} chunks")

