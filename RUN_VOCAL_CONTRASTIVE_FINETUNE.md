# Vocal contrastive finetune

Self-supervised **contrastive** finetune on vocal regions from CSV labels. Same-vocalization frames are pulled together; non-vocal and cross-file frames are pushed apart. Output checkpoint works with `animal2vec_inference.py` and the UI clustering pipeline unchanged.

## What it does

- Loads a pretrained `data2vec_multi` checkpoint (student only).
- Reads wav paths from a Fairseq manifest (`train_0.tsv`).
- Uses CSV for vocalization `[start, end)` intervals; class ids are used for optional class-aware sampling and sampling diagnostics.
- Loss: **triplet margin** on cosine distance (no teacher anchor term).
- Negative sampling: **50/50** noise in same clip vs vocal frame from another file in the batch (no same-clip other-vocal negatives).

## Data layout

```
dataset_root/
  wav/.../*.wav
  csv/.../{recording_id}.csv    # or next to wavs
  manifest/
    train_0.tsv
```

Manifest format (first line = root dir):

```
/path/to/dataset_root
relative/path.wav<TAB>num_samples
```

CSV formats supported (`--label-csv-format auto`):

- NIPS indices: `start_idx,end_idx,class_id`
- Seconds + duration: `start_s,duration_s,class_name`
- Audacity export: `Name, Start, Duration, ...`

Segment wavs like `foo_00000s_00005s.wav` use recording CSV `foo.csv` with a 0 s offset applied to intervals.

## Train

From repo root:

```bash
cd Mar-Lab-Animal2vec

python scripts/vocal_contrastive_finetune.py \
  --pretrain-ckpt /path/to/xeno_pretrain.pt \
  --manifest-dir /path/to/manifest \
  --train-subset train_0 \
  --label-csv-dir /path/to/csv \
  --label-csv-format auto \
  --label-index-rate-hz 44100 \
  --save-dir /path/to/vocal_contrastive_runs \
  --batch-size 8 \
  --max-updates 5000 \
  --device cuda
```

Optional: `--freeze-feature-extractor` trains only the shared transformer (safer on small data).
Optional: `--class-aware` uses same-class positives and different-class negatives from **other files in the batch only** (never another vocalization in the same clip).
Optional: `--noise-negative-prob 0.5` controls noise vs other-file negative mix (try `0.25` in a later experiment only).

Logs: `pos_dist` should fall below `neg_dist`; `loss_triplet` should decrease. Each log interval reports sampling rates (`neg_diff`, `neg_same`, `pos_same`) and `sec/up`.

After training, `{save_dir}/training_summary.json` contains args, sampling config, runtime, and aggregated per-class sampling stats. `{save_dir}/training_stats.jsonl` has per-log-interval rows.

## Embeddings (same as pretrain)

```bash
python animal2vec_inference.py \
  --path /path/to/wavs \
  --model-path /path/to/vocal_contrastive_runs/checkpoint_last.pt \
  --out-path /path/to/embeddings \
  --write-embeddings True \
  --unique-values "[]" \
  --sample-rate 24000
```

## UI clustering

1. Open `UI/inference_app.py` (Streamlit).
2. Point embedding directory at `--out-path` from inference.
3. Run k-means or HDBSCAN on frame embeddings.

Compare against baseline: same wavs with the **pretrain** checkpoint only.

## Files

| File | Role |
|------|------|
| `nn/vocal_contrastive.py` | Dataset, CSV parsing, triplet loss |
| `scripts/vocal_contrastive_finetune.py` | Training entry point |
| `configs/vocal_contrastive_finetune.yaml` | Reference hyperparameters |

No changes to `data2vec2.py`, `animal2vec_train.py`, inference, or UI.
