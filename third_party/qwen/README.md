# third_party/qwen

Vendored, lightly-patched code from [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
(@ commit `1ab0dd7`), redistributed under its original Apache-2.0 license.

## `sft_12hz.py`

Near-verbatim copy of upstream `finetuning/sft_12hz.py`, used for the full
fine-tuning step that produces the per-emotion weights from which the task
vectors (`τ`) are derived. We ship only the resulting `τ` embeddings, not the
fine-tuned checkpoints — this script is included so the procedure is auditable
and reproducible.

Changes vs upstream (see the header comment in the file for line-level detail):

1. **Bug fix** — hidden-state / `codec_mask` alignment. The talker is fed
   `input_embeddings[:, :-1, :]`, so its hidden states align with positions
   `[:-1]`; upstream selected them with `codec_mask[:, 1:]` (off-by-one). This
   is the only behavioural change, and the only one we'd suggest upstreaming.
2. `attn_implementation` exposed as `--attn` (default `flash_attention_2`,
   matching upstream; use `--attn sdpa` without FA2).
3. Behaviour-neutral conveniences: auto `snapshot_download` for HF repo ids and
   a tensorboard `project_dir`.

Upstream's sub-talker loss weight (`1.0`) is preserved.

### Running

The script depends on upstream's unmodified `finetuning/dataset.py` and the
installed `qwen_tts` package. Run it from an environment where both are
importable, e.g.:

```bash
PYTHONPATH=/path/to/Qwen3-TTS/finetuning \
  uv run python third_party/qwen/sft_12hz.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --train_jsonl <emotion>_coded.jsonl \
    --output_model_path output_<emotion> \
    --speaker_name <speaker>_<emotion> \
    --batch_size 2 --lr 2e-6 --num_epochs 3
```
