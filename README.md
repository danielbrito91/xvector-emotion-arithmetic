# Task-Vector Arithmetic for Emotional Expressivity Control in LM-TTS

[![arXiv](https://img.shields.io/badge/arXiv-2606.05367-b31b1b.svg)](https://arxiv.org/abs/2606.05367)

Reproducibility code for the paper *"Task-Vector Arithmetic for Emotional
Expressivity Control in Language-Model-Based Text-to-Speech"*
([arXiv:2606.05367](https://arxiv.org/abs/2606.05367)).

We localize the dominant carrier of emotional prosody in an LM-TTS
(Qwen3-TTS-12Hz-1.7B-Base) through a four-step elimination study, and show that
**cross-speaker and cross-lingual emotional transfer is achievable with simple,
training-free vector arithmetic over the speaker x-vector**:

```
tau      = mean_i xvec(speaker_i, emotion) - mean_i xvec(speaker_i, neutral)
xvec_new = xvec(target, neutral) + alpha * tau
```

## Citation

If you use this code, please cite:

```bibtex
@misc{brito2026taskvector,
  title         = {Task-Vector Arithmetic for Emotional Expressivity Control in Language-Model-Based Text-to-Speech},
  author        = {Brito, Daniel Oliveira de and Candido Junior, Arnaldo},
  year          = {2026},
  eprint        = {2606.05367},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  url           = {https://arxiv.org/abs/2606.05367}
}
```

## Elimination study (where is the emotion?)

| Step | Operand | Intervention | Transfers emotion? |
|------|---------|--------------|--------------------|
| 1 | Backbone weights | Task vector via full FT / LoRA | No |
| 2 | Codec embeddings (continuous) | Per-codebook centroid arithmetic | No (-> noise) |
| 3 | Discrete codec tokens | `full_swap` (*angry* tokens + *neutral* x-vec) | No (coherent, but neutral) |
| **4** | **x-vector (ECAPA-TDNN)** | **Centroid arithmetic (Eq. 1)** | **Yes** |

## Setup

```bash
uv sync
```

Download the model and tokenizer (~3.5 GB):

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base   --local-dir ./Qwen3-TTS-12Hz-1.7B-Base
HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download Qwen/Qwen3-TTS-Tokenizer-12Hz   --local-dir ./Qwen3-TTS-Tokenizer-12Hz
```

The internal experiment scripts (`scripts/*.py`) expect `PYTHONPATH=.`.

## Data

- **ESD** (Zhou et al., 2022) — English. tau source + English held-out targets.
  Speakers `{0011, 0014, 0017, 0020}` (tau), `{0013, 0019}` (held-out).
- **emoUERJ** ([Germano et al., 2021](https://doi.org/10.5281/zenodo.5427549)) —
  Brazilian Portuguese. cross-lingual targets with ground truth `{m03, m04, w04}`.

Both are resampled to 24 kHz mono. Layout expected per dataset root:
`<root>/<speaker>/<Emotion>/<utt_id>.wav`. Resample ESD with
`scripts/resample_esd.py` (see `--help`).

Dataset roots are resolved via environment variables (no hard-coded paths in the
code). Set them before running the pipeline:

```bash
export DATA_ROOT="$HOME/data/processed"   # parent directory of the resampled datasets
# optional — override each dataset individually:
export ESD_ROOT="$DATA_ROOT/esd_24k"
export EMOUERJ_ROOT="$DATA_ROOT/emouerj_24k"
```

Defaults are `$DATA_ROOT/esd_24k` and `$DATA_ROOT/emouerj_24k`, with `DATA_ROOT`
falling back to `~/data/processed` when unset. The script flags (`--esd_root`,
`--esd_dir`, `--emouerj_root`) still take precedence over the environment
variables.

### Pre-computed tau vectors

The emotional direction vectors used at deploy time ship with the repository in
`data/tau/*.pt` (~450 KB total), so **you do not need to run the extraction**
(step 1 below) to use `scripts/deploy/emotionize_audio.py`. Each file is a tensor
$\tau \in \mathbb{R}^{2048}$ (difference of x-vector centroids), derived from ESD.

## Reproducing the paper

The commands below use the roots defined in `$DATA_ROOT` (see the **Data**
section). You can override case by case with the script flags.

### 1. Extract the tau directions (Eq. 1)

```bash
# multi-speaker tau (avg4spk) over {0011,0014,0017,0020}
PYTHONPATH=. uv run python scripts/repro/extract_xvec_tau.py \
    --esd_dir /path/to/esd_24k --speakers 0011 0014 0017 0020 \
    --emotions Angry Happy Sad --n_pairs 50 --output_dir data/tau

# reference single-speaker tau (0017)
PYTHONPATH=. uv run python scripts/repro/extract_xvec_tau.py \
    --esd_dir /path/to/esd_24k --speakers 0017 \
    --emotions Angry Happy Sad --n_pairs 50 --output_dir data/tau
```

Produces `data/tau/tau_{angry,happy,sad}_{avg4spk,single0017}.pt`.

### 2. Geometry of tau (Section 4.2)

```bash
PYTHONPATH=. uv run python scripts/repro/analyze_tau_directions.py
```

### 3. Cross-speaker EN -> EN sweep (Table 2, Figure 3)

```bash
PYTHONPATH=. uv run python scripts/repro/run_en2en_sweep.py
PYTHONPATH=. uv run python scripts/repro/run_en2en_sweep.py --n_sentences 5  # quick smoke test
```

### 4. Cross-lingual EN -> PT-BR sweep (Table 3, Figures 4-5)

```bash
PYTHONPATH=. uv run python scripts/data/transcribe_emouerj.py   # once, generates the transcriptions
PYTHONPATH=. uv run python scripts/repro/run_ptbr_sweep.py
```

### 5. Ground-truth ceilings + UTMOS

```bash
PYTHONPATH=. uv run python scripts/repro/compute_gt_ceiling.py        # EN
PYTHONPATH=. uv run python scripts/repro/compute_gt_ceiling_ptbr.py   # PT-BR
PYTHONPATH=. uv run python scripts/repro/add_utmos_to_results.py      # UTMOS post-hoc, no re-synthesis
```

### Elimination study (Steps 1-3, negative results; Table 1)

```bash
PYTHONPATH=. uv run python scripts/elimination/exp01.py                    # Step 1 baseline
PYTHONPATH=. uv run python src/lora.py                                     # Step 1 LoRA fine-tuning
PYTHONPATH=. uv run python scripts/elimination/infer_lora.py               # Step 1 LoRA inference
PYTHONPATH=. uv run python scripts/elimination/extract_emotion_centroids.py # Step 2 codec-embedding tau
PYTHONPATH=. uv run python scripts/elimination/infer_emotion_vector.py     # Step 2 injection
PYTHONPATH=. uv run python scripts/elimination/exp_token_swap.py           # Step 3 discrete token swap
```

## Inference with x-vectors

### Quick single experiment (Step 4)

```bash
# Same-speaker interpolation
PYTHONPATH=. uv run python scripts/repro/exp_xvec_interpolation.py \
    --esd_dir /path/to/esd_24k/0017 --output_dir data/experiments/xvec_interp \
    --model_path ./Qwen3-TTS-12Hz-1.7B-Base --mode interp \
    --alpha 0.0 0.25 0.5 0.75 1.0

# Cross-speaker task arithmetic (target voice + 0017 anger direction)
PYTHONPATH=. uv run python scripts/repro/exp_xvec_interpolation.py \
    --esd_dir /path/to/esd_24k/0017 --output_dir data/experiments/xvec_task_arith \
    --model_path ./Qwen3-TTS-12Hz-1.7B-Base --mode task_arith \
    --ref_audio data/ref/neutral.wav --ref_text "Transcript of the reference audio" \
    --alpha 0.0 0.5 1.0 1.5 2.0
```

### Emotionize any audio (deploy)

Once `data/tau/*.pt` exists, apply an emotion to **any** input audio
(wav/opus/mp3/m4a) without retraining:

```bash
# Clone the input words, but angry (defaults: avg4spk, alpha=2.5)
uv run python scripts/deploy/emotionize_audio.py --input data/zap.opus --output data/angry.wav

# Same voice, saying something new, angry
uv run python scripts/deploy/emotionize_audio.py --input data/zap.opus --output data/new.wav \
    --text "Cara, eu nao acredito que voce fez isso de novo!"

# Override emotion / variant / intensity
uv run python scripts/deploy/emotionize_audio.py --input data/zap.opus --output data/happy.wav \
    --emotion happy --tau-variant avg4spk --alpha 2.0
```

Programmatic API (Receive an Object, Return an Object):

```python
from src.emotionize import emotionize_audio

result = emotionize_audio(
    base_audio="data/zap.opus",
    output_path="data/angry.wav",
    text="Cara, eu nao acredito que voce fez isso de novo!",
    emotion="angry", tau_variant="avg4spk", alpha=2.5,
)
print(result.as_dict())
```

**Defaults and caveats:** `tau_variant=avg4spk` preserves identity better than
`single0017`; the useful `alpha` range is `[1.5, 2.5]` (do not extrapolate beyond
2.5 — you get *over-projection*, not "more emotion"). Emotion strength is
asymmetric (`angry` > `happy` > `sad`). Use >= 4 s of synthesis text and trim long
references to 3-6 s (`--ref-start`, `--ref-duration`) to avoid prosody leakage via
ICL.

## Project structure

Scripts depend on the `src/` library; they never import one another.

```
src/                      # the library
  metrics/                # objective metrics
    emotion.py            #   EECS  (emotion2vec cosine)
    speaker.py            #   SECS_W (independent WavLM x-vector)
    asr.py                #   WER   (Whisper-large-v3 + jiwer)
    utmos.py              #   naturalness UTMOSv2
  xvec.py                 # x-vector arithmetic + synthesis (core)
  sweep.py                # alpha-sweep aggregation + markdown tables
  emouerj.py              # emoUERJ dataset layout (PT-BR)
  emotionize.py           # deploy: any audio -> emotional clone
  lora.py                 # LoRA fine-tuning (Step 1)

scripts/                  # thin CLI entrypoints
  repro/                  # paper pipeline
    extract_xvec_tau.py        # tau extraction (Eq. 1)
    exp_xvec_interpolation.py  # single Step 4 experiment
    run_en2en_sweep.py         # EN -> EN sweep
    run_ptbr_sweep.py          # EN -> PT-BR sweep
    analyze_tau_directions.py  # tau geometry (Section 4.2)
    compute_gt_ceiling{,_ptbr}.py  # ground-truth ceilings
    add_utmos_to_results.py    # post-hoc UTMOS
  elimination/            # Steps 1-3 (negative results)
    exp01.py / exp_token_swap.py / extract_emotion_centroids.py /
    infer_emotion_vector.py / infer_lora.py
  data/                   # data preparation
    prepare_esd_jsonl.py / resample_esd.py / transcribe_emouerj.py
  deploy/                 # emotionize_audio.py (CLI)
```

`data/`, `models/`, the downloaded `Qwen3-TTS-*/` weights and notebooks are
gitignored — they are regenerated by the pipeline or downloaded as shown above.
