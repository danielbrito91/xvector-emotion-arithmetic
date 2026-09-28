# Training-Free Cross-Lingual Emotion Control in LM-TTS via Speaker-Channel Editing

Code for the emotion-control study on Qwen3-TTS-12Hz-1.7B-Base by Daniel O. Brito, Sidney E. Leal, and Arnaldo Candido Junior. This release includes the method and part of the analyses from the manuscript revised in September 2026. The remaining experimental recipes are still being prepared for release.

The method extracts emotion directions from English recordings and applies them to the speaker embedding of a target voice while retaining the tokens from its neutral reference:

```python
tau = mean_s(centroid(s, emotion) - centroid(s, neutral))
x_edited = x_target_neutral + alpha * tau
```

The neutral/angry factorial shows a larger mean effect from the speaker embedding, with contributions from both tokens and embedding and a reversal in their ordering for one of four speakers. The directions transfer emotion control to English and Brazilian Portuguese target voices. Evaluation covers ESD, emoUERJ, CREMA-D, and VERBO; the latter two are exploratory evaluations with alpha selected using English data only. The listening test uses speaker-calibrated strengths fixed before data collection.

## Installation

Requires Python 3.12+, `uv`, and `ffmpeg`. Synthesis uses PyTorch/CUDA. Python dependencies are locked in `uv.lock`.

```bash
uv sync --locked
uv run hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./Qwen3-TTS-12Hz-1.7B-Base
uv run hf download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ./Qwen3-TTS-Tokenizer-12Hz
export PYTHONPATH=.
```

The metrics download their own models on first use: emotion2vec+ large, SenseVoiceSmall, WavLM-base-plus-sv, and Whisper-large-v3. Historical scripts also compute UTMOSv2, which is not included in the current manuscript tables.

## Inference

The six emotion directions in `data/tau/` are included. Each `.pt` file contains a 2,048-dimensional vector, centroids, and extraction metadata.

```bash
uv run python scripts/deploy/emotionize_audio.py \
  --input data/reference.wav --output data/angry.wav \
  --emotion angry --tau-variant avg4spk --alpha 2.5 \
  --ref-text "Transcrição da referência." --text "Texto a sintetizar."
```

Without `--ref-text`, Whisper transcribes the reference. Without `--text`, the system synthesizes that transcription. The English-only selection uses `avg4spk`, with alpha 2.5 for angry/sad and 1.5 for happy. These are operating points from the study, with no guarantee of uniform intensity across voices.

## Data and reproduction

ESD provides the directions, using 50 recordings per emotion and speaker: 0011, 0014, 0017, and 0020. The single-source comparator uses 0017. English evaluation uses 0013 and 0019; Brazilian Portuguese evaluation uses m03, m04, and w04 from emoUERJ. CREMA-D and VERBO add 12 speakers per corpus.

```bash
export DATA_ROOT="$HOME/data/processed"
export ESD_ROOT="$DATA_ROOT/esd_24k"
export EMOUERJ_ROOT="$DATA_ROOT/emouerj_24k"
export ESD_RAW_ROOT="$HOME/data/external/Emotional Speech Dataset (ESD)/Emotion Speech Dataset"
```

ESD uses speaker/emotion subdirectories; emoUERJ uses files such as `m03a01.wav` at the root of the resampled corpus. `scripts/data/resample_esd.py --help` describes ESD preparation. Corpora, synthesized audio, checkpoints, and individual MOS responses are excluded from Git.

| Result | Available in this release |
| --- | --- |
| Inference and direction extraction | `scripts/deploy/emotionize_audio.py`, `scripts/repro/extract_xvec_tau.py` |
| ESD/emoUERJ sweeps and Table 2 | `run_en2en_sweep.py`, `run_ptbr_sweep.py`, `analyze_gap4_eca_esim.py`, `analyze_icassp_revision.py` |
| Tokens × embedding factorial, Table 1 | Analysis of existing scores in `analyze_icassp_revision.py`; generation of the four conditions is pending |
| CREMA-D/VERBO, Table 2 | Analysis of existing manifests and scores in `analyze_cross_corpus.py`; corpus acquisition and synthesis are pending |
| EMOS/NMOS, Table 3 | `scripts/mos_power/analyze_mos.py`; anonymized public inputs are pending |
| Matched-norm controls | Recipe pending in this release |

Paths shown without a directory are under `scripts/repro/`. The [reproduction guide](scripts/repro/README.md) describes inputs and commands. This partial release cannot reproduce every table from the corpora alone. Previously published explorations in `scripts/elimination/`, `src/lora.py`, and `third_party/qwen/` remain available.

## Repository layout

```text
src/                    vector arithmetic, synthesis, data, and metrics
scripts/data/           corpus preparation
scripts/deploy/         reference-based inference
scripts/repro/          experiments, analyses, and configurations
scripts/mos_power/      listening-test analysis
third_party/qwen/       Qwen3-TTS fine-tuning adaptation
data/tau/               precomputed emotion directions
```

## Citation

For citation, use the current metadata at [arXiv:2606.05367](https://arxiv.org/abs/2606.05367).
