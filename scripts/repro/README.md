# Reproduction and analyses

Run these commands from the repository root with `PYTHONPATH=.` and the environment specified by `uv.lock`. Corpus paths are defined in the [main README](../../README.md). This release includes inference, direction extraction, ESD/emoUERJ sweeps, and analyses of existing scores. Public recipes for factorial generation, cross-corpus synthesis, and matched-norm controls, as well as anonymized MOS inputs, are still pending.

## Directions and sweeps

The `.pt` files contain tau, centroids, configuration, and statistics. To reconstruct them from ESD:

```bash
PYTHONPATH=. uv run python scripts/repro/extract_xvec_tau.py \
  --esd_dir "$ESD_ROOT" --speakers 0011 0014 0017 0020 \
  --emotions Angry Happy Sad --n_pairs 50 --output_dir data/tau_recomputed
PYTHONPATH=. uv run python scripts/repro/extract_xvec_tau.py \
  --esd_dir "$ESD_ROOT" --speakers 0017 \
  --emotions Angry Happy Sad --n_pairs 50 --output_dir data/tau_recomputed
PYTHONPATH=. uv run python scripts/repro/run_en2en_sweep.py
PYTHONPATH=. uv run python scripts/data/transcribe_emouerj.py
PYTHONPATH=. uv run python scripts/repro/run_ptbr_sweep.py
PYTHONPATH=. uv run python scripts/repro/analyze_gap4_eca_esim.py \
  --summaries en2en_summary.json ptbr_summary.json
PYTHONPATH=. uv run python scripts/repro/analyze_icassp_revision.py --sections objective \
  --output data/experiments/icassp_objective.json
```

The reanalysis uses `data/experiments/gap4_eca_esim.parquet`, produced by the scoring pass, and `results_long.parquet`/`results_long_ptbr.parquet`, produced by the sweeps. Speaker-calibrated alpha maximizes emotion similarity within each evaluated cell. English-only alpha selection uses `avg4spk` and the shared grid `{0, 1, 1.5, 2, 2.5}`, transferring angry=2.5, happy=1.5, and sad=2.5 to Portuguese. The tests verify that this selection does not depend on Portuguese scores.

The `--selftest` option of `analyze_gap4_eca_esim.py` computes natural ECA and similarity references. `compute_gt_ceiling.py` and `compute_gt_ceiling_ptbr.py` also evaluate identity, WER, and UTMOS. UTMOS remains in historical outputs but is not included in the current tables. The sweeps accept `--combination convex` for interpolation toward the emotional centroid, including full replacement at beta=1.

## Conditioning factorial

The analysis takes 1,600 records: four speakers × ten references × ten texts × four conditions. In the `balanced` layout, each `scores.jsonl` row contains `key`, `speaker` as a string, `ref`, `utt`, `condition`, `wav_path`, `sha256`, `eca_e2v`, `eca_sv`, `duration_s`, `wer_norm`, and `spk_cos_sim_neutral_wavlm`. Audio paths must be accessible to verify their hashes. The conditions are `neutral_baseline`, `full_swap`, `xvec_swap`, and `angry_baseline`.

```bash
PYTHONPATH=. uv run python scripts/repro/analyze_icassp_revision.py \
  --sections factorial --factorial-layout balanced \
  --factorial data/experiments/conditioning_factorial
```

The speakers are 0011, 0013, 0017, and 0019; references use indices 1–10 and texts use 321–330. The bootstrap crosses reference and text while conditioning on the four observed speakers. The sensitivity analysis retains 380 units, excluding reference 6 for 0011 and 0013 because of a lexical discrepancy identified before scoring.

The `historical` reader also accepts the original run: `ar2_token_swap/{results,eca}.jsonl` and `conditioning_factorial_2026-09-06/{scores,sentinel_checks}.jsonl` under `data/experiments/`. It verifies the 12 sentinels used to reuse the three earlier arms. These artifacts are not included with the code. The generator of the four conditions is still outside this release.

## CREMA-D and VERBO

```bash
PYTHONPATH=. uv run python scripts/repro/analyze_cross_corpus.py \
  --run data/experiments/cross_corpus_source_only_v2_exploratory
```

Inputs are `scores.jsonl` and `manifest_synthesis.jsonl`. `cross_corpus_support.py` defines the fields, validates pairing, and computes aggregates. Each corpus has 12 speakers, six texts, and three seeds, yielding 864 outputs and 648 contrasts. Synthesis failures remain in the denominator; the neutral baseline is shared across the three emotion contrasts. The bootstrap crosses speaker and text after averaging over seeds and emotions.

The configuration in `configs/cross_corpus_source_only.json` documents the grid, tau hashes, and English-only alpha selection. In the reported run, the VERBO l5 transcripts for f4/f6 were corrected from “de seu João” to “do seu João” after listening to the pilot. This correction affects 144 manifest rows without changing the reference audio. The evaluation is exploratory. Corpus acquisition and audio generation are not yet included.

## Human listening test

The analysis removes responses from participants who failed Portuguese attention probes, English blocks from participants below CEFR B1, and pages with insufficient duration. Low ratings of natural recordings and extreme scores are retained. Naturalness questions that were not asked remain missing. The condition mapping is joined only after cleaning.

```bash
PYTHONPATH=. uv run python scripts/mos_power/analyze_mos.py \
  --submissions data/experiments/mos_field/submissions.jsonl \
  --manifest data/experiments/mos_stimuli/deploy/manifest.json \
  --mapping data/experiments/mos_stimuli/private/mapping.csv \
  --objective data/experiments/mos_stimuli/private/frozen_objective.csv \
  --out_dir data/experiments/mos_reanalysis --unblind
```

Without `--unblind`, the command produces cleaning data only. EMOS uses a paired Wilcoxon test by utterance and 10,000 bootstrap replicates; NMOS includes a non-inferiority test with a 0.5 MOS-point margin. Listening-test alpha values were calibrated per speaker before data collection, using `single0017` for English and `avg4spk` for Brazilian Portuguese; this evaluation does not perceptually test English-only alpha selection.

The raw data contain personal identifiers and are excluded from Git. An anonymized version of the responses and stimulus artifacts is still needed for public reproduction. The interface is in the [tts-eval repository](https://github.com/danielbrito91/tts-eval).

## Local verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. uv run python scripts/repro/test_cross_corpus_source_only.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. uv run python scripts/repro/test_icassp_analysis.py
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/repro/check_release.py
```

The tests use synthetic data to verify alpha selection, pairing, denominators, and bootstrap behavior. `check_release.py` checks the files listed in `release_files.txt`, their syntax, and local dependencies. None of these commands runs GPU synthesis. The fine-tuning explorations depend on Qwen3-TTS's `finetuning/dataset.py`; see [third_party/qwen/README.md](../../third_party/qwen/README.md).
