"""EN→EN cross-speaker α-sweep matrix (Phase B v2 — n=30 ESD official test split).

Matrix: 2 τ variants × 2 target speakers × 3 emotions × n_sentences × |α grid|.
Default sentence set = ESD canonical test split (Zhou 2022): local idx 321–350,
N=30. Same split used by EmoSphere++ 2025 (Tabela III, unseen speakers 0013/0019).

α grids include α=0.0 as Shaheen-style ICL-pure anchor:
  single0017 → {0.0, 0.5, 1.0, 1.5, 2.0, 2.5}   # extended 2026-05-20 to match avg4spk
  avg4spk    → {0.0, 0.5, 1.0, 1.5, 2.0, 2.5}

Output structure:
    data/experiments/xvec_en2en_{target}_{tau_variant}_{emotion}/
        utt_{idx:03d}/alpha_{α}.wav      (one wav per (sentence, α))
        results.json                      (full per-(sentence, α) metrics + agg)
    data/experiments/en2en_summary.{json,md}

Aggregation: mean ± stderr over the N sentences per (target × τ × emotion × α).
Best α per cell picked by argmax_α mean(emo_cos_sim_gt). α=0 reported as baseline.

Usage:
    PYTHONPATH=. uv run python scripts/repro/run_en2en_sweep.py
    PYTHONPATH=. uv run python scripts/repro/run_en2en_sweep.py --summary_only
    PYTHONPATH=. uv run python scripts/repro/run_en2en_sweep.py --n_sentences 5  # quick smoke
"""

import argparse
import json
import os
from pathlib import Path

from src.paths import esd_root

from src.sweep import aggregate_over_sentences, build_summary, render_markdown
from src.xvec import (
    compute_metrics_v3,
    cos_flat,
    extract_xvec,
    load_tau_artifact,
    load_tts,
    match_shape,
    precompute_refs,
    save_wav,
    synthesize_with_xvec,
)


TARGETS = ["0013", "0019"]
TAU_VARIANTS = ["single0017", "avg4spk"]
EMOTIONS = ["angry", "happy", "sad"]
EMO_OFFSET = {"angry": 350, "happy": 700, "sad": 1050}
ALPHA_GRID = {
    "single0017": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
    "avg4spk":    [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
}
DEFAULT_LOCAL_IDX_RANGE = list(range(321, 351))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--esd_root", default=esd_root())
    p.add_argument("--tau_dir", default="data/tau")
    p.add_argument("--output_root", default="data/experiments")
    p.add_argument("--model_path", default="./Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--n_sentences", type=int, default=30,
                   help="How many sentences from the local idx range to use (≤ len(range))")
    p.add_argument("--local_idx_start", type=int, default=321,
                   help="First local idx (default 321 = ESD test split start, Zhou 2022)")
    p.add_argument("--local_idx_end", type=int, default=350,
                   help="Last local idx inclusive (default 350 = ESD test split end)")
    p.add_argument("--language", default="Auto")
    p.add_argument("--wer_language", default="en")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--targets", nargs="+", default=TARGETS)
    p.add_argument("--tau_variants", nargs="+", default=TAU_VARIANTS)
    p.add_argument("--emotions", nargs="+", default=EMOTIONS)
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip cells whose results.json already exists")
    p.add_argument("--summary_only", action="store_true",
                   help="Skip generation; re-aggregate from existing per-cell results.json")
    p.add_argument("--extend_existing", action="store_true",
                   help="If a cell's results.json exists, only synthesize (sentence, α) "
                        "combinations missing relative to the current ALPHA_GRID; merge "
                        "into existing per_sentence and re-aggregate.")
    return p.parse_args()


def neutral_ref_path(esd_root: str, spk: str, idx: int) -> str:
    return f"{esd_root}/{spk}/Neutral/{spk}_{idx:06d}.wav"


def gt_emo_path(esd_root: str, spk: str, emotion: str, idx: int) -> str:
    abs_idx = EMO_OFFSET[emotion] + idx
    sub = emotion.capitalize()
    return f"{esd_root}/{spk}/{sub}/{spk}_{abs_idx:06d}.wav"


def parse_transcripts(esd_root: str, spk: str) -> dict[int, tuple[str, str]]:
    """Return {local_idx: (text, emotion)} for the Neutral block (the canonical text)."""
    path = f"{esd_root}/{spk}/{spk}.txt"
    out: dict[int, tuple[str, str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            utt_id, text, emotion = parts
            if emotion != "Neutral":
                continue
            local_idx = int(utt_id.split("_")[1])
            out[local_idx] = (text, emotion)
    return out


def select_sentences(esd_root: str, spk: str, n: int, idx_start: int, idx_end: int) -> list[tuple[int, str]]:
    transcripts = parse_transcripts(esd_root, spk)
    out = []
    for idx in range(idx_start, idx_end + 1):
        if idx in transcripts:
            out.append((idx, transcripts[idx][0]))
        if len(out) == n:
            break
    return out


def run_cell(
    tts,
    *,
    tau_file: str,
    target_spk: str,
    emotion: str,
    sentences: list[tuple[int, str]],
    alphas: list[float],
    esd_root: str,
    out_dir: str,
    language: str,
    wer_language: str,
    max_new_tokens: int,
    existing_cell: dict | None = None,
) -> dict:
    """Run (or extend) a single cell.

    If `existing_cell` is provided, only (sentence, α) combos missing from it are
    synthesized; the new conditions are merged into the existing per_sentence
    structure and the union grid is re-aggregated.
    """
    art = load_tau_artifact(tau_file)
    tau = art["tau"]
    neutral_centroid = art["neutral_centroid"]
    emotion_centroid = art["emotion_centroid"]

    print(f"  ‖τ‖={tau.norm().item():.3f}  ‖neutral_c‖={neutral_centroid.norm().item():.3f}  "
          f"cos(neutral_c, emo_c)={cos_flat(neutral_centroid, emotion_centroid):.4f}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if existing_cell is None:
        cell: dict = {
            "config": {
                "tau_file": tau_file,
                "tau_config": art["config"],
                "tau_stats": art["stats"],
                "target_speaker": target_spk,
                "emotion": emotion,
                "alphas": alphas,
                "language": language,
                "wer_language": wer_language,
                "n_sentences": len(sentences),
                "sentences": [{"local_idx": i, "text": t} for i, t in sentences],
            },
            "xvec_stats": {
                "tau_norm": tau.norm().item(),
                "neutral_centroid_norm": neutral_centroid.norm().item(),
                "emotion_centroid_norm": emotion_centroid.norm().item(),
                "cos_neutral_emotion_centroid": cos_flat(neutral_centroid, emotion_centroid),
            },
            "per_sentence": {},
        }
        union_alphas = list(alphas)
    else:
        cell = existing_cell
        prior_alphas = cell.get("config", {}).get("alphas") or []
        union_alphas = sorted(set(prior_alphas) | set(alphas))
        cell.setdefault("config", {})["alphas"] = union_alphas
        cell["config"]["sentences"] = [{"local_idx": i, "text": t} for i, t in sentences]
        cell["config"]["n_sentences"] = len(sentences)
        print(f"  [extend] prior α={prior_alphas} → union α={union_alphas}; "
              f"new={sorted(set(union_alphas) - set(prior_alphas))}")

    for sent_idx, text in sentences:
        utt_key = f"utt_{sent_idx:03d}"
        ref_audio = neutral_ref_path(esd_root, target_spk, sent_idx)
        gt_audio = gt_emo_path(esd_root, target_spk, emotion, sent_idx)
        if not os.path.exists(ref_audio) or not os.path.exists(gt_audio):
            print(f"    [skip] {utt_key}: missing {ref_audio} or {gt_audio}")
            continue

        sent_block = cell["per_sentence"].get(utt_key)
        existing_conds = (sent_block or {}).get("conditions") or {}
        alphas_to_run = [
            a for a in alphas
            if f"alpha_{a:.2f}" not in existing_conds
            or "error" in existing_conds[f"alpha_{a:.2f}"]
        ]
        if not alphas_to_run:
            continue

        refs = precompute_refs(tts, ref_audio, gt_audio)
        base_xvec = extract_xvec(tts, ref_audio)
        tau_dev = match_shape(tau, base_xvec)

        if sent_block is None:
            sent_block = {
                "text": text,
                "ref_audio": ref_audio,
                "gt_emo_audio": gt_audio,
                "base_xvec_norm": float(base_xvec.norm().item()),
                "conditions": {},
            }
            cell["per_sentence"][utt_key] = sent_block

        sub_dir = os.path.join(out_dir, utt_key)
        Path(sub_dir).mkdir(parents=True, exist_ok=True)

        for alpha in alphas_to_run:
            cond_name = f"alpha_{alpha:.2f}"
            hybrid = base_xvec + alpha * tau_dev
            wav, sr = synthesize_with_xvec(
                tts, text, ref_audio, text, hybrid,
                language=language, max_new_tokens=max_new_tokens,
            )
            if wav is None:
                sent_block["conditions"][cond_name] = {"alpha": alpha, "error": "no audio"}
                continue
            out_path = os.path.join(sub_dir, f"{cond_name}.wav")
            duration = save_wav(wav, sr, out_path)
            try:
                metrics = compute_metrics_v3(
                    tts, synth_wav=out_path, refs=refs,
                    target_text=text, wer_language=wer_language,
                )
            except Exception as e:
                metrics = {"error": str(e)}
            sent_block["conditions"][cond_name] = {
                "alpha": alpha,
                "wav_path": out_path,
                "duration_s": round(duration, 2),
                "hybrid_xvec_norm": round(float(hybrid.norm().item()), 4),
                "metrics": metrics,
            }

        eco = [
            sent_block["conditions"].get(f"alpha_{a:.2f}", {}).get("metrics", {}).get("emo_cos_sim_gt")
            for a in alphas_to_run
        ]
        eco_str = " ".join(f"{(v if v is not None else float('nan')):+.3f}" for v in eco)
        print(f"    {utt_key}  new α={alphas_to_run}  emo@α: {eco_str}")

    cell["aggregates"] = aggregate_over_sentences(cell, union_alphas)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(cell, f, indent=2, default=str)
    return cell


def main():
    args = parse_args()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    tts = None if args.summary_only else load_tts(args.model_path)
    if tts is None:
        print("[summary_only] skipping model load and generation; re-aggregating only.")
    else:
        print(f"Loading model: {args.model_path}")

    all_results: dict = {}
    for tau_variant in args.tau_variants:
        for emotion in args.emotions:
            tau_file = os.path.join(args.tau_dir, f"tau_{emotion}_{tau_variant}.pt")
            if not os.path.exists(tau_file):
                print(f"!! missing τ artifact: {tau_file} — skipping")
                continue
            for target_spk in args.targets:
                cell_key = f"{target_spk}__{tau_variant}__{emotion}"
                out_dir = os.path.join(
                    args.output_root,
                    f"xvec_en2en_{target_spk}_{tau_variant}_{emotion}",
                )
                results_json = os.path.join(out_dir, "results.json")
                existing_cell = None
                if os.path.exists(results_json):
                    with open(results_json) as f:
                        existing_cell = json.load(f)

                if existing_cell is not None and not args.extend_existing:
                    if args.skip_existing or args.summary_only:
                        all_results[cell_key] = existing_cell
                        print(f"[reuse] {cell_key}")
                        continue
                if args.summary_only:
                    print(f"[summary_only] missing {results_json}; skipping {cell_key}")
                    continue

                alphas = ALPHA_GRID[tau_variant]
                if existing_cell is not None and args.extend_existing:
                    prior = set(existing_cell.get("config", {}).get("alphas") or [])
                    missing = [a for a in alphas if a not in prior]
                    if not missing:
                        all_results[cell_key] = existing_cell
                        print(f"[extend:noop] {cell_key} — all α already present")
                        continue
                    alphas = missing  # only synthesize the missing ones

                sentences = select_sentences(
                    args.esd_root, target_spk, args.n_sentences,
                    args.local_idx_start, args.local_idx_end,
                )
                if not sentences:
                    print(f"!! no sentences resolved for {cell_key}")
                    continue
                print(f"\n{'=' * 70}\n{cell_key}  N={len(sentences)} sents × "
                      f"{len(alphas)} α (target α={alphas})\n{'=' * 70}")
                print(f"  τ_file = {tau_file}")
                cell = run_cell(
                    tts,
                    tau_file=tau_file,
                    target_spk=target_spk,
                    emotion=emotion,
                    sentences=sentences,
                    alphas=alphas,
                    esd_root=args.esd_root,
                    out_dir=out_dir,
                    language=args.language,
                    wer_language=args.wer_language,
                    max_new_tokens=args.max_new_tokens,
                    existing_cell=existing_cell if args.extend_existing else None,
                )
                all_results[cell_key] = cell

    summary = build_summary(all_results)
    summary_json = os.path.join(args.output_root, "en2en_summary.json")
    summary_md = os.path.join(args.output_root, "en2en_summary.md")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(summary_md, "w") as f:
        f.write(render_markdown(summary))
    print(f"\nSaved {summary_json}\nSaved {summary_md}")


if __name__ == "__main__":
    main()
