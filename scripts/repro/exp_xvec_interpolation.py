"""X-vector interpolation / task-arithmetic experiment for Qwen3-TTS.

Two operating paths:

1) Single-pair (legacy, --esd_dir): build τ on the fly from one ESD parallel pair
   (utterance_id Neutral vs +offset Angry). Backwards compatible.

2) τ-file (Phase A, --tau_file): load a pre-computed multi-utt / multi-spk τ artifact
   from scripts/repro/extract_xvec_tau.py. Skips the in-script angry-neutral diff.

Modes:
  - interp:     xvec = (1-α)·neutral + α·emotion (centroids when τ-file)
  - task_arith: xvec = xvec(ref_audio) + α·τ  (--ref_audio is target speaker's neutral)

Phase B metrics (when --target_emo_audio is set, the EN→EN v2 metric set):
  emo_cos_sim_gt, xvec_cos_sim_gt, spk_cos_sim_neutral, wer  vs the target speaker's own GT.

Usage:
  # Legacy single-pair task arithmetic (unchanged):
  PYTHONPATH=. uv run python scripts/exp_xvec_interpolation.py \\
      --esd_dir /home/daniel/data/processed/esd_24k/0017 \\
      --output_dir data/experiments/xvec_task_arith \\
      --mode task_arith \\
      --ref_audio data/ref/dani-neutro.wav \\
      --ref_text "..." --alpha 0.0 0.5 1.0 1.5 2.0

  # Phase A: τ-file driven, EN→EN target speaker 0013, angry, avg4spk τ:
  PYTHONPATH=. uv run python scripts/exp_xvec_interpolation.py \\
      --tau_file data/tau/tau_angry_avg4spk.pt \\
      --mode task_arith \\
      --ref_audio /home/daniel/data/processed/esd_24k/0013/Neutral/0013_000010.wav \\
      --ref_text "A nauseous draught." \\
      --target_emo_audio /home/daniel/data/processed/esd_24k/0013/Angry/0013_000360.wav \\
      --target_emo_label angry \\
      --text "A nauseous draught." \\
      --output_dir data/experiments/xvec_en2en_0013_avg4spk_angry \\
      --alpha 0.5 1.0 1.5 2.0 2.5
"""

import argparse
import json
import os
from pathlib import Path

import torch

from src.xvec import (
    compute_metrics_target_gt,
    cos_flat,
    extract_xvec,
    load_tau_artifact,
    load_tts,
    match_shape,
    save_wav,
    synthesize_with_xvec,
)


UTTERANCE_ID_OFFSET = {"Angry": 350, "Happy": 700, "Sad": 1050, "Surprise": 1400}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--esd_dir", default=None,
                   help="ESD speaker dir (legacy single-pair path). Optional if --tau_file is set.")
    p.add_argument("--tau_file", default=None,
                   help="Pre-computed τ artifact (.pt) from extract_xvec_tau.py")
    p.add_argument("--output_dir", default="data/experiments/xvec_interp")
    p.add_argument("--model_path", default="./Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--text", default="Estou muito brabo com essa situação toda!")
    p.add_argument("--language", default="Auto")
    p.add_argument("--alpha", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--utterance_id", default="0017_000001",
                   help="Legacy: neutral utt id for in-script parallel pair (single-pair path)")
    p.add_argument("--target_emotion", default="Angry",
                   choices=list(UTTERANCE_ID_OFFSET.keys()),
                   help="Legacy: target emotion for in-script parallel pair (single-pair path)")
    p.add_argument("--mode", choices=["interp", "task_arith"], default="interp")
    p.add_argument("--ref_audio", default=None,
                   help="ICL ref audio. task_arith: also the base x-vector source.")
    p.add_argument("--ref_text", default=None)
    p.add_argument("--target_emo_audio", default=None,
                   help="Target speaker's GT emotional recording (Phase B v2 metrics)")
    p.add_argument("--target_emo_label", default=None,
                   help="Tag for the GT emotion (e.g. 'angry'), recorded in results.json")
    p.add_argument("--wer_language", default="en")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--skip_metrics", action="store_true")
    p.add_argument("--skip_controls", action="store_true",
                   help="Skip the ctrl_neutral / ctrl_angry control generations")
    return p.parse_args()


def parse_transcripts(esd_dir: str) -> dict[str, tuple[str, str]]:
    speaker_id = Path(esd_dir).name
    out: dict[str, tuple[str, str]] = {}
    with open(os.path.join(esd_dir, f"{speaker_id}.txt"), encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                out[parts[0]] = (parts[1], parts[2])
    return out


def find_wav(esd_dir: str, emotion: str, utt_id: str) -> str | None:
    candidates = list(Path(esd_dir, emotion).rglob(f"{utt_id}.wav"))
    return str(candidates[0]) if candidates else None


def resolve_tau_and_base(args, tts) -> dict:
    """Return dict with tau, neutral, optional emotion centroid, and provenance."""
    if args.tau_file:
        art = load_tau_artifact(args.tau_file)
        return {
            "source": "tau_file",
            "tau_file": args.tau_file,
            "tau": art["tau"],
            "neutral_centroid": art["neutral_centroid"],
            "emotion_centroid": art["emotion_centroid"],
            "tau_config": art["config"],
            "tau_stats": art["stats"],
        }

    if not args.esd_dir:
        raise SystemExit("Either --tau_file or --esd_dir is required.")
    if args.target_emotion not in UTTERANCE_ID_OFFSET:
        raise SystemExit(f"--target_emotion must be one of {list(UTTERANCE_ID_OFFSET.keys())}")

    transcripts = parse_transcripts(args.esd_dir)
    speaker_id = Path(args.esd_dir).name
    nid = args.utterance_id
    num = int(nid.split("_")[1])
    offset = UTTERANCE_ID_OFFSET[args.target_emotion]
    aid = f"{speaker_id}_{num + offset:06d}"

    neutral_wav = find_wav(args.esd_dir, "Neutral", nid)
    emo_wav = find_wav(args.esd_dir, args.target_emotion, aid)
    if not neutral_wav or not emo_wav:
        raise FileNotFoundError(f"Missing wav: neutral={neutral_wav}, emo={emo_wav}")

    neutral_xvec = extract_xvec(tts, neutral_wav).detach().cpu()
    emo_xvec = extract_xvec(tts, emo_wav).detach().cpu()
    tau = emo_xvec - neutral_xvec

    pair_text = transcripts.get(nid, (None,))[0]
    return {
        "source": "single_pair",
        "tau": tau,
        "neutral_centroid": neutral_xvec,
        "emotion_centroid": emo_xvec,
        "neutral_wav": neutral_wav,
        "emo_wav": emo_wav,
        "pair_text": pair_text,
        "pair_neutral_id": nid,
        "pair_emo_id": aid,
        "pair_emotion": args.target_emotion,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    tts = load_tts(args.model_path)

    info = resolve_tau_and_base(args, tts)
    tau = info["tau"]
    neutral_centroid = info["neutral_centroid"]
    emotion_centroid = info["emotion_centroid"]

    print(f"\nτ source: {info['source']}")
    print(f"  ‖τ‖              = {tau.norm().item():.4f}")
    print(f"  ‖neutral_c‖      = {neutral_centroid.norm().item():.4f}")
    print(f"  ‖emo_c‖          = {emotion_centroid.norm().item():.4f}")
    print(f"  cos(neutral,emo) = {cos_flat(neutral_centroid, emotion_centroid):.4f}")

    if args.mode == "task_arith":
        if not args.ref_audio:
            raise SystemExit("--ref_audio required in task_arith mode")
        base_xvec = extract_xvec(tts, args.ref_audio)
        tau_dev = match_shape(tau, base_xvec)
        print(f"  base xvec (ref_audio) ‖x‖ = {base_xvec.norm().item():.4f}")
        print(f"  cos(base, neutral_c)      = {cos_flat(base_xvec, neutral_centroid):.4f}")
    else:
        base_xvec = None
        tau_dev = None

    if args.ref_audio:
        if not args.ref_text:
            raise SystemExit("--ref_text required when --ref_audio is set")
        icl_audio, icl_text = args.ref_audio, args.ref_text
    elif info["source"] == "single_pair":
        icl_audio, icl_text = info["neutral_wav"], info["pair_text"]
    else:
        raise SystemExit("--ref_audio is required when using --tau_file")

    results: dict = {
        "config": {
            "mode": args.mode,
            "tau_source": info["source"],
            "tau_file": args.tau_file,
            "tau_config": info.get("tau_config"),
            "tau_stats": info.get("tau_stats"),
            "icl_ref_audio": icl_audio,
            "icl_ref_text": icl_text,
            "ref_audio": args.ref_audio,
            "target_emo_audio": args.target_emo_audio,
            "target_emo_label": args.target_emo_label,
            "text": args.text,
            "alphas": args.alpha,
            "model_path": args.model_path,
            "wer_language": args.wer_language,
        },
        "xvec_stats": {
            "tau_norm": tau.norm().item(),
            "neutral_centroid_norm": neutral_centroid.norm().item(),
            "emotion_centroid_norm": emotion_centroid.norm().item(),
            "cos_neutral_emotion": cos_flat(neutral_centroid, emotion_centroid),
            "base_xvec_norm": base_xvec.norm().item() if base_xvec is not None else None,
        },
        "conditions": {},
    }
    if info["source"] == "single_pair":
        results["config"].update({
            "pair_neutral_id": info["pair_neutral_id"],
            "pair_emo_id": info["pair_emo_id"],
            "pair_emotion": info["pair_emotion"],
            "pair_text": info["pair_text"],
        })

    use_v2_metrics = bool(args.target_emo_audio)

    def _metrics(out_path: str) -> dict:
        if args.skip_metrics:
            return {}
        if use_v2_metrics:
            return compute_metrics_target_gt(
                tts,
                synth_wav=out_path,
                gt_emo_audio=args.target_emo_audio,
                ref_neutral_audio=args.ref_audio or info.get("neutral_wav"),
                target_text=args.text,
                wer_language=args.wer_language,
            )
        # legacy 3-metric path (only valid in single_pair branch)
        from src.metrics.emotion import get_emotion
        from src.metrics.speaker import get_speaker_embedding

        emo_synth = torch.tensor(get_emotion(out_path)["embedding"])
        emo_emo = torch.tensor(get_emotion(info["emo_wav"])["embedding"])
        emo_neu = torch.tensor(get_emotion(info["neutral_wav"])["embedding"])
        spk_synth = get_speaker_embedding(out_path)
        spk_neu = get_speaker_embedding(info["neutral_wav"])
        cos = torch.nn.functional.cosine_similarity
        return {
            "emo_sim_emo": cos(emo_synth, emo_emo, dim=0).item(),
            "emo_sim_neutral": cos(emo_synth, emo_neu, dim=0).item(),
            "spk_sim_neutral": cos(spk_synth, spk_neu).item(),
        }

    if not args.skip_controls and info["source"] == "single_pair":
        for ctrl_name, ctrl_wav, ctrl_text in [
            ("ctrl_neutral", info["neutral_wav"], info["pair_text"]),
            (f"ctrl_{info['pair_emotion'].lower()}", info["emo_wav"], info["pair_text"]),
        ]:
            print(f"\n--- {ctrl_name} ---")
            prompt_xvec = extract_xvec(tts, ctrl_wav)
            wav, sr = synthesize_with_xvec(
                tts, args.text, ctrl_wav, ctrl_text, prompt_xvec,
                language=args.language, max_new_tokens=args.max_new_tokens,
            )
            if wav is None:
                print("  ERROR: no audio")
                continue
            out_path = os.path.join(args.output_dir, f"{ctrl_name}.wav")
            duration = save_wav(wav, sr, out_path)
            print(f"  Saved: {out_path} ({duration:.1f}s)")
            cond = {"wav_path": out_path, "duration_s": round(duration, 2)}
            try:
                cond["metrics"] = _metrics(out_path)
            except Exception as e:
                print(f"  WARN: metrics failed: {e}")
            results["conditions"][ctrl_name] = cond

    print(f"\n{'=' * 70}\nα SWEEP\n{'=' * 70}")
    for alpha in args.alpha:
        cond_name = f"alpha_{alpha:.2f}"
        if args.mode == "task_arith":
            hybrid = base_xvec + alpha * tau_dev
        else:
            nc = match_shape(neutral_centroid, base_xvec) if base_xvec is not None else neutral_centroid
            ec = match_shape(emotion_centroid, base_xvec) if base_xvec is not None else emotion_centroid
            hybrid = (1 - alpha) * nc + alpha * ec

        print(f"\n--- α={alpha:.2f} | ‖hybrid‖={hybrid.norm().item():.4f} ---")
        wav, sr = synthesize_with_xvec(
            tts, args.text, icl_audio, icl_text, hybrid,
            language=args.language, max_new_tokens=args.max_new_tokens,
        )
        if wav is None:
            print("  ERROR: no audio")
            results["conditions"][cond_name] = {"error": "no audio", "alpha": alpha}
            continue
        out_path = os.path.join(args.output_dir, f"{cond_name}.wav")
        duration = save_wav(wav, sr, out_path)
        print(f"  Saved: {out_path} ({duration:.1f}s)")

        cond = {
            "alpha": alpha,
            "wav_path": out_path,
            "duration_s": round(duration, 2),
            "hybrid_xvec_norm": round(hybrid.norm().item(), 4),
        }
        try:
            cond["metrics"] = _metrics(out_path)
            for k, v in cond["metrics"].items():
                if isinstance(v, float):
                    print(f"  {k:<22} = {v:.4f}")
        except Exception as e:
            print(f"  WARN: metrics failed: {e}")
        results["conditions"][cond_name] = cond

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    rows = []
    for name, cond in results["conditions"].items():
        m = cond.get("metrics", {})
        rows.append((name, cond.get("duration_s", float("nan")), m))
    if use_v2_metrics:
        print(f"{'cond':<20} {'dur':>6}  {'emo_gt':>8} {'xvec_gt':>8} {'spk_neu':>8} {'wer':>6}")
        for name, dur, m in rows:
            print(f"  {name:<18} {dur:>5.1f}s  "
                  f"{m.get('emo_cos_sim_gt', float('nan')):>8.4f} "
                  f"{m.get('xvec_cos_sim_gt', float('nan')):>8.4f} "
                  f"{m.get('spk_cos_sim_neutral', float('nan')):>8.4f} "
                  f"{m.get('wer', float('nan')):>6.3f}")
    else:
        print(f"{'cond':<20} {'dur':>6}  {'emo_emo':>10} {'emo_neu':>10} {'spk_neu':>10}")
        for name, dur, m in rows:
            print(f"  {name:<18} {dur:>5.1f}s  "
                  f"{m.get('emo_sim_emo', float('nan')):>10.4f} "
                  f"{m.get('emo_sim_neutral', float('nan')):>10.4f} "
                  f"{m.get('spk_sim_neutral', float('nan')):>10.4f}")
    print("\nDone!")


if __name__ == "__main__":
    main()
