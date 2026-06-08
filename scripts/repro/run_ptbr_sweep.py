"""PT-BR cross-lingual α-sweep (emoUERJ) — mirror of run_en2en_sweep.py.

τ derivado do ESD inglês (`data/tau/tau_{angry,happy,sad}_{single0017,avg4spk}.pt`)
aplicado cross-lingual a falantes PT-BR do emoUERJ.

Dois designs de célula, unificados num só loop via o modelo de utterance
`(ref_audio, ref_text, gt_audio, synth_text)`, com `synth_text` = texto do alvo:
  - **paired** (m03, m04): mesma frase em neutral→emoção (ref_text == synth_text);
    1 GT por frase. Espelha EN→EN exatamente.
  - **xtext**  (w04): ref = tomada neutral (frase G/T); synth_text = frase-âncora
    da emoção (que tem GT). Zip (ref_take, gt_take) → n sínteses distintas p/ variância.

Saída idêntica em estrutura ao EN→EN:
    data/experiments/xvec_ptbr_{spk}_{tau}_{emotion}/utt_*/alpha_{α}.wav
    data/experiments/xvec_ptbr_{spk}_{tau}_{emotion}/results.json
    data/experiments/ptbr_summary.{json,md}
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src import emouerj
from src.emouerj import (
    EMOUERJ_ROOT,
    PAIRED,
    PAIRED_SPK,
    XTEXT_SPK,
    utterances_for,
    wav_path,
)
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


EMOTIONS = ["angry", "happy", "sad"]
TAU_VARIANTS = ["single0017", "avg4spk"]
ALPHA_GRID = [0.0, 1.0, 1.5, 2.0, 2.5]


def run_cell(
    tts,
    *,
    tau_file: str,
    spk: str,
    emotion: str,
    utts: list[dict],
    alphas: list[float],
    out_dir: str,
    language: str,
    wer_language: str,
    max_new_tokens: int,
) -> dict:
    art = load_tau_artifact(tau_file)
    tau, neu_c, emo_c = art["tau"], art["neutral_centroid"], art["emotion_centroid"]
    print(
        f"  ‖τ‖={tau.norm().item():.3f}  ‖neutral_c‖={neu_c.norm().item():.3f}  "
        f"cos(neutral_c, emo_c)={cos_flat(neu_c, emo_c):.4f}"
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    design = "paired" if spk in PAIRED else "xtext"
    cell: dict = {
        "config": {
            "tau_file": tau_file,
            "tau_config": art["config"],
            "tau_stats": art["stats"],
            "target_speaker": spk,
            "emotion": emotion,
            "alphas": alphas,
            "language": language,
            "wer_language": wer_language,
            "n_sentences": len(utts),
            "design": design,
            "utts": [
                {k: v for k, v in u.items() if k != "frase_code"} | {"frase_code": u["frase_code"]}
                for u in utts
            ],
        },
        "xvec_stats": {
            "tau_norm": tau.norm().item(),
            "neutral_centroid_norm": neu_c.norm().item(),
            "emotion_centroid_norm": emo_c.norm().item(),
            "cos_neutral_emotion_centroid": cos_flat(neu_c, emo_c),
        },
        "per_sentence": {},
    }

    for u in utts:
        utt_key = u["key"]
        ref_audio, gt_audio = wav_path(u["ref_stem"]), wav_path(u["gt_stem"])
        if not (os.path.exists(ref_audio) and os.path.exists(gt_audio)):
            print(f"    [skip] {utt_key}: missing {ref_audio} or {gt_audio}")
            continue

        refs = precompute_refs(tts, ref_audio, gt_audio)
        base_xvec = extract_xvec(tts, ref_audio)
        tau_dev = match_shape(tau, base_xvec)

        sub_dir = os.path.join(out_dir, utt_key)
        Path(sub_dir).mkdir(parents=True, exist_ok=True)

        sent_block = {
            "frase_code": u["frase_code"],
            "ref_audio": ref_audio,
            "ref_text": u["ref_text"],
            "gt_emo_audio": gt_audio,
            "synth_text": u["synth_text"],
            "base_xvec_norm": float(base_xvec.norm().item()),
            "conditions": {},
        }
        cell["per_sentence"][utt_key] = sent_block

        for alpha in alphas:
            cond_name = f"alpha_{alpha:.2f}"
            hybrid = base_xvec + alpha * tau_dev
            wav, sr = synthesize_with_xvec(
                tts, u["synth_text"], ref_audio, u["ref_text"], hybrid,
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
                    target_text=u["synth_text"], wer_language=wer_language,
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
            (sent_block["conditions"].get(f"alpha_{a:.2f}", {}).get("metrics", {}) or {}).get("emo_cos_sim_gt")
            for a in alphas
        ]
        eco_str = " ".join(f"{(v if v is not None else float('nan')):+.3f}" for v in eco)
        print(f"    {utt_key}  emo@α: {eco_str}")

    cell["aggregates"] = aggregate_over_sentences(cell, alphas)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(cell, f, indent=2, default=str)
    return cell


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--emouerj_root", default=EMOUERJ_ROOT)
    p.add_argument("--tau_dir", default="data/tau")
    p.add_argument("--output_root", default="data/experiments")
    p.add_argument("--model_path", default="./Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--speakers", nargs="+", default=PAIRED_SPK + XTEXT_SPK)
    p.add_argument("--tau_variants", nargs="+", default=TAU_VARIANTS)
    p.add_argument("--emotions", nargs="+", default=EMOTIONS)
    p.add_argument(
        "--max_per_anchor", type=int, default=6,
        help="(xtext/w04) nº de tomadas de GT por frase-âncora = nº de sínteses",
    )
    p.add_argument("--language", default="Auto")
    p.add_argument("--wer_language", default="pt")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--summary_only", action="store_true",
                   help="Skip generation; re-aggregate from existing per-cell results.json")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip cells whose results.json already exists")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    emouerj.EMOUERJ_ROOT = args.emouerj_root

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
            for spk in args.speakers:
                cell_key = f"{spk}__{tau_variant}__{emotion}"
                out_dir = os.path.join(
                    args.output_root, f"xvec_ptbr_{spk}_{tau_variant}_{emotion}",
                )
                results_json = os.path.join(out_dir, "results.json")

                if os.path.exists(results_json) and (args.skip_existing or args.summary_only):
                    with open(results_json) as f:
                        all_results[cell_key] = json.load(f)
                    print(f"[reuse] {cell_key}")
                    continue
                if args.summary_only:
                    print(f"[summary_only] missing {results_json}; skipping {cell_key}")
                    continue

                utts = utterances_for(spk, emotion, args.max_per_anchor)
                if not utts:
                    print(f"!! no utts for {cell_key}")
                    continue

                design = "paired" if spk in PAIRED else "xtext"
                print(f"\n{'=' * 70}\n{cell_key}  design={design}  "
                      f"N={len(utts)} × {len(ALPHA_GRID)} α\n{'=' * 70}")
                print(f"  τ_file = {tau_file}")
                cell = run_cell(
                    tts,
                    tau_file=tau_file,
                    spk=spk,
                    emotion=emotion,
                    utts=utts,
                    alphas=ALPHA_GRID,
                    out_dir=out_dir,
                    language=args.language,
                    wer_language=args.wer_language,
                    max_new_tokens=args.max_new_tokens,
                )
                all_results[cell_key] = cell

    summary = build_summary(all_results)
    summary_json = os.path.join(args.output_root, "ptbr_summary.json")
    summary_md = os.path.join(args.output_root, "ptbr_summary.md")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    md = (
        render_markdown(summary)
        .replace("EN→EN cross-speaker α-sweep (n=30, ESD test split 321–350)",
                 "PT-BR cross-lingual α-sweep (emoUERJ; m03, m04 paired + w04 xtext)")
        .replace("EnglishTextNormalizer", "BasicTextNormalizer (whisper PT)")
    )
    md += (
        "\n_PT-BR caveats: (i) `UTMOS` (UTMOSv2) não foi treinado em PT-BR — usar só\n"
        "como proxy relativo intra-experimento, não comparar 1:1 com nMOS nem com\n"
        "EN→EN. (ii) `xvec_cos_sim_gt` (Qwen ECAPA): synth e GT são o MESMO falante\n"
        "PT-BR, então a métrica pode saturar como no EN→EN — o sinal informativo é\n"
        "**Δ vs α=0** (o τ inglês move o x-vec PT-BR rumo ao GT emocional?), não o\n"
        "valor absoluto. (iii) `WER_norm` usa o `BasicTextNormalizer` (não o\n"
        "`EnglishTextNormalizer`). (iv) `w04` é probe cross-text (design B) e está\n"
        "tabulado junto com m03/m04 só por conveniência; comparações de SECS_W \n"
        "entre paired (mesma frase) e xtext (frase diferente) **não são**\n"
        "diretamente comparáveis._\n"
    )
    with open(summary_md, "w") as f:
        f.write(md)
    print(f"\nSaved {summary_json}\nSaved {summary_md}")


if __name__ == "__main__":
    main()
