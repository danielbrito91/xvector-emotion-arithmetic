"""GT ceiling para o experimento PT-BR (emoUERJ) — espelha compute_gt_ceiling.py.

Dois designs:

(A) **m03, m04** (paired, same-text neutral/emo por frase):
    - `emo_cos_self_pairwise` = média sobre C(n,2) pares within-emo no GT_emo. n=8–10.
    - `secs_w_paired`         = cos(WavLM(GT_emo_X), WavLM(GT_neutral_X)) per utt.
    - utmos, wer_norm (single-arg) sobre as n gravações GT_emo.

(B) **w04** (xtext, design cross-text):
    Por anchor (frase com GT emocional, e.g. B/M para anger), reportamos:
    - `emo_cos_within_anchor`  = pairwise cos(GT_emo_take_i, GT_emo_take_j) sobre
                                 todas as tomadas da mesma anchor (C(n,2)).
    - `secs_w_xtext`           = média sobre (i, ref_i) ciclados — mesmas tuplas
                                 (ref_neutral, gt_emo) que `run_ptbr_sweep.py`
                                 usa para sintetizar. Compara identidade entre
                                 neutral take e emo take da mesma w04.
    - utmos, wer_norm sobre as tomadas GT_emo (todas as anchors juntas).

Saída: data/experiments/gt_ceiling_ptbr.{json,md}
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
from pathlib import Path

import torch

from src import emouerj
from src.emouerj import (
    EMOUERJ_ROOT,
    FRASE_TEXT,
    PAIRED,
    PAIRED_SPK,
    XTEXT,
    XTEXT_SPK,
    wav_path,
)

EMOTIONS = ["angry", "happy", "sad"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--emouerj_root", default=EMOUERJ_ROOT)
    p.add_argument("--speakers", nargs="+", default=PAIRED_SPK + XTEXT_SPK)
    p.add_argument("--emotions", nargs="+", default=EMOTIONS)
    p.add_argument("--max_per_anchor", type=int, default=6,
                   help="(xtext/w04) tomadas por anchor a usar — espelha run_ptbr_sweep")
    p.add_argument("--out_json", default="data/experiments/gt_ceiling_ptbr.json")
    p.add_argument("--out_md", default="data/experiments/gt_ceiling_ptbr.md")
    p.add_argument("--wer_language", default="pt")
    return p.parse_args()


def agg(vs: list[float]) -> dict:
    if not vs:
        return {"mean": float("nan"), "stderr": float("nan"), "n": 0}
    return {
        "mean": statistics.fmean(vs),
        "stderr": statistics.stdev(vs) / len(vs) ** 0.5 if len(vs) > 1 else 0.0,
        "n": len(vs),
    }


def paired_pairs(spk: str, emotion: str) -> list[dict]:
    """One (neutral, emo) pair per frase available in PAIRED[spk] for this emotion."""
    out = []
    for code, files in PAIRED[spk].items():
        if "neutral" not in files or emotion not in files:
            continue
        out.append({
            "frase_code": code,
            "ref_stem": files["neutral"],
            "gt_stem": files[emotion],
            "text": FRASE_TEXT[code],
        })
    return out


def xtext_tuples(spk: str, emotion: str, max_per_anchor: int) -> list[dict]:
    """Ciclando neutral refs entre takes de cada anchor (espelha run_ptbr_sweep)."""
    out = []
    spec = XTEXT[spk]
    neu = spec["neutral_refs"]
    for anchor, gt_takes in spec[emotion].items():
        for i, gt in enumerate(gt_takes[:max_per_anchor]):
            ref = neu[i % len(neu)]
            out.append({
                "frase_code": anchor,
                "ref_stem": ref,
                "gt_stem": gt,
                "text": FRASE_TEXT[anchor],
            })
    return out


def paired_cell(spk: str, emotion: str, wer_language: str) -> dict:
    from src.metrics.asr import compute_wer_norm, transcribe
    from src.metrics.emotion import get_emotion
    from src.metrics.speaker import get_speaker_embedding
    from src.metrics.utmos import score_wavs
    from src.xvec import cos_flat

    pairs = paired_pairs(spk, emotion)
    print(f"  [{spk}/{emotion}/paired] n={len(pairs)} frases", flush=True)
    emo_paths = [wav_path(p["gt_stem"]) for p in pairs]
    neu_paths = [wav_path(p["ref_stem"]) for p in pairs]

    print(f"  [{spk}/{emotion}] UTMOS...", flush=True)
    utmos_map = score_wavs(emo_paths)
    utmos_vals = [utmos_map[p] for p in emo_paths if p in utmos_map]

    print(f"  [{spk}/{emotion}] WavLM + SECS_W paired...", flush=True)
    wavlm_emo = [get_speaker_embedding(p) for p in emo_paths]
    wavlm_neu = [get_speaker_embedding(p) for p in neu_paths]
    secs_paired = [cos_flat(e, n) for e, n in zip(wavlm_emo, wavlm_neu)]

    print(f"  [{spk}/{emotion}] emotion2vec pairwise self-sim...", flush=True)
    emo_embs = [torch.tensor(get_emotion(p)["embedding"]) for p in emo_paths]
    pairs_idx = list(itertools.combinations(range(len(emo_embs)), 2))
    eecs_self = [cos_flat(emo_embs[i], emo_embs[j]) for i, j in pairs_idx]

    print(f"  [{spk}/{emotion}] WER...", flush=True)
    wer_vals: list[float] = []
    for p, ref_text in zip(emo_paths, [pp["text"] for pp in pairs]):
        hyp = transcribe(p, language=wer_language)
        wer_vals.append(compute_wer_norm(ref_text, hyp, language=wer_language))

    return {
        "spk": spk,
        "emotion": emotion,
        "design": "paired",
        "n_pairs": len(pairs),
        "frase_codes": [p["frase_code"] for p in pairs],
        "emo_cos_self_pairwise": agg(eecs_self),
        "secs_w_paired": agg(secs_paired),
        "utmos": agg(utmos_vals),
        "wer_norm": agg(wer_vals),
    }


def xtext_cell(spk: str, emotion: str, max_per_anchor: int, wer_language: str) -> dict:
    from src.metrics.asr import compute_wer_norm, transcribe
    from src.metrics.emotion import get_emotion
    from src.metrics.speaker import get_speaker_embedding
    from src.metrics.utmos import score_wavs
    from src.xvec import cos_flat

    tuples = xtext_tuples(spk, emotion, max_per_anchor)
    print(f"  [{spk}/{emotion}/xtext] n={len(tuples)} tuplas (anchors × takes)", flush=True)
    emo_paths = [wav_path(t["gt_stem"]) for t in tuples]
    neu_paths = [wav_path(t["ref_stem"]) for t in tuples]

    print(f"  [{spk}/{emotion}] UTMOS...", flush=True)
    utmos_map = score_wavs(emo_paths)
    utmos_vals = [utmos_map[p] for p in emo_paths if p in utmos_map]

    print(f"  [{spk}/{emotion}] WavLM + SECS_W xtext (mesma tupla do sweep)...", flush=True)
    wavlm_emo = [get_speaker_embedding(p) for p in emo_paths]
    wavlm_neu = [get_speaker_embedding(p) for p in neu_paths]
    secs_xtext = [cos_flat(e, n) for e, n in zip(wavlm_emo, wavlm_neu)]

    print(f"  [{spk}/{emotion}] emotion2vec within-anchor pairwise...", flush=True)
    emo_embs = [torch.tensor(get_emotion(p)["embedding"]) for p in emo_paths]
    by_anchor: dict[str, list[int]] = {}
    for i, t in enumerate(tuples):
        by_anchor.setdefault(t["frase_code"], []).append(i)
    eecs_within: list[float] = []
    per_anchor_eecs: dict[str, dict] = {}
    for anchor, idxs in by_anchor.items():
        vals = [cos_flat(emo_embs[i], emo_embs[j]) for i, j in itertools.combinations(idxs, 2)]
        per_anchor_eecs[anchor] = agg(vals)
        eecs_within.extend(vals)

    print(f"  [{spk}/{emotion}] WER...", flush=True)
    wer_vals: list[float] = []
    for p, ref_text in zip(emo_paths, [t["text"] for t in tuples]):
        hyp = transcribe(p, language=wer_language)
        wer_vals.append(compute_wer_norm(ref_text, hyp, language=wer_language))

    return {
        "spk": spk,
        "emotion": emotion,
        "design": "xtext",
        "n_tuples": len(tuples),
        "anchors": list(by_anchor.keys()),
        "emo_cos_within_anchor_pairwise": agg(eecs_within),
        "emo_cos_within_anchor_per_anchor": per_anchor_eecs,
        "secs_w_xtext": agg(secs_xtext),
        "utmos": agg(utmos_vals),
        "wer_norm": agg(wer_vals),
    }


def render_md(rows: list[dict]) -> str:
    def fmt(d, f="{:.4f}"):
        return f"{f.format(d['mean'])} ± {f.format(d['stderr'])}"

    paired = [r for r in rows if r["design"] == "paired"]
    xtext = [r for r in rows if r["design"] == "xtext"]

    head_a = (
        "# GT (gravação real emoUERJ) ceiling — PT-BR cross-lingual\n\n"
        "Duas metodologias complementares, conforme o design de cada falante:\n\n"
        "- **(A) Paired (m03, m04)** — mesma frase em neutral + emoção. Literal\n"
        "  substituição `synth_X → GT_X` no pipeline v3 (que usa GT_emo_X /\n"
        "  GT_neutral_X como RHS, por utt). `EECS_self_pairwise` é a média de\n"
        "  C(n,2) pares within-emo sobre as frases paralelas; SECS_W_paired é\n"
        "  per-utt vs `GT_neutral_X` (literal v3, same-text). n = 8 (m04 sad,\n"
        "  sem G e Q) ou 10.\n"
        "- **(B) Xtext (w04)** — não há same-text neutral/emo; ceiling é da\n"
        "  variância natural cross-text within (spk, emo). `EECS_within_anchor`\n"
        "  é a média pairwise dentro de cada anchor (B, M, Q, D, F). SECS_W_xtext\n"
        "  é a média sobre as **mesmas tuplas** (ref_neutral, gt_emo) que o\n"
        "  sweep usa, ciclando ref entre as 10 tomadas neutrais.\n\n"
        "## Tabela (A): paired — diretamente comparável às cells de m03/m04 synth\n\n"
        "| spk | emotion | n | EECS_self pairwise | SECS_W paired (vs GT_neutral_X) | UTMOS | WER_norm |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    body_a = "".join(
        f"| {r['spk']} | {r['emotion']} | {r['n_pairs']} | "
        f"{fmt(r['emo_cos_self_pairwise'])} | {fmt(r['secs_w_paired'])} | "
        f"{fmt(r['utmos'], '{:.3f}')} | {fmt(r['wer_norm'], '{:.3f}')} |\n"
        for r in paired
    )

    head_b = (
        "\n## Tabela (B): xtext — w04 (probe cross-text, design cruzado com o sweep)\n\n"
        "| spk | emotion | n_tuples | EECS within-anchor (pairwise) | SECS_W xtext (vs neutral cíclico) | UTMOS | WER_norm |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    body_b = "".join(
        f"| {r['spk']} | {r['emotion']} | {r['n_tuples']} | "
        f"{fmt(r['emo_cos_within_anchor_pairwise'])} | {fmt(r['secs_w_xtext'])} | "
        f"{fmt(r['utmos'], '{:.3f}')} | {fmt(r['wer_norm'], '{:.3f}')} |\n"
        for r in xtext
    )

    head_b2 = (
        "\n### Tabela (B'): EECS within-anchor por (spk, emoção, anchor)\n\n"
        "| spk | emotion | anchor | n_pairs (C(takes,2)) | EECS within-anchor |\n|---|---|---|---|---|\n"
    )
    body_b2 = ""
    for r in xtext:
        for anchor, a in r["emo_cos_within_anchor_per_anchor"].items():
            body_b2 += f"| {r['spk']} | {r['emotion']} | {anchor} | {a['n']} | {fmt(a)} |\n"

    note = (
        "\n_Como ler: (A) é o teto natural intrínseco da métrica no setup paired\n"
        "(mesma frase em duas emoções, n=8–10 por cell). (B) é o teto cross-text\n"
        "do w04 — não comparável diretamente com (A) e nem com as cells de m03/m04.\n"
        "Cells de synth devem ficar **abaixo** dos respectivos tetos; se ultrapassam,\n"
        "indica saturação da métrica ou collapse._\n"
        "\n_UTMOS é proxy relativo (modelo não treinado em PT-BR). WER usa Whisper-large-v3\n"
        "language=pt + `BasicTextNormalizer` (não `EnglishTextNormalizer`)._\n"
    )
    return head_a + body_a + head_b + body_b + head_b2 + body_b2 + note


def main():
    args = parse_args()
    emouerj.EMOUERJ_ROOT = args.emouerj_root
    rows: list[dict] = []
    for spk in args.speakers:
        for emotion in args.emotions:
            print(f"--- {spk} / {emotion} ---", flush=True)
            if spk in PAIRED:
                r = paired_cell(spk, emotion, args.wer_language)
            elif spk in XTEXT:
                r = xtext_cell(spk, emotion, args.max_per_anchor, args.wer_language)
            else:
                print(f"!! unknown speaker {spk}; skipping")
                continue
            key = (
                f"  → n={r.get('n_pairs') or r.get('n_tuples')} "
                f"EECS={r.get('emo_cos_self_pairwise', r.get('emo_cos_within_anchor_pairwise'))['mean']:.4f} "
                f"SECS_W={r.get('secs_w_paired', r.get('secs_w_xtext'))['mean']:.4f} "
                f"UTMOS={r['utmos']['mean']:.3f} WER={r['wer_norm']['mean']:.3f}"
            )
            print(key, flush=True)
            rows.append(r)

    out = {
        "config": {
            "emouerj_root": args.emouerj_root,
            "speakers": args.speakers,
            "emotions": args.emotions,
            "max_per_anchor": args.max_per_anchor,
            "wer_language": args.wer_language,
        },
        "rows": rows,
    }
    os.makedirs(Path(args.out_json).parent, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, default=str)
    with open(args.out_md, "w") as f:
        f.write(render_md(rows))
    print(f"\nSaved {args.out_json}\nSaved {args.out_md}")


if __name__ == "__main__":
    main()
