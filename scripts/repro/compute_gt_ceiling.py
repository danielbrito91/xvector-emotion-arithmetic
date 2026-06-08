"""Compute the GT (ground-truth recording) ceiling row for the metric table.

Two complementary ceiling methodologies are reported:

(A) Per-utt parallel (literal "replace synth with GT" using v2/v3 synth RHS):
    secs_w           = cos(WavLM(GT_emo_X), WavLM(GT_neutral_X))      per utt
    emo_cos_self     = pairwise cos(emo2vec(GT_X), emo2vec(GT_Y))     C(30,2)
                       (per-utt would give trivial 1.0 — use pairwise instead).

(B) Anchor at utt 10 (v1-pilot reference; cross-text natural variance):
    emo_cos_anchor   = cos(emo2vec(GT_emo_X), emo2vec(GT_emo_utt10))  per utt
    secs_w_anchor    = cos(WavLM(GT_emo_X),  WavLM(GT_neutral_utt10)) per utt

Both anchored at the same speaker; X ranges over test split (local idx 321–350).

Other metrics (single-arg, no comparison):
    utmos            = UTMOSv2 directly on the 30 GT_emo recordings
    wer_norm         = Whisper-large-v3 + EnglishTextNormalizer on GT_emo

Output: data/experiments/gt_ceiling.{json,md}
"""

import argparse
import itertools
import json
import os
import statistics
from pathlib import Path

import torch

from src.paths import esd_root

EMO_OFFSET = {"angry": 350, "happy": 700, "sad": 1050}
ANCHOR_UTT = 10  # v1 pilot reference utterance for the anchored-ceiling variant


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--esd_root", default=esd_root())
    p.add_argument("--speakers", nargs="+", default=["0013", "0019"])
    p.add_argument("--emotions", nargs="+", default=["angry", "happy", "sad"])
    p.add_argument("--idx_start", type=int, default=321)
    p.add_argument("--idx_end", type=int, default=350)
    p.add_argument("--anchor_utt", type=int, default=ANCHOR_UTT,
                   help="utt idx used for the anchored-ceiling variant (v1 pilot ref)")
    p.add_argument("--out_json", default="data/experiments/gt_ceiling.json")
    p.add_argument("--out_md", default="data/experiments/gt_ceiling.md")
    return p.parse_args()


def neutral_path(esd: str, spk: str, idx: int) -> str:
    return f"{esd}/{spk}/Neutral/{spk}_{idx:06d}.wav"


def emo_path(esd: str, spk: str, emo: str, idx: int) -> str:
    return f"{esd}/{spk}/{emo.capitalize()}/{spk}_{idx + EMO_OFFSET[emo]:06d}.wav"


def parse_transcripts(esd: str, spk: str) -> dict[int, str]:
    txt = f"{esd}/{spk}/{spk}.txt"
    out: dict[int, str] = {}
    with open(txt) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            uid = parts[0]
            try:
                global_idx = int(uid.split("_")[-1])
            except ValueError:
                continue
            out[global_idx] = parts[1]
    return out


def utt_local_to_global(idx: int) -> int:
    return idx


def cell_metrics(
    esd: str,
    spk: str,
    emotion: str,
    indices: list[int],
    transcripts: dict[int, str],
    anchor_utt: int,
) -> dict:
    from src.metrics.asr import compute_wer_norm, transcribe
    from src.metrics.emotion import get_emotion
    from src.metrics.speaker import get_speaker_embedding
    from src.metrics.utmos import score_wavs
    from src.xvec import cos_flat

    emo_paths = [emo_path(esd, spk, emotion, i) for i in indices]
    neu_paths = [neutral_path(esd, spk, i) for i in indices]
    anchor_emo_path = emo_path(esd, spk, emotion, anchor_utt)
    anchor_neu_path = neutral_path(esd, spk, anchor_utt)

    # (A) per-utt parallel paths (literal v2/v3 substitution)
    # 1) UTMOS on GT_emo
    print(f"  [{spk}/{emotion}] UTMOS on {len(emo_paths)} GTs...", flush=True)
    utmos_map = score_wavs(emo_paths)
    utmos_vals = [utmos_map[p] for p in emo_paths if p in utmos_map]

    # 2) SECS_W (per-utt paired) and WavLM embeddings (reused below)
    print(f"  [{spk}/{emotion}] WavLM embeddings + SECS_W (paired)...", flush=True)
    wavlm_emo: list[torch.Tensor] = []
    wavlm_neu: list[torch.Tensor] = []
    for ep, np_ in zip(emo_paths, neu_paths):
        wavlm_emo.append(get_speaker_embedding(ep))
        wavlm_neu.append(get_speaker_embedding(np_))
    secs_paired_vals = [cos_flat(we, wn) for we, wn in zip(wavlm_emo, wavlm_neu)]

    # 3) emo_cos_self (pairwise within-emo across test split)
    print(f"  [{spk}/{emotion}] emotion2vec embeddings + pairwise self-sim...", flush=True)
    emo_embs: list[torch.Tensor] = [
        torch.tensor(get_emotion(p)["embedding"]) for p in emo_paths
    ]
    pairs = list(itertools.combinations(range(len(emo_embs)), 2))
    eecs_self_vals = [cos_flat(emo_embs[i], emo_embs[j]) for i, j in pairs]

    # (B) anchored at utt `anchor_utt` (v1-pilot reference; cross-text variance vs fixed point)
    print(f"  [{spk}/{emotion}] anchor at utt {anchor_utt} ({anchor_emo_path.split('/')[-1]})...", flush=True)
    anchor_emo_emb = torch.tensor(get_emotion(anchor_emo_path)["embedding"])
    anchor_neu_wavlm = get_speaker_embedding(anchor_neu_path)
    eecs_anchor_vals = [cos_flat(e, anchor_emo_emb) for e in emo_embs]
    secs_anchor_vals = [cos_flat(we, anchor_neu_wavlm) for we in wavlm_emo]

    # 4) WER_norm on GT_emo (its own canonical text; same as v2/v3 synth uses)
    print(f"  [{spk}/{emotion}] WER on GTs...", flush=True)
    wer_vals: list[float] = []
    for ep, idx in zip(emo_paths, indices):
        ref_text = transcripts.get(utt_local_to_global(idx), "")
        if not ref_text:
            continue
        hyp = transcribe(ep, language="en")
        wer_vals.append(compute_wer_norm(ref_text, hyp, language="en"))

    def agg(vs: list[float]) -> dict:
        if not vs:
            return {"mean": float("nan"), "stderr": float("nan"), "n": 0}
        return {
            "mean": statistics.fmean(vs),
            "stderr": statistics.stdev(vs) / len(vs) ** 0.5 if len(vs) > 1 else 0.0,
            "n": len(vs),
        }

    return {
        "spk": spk,
        "emotion": emotion,
        "anchor_utt": anchor_utt,
        "anchor_emo_path": anchor_emo_path,
        "anchor_neu_path": anchor_neu_path,
        # (A) literal v2/v3 substitution
        "emo_cos_self_pairwise": agg(eecs_self_vals),
        "secs_w_paired": agg(secs_paired_vals),
        # (B) anchored at utt 10
        "emo_cos_anchor_utt10": agg(eecs_anchor_vals),
        "secs_w_anchor_utt10": agg(secs_anchor_vals),
        # single-arg metrics
        "utmos": agg(utmos_vals),
        "wer_norm": agg(wer_vals),
    }


def render_md(rows: list[dict]) -> str:
    def fmt(d, f="{:.4f}"):
        return f"{f.format(d['mean'])} ± {f.format(d['stderr'])}"

    head = (
        "# GT (real ESD recording) ceiling — duas metodologias complementares\n\n"
        "Métricas calculadas nos áudios humanos do ESD test split (local idx\n"
        "321–350). Duas abordagens de _ceiling_:\n\n"
        "- **(A) Per-utt paralelo** = literal substituição `synth_X → GT_X` no\n"
        "  pipeline v2/v3 (que usa GT_emo_X / GT_neutral_X como RHS, por utt).\n"
        "  Como `emo_cos_sim_gt(GT_X, GT_X) = 1.0` é trivial, reportamos no lugar\n"
        "  `EECS_self_pairwise` = média de C(30,2)=435 pares disjuntos within-emo\n"
        "  no test split (mede variância natural intra-classe do emotion2vec).\n"
        "- **(B) Anchored em utt 10** (referência v1-pilot, paralela por\n"
        "  construção): mede `cos(GT_emo_X, GT_emo_utt10)` para cada X — proxy de\n"
        "  consistência cross-text dentro de (spk, emo), ancorada num ponto fixo.\n\n"
        "Para SECS_W, paired (A) usa `GT_neutral_X` per-utt (literal v2/v3); (B) usa\n"
        "`GT_neutral_utt10` fixo. Para UTMOS e WER_norm, métricas single-arg sobre\n"
        "as 30 GTs (idênticas nas duas metodologias).\n\n"
        "## Tabela (A): per-utt parallel — diretamente comparável às cells de synth\n\n"
        "| spk | emotion | EECS_self pairwise | SECS_W paired (vs GT_neutral_X) | UTMOS | WER_norm |\n"
        "|---|---|---|---|---|---|\n"
    )
    body = ""
    for r in rows:
        body += (
            f"| {r['spk']} | {r['emotion']} | {fmt(r['emo_cos_self_pairwise'])} | "
            f"{fmt(r['secs_w_paired'])} | {fmt(r['utmos'], '{:.3f}')} | "
            f"{fmt(r['wer_norm'], '{:.3f}')} |\n"
        )

    head_b = (
        "\n## Tabela (B): anchored em utt 10 — cross-text natural variance\n\n"
        "| spk | emotion | EECS anchor (vs GT_emo_utt10) | SECS_W anchor (vs GT_neutral_utt10) | UTMOS | WER_norm |\n"
        "|---|---|---|---|---|---|\n"
    )
    body_b = ""
    for r in rows:
        body_b += (
            f"| {r['spk']} | {r['emotion']} | {fmt(r['emo_cos_anchor_utt10'])} | "
            f"{fmt(r['secs_w_anchor_utt10'])} | {fmt(r['utmos'], '{:.3f}')} | "
            f"{fmt(r['wer_norm'], '{:.3f}')} |\n"
        )

    note = (
        "\n_Como ler: (A) é o teto natural intrínseco da métrica para o setup\n"
        "v2/v3 (paired). (B) é o teto ancorado — útil quando o synth também usaria\n"
        "RHS fixo (caso v1 pilot, ou caso o reviewer pedir a variância cross-text).\n"
        "Cells de synth devem ficar **abaixo** desses tetos; se ultrapassam, indica\n"
        "saturação da métrica ou collapse (e.g., synth exagerando atributos\n"
        "prosódicos vs o GT médio)._\n"
    )
    return head + body + head_b + body_b + note


def main():
    args = parse_args()
    indices = list(range(args.idx_start, args.idx_end + 1))
    rows = []
    for spk in args.speakers:
        transcripts = parse_transcripts(args.esd_root, spk)
        for emo in args.emotions:
            print(f"--- {spk} / {emo} ---", flush=True)
            r = cell_metrics(args.esd_root, spk, emo, indices, transcripts, args.anchor_utt)
            print(
                f"  (A) EECS_self_pair={r['emo_cos_self_pairwise']['mean']:.4f} "
                f"SECS_paired={r['secs_w_paired']['mean']:.4f} | "
                f"(B) EECS_anchor={r['emo_cos_anchor_utt10']['mean']:.4f} "
                f"SECS_anchor={r['secs_w_anchor_utt10']['mean']:.4f} | "
                f"UTMOS={r['utmos']['mean']:.3f} "
                f"WER={r['wer_norm']['mean']:.3f}",
                flush=True,
            )
            rows.append(r)

    out = {
        "config": {
            "esd_root": args.esd_root,
            "speakers": args.speakers,
            "emotions": args.emotions,
            "idx_range": [args.idx_start, args.idx_end],
            "n_per_cell": len(indices),
            "anchor_utt": args.anchor_utt,
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
