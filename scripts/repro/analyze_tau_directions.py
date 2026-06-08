"""Pairwise cosine analysis of τ directions in Qwen3-TTS x-vector space.

Loads the 6 τ artifacts in data/tau/ and computes, per τ family
({single0017, avg4spk}), the pairwise cos(τ_e1, τ_e2) over
{angry, happy, sad}, plus ‖τ‖ for each.

Writes:
    data/tau/tau_direction_analysis.json
    stdout: 2 markdown tables (norms + pairwise cosines)

Usage:
    PYTHONPATH=. uv run python scripts/repro/analyze_tau_directions.py
"""

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F


EMOTIONS = ["angry", "happy", "sad"]
FAMILIES = ["single0017", "avg4spk"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tau_dir", default="data/tau")
    p.add_argument("--output", default="data/tau/tau_direction_analysis.json")
    return p.parse_args()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def load_taus(tau_dir: str) -> dict:
    out: dict = {}
    for fam in FAMILIES:
        out[fam] = {}
        for emo in EMOTIONS:
            path = os.path.join(tau_dir, f"tau_{emo}_{fam}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            art = torch.load(path, map_location="cpu", weights_only=False)
            out[fam][emo] = {
                "path": path,
                "tau": art["tau"].float(),
                "norm": art["tau"].float().norm().item(),
                "config_speakers": art["config"]["speakers"],
                "stats": art["stats"],
            }
    return out


def pairwise(family_taus: dict) -> dict[str, float]:
    return {
        f"cos(τ_{e1}, τ_{e2})": cos(family_taus[e1]["tau"], family_taus[e2]["tau"])
        for e1, e2 in combinations(EMOTIONS, 2)
    }


def render_norms_table(taus: dict) -> str:
    head = "| τ variant | " + " | ".join(f"‖τ_{e}‖" for e in EMOTIONS) + " |\n"
    head += "|---|" + "|".join("---" for _ in EMOTIONS) + "|\n"
    body = ""
    for fam in FAMILIES:
        cells = " | ".join(f"{taus[fam][e]['norm']:.4f}" for e in EMOTIONS)
        body += f"| {fam} | {cells} |\n"
    return head + body


def render_pairwise_table(pairs: dict) -> str:
    cols = list(next(iter(pairs.values())).keys())
    head = "| τ variant | " + " | ".join(cols) + " |\n"
    head += "|---|" + "|".join("---" for _ in cols) + "|\n"
    body = ""
    for fam in FAMILIES:
        cells = " | ".join(f"{pairs[fam][c]:+.4f}" for c in cols)
        body += f"| {fam} | {cells} |\n"
    return head + body


def main():
    args = parse_args()
    taus = load_taus(args.tau_dir)
    pairs = {fam: pairwise(taus[fam]) for fam in FAMILIES}

    norms_md = render_norms_table(taus)
    pairs_md = render_pairwise_table(pairs)
    print("## τ norms\n")
    print(norms_md)
    print("\n## Pairwise cosine similarities between emotion τ directions\n")
    print(pairs_md)

    out_payload = {
        "tau_dir": args.tau_dir,
        "norms": {
            fam: {e: taus[fam][e]["norm"] for e in EMOTIONS}
            for fam in FAMILIES
        },
        "pairwise_cosines": pairs,
        "config_speakers": {
            fam: taus[fam][EMOTIONS[0]]["config_speakers"] for fam in FAMILIES
        },
        "tau_stats": {
            fam: {e: taus[fam][e]["stats"] for e in EMOTIONS}
            for fam in FAMILIES
        },
        "markdown": {
            "norms": norms_md,
            "pairwise_cosines": pairs_md,
        },
    }
    Path(os.path.dirname(args.output) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out_payload, f, indent=2, default=str)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
