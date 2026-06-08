"""Patch UTMOSv2 scores into existing per-cell results.json (no re-synthesis).

Walks every data/experiments/xvec_en2en_*/results.json, scores each wav with
UTMOSv2 (batched per utt_xxx dir for speed), writes back metrics.utmos, and
re-runs the per-cell aggregator.

Usage:
    PYTHONPATH=. uv run python scripts/add_utmos_to_results.py
    PYTHONPATH=. uv run python scripts/add_utmos_to_results.py --skip_existing
"""

import argparse
import json
import time
from glob import glob

from src.metrics.utmos import score_wavs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments_root", default="data/experiments")
    p.add_argument("--glob", default="xvec_en2en_*/results.json")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip wavs that already have a utmos field")
    return p.parse_args()


def patch_cell(results_path: str, skip_existing: bool) -> tuple[int, int]:
    with open(results_path) as f:
        cell = json.load(f)

    wavs_to_score: list[str] = []
    wav_to_cond: dict[str, tuple[str, str]] = {}
    skipped = 0
    for utt_key, sent_block in cell.get("per_sentence", {}).items():
        for cond_name, cond in (sent_block.get("conditions") or {}).items():
            wav = cond.get("wav_path")
            if not wav:
                continue
            metrics = cond.get("metrics") or {}
            if skip_existing and isinstance(metrics.get("utmos"), (int, float)):
                skipped += 1
                continue
            wavs_to_score.append(wav)
            wav_to_cond[wav] = (utt_key, cond_name)

    if not wavs_to_score:
        return 0, skipped

    scores = score_wavs(wavs_to_score)

    scored = 0
    for wav, mos in scores.items():
        utt_key, cond_name = wav_to_cond[wav]
        cond = cell["per_sentence"][utt_key]["conditions"][cond_name]
        cond.setdefault("metrics", {})["utmos"] = float(mos)
        scored += 1

    cell["aggregates"] = _reaggregate(cell)

    with open(results_path, "w") as f:
        json.dump(cell, f, indent=2, default=str)
    return scored, skipped


def _reaggregate(cell: dict) -> dict:
    from src.sweep import aggregate_over_sentences

    alphas = cell.get("config", {}).get("alphas") or []
    return aggregate_over_sentences(cell, alphas)


def main():
    args = parse_args()
    paths = sorted(glob(f"{args.experiments_root}/{args.glob}"))
    print(f"Found {len(paths)} cell results.json files", flush=True)
    total_scored = 0
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        ts = time.time()
        scored, skipped = patch_cell(p, args.skip_existing)
        total_scored += scored
        cell_name = p.split("/")[-2]
        dt = time.time() - ts
        print(f"  [{i}/{len(paths)}] {cell_name}: scored {scored}, skipped {skipped} ({dt:.1f}s)",
              flush=True)
    print(f"\nTotal UTMOSv2 scores written: {total_scored} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
