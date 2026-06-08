"""Extract per-emotion x-vector centroids and τ = E[xvec(emo)] − E[xvec(neutral)] from ESD.

Implements the multi-speaker centroid arithmetic from §3.2.4 of artigo_v2.tex:

    τ_emo = E_{s ∈ S}[xvec(s, emo)] − E_s[xvec(s, neutral)]

X-vectors are extracted by Qwen3-TTS's own ECAPA-TDNN speaker encoder (24 kHz),
matching the encoder that conditions the LM at inference (consistent operand).

Outputs one .pt per (emotion, source_set), self-describing:

    {
      "tau": Tensor,                                # the centroid difference
      "emotion_centroid": Tensor,                   # mean over (spk, emo) xvecs
      "neutral_centroid": Tensor,                   # mean over (spk, neutral) xvecs
      "per_speaker_xvec": {spk: {emo: Tensor, "neutral": Tensor}},
      "config": {speakers, emotion, n_pairs, utt_ids, model_path, ...},
      "stats": {tau_norm, neutral_norm, emo_norm, cos_neutral_emo, tau_over_neutral},
    }

Filename convention:
  - 1 speaker  →  tau_{emo}_single{spk}.pt
  - N speakers →  tau_{emo}_avg{N}spk.pt

Usage:
    # Multi-speaker τ_avg over {0011, 0014, 0017, 0020} for Angry, Happy, Sad
    PYTHONPATH=. uv run python scripts/repro/extract_xvec_tau.py \\
        --esd_dir /home/daniel/data/processed/esd_24k \\
        --speakers 0011 0014 0017 0020 \\
        --emotions Angry Happy Sad \\
        --n_pairs 50 \\
        --output_dir data/tau

    # Single-speaker τ baseline (0017 only)
    PYTHONPATH=. uv run python scripts/repro/extract_xvec_tau.py \\
        --esd_dir /home/daniel/data/processed/esd_24k \\
        --speakers 0017 \\
        --emotions Angry Happy Sad \\
        --n_pairs 50 \\
        --output_dir data/tau
"""

import argparse
import os
from pathlib import Path

import librosa
import torch
from qwen_tts import Qwen3TTSModel


EMOTION_OFFSETS = {
    "Neutral": 0,
    "Angry": 350,
    "Happy": 700,
    "Sad": 1050,
    "Surprise": 1400,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--esd_dir", required=True,
                   help="Root of resampled ESD (contains 0011/Neutral/*.wav etc.)")
    p.add_argument("--speakers", nargs="+", required=True,
                   help="Source speaker IDs, e.g. 0011 0014 0017 0020")
    p.add_argument("--emotions", nargs="+", default=["Angry", "Happy", "Sad"],
                   choices=["Angry", "Happy", "Sad", "Surprise"],
                   help="Target emotions for τ (Neutral is always the baseline)")
    p.add_argument("--n_pairs", type=int, default=50,
                   help="Utterances per (speaker, emotion) for the centroid")
    p.add_argument("--output_dir", default="data/tau")
    p.add_argument("--model_path", default="./Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--start_utt_id", type=int, default=1,
                   help="First utterance id within each emotion block (1-based)")
    return p.parse_args()


def utt_id_for(speaker: str, emotion: str, local_idx: int) -> str:
    """Map (speaker, emotion, 1-based local index) → ESD filename stem."""
    abs_idx = EMOTION_OFFSETS[emotion] + local_idx
    return f"{speaker}_{abs_idx:06d}"


def find_wav(esd_dir: str, speaker: str, emotion: str, utt_id: str) -> str | None:
    candidates = list(Path(esd_dir, speaker, emotion).rglob(f"{utt_id}.wav"))
    return str(candidates[0]) if candidates else None


def extract_xvec(tts: Qwen3TTSModel, wav_path: str) -> torch.Tensor:
    spk_sr = tts.model.speaker_encoder_sample_rate
    audio, _ = librosa.load(wav_path, sr=spk_sr, mono=True)
    return tts.model.extract_speaker_embedding(audio, sr=spk_sr).detach().cpu().float()


def collect_xvecs(
    tts: Qwen3TTSModel,
    esd_dir: str,
    speaker: str,
    emotion: str,
    n_pairs: int,
    start_utt_id: int,
) -> tuple[torch.Tensor, list[str]]:
    """Return (mean_xvec, utt_ids_used) for one (speaker, emotion)."""
    xvecs: list[torch.Tensor] = []
    utt_ids: list[str] = []
    local_idx = start_utt_id
    while len(xvecs) < n_pairs:
        utt_id = utt_id_for(speaker, emotion, local_idx)
        wav_path = find_wav(esd_dir, speaker, emotion, utt_id)
        local_idx += 1
        if wav_path is None:
            if local_idx > start_utt_id + 350:
                raise RuntimeError(
                    f"Ran out of utterances for {speaker}/{emotion} after "
                    f"{len(xvecs)} / {n_pairs}"
                )
            continue
        xvecs.append(extract_xvec(tts, wav_path).flatten())
        utt_ids.append(utt_id)
    return torch.stack(xvecs).mean(dim=0), utt_ids


def compute_stats(neutral_c: torch.Tensor, emo_c: torch.Tensor, tau: torch.Tensor) -> dict:
    cos = torch.nn.functional.cosine_similarity
    return {
        "tau_norm": tau.norm().item(),
        "neutral_norm": neutral_c.norm().item(),
        "emo_norm": emo_c.norm().item(),
        "cos_neutral_emo": cos(neutral_c, emo_c, dim=0).item(),
        "tau_over_neutral": (tau.norm() / neutral_c.norm()).item(),
    }


def output_filename(emotion: str, speakers: list[str]) -> str:
    suffix = f"single{speakers[0]}" if len(speakers) == 1 else f"avg{len(speakers)}spk"
    return f"tau_{emotion.lower()}_{suffix}.pt"


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = Qwen3TTSModel.from_pretrained(
        args.model_path, device_map=device, dtype=torch.bfloat16,
    )

    # Neutral centroids once per speaker (shared across all τ_emo)
    print(f"\n{'=' * 70}")
    print(f"Extracting neutral centroids ({args.n_pairs} utt/spk)")
    print(f"{'=' * 70}")
    per_spk_neutral: dict[str, torch.Tensor] = {}
    per_spk_neutral_utts: dict[str, list[str]] = {}
    for spk in args.speakers:
        mean_xvec, utt_ids = collect_xvecs(
            tts, args.esd_dir, spk, "Neutral", args.n_pairs, args.start_utt_id,
        )
        per_spk_neutral[spk] = mean_xvec
        per_spk_neutral_utts[spk] = utt_ids
        print(f"  {spk}/Neutral: mean over {len(utt_ids)} utt, "
              f"‖x̄‖={mean_xvec.norm().item():.4f}")

    neutral_centroid = torch.stack(list(per_spk_neutral.values())).mean(dim=0)

    for emotion in args.emotions:
        print(f"\n{'=' * 70}")
        print(f"τ for emotion = {emotion}")
        print(f"{'=' * 70}")

        per_spk_emo: dict[str, torch.Tensor] = {}
        per_spk_emo_utts: dict[str, list[str]] = {}
        for spk in args.speakers:
            mean_xvec, utt_ids = collect_xvecs(
                tts, args.esd_dir, spk, emotion, args.n_pairs, args.start_utt_id,
            )
            per_spk_emo[spk] = mean_xvec
            per_spk_emo_utts[spk] = utt_ids
            print(f"  {spk}/{emotion}: mean over {len(utt_ids)} utt, "
                  f"‖x̄‖={mean_xvec.norm().item():.4f}")

        emo_centroid = torch.stack(list(per_spk_emo.values())).mean(dim=0)
        tau = emo_centroid - neutral_centroid
        stats = compute_stats(neutral_centroid, emo_centroid, tau)

        print(f"\n  τ_{emotion.lower()} stats:")
        for k, v in stats.items():
            print(f"    {k:20s} = {v:.4f}")

        artifact = {
            "tau": tau,
            "emotion_centroid": emo_centroid,
            "neutral_centroid": neutral_centroid,
            "per_speaker_xvec": {
                spk: {"neutral": per_spk_neutral[spk], emotion.lower(): per_spk_emo[spk]}
                for spk in args.speakers
            },
            "config": {
                "speakers": list(args.speakers),
                "emotion": emotion,
                "baseline_emotion": "Neutral",
                "n_pairs_per_speaker": args.n_pairs,
                "start_utt_id": args.start_utt_id,
                "utt_ids": {
                    spk: {
                        "neutral": per_spk_neutral_utts[spk],
                        emotion.lower(): per_spk_emo_utts[spk],
                    }
                    for spk in args.speakers
                },
                "model_path": args.model_path,
                "esd_dir": args.esd_dir,
                "speaker_encoder_sample_rate": tts.model.speaker_encoder_sample_rate,
            },
            "stats": stats,
        }

        out_path = os.path.join(args.output_dir, output_filename(emotion, args.speakers))
        torch.save(artifact, out_path)
        print(f"  → saved: {out_path}")

    print(f"\nDone. {len(args.emotions)} τ artifact(s) in {args.output_dir}/")


if __name__ == "__main__":
    main()
