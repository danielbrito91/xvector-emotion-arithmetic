"""
Discrete token swap experiment: test whether emotion flows through ICL via codec tokens.

Instead of shifting embeddings (which failed — see docs/token_space_emotion_vectors.md),
directly replace codec tokens in the reference sequence. Uses parallel utterances from
ESD speaker 0017 (same text spoken angry vs neutral) so ref_code tensors are comparable.

Experiments:
  A) Full ref_code swap: angry ref_code + neutral speaker embedding
  B) Selective codebook swap: only replace specific codebook layers in ref_code
     - cb1 only (highest τ norm)
     - cb0-5 (top codebooks)
     - cb0 only (semantic)

For each condition, generates speech and measures emotion similarity (emotion2vec)
and speaker similarity (WavLM x-vector) against references.

Usage:
    uv run python scripts/exp_token_swap.py \
        --esd_dir /home/daniel/data/processed/esd_24k/0017 \
        --output_dir data/experiments/token_swap \
        --num_pairs 5

    # Use a specific utterance pair
    uv run python scripts/exp_token_swap.py \
        --esd_dir /home/daniel/data/processed/esd_24k/0017 \
        --output_dir data/experiments/token_swap \
        --utterance_ids 0017_000001
"""

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import librosa
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


UTTERANCE_ID_OFFSET_ANGRY = 350  # ESD: neutral=1-350, angry=351-700


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--esd_dir", required=True,
                   help="ESD speaker dir (e.g. .../esd_24k/0017) with Angry/, Neutral/ subdirs")
    p.add_argument("--output_dir", default="data/experiments/token_swap")
    p.add_argument("--model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--text", default="Estou muito brabo com essa situação toda!",
                   help="Text to synthesize")
    p.add_argument("--language", default="Auto")
    p.add_argument("--num_pairs", type=int, default=5,
                   help="Number of parallel utterance pairs to use as references")
    p.add_argument("--utterance_ids", nargs="*", default=None,
                   help="Specific neutral utterance IDs (e.g. 0017_000001). "
                        "Overrides --num_pairs.")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--skip_metrics", action="store_true",
                   help="Skip emotion/speaker similarity computation")
    return p.parse_args()


def parse_transcripts(esd_dir: str) -> dict[str, tuple[str, str]]:
    """Parse {speaker}.txt → {utt_id: (text, emotion)}."""
    speaker_id = Path(esd_dir).name
    transcript_path = os.path.join(esd_dir, f"{speaker_id}.txt")
    transcripts = {}
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            utt_id, text, emotion = parts
            transcripts[utt_id] = (text, emotion)
    return transcripts


def find_parallel_pairs(
    esd_dir: str,
    transcripts: dict,
    num_pairs: int | None = None,
    utterance_ids: list[str] | None = None,
) -> list[dict]:
    """Find matching (neutral, angry) wav pairs with same text.

    ESD convention: utterance N in Neutral → utterance N+350 in Angry, same text.
    """
    speaker_id = Path(esd_dir).name

    if utterance_ids is not None:
        neutral_ids = utterance_ids
    else:
        neutral_ids = [
            f"{speaker_id}_{i:06d}" for i in range(1, (num_pairs or 5) + 1)
        ]

    pairs = []
    for nid in neutral_ids:
        num = int(nid.split("_")[1])
        aid = f"{speaker_id}_{num + UTTERANCE_ID_OFFSET_ANGRY:06d}"

        neutral_wav = _find_wav(esd_dir, "Neutral", nid)
        angry_wav = _find_wav(esd_dir, "Angry", aid)

        if neutral_wav is None or angry_wav is None:
            print(f"  WARN: skipping {nid} — missing wav (neutral={neutral_wav}, angry={angry_wav})")
            continue

        n_text = transcripts.get(nid, (None, None))[0]
        a_text = transcripts.get(aid, (None, None))[0]

        if n_text != a_text:
            print(f"  WARN: text mismatch for {nid}/{aid}: {n_text!r} vs {a_text!r}")

        pairs.append({
            "neutral_id": nid,
            "angry_id": aid,
            "neutral_wav": neutral_wav,
            "angry_wav": angry_wav,
            "text": n_text or a_text,
        })

    return pairs


def _find_wav(esd_dir: str, emotion: str, utt_id: str) -> str | None:
    candidates = list(Path(esd_dir, emotion).rglob(f"{utt_id}.wav"))
    return str(candidates[0]) if candidates else None


def encode_audio(wav_path: str, speech_tokenizer, tokenizer_sr: int) -> torch.Tensor:
    """Encode a single wav file → ref_code (T, 16)."""
    audio, sr = librosa.load(wav_path, sr=tokenizer_sr, mono=True)
    with torch.no_grad():
        enc = speech_tokenizer.encode(audio, sr=tokenizer_sr)
    return enc.audio_codes[0]


def build_prompt(
    tts: Qwen3TTSModel,
    wav_path: str,
    ref_text: str,
) -> list:
    """Build voice clone prompt items from a single reference."""
    return tts.create_voice_clone_prompt(
        ref_audio=wav_path,
        ref_text=ref_text,
        x_vector_only_mode=False,
    )


def swap_ref_code(prompt_items: list, new_ref_code: torch.Tensor) -> list:
    """Return a copy of prompt_items with ref_code replaced."""
    items = deepcopy(prompt_items)
    items[0].ref_code = new_ref_code
    return items


def selective_codebook_swap(
    neutral_code: torch.Tensor,
    angry_code: torch.Tensor,
    codebooks: list[int],
) -> torch.Tensor:
    """Swap specific codebook columns from angry into neutral ref_code.

    Truncates to min length if T differs.
    """
    min_t = min(neutral_code.shape[0], angry_code.shape[0])
    result = neutral_code[:min_t].clone()
    for cb in codebooks:
        result[:, cb] = angry_code[:min_t, cb]
    return result


def generate_condition(
    tts: Qwen3TTSModel,
    prompt_items: list,
    text: str,
    language: str,
    max_new_tokens: int,
) -> tuple:
    """Generate speech and return (wav_array, sample_rate)."""
    wavs, sr = tts.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=prompt_items,
        max_new_tokens=max_new_tokens,
    )
    if not wavs:
        return None, None
    return wavs[0], sr


def compute_metrics(wav_path: str, ref_angry_path: str, ref_neutral_path: str) -> dict:
    """Compute emotion and speaker similarity metrics."""
    from src.metrics.emotion import get_emotion
    from src.metrics.speaker import get_speaker_embedding

    emo_synth = torch.tensor(get_emotion(wav_path)["embedding"])
    emo_angry = torch.tensor(get_emotion(ref_angry_path)["embedding"])
    emo_neutral = torch.tensor(get_emotion(ref_neutral_path)["embedding"])

    spk_synth = get_speaker_embedding(wav_path)
    spk_neutral = get_speaker_embedding(ref_neutral_path)

    cos = torch.nn.functional.cosine_similarity
    return {
        "emo_sim_angry": cos(emo_synth, emo_angry, dim=0).item(),
        "emo_sim_neutral": cos(emo_synth, emo_neutral, dim=0).item(),
        "spk_sim_neutral": cos(spk_synth, spk_neutral).item(),
    }


# Each condition: (name, description, codebooks_to_swap)
# None means full swap; list means selective swap
CONDITIONS = [
    ("neutral_baseline", "Neutral ref_code (control)", None),
    ("angry_baseline", "Angry ref_code from angry prompt (angry control)", None),
    ("full_swap", "Full ref_code swap (angry codes, neutral speaker embed)", None),
    ("swap_cb1", "Swap codebook 1 only (highest τ norm)", [1]),
    ("swap_cb0", "Swap codebook 0 only (semantic)", [0]),
    ("swap_cb0_5", "Swap codebooks 0-5", list(range(6))),
    ("swap_cb1_5", "Swap codebooks 1-5 (skip semantic)", list(range(1, 6))),
    ("swap_cb6_15", "Swap codebooks 6-15 (lower layers)", list(range(6, 16))),
]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    transcripts = parse_transcripts(args.esd_dir)
    print(f"Loaded {len(transcripts)} transcript entries")

    pairs = find_parallel_pairs(
        args.esd_dir, transcripts,
        num_pairs=args.num_pairs,
        utterance_ids=args.utterance_ids,
    )
    print(f"Found {len(pairs)} parallel pairs")
    if not pairs:
        raise RuntimeError("No valid parallel pairs found")

    for p in pairs:
        print(f"  {p['neutral_id']} / {p['angry_id']}: {p['text']!r}")

    print(f"\nLoading model: {args.model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=device,
        dtype=torch.bfloat16,
    )

    all_results = []

    for pair_idx, pair in enumerate(pairs):
        print(f"\n{'=' * 70}")
        print(f"Pair {pair_idx + 1}/{len(pairs)}: {pair['neutral_id']} / {pair['angry_id']}")
        print(f"Text: {pair['text']!r}")
        print(f"{'=' * 70}")

        ref_text = pair["text"]

        # Build prompts from both emotions
        neutral_prompt = build_prompt(tts, pair["neutral_wav"], ref_text)
        angry_prompt = build_prompt(tts, pair["angry_wav"], ref_text)

        neutral_code = neutral_prompt[0].ref_code
        angry_code = angry_prompt[0].ref_code

        print(f"  Neutral ref_code: {neutral_code.shape}")
        print(f"  Angry ref_code:   {angry_code.shape}")

        # Token-level comparison
        min_t = min(neutral_code.shape[0], angry_code.shape[0])
        matching = (neutral_code[:min_t] == angry_code[:min_t])
        per_cb_match = matching.float().mean(dim=0)
        overall_match = matching.float().mean().item()
        print(f"  Token overlap (T={min_t}):")
        print(f"    Overall: {overall_match:.1%}")
        for cb in range(neutral_code.shape[1]):
            print(f"    CB{cb:2d}: {per_cb_match[cb].item():.1%}")

        pair_dir = os.path.join(args.output_dir, pair["neutral_id"])
        os.makedirs(pair_dir, exist_ok=True)

        pair_results = {
            "pair": pair,
            "neutral_code_shape": list(neutral_code.shape),
            "angry_code_shape": list(angry_code.shape),
            "token_overlap_overall": overall_match,
            "token_overlap_per_cb": per_cb_match.tolist(),
            "conditions": {},
        }

        for cond_name, cond_desc, cond_codebooks in CONDITIONS:
            print(f"\n  --- {cond_name}: {cond_desc} ---")

            if cond_name == "neutral_baseline":
                prompt = neutral_prompt
            elif cond_name == "angry_baseline":
                prompt = angry_prompt
            elif cond_name == "full_swap":
                prompt = swap_ref_code(neutral_prompt, angry_code)
            else:
                hybrid_code = selective_codebook_swap(
                    neutral_code, angry_code, cond_codebooks,
                )
                prompt = swap_ref_code(neutral_prompt, hybrid_code)

            wav, sr = generate_condition(
                tts, prompt, args.text, args.language, args.max_new_tokens,
            )

            if wav is None:
                print(f"    ERROR: no audio generated")
                pair_results["conditions"][cond_name] = {"error": "no audio"}
                continue

            duration = len(wav) / sr
            out_path = os.path.join(pair_dir, f"{cond_name}.wav")
            sf.write(out_path, wav, sr)
            print(f"    Saved: {out_path} ({duration:.1f}s)")

            cond_result = {
                "description": cond_desc,
                "wav_path": out_path,
                "duration_s": round(duration, 2),
            }

            if not args.skip_metrics:
                try:
                    metrics = compute_metrics(
                        out_path, pair["angry_wav"], pair["neutral_wav"],
                    )
                    cond_result["metrics"] = metrics
                    print(f"    emo_sim(angry):  {metrics['emo_sim_angry']:.4f}")
                    print(f"    emo_sim(neutral):{metrics['emo_sim_neutral']:.4f}")
                    print(f"    spk_sim(neutral):{metrics['spk_sim_neutral']:.4f}")
                except Exception as e:
                    print(f"    WARN: metrics failed: {e}")

            pair_results["conditions"][cond_name] = cond_result

        all_results.append(pair_results)

    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Condition':<20} {'emo_angry':>10} {'emo_neutral':>12} {'spk_sim':>10} {'dur':>6}")
    print("-" * 60)

    for pair_result in all_results:
        pair_id = pair_result["pair"]["neutral_id"]
        print(f"\n  Pair: {pair_id}")
        for cond_name in [c[0] for c in CONDITIONS]:
            cond = pair_result["conditions"].get(cond_name, {})
            m = cond.get("metrics", {})
            dur = cond.get("duration_s", 0)
            ea = m.get("emo_sim_angry", float("nan"))
            en = m.get("emo_sim_neutral", float("nan"))
            ss = m.get("spk_sim_neutral", float("nan"))
            print(f"  {cond_name:<20} {ea:>10.4f} {en:>12.4f} {ss:>10.4f} {dur:>5.1f}s")

    print("\nDone!")


if __name__ == "__main__":
    main()
