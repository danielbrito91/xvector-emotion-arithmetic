"""
Build token-space emotion vectors from ESD data.

For a given ESD speaker, encodes all angry and neutral utterances through the
Qwen3-TTS speech tokenizer, looks up the codec embeddings in the talker's
embedding tables, and computes per-codebook-layer mean embeddings for each
emotion. Saves the emotion vector τ = mean(angry) − mean(neutral).

No training. Pure arithmetic on pre-existing representations.

Usage:
    uv run python scripts/extract_emotion_centroids.py \
        --esd_dir /home/daniel/data/processed/esd_24k/0017 \
        --output_path data/emotion_vectors/0017_angry.pt \
        --speaker_id 0017
"""

import argparse
import os
from pathlib import Path

import librosa
import torch
from qwen_tts import Qwen3TTSModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--esd_dir", required=True,
        help="ESD speaker directory (e.g. .../esd_24k/0017) containing Angry/ and Neutral/ subdirs",
    )
    p.add_argument("--output_path", default="data/emotion_vectors/0017_angry.pt")
    p.add_argument("--model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--speaker_id", default="0017")
    p.add_argument("--max_samples", type=int, default=0, help="0 = all")
    p.add_argument("--target_emotion", default="Angry", help="Target emotion directory name")
    p.add_argument("--baseline_emotion", default="Neutral", help="Baseline emotion directory name")
    return p.parse_args()


def collect_wav_paths(esd_dir: str, emotion: str) -> list[str]:
    """Collect wav files for a given emotion from the ESD speaker directory.

    Supports both flat ({speaker}/{emotion}/*.wav) and split-based
    ({speaker}/{emotion}/{train,test,evaluation}/*.wav) structures.
    """
    emotion_dir = os.path.join(esd_dir, emotion)
    if not os.path.isdir(emotion_dir):
        raise FileNotFoundError(f"Emotion directory not found: {emotion_dir}")

    wavs = sorted(str(p) for p in Path(emotion_dir).rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No wav files found in {emotion_dir}")
    return wavs


def encode_utterances(
    wav_paths: list[str],
    speech_tokenizer,
    tokenizer_sr: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Encode wav files to codec codes via the speech tokenizer.

    Returns list of tensors, each (T_i, num_codebooks).
    """
    all_codes = []
    for i, path in enumerate(wav_paths):
        audio, sr = librosa.load(path, sr=tokenizer_sr, mono=True)
        if len(audio) < 1600:
            continue

        with torch.no_grad():
            enc = speech_tokenizer.encode(audio, sr=tokenizer_sr)
            codes = enc.audio_codes[0]  # (T, num_codebooks)

        all_codes.append(codes.to(device))

        if (i + 1) % 50 == 0:
            print(f"  Encoded {i + 1}/{len(wav_paths)}")

    return all_codes


def compute_embedding_centroid(
    all_codes: list[torch.Tensor],
    main_embedding: torch.nn.Embedding,
    code_predictor_embeddings: torch.nn.ModuleList,
    num_codebooks: int = 16,
) -> dict:
    """Compute mean embedding per codebook layer across all utterances.

    Returns dict with per_layer means (list of 16 [D] tensors),
    summed mean ([D] tensor), and total frame count.
    """
    D = main_embedding.weight.shape[1]
    layer_sums = [
        torch.zeros(D, device=main_embedding.weight.device, dtype=torch.float32)
        for _ in range(num_codebooks)
    ]
    total_frames = 0

    with torch.no_grad():
        for codes in all_codes:
            T = codes.shape[0]
            total_frames += T

            for k in range(num_codebooks):
                code_ids = codes[:, k]  # (T,)
                if k == 0:
                    emb = main_embedding(code_ids)  # (T, D)
                else:
                    emb = code_predictor_embeddings[k - 1](code_ids)
                layer_sums[k] += emb.float().sum(dim=0)

    per_layer = [s / total_frames for s in layer_sums]

    summed = sum(layer_sums) / total_frames

    return {
        "per_layer": per_layer,
        "summed": summed,
        "num_frames": total_frames,
    }


def main():
    args = parse_args()

    print(f"Loading model: {args.model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=device,
        dtype=torch.bfloat16,
    )

    speech_tokenizer = tts.model.speech_tokenizer
    tokenizer_sr = speech_tokenizer.feature_extractor.sampling_rate
    main_embedding = tts.model.talker.get_input_embeddings()
    code_predictor_embeddings = tts.model.talker.code_predictor.get_input_embeddings()
    num_codebooks = tts.model.config.talker_config.num_code_groups

    print(f"  Tokenizer sample rate: {tokenizer_sr}")
    print(f"  Num codebooks: {num_codebooks}")
    print(f"  Embedding dim: {main_embedding.weight.shape[1]}")

    # --- Target emotion ---
    print(f"\nCollecting {args.target_emotion} utterances from {args.esd_dir}/{args.target_emotion}/")
    target_wavs = collect_wav_paths(args.esd_dir, args.target_emotion)
    if args.max_samples > 0:
        target_wavs = target_wavs[:args.max_samples]
    print(f"  Found {len(target_wavs)} files")

    print(f"Encoding {args.target_emotion} utterances...")
    target_codes = encode_utterances(target_wavs, speech_tokenizer, tokenizer_sr, tts.device)
    print(f"  Encoded {len(target_codes)} utterances")

    print(f"Computing {args.target_emotion} centroid...")
    target_centroid = compute_embedding_centroid(
        target_codes, main_embedding, code_predictor_embeddings, num_codebooks,
    )
    print(f"  Total frames: {target_centroid['num_frames']}")

    # --- Baseline emotion ---
    print(f"\nCollecting {args.baseline_emotion} utterances from {args.esd_dir}/{args.baseline_emotion}/")
    baseline_wavs = collect_wav_paths(args.esd_dir, args.baseline_emotion)
    if args.max_samples > 0:
        baseline_wavs = baseline_wavs[:args.max_samples]
    print(f"  Found {len(baseline_wavs)} files")

    print(f"Encoding {args.baseline_emotion} utterances...")
    baseline_codes = encode_utterances(baseline_wavs, speech_tokenizer, tokenizer_sr, tts.device)
    print(f"  Encoded {len(baseline_codes)} utterances")

    print(f"Computing {args.baseline_emotion} centroid...")
    baseline_centroid = compute_embedding_centroid(
        baseline_codes, main_embedding, code_predictor_embeddings, num_codebooks,
    )
    print(f"  Total frames: {baseline_centroid['num_frames']}")

    # --- τ = target − baseline ---
    print(f"\nComputing τ = {args.target_emotion} − {args.baseline_emotion}")
    tau_per_layer = [
        (t - b) for t, b in zip(target_centroid["per_layer"], baseline_centroid["per_layer"])
    ]
    tau_summed = target_centroid["summed"] - baseline_centroid["summed"]

    norms = [t.norm().item() for t in tau_per_layer]
    print(f"  Per-layer L2 norms: {['%.4f' % n for n in norms]}")
    print(f"  Summed vector L2 norm: {tau_summed.norm().item():.4f}")

    # --- Save ---
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    save_dict = {
        "tau_per_layer": [t.cpu() for t in tau_per_layer],
        "tau_summed": tau_summed.cpu(),
        "target_centroid_per_layer": [c.cpu() for c in target_centroid["per_layer"]],
        "baseline_centroid_per_layer": [c.cpu() for c in baseline_centroid["per_layer"]],
        "target_centroid_summed": target_centroid["summed"].cpu(),
        "baseline_centroid_summed": baseline_centroid["summed"].cpu(),
        "target_frames": target_centroid["num_frames"],
        "baseline_frames": baseline_centroid["num_frames"],
        "target_emotion": args.target_emotion,
        "baseline_emotion": args.baseline_emotion,
        "speaker_id": args.speaker_id,
        "num_codebooks": num_codebooks,
        "embedding_dim": main_embedding.weight.shape[1],
    }
    torch.save(save_dict, args.output_path)
    print(f"\nSaved emotion vector to {args.output_path}")


if __name__ == "__main__":
    main()
