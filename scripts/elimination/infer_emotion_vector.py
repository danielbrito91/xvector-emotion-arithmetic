"""
Apply token-space emotion vectors to Qwen3-TTS voice cloning at inference time.

Loads a pre-computed emotion vector (from extract_emotion_centroids.py) and injects
it into the voice cloning ICL prompt. The emotion vector is added to the reference
audio's codec embeddings AFTER lookup from the embedding tables but BEFORE they
enter the LLM backbone.

No training. No weight modification. Pure inference-time arithmetic.

Usage:
    # Single alpha (ref_text MUST be the exact transcript of the reference audio)
    uv run python scripts/infer_emotion_vector.py \
        --ref_audio data/ref/dani-neutro.wav \
        --ref_text "Exact transcript of dani-neutro.wav goes here" \
        --text "Text to synthesize" \
        --emotion_vector data/emotion_vectors/0017_angry.pt \
        --alpha 1.0 \
        --output output_angry_alpha1.0.wav

    # Alpha sweep
    uv run python scripts/infer_emotion_vector.py \
        --ref_audio data/ref/dani-neutro.wav \
        --ref_text "Exact transcript of dani-neutro.wav goes here" \
        --text "Text to synthesize" \
        --emotion_vector data/emotion_vectors/0017_angry.pt \
        --alpha 0.0 0.5 1.0 1.5 2.0 3.0 \
        --output data/experiments/emotion_vector_sweep/
"""

import argparse
import os
from functools import wraps

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_audio", required=True)
    p.add_argument("--ref_text", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--emotion_vector", required=True,
                   help="Path to .pt file from extract_emotion_centroids.py")
    p.add_argument("--alpha", nargs="+", type=float, default=[1.0])
    p.add_argument("--output", default="output_emotion.wav",
                   help="Output path. If multiple alphas, treated as directory.")
    p.add_argument("--model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--language", default="Auto")
    p.add_argument("--mode", choices=["summed", "per_layer"], default="summed",
                   help="'summed': single τ on combined embedding. "
                        "'per_layer': per-codebook τ_k before summation.")
    p.add_argument("--codebooks", nargs="*", type=int, default=None,
                   help="Apply τ only to these codebook indices (0-15). "
                        "Default: all. Example: --codebooks 1 to target only codebook 1.")
    p.add_argument("--non_streaming_mode", action="store_true", default=False)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    return p.parse_args()


def patch_generate_icl_prompt(
    core_model,
    tau_per_layer: list[torch.Tensor],
    tau_summed: torch.Tensor,
    alpha: float,
    mode: str,
    codebooks: list[int] | None = None,
):
    """Monkey-patch generate_icl_prompt on the Qwen3TTSForConditionalGeneration instance.

    Reproduces the original codec embedding computation from modeling_qwen3_tts.py
    lines 1968-2019 with the emotion vector injected between embedding lookup and
    LLM input.

    Patching the instance attribute means self.generate_icl_prompt(...) inside
    generate() will call our function (without passing self — we close over
    core_model instead).
    """
    original_fn = core_model.generate_icl_prompt
    talker = core_model.talker
    config = core_model.config
    device = talker.device
    dtype = next(talker.parameters()).dtype
    num_code_groups = talker.config.num_code_groups

    @wraps(original_fn)
    def patched(text_id, ref_id, ref_code, tts_pad_embed, tts_eos_embed, non_streaming_mode):
        # text embed: (ref_id ++ text_id ++ eos) → [1, T1, D]
        text_embed = talker.text_projection(
            talker.get_text_embeddings()(torch.cat([ref_id, text_id], dim=-1))
        )
        text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)

        # codec embed per codebook layer
        codec_embeds = []
        for i in range(num_code_groups):
            if i == 0:
                emb = talker.get_input_embeddings()(ref_code[:, :1])   # (T, 1, D)
            else:
                emb = talker.code_predictor.get_input_embeddings()[i - 1](ref_code[:, i:i + 1])
            codec_embeds.append(emb)

        # --- INJECT EMOTION VECTOR ---
        active_codebooks = set(codebooks) if codebooks is not None else set(range(num_code_groups))

        if mode == "per_layer":
            for k in active_codebooks:
                tau_k = tau_per_layer[k].to(device=device, dtype=dtype)
                codec_embeds[k] = codec_embeds[k] + alpha * tau_k  # broadcasts (D,) → (T, 1, D)

        # sum across codebook layers: (T, 16, D) → (T, D) → (1, T, D)
        codec_embed = torch.cat(codec_embeds, dim=1).sum(1).unsqueeze(0)

        if mode == "summed":
            tau_s = tau_summed.to(device=device, dtype=dtype)
            codec_embed = codec_embed + alpha * tau_s  # broadcasts (D,) → (1, T, D)

        # prepend codec_bos
        codec_embed = torch.cat([
            talker.get_input_embeddings()(
                torch.tensor([[config.talker_config.codec_bos_id]],
                             device=device, dtype=text_id.dtype)
            ),
            codec_embed,
        ], dim=1)

        # --- rest identical to original ---
        text_lens = text_embed.shape[1]
        codec_lens = codec_embed.shape[1]

        if non_streaming_mode:
            icl_input_embed = text_embed + talker.get_input_embeddings()(
                torch.tensor(
                    [[config.talker_config.codec_pad_id] * text_lens],
                    device=device, dtype=text_id.dtype,
                )
            )
            icl_input_embed = torch.cat([icl_input_embed, codec_embed + tts_pad_embed], dim=1)
            return icl_input_embed, tts_pad_embed
        else:
            if text_lens > codec_lens:
                return text_embed[:, :codec_lens] + codec_embed, text_embed[:, codec_lens:]
            else:
                text_embed = torch.cat(
                    [text_embed] + [tts_pad_embed] * (codec_lens - text_lens), dim=1
                )
                return text_embed + codec_embed, tts_pad_embed

    core_model.generate_icl_prompt = patched
    return original_fn  # caller can restore if needed


def main():
    args = parse_args()

    # Load emotion vector
    print(f"Loading emotion vector from {args.emotion_vector}")
    emo = torch.load(args.emotion_vector, map_location="cpu", weights_only=True)
    tau_per_layer = emo["tau_per_layer"]
    tau_summed = emo["tau_summed"]
    print(f"  Speaker: {emo.get('speaker_id', '?')}, "
          f"Emotion: {emo.get('target_emotion', '?')}")
    print(f"  Dim: {emo.get('embedding_dim', '?')}, "
          f"τ_summed L2: {tau_summed.norm().item():.4f}")
    per_layer_norms = [t.norm().item() for t in tau_per_layer]
    print(f"  Per-layer L2 norms: {['%.4f' % n for n in per_layer_norms]}")

    # Load model
    print(f"\nLoading model: {args.model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=device,
        dtype=torch.bfloat16,
    )

    # Build voice clone prompt once (reuse across alpha values)
    print(f"\nBuilding voice clone prompt from {args.ref_audio}")
    prompt_items = tts.create_voice_clone_prompt(
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        x_vector_only_mode=False,
    )

    is_sweep = len(args.alpha) > 1
    if is_sweep:
        os.makedirs(args.output, exist_ok=True)

    for alpha in args.alpha:
        print(f"\n{'=' * 60}")
        print(f"Generating with α = {alpha}, mode = {args.mode}")
        print(f"{'=' * 60}")

        original_fn = patch_generate_icl_prompt(
            tts.model, tau_per_layer, tau_summed,
            alpha=alpha, mode=args.mode, codebooks=args.codebooks,
        )

        try:
            wavs, sr = tts.generate_voice_clone(
                text=args.text,
                language=args.language,
                voice_clone_prompt=prompt_items,
                non_streaming_mode=args.non_streaming_mode,
                max_new_tokens=args.max_new_tokens,
            )
        finally:
            tts.model.generate_icl_prompt = original_fn

        if not wavs:
            print(f"  ERROR: No audio generated for α={alpha}")
            continue

        audio = wavs[0]
        duration = len(audio) / sr

        if is_sweep:
            out_path = os.path.join(args.output, f"alpha_{alpha:.2f}.wav")
        else:
            out_path = args.output

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        sf.write(out_path, audio, sr)
        print(f"  Saved: {out_path} ({duration:.1f}s)")

    print("\nDone!")


if __name__ == "__main__":
    main()
