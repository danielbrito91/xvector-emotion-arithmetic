"""Synthesize from a full-FT Qwen3-TTS checkpoint via generate_custom_voice.

Each sft_12hz.py checkpoint dir is a complete model (base copied in, weights
overwritten, config patched with tts_model_type=custom_voice + spk_id), so we
just load the dir directly and synthesize with the trained speaker slot.
"""
import argparse
import os

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint dir (contains model.safetensors + config.json)")
    p.add_argument("--speaker", required=True, help="trained speaker slot, e.g. 0017_angry")
    p.add_argument("--text", default="She said she would be here by noon, but the train was late.")
    p.add_argument("--output", required=True)
    p.add_argument("--max_new_tokens", type=int, default=500)
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tts = Qwen3TTSModel.from_pretrained(args.ckpt, device_map=device, dtype=torch.bfloat16)

    # Defensive: make sure the speaker slot is registered for custom-voice synthesis.
    tts.model.tts_model_type = "custom_voice"
    tts.model.config.tts_model_type = "custom_voice"
    tts.model.supported_speakers = tts.model.config.talker_config.spk_id.keys()

    wavs, sr = tts.generate_custom_voice(text=args.text, speaker=args.speaker, max_new_tokens=args.max_new_tokens)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sf.write(args.output, wavs[0], sr)
    print(f"Saved {args.output}  (sr={sr}, dur={len(wavs[0])/sr:.1f}s)")


if __name__ == "__main__":
    main()
