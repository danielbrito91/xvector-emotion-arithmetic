import argparse
import json
import os

import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from qwen_tts import Qwen3TTSModel


def main():
    parser = argparse.ArgumentParser(description="Run inference with a LoRA-finetuned Qwen3-TTS model")
    parser.add_argument("--base_model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        help="Base model path or HF repo ID")
    parser.add_argument("--lora_path", required=True,
                        help="Path to the LoRA checkpoint directory (contains adapter_config.json)")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--speaker", required=True, help="Speaker name (e.g. 0017_angry)")
    parser.add_argument("--output", default="output.wav", help="Output wav file path")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Max new tokens (~40s at 500)")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Resolve base model path
    base_path = args.base_model_path
    if not os.path.isdir(base_path):
        base_path = snapshot_download(base_path)

    # Load base model
    tts = Qwen3TTSModel.from_pretrained(
        base_path,
        device_map=device,
        dtype=torch.bfloat16,
    )

    # Apply LoRA adapter to the talker
    tts.model.talker = PeftModel.from_pretrained(
        tts.model.talker,
        args.lora_path,
        is_trainable=False,
    )

    # Merge LoRA weights into the base model so the attribute structure is restored
    tts.model.talker = tts.model.talker.merge_and_unload()

    # Inject the trained speaker embedding into codec_embedding slot 3000
    spk_emb_path = os.path.join(args.lora_path, "speaker_embedding.pt")
    if os.path.exists(spk_emb_path):
        speaker_embedding = torch.load(spk_emb_path, map_location=device)
        weight = tts.model.talker.model.codec_embedding.weight
        tts.model.talker.model.codec_embedding.weight.data[3000] = (
            speaker_embedding[0].to(weight.device).to(weight.dtype)
        )
    else:
        print(f"Warning: speaker_embedding.pt not found at {spk_emb_path}")

    # Patch the config so the model knows about the speaker slot
    lora_config_path = os.path.join(args.lora_path, "config.json")
    if os.path.exists(lora_config_path):
        with open(lora_config_path) as f:
            lora_cfg = json.load(f)
        talker_cfg = lora_cfg.get("talker_config", {})
        spk_id = talker_cfg.get("spk_id", {})
        spk_is_dialect = talker_cfg.get("spk_is_dialect", {})
        tts.model.config.talker_config.spk_id.update(spk_id)
        tts.model.config.talker_config.spk_is_dialect.update(spk_is_dialect)
        tts.model.supported_speakers = tts.model.config.talker_config.spk_id.keys()
        tts.model.tts_model_type = "custom_voice"
        tts.model.config.tts_model_type = "custom_voice"

    wavs, sr = tts.generate_custom_voice(
        text=args.text,
        speaker=args.speaker,
        max_new_tokens=args.max_new_tokens,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sf.write(args.output, wavs[0], sr)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
