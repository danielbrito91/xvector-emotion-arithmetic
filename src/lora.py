# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# LoRA fine-tuning variant of sft_12hz.py.
# Only the talker's LoRA adapter and speaker embedding are saved per checkpoint.
import argparse
import json
import os

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from peft import LoraConfig, get_peft_model
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download
from transformers import AutoConfig

target_speaker_embedding = None


def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=args.output_model_path,
    )

    MODEL_PATH = args.init_model_path
    if not os.path.isdir(MODEL_PATH):
        MODEL_PATH = snapshot_download(MODEL_PATH)

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    # Apply LoRA to talker only.
    # Targets attention projections + codec_head (hidden→codec-0 logit) + lm_head
    # (the 15 code_predictor heads for acoustic codebooks 1-15).
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "codec_head",
        ] + [f"lm_head.{i}" for i in range(15)],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    qwen3tts.model.talker = get_peft_model(qwen3tts.model.talker, lora_config)
    qwen3tts.model.talker.print_trainable_parameters()

    # Freeze speaker encoder and speech tokenizer — only LoRA params need gradients
    for param in qwen3tts.model.speaker_encoder.parameters():
        param.requires_grad = False

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn
    )

    optimizer = AdamW(
        [p for p in qwen3tts.model.talker.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    model.train()

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                speaker_embedding = model.speaker_encoder(
                    ref_mels.to(model.device).to(model.dtype)
                ).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                # After LoRA wrapping, model.talker is a PeftModel, so
                # model.talker.model is the original Qwen3TTSTalkerForConditionalGeneration
                # and model.talker.model.model is the inner transformer with the embeddings.
                talker_inner = model.talker.model.model
                input_text_embedding = (
                    talker_inner.text_embedding(input_text_ids) * text_embedding_mask
                )
                input_codec_embedding = (
                    talker_inner.codec_embedding(input_codec_ids) * codec_embedding_mask
                )
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](
                        codec_ids[:, :, i]
                    )
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = (
                    model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
                )

                loss = outputs.loss + 0.3 * sub_talker_loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            os.makedirs(output_dir, exist_ok=True)

            # Save only the LoRA adapter weights
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.talker.save_pretrained(output_dir)

            # Save speaker embedding for inference
            torch.save(
                target_speaker_embedding,
                os.path.join(output_dir, "speaker_embedding.pt"),
            )

            # Save config with custom_voice speaker slot
            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {args.speaker_name: 3000}
            talker_config["spk_is_dialect"] = {args.speaker_name: False}
            config_dict["talker_config"] = talker_config
            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            accelerator.print(f"Saved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    train()
