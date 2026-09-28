# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adaptado de QwenLM/Qwen3-TTS, commit 1ab0dd7 (finetuning/sft_12hz.py).
# Alterações: alinhamento de codec_mask com input_embeddings[:, :-1], opção
# --attn, resolução de modelos HF, diretório de logs e sonda opcional de
# gradiente em --grad_probe_tau. A sonda mede o gradiente do embedding injetado
# sem incluí-lo no otimizador. A perda do sub-talker mantém peso 1.0.
# Requer finetuning/dataset.py do upstream no PYTHONPATH.
import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
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
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    # sdpa permite executar sem FlashAttention 2.
    parser.add_argument("--attn", type=str, default="flash_attention_2",
                        help="attn_implementation: flash_attention_2 (upstream default) or sdpa")
    # Sonda opcional do gradiente no embedding de falante.
    parser.add_argument("--grad_probe_tau", type=str, default=None,
                        help="Path to a τ artifact (.pt with key 'tau', or a raw tensor). "
                             "Enables logging cos(−∂L/∂spk_emb, τ̂) per micro-step to "
                             "<output_model_path>/grad_probe.jsonl.")
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard", project_dir=args.output_model_path)

    MODEL_PATH = args.init_model_path
    if not os.path.isdir(MODEL_PATH):
        MODEL_PATH = snapshot_download(MODEL_PATH)

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn,
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    probe_tau = None
    probe_log_path = None
    if args.grad_probe_tau:
        art = torch.load(args.grad_probe_tau, map_location="cpu", weights_only=False)
        probe_tau = (art["tau"] if isinstance(art, dict) else art).flatten().float()
        probe_tau_hat = probe_tau / probe_tau.norm()
        os.makedirs(args.output_model_path, exist_ok=True)
        probe_log_path = os.path.join(args.output_model_path, "grad_probe.jsonl")
        accelerator.print(f"[grad_probe] τ dim={probe_tau.numel()} ‖τ‖={probe_tau.norm():.3f} "
                          f"→ {probe_log_path}")

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    model.train()

    for epoch in range(num_epochs):
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

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                # Folha com gradiente para a sonda; fica fora do otimizador.
                probe_emb = None
                if probe_tau is not None:
                    assert probe_tau.numel() == speaker_embedding.shape[-1], (
                        f"grad_probe: τ dim {probe_tau.numel()} != speaker embedding dim "
                        f"{speaker_embedding.shape[-1]} — extract τ from this model size."
                    )
                    speaker_embedding = speaker_embedding.clone().requires_grad_(True)
                    probe_emb = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                hidden_states = outputs.hidden_states[0][-1]
                # A máscara acompanha input_embeddings[:, :-1, :].
                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + sub_talker_loss

                accelerator.backward(loss)

                # Projeção da direção de descida sobre tau normalizado.
                if probe_emb is not None and probe_emb.grad is not None and accelerator.is_main_process:
                    g = probe_emb.grad.detach().float().cpu()           # (b, d)
                    tau_hat = probe_tau_hat
                    desc = -g                                            # descent direction
                    cos_i = torch.nn.functional.cosine_similarity(
                        desc, tau_hat.unsqueeze(0).expand_as(desc), dim=1)
                    proj_i = desc @ tau_hat                              # signed magnitude along τ̂
                    with open(probe_log_path, "a") as pf:
                        pf.write(json.dumps({
                            "epoch": epoch, "step": step,
                            "loss": round(loss.item(), 4),
                            "cos_desc_tau_mean": round(cos_i.mean().item(), 6),
                            "cos_desc_tau_std": round(cos_i.std(unbiased=False).item(), 6),
                            "proj_desc_tau_mean": round(proj_i.mean().item(), 8),
                            "grad_norm_mean": round(g.norm(dim=1).mean().item(), 8),
                        }) + "\n")

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {
                args.speaker_name: 3000
            }
            talker_config["spk_is_dialect"] = {
                args.speaker_name: False
            }
            config_dict["talker_config"] = talker_config

            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            weight = state_dict['talker.model.codec_embedding.weight']
            state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)

if __name__ == "__main__":
    train()
