# Fine-tuning Qwen3-TTS

`sft_12hz.py` adapta o [código QwenLM/Qwen3-TTS no commit 1ab0dd7](https://github.com/QwenLM/Qwen3-TTS/tree/1ab0dd7/finetuning), sob Apache-2.0. Ele serve às explorações em pesos e à sonda de gradiente. As direções distribuídas em `data/tau/` são diferenças de centroides extraídos pelo encoder de falante; não são diferenças de pesos produzidas por este treino.

Alterações em relação ao upstream:

- Alinhamento de `codec_mask[:, :-1]` com os estados de `input_embeddings[:, :-1]`.
- Seleção da implementação de atenção por `--attn`, resolução de modelos Hugging Face e diretório de logs do TensorBoard.
- Sonda opcional `--grad_probe_tau` que registra a projeção do gradiente do embedding de falante sobre tau. O embedding continua fora do otimizador. A dimensão de tau precisa corresponder à do modelo.

O peso da perda do sub-talker permanece 1.0. O script requer `finetuning/dataset.py` do upstream no `PYTHONPATH` e o pacote `qwen_tts` instalado.

```bash
PYTHONPATH=/path/to/Qwen3-TTS/finetuning \
  uv run python third_party/qwen/sft_12hz.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --train_jsonl data/esd/0017_angry_coded.jsonl \
    --output_model_path output_angry \
    --speaker_name speaker_angry \
    --batch_size 2 --lr 2e-6 --num_epochs 3
```

O padrão de atenção é `flash_attention_2`; use `--attn sdpa` quando FlashAttention 2 não estiver disponível.
