# Training-Free Cross-Lingual Emotion Control in LM-TTS via Speaker-Channel Editing

Código do estudo de controle emocional em Qwen3-TTS-12Hz-1.7B-Base, por Daniel O. Brito, Sidney E. Leal e Arnaldo Candido Junior. Esta versão contém o método e parte das análises do manuscrito revisado em setembro de 2026; a disponibilização das receitas experimentais está em andamento. O [preprint público, arXiv v2](https://arxiv.org/abs/2606.05367v2), contém a versão anterior do estudo.

O método extrai direções emocionais de gravações inglesas e as aplica ao embedding de falante de uma voz alvo, mantendo os tokens da referência neutra:

```python
tau = mean_s(centroid(s, emotion) - centroid(s, neutral))
x_edited = x_target_neutral + alpha * tau
```

O fatorial neutro/raiva mostra maior efeito médio do embedding, com contribuição conjunta dos tokens e inversão da ordem em um dos quatro falantes. As direções transferem controle a vozes inglesas e brasileiras. A avaliação cobre ESD, emoUERJ, CREMA-D e VERBO; os dois últimos são avaliações exploratórias com força selecionada apenas em inglês. O teste de escuta usa forças calibradas por falante, fixadas antes da coleta.

## Instalação

Requer Python 3.12+, `uv` e `ffmpeg`. A síntese usa PyTorch/CUDA. O ambiente Python está fixado em `uv.lock`.

```bash
uv sync --locked
uv run hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./Qwen3-TTS-12Hz-1.7B-Base
uv run hf download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ./Qwen3-TTS-Tokenizer-12Hz
export PYTHONPATH=.
```

As métricas baixam seus próprios modelos no primeiro uso: emotion2vec+ large, SenseVoiceSmall, WavLM-base-plus-sv e Whisper-large-v3. Os scripts históricos também calculam UTMOSv2, que não integra as tabelas do manuscrito atual.

## Inferência

As seis direções em `data/tau/` acompanham o código. Cada `.pt` contém o vetor de 2048 dimensões, centroides e metadados de extração.

```bash
uv run python scripts/deploy/emotionize_audio.py \
  --input data/reference.wav --output data/angry.wav \
  --emotion angry --tau-variant avg4spk --alpha 2.5 \
  --ref-text "Transcrição da referência." --text "Texto a sintetizar."
```

Sem `--ref-text`, Whisper transcreve a referência. Sem `--text`, o sistema sintetiza sua transcrição. A receita selecionada apenas em inglês usa `avg4spk`, com alpha 2.5 para angry/sad e 1.5 para happy. Esses valores são pontos de operação do estudo, sem garantia de intensidade uniforme em qualquer voz.

## Dados e reprodução

ESD fornece as direções, com 50 gravações por emoção e falante: 0011, 0014, 0017 e 0020. O comparador de fonte única usa 0017. A avaliação inglesa usa 0013 e 0019; a brasileira usa m03, m04 e w04 do emoUERJ. CREMA-D e VERBO acrescentam 12 falantes por corpus.

```bash
export DATA_ROOT="$HOME/data/processed"
export ESD_ROOT="$DATA_ROOT/esd_24k"
export EMOUERJ_ROOT="$DATA_ROOT/emouerj_24k"
export ESD_RAW_ROOT="$HOME/data/external/Emotional Speech Dataset (ESD)/Emotion Speech Dataset"
```

O ESD usa subpastas de falante/emoção; o emoUERJ usa arquivos como `m03a01.wav` na raiz reamostrada. `scripts/data/resample_esd.py --help` descreve a preparação do ESD. Corpora, áudios sintetizados, checkpoints e respostas individuais do MOS ficam fora do Git.

| Resultado | Disponível neste recorte |
| --- | --- |
| Inferência e extração de direções | `scripts/deploy/emotionize_audio.py`, `scripts/repro/extract_xvec_tau.py` |
| ESD/emoUERJ: sweeps e Tabela 2 | `run_en2en_sweep.py`, `run_ptbr_sweep.py`, `analyze_gap4_eca_esim.py`, `analyze_icassp_revision.py` |
| Fatorial tokens × embedding, Tabela 1 | Análise de escores existentes em `analyze_icassp_revision.py`; geração das quatro condições pendente |
| CREMA-D/VERBO, Tabela 2 | Análise de manifestos e escores existentes em `analyze_cross_corpus.py`; aquisição e síntese pendentes |
| EMOS/NMOS, Tabela 3 | `scripts/mos_power/analyze_mos.py`; entradas públicas anonimizadas pendentes |
| Controles de norma casada | Receita ainda pendente neste recorte |

Os nomes sem diretório estão em `scripts/repro/`. O [guia de reprodução](scripts/repro/README.md) descreve entradas e comandos. A publicação parcial não permite reproduzir todas as tabelas a partir dos corpora. As explorações já publicadas em `scripts/elimination/`, `src/lora.py` e `third_party/qwen/` permanecem disponíveis.

## Organização

```text
src/                    aritmética, síntese, dados e métricas
scripts/data/           preparação dos corpora
scripts/deploy/         inferência a partir de uma referência
scripts/repro/          experimentos, análises e configurações
scripts/mos_power/      análise da escuta
third_party/qwen/       adaptação do fine-tuning Qwen3-TTS
data/tau/               direções emocionais pré-calculadas
```

## Citação

A referência abaixo identifica o preprint público v2. Seu título e autoria antecedem o manuscrito revisado descrito nesta versão do código.

```bibtex
@misc{brito2026taskvector,
  title = {Task-Vector Arithmetic for Emotional Expressivity Control in Language-Model-Based Text-to-Speech},
  author = {Brito, Daniel Oliveira de and Candido Junior, Arnaldo},
  year = {2026},
  eprint = {2606.05367},
  archivePrefix = {arXiv},
  primaryClass = {cs.SD},
  url = {https://arxiv.org/abs/2606.05367v2}
}
```
