# Reprodução e análises

Execute os comandos na raiz do repositório com `PYTHONPATH=.` e o ambiente de `uv.lock`. Os caminhos dos corpora são definidos no [README principal](../../README.md). Este recorte inclui inferência, extração de direções, sweeps ESD/emoUERJ e análises de escores existentes. Ainda faltam receitas públicas de geração do fatorial, síntese cross-corpus e controles de norma casada, além das entradas anonimizadas do MOS.

## Direções e sweeps

Os arquivos `.pt` contêm tau, centroides, configuração e estatísticas. Para reconstruí-los a partir do ESD:

```bash
PYTHONPATH=. uv run python scripts/repro/extract_xvec_tau.py \
  --esd_dir "$ESD_ROOT" --speakers 0011 0014 0017 0020 \
  --emotions Angry Happy Sad --n_pairs 50 --output_dir data/tau_recomputed
PYTHONPATH=. uv run python scripts/repro/extract_xvec_tau.py \
  --esd_dir "$ESD_ROOT" --speakers 0017 \
  --emotions Angry Happy Sad --n_pairs 50 --output_dir data/tau_recomputed
PYTHONPATH=. uv run python scripts/repro/run_en2en_sweep.py
PYTHONPATH=. uv run python scripts/data/transcribe_emouerj.py
PYTHONPATH=. uv run python scripts/repro/run_ptbr_sweep.py
PYTHONPATH=. uv run python scripts/repro/analyze_gap4_eca_esim.py \
  --summaries en2en_summary.json ptbr_summary.json
PYTHONPATH=. uv run python scripts/repro/analyze_icassp_revision.py --sections objective \
  --output data/experiments/icassp_objective.json
```

A reanálise usa `data/experiments/gap4_eca_esim.parquet`, gerado pelo passe de classificação, e `results_long.parquet`/`results_long_ptbr.parquet`, gerados pelos sweeps. Alpha calibrado por falante maximiza a similaridade emocional na própria célula avaliada. Alpha selecionado apenas em inglês usa `avg4spk` e a grade comum `{0, 1, 1.5, 2, 2.5}`, transferindo angry=2.5, happy=1.5 e sad=2.5 ao português. Os testes verificam a independência dessa seleção em relação aos escores portugueses.

A opção `--selftest` de `analyze_gap4_eca_esim.py` calcula referências naturais de ECA e similaridade. `compute_gt_ceiling.py` e `compute_gt_ceiling_ptbr.py` também avaliam identidade, WER e UTMOS. O UTMOS permanece nas saídas históricas, embora não integre as tabelas atuais. Os sweeps aceitam `--combination convex` para interpolação com o centroide emocional, incluindo substituição em beta=1.

## Fatorial de condicionamento

A análise recebe 1.600 registros: quatro falantes × dez referências × dez textos × quatro condições. Em `balanced`, cada linha de `scores.jsonl` contém `key`, `speaker` como string, `ref`, `utt`, `condition`, `wav_path`, `sha256`, `eca_e2v`, `eca_sv`, `duration_s`, `wer_norm` e `spk_cos_sim_neutral_wavlm`. Os caminhos de áudio precisam ser acessíveis para conferir os hashes. As condições são `neutral_baseline`, `full_swap`, `xvec_swap` e `angry_baseline`.

```bash
PYTHONPATH=. uv run python scripts/repro/analyze_icassp_revision.py \
  --sections factorial --factorial-layout balanced \
  --factorial data/experiments/conditioning_factorial
```

Os falantes são 0011, 0013, 0017 e 0019; as referências usam índices 1–10 e os textos, 321–330. O bootstrap cruza referência e texto, condicionado aos quatro falantes observados. A sensibilidade mantém 380 unidades, retirando a referência 6 de 0011 e 0013 por discrepância lexical identificada antes da pontuação.

O leitor `historical` também aceita a execução original: `ar2_token_swap/{results,eca}.jsonl` e `conditioning_factorial_2026-09-06/{scores,sentinel_checks}.jsonl`, sob `data/experiments/`. Ele verifica as 12 sentinelas usadas para reaproveitar os três braços anteriores. Esses artefatos não acompanham o código. O gerador das quatro condições ainda não integra este recorte.

## CREMA-D e VERBO

```bash
PYTHONPATH=. uv run python scripts/repro/analyze_cross_corpus.py \
  --run data/experiments/cross_corpus_source_only_v2_exploratory
```

Entradas: `scores.jsonl` e `manifest_synthesis.jsonl`. `cross_corpus_support.py` define os campos, valida o pareamento e calcula os agregados. São 12 falantes, seis textos e três sementes por corpus, com 864 saídas e 648 contrastes. Falhas de síntese permanecem no denominador; a baseline neutra é compartilhada pelos três contrastes emocionais. O bootstrap cruza falante e texto após média das sementes e emoções.

A configuração em `configs/cross_corpus_source_only.json` documenta a grade, os hashes de tau e a seleção de alpha em inglês. Na execução reportada, as transcrições VERBO l5 de f4/f6 foram corrigidas de “de seu João” para “do seu João” após escuta do piloto. A correção afeta 144 linhas de manifesto, sem alterar os áudios de referência. A avaliação é exploratória. A aquisição dos corpora e a geração dos áudios ainda não estão incluídas.

## Escuta humana

A análise remove respostas de participantes que erraram sondas PT, blocos EN sem CEFR B1 ou superior e páginas com duração insuficiente. Notas baixas em gravações naturais e escores extremos são mantidos. Naturalidade não perguntada permanece ausente. O mapping das condições é associado depois da limpeza.

```bash
PYTHONPATH=. uv run python scripts/mos_power/analyze_mos.py \
  --submissions data/experiments/mos_field/submissions.jsonl \
  --manifest data/experiments/mos_stimuli/deploy/manifest.json \
  --mapping data/experiments/mos_stimuli/private/mapping.csv \
  --objective data/experiments/mos_stimuli/private/frozen_objective.csv \
  --out_dir data/experiments/mos_reanalysis --unblind
```

Sem `--unblind`, o comando produz somente os dados de limpeza. EMOS usa Wilcoxon pareado por enunciado e bootstrap de 10.000 réplicas; NMOS inclui não inferioridade com margem 0,5. Os alpha da escuta foram calibrados por falante antes da coleta, usando `single0017` em EN e `avg4spk` em PT-BR; essa avaliação não testa perceptualmente a seleção apenas em inglês.

Os dados brutos contêm identificadores pessoais e ficam fora do Git. Ainda é necessária uma versão anonimizada das respostas e dos artefatos dos estímulos para reprodução pública. A interface está no [repositório tts-eval](https://github.com/danielbrito91/tts-eval).

## Verificação local

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. uv run python scripts/repro/test_cross_corpus_source_only.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. uv run python scripts/repro/test_icassp_analysis.py
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/repro/check_release.py
```

Os testes usam dados artificiais para verificar seleção de alpha, pareamento, denominadores e bootstrap. `check_release.py` confere os arquivos enumerados em `release_files.txt`, sua sintaxe e dependências locais. Nenhum desses comandos executa síntese em GPU. As explorações de fine-tuning dependem do `finetuning/dataset.py` do Qwen3-TTS; consulte [third_party/qwen/README.md](../../third_party/qwen/README.md).
