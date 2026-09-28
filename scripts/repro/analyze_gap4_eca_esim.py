"""Calcula ECA e similaridade emocional com emotion2vec+ e SenseVoice.

Avalia sínteses existentes e gravações naturais pareadas. Reporta argmax completo
e restrito a neutral/angry/happy/sad. Antes de pontuar, confere uma amostra das
similaridades gravadas; --selftest executa a checagem e a avaliação do GT."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

EXP_DIR = Path('data/experiments')
DEFAULT_SUMMARIES = [
    'en2en_summary.json',
    'en2en_summary_convex.json',
    'ptbr_summary.json',
    'ptbr_summary_convex.json',
]
ESD_CLASSES = ['angry', 'happy', 'sad', 'neutral']
EECS_TOL = 0.02  # tolerância do check de integridade contra o valor gravado


def _e2v_label_probs(path: str) -> tuple[str, dict[str, float], torch.Tensor]:
    """emotion2vec_plus_large → (label, probs por classe em inglês, embedding)."""
    from src.metrics.emotion import get_emotion

    r = get_emotion(path)
    probs = {lab.split('/')[-1]: float(s) for lab, s in zip(r['label'], r['scores'])}
    label = max(probs, key=probs.get)
    return label, probs, torch.tensor(r['embedding'])


def _restrict(probs: dict[str, float], classes: list[str]) -> dict[str, float]:
    sub = {c: probs.get(c, 0.0) for c in classes}
    tot = sum(sub.values())
    if tot <= 0:
        return {c: float('nan') for c in classes}
    return {c: v / tot for c, v in sub.items()}


def _sv_lang(lang: str) -> str:
    """SenseVoice cobre auto/zh/en/yue/ja/ko — PT-BR entra como `auto` (fora do domínio)."""
    return 'en' if lang.lower().startswith('en') else 'auto'


def score_wav(path: str, lang: str) -> dict:
    from src.metrics.emotion_sv import predict as sv_predict

    e2v_label, e2v_probs, e2v_emb = _e2v_label_probs(path)
    sv = sv_predict(path, language=_sv_lang(lang))
    return {
        'e2v_label': e2v_label,
        'e2v_probs': e2v_probs,
        'e2v_emb': e2v_emb,
        'sv_label': sv['label'],
        'sv_probs': sv['probs'],
        'sv_prob_unk': sv['prob_unk'],
        'sv_emb': sv['embedding'],
    }


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))


def build_rows(summaries: list[str]) -> list[dict]:
    rows: list[dict] = []
    for name in summaries:
        path = EXP_DIR / name
        if not path.exists():
            print(f'[!] summary ausente, pulando: {path}')
            continue
        data = json.loads(path.read_text())
        for cell_key, cell in data['per_cell'].items():
            cfg = cell['config']
            target, tau_variant, emotion = cell_key.split('__')
            # `config.language` é o parâmetro de síntese do Qwen ("Auto" em todas as
            # células) — a língua real do bloco está em `wer_language`.
            lang = cfg.get('wer_language') or ('pt' if 'ptbr' in name else 'en')
            combination = cfg.get('combination') or 'additive'
            best = cell['aggregates'].get('best_alpha')
            base = cell['aggregates'].get('baseline_alpha')
            for utt, sent in cell['per_sentence'].items():
                for cond_name, cond in sent['conditions'].items():
                    wav = cond.get('wav_path')
                    if not wav:
                        continue
                    rows.append({
                        'summary': name,
                        'cell': cell_key,
                        'target': target,
                        'tau_variant': tau_variant,
                        'emotion': emotion.lower(),
                        'language': lang,
                        'combination': combination,
                        'utt': utt,
                        'cond': cond_name,
                        'alpha': cond.get('alpha'),
                        'is_baseline': cond.get('alpha') == base,
                        'is_best': cond.get('alpha') == best,
                        'wav_path': wav,
                        'gt_emo_audio': sent['gt_emo_audio'],
                        'ref_audio': sent['ref_audio'],
                        'eecs_stored': (cond.get('metrics') or {}).get('emo_cos_sim_gt'),
                    })
    return rows


def selftest(rows: list[dict], ceiling_path: str | None = None) -> None:
    """Calcula referências naturais de ECA e similaridade emocional por língua e emoção."""
    print('\n=== teto de ECA no GT (por língua × emoção × encoder) ===')
    ceiling: dict[str, dict] = {}
    seen: dict[tuple[str, str], set[str]] = {}
    for r in rows:
        seen.setdefault((r['language'], r['emotion']), set()).add(r['gt_emo_audio'])
    neutral_refs: dict[str, set[str]] = {}
    for r in rows:
        neutral_refs.setdefault(r['language'], set()).add(r['ref_audio'])

    print(
        f'{"lang":5s} {"emoção":8s} {"n":>4s}  {"ECA e2v":>8s} {"ECA sv":>7s} '
        f'{"E-SIM e2v gt~ref":>17s} {"E-SIM sv gt~ref":>16s}'
    )
    for (lang, emo), files in sorted(seen.items()):
        files = sorted(files)
        hits_e2v = hits_sv = 0
        embs_e2v, embs_sv = [], []
        for f in files:
            s = score_wav(f, lang)
            hits_e2v += int(
                max(
                    _restrict(s['e2v_probs'], ESD_CLASSES),
                    key=_restrict(s['e2v_probs'], ESD_CLASSES).get,
                )
                == emo
            )
            hits_sv += int(
                max(
                    _restrict(s['sv_probs'], ESD_CLASSES),
                    key=_restrict(s['sv_probs'], ESD_CLASSES).get,
                )
                == emo
            )
            embs_e2v.append(s['e2v_emb'])
            embs_sv.append(s['sv_emb'])
        # piso: quanto o próprio GT neutro já pontua contra o GT emocional
        refs = sorted(neutral_refs[lang])[: len(files)]
        floor_e2v, floor_sv = [], []
        for f_ref, e_gt, s_gt in zip(refs, embs_e2v, embs_sv):
            s = score_wav(f_ref, lang)
            floor_e2v.append(cos(s['e2v_emb'], e_gt))
            floor_sv.append(cos(s['sv_emb'], s_gt))
        n = len(files)
        print(
            f'{lang:5s} {emo:8s} {n:4d}  {hits_e2v / n:8.2f} {hits_sv / n:7.2f} '
            f'{sum(floor_e2v) / len(floor_e2v):17.3f} {sum(floor_sv) / len(floor_sv):16.3f}'
        )

        # teto de E-SIM: similaridade média entre GTs emocionais distintos do bloco
        # entre textos; referência empírica, sem constituir um limite matemático.
        def _pairwise(embs: list[torch.Tensor]) -> float:
            vals = [
                cos(embs[i], embs[j]) for i in range(len(embs)) for j in range(i + 1, len(embs))
            ]
            return sum(vals) / len(vals) if vals else float('nan')

        ceiling[f'{lang}|{emo}'] = {
            'n': n,
            'eca_e2v_ceiling': hits_e2v / n,
            'eca_sv_ceiling': hits_sv / n,
            'esim_e2v_floor': sum(floor_e2v) / len(floor_e2v),
            'esim_sv_floor': sum(floor_sv) / len(floor_sv),
            'esim_e2v_ceiling': _pairwise(embs_e2v),
            'esim_sv_ceiling': _pairwise(embs_sv),
        }
    print(
        '(as duas últimas colunas são o PISO: GT neutro vs GT emocional — '
        'E-SIM sintético abaixo disso não indica transferência)'
    )
    if ceiling_path:
        Path(ceiling_path).write_text(json.dumps(ceiling, indent=2))
        print(f'teto/piso → {ceiling_path}')


def integrity_check(rows: list[dict], n: int = 5) -> None:
    """Recalcula o EECS de uma amostra e confere contra o summary."""
    sample = [r for r in rows if isinstance(r['eecs_stored'], float)][:: max(1, len(rows) // n)][
        :n
    ]
    print(f'\n=== check de integridade (EECS recalculado vs gravado, n={len(sample)}) ===')
    bad = []
    for r in sample:
        if not os.path.exists(r['wav_path']):
            raise SystemExit(f'wav ausente — {r["wav_path"]}')
        _, _, emb = _e2v_label_probs(r['wav_path'])
        _, _, gt = _e2v_label_probs(r['gt_emo_audio'])
        got = cos(emb, gt)
        delta = abs(got - r['eecs_stored'])
        flag = 'ok' if delta <= EECS_TOL else 'DIVERGE'
        print(
            f'  {r["cell"]:34s} {r["utt"]:8s} {r["cond"]:11s} '
            f'gravado={r["eecs_stored"]:.4f} recalc={got:.4f} Δ={delta:.4f} {flag}'
        )
        if delta > EECS_TOL:
            bad.append(r)
    if bad:
        raise SystemExit(
            'EECS recalculado diverge do gravado — os wavs em disco não são os medidos no summary.'
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--summaries', nargs='*', default=DEFAULT_SUMMARIES)
    p.add_argument('--out_prefix', default=str(EXP_DIR / 'gap4_eca_esim'))
    p.add_argument('--selftest', action='store_true', help='só teto no GT + integridade')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--no_resume', action='store_true')
    args = p.parse_args()

    rows = build_rows(args.summaries)
    print(
        f'{len(rows)} wavs em {len({r["cell"] for r in rows})} células '
        f'({len(args.summaries)} summaries)'
    )

    ceiling_path = f'{args.out_prefix}_gt_ceiling.json'
    integrity_check(rows)
    if args.selftest or not os.path.exists(ceiling_path):
        selftest(rows, ceiling_path)
    if args.selftest:
        return

    if args.limit:
        rows = rows[: args.limit]

    jsonl_path = f'{args.out_prefix}.jsonl'
    done: set[str] = set()
    if os.path.exists(jsonl_path) and not args.no_resume:
        with open(jsonl_path) as f:
            done = {json.loads(line)['wav_path'] for line in f if line.strip()}
        print(f'resume: {len(done)} wavs já pontuados')
    mode = 'a' if done else 'w'

    gt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    t0 = time.time()
    written = 0
    with open(jsonl_path, mode) as out:
        for i, r in enumerate(rows):
            if r['wav_path'] in done:
                continue
            if not os.path.exists(r['wav_path']):
                print(f'[!] wav ausente: {r["wav_path"]}')
                continue
            gt = r['gt_emo_audio']
            if gt not in gt_cache:
                s = score_wav(gt, r['language'])
                gt_cache[gt] = (s['e2v_emb'], s['sv_emb'])
            gt_e2v, gt_sv = gt_cache[gt]

            s = score_wav(r['wav_path'], r['language'])
            e2v_4 = _restrict(s['e2v_probs'], ESD_CLASSES)
            sv_4 = _restrict(s['sv_probs'], ESD_CLASSES)
            emo = r['emotion']
            rec = {k: v for k, v in r.items()}
            rec.update({
                'eca_e2v_strict': int(s['e2v_label'] == emo),
                'eca_e2v_4way': int(max(e2v_4, key=e2v_4.get) == emo),
                'p_e2v_target': e2v_4.get(emo),
                'e2v_label': s['e2v_label'],
                'esim_e2v_gt': cos(s['e2v_emb'], gt_e2v),
                'eca_sv_strict': int(s['sv_label'] == emo),
                'eca_sv_4way': int(max(sv_4, key=sv_4.get) == emo),
                'p_sv_target': sv_4.get(emo),
                'sv_label': s['sv_label'],
                'sv_prob_unk': s['sv_prob_unk'],
                'esim_sv_gt': cos(s['sv_emb'], gt_sv),
            })
            out.write(json.dumps(rec) + '\n')
            out.flush()
            written += 1
            if written == 1:
                print(
                    f'Primeiro áudio pontuado em {time.time() - t0:.1f}s: '
                    f'ECA_e2v={rec["eca_e2v_4way"]} E-SIM_sv={rec["esim_sv_gt"]:.3f} '
                    f'(EECS gravado {r["eecs_stored"]})'
                )
            if written % 250 == 0:
                el = time.time() - t0
                print(
                    f'  {written}/{len(rows) - len(done)} · {el / written:.2f}s/wav · '
                    f'ETA {(len(rows) - len(done) - written) * el / written / 60:.1f} min',
                    flush=True,
                )

    print(f'\n{written} wavs pontuados em {(time.time() - t0) / 60:.1f} min → {jsonl_path}')
    aggregate(jsonl_path, args.out_prefix)


def aggregate(jsonl_path: str, out_prefix: str) -> None:
    import pandas as pd

    df = pd.read_json(jsonl_path, lines=True)
    df.to_parquet(f'{out_prefix}.parquet')

    metrics = [
        'eca_e2v_4way',
        'eca_e2v_strict',
        'p_e2v_target',
        'esim_e2v_gt',
        'eca_sv_4way',
        'eca_sv_strict',
        'p_sv_target',
        'esim_sv_gt',
        'sv_prob_unk',
    ]
    lines = [
        '# ECA e similaridade emocional das sínteses',
        '',
        f'Fonte: `{jsonl_path}` · {len(df)} wavs · '
        f'{df["cell"].nunique()} células · encoders: emotion2vec_plus_large + SenseVoiceSmall',
        '',
        'ECA_4way = forced-choice restrito a angry/happy/sad/neutral. '
        'Ler sempre contra o teto no GT (bloco `--selftest`).',
        '',
    ]

    for combo in sorted(df['combination'].unique()):
        for lang in sorted(df[df.combination == combo]['language'].unique()):
            sub = df[(df.combination == combo) & (df.language == lang)]
            lines += [
                f'## {combo} · {lang}',
                '',
                '| célula | cond | n | ' + ' | '.join(metrics) + ' |',
                '|---|---|---|' + '---|' * len(metrics),
            ]
            for cell in sorted(sub['cell'].unique()):
                for label, mask in (('α=0', sub.is_baseline), ('α*', sub.is_best)):
                    blk = sub[(sub.cell == cell) & mask]
                    if blk.empty:
                        continue
                    vals = ' | '.join(f'{blk[m].mean():.3f}' for m in metrics)
                    lines.append(f'| {cell} | {label} | {len(blk)} | {vals} |')
            lines.append('')

    ceiling_file = Path(f'{out_prefix}_gt_ceiling.json')
    if ceiling_file.exists():
        ceil = json.loads(ceiling_file.read_text())
        lines += [
            '## Leitura normalizada pelo GT — `(α* − α0) / (teto − α0)`',
            '',
            'A normalização usa as referências naturais calculadas nesta execução.',
            '',
            '| lang | emoção | ECA e2v α0→α* (teto) | ECA norm | E-SIM e2v α0→α* '
            '(piso→teto) | E-SIM norm |',
            '|---|---|---|---|---|---|',
        ]
        add = df[df.combination == 'additive']
        for key, c in sorted(ceil.items()):
            lang, emo = key.split('|')
            blk = add[(add.language == lang) & (add.emotion == emo)]
            b, k = blk[blk.is_baseline], blk[blk.is_best]
            if b.empty or k.empty:
                continue
            e0, e1, et = b.eca_e2v_4way.mean(), k.eca_e2v_4way.mean(), c['eca_e2v_ceiling']
            s0, s1 = b.esim_e2v_gt.mean(), k.esim_e2v_gt.mean()
            sf, st = c['esim_e2v_floor'], c['esim_e2v_ceiling']
            eca_n = (e1 - e0) / (et - e0) if et > e0 else float('nan')
            esim_n = (s1 - sf) / (st - sf) if st > sf else float('nan')
            lines.append(
                f'| {lang} | {emo} | {e0:.2f}→{e1:.2f} ({et:.2f}) | {eca_n:.2f} | '
                f'{s0:.3f}→{s1:.3f} ({sf:.3f}→{st:.3f}) | {esim_n:.2f} |'
            )
        lines.append('')

    lines += [
        '## Por α (agregado sobre células)',
        '',
        '| combination | lang | emoção | α | n | ' + ' | '.join(metrics) + ' |',
        '|---|---|---|---|---|' + '---|' * len(metrics),
    ]
    g = df.groupby(['combination', 'language', 'emotion', 'alpha'])
    for (combo, lang, emo, a), blk in g:
        vals = ' | '.join(f'{blk[m].mean():.3f}' for m in metrics)
        lines.append(f'| {combo} | {lang} | {emo} | {a} | {len(blk)} | {vals} |')

    md = f'{out_prefix}.md'
    Path(md).write_text('\n'.join(lines) + '\n')
    print(f'Resumo → {md}\nParquet → {out_prefix}.parquet')


if __name__ == '__main__':
    main()
