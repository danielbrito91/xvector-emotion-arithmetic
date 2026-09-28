"""Calcula as tabelas objetivas, o MOS pareado e os contrastes do fatorial de condicionamento."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ['eca_e2v_4way', 'eca_sv_4way', 'spk_cos_sim_neutral_wavlm', 'wer_norm']
GRID = [0.0, 1.0, 1.5, 2.0, 2.5]


def select_source_alpha(frame):
    source = frame[
        frame.language.eq('en') & frame.tau_variant.eq('avg4spk') & frame.alpha.isin(GRID)
    ]
    scores = source.groupby(['emotion', 'alpha']).esim_e2v_gt.mean().reset_index()
    assert set(scores.emotion) == {'angry', 'happy', 'sad'}
    assert (scores.groupby('emotion').size() == len(GRID)).all()
    return (
        scores
        .sort_values(['emotion', 'esim_e2v_gt', 'alpha'], ascending=[True, False, True])
        .groupby('emotion')
        .alpha.first()
        .to_dict()
    )


def paired_summary(baseline, endpoint):
    key = ['language', 'target', 'emotion', 'utt', 'tau_variant']
    q = endpoint[key + METRICS].merge(
        baseline[key + METRICS], on=key, validate='one_to_one', suffixes=('_end', '_base')
    )
    assert len(q) == len(endpoint) == len(baseline), 'Pareamento incompleto.'
    return {
        'n': len(q),
        'metrics': {
            m: {
                'baseline': float(q[m + '_base'].mean()),
                'endpoint': float(q[m + '_end'].mean()),
                'delta': float((q[m + '_end'] - q[m + '_base']).mean()),
            }
            for m in METRICS
        },
    }


def objective():
    d = pd.read_parquet('data/experiments/gap4_eca_esim.parquet')
    d = d[d.combination.eq('additive')].copy()
    raw = pd.concat([
        pd.read_parquet('data/experiments/' + p)
        for p in ['results_long.parquet', 'results_long_ptbr.parquet']
    ])
    raw = raw[['wav_path', 'spk_cos_sim_neutral_wavlm', 'wer_norm']]
    assert not raw.wav_path.duplicated().any()
    d = d.merge(raw, on='wav_path', validate='one_to_one')
    assert not d[METRICS].isna().any().any()
    choices = select_source_alpha(d)
    result = {
        'source_only_alpha': choices,
        'posthoc': True,
        'calibrated': {},
        'source_only': {},
        'per_emotion_multi': {},
    }
    for (lang, variant), q in d.groupby(['language', 'tau_variant']):
        result['calibrated'][f'{lang}/{variant}'] = paired_summary(q[q.is_baseline], q[q.is_best])
        if variant == 'avg4spk':
            endpoint = q[q.alpha.eq(q.emotion.map(choices))]
            result['source_only'][lang] = paired_summary(q[q.is_baseline], endpoint)
            for emotion, sub in q.groupby('emotion'):
                result['per_emotion_multi'][f'{lang}/{emotion}'] = paired_summary(
                    sub[sub.is_baseline], sub[sub.is_best]
                )
    return result


def human(
    submissions='data/experiments/mos_field/snapshots/submissions_2026-09-03_n548.jsonl',
    coauthor_raters=(),
):
    from scripts.mos_power import analyze_mos as mod

    manifest = json.loads(Path('data/experiments/mos_stimuli/deploy/manifest.json').read_text())
    subs = mod.load_submissions(submissions)
    votes, participants = mod.build_ratings(subs, manifest)
    clean, ledger = mod.clean(
        votes,
        participants,
        mod.probe_errors(subs, manifest),
        mod.qualification_levels(subs),
        manifest,
    )
    votes = mod.join_conditions(clean, 'data/experiments/mos_stimuli/private/mapping.csv')
    assert not votes.duplicated(['rater', 'sid']).any()
    result = {'all': {}, 'n_raters': ledger['n_raters_by_language']}
    groups = [('all', votes)]
    if coauthor_raters:
        result['without_coauthors'] = {}
        groups.append(('without_coauthors', votes[~votes.rater.isin(coauthor_raters)]))
    for label, frame in groups:
        for lang in ['en', 'pt']:
            for scale, fn in [('emos', mod.h1), ('nat', mod.h2)]:
                r = fn(frame, lang)
                result[label][f'{lang}/{scale}'] = {
                    k: r[k] for k in ['mean_diff', 'p_value', 'ci95_mean_diff']
                }
    return result


def historical_factorial_rows(folder):
    checks = pd.read_json(folder / 'sentinel_checks.jsonl', lines=True)
    assert len(checks) == 12 and checks.identical.all()
    old = pd.read_json(
        'data/experiments/ar2_token_swap/results.jsonl', lines=True, dtype={'speaker': str}
    )
    old_eca = pd.read_json('data/experiments/ar2_token_swap/eca.jsonl', lines=True)
    old = old.drop(columns=['eca_e2v', 'eca_sv'], errors='ignore').merge(
        old_eca[['key', 'eca_e2v', 'eca_sv']], on='key', validate='one_to_one'
    )
    old = old[old.condition.isin(['neutral_baseline', 'full_swap', 'angry_baseline'])].copy()
    old['ref'] = old.key.str.split('/').str[1].str[-6:].astype(int)
    old['utt'] = old.key.str.split('/').str[2].str[-3:].astype(int)
    new = pd.read_json(folder / 'scores.jsonl', lines=True, dtype={'speaker': str})
    for row in new.itertuples():
        assert hashlib.sha256(Path(row.wav_path).read_bytes()).hexdigest() == row.sha256, (
            'Áudio novo alterado.'
        )
    # Rótulos de sentinelas devem reproduzir os dois classificadores históricos.
    sentinel = new[new.condition.ne('xvec_swap')].merge(
        old[['key', 'eca_e2v', 'eca_sv']],
        on='key',
        suffixes=('_new', '_old'),
        validate='one_to_one',
    )
    assert len(sentinel) == 12
    for metric in ['eca_e2v', 'eca_sv']:
        assert sentinel[f'{metric}_new'].eq(sentinel[f'{metric}_old']).all(), (
            'Deriva do classificador.'
        )
    cols = [
        'key',
        'speaker',
        'ref',
        'utt',
        'condition',
        'eca_e2v',
        'eca_sv',
        'duration_s',
        'wer_norm',
        'spk_cos_sim_neutral_wavlm',
    ]
    return pd.concat([old[cols], new[new.condition.eq('xvec_swap')][cols]], ignore_index=True)


def factorial(folder, layout='historical'):
    if layout == 'balanced':
        d = pd.read_json(folder / 'scores.jsonl', lines=True, dtype={'speaker': str})
        for row in d.itertuples():
            assert hashlib.sha256(Path(row.wav_path).read_bytes()).hexdigest() == row.sha256, (
                'Áudio alterado.'
            )
    elif layout == 'historical':
        d = historical_factorial_rows(folder)
    else:
        raise ValueError(f'Layout de fatorial desconhecido: {layout}')
    assert len(d) == 1600 and not d.key.duplicated().any()
    assert not d.duplicated(['speaker', 'ref', 'utt', 'condition']).any()
    assert set(d.speaker) == {'0011', '0013', '0017', '0019'}
    assert set(d.condition) == {'neutral_baseline', 'full_swap', 'xvec_swap', 'angry_baseline'}
    assert np.isfinite(
        d[['eca_e2v', 'eca_sv', 'duration_s', 'wer_norm', 'spk_cos_sim_neutral_wavlm']].to_numpy()
    ).all()
    assert (d.groupby(['speaker', 'condition']).size() == 100).all()
    assert d.ref.between(1, 10).all() and d.utt.between(321, 330).all()
    result = {
        'n_units': 400,
        'n_audio_conditions': 1600,
        'sentinels_identical': 12 if layout == 'historical' else 0,
        'classifier_sentinels_identical': True if layout == 'historical' else None,
        'duration_runaways': int((d.duration_s >= 0.95 * 1024 / 12.5).sum()),
        'inference': 'conditional on four observed speakers; crossed reference/text bootstrap',
        'wer_gt_1': int((d.wer_norm > 1).sum()),
        'quality': d
        .groupby('condition')[['wer_norm', 'spk_cos_sim_neutral_wavlm', 'duration_s']]
        .mean()
        .to_dict('index'),
        'per_speaker': {},
        'aggregate': {},
        'parallel_only_sensitivity': {},
    }
    for metric in ['eca_e2v', 'eca_sv']:
        result['per_speaker'][metric] = {
            str(s): q.groupby('condition')[metric].mean().to_dict()
            for s, q in d.groupby('speaker')
        }
        pivot = d.pivot(index=['speaker', 'ref', 'utt'], columns='condition', values=metric)
        contrasts = pd.DataFrame({
            'x_only_gain': pivot.xvec_swap - pivot.neutral_baseline,
            'token_only_gain': pivot.full_swap - pivot.neutral_baseline,
            'x_minus_tokens': pivot.xvec_swap - pivot.full_swap,
            'interaction': pivot.angry_baseline
            - pivot.full_swap
            - pivot.xvec_swap
            + pivot.neutral_baseline,
        })
        result['aggregate'][metric] = {
            'conditions': d.groupby('condition')[metric].mean().to_dict(),
            'contrasts': {},
        }
        rng = np.random.default_rng(20260906)
        sampled = np.zeros((10000, len(contrasts.columns)))
        texts = rng.integers(0, 10, size=(10000, 10))
        # Réplicas condicionais aos falantes: referências e textos são fatores cruzados.
        for spk, frame in contrasts.groupby(level='speaker'):
            arr = frame.to_numpy().reshape(10, 10, -1)
            refs = rng.integers(0, 10, size=(10000, 10))
            sampled += arr[refs[:, :, None], texts[:, None, :], :].mean(axis=(1, 2)) / 4
        for j, name in enumerate(contrasts):
            result['aggregate'][metric]['contrasts'][name] = {
                'delta': float(contrasts[name].mean()),
                'ci95': np.percentile(sampled[:, j], [2.5, 97.5]).tolist(),
            }
        # Sensibilidade pré-scoring: remove integralmente os dois pares com
        # discrepância lexical; mantém todas as quatro condições dos demais.
        strict = d[~(d.speaker.isin(['0011', '0013']) & d.ref.eq(6))]
        strict_p = strict.pivot(
            index=['speaker', 'ref', 'utt'], columns='condition', values=metric
        )
        delta = strict_p.xvec_swap - strict_p.full_swap
        sampled = np.zeros(10000)
        rng = np.random.default_rng(20260906)
        texts = rng.integers(0, 10, size=(10000, 10))
        for spk, series in delta.groupby(level='speaker'):
            nr = len(series) // 10
            arr = series.to_numpy().reshape(nr, 10)
            refs = rng.integers(0, nr, size=(10000, nr))
            sampled += arr[refs[:, :, None], texts[:, None, :]].mean(axis=(1, 2)) * (
                len(series) / len(delta)
            )
        result['parallel_only_sensitivity'][metric] = {
            'n_units': len(delta),
            'x_minus_tokens': float(delta.mean()),
            'ci95': np.percentile(sampled, [2.5, 97.5]).tolist(),
            'conditions': strict.groupby('condition')[metric].mean().to_dict(),
            'per_speaker_x_minus_tokens': delta.groupby(level='speaker').mean().to_dict(),
        }
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--factorial', type=Path)
    p.add_argument('--factorial-layout', choices=['historical', 'balanced'], default='historical')
    p.add_argument(
        '--sections',
        nargs='+',
        choices=['objective', 'human', 'factorial'],
        default=['objective', 'human'],
        help='Análises a executar; --factorial também inclui o fatorial.',
    )
    p.add_argument(
        '--submissions',
        default='data/experiments/mos_field/snapshots/submissions_2026-09-03_n548.jsonl',
    )
    p.add_argument('--output', type=Path)
    p.add_argument(
        '--coauthor-raters',
        nargs='*',
        default=[],
        help='IDs fornecidos localmente para a sensibilidade sem coautores.',
    )
    args = p.parse_args()
    if 'factorial' in args.sections and not args.factorial:
        p.error('--sections factorial exige --factorial com o diretório da corrida.')
    result = {}
    if 'objective' in args.sections:
        result['objective'] = objective()
    if 'human' in args.sections:
        result['human'] = human(args.submissions, args.coauthor_raters)
    if args.factorial:
        result['factorial'] = factorial(args.factorial, args.factorial_layout)
    text = json.dumps(result, indent=2, default=float, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    print(text)
