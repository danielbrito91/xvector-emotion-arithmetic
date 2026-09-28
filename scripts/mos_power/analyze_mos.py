"""Analisa o teste de escuta com limpeza cega às condições.

O mapping é associado somente após clean(). EMOS usa Wilcoxon pareado por
utterance e bootstrap de 10.000 réplicas. Naturalidade inclui o teste unilateral
de não-inferioridade, margem 0,5. Sem --unblind, executa apenas a limpeza."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

Z_OUTLIER = 3.29  # Sinalização diagnóstica; não exclui votos.
PROBE_LANG_GATE = 'pt'  # As sondas EN são apenas descritivas.
PROBE_MAX_ERRORS = 0  # Dois itens de atenção em português.
CEFR_MAIN_EN = {'B1', 'B2', 'C1', 'C2'}  # abaixo de B1: só o bloco EN sai
CEFR_SENSITIVITY = {'B2', 'C1', 'C2'}

H2_MARGIN = 0.5  # margem de não-inferioridade, em pontos de MOS
ALPHA = 0.05
N_BOOT = 10000
BOOT_SEED = 20260819


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--submissions', default='data/experiments/mos_field/submissions.jsonl')
    p.add_argument('--manifest', default='data/experiments/mos_stimuli/deploy/manifest.json')
    p.add_argument('--mapping', default='data/experiments/mos_stimuli/private/mapping.csv')
    p.add_argument(
        '--objective', default='data/experiments/mos_stimuli/private/frozen_objective.csv'
    )
    p.add_argument('--out_dir', default='data/experiments/mos_field/analysis')
    p.add_argument(
        '--unblind',
        action='store_true',
        help='Junta a condição e roda os testes. Sem isto, só o cleansing.',
    )
    p.add_argument('--min_raters', type=int, default=1)
    return p.parse_args()


def load_submissions(path: str) -> pd.DataFrame:
    """Lê JSONL de submissões, incluindo registros dos buckets verified e spam."""
    rows = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        d = r.get('data', {})
        try:
            payload = json.loads(d.get('payload_json') or '{}')
        except json.JSONDecodeError:
            payload = {}
        rows.append({
            'sub_id': r.get('id'),
            'bucket': r.get('_bucket'),
            'rater': str(d.get('rater_seed')),
            'kind': d.get('kind'),
            'step_id': d.get('step_id'),
            'ui_version': d.get('ui_version'),
            'timestamp': d.get('timestamp'),
            'payload': payload,
        })
    df = pd.DataFrame(rows).drop_duplicates('sub_id')

    drop = []
    debug = df.payload.apply(lambda p: bool(p.get('debug')))
    drop.append(('payload debug:true', int(debug.sum())))
    smoke = df.step_id.eq('smoke_test') | df.ui_version.fillna('0').lt('3.')
    drop.append(('smoke test / UI anterior à v3', int((smoke & ~debug).sum())))
    df = df[~(debug | smoke)].copy()
    for name, n in drop:
        if n:
            print(f'    descartado: {name} ({n})')
    return df


def build_ratings(df: pd.DataFrame, manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Converte respostas em tabelas de votos e páginas, excluindo os itens de treino."""
    items = {i['id']: i for i in manifest['items']}
    votes, pages = [], []
    for r in df[df.kind == 'rating_page'].itertuples():
        p = r.payload
        if p.get('training') or p.get('block') == 'training':
            continue
        page_key = (r.rater, p.get('block'), p.get('page_index'))
        pages.append({
            'rater': r.rater,
            'block': p.get('block'),
            'page_index': p.get('page_index'),
            'shown_at': p.get('shown_at'),
            'submitted_at': p.get('submitted_at'),
            'min_audio_s': p.get('min_audio_s'),
            'page_key': page_key,
        })
        for it in p.get('items', []):
            meta = items.get(it['id'], {})
            votes.append({
                'rater': r.rater,
                'sid': it['id'],
                'block': p.get('block'),
                'page_key': page_key,
                'pos': it.get('pos'),
                'emos': it.get('emos'),
                'nat': it.get('nat'),
                # Naturalidade não perguntada fica ausente; não representa nota zero.
                'asked_nat': bool(it.get('asked_nat')),
                'plays': it.get('plays'),
                'played_ms': it.get('played_ms'),
                'language': meta.get('language'),
                'is_gold': bool(meta.get('is_gold')),
                'target_emotion': meta.get('target_emotion'),
            })
    vcols = [
        'rater',
        'sid',
        'block',
        'page_key',
        'pos',
        'emos',
        'nat',
        'asked_nat',
        'plays',
        'played_ms',
        'language',
        'is_gold',
        'target_emotion',
    ]
    pcols = ['rater', 'block', 'page_index', 'shown_at', 'submitted_at', 'min_audio_s', 'page_key']
    return (
        pd.DataFrame(votes, columns=None if votes else vcols),
        pd.DataFrame(pages, columns=None if pages else pcols),
    )


def assert_blind(*frames: pd.DataFrame) -> None:
    """Rejeita tabelas com condição ou alpha antes da limpeza."""
    for f in frames:
        leak = {'condition', 'alpha'} & set(f.columns)
        if leak:
            raise RuntimeError(
                f'Coluna de condição {sorted(leak)} presente durante o cleansing. '
                'Associe as condições somente depois de clean().'
            )


def probe_errors(df: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    """Compara a resposta da sonda ao texto do estímulo no manifesto."""
    items = {i['id']: i for i in manifest['items']}
    rows: list[dict] = []
    for r in df[df.kind == 'probe'].itertuples():
        p = r.payload
        tgt = items.get(p.get('target_id'), {})
        rows.append({
            'rater': r.rater,
            'block': p.get('block'),
            'language': tgt.get('language'),
            'correct': (p.get('chosen_text') or '').strip() == (tgt.get('text') or '').strip(),
        })
    return pd.DataFrame(rows, columns=None if rows else ['rater', 'block', 'language', 'correct'])


def qualification_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Uma qualificação CEFR por rater, sem escolher silenciosamente conflitos."""
    rows: list[dict] = []
    for r in df[df.kind == 'qualification'].itertuples():
        cefr = str(r.payload.get('cefr_en') or '').strip().upper() or None
        rows.append({'rater': r.rater, 'cefr_en': cefr})
    q = pd.DataFrame(rows, columns=None if rows else ['rater', 'cefr_en'])
    if q.empty:
        return q
    conflicts = q.dropna(subset=['cefr_en']).groupby('rater').cefr_en.nunique()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        raise RuntimeError(
            'CEFR conflitante em submissões de qualificação do(s) rater(s): '
            + ', '.join(conflicts.index.astype(str))
        )
    return q.drop_duplicates('rater', keep='last')


def clean(
    votes: pd.DataFrame,
    pages: pd.DataFrame,
    probes: pd.DataFrame,
    qualifications: pd.DataFrame,
    manifest: dict,
) -> tuple[pd.DataFrame, dict]:
    """Filtra participantes e páginas e devolve os votos retidos e os motivos dos cortes."""
    assert_blind(votes, pages, probes, qualifications)
    if votes.empty:
        # Estado normal durante o campo: ninguem completou uma pagina valida ainda.
        return votes, {
            'raters_in': [],
            'raters_out': [],
            'n_raters_in': 0,
            'n_raters_out': 0,
            'dropped_by_probe': [],
            'dropped_en_by_cefr': [],
            'gold_responses_preserved': 0,
            'dropped_pages_too_fast': [],
            'flagged_votes_outlier': {},
            'flagged_vote_details': [],
            'dropped_votes_outlier': {'emos': 0, 'nat': 0},
            'note': 'nenhum voto valido: campo vazio ou so sessoes de debug',
        }
    ledger: dict = {'raters_in': sorted(votes.rater.unique().tolist())}

    # Erro em uma sonda PT exclui o participante.
    dropped_probe = []
    if len(probes):
        gate = probes[probes.language == PROBE_LANG_GATE]
        for rater, g in gate.groupby('rater'):
            errs = int((~g.correct).sum())
            if errs > PROBE_MAX_ERRORS:
                dropped_probe.append({'rater': rater, 'errors_pt': errs, 'n_probes_pt': len(g)})
        en = probes[probes.language != PROBE_LANG_GATE]
        ledger['probes_en_descriptive'] = {
            'n': len(en),
            'accuracy': float(en.correct.mean()) if len(en) else None,
            'note': 'fora da regra: frases do ESD curtas e agramaticais para painel B1+',
        }
    ledger['dropped_by_probe'] = dropped_probe
    votes = votes[~votes.rater.isin([d['rater'] for d in dropped_probe])]

    # CEFR abaixo de B1 ou ausente exclui somente o bloco EN.
    cefr_by_rater = (
        qualifications.set_index('rater').cefr_en.to_dict() if len(qualifications) else {}
    )
    votes = votes.copy()
    votes['cefr_en'] = votes.rater.map(cefr_by_rater)
    bad_en = votes.language.eq('en') & ~votes.cefr_en.isin(CEFR_MAIN_EN)
    dropped_en = []
    for rater, g in votes[bad_en].groupby('rater'):
        dropped_en.append({
            'rater': rater,
            'cefr_en': cefr_by_rater.get(rater),
            'n_votes': int(len(g)),
        })
    ledger['dropped_en_by_cefr'] = dropped_en
    votes = votes[~bad_en]

    # is_gold identifica a origem do áudio; notas baixas no GT são mantidas.
    ledger['gold_responses_preserved'] = int(votes.is_gold.sum())
    ledger['gold_rule'] = (
        'nenhuma exclusão: is_gold marca GT usado como referência perceptual; '
        'EMOS/naturalidade não têm resposta correta'
    )

    # Retém páginas com duração suficiente para ouvir todos os áudios.
    bad_pages = []
    for r in pages.itertuples():
        if not (r.shown_at and r.submitted_at and r.min_audio_s):
            continue
        dt = (pd.Timestamp(r.submitted_at) - pd.Timestamp(r.shown_at)).total_seconds()
        if dt < float(r.min_audio_s):
            bad_pages.append({
                'page_key': list(r.page_key),
                'elapsed_s': round(dt, 1),
                'min_audio_s': float(r.min_audio_s),
            })
    ledger['dropped_pages_too_fast'] = bad_pages
    bad_keys = {tuple(b['page_key']) for b in bad_pages}
    votes = votes[~votes.page_key.isin(bad_keys)]

    # Sinaliza |z| > 3.29 por estímulo e escala, preservando o voto.
    out = {}
    flagged = []
    for scale in ['emos', 'nat']:
        v = (
            votes[votes[scale].notna()]
            if scale == 'emos'
            else votes[votes.asked_nat & votes.nat.notna()]
        )
        if v.empty:
            out[scale] = 0
            continue
        g = v.groupby('sid')[scale]
        mu, sd = g.transform('mean'), g.transform('std')
        z = (v[scale] - mu) / sd.replace(0, np.nan)
        mask = z.abs() > Z_OUTLIER
        out[scale] = int(mask.sum())
        for idx in mask[mask].index:
            flagged.append({
                'rater': str(v.at[idx, 'rater']),
                'sid': str(v.at[idx, 'sid']),
                'language': str(v.at[idx, 'language']),
                'scale': scale,
                'value': float(v.at[idx, scale]),
                'abs_z': float(abs(z.at[idx])),
            })
    ledger['flagged_votes_outlier'] = out
    ledger['flagged_vote_details'] = flagged
    # Compatibilidade explícita com consumidores do ledger anterior.
    ledger['dropped_votes_outlier'] = {'emos': 0, 'nat': 0}

    # Participantes incompletos contribuem com suas páginas completas.
    ledger['raters_out'] = sorted(votes.rater.unique().tolist())
    ledger['n_raters_in'] = len(ledger['raters_in'])
    ledger['n_raters_out'] = len(ledger['raters_out'])
    ledger['raters_by_language'] = {
        lang: sorted(votes.loc[votes.language.eq(lang), 'rater'].unique().tolist())
        for lang in ['pt', 'en']
    }
    ledger['n_raters_by_language'] = {
        lang: len(raters) for lang, raters in ledger['raters_by_language'].items()
    }
    return votes.copy(), ledger


def join_conditions(votes: pd.DataFrame, mapping: str) -> pd.DataFrame:
    """Associa as condições aos votos já limpos."""
    m = pd.read_csv(mapping)[['sid', 'condition', 'alpha', 'target', 'emotion', 'utt', 'language']]
    return votes.drop(columns=['language']).merge(m, on='sid', how='inner')


def paired_by_utterance(v: pd.DataFrame, scale: str, language: str) -> pd.DataFrame:
    """Pareia médias de ouvintes por frase, falante e referência ICL."""
    d = v[
        (v.language == language) & (v.condition.isin(['alpha0', 'alphastar'])) & v[scale].notna()
    ]
    if scale == 'nat':
        d = d[d.asked_nat]
    agg = d.groupby(['target', 'emotion', 'utt', 'condition'])[scale].mean().unstack('condition')
    return agg.dropna(subset=['alpha0', 'alphastar'])


def boot_ci(x: np.ndarray, seed: int = BOOT_SEED, n: int = N_BOOT) -> list[float]:
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def h1(v: pd.DataFrame, language: str) -> dict:
    t = paired_by_utterance(v, 'emos', language)
    n = len(t)
    if n < 3:
        return {
            'language': language,
            'n_pairs': n,
            'testable': False,
            'reason': 'menos de 3 pares completos',
        }
    diff = (t.alphastar - t.alpha0).to_numpy(float)
    w = stats.wilcoxon(
        t.alphastar, t.alpha0, alternative='two-sided', zero_method='wilcox', method='auto'
    )
    # correlação bisserial de postos para pares casados
    nz = diff[diff != 0]
    rb = float('nan')
    if len(nz):
        ranks = stats.rankdata(np.abs(nz))
        rb = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
    return {
        'language': language,
        'n_pairs': n,
        'testable': True,
        'mean_alpha0': float(t.alpha0.mean()),
        'mean_alphastar': float(t.alphastar.mean()),
        'mean_diff': float(diff.mean()),
        'median_diff': float(np.median(diff)),
        'ci95_mean_diff': boot_ci(diff),
        'rank_biserial': rb,
        'wilcoxon_stat': float(w.statistic),
        'p_value': float(w.pvalue),
        'reject_h0': bool(w.pvalue < ALPHA),
    }


def h2(v: pd.DataFrame, language: str) -> dict:
    """Não-inferioridade: H0 μ_diff <= −margem contra H1 μ_diff > −margem."""
    t = paired_by_utterance(v, 'nat', language)
    n = len(t)
    if n < 3:
        return {
            'language': language,
            'n_pairs': n,
            'testable': False,
            'reason': 'menos de 3 pares completos com naturalidade perguntada',
        }
    diff = (t.alphastar - t.alpha0).to_numpy(float)
    tt = stats.ttest_1samp(diff, -H2_MARGIN, alternative='greater')
    return {
        'language': language,
        'n_pairs': n,
        'testable': True,
        'margin': H2_MARGIN,
        'mean_diff': float(diff.mean()),
        'ci95_mean_diff': boot_ci(diff),
        't_stat': float(tt.statistic),
        'p_value': float(tt.pvalue),
        'non_inferior': bool(tt.pvalue < ALPHA),
    }


def sanity_by_rater(v: pd.DataFrame, language: str) -> dict:
    """Calcula a análise secundária com agregação por ouvinte."""
    d = v[(v.language == language) & v.condition.isin(['alpha0', 'alphastar']) & v.emos.notna()]
    agg = d.groupby(['rater', 'condition']).emos.mean().unstack('condition').dropna()
    if len(agg) < 3:
        return {'language': language, 'n_raters': len(agg), 'testable': False}
    w = stats.wilcoxon(agg.alphastar, agg.alpha0, alternative='two-sided')
    return {
        'language': language,
        'n_raters': len(agg),
        'mean_diff': float((agg.alphastar - agg.alpha0).mean()),
        'p_value': float(w.pvalue),
        'testable': True,
    }


def exploratory(v: pd.DataFrame, objective: str) -> dict:
    """Calcula associações exploratórias sem correção de múltiplas comparações."""
    out: dict = {'label': 'exploratório pré-declarado, sem correção de múltiplas comparações'}

    # teto GT por célula (PT) e efeito normalizado
    gt = v[(v.condition == 'gt') & (v.language == 'pt')]
    if len(gt):
        ceil = gt.groupby(['target', 'emotion']).emos.mean()
        pt = v[(v.language == 'pt') & v.condition.isin(['alpha0', 'alphastar'])]
        cell = pt.groupby(['target', 'emotion', 'condition']).emos.mean().unstack('condition')
        cell = cell.join(ceil.rename('gt'))
        cell['normalized'] = (cell['alphastar'] - cell['alpha0']) / (cell['gt'] - cell['alpha0'])
        out['gt_ceiling_pt'] = json.loads(cell.round(4).reset_index().to_json(orient='records'))

    # Associações entre as métricas objetivas e as médias dos ouvintes.
    op = Path(objective)
    if op.exists():
        o = pd.read_csv(op)[['sid', 'eecs', 'utmos']]
        per = v.groupby('sid').agg(emos=('emos', 'mean'), nat=('nat', 'mean')).reset_index()
        j = per.merge(o, on='sid', how='inner')
        for a, b in [('emos', 'eecs'), ('nat', 'utmos')]:
            d = j[[a, b]].dropna()
            if len(d) >= 4:
                r = stats.pearsonr(d[a], d[b])
                s = stats.spearmanr(d[a], d[b])
                out[f'{a}_vs_{b}'] = {
                    'n': len(d),
                    'pearson_r': float(r.statistic),
                    'pearson_p': float(r.pvalue),
                    'spearman_rho': float(s.statistic),
                }
    return out


def write_report(out: Path, ledger: dict, res: dict | None) -> None:
    n_lang = ledger.get('n_raters_by_language', {'pt': 0, 'en': 0})
    L = [
        '# Análise do teste de escuta',
        '',
        'Gerado por `scripts/mos_power/analyze_mos.py`. As regras de limpeza '
        'são aplicadas antes da associação das condições.',
        '',
        '## Limpeza das respostas',
        '',
        f'- Raters que entraram: **{ledger["n_raters_in"]}** → sobraram: **{ledger["n_raters_out"]}**',
        f'- Raters com ao menos uma página retida: PT-BR **N={n_lang.get("pt", 0)}** · EN **N={n_lang.get("en", 0)}**',
        f'- Descartados por sonda de conteúdo (PT, tolerância zero): {len(ledger["dropped_by_probe"])}',
        f'- Blocos EN removidos por CEFR abaixo de B1 ou ausente: {len(ledger["dropped_en_by_cefr"])}',
        '- Descartados por nota em GT/is_gold: 0 (respostas perceptuais preservadas)',
        f'- Páginas descartadas por duração: {len(ledger["dropped_pages_too_fast"])}',
        f'- Votos sinalizados por |z| > {Z_OUTLIER}: {ledger["flagged_votes_outlier"]} (nenhum descartado)',
        '',
    ]
    if ledger.get('probes_en_descriptive', {}).get('n'):
        e = ledger['probes_en_descriptive']
        L += [
            f'- Sondas em inglês (descritivo, fora da regra): {e["n"]} respostas, '
            f'acurácia {e["accuracy"]:.1%}.',
            '',
        ]
    L += [
        'As marcações `is_gold` identificam gravações GT usadas como referência perceptual; '
        'não constituem gabarito de EMOS ou naturalidade.',
        '',
    ]

    if res is None:
        L += [
            '## Testes',
            '',
            'Não rodados: use `--unblind` depois de fechar o campo e conferir o cleansing acima.',
        ]
    else:
        L += [
            '## Testes primários',
            '',
            '| teste | n pares | α=0 | α\\* | Δ | IC95 de Δ | p | rejeita H0 |',
            '|---|---|---|---|---|---|---|---|',
        ]
        for k in ['H1-PT', 'H1-EN']:
            h = res[k]
            if not h.get('testable'):
                L.append(f'| {k} | {h["n_pairs"]} | — | — | — | — | — | não testável |')
                continue
            L.append(
                f'| {k} | {h["n_pairs"]} | {h["mean_alpha0"]:.3f} | {h["mean_alphastar"]:.3f} | '
                f'{h["mean_diff"]:+.3f} | {h["ci95_mean_diff"][0]:+.3f} a '
                f'{h["ci95_mean_diff"][1]:+.3f} | {h["p_value"]:.4f} | '
                f'{"sim" if h["reject_h0"] else "não"} |'
            )
        L += [
            '',
            '## H2 — não-inferioridade de naturalidade (margem 0,5)',
            '',
            '| língua | n pares | Δ | IC95 | p | não-inferior |',
            '|---|---|---|---|---|---|',
        ]
        for k in ['H2-PT', 'H2-EN']:
            h = res[k]
            if not h.get('testable'):
                L.append(f'| {k} | {h["n_pairs"]} | — | — | — | não testável |')
                continue
            L.append(
                f'| {k} | {h["n_pairs"]} | {h["mean_diff"]:+.3f} | '
                f'{h["ci95_mean_diff"][0]:+.3f} a {h["ci95_mean_diff"][1]:+.3f} | '
                f'{h["p_value"]:.4f} | {"sim" if h["non_inferior"] else "não"} |'
            )
        sens = res['sensitivity_b2plus']
        L += [
            '',
            '## Sensibilidade — painel B2 ou superior',
            '',
            '| teste | N raters | n pares | Δ | p |',
            '|---|---|---|---|---|',
        ]
        for k in ['H1-PT', 'H1-EN', 'H2-PT', 'H2-EN']:
            h = sens[k]
            if not h.get('testable'):
                L.append(f'| {k} | {sens["n_raters"]} | {h["n_pairs"]} | — | — |')
                continue
            L.append(
                f'| {k} | {sens["n_raters"]} | {h["n_pairs"]} | '
                f'{h["mean_diff"]:+.3f} | {h["p_value"]:.4f} |'
            )
    (out / 'report.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))

    print('ingestão')
    subs = load_submissions(args.submissions)
    votes, pages = build_ratings(subs, manifest)
    probes = probe_errors(subs, manifest)
    qualifications = qualification_levels(subs)
    print(
        f'    {len(subs)} submissões, {len(votes)} votos, {len(pages)} páginas, '
        f'{len(probes)} sondas, {len(qualifications)} qualificações'
    )

    print('limpeza das respostas')
    clean_votes, ledger = clean(votes, pages, probes, qualifications, manifest)
    n_lang = ledger.get('n_raters_by_language', {'pt': 0, 'en': 0})
    print(
        f'    raters {ledger["n_raters_in"]} → {ledger["n_raters_out"]}; '
        f'contribuintes PT-BR={n_lang.get("pt", 0)}, EN={n_lang.get("en", 0)}; '
        f'votos limpos: {len(clean_votes)}'
    )
    (out / 'cleansing_ledger.json').write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    res = None
    if not args.unblind:
        print('\nLimpeza concluída. Use --unblind para calcular os testes por condição.')
    elif ledger['n_raters_out'] < args.min_raters:
        print(f'\n[cego] só {ledger["n_raters_out"]} raters sobreviveram; nada a testar.')
    else:
        print('testes por condição')
        v = join_conditions(clean_votes, args.mapping)
        v_b2 = v[v.cefr_en.isin(CEFR_SENSITIVITY)].copy()
        sensitivity = {
            'n_raters': int(v_b2.rater.nunique()),
            'H1-PT': h1(v_b2, 'pt'),
            'H1-EN': h1(v_b2, 'en'),
            'H2-PT': h2(v_b2, 'pt'),
            'H2-EN': h2(v_b2, 'en'),
        }
        res = {
            'H1-PT': h1(v, 'pt'),
            'H1-EN': h1(v, 'en'),
            'H2-PT': h2(v, 'pt'),
            'H2-EN': h2(v, 'en'),
            'sanity_by_rater': {'pt': sanity_by_rater(v, 'pt'), 'en': sanity_by_rater(v, 'en')},
            'sensitivity_b2plus': sensitivity,
            'exploratory': exploratory(v, args.objective),
            'params': {
                'alpha': ALPHA,
                'n_boot': N_BOOT,
                'boot_seed': BOOT_SEED,
                'h2_margin': H2_MARGIN,
                'z_outlier': Z_OUTLIER,
                'z_outlier_action': 'flag_only',
                'cefr_main_en': sorted(CEFR_MAIN_EN),
                'cefr_sensitivity': sorted(CEFR_SENSITIVITY),
            },
        }
        (out / 'results.json').write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8'
        )
        for k in ['H1-PT', 'H1-EN', 'H2-PT', 'H2-EN']:
            h = res[k]
            print(
                f'    {k}: '
                + (
                    f'n={h["n_pairs"]} Δ={h["mean_diff"]:+.3f} p={h["p_value"]:.4f}'
                    if h.get('testable')
                    else f'não testável ({h.get("reason")})'
                )
            )

    write_report(out, ledger, res)
    print(f'\n→ {out}/report.md, cleansing_ledger.json' + (', results.json' if res else ''))


if __name__ == '__main__':
    main()
