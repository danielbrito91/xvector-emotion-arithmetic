"""Shared aggregation + markdown rendering for the α-sweep experiments.

Both the EN→EN and EN→PT-BR sweeps (scripts/repro/run_*_sweep.py) and the
PT-BR ground-truth ceiling produce the same per-cell results structure, so the
summary table builders live here instead of inside a single CLI script.
"""

from __future__ import annotations

import statistics


def aggregate_over_sentences(cell: dict, alphas: list[float]) -> dict:
    """Mean ± stderr per α over the sentences of one cell; best α by emo_cos_sim_gt."""
    metric_keys = [
        'emo_cos_sim_gt', 'xvec_cos_sim_gt',
        'spk_cos_sim_neutral_qwen', 'spk_cos_sim_neutral_wavlm',
        'xvec_cos_sim_gt_wavlm', 'wer_raw', 'wer_norm', 'utmos',
    ]
    out: dict = {'per_alpha': {}}
    for alpha in alphas:
        cond_name = f'alpha_{alpha:.2f}'
        agg: dict = {}
        for k in metric_keys:
            vals: list[float] = []
            for blk in cell['per_sentence'].values():
                m = ((blk['conditions'].get(cond_name) or {}).get('metrics')) or {}
                v = m.get(k)
                if isinstance(v, (int, float)) and v == v:
                    vals.append(float(v))
            if not vals:
                agg[k] = {'mean': float('nan'), 'stderr': float('nan'), 'n': 0}
                continue
            mean = statistics.fmean(vals)
            stderr = statistics.stdev(vals) / (len(vals) ** 0.5) if len(vals) > 1 else 0.0
            agg[k] = {'mean': mean, 'stderr': stderr, 'n': len(vals)}
        out['per_alpha'][cond_name] = {'alpha': alpha, **agg}

    best_alpha, best_emo = None, float('-inf')
    for blk in out['per_alpha'].values():
        m = blk['emo_cos_sim_gt']['mean']
        if m == m and m > best_emo:
            best_emo, best_alpha = m, blk['alpha']
    out['best_alpha'] = best_alpha
    out['baseline_alpha'] = 0.0
    return out


def build_summary(all_results: dict) -> dict:
    rows = []
    baseline_rows = []
    for cell_key, cell in all_results.items():
        target, tau_variant, emotion = cell_key.split('__')
        agg = cell.get('aggregates') or {}
        best_alpha = agg.get('best_alpha')
        per_alpha = agg.get('per_alpha') or {}
        best_blk = per_alpha.get(f'alpha_{best_alpha:.2f}') if best_alpha is not None else None
        base_blk = per_alpha.get('alpha_0.00')

        def m(blk, k):
            if not blk:
                return None
            v = blk.get(k) or {}
            return v.get('mean')

        def se(blk, k):
            if not blk:
                return None
            v = blk.get(k) or {}
            return v.get('stderr')

        def row(blk, alpha_label):
            return {
                'target': target,
                'tau_variant': tau_variant,
                'emotion': emotion,
                'alpha': alpha_label,
                'n': (blk.get('emo_cos_sim_gt') or {}).get('n') if blk else 0,
                'emo_cos_sim_gt': (m(blk, 'emo_cos_sim_gt'), se(blk, 'emo_cos_sim_gt')),
                'xvec_cos_sim_gt': (m(blk, 'xvec_cos_sim_gt'), se(blk, 'xvec_cos_sim_gt')),
                'spk_cos_sim_neutral_qwen': (
                    m(blk, 'spk_cos_sim_neutral_qwen'), se(blk, 'spk_cos_sim_neutral_qwen'),
                ),
                'spk_cos_sim_neutral_wavlm': (
                    m(blk, 'spk_cos_sim_neutral_wavlm'), se(blk, 'spk_cos_sim_neutral_wavlm'),
                ),
                'xvec_cos_sim_gt_wavlm': (
                    m(blk, 'xvec_cos_sim_gt_wavlm'), se(blk, 'xvec_cos_sim_gt_wavlm'),
                ),
                'wer_raw': (m(blk, 'wer_raw'), se(blk, 'wer_raw')),
                'wer_norm': (m(blk, 'wer_norm'), se(blk, 'wer_norm')),
                'utmos': (m(blk, 'utmos'), se(blk, 'utmos')),
            }

        if best_blk:
            rows.append(row(best_blk, best_alpha))
        if base_blk:
            baseline_rows.append(row(base_blk, 0.0))

    return {'per_cell': all_results, 'best_per_cell': rows, 'baseline_per_cell': baseline_rows}


def _fmt(pair, fmt='{:+.4f}'):
    if pair is None:
        return '—'
    mean, se = pair
    if mean is None or mean != mean:
        return '—'
    return f'{fmt.format(mean)} ± {se:.4f}' if se is not None and se == se else fmt.format(mean)


def render_markdown(summary: dict) -> str:
    rows = sorted(
        summary['best_per_cell'], key=lambda r: (r['target'], r['tau_variant'], r['emotion'])
    )
    baselines = {
        (r['target'], r['tau_variant'], r['emotion']): r
        for r in summary['baseline_per_cell']
    }
    head = (
        '# EN→EN cross-speaker α-sweep (n=30, ESD test split 321–350)\n\n'
        '## Best α per (target × τ × emotion), mean ± stderr over N sentences\n\n'
        '| target | τ | emotion | best α | N | emo_cos_sim_gt | xvec_cos_sim_gt | '
        'spk_cos_sim_neu (Qwen ECAPA) | SECS_W (WavLM, indep.) | UTMOS | WER_raw | WER_norm |\n'
        '|---|---|---|---|---|---|---|---|---|---|---|---|\n'
    )

    def row_md(r, alpha_label):
        return (
            f"| {r['target']} | {r['tau_variant']} | {r['emotion']} | {alpha_label} | "
            f"{r['n']} | {_fmt(r['emo_cos_sim_gt'])} | {_fmt(r['xvec_cos_sim_gt'])} | "
            f"{_fmt(r['spk_cos_sim_neutral_qwen'])} | {_fmt(r['spk_cos_sim_neutral_wavlm'])} | "
            f"{_fmt(r['utmos'], '{:.3f}')} | "
            f"{_fmt(r['wer_raw'], '{:.3f}')} | {_fmt(r['wer_norm'], '{:.3f}')} |\n"
        )

    body = ''.join(row_md(r, r['alpha']) for r in rows)

    head_base = (
        '\n## α=0 baseline (Shaheen-style ICL-pure, no x-vector manipulation)\n\n'
        '| target | τ | emotion | N | emo_cos_sim_gt | xvec_cos_sim_gt | '
        'spk_cos_sim_neu (Qwen) | SECS_W (WavLM) | UTMOS | WER_raw | WER_norm |\n'
        '|---|---|---|---|---|---|---|---|---|---|---|\n'
    )
    body_base = ''
    for r in rows:
        b = baselines.get((r['target'], r['tau_variant'], r['emotion']))
        if not b:
            continue
        body_base += (
            f"| {b['target']} | {b['tau_variant']} | {b['emotion']} | {b['n']} | "
            f"{_fmt(b['emo_cos_sim_gt'])} | {_fmt(b['xvec_cos_sim_gt'])} | "
            f"{_fmt(b['spk_cos_sim_neutral_qwen'])} | {_fmt(b['spk_cos_sim_neutral_wavlm'])} | "
            f"{_fmt(b['utmos'], '{:.3f}')} | "
            f"{_fmt(b['wer_raw'], '{:.3f}')} | {_fmt(b['wer_norm'], '{:.3f}')} |\n"
        )

    note = (
        '\n_Best α picked by argmax over `mean(emo_cos_sim_gt)`. '
        '`SECS_W (WavLM)` is the independent-encoder SECS (`microsoft/wavlm-base-plus-sv`), '
        'ortogonal ao ECAPA-TDNN interno do Qwen3-TTS (onde τ vive). `UTMOS` = MOS predicted '
        'via UTMOSv2 (Baba et al. 2024 SLT, VoiceMOS Challenge 2024 #1). '
        '`WER_norm` aplica `whisper.normalizers.EnglishTextNormalizer` (sem custom alias map)._\n'
    )
    return head + body + head_base + body_base + note
