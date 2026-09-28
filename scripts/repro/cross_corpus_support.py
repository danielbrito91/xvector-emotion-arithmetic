"""Manifestos, pareamento e bootstrap da avaliação em CREMA-D e VERBO."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

CLASSES = ('neutral', 'angry', 'happy', 'sad')
EMOTIONS = CLASSES[1:]
ALPHA = dict(angry=2.5, happy=1.5, sad=2.5)
CREMA_TRANSCRIPTS = {
    'IEO': "It's eleven o'clock.",
    'DFA': "Don't forget a jacket.",
    'IOM': "I'm on my way to the meeting.",
    'ITH': "I think I have a doctor's appointment.",
    'ITS': "I think I've seen this before.",
    'IWL': 'I would like a new alarm clock.',
    'IWW': 'I wonder what this is about.',
    'WSI': "We'll stop in a couple of minutes.",
}
VERBO_TRANSCRIPTS = {
    's1': 'Os operários levantam cedo.',
    'l1': 'Os bombeiros estão equipados com uma arma.',
    'l2': 'No próximo outono, Antônio vai a Minas em quinze de outubro.',
    'l3': 'Agora vou pôr a camiseta e sair para uma caminhada.',
    'l4': 'Um momento depois, ele caminhou e tropeçou.',
    'l5': 'Eu queria o número de telefone de seu João.',
    'q1': 'Sábado à noite, o que vai fazer?',
    'q2': 'Você vai trazer aquela coisa com você?',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':')
    )


def objhash(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_once(path, obj, lines=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        ''.join(canonical(x) + '\n' for x in obj)
        if lines
        else json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    )
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(
                f'Conflito com artefato existente: {path}; preservar e abrir versão explícita.'
            )
        return
    with path.open('x') as f:
        f.write(text)


def append_event(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(canonical(obj) + '\n')
        f.flush()


def validate_contract(c, repo=None):
    expected = {
        'model': 'Qwen3-TTS-12Hz-1.7B-Base',
        'tau_variant': 'avg4spk',
        'alpha': ALPHA,
        'conditions': list(CLASSES),
        'seed_master': 20260907,
        'seed_replicates': [0, 1, 2],
        'generation_language': 'Auto',
        'max_new_tokens': 2048,
        'x_vector_only_mode': False,
        'include_utmos': False,
        'primary_metric': 'eca_sv_4way',
        'bootstrap_replicates': 10000,
        'bootstrap_seed': 20260908,
        'report_ci': 0.95,
        'simultaneous_primary_ci_per_contrast': 0.975,
    }
    for k, v in expected.items():
        if c[k] != v:
            raise ValueError(f'Contrato fora do recorte: {k}')
    if set(c['corpora']) != {'crema_d', 'verbo'}:
        raise ValueError('Corpora')
    for corpus, cc in c['corpora'].items():
        refs, texts = cc['reference_text_ids'], cc['synthesis_text_ids']
        if (
            len(refs) != 1
            or len(set(texts)) != 6
            or set(refs) & set(texts)
            or cc['pilot_text_id'] in refs + texts
            or cc['speakers_n'] != 12
        ):
            raise ValueError('Grade ou referência')
    if c['corpora']['crema_d']['intensity'] != 'XX':
        raise ValueError('Intensidade')
    for corpus, refs, texts, pilot in [
        ('crema_d', ['IEO'], ['DFA', 'IOM', 'ITH', 'ITS', 'IWL', 'IWW'], 'WSI'),
        ('verbo', ['l5'], ['s1', 'l1', 'l2', 'l3', 'l4', 'q1'], 'q2'),
    ]:
        cc = c['corpora'][corpus]
        if (cc['reference_text_ids'], cc['synthesis_text_ids'], cc['pilot_text_id']) != (
            refs,
            texts,
            pilot,
        ):
            raise ValueError('Textos fora do contrato')
    if repo:
        repo = Path(repo)
        if read_json(repo / c['alpha_provenance'])['objective']['source_only_alpha'] != c['alpha']:
            raise ValueError('α difere da fonte')
        for r in c['tau_files'].values():
            if digest(repo / r['path']) != r['sha256']:
                raise ValueError('τ difere da fonte')


def rng_seed(corpus, speaker, text, reference, replicate):
    key = f'20260907|{corpus}|{speaker}|{text}|{reference}|{replicate}'
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % (2**63)
    return key, seed, seed % (2**32)


def parse_audio(corpus, path):
    p = Path(path)
    if corpus == 'verbo':
        emo, sp, txt = p.stem.split('-')
        if p.parent.name != sp:
            raise ValueError('Falante/path divergentes')
        mapping = {'neu': 'neutral', 'rai': 'angry', 'ale': 'happy', 'tri': 'sad'}
    else:
        sp, txt, emo, intensity = p.stem.split('_')
        if intensity != 'XX':
            raise ValueError('Intensidade CREMA-D esperada: XX')
        mapping = {'NEU': 'neutral', 'ANG': 'angry', 'HAP': 'happy', 'SAD': 'sad'}
    return sp, txt, mapping[emo]


def audio_info(path):
    import soundfile as sf

    data, sr = sf.read(path, dtype='float32', always_2d=True)
    if data.size == 0 or not np.isfinite(data).all() or not np.any(data):
        raise ValueError(f'Áudio ilegível/vazio/não finito/silencioso: {path}')
    info = sf.info(path)
    return dict(
        sha256=digest(path),
        sample_rate=sr,
        channels=data.shape[1],
        frames=len(data),
        duration_s=len(data) / sr,
        peak=float(np.abs(data).max()),
        rms=float(np.sqrt(np.mean(data.astype('float64') ** 2))),
        clipping_fraction=float(np.mean(np.abs(data) >= 0.999)),
        subtype=info.subtype,
        normalization='original; nenhuma alteração',
    )


def build_manifests(c, speakers, roots, transcripts, model_revision, config_hash):
    """Monta os manifestos de síntese a partir de referências neutras e textos."""
    synthesis = []
    natural = []
    pilot = []
    for corpus, cc in c['corpora'].items():
        root = Path(roots[corpus])
        refid = cc['reference_text_ids'][0]

        def path(sp, t, e):
            alias = next(k for k, v in cc['label_map'].items() if v == e)
            return (
                root / sp / f'{alias}-{sp}-{t}.wav'
                if corpus == 'verbo'
                else root / f'{sp}_{t}_{alias}_XX.wav'
            )

        selected = speakers[corpus]
        if len(selected) != 12 or len(set(selected)) != 12:
            raise ValueError('12 falantes necessários')
        pilot_sp = cc.get('pilot_speaker_ids', [selected[0], selected[6]])
        for sp in selected:
            ref = path(sp, refid, 'neutral')
            refhash = digest(ref)
            for t in [refid, *cc['synthesis_text_ids']]:
                for e in CLASSES if t != refid else ('neutral',):
                    wav = path(sp, t, e)
                    natural.append(
                        dict(
                            output_id=f'natural_{corpus}_{sp}_{t}_{e}',
                            corpus=corpus,
                            language=cc['language'],
                            speaker_id=sp,
                            text_id=t,
                            role='reference' if t == refid else 'evaluation',
                            emotion=e,
                            audio_path=str(wav),
                            text=transcripts[corpus][t],
                            reference_path=str(ref),
                            reference_sha256=refhash,
                        )
                    )
            for stage, texts, reps in [
                ('main', cc['synthesis_text_ids'], c['seed_replicates']),
                ('pilot', [cc['pilot_text_id']] if sp in pilot_sp else [], [0]),
            ]:
                for t, r in itertools.product(texts, reps):
                    key, seed, npseed = rng_seed(corpus, sp, t, refid, r)
                    uid = f'{corpus}_{sp}_{t}_{refid}_r{r}'
                    for condition in c['conditions']:
                        tau = c['tau_files'].get(condition)
                        row = dict(
                            corpus=corpus,
                            language=cc['language'],
                            speaker_id=sp,
                            text_id=t,
                            reference_id=refid,
                            replicate=r,
                            unit_id=uid,
                            seed_key=key,
                            seed=seed,
                            numpy_seed=npseed,
                            condition=condition,
                            alpha=c['alpha'].get(condition, 0.0),
                            tau_sha256=tau['sha256'] if tau else None,
                            tau_path=tau['path'] if tau else None,
                            reference_path=str(ref),
                            reference_sha256=refhash,
                            reference_text=transcripts[corpus][refid],
                            text=transcripts[corpus][t],
                            transcript_hash=objhash({
                                'reference': transcripts[corpus][refid],
                                'target': transcripts[corpus][t],
                            }),
                            model_revision=model_revision,
                            generation_config_hash=config_hash,
                            output_id=uid + '_' + condition,
                            baseline_output_id=uid + '_neutral',
                        )
                        (synthesis if stage == 'main' else pilot).append(row)
    validate_worklist(synthesis, True)
    if len(natural) != 600 or len(pilot) != 16:
        raise ValueError('Cardinalidade natural/piloto')
    return synthesis, natural, pilot


def validate_worklist(rows, full=False):
    ids = [r['output_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('output_id duplicado')
    units = {}
    for r in rows:
        if r['text_id'] == r['reference_id']:
            raise ValueError('Referência é alvo')
        if any(k.startswith('gt') or k in ('emotion', 'scores', 'audio_path') for k in r):
            raise ValueError('Manifesto de síntese contém campos de avaliação')
        if r['alpha'] != ALPHA.get(r['condition'], 0.0):
            raise ValueError('Alpha difere da configuração source-only')
        key, seed, nps = rng_seed(
            r['corpus'], r['speaker_id'], r['text_id'], r['reference_id'], r['replicate']
        )
        if (key, seed, nps) != (r['seed_key'], r['seed'], r['numpy_seed']):
            raise ValueError('Semente difere da chave da unidade')
        units.setdefault(r['unit_id'], []).append(r)
    if len({rs[0]['seed'] for rs in units.values()}) != len(units):
        raise ValueError('Colisão de semente')
    for rs in units.values():
        if sorted(r['condition'] for r in rs) != sorted(CLASSES):
            raise ValueError('Braço ausente/duplicado')
        base = next(r for r in rs if r['condition'] == 'neutral')
        if any(r['baseline_output_id'] != base['output_id'] for r in rs):
            raise ValueError('Baseline não compartilhado')
        constants = (
            'seed',
            'reference_sha256',
            'transcript_hash',
            'model_revision',
            'generation_config_hash',
            'reference_path',
            'reference_text',
            'text',
        )
        if any(len({canonical(r[k]) for r in rs}) != 1 for k in constants):
            raise ValueError('Campos da unidade divergem entre condições')
    if full and (
        len(rows) != 1728
        or any(sum(r['corpus'] == corpus for r in rows) != 864 for corpus in ('verbo', 'crema_d'))
    ):
        raise ValueError('Grade incompleta')
    return {'unique_wavs': len(rows), 'units': len(units), 'contrasts': len(units) * 3}


ALIASES = {
    'anger': 'angry',
    'happiness': 'happy',
    'sadness': 'sad',
    'neu': 'neutral',
    'rai': 'angry',
    'ale': 'happy',
    'tri': 'sad',
    'hap': 'happy',
    'ang': 'angry',
}


def restrict_probs(probs):
    mapped = {}
    for k, v in probs.items():
        key = k.split('/')[-1].lower()
        key = ALIASES.get(key, key)
        if not math.isfinite(v) or v < 0:
            raise ValueError('Massa SER inválida')
        if key in mapped:
            raise ValueError('Alias duplicado')
        mapped[key] = float(v)
    if not all(k in mapped for k in CLASSES):
        raise ValueError('Classe SER ausente')
    mass = sum(mapped[k] for k in CLASSES)
    if mass <= 0:
        raise ValueError('Massa four-way nula')
    restricted = {k: mapped[k] / mass for k in CLASSES}
    return max(restricted, key=restricted.get), restricted, mass


def bootstrap_weights(ns, nt, n=10000, seed=20260908):
    rng = np.random.default_rng(seed)
    s = rng.multinomial(ns, np.full(ns, 1 / ns), n) / ns
    t = rng.multinomial(nt, np.full(nt, 1 / nt), n) / nt
    return (s[:, :, None] * t[:, None, :]).reshape(n, ns * nt)


def summarize_grid(grid, weights=None):
    """Sementes já promediadas. Texto sorteado uma vez por réplica, comum a falantes/braços/emoções."""
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 3 or grid.shape[2] != 3 or not np.isfinite(grid).all():
        raise ValueError('Grade inválida')
    w = bootstrap_weights(*grid.shape[:2]) if weights is None else weights
    draws = w @ grid.reshape(-1, 3)
    macro = draws.mean(axis=1)
    return dict(
        delta=float(grid.mean()),
        ci95=np.quantile(macro, [0.025, 0.975]).tolist(),
        ci975=np.quantile(macro, [0.0125, 0.9875]).tolist(),
        by_emotion={
            e: {
                'delta': float(grid[:, :, i].mean()),
                'ci95': np.quantile(draws[:, i], [0.025, 0.975]).tolist(),
            }
            for i, e in enumerate(EMOTIONS)
        },
        leave_one_speaker_out=[
            float(np.delete(grid, i, axis=0).mean()) for i in range(grid.shape[0])
        ],
    )


def calibration_summary(rows):
    result = {}
    for corpus in ('crema_d', 'verbo'):
        subset = [r for r in rows if r['corpus'] == corpus and r['role'] == 'evaluation']
        speakers = sorted({r['speaker_id'] for r in subset})
        texts = sorted({r['text_id'] for r in subset})
        if len(speakers) != 12 or len(texts) != 6 or len(subset) != 288:
            raise ValueError('Calibração incompleta')
        lookup = {(r['speaker_id'], r['text_id'], r['emotion']): r for r in subset}
        if len(lookup) != 288:
            raise ValueError('Natural duplicado')
        out = {}
        for judge in ('sv', 'e2v'):
            grid = np.zeros((12, 6, 3))
            conf = np.zeros((4, 4), int)
            for si, sp in enumerate(speakers):
                for ti, txt in enumerate(texts):
                    base = lookup[sp, txt, 'neutral'][judge]['label']
                    for ei, e in enumerate(EMOTIONS):
                        grid[si, ti, ei] = int(lookup[sp, txt, e][judge]['label'] == e) - int(
                            base == e
                        )
                    for e in CLASSES:
                        label = lookup[sp, txt, e][judge]['label']
                        if label not in CLASSES:
                            raise ValueError('Scorer não produziu predição válida')
                        conf[CLASSES.index(e), CLASSES.index(label)] += 1
            summary = summarize_grid(grid)
            summary.update(
                confusion=conf.tolist(),
                class_order=list(CLASSES),
                neutral_accuracy=float(conf[0, 0] / conf[0].sum()),
                by_speaker={sp: grid[i].mean(axis=0).tolist() for i, sp in enumerate(speakers)},
                gate_pass=summary['ci95'][0] > 0,
            )
            out[judge] = summary
        result[corpus] = out
    return result


def analyze_scores(rows, manifest):
    validate_worklist(manifest, True)
    byid = {r['output_id']: r for r in rows}
    if len(byid) != len(rows) or set(byid) != {r['output_id'] for r in manifest}:
        raise ValueError('Scores duplicados/incompletos')
    result = {}
    for corpus in ('crema_d', 'verbo'):
        meta = [r for r in manifest if r['corpus'] == corpus]
        speakers = sorted({r['speaker_id'] for r in meta})
        texts = sorted({r['text_id'] for r in meta})
        ix = {
            (r['speaker_id'], r['text_id'], r['replicate'], r['condition']): byid[r['output_id']]
            for r in meta
        }
        out = {
            'n_speakers': 12,
            'n_texts': 6,
            'n_units': 216,
            'n_wavs': 864,
            'n_contrasts': 648,
            'synthesis_failures': sum(r['status'] == 'synthesis_failure' for r in ix.values()),
        }

        def prediction(r, j):
            if r['status'] == 'synthesis_failure':
                return None
            return r[j]['label']

        for judge in ('sv', 'e2v'):
            base = np.zeros((12, 6, 3, 3))
            edit = np.zeros_like(base)
            for si, sp in enumerate(speakers):
                for ti, t in enumerate(texts):
                    for r in range(3):
                        for ei, e in enumerate(EMOTIONS):
                            base[si, ti, r, ei] = prediction(ix[sp, t, r, 'neutral'], judge) == e
                            edit[si, ti, r, ei] = prediction(ix[sp, t, r, e], judge) == e
            delta = edit - base
            summary = summarize_grid(delta.mean(axis=2))
            summary.update(
                baseline=float(base.mean()),
                edited=float(edit.mean()),
                neutral_accuracy=float(
                    np.mean([
                        prediction(v, judge) == 'neutral'
                        for k, v in ix.items()
                        if k[-1] == 'neutral'
                    ])
                ),
                by_speaker={
                    sp: {
                        'baseline': float(base[i].mean()),
                        'edited': float(edit[i].mean()),
                        'delta': float(delta[i].mean()),
                        'by_emotion': delta[i].mean(axis=(0, 1)).tolist(),
                    }
                    for i, sp in enumerate(speakers)
                },
                leave_one_seed_out={
                    str(r): summarize_grid(np.delete(delta, r, axis=2).mean(axis=2))
                    for r in range(3)
                },
            )
            for ei, e in enumerate(EMOTIONS):
                summary['by_emotion'][e].update(
                    baseline=float(base[:, :, :, ei].mean()),
                    edited=float(edit[:, :, :, ei].mean()),
                )
            summary['by_speaker_emotion'] = {
                sp: {
                    e: {
                        'baseline': float(base[si, :, :, ei].mean()),
                        'edited': float(edit[si, :, :, ei].mean()),
                        'delta': float(delta[si, :, :, ei].mean()),
                    }
                    for ei, e in enumerate(EMOTIONS)
                }
                for si, sp in enumerate(speakers)
            }
            out[judge] = summary
        out['continuous'] = {}
        for metric in ('secs_w', 'wer_norm', 'duration_s'):
            pairs = []
            missing = 0
            for sp, t, r, e in itertools.product(speakers, texts, range(3), EMOTIONS):
                b = ix[sp, t, r, 'neutral'].get(metric)
                v = ix[sp, t, r, e].get(metric)
                if b is None or v is None:
                    missing += 1
                else:
                    pairs.append((sp, t, r, e, float(b), float(v)))
            out['continuous'][metric] = {
                'complete_pairs': len(pairs),
                'missing_pairs': missing,
                'baseline_complete_case': float(np.mean([p[-2] for p in pairs]))
                if pairs
                else None,
                'edited_complete_case': float(np.mean([p[-1] for p in pairs])) if pairs else None,
                'delta_complete_case': float(np.mean([p[-1] - p[-2] for p in pairs]))
                if pairs
                else None,
            }
        result[corpus] = out
    return result


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [(p + z * z / (2 * n)) / d - h, (p + z * z / (2 * n)) / d + h]


def simulate(n_worlds=2000):
    """DGP pareado com neutro categórico compartilhado, efeitos cruzados e 3 sementes.
    Probabilidades complementares e clamp simétrico garantem delta médio zero no nulo.
    O gate MC é fixado antes da execução: limite superior FWER <= .075.
    """
    rng = np.random.default_rng(20260909)
    w = bootstrap_weights(12, 6)
    scenarios = []
    for heterogeneity in (0.0, 0.04, 0.08):
        for baseline_target_p in (0.10, 0.25):
            for delta in (0.0, 0.05, 0.10, 0.20):
                detected = []
                family_errors = []
                covered = []
                means = []
                for start in range(0, n_worlds, 50):
                    nw = min(50, n_worlds - start)
                    nc = nw * 2
                    # A mesma probabilidade de neutral cair em cada classe solicitada.
                    neutral_u = rng.random((nc, 12, 6, 3))
                    neutral = np.stack(
                        [
                            (neutral_u >= i * baseline_target_p)
                            & (neutral_u < (i + 1) * baseline_target_p)
                            for i in range(3)
                        ],
                        axis=-1,
                    )
                    s = rng.uniform(-heterogeneity, heterogeneity, (nc, 12, 1, 1, 3))
                    t = rng.uniform(-heterogeneity, heterogeneity, (nc, 1, 6, 1, 3))
                    # Cap simétrico evita viés no nulo quando a heterogeneidade excede p_base.
                    effect = np.clip(s + t, -baseline_target_p, baseline_target_p)
                    p = baseline_target_p + delta + effect
                    # Parte do RNG comum (50%) e parte ruído de síntese independente.
                    independent = rng.random((nc, 12, 6, 3, 3))
                    common = np.mod(neutral_u[..., None] - np.arange(3) * baseline_target_p, 1.0)
                    use_common = rng.random((nc, 12, 6, 3, 1)) < 0.5
                    edited = np.where(use_common, common, independent) < p
                    grid = (edited.astype(float) - neutral).mean(axis=(3, 4))
                    boot = w @ grid.reshape(nc, 72).T
                    lo, hi = np.quantile(boot, [0.0125, 0.9875], axis=0)
                    detected.extend((lo.reshape(nw, 2) > 0).any(axis=1).tolist())
                    family_errors.extend(
                        ((lo.reshape(nw, 2) > delta) | (hi.reshape(nw, 2) < delta))
                        .any(axis=1)
                        .tolist()
                    )
                    covered.extend(((lo <= delta) & (hi >= delta)).tolist())
                    means.extend(grid.mean(axis=(1, 2)).tolist())
                k = sum(detected)
                scenarios.append(
                    dict(
                        heterogeneity=heterogeneity,
                        baseline_target_p=baseline_target_p,
                        delta=delta,
                        n_worlds=n_worlds,
                        family_any_positive=k / n_worlds,
                        mc_ci95=wilson(k, n_worlds),
                        family_noncoverage=float(np.mean(family_errors)),
                        family_noncoverage_mc_ci95=wilson(sum(family_errors), n_worlds),
                        coverage975=float(np.mean(covered)),
                        coverage_mc_ci95=wilson(sum(covered), len(covered)),
                        empirical_delta=float(np.mean(means)),
                    )
                )
    null = [s for s in scenarios if s['delta'] == 0]
    return dict(
        scenarios=scenarios,
        bootstrap_replicates=10000,
        seed=20260909,
        acceptance_rule='Antes dos scores: upper Wilson95 da taxa familiar bilateral de exclusão de zero <= 0.075 em cada cenário nulo (nominal FWER 0.05, tolerância diagnóstica de 0.025). Cobertura marginal e falso ganho positivo reportados separadamente.',
        gate_pass=all(s['family_noncoverage_mc_ci95'][1] <= 0.075 for s in null),
    )


def apply_reference_overrides(rows, overrides):
    """Corrige de/do nas referências VERBO l5 de f4 e f6, preservando o áudio."""
    import copy

    expected_before = 'Eu queria o número de telefone de seu João.'
    expected_after = 'Eu queria o número de telefone do seu João.'
    lookup = {(r['corpus'], r['speaker_id'], r['reference_id']): r for r in overrides}
    if len(overrides) != 2 or set(lookup) != {('verbo', 'f4', 'l5'), ('verbo', 'f6', 'l5')}:
        raise ValueError('Correção limitada às referências VERBO l5 de f4 e f6')
    if any(
        r['before'] != expected_before or r['proposed_after'] != expected_after for r in overrides
    ):
        raise ValueError('Correção esperada: de seu João → do seu João')
    result = copy.deepcopy(rows)
    changed = 0
    for row in result:
        entry = lookup.get((row['corpus'], row['speaker_id'], row['reference_id']))
        if not entry:
            continue
        if (
            row['reference_text'] != entry['before']
            or row['reference_sha256'] != entry['audio_sha256']
        ):
            raise ValueError('Referência não corresponde à revisão ou já foi emendada')
        row['reference_text'] = entry['proposed_after']
        row['transcript_hash'] = objhash({
            'reference': row['reference_text'],
            'target': row['text'],
        })
        changed += 1
    if changed != 144:
        raise ValueError('A emenda deve atingir exatamente 144 linhas')
    validate_worklist(result, True)
    return result
