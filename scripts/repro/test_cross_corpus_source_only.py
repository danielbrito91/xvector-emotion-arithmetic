#!/usr/bin/env python3
"""Testes de manifestos, denominadores e bootstrap cross-corpus."""

import argparse
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
from cross_corpus_support import (
    CLASSES,
    CREMA_TRANSCRIPTS,
    VERBO_TRANSCRIPTS,
    analyze_scores,
    apply_reference_overrides,
    bootstrap_weights,
    build_manifests,
    canonical,
    parse_audio,
    read_json,
    read_jsonl,
    restrict_probs,
    summarize_grid,
    validate_contract,
    validate_worklist,
    write_once,
)

OUT = None


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix='cross_corpus_tests_')
        cls.contract = (
            read_json(OUT / 'protocol.json')
            if OUT
            else read_json(Path(__file__).parent / 'configs/cross_corpus_source_only.json')
        )
        cls.speakers = {
            'crema_d': [str(1000 + i) for i in range(12)],
            'verbo': cls.contract['corpora']['verbo']['speaker_ids'],
        }
        cls.roots = {c: str(Path(cls.tmp) / c) for c in cls.speakers}
        for c, ss in cls.speakers.items():
            for s in ss:
                p = (
                    Path(cls.roots[c]) / s / f'neu-{s}-l5.wav'
                    if c == 'verbo'
                    else Path(cls.roots[c]) / f'{s}_IEO_NEU_XX.wav'
                )
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b'referencia artificial para hash; nao e WAV')
        cls.transcripts = {'crema_d': CREMA_TRANSCRIPTS, 'verbo': VERBO_TRANSCRIPTS}
        cls.rows, cls.natural, cls.pilot = cls.build()

    @classmethod
    def build(cls):
        return build_manifests(
            cls.contract, cls.speakers, cls.roots, cls.transcripts, 'model_test', 'config_test'
        )

    def test_source_contract_mutations(self):
        for key, value in [
            ('alpha', dict(angry=1.0, happy=1.5, sad=2.5)),
            ('conditions', ['neutral', 'neutral', 'happy', 'sad']),
            ('max_new_tokens', 1024),
            ('include_utmos', True),
            ('seed_replicates', [0]),
        ]:
            bad = copy.deepcopy(self.contract)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_contract(bad)

    def test_generation_independent_of_gt_and_scores(self):
        before = canonical(self.rows)
        p = Path(self.tmp) / 'natural_gt_and_scores.json'
        p.write_text(canonical(self.natural))
        p.rename(p.with_suffix('.hidden'))
        p.write_text(canonical({'all_labels': 'wrong', 'score': 1e200}))
        after, _, _ = self.build()
        self.assertEqual(before, canonical(after))

    def test_counts_and_neutral_sharing(self):
        self.assertEqual(
            validate_worklist(self.rows, True), dict(unique_wavs=1728, units=432, contrasts=1296)
        )
        self.assertEqual(len(self.natural), 600)
        self.assertEqual(len(self.pilot), 16)
        self.assertEqual(len({r['baseline_output_id'] for r in self.rows}), 432)

    def test_pairing_counterexamples(self):
        mutations = [
            lambda r: r.append(r[0]),
            lambda r: r[0].update(reference_id=r[0]['text_id']),
            lambda r: r[1].update(baseline_output_id='fake'),
            lambda r: r[1].update(reference_sha256='fake'),
            lambda r: r[0].update(gt_path='forbidden'),
            lambda r: r[0].update(numpy_seed=123),
        ]
        for mutate in mutations:
            bad = copy.deepcopy(self.rows)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_worklist(bad, True)

    def test_rng_matches_independent_sha_definition(self):
        import hashlib

        r = self.rows[100]
        key = '|'.join(
            map(
                str,
                [
                    20260907,
                    r['corpus'],
                    r['speaker_id'],
                    r['text_id'],
                    r['reference_id'],
                    r['replicate'],
                ],
            )
        )
        expected = int(hashlib.sha256(key.encode('utf8')).hexdigest()[:16], 16) & ((1 << 63) - 1)
        self.assertEqual(r['seed'], expected)
        self.assertEqual(r['numpy_seed'], expected % (2**32))
        self.assertEqual(
            sorted((r['output_id'], r['seed']) for r in self.rows),
            sorted((r['output_id'], r['seed']) for r in reversed(self.rows)),
        )

    def test_fourway_alias_unknown_and_invalid(self):
        label, p, m = restrict_probs(
            dict(anger=0.1, happiness=0.2, sadness=0.3, neutral=0.4, unk=99.0)
        )
        self.assertEqual(label, 'neutral')
        self.assertAlmostEqual(sum(p.values()), 1)
        self.assertAlmostEqual(m, 1)
        for bad in (
            dict.fromkeys(CLASSES, 0.0),
            dict(neutral=np.nan, angry=1.0, happy=1.0, sad=1.0),
            dict(neutral=1.0, angry=-1.0, happy=1.0, sad=1.0),
            dict(neutral=1.0, angry=1.0, happy=1.0),
        ):
            with self.assertRaises(ValueError):
                restrict_probs(bad)

    def test_parser_separates_speaker_text_emotion(self):
        self.assertEqual(parse_audio('verbo', '/x/f1/rai-f1-l3.wav'), ('f1', 'l3', 'angry'))
        with self.assertRaises(ValueError):
            parse_audio('verbo', '/x/m1/rai-f1-l3.wav')
        with self.assertRaises(ValueError):
            parse_audio('crema_d', '1001_DFA_ANG_HI.wav')

    def test_crossed_bootstrap_against_manual_draws(self):
        ns, nt, n, seed = 3, 2, 10, 91
        rng = np.random.default_rng(seed)
        sc = rng.multinomial(ns, [1 / ns] * ns, n)
        tc = rng.multinomial(nt, [1 / nt] * nt, n)
        grid = np.arange(ns * nt * 3).reshape(ns, nt, 3)
        actual = bootstrap_weights(ns, nt, n, seed) @ grid.reshape(-1, 3)
        expected = np.array([
            np.mean(grid[np.repeat(np.arange(ns), s)][:, np.repeat(np.arange(nt), t)], axis=(0, 1))
            for s, t in zip(sc, tc)
        ])
        np.testing.assert_allclose(actual, expected)

    def test_equal_arms_and_swap(self):
        rng = np.random.default_rng(9)
        grid = rng.normal(0.1, 0.2, (12, 6, 3))
        zero = summarize_grid(grid - grid)
        self.assertEqual(zero['delta'], 0.0)
        self.assertEqual(zero['ci95'], [0.0, 0.0])
        pos = summarize_grid(grid)
        neg = summarize_grid(-grid)
        np.testing.assert_allclose(neg['ci975'], -np.array(pos['ci975'])[::-1])
        self.assertAlmostEqual(pos['delta'], -neg['delta'])

    def test_outlier_exposed(self):
        grid = np.ones((12, 6, 3)) * 0.1
        grid[0] = -2
        summary = summarize_grid(grid)
        self.assertLess(summary['delta'], 0)
        self.assertGreater(summary['leave_one_speaker_out'][0], 0)

    def test_analysis_denominator_failures_and_duplicates(self):
        rows = []
        for r in self.rows:
            rows.append(
                dict(
                    output_id=r['output_id'],
                    status='success',
                    sv={'label': 'neutral'},
                    e2v={'label': 'neutral'},
                    secs_w=0.8,
                    wer_norm=0.0,
                    duration_s=2.0,
                )
            )
        summary = analyze_scores(rows, self.rows)
        self.assertEqual(summary['crema_d']['sv']['delta'], 0)
        self.assertEqual(summary['verbo']['sv']['neutral_accuracy'], 1)
        with self.assertRaises(ValueError):
            analyze_scores(rows + [rows[0]], self.rows)
        with self.assertRaises(ValueError):
            analyze_scores(rows[:-1], self.rows)
        rows[1] = dict(output_id=rows[1]['output_id'], status='synthesis_failure')
        failed = analyze_scores(rows, self.rows)['crema_d']
        self.assertEqual(failed['n_contrasts'], 648)
        self.assertEqual(failed['synthesis_failures'], 1)
        self.assertEqual(failed['continuous']['secs_w']['missing_pairs'], 1)

    def test_existing_output_never_overwritten(self):
        p = Path(self.tmp) / 'exclusive.json'
        write_once(p, {'test': 1})
        write_once(p, {'test': 1})
        with self.assertRaises(RuntimeError):
            write_once(p, {'test': 2})
        self.assertEqual(read_json(p), {'test': 1})

    def test_approved_reference_changes_only(self):
        overrides = [
            dict(
                corpus='verbo',
                speaker_id=sp,
                reference_id='l5',
                before=VERBO_TRANSCRIPTS['l5'],
                proposed_after=VERBO_TRANSCRIPTS['l5'].replace('de seu João', 'do seu João'),
                audio_sha256=next(
                    r['reference_sha256']
                    for r in self.rows
                    if r['corpus'] == 'verbo' and r['speaker_id'] == sp
                ),
            )
            for sp in ('f4', 'f6')
        ]
        amended = apply_reference_overrides(self.rows, overrides)
        changed = [(a, b) for a, b in zip(self.rows, amended) if a != b]
        self.assertEqual(len(changed), 144)
        for a, b in changed:
            self.assertEqual({k for k in a if a[k] != b[k]}, {'reference_text', 'transcript_hash'})
        for mutate in (
            lambda x: x[0].update(speaker_id='f1'),
            lambda x: x[0].update(proposed_after='texto escolhido por score'),
            lambda x: x[0].update(audio_sha256='outro_audio'),
            lambda x: x.append(x[0]),
        ):
            bad = copy.deepcopy(overrides)
            mutate(bad)
            with self.assertRaises(ValueError):
                apply_reference_overrides(self.rows, bad)
        with self.assertRaises(ValueError):
            apply_reference_overrides(amended, overrides)

    def test_real_artifact_manifest_if_present(self):
        if OUT and (OUT / 'manifest_synthesis.jsonl').exists():
            validate_worklist(read_jsonl(OUT / 'manifest_synthesis.jsonl'), True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path)
    a, remaining = p.parse_known_args()
    OUT = a.out
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
