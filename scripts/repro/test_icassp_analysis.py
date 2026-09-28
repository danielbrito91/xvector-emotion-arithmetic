"""Testes de seleção de alpha, pareamento e leitura do fatorial."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_icassp_revision import METRICS, factorial, paired_summary, select_source_alpha


class AnalysisTests(unittest.TestCase):
    def test_portuguese_scores_do_not_select_alpha(self):
        rows = [
            dict(
                language=lang,
                tau_variant='avg4spk',
                emotion=emotion,
                alpha=alpha,
                esim_e2v_gt=score,
            )
            for lang in ['en', 'pt']
            for emotion in ['angry', 'happy', 'sad']
            for alpha, score in [(0.0, 0.0), (1.0, 0.2), (1.5, 0.8), (2.0, 0.3), (2.5, 0.4)]
        ]
        frame = pd.DataFrame(rows)
        expected = dict.fromkeys(['angry', 'happy', 'sad'], 1.5)
        self.assertEqual(select_source_alpha(frame), expected)
        frame.loc[frame.language.eq('pt') & frame.alpha.eq(2.5), 'esim_e2v_gt'] = 1e20
        self.assertEqual(select_source_alpha(frame), expected)

    def test_pairing_rejects_missing_and_duplicate_endpoints(self):
        base = pd.DataFrame([
            dict(
                language='pt',
                target='m03',
                emotion='angry',
                utt=str(i),
                tau_variant='avg4spk',
                **dict.fromkeys(METRICS, 0.2),
            )
            for i in range(2)
        ])
        self.assertEqual(paired_summary(base, base)['n'], 2)
        with self.assertRaises(AssertionError):
            paired_summary(base, base.iloc[:1])
        with self.assertRaises(pd.errors.MergeError):
            paired_summary(base, pd.concat([base, base.iloc[:1]]))

    def test_balanced_factorial_and_invalid_conditions(self):
        folder = Path(tempfile.mkdtemp(prefix='icassp_analysis_test_'))
        artifact = folder / 'synthetic_fixture.bin'
        artifact.write_bytes(b'fixture de teste; nao e audio')
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        rows = []
        for speaker in ['0011', '0013', '0017', '0019']:
            for ref in range(1, 11):
                for utt in range(321, 331):
                    for condition in [
                        'neutral_baseline',
                        'full_swap',
                        'xvec_swap',
                        'angry_baseline',
                    ]:
                        effect = int(condition in ['xvec_swap', 'angry_baseline'])
                        rows.append(
                            dict(
                                key=f'{speaker}/{ref}/{utt}/{condition}',
                                speaker=speaker,
                                ref=ref,
                                utt=utt,
                                condition=condition,
                                eca_e2v=effect,
                                eca_sv=effect,
                                duration_s=2.0,
                                wer_norm=0.0,
                                spk_cos_sim_neutral_wavlm=0.9,
                                wav_path=str(artifact),
                                sha256=digest,
                            )
                        )
        frame = pd.DataFrame(rows)
        path = folder / 'scores.jsonl'
        frame.to_json(path, orient='records', lines=True)
        result = factorial(folder, 'balanced')
        self.assertEqual(result['n_units'], 400)
        for metric in ['eca_e2v', 'eca_sv']:
            contrast = result['aggregate'][metric]['contrasts']['x_minus_tokens']
            self.assertEqual(contrast['delta'], 1.0)
            np.testing.assert_array_equal(contrast['ci95'], [1.0, 1.0])
            self.assertEqual(result['parallel_only_sensitivity'][metric]['n_units'], 380)
        frame.loc[frame.condition.eq('xvec_swap'), 'condition'] = 'missing_arm'
        frame.to_json(path, orient='records', lines=True)
        with self.assertRaises(AssertionError):
            factorial(folder, 'balanced')


if __name__ == '__main__':
    unittest.main()
