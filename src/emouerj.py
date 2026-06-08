"""emoUERJ (Brazilian Portuguese) dataset layout for the cross-lingual sweep.

Shared between the PT-BR α-sweep (scripts/repro/run_ptbr_sweep.py) and the PT-BR
ground-truth ceiling (scripts/repro/compute_gt_ceiling_ptbr.py).

Two cell designs, unified via the utterance model
`(ref_audio, ref_text, gt_audio, synth_text)`:
  - **paired** (m03, m04): same sentence neutral→emotion (ref_text == synth_text).
  - **xtext**  (w04): neutral ref take + anchor emotion sentence (cross-text).
"""

from __future__ import annotations

from src.paths import emouerj_root

EMOUERJ_ROOT = emouerj_root()

PAIRED_SPK = ['m03', 'm04']
XTEXT_SPK = ['w04']

# Canonical text per sentence code (mirrors the emoUERJ transcription).
FRASE_TEXT = {
    'G': 'A garrafa está na geladeira.',
    'C': 'Não importa quem está certo.',
    'M': 'De quem são essas malas que estão debaixo da mesa?',
    'Q': 'Ele volta na quarta-feira.',
    'F': 'Nos fins de semana, eu sempre ia para a casa dele.',
    'A': 'Você poderia arrumar a mesa, por favor?',
    'I': 'Você perde tempo demais com a Internet.',
    'T': 'Eu estou um pouco atrasado.',
    'D': 'Eu estou me sentindo doente hoje.',
    'B': 'Já chega! Eu vou tomar um banho e ir para a cama.',
}

# Paired design (same sentence neutral + emotion). Stems without `.wav`.
# m03 = 10 sentences × 4 emotions (all present).
# m04 = sad missing G and Q (n=8 pairs for sad).
PAIRED: dict[str, dict[str, dict[str, str]]] = {
    'm03': {
        'G': {'neutral': 'm03n03', 'angry': 'm03a03', 'happy': 'm03h03', 'sad': 'm03s03'},
        'C': {'neutral': 'm03n01', 'angry': 'm03a01', 'happy': 'm03h01', 'sad': 'm03s01'},
        'M': {'neutral': 'm03n07', 'angry': 'm03a07', 'happy': 'm03h07', 'sad': 'm03s07'},
        'Q': {'neutral': 'm03n08', 'angry': 'm03a08', 'happy': 'm03h08', 'sad': 'm03s08'},
        'F': {'neutral': 'm03n06', 'angry': 'm03a06', 'happy': 'm03h06', 'sad': 'm03s06'},
        'A': {'neutral': 'm03n10', 'angry': 'm03a10', 'happy': 'm03h10', 'sad': 'm03s10'},
        'I': {'neutral': 'm03n02', 'angry': 'm03a02', 'happy': 'm03h02', 'sad': 'm03s02'},
        'T': {'neutral': 'm03n05', 'angry': 'm03a05', 'happy': 'm03h05', 'sad': 'm03s05'},
        'D': {'neutral': 'm03n04', 'angry': 'm03a04', 'happy': 'm03h04', 'sad': 'm03s04'},
        'B': {'neutral': 'm03n09', 'angry': 'm03a09', 'happy': 'm03h09', 'sad': 'm03s09'},
    },
    'm04': {
        'G': {'neutral': 'm04n03', 'angry': 'm04a03', 'happy': 'm04h03'},
        'C': {'neutral': 'm04n01', 'angry': 'm04a01', 'happy': 'm04h01', 'sad': 'm04s01'},
        'M': {'neutral': 'm04n07', 'angry': 'm04a07', 'happy': 'm04h07', 'sad': 'm04s06'},
        'Q': {'neutral': 'm04n08', 'angry': 'm04a08', 'happy': 'm04h08'},
        'F': {'neutral': 'm04n06', 'angry': 'm04a06', 'happy': 'm04h06', 'sad': 'm04s04'},
        'A': {'neutral': 'm04n10', 'angry': 'm04a10', 'happy': 'm04h10', 'sad': 'm04s09'},
        'I': {'neutral': 'm04n02', 'angry': 'm04a02', 'happy': 'm04h02', 'sad': 'm04s02'},
        'T': {'neutral': 'm04n05', 'angry': 'm04a05', 'happy': 'm04h05', 'sad': 'm04s05'},
        'D': {'neutral': 'm04n04', 'angry': 'm04a04', 'happy': 'm04h04', 'sad': 'm04s03'},
        'B': {'neutral': 'm04n09', 'angry': 'm04a09', 'happy': 'm04h09', 'sad': 'm04s07'},
    },
}

# Cross-text design (w04): neutral ref ≠ synthesized text; GT same-text as synth.
# `neutral_refs` is cycled across the GT takes to vary the x-vec base.
XTEXT: dict[str, dict] = {
    'w04': {
        'neutral_refs': [
            'w04n06', 'w04n07', 'w04n08', 'w04n09', 'w04n10',   # G - "A garrafa…"
            'w04n01', 'w04n02', 'w04n03', 'w04n04', 'w04n05',   # T - "Eu estou um pouco atrasado."
        ],
        'angry': {
            'B': [f'w04a{i:02d}' for i in range(1, 13)],         # n=12
            'M': [f'w04a{i:02d}' for i in range(13, 17)],        # n=4
        },
        'happy': {
            # w04h01 = "Não importa quem está certo." (n=1) → omitted by default.
            'Q': [f'w04h{i:02d}' for i in range(2, 13)],         # n=11
        },
        'sad': {
            'D': [f'w04s{i:02d}' for i in range(1, 11)],         # n=10
            'F': [f'w04s{i:02d}' for i in range(11, 15)],        # n=4
            'B': [f'w04s{i:02d}' for i in range(15, 19)],        # n=4
        },
    },
}
# Maps each w04 neutral_ref to the sentence code it speaks.
W04_NEU_CODE = {f'w04n{n:02d}': ('G' if n >= 6 else 'T') for n in range(1, 11)}


def wav_path(stem: str) -> str:
    return f'{EMOUERJ_ROOT}/{stem}.wav'


def paired_utterances(spk: str, emotion: str) -> list[dict]:
    out = []
    for code, files in PAIRED[spk].items():
        if 'neutral' not in files or emotion not in files:
            continue
        out.append({
            'key': f'utt_{code}',
            'frase_code': code,
            'ref_stem': files['neutral'],
            'ref_text': FRASE_TEXT[code],
            'gt_stem': files[emotion],
            'synth_text': FRASE_TEXT[code],
        })
    return out


def xtext_utterances(spk: str, emotion: str, max_per_anchor: int) -> list[dict]:
    out = []
    spec = XTEXT[spk]
    neu = spec['neutral_refs']
    for anchor, gt_takes in spec[emotion].items():
        for i, gt in enumerate(gt_takes[:max_per_anchor]):
            ref = neu[i % len(neu)]
            out.append({
                'key': f'utt_{anchor}_{i:02d}',
                'frase_code': anchor,
                'ref_stem': ref,
                'ref_text': FRASE_TEXT[W04_NEU_CODE[ref]],
                'gt_stem': gt,
                'synth_text': FRASE_TEXT[anchor],
            })
    return out


def utterances_for(spk: str, emotion: str, max_per_anchor: int) -> list[dict]:
    if spk in PAIRED:
        return paired_utterances(spk, emotion)
    return xtext_utterances(spk, emotion, max_per_anchor)
