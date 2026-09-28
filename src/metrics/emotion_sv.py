"""Rótulos, probabilidades e embeddings emocionais do SenseVoiceSmall.

Usa a posição 1 do encoder, correspondente ao token CTC de emoção. O argmax
exclui EMO_UNKNOWN e renormaliza as classes emocionais; prob_unk preserva sua
massa original para diagnóstico. Português usa language="auto"."""

from __future__ import annotations

from functools import lru_cache

import torch

MODEL_ID = 'FunAudioLLM/SenseVoiceSmall'
EMO_POS = 1  # Posição do token de emoção na saída CTC.
EMO_TAGS = [
    'HAPPY',
    'SAD',
    'ANGRY',
    'NEUTRAL',
    'FEARFUL',
    'DISGUSTED',
    'SURPRISED',
]
UNK_TAG = 'EMO_UNKNOWN'


class _EncoderTap(torch.nn.Module):
    """Wrapper que guarda a saída do encoder sem alterar o forward."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner
        self.last: torch.Tensor | None = None

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        enc = out[0] if isinstance(out, tuple) else out
        self.last = enc.detach()
        return out


@lru_cache(maxsize=1)
def _load():
    from funasr import AutoModel

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    am = AutoModel(model=MODEL_ID, hub='hf', device=device, disable_update=True)
    core = am.model
    tap = _EncoderTap(core.encoder)
    core.encoder = tap
    tokenizer = am.kwargs['tokenizer']

    tag_ids: dict[str, int] = {}
    for tag in [*EMO_TAGS, UNK_TAG]:
        ids = tokenizer.encode(f'<|{tag}|>')
        tag_ids[tag] = int(ids[0]) if len(ids) == 1 else int(ids[0])
    return am, core, tap, tag_ids


def predict(audio_path: str, language: str = 'auto') -> dict:
    """Rótulo, distribuição e embedding de emoção para um wav."""
    am, core, tap, tag_ids = _load()
    am.generate(input=audio_path, language=language, use_itn=False)
    enc = tap.last
    if enc is None:
        raise RuntimeError(f'SenseVoice: encoder não capturado para {audio_path}')

    logp = core.ctc.log_softmax(enc)[0, EMO_POS]
    probs_all = logp.exp().float().cpu()

    unk = float(probs_all[tag_ids[UNK_TAG]])
    raw = {t: float(probs_all[tag_ids[t]]) for t in EMO_TAGS}
    total = sum(raw.values())
    probs = {t.lower(): (v / total if total > 0 else float('nan')) for t, v in raw.items()}

    return {
        'label': max(probs, key=probs.get) if total > 0 else 'unk',
        'probs': probs,
        'prob_unk': unk,
        'embedding': enc[0, EMO_POS, :].float().cpu(),
    }


def restrict(probs: dict[str, float], classes: list[str]) -> dict[str, float]:
    """Renormaliza a distribuição sobre um subconjunto de classes (forced-choice)."""
    sub = {c: probs.get(c, 0.0) for c in classes}
    total = sum(sub.values())
    if total <= 0:
        return {c: float('nan') for c in classes}
    return {c: v / total for c, v in sub.items()}
