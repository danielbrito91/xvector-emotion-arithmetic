"""Deploy-time helper: turn any base audio into an emotion-shifted clone.

Pipeline (mirrors `scripts/run_ptbr_sweep.py`, single-utt edition):

    1. ffmpeg-preprocess input → 24 kHz mono WAV with N dB of headroom
       (avoids 44.1k→24k overshoot + clip; cf. session_report_2026-05-21 §8).
    2. Optional Whisper transcription → `ref_text` (= `synth_text` by default).
    3. Extract base x-vec via Qwen3-TTS ECAPA speaker encoder.
    4. Hybrid x-vec = base + α · τ  (τ from `data/tau/tau_{emotion}_{variant}.pt`).
    5. Voice-clone synth with the hybrid x-vec, save with peak-norm guard.

Recommended defaults follow §4.2 of session_report_2026-05-21:
  - `tau_variant = "avg4spk"`  (preserves identity better cross-lingual)
  - `alpha       = 2.5`         (best-α for PT-BR angry in m03/m04 paired)

API entry point: `emotionize_audio(...)`. Single dict argument is RORO-friendly
but, since most callers want positional defaults, we expose keyword args.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from src.xvec import (
    cos_flat,
    extract_xvec,
    load_tau_artifact,
    load_tts,
    match_shape,
    save_wav,
    synthesize_with_xvec,
)

DEFAULT_TAU_DIR = 'data/tau'
DEFAULT_MODEL_PATH = './Qwen3-TTS-12Hz-1.7B-Base'
DEFAULT_EMOTION = 'angry'
DEFAULT_TAU_VARIANT = 'avg4spk'
DEFAULT_ALPHA = 2.5
DEFAULT_HEADROOM_DB = 1.0
DEFAULT_TARGET_SR = 24_000
DEFAULT_LANGUAGE = 'Auto'
DEFAULT_ASR_LANGUAGE = 'pt'


@dataclass
class EmotionizeResult:
    output_path: str
    preprocessed_ref_path: str
    ref_text: str
    synth_text: str
    emotion: str
    tau_variant: str
    tau_file: str
    alpha: float
    tau_norm: float
    base_xvec_norm: float
    hybrid_xvec_norm: float
    duration_s: float
    cos_base_tau: float
    """cos(base_xvec, τ). Predicts τ responsiveness for this speaker:
    >0.1 strong direction overlap; near 0 mostly orthogonal (push is angular);
    <0 partially cancels the base direction. cf. docs/deploy_caveats.md §3."""
    cos_base_hybrid: float
    """cos(base_xvec, hybrid_xvec). Realized angular shift in x-vec space.
    Close to 1.0 → barely moved; <0.99 → noticeable rotation."""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _f32_norm(t: torch.Tensor) -> float:
    """Norm in float32 (base xvec is bfloat16 from the model; bf16 step in
    [16, 32) is 0.125, so bf16 norm reporting can mask small but real
    α-induced changes)."""
    return float(t.detach().float().norm().item())


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError('ffmpeg not found on PATH; required for preprocessing.')
    return ffmpeg


def preprocess_to_24k_mono(
    in_path: str,
    out_path: str,
    target_sr: int = DEFAULT_TARGET_SR,
    headroom_db: float = DEFAULT_HEADROOM_DB,
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> str:
    """Resample to mono 24 kHz with N dB of headroom (clip-safe), optional crop.

    Matches the emoUERJ pre-proc from session_report_2026-05-21 §1/§8:
    `ffmpeg -af "volume=-1dB" -ar 24000 -ac 1`. The optional crop
    (`-ss start_s -t duration_s`, placed after `-i` for accurate seek)
    mitigates ICL ref-prosody leakage on long calm references —
    cf. `docs/deploy_caveats.md` §1 axis 5.
    """
    ffmpeg = _require_ffmpeg()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = [ffmpeg, '-y', '-i', in_path]
    if start_s > 0.0:
        cmd += ['-ss', f'{start_s:.3f}']
    if duration_s is not None:
        cmd += ['-t', f'{duration_s:.3f}']
    cmd += [
        '-af', f'volume=-{headroom_db}dB',
        '-ar', str(target_sr),
        '-ac', '1',
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def transcribe_ref(audio_path: str, language: str = DEFAULT_ASR_LANGUAGE) -> str:
    """Whisper-large-v3 transcription, default language=PT (deployment scenario)."""
    from src.metrics.asr import transcribe
    return transcribe(audio_path, language=language)


def resolve_tau_file(
    emotion: str,
    tau_variant: str = DEFAULT_TAU_VARIANT,
    tau_dir: str = DEFAULT_TAU_DIR,
) -> str:
    path = Path(tau_dir) / f'tau_{emotion}_{tau_variant}.pt'
    if not path.exists():
        raise FileNotFoundError(
            f'τ artifact not found: {path}. '
            f'Available emotions: angry, happy, sad. '
            f'Available variants: avg4spk, single0017. '
            f'Extract via `scripts/extract_xvec_tau.py`.'
        )
    return str(path)


def emotionize_audio(
    base_audio: str,
    output_path: str,
    *,
    text: str | None = None,
    ref_text: str | None = None,
    emotion: str = DEFAULT_EMOTION,
    tau_variant: str = DEFAULT_TAU_VARIANT,
    alpha: float = DEFAULT_ALPHA,
    tau_dir: str = DEFAULT_TAU_DIR,
    model_path: str = DEFAULT_MODEL_PATH,
    language: str = DEFAULT_LANGUAGE,
    asr_language: str = DEFAULT_ASR_LANGUAGE,
    target_sr: int = DEFAULT_TARGET_SR,
    headroom_db: float = DEFAULT_HEADROOM_DB,
    ref_start_s: float = 0.0,
    ref_duration_s: float | None = None,
    preprocessed_path: str | None = None,
    tts: Any | None = None,
    max_new_tokens: int = 2048,
) -> EmotionizeResult:
    """Emotion-shift `base_audio` and write the result to `output_path`.

    `text` (what to *say*) and `ref_text` (transcript of the *reference audio*)
    are independent:
      - `text=None, ref_text=None`  → both auto-transcribed (clone same words).
      - `text="...", ref_text=None` → say new text; ref transcript auto-transcribed.
      - `text="...", ref_text="..."`→ both explicit (skips Whisper entirely).

    Args:
        base_audio: any ffmpeg-readable file (wav, opus, mp3, m4a, …).
        output_path: where to write the emotionized 24 kHz mono WAV.
        text: synth text — what the output will say. None → mirror `ref_text`.
        ref_text: transcript of `base_audio` for the voice-clone prompt.
                  None → auto-transcribe with Whisper (`asr_language`).
        emotion: one of {"angry", "happy", "sad"}.
        tau_variant: one of {"avg4spk", "single0017"} (see §4.2 of session report).
        alpha: τ scaling factor. Best-α for PT-BR angry/avg4spk paired = 2.5.
        tau_dir: directory containing tau_<emo>_<variant>.pt artifacts.
        model_path: Qwen3-TTS checkpoint dir.
        language: Qwen `generate_voice_clone` language flag.
        asr_language: Whisper language for auto-transcription.
        target_sr: target sample rate for the preprocessed reference (24 kHz).
        headroom_db: pre-resample attenuation, dB (-1 dB recommended).
        ref_start_s: ffmpeg `-ss` (seek) into `base_audio` before clipping.
        ref_duration_s: ffmpeg `-t` (duration in s) to clip the reference.
                        Recommended 3–6 s on long calm refs to match the
                        ESD/emoUERJ training distribution and mitigate ICL
                        prosody leakage (cf. docs/deploy_caveats.md §1 axis 5).
                        None → no crop (use full input).
        preprocessed_path: where to cache the 24k mono ref; default: alongside output.
        tts: pre-loaded Qwen3TTSModel to avoid the ~10 s reload (optional).
        max_new_tokens: cap on generated audio tokens.

    Returns:
        EmotionizeResult with all paths + norms for downstream logging.
    """
    base_path = Path(base_audio)
    if not base_path.exists():
        raise FileNotFoundError(f'base_audio not found: {base_audio}')

    tau_file = resolve_tau_file(emotion, tau_variant=tau_variant, tau_dir=tau_dir)

    if preprocessed_path is None:
        preprocessed_path = str(
            Path(output_path).with_name(f'{Path(output_path).stem}__ref24k.wav')
        )
    preprocess_to_24k_mono(
        str(base_path), preprocessed_path,
        target_sr=target_sr, headroom_db=headroom_db,
        start_s=ref_start_s, duration_s=ref_duration_s,
    )

    if ref_text is None:
        ref_text = transcribe_ref(preprocessed_path, language=asr_language)
    if not ref_text.strip():
        raise ValueError(
            'Empty reference text (transcription returned ""). '
            'Provide `ref_text=...` explicitly.'
        )
    synth_text = text if (text is not None and text.strip()) else ref_text

    tts = tts if tts is not None else load_tts(model_path)

    art = load_tau_artifact(tau_file)
    tau = art['tau']
    base_xvec = extract_xvec(tts, preprocessed_path)
    tau_dev = match_shape(tau, base_xvec)
    hybrid_xvec = base_xvec + alpha * tau_dev

    wav, sr = synthesize_with_xvec(
        tts, synth_text, preprocessed_path, ref_text, hybrid_xvec,
        language=language, max_new_tokens=max_new_tokens,
    )
    if wav is None:
        raise RuntimeError('Synthesis returned no audio (empty wavs list).')

    duration_s = save_wav(wav, sr, output_path)

    return EmotionizeResult(
        output_path=output_path,
        preprocessed_ref_path=preprocessed_path,
        ref_text=ref_text,
        synth_text=synth_text,
        emotion=emotion,
        tau_variant=tau_variant,
        tau_file=tau_file,
        alpha=float(alpha),
        tau_norm=_f32_norm(tau),
        base_xvec_norm=_f32_norm(base_xvec),
        hybrid_xvec_norm=_f32_norm(hybrid_xvec),
        duration_s=float(duration_s),
        cos_base_tau=cos_flat(base_xvec, tau_dev),
        cos_base_hybrid=cos_flat(base_xvec, hybrid_xvec),
    )


def emotionize_many(
    base_audio: str,
    output_dir: str,
    alphas: list[float],
    *,
    emotion: str = DEFAULT_EMOTION,
    tau_variant: str = DEFAULT_TAU_VARIANT,
    text: str | None = None,
    ref_text: str | None = None,
    tau_dir: str = DEFAULT_TAU_DIR,
    model_path: str = DEFAULT_MODEL_PATH,
    language: str = DEFAULT_LANGUAGE,
    asr_language: str = DEFAULT_ASR_LANGUAGE,
    target_sr: int = DEFAULT_TARGET_SR,
    headroom_db: float = DEFAULT_HEADROOM_DB,
    ref_start_s: float = 0.0,
    ref_duration_s: float | None = None,
    max_new_tokens: int = 2048,
) -> list[EmotionizeResult]:
    """Sweep α for a single base audio (useful for picking operating point)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tts = load_tts(model_path)
    preprocessed_path = str(out_dir / f'{Path(base_audio).stem}__ref24k.wav')
    preprocess_to_24k_mono(
        base_audio, preprocessed_path,
        target_sr=target_sr, headroom_db=headroom_db,
        start_s=ref_start_s, duration_s=ref_duration_s,
    )
    if ref_text is None:
        ref_text = transcribe_ref(preprocessed_path, language=asr_language)

    results: list[EmotionizeResult] = []
    for alpha in alphas:
        out_path = str(out_dir / f'{emotion}_alpha_{alpha:.2f}.wav')
        with torch.no_grad():
            r = emotionize_audio(
                base_audio=base_audio,
                output_path=out_path,
                text=text,
                ref_text=ref_text,
                emotion=emotion,
                tau_variant=tau_variant,
                alpha=alpha,
                tau_dir=tau_dir,
                model_path=model_path,
                language=language,
                asr_language=asr_language,
                target_sr=target_sr,
                headroom_db=headroom_db,
                ref_start_s=ref_start_s,
                ref_duration_s=ref_duration_s,
                preprocessed_path=preprocessed_path,
                tts=tts,
                max_new_tokens=max_new_tokens,
            )
        results.append(r)
    return results
