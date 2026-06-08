"""X-vector arithmetic + voice-clone synthesis helpers (model-agnostic side effects)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import librosa
import soundfile as sf
import torch
import torch.nn.functional as F
from qwen_tts import Qwen3TTSModel


def load_tts(model_path: str, device: str | None = None) -> Qwen3TTSModel:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return Qwen3TTSModel.from_pretrained(
        model_path, device_map=device, dtype=torch.bfloat16,
    )


def extract_xvec(tts: Qwen3TTSModel, wav_path: str) -> torch.Tensor:
    spk_sr = tts.model.speaker_encoder_sample_rate
    audio, _ = librosa.load(wav_path, sr=spk_sr, mono=True)
    with torch.no_grad():
        return tts.model.extract_speaker_embedding(audio, sr=spk_sr)


def load_tau_artifact(tau_file: str) -> dict[str, Any]:
    art = torch.load(tau_file, map_location="cpu", weights_only=False)
    required = {"tau", "emotion_centroid", "neutral_centroid", "config", "stats"}
    missing = required - set(art.keys())
    if missing:
        raise ValueError(f"Tau artifact {tau_file} missing keys: {missing}")
    return art


def build_prompt_with_xvec(
    tts: Qwen3TTSModel, ref_audio: str, ref_text: str, custom_xvec: torch.Tensor,
) -> list:
    items = tts.create_voice_clone_prompt(
        ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=False,
    )
    items = deepcopy(items)
    items[0].ref_spk_embedding = custom_xvec
    return items


def synthesize_with_xvec(
    tts: Qwen3TTSModel,
    text: str,
    ref_audio: str,
    ref_text: str,
    xvec: torch.Tensor,
    language: str = "Auto",
    max_new_tokens: int = 2048,
) -> tuple[Any, int] | tuple[None, None]:
    prompt = build_prompt_with_xvec(tts, ref_audio, ref_text, xvec)
    wavs, sr = tts.generate_voice_clone(
        text=text, language=language,
        voice_clone_prompt=prompt, max_new_tokens=max_new_tokens,
    )
    if not wavs:
        return None, None
    return wavs[0], sr


def save_wav(wav, sr: int, out_path: str, peak_target: float = 0.97) -> float:
    """Save wav to disk with peak-norm guard.

    At high α the vocoder can output samples > 1.0 (out-of-distribution x-vec
    push). Hard-clip on PCM write produces audible clicks. We scale down by a
    constant factor if peak exceeds `peak_target`, preserving relative
    amplitudes (encoders downstream are amplitude-invariant up to ~0dB).
    """
    import numpy as np

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(wav, dtype=np.float32)
    if a.size:
        peak = float(np.max(np.abs(a)))
        if peak > peak_target:
            a = a * (peak_target / peak)
    sf.write(out_path, a, sr)
    return len(a) / sr


def match_shape(target: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Align a 1-D tau to ref's shape/device/dtype (xvec ops are shape-sensitive)."""
    t = target.to(device=ref.device, dtype=ref.dtype)
    return t.view_as(ref) if t.shape != ref.shape else t


def cos_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().flatten().float().cpu()
    bf = b.detach().flatten().float().cpu()
    return F.cosine_similarity(af, bf, dim=0).item()


def compute_metrics_target_gt(
    tts: Qwen3TTSModel,
    synth_wav: str,
    gt_emo_audio: str,
    ref_neutral_audio: str,
    target_text: str | None = None,
    wer_language: str = "en",
) -> dict[str, float | str]:
    """Phase B v2 (single-shot, no pre-cache). Kept for back-compat / single-cell script."""
    from src.metrics.emotion import get_emotion

    emo_synth = torch.tensor(get_emotion(synth_wav)["embedding"])
    emo_gt = torch.tensor(get_emotion(gt_emo_audio)["embedding"])
    xvec_synth = extract_xvec(tts, synth_wav)
    xvec_gt_emo = extract_xvec(tts, gt_emo_audio)
    xvec_neutral = extract_xvec(tts, ref_neutral_audio)

    out: dict[str, float | str] = {
        "emo_cos_sim_gt": cos_flat(emo_synth, emo_gt),
        "xvec_cos_sim_gt": cos_flat(xvec_synth, xvec_gt_emo),
        "spk_cos_sim_neutral_qwen": cos_flat(xvec_synth, xvec_neutral),
    }
    if target_text is not None:
        from src.metrics.asr import compute_wer_norm, compute_wer_raw, transcribe
        hyp = transcribe(synth_wav, language=wer_language)
        out["transcription"] = hyp
        out["wer_raw"] = compute_wer_raw(target_text, hyp)
        out["wer_norm"] = compute_wer_norm(target_text, hyp, language=wer_language)
    return out


def precompute_refs(
    tts: Qwen3TTSModel,
    ref_neutral_audio: str,
    gt_emo_audio: str,
) -> dict[str, torch.Tensor]:
    """Pre-cache GT/neutral embeddings once per (target, emo, sentence) — reused across α."""
    from src.metrics.emotion import get_emotion
    from src.metrics.speaker import get_speaker_embedding

    return {
        "emo_gt": torch.tensor(get_emotion(gt_emo_audio)["embedding"]),
        "qwen_xvec_gt": extract_xvec(tts, gt_emo_audio).detach().cpu(),
        "qwen_xvec_neutral": extract_xvec(tts, ref_neutral_audio).detach().cpu(),
        "wavlm_xvec_gt": get_speaker_embedding(gt_emo_audio),
        "wavlm_xvec_neutral": get_speaker_embedding(ref_neutral_audio),
    }


def compute_metrics_v3(
    tts: Qwen3TTSModel,
    synth_wav: str,
    refs: dict[str, torch.Tensor],
    target_text: str,
    wer_language: str = "en",
    include_utmos: bool = True,
) -> dict[str, float | str]:
    """v3 metric set with pre-cached refs (`refs` from precompute_refs).

    Reported per synth wav:
      - emo_cos_sim_gt              (emotion2vec, primary emotion-transfer signal — EmoSphere++ EECS)
      - xvec_cos_sim_gt             (Qwen3-TTS ECAPA — falsification in τ space)
      - spk_cos_sim_neutral_qwen    (Qwen3-TTS ECAPA — identity in τ space, circular)
      - spk_cos_sim_neutral_wavlm   (WavLM-base-plus-sv — independent SECS, EmoSphere++ SECS_W-like)
      - xvec_cos_sim_gt_wavlm       (WavLM — emo-side identity in independent space)
      - wer_raw, wer_norm           (Whisper-large-v3 + jiwer; norm = whisper EN normalizer)
      - utmos                       (UTMOSv2 predicted MOS, naturalness — single-file mode)
      - transcription
    """
    from src.metrics.asr import compute_wer_norm, compute_wer_raw, transcribe
    from src.metrics.emotion import get_emotion
    from src.metrics.speaker import get_speaker_embedding

    emo_synth = torch.tensor(get_emotion(synth_wav)["embedding"])
    qwen_xvec_synth = extract_xvec(tts, synth_wav).detach().cpu()
    wavlm_xvec_synth = get_speaker_embedding(synth_wav)
    hyp = transcribe(synth_wav, language=wer_language)
    out: dict[str, float | str] = {
        "emo_cos_sim_gt": cos_flat(emo_synth, refs["emo_gt"]),
        "xvec_cos_sim_gt": cos_flat(qwen_xvec_synth, refs["qwen_xvec_gt"]),
        "spk_cos_sim_neutral_qwen": cos_flat(qwen_xvec_synth, refs["qwen_xvec_neutral"]),
        "spk_cos_sim_neutral_wavlm": cos_flat(wavlm_xvec_synth, refs["wavlm_xvec_neutral"]),
        "xvec_cos_sim_gt_wavlm": cos_flat(wavlm_xvec_synth, refs["wavlm_xvec_gt"]),
        "wer_raw": compute_wer_raw(target_text, hyp),
        "wer_norm": compute_wer_norm(target_text, hyp, language=wer_language),
        "transcription": hyp,
    }
    if include_utmos:
        from src.metrics.utmos import score_wav

        out["utmos"] = score_wav(synth_wav)
    return out
