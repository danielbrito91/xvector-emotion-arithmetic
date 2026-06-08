"""Deploy-time CLI: emotion-shift any audio file via x-vector arithmetic.

Wraps `src.emotionize` (single α + α-sweep). Whisper PT auto-transcribes the
reference audio; `--text` overrides what the output *says* (clone the voice but
say something new); `--ref-text` overrides the reference transcript itself
(skips Whisper entirely if you know it).

Examples:

    # Clone same words as zap.opus, but angry (recommended deploy defaults):
    PYTHONPATH=. uv run python scripts/deploy/emotionize_audio.py \\
        --input data/zap.opus --output data/angry_zap.wav

    # Same voice (zap.opus), but say something new — angrily:
    PYTHONPATH=. uv run python scripts/deploy/emotionize_audio.py \\
        --input data/zap.opus --output data/angry_new.wav \\
        --text "Eu não acredito que você fez isso de novo!"

    # α-sweep (operating-point search):
    PYTHONPATH=. uv run python scripts/deploy/emotionize_audio.py \\
        --input data/zap.opus --output-dir data/experiments/zap_sweep \\
        --alpha 0.0 --alpha 1.0 --alpha 1.5 --alpha 2.0 --alpha 2.5
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Annotated

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import typer  # noqa: E402

from src.emotionize import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_ASR_LANGUAGE,
    DEFAULT_HEADROOM_DB,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL_PATH,
    DEFAULT_TARGET_SR,
    DEFAULT_TAU_DIR,
    emotionize_audio,
    emotionize_many,
)


class Emotion(str, Enum):
    angry = 'angry'
    happy = 'happy'
    sad = 'sad'


class TauVariant(str, Enum):
    avg4spk = 'avg4spk'
    single0017 = 'single0017'


app = typer.Typer(
    add_completion=False,
    help='Emotion-shift any audio via x-vector arithmetic (Qwen3-TTS + τ).',
    rich_markup_mode='rich',
)


@app.command()
def main(
    input: Annotated[str, typer.Option(
        '--input', '-i', help='Base audio (wav/opus/mp3/m4a/…).',
    )],
    output: Annotated[str | None, typer.Option(
        '--output', '-o', help='Output wav (single-α mode).',
    )] = None,
    output_dir: Annotated[str | None, typer.Option(
        '--output-dir', help='Output dir (α-sweep mode; one wav per α).',
    )] = None,
    alpha: Annotated[list[float], typer.Option(
        '--alpha', '-a',
        help='τ scaling factor. Pass multiple --alpha for sweep mode.',
    )] = [DEFAULT_ALPHA],
    emotion: Annotated[Emotion, typer.Option(
        '--emotion', '-e', help='Target emotion.',
    )] = Emotion.angry,
    tau_variant: Annotated[TauVariant, typer.Option(
        '--tau-variant', help='τ artifact variant (see §4.2 of session report).',
    )] = TauVariant.avg4spk,
    text: Annotated[str | None, typer.Option(
        '--text', '-t',
        help='What the output should SAY. Defaults to ref transcript (same words).',
    )] = None,
    ref_text: Annotated[str | None, typer.Option(
        '--ref-text',
        help='Transcript of the input audio. Defaults to Whisper auto-transcription.',
    )] = None,
    asr_language: Annotated[str, typer.Option(
        '--asr-language', help='Whisper language for auto-transcription.',
    )] = DEFAULT_ASR_LANGUAGE,
    language: Annotated[str, typer.Option(
        '--language', help='Qwen voice-clone language flag.',
    )] = DEFAULT_LANGUAGE,
    ref_start: Annotated[float, typer.Option(
        '--ref-start',
        help='ffmpeg -ss (s) — start time to crop the reference audio.',
    )] = 0.0,
    ref_duration: Annotated[float | None, typer.Option(
        '--ref-duration',
        help='ffmpeg -t (s) — duration to crop. 3–6 s recommended on '
             'long calm refs to match ESD/emoUERJ distribution. None = full input.',
    )] = None,
    tau_dir: Annotated[str, typer.Option('--tau-dir')] = DEFAULT_TAU_DIR,
    model_path: Annotated[str, typer.Option('--model-path')] = DEFAULT_MODEL_PATH,
    target_sr: Annotated[int, typer.Option('--target-sr')] = DEFAULT_TARGET_SR,
    headroom_db: Annotated[float, typer.Option('--headroom-db')] = DEFAULT_HEADROOM_DB,
    max_new_tokens: Annotated[int, typer.Option('--max-new-tokens')] = 2048,
) -> None:
    """Emotion-shift `--input` and write to `--output` (single α) or `--output-dir` (sweep)."""
    is_sweep = len(alpha) > 1 or output_dir is not None
    if is_sweep:
        if not output_dir:
            raise typer.BadParameter('--output-dir is required for α-sweep.')
        results = emotionize_many(
            base_audio=input,
            output_dir=output_dir,
            alphas=alpha,
            emotion=emotion.value,
            tau_variant=tau_variant.value,
            text=text,
            ref_text=ref_text,
            tau_dir=tau_dir,
            model_path=model_path,
            language=language,
            asr_language=asr_language,
            target_sr=target_sr,
            headroom_db=headroom_db,
            ref_start_s=ref_start,
            ref_duration_s=ref_duration,
            max_new_tokens=max_new_tokens,
        )
        typer.echo(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False))
        return

    if not output:
        raise typer.BadParameter('--output is required for single-α mode.')
    result = emotionize_audio(
        base_audio=input,
        output_path=output,
        text=text,
        ref_text=ref_text,
        emotion=emotion.value,
        tau_variant=tau_variant.value,
        alpha=alpha[0],
        tau_dir=tau_dir,
        model_path=model_path,
        language=language,
        asr_language=asr_language,
        target_sr=target_sr,
        headroom_db=headroom_db,
        ref_start_s=ref_start,
        ref_duration_s=ref_duration,
        max_new_tokens=max_new_tokens,
    )
    typer.echo(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    app()
