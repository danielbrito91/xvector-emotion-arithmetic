"""
Resample ESD wav files from 16kHz to 24kHz.

Usage:
    python resample_esd.py \
        --esd_path "/home/daniel/data/external/Emotional Speech Dataset (ESD)/Emotion Speech Dataset" \
        --output_dir /home/daniel/data/processed/esd_24k \
        --speakers 0011 \
        --emotions Neutral Angry \
        --target_sr 24000
"""

import argparse
import os
from pathlib import Path

import librosa
import soundfile as sf


def resample_speaker_emotion(
    esd_path: str,
    output_dir: str,
    speaker: str,
    emotion: str,
    target_sr: int,
):
    src_dir = os.path.join(esd_path, speaker, emotion)
    dst_dir = os.path.join(output_dir, speaker, emotion)
    os.makedirs(dst_dir, exist_ok=True)

    wav_files = sorted(Path(src_dir).glob('*.wav'))
    if not wav_files:
        print(f'  {speaker}/{emotion}: no wav files found')
        return 0

    for wav_path in wav_files:
        dst_path = os.path.join(dst_dir, wav_path.name)
        audio, sr = librosa.load(str(wav_path), sr=target_sr, mono=True)
        sf.write(dst_path, audio, target_sr, subtype='PCM_16')

    print(f'  {speaker}/{emotion}: {len(wav_files)} files → {dst_dir}')
    return len(wav_files)


def copy_transcript(esd_path: str, output_dir: str, speaker: str):
    """Copy transcript file unchanged (it has no audio, just text)."""
    import shutil
    src = os.path.join(esd_path, speaker, f'{speaker}.txt')
    dst_dir = os.path.join(output_dir, speaker)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, dst_dir)


def main():
    parser = argparse.ArgumentParser(description='Resample ESD to target sample rate')
    parser.add_argument('--esd_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--speakers', nargs='+', default=['0011'])
    parser.add_argument(
        '--emotions',
        nargs='+',
        default=['Neutral', 'Angry', 'Happy', 'Sad', 'Surprise'],
    )
    parser.add_argument('--target_sr', type=int, default=24000)
    args = parser.parse_args()

    print(f'Resampling ESD to {args.target_sr}Hz → {args.output_dir}\n')

    total = 0
    for spk in args.speakers:
        copy_transcript(args.esd_path, args.output_dir, spk)
        for emo in args.emotions:
            total += resample_speaker_emotion(
                args.esd_path, args.output_dir, spk, emo, args.target_sr
            )

    print(f'\nDone. {total} files resampled.')
    print(f'\nNext: regenerate JSONLs pointing to {args.output_dir}')
    print(f'  uv run task prepare_esd_24k')


if __name__ == '__main__':
    main()
