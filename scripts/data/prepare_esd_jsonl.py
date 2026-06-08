"""
Prepare ESD (Emotional Speech Dataset) into Qwen3-TTS fine-tuning JSONL format.

For task vector experiments, we need per-speaker, per-emotion JSONLs:
  - Speaker X, Neutral  → fine-tune → checkpoint_neutral
  - Speaker X, Angry    → fine-tune → checkpoint_angry
  - τ_angry = checkpoint_angry - checkpoint_neutral

The Qwen3-TTS raw JSONL format requires:
  - audio: path to target utterance wav
  - text: transcript
  - ref_audio: path to reference speaker audio (same for all samples)

Then run prepare_data.py to add audio_codes before fine-tuning.

Usage:
    python prepare_esd_jsonl.py \
        --esd_path \
            "/home/daniel/data/external/Emotional Speech Dataset (ESD)/Emotion Speech Dataset" \
        --output_dir /home/daniel/data/processed/esd_qwen3 \
        --speakers 0011 \
        --emotions Neutral Angry Happy Sad Surprise
"""

import argparse
import json
import os
from pathlib import Path


def parse_transcript_file(transcript_path: str) -> dict[str, tuple[str, str]]:
    """Parse ESD transcript file.

    Format: {speaker}_{utterance_id}\t{text}\t{emotion}
    Returns: dict mapping filename_stem -> (text, emotion)
    """
    transcripts = {}
    with open(transcript_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            utt_id, text, emotion = parts
            transcripts[utt_id] = (text, emotion)
    return transcripts


def get_wav_files(speaker_dir: str, emotion: str) -> list[str]:
    """Get sorted list of wav files for a speaker/emotion combo."""
    emotion_dir = os.path.join(speaker_dir, emotion)
    if not os.path.isdir(emotion_dir):
        return []
    return sorted([
        os.path.join(emotion_dir, f) for f in os.listdir(emotion_dir) if f.endswith('.wav')
    ])


def select_ref_audio(wav_files: list[str], idx: int = 0) -> str:
    """Select a reference audio file.

    Per Qwen docs: 'use the same ref_audio for all samples' for consistency.
    We pick the first file by default — you might want to pick one manually
    that sounds clean and representative.
    """
    return wav_files[idx]


def build_jsonl_entries(
    speaker_id: str,
    emotion: str,
    speaker_dir: str,
    transcripts: dict[str, tuple[str, str]],
    ref_audio: str,
) -> list[dict]:
    """Build JSONL entries for one speaker/emotion pair."""
    wav_files = get_wav_files(speaker_dir, emotion)
    entries = []

    for wav_path in wav_files:
        stem = Path(wav_path).stem  # e.g., "0011_000001"
        if stem not in transcripts:
            print(f'  WARNING: no transcript for {stem}, skipping')
            continue

        text, transcript_emotion = transcripts[stem]

        # Sanity check: transcript emotion should match directory
        if transcript_emotion != emotion:
            print(
                f'  WARNING: {stem} transcript says {transcript_emotion!r} '
                f'but in {emotion!r} dir, skipping'
            )
            continue

        entries.append({
            'audio': wav_path,
            'text': text,
            'ref_audio': ref_audio,
        })

    return entries


def compute_stats(esd_path: str, speakers: list[str], emotions: list[str]):
    """Print a summary table of utterance counts per speaker/emotion."""
    print('\n=== ESD Data Inventory ===\n')
    header = f'{"Speaker":<10}' + ''.join(f'{e:<12}' for e in emotions) + 'Total'
    print(header)
    print('-' * len(header))

    for spk in speakers:
        spk_dir = os.path.join(esd_path, spk)
        if not os.path.isdir(spk_dir):
            print(f'{spk:<10} (not found)')
            continue

        counts = [len(get_wav_files(spk_dir, emo)) for emo in emotions]
        row = f'{spk:<10}' + ''.join(f'{c:<12}' for c in counts) + str(sum(counts))
        print(row)

    print('\n(ESD utterances are typically 2-5s each)')
    print('350 utterances × ~3.5s avg ≈ ~20 min per speaker per emotion')


def main():
    parser = argparse.ArgumentParser(description='Prepare ESD for Qwen3-TTS fine-tuning')
    parser.add_argument(
        '--esd_path',
        type=str,
        required=True,
        help="Path to ESD 'Emotion Speech Dataset' directory",
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for JSONL files',
    )
    parser.add_argument(
        '--speakers',
        nargs='+',
        default=['0011'],
        help='Speaker IDs to process (default: 0011, an English male speaker)',
    )
    parser.add_argument(
        '--emotions',
        nargs='+',
        default=['Neutral', 'Angry', 'Happy', 'Sad', 'Surprise'],
        help='Emotions to process',
    )
    parser.add_argument(
        '--ref_audio_index',
        type=int,
        default=0,
        help='Index of wav file to use as ref_audio (0 = first file)',
    )
    parser.add_argument(
        '--stats_only',
        action='store_true',
        help="Only print data inventory, don't write JSONLs",
    )
    args = parser.parse_args()

    compute_stats(args.esd_path, args.speakers, args.emotions)

    if args.stats_only:
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for spk in args.speakers:
        spk_dir = os.path.join(args.esd_path, spk)
        transcript_path = os.path.join(spk_dir, f'{spk}.txt')

        if not os.path.isfile(transcript_path):
            print(f'\nERROR: transcript not found: {transcript_path}')
            continue

        transcripts = parse_transcript_file(transcript_path)
        print(f'\n{spk}: loaded {len(transcripts)} transcript entries')

        # Use a Neutral utterance as ref_audio (consistent speaker identity)
        neutral_wavs = get_wav_files(spk_dir, 'Neutral')
        if not neutral_wavs:
            print(f'  ERROR: no Neutral wavs found for {spk}')
            continue

        ref_audio = select_ref_audio(neutral_wavs, args.ref_audio_index)
        print(f'  ref_audio: {ref_audio}')

        for emo in args.emotions:
            entries = build_jsonl_entries(spk, emo, spk_dir, transcripts, ref_audio)

            if not entries:
                print(f'  {emo}: 0 entries, skipping')
                continue

            out_file = os.path.join(args.output_dir, f'{spk}_{emo.lower()}_raw.jsonl')
            with open(out_file, 'w', encoding='utf-8') as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            print(f'  {emo}: {len(entries)} entries → {out_file}')

    print(f"""
=== Next Steps ===

1. Run prepare_data.py for each raw JSONL to add audio_codes:

   for f in {args.output_dir}/*_raw.jsonl; do
       out="${{f/_raw.jsonl/_coded.jsonl}}"
       python prepare_data.py \\
           --device cuda:0 \\
           --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \\
           --input_jsonl "$f" \\
           --output_jsonl "$out"
   done

2. Fine-tune separately for each emotion (task vector experiment):

   # Neutral baseline
   uv run python third_party/qwen/sft_12hz.py \\
       --init_model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \\
       --output_model_path output_{args.speakers[0]}_neutral \\
       --train_jsonl {args.output_dir}/{args.speakers[0]}_neutral_coded.jsonl \\
       --batch_size 2 --lr 2e-6 --num_epochs 3 \\
       --speaker_name {args.speakers[0]}_neutral

   # Expressive (e.g., angry)
   uv run python third_party/qwen/sft_12hz.py \\
       --init_model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \\
       --output_model_path output_{args.speakers[0]}_angry \\
       --train_jsonl {args.output_dir}/{args.speakers[0]}_angry_coded.jsonl \\
       --batch_size 2 --lr 2e-6 --num_epochs 3 \\
       --speaker_name {args.speakers[0]}_angry

3. Extract task vector:
   τ_angry = θ_angry - θ_neutral
   θ_expressive = θ_base + α * τ_angry
""")


if __name__ == '__main__':
    main()
