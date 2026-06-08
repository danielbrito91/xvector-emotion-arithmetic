import argparse
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.metrics.asr import MODEL_NAME, transcribe

CANONICAL_SENTENCES: list[str] = [
    'Não importa quem está certo.',
    'Você perde tempo demais com a Internet.',
    'A garrafa está na geladeira.',
    'Eu estou me sentindo doente hoje.',
    'Eu estou um pouco atrasado.',
    'Nos fins de semana, eu sempre ia para a casa dele.',
    'Nos fins de semana, eu sempre ia para a casa dela.',
    'De quem são essas malas que estão debaixo da mesa?',
    'Ele volta na quarta-feira.',
    'Já chega! Eu vou tomar um banho e ir para a cama.',
    'Você poderia arrumar a mesa, por favor?',
]


def _normalize(text: str) -> str:
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


_NORMALIZED_CANDIDATES: list[tuple[str, str]] = [
    (s, _normalize(s)) for s in CANONICAL_SENTENCES
]


def match_canonical(hypothesis: str) -> tuple[str, float]:
    """Return (canonical_sentence, similarity_ratio) best matching the hypothesis."""
    norm_hyp = _normalize(hypothesis)
    if not norm_hyp:
        return '', 0.0
    best_sentence = ''
    best_score = -1.0
    for sentence, norm_sentence in _NORMALIZED_CANDIDATES:
        score = SequenceMatcher(None, norm_hyp, norm_sentence).ratio()
        if score > best_score:
            best_score = score
            best_sentence = sentence
    return best_sentence, best_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run Whisper-large-v3 ASR over the emoUERJ dataset.'
    )
    parser.add_argument(
        '--input_csv',
        type=str,
        default='data/emouerj/emo_uerj.csv',
        help='Path to the input emo_uerj.csv file.',
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='data/emouerj/emo_uerj_transcripts.csv',
        help='Path where the transcripts CSV will be written.',
    )
    parser.add_argument(
        '--language',
        type=str,
        default='portuguese',
        help='Language hint for Whisper (e.g. "portuguese", "english").',
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default=MODEL_NAME,
        help='Whisper model name on Hugging Face Hub.',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip rows whose file_name is already present in the output CSV.',
    )
    parser.add_argument(
        '--no_canonical_match',
        action='store_true',
        help='Disable matching Whisper output to the known emoUERJ sentence list.',
    )
    parser.add_argument(
        '--low_score_threshold',
        type=float,
        default=0.6,
        help='Warn when the best canonical match has similarity below this value.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    already_done: set[str] = set()
    if args.resume and output_path.exists():
        done_df = pd.read_csv(output_path)
        already_done = set(done_df['file_name'].astype(str).tolist())
        print(f'Resuming: {len(already_done)} rows already transcribed.')

    write_header = not (args.resume and output_path.exists())

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Transcribing'):
        file_name = str(row['file_name'])
        if file_name in already_done:
            continue

        audio_path = str(row['file_path'])
        try:
            text = transcribe(
                audio_path,
                language=args.language,
                model_name=args.model_name,
            )
        except Exception as e:  # noqa: BLE001
            print(f'[error] {file_name}: {e}')
            text = ''

        out_row = row.to_dict()
        out_row['whisper_transcript'] = text

        if args.no_canonical_match or not text:
            out_row['transcript'] = text
            out_row['match_score'] = ''
        else:
            canonical, score = match_canonical(text)
            out_row['transcript'] = canonical
            out_row['match_score'] = round(score, 4)
            if score < args.low_score_threshold:
                print(
                    f'[low-match {score:.2f}] {file_name}: '
                    f'whisper="{text}" -> "{canonical}"'
                )

        pd.DataFrame([out_row]).to_csv(
            output_path,
            mode='a',
            header=write_header,
            index=False,
        )
        write_header = False

    print(f'Done. Wrote transcripts to {output_path}')


if __name__ == '__main__':
    main()
