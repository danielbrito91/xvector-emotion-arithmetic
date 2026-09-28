"""Recalcula os agregados cross-corpus a partir de manifestos e escores salvos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cross_corpus_support import analyze_scores, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze_scores(
        read_jsonl(args.run / 'scores.jsonl'),
        read_jsonl(args.run / 'manifest_synthesis.jsonl'),
    )
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
