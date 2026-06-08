from functools import lru_cache

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

MODEL_NAME = 'openai/whisper-large-v3'


@lru_cache(maxsize=1)
def get_asr_pipeline(
    model_name: str = MODEL_NAME,
    device: str | None = None,
    torch_dtype: torch.dtype | None = None,
):
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if torch_dtype is None:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_name)

    return pipeline(
        'automatic-speech-recognition',
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        chunk_length_s=30,
        return_timestamps=False,
    )


def transcribe(
    audio_path: str,
    language: str | None = None,
    task: str = 'transcribe',
    model_name: str = MODEL_NAME,
) -> str:
    asr = get_asr_pipeline(model_name=model_name)
    generate_kwargs: dict = {'task': task}
    if language is not None:
        generate_kwargs['language'] = language
    result = asr(audio_path, generate_kwargs=generate_kwargs)
    return result['text'].strip()


def _basic_normalize(s: str) -> str:
    import re

    s = s.lower().strip()
    s = re.sub(r'[^\w\s\']', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


@lru_cache(maxsize=1)
def _english_text_normalizer():
    from whisper_normalizer.english import EnglishTextNormalizer

    return EnglishTextNormalizer()


def normalize_en(s: str) -> str:
    return _english_text_normalizer()(s)


def compute_wer_raw(reference: str, hypothesis: str) -> float:
    from jiwer import wer

    ref, hyp = _basic_normalize(reference), _basic_normalize(hypothesis)
    return float('nan') if not ref else wer(ref, hyp)


def compute_wer_norm(reference: str, hypothesis: str, language: str = 'en') -> float:
    from jiwer import wer

    if language == 'en':
        ref, hyp = normalize_en(reference), normalize_en(hypothesis)
    else:
        from whisper_normalizer.basic import BasicTextNormalizer

        n = BasicTextNormalizer()
        ref, hyp = n(reference), n(hypothesis)
    return float('nan') if not ref else wer(ref, hyp)
