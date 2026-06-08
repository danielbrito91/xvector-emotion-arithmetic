from functools import lru_cache

import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

MODEL_NAME = 'microsoft/wavlm-base-plus-sv'


@lru_cache(maxsize=1)
def _get_wavlm(model_name: str = MODEL_NAME):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WavLMForXVector.from_pretrained(model_name).to(device).eval()
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    return model, extractor, device


def get_speaker_embedding(audio_path: str, model_name: str = MODEL_NAME) -> torch.Tensor:
    model, extractor, device = _get_wavlm(model_name)
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    inputs = extractor(
        waveform.squeeze().numpy(), sampling_rate=16000, return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model(**inputs).embeddings
    return torch.nn.functional.normalize(emb, dim=-1).detach().cpu()
