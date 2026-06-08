import json

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from src.metrics.emotion import get_emotion
from src.metrics.speaker import get_speaker_embedding

TOKENIZER_PATH = 'Qwen/Qwen3-TTS-Tokenizer-12Hz'
BASE_MODEL_PATH = 'Qwen/Qwen3-TTS-12Hz-1.7B-Base'
CUSTOM_VOICE_PATH = 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'
LANGUAGE = 'Portuguese'
TEMPERATURA = 0.9

REFERENCES = {
    'neutral': {
        'audio': 'data/ref/dani-neutro.wav',
        'text': (
            'Esse áudio serve como referência para um modelo de síntese '
            'de fala. Estou falando agora com uma voz neutra, esperando '
            'aqui que eu consiga atingir uma marca de dez segundos de fala.'
        ),
    },
    'angry': {
        'audio': 'data/ref/eu-to-exausta.wav',
        'text': 'Eu tô exausta!',
    },
}

TEXT_TO_SYNTHESIZE = 'Estou muito brabo com essa situação toda!'


def experiment_emotion(model: Qwen3TTSModel) -> dict:

    final_results = {
        'speaker_similarity': {},
        'emotion_similarity': {},
    }

    for emotion, ref in REFERENCES.items():
        ref_audio = ref['audio']
        ref_text = ref['text']

        # Generate speech with x_vector_only_true
        wavs, sr = model.generate_voice_clone(
            text=TEXT_TO_SYNTHESIZE,
            language=LANGUAGE,
            ref_audio=ref_audio,
            x_vector_only_mode=True,
            max_new_tokens=2048,
            temperature=TEMPERATURA,
        )
        # Save output
        output_path = f'data/outputs/exp01_ref-{emotion}_vector_only.wav'
        sf.write(output_path, wavs[0], sr)
        print(f'Saved {emotion} audio to {output_path}')

        # Compara spk sim
        spk_sim = compare_speaker_similarity(ref_audio, output_path)
        emo_sim = compare_emotions(ref_audio, output_path)
        final_results['speaker_similarity'][f'{emotion}_x_vector_only'] = spk_sim
        final_results['emotion_similarity'][f'{emotion}_x_vector_only'] = emo_sim

        # Generate speech with ref_text
        wavs, sr = model.generate_voice_clone(
            text=TEXT_TO_SYNTHESIZE,
            language=LANGUAGE,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=False,
            max_new_tokens=2048,
            temperature=TEMPERATURA,
        )
        # Save output
        output_path = f'data/outputs/exp01_ref-{emotion}_with_text.wav'
        sf.write(output_path, wavs[0], sr)
        print(f'Saved {emotion} audio to {output_path}')

        spk_sim = compare_speaker_similarity(ref_audio, output_path)
        emo_sim = compare_emotions(ref_audio, output_path)
        final_results['speaker_similarity'][f'{emotion}_with_text'] = spk_sim
        final_results['emotion_similarity'][f'{emotion}_with_text'] = emo_sim

    return final_results


def compare_emotions(ref_audio, synth_audio) -> float:
    emb_ref = torch.tensor(get_emotion(ref_audio)['embedding'])
    emb_gen = torch.tensor(get_emotion(synth_audio)['embedding'])
    sim = torch.nn.functional.cosine_similarity(emb_ref, emb_gen, dim=0).item()
    return sim


def compare_speaker_similarity(ref_audio: str, synth_audio: str) -> float:
    emb_ref = get_speaker_embedding(ref_audio)
    emb_gen = get_speaker_embedding(synth_audio)
    sim = torch.nn.functional.cosine_similarity(emb_ref, emb_gen).item()
    return sim


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Qwen3TTSModel.from_pretrained(
        BASE_MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    )

    results = experiment_emotion(model)
    with open('data/outputs/exp01_results.json', 'w') as f:
        json.dump(results, f, indent=4)
