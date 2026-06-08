from funasr import AutoModel

emotion_model = AutoModel(model='iic/emotion2vec_plus_large')


def get_emotion(audio_path: str) -> dict:
    result = emotion_model.generate(audio_path, extract_embedding=True)
    return {
        'label': result[0]['labels'],
        'scores': result[0]['scores'],
        'embedding': result[0]['feats'],
    }
