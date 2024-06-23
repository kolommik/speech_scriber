import os
from dotenv import load_dotenv, find_dotenv
from pyannote.audio import Pipeline
import pandas as pd
import torch


def process_diarization(audio_path):
    # Ваш токен доступа Hugging Face
    load_dotenv(find_dotenv())
    AUTH_TOKEN = os.getenv("HUGGINGFACE_AUTH_TOKEN")

    if not AUTH_TOKEN:
        raise ValueError(
            "Hugging Face auth token not found. Please set it in the .env file."
        )

    # Инициализация пайплайна диаризации
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=AUTH_TOKEN
    )

    # send pipeline to GPU (when available)
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Применение диаризации к аудиофайлу
    diarization = pipeline(audio_path)

    # Обработка результатов диаризации
    diarization_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_segments.append(
            {"start": turn.start, "end": turn.end, "speaker": speaker}
        )

    # Преобразование в DataFrame
    diarization_df = pd.DataFrame(diarization_segments)
    return diarization_df
