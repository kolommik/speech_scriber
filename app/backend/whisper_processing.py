import whisper
import pandas as pd
import torch
from logging_config import logger


def process_whisper(audio_path):
    # Загрузка модели Whisper
    model = whisper.load_model("large-v3", "cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"Whisper model loaded and moved to {device}")

    # Загрузка и обработка аудиофайла с помощью Whisper
    audio = whisper.load_audio(audio_path)
    result = model.transcribe(audio, language="ru")

    # Обработка результатов Whisper
    whisper_segments = []
    for segment in result["segments"]:
        whisper_segments.append(
            {"start": segment["start"], "end": segment["end"], "text": segment["text"]}
        )

    # Преобразование в DataFrame
    whisper_df = pd.DataFrame(whisper_segments)
    return whisper_df
