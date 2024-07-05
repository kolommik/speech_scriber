import os
import time
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from backend.whisper_processing import process_whisper
from backend.diarization_processing import process_diarization
from backend.merge_results import merge_results
from logging_config import logger
import torch

# Инициализация сессионного состояния
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "result_df" not in st.session_state:
    st.session_state.result_df = None
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "processing_times" not in st.session_state:
    st.session_state.processing_times = {}

st.title("Speech Scriber")

# Проверка доступности CUDA
cuda_available = torch.cuda.is_available()
if cuda_available:
    cuda_info = f"CUDA is available. Version: {torch.version.cuda}, Device: {torch.cuda.get_device_name(0)}"
else:
    cuda_info = "CUDA is not available."

# Логирование информации о CUDA
logger.info(cuda_info)

# Отображение информации о CUDA в интерфейсе
st.write(cuda_info)

uploaded_file = st.file_uploader("Загрузите MP3 файл", type=["mp3"])

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    # Создание директории 'data', если она не существует
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    audio_path = os.path.join(data_dir, uploaded_file.name)
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state.audio_path = audio_path
    st.write("Файл загружен. Начинаем обработку...")
    logger.info(f"Файл {uploaded_file.name} загружен и сохранен в {audio_path}")

    # Параллельная обработка
    try:
        with st.spinner("Обработка через Whisper и Pyannote..."):
            start_time = time.time()
            with ThreadPoolExecutor() as executor:
                whisper_future = executor.submit(process_whisper, audio_path)
                diarization_future = executor.submit(process_diarization, audio_path)

                logger.info("Начало обработки через Whisper.")
                whisper_df = whisper_future.result()
                logger.info("Обработка через Whisper завершена.")
                whisper_time = time.time() - start_time
                st.session_state.processing_times["Whisper"] = whisper_time

                diarization_start_time = time.time()
                logger.info("Начало обработки через Pyannote.")
                diarization_df = diarization_future.result()
                logger.info("Обработка через Pyannote завершена.")
                diarization_time = time.time() - diarization_start_time
                st.session_state.processing_times["Diarization"] = diarization_time

        st.write("Обработка завершена. Объединение результатов...")
        logger.info("Обработка завершена. Объединение результатов...")

        # Объединение результатов
        start_time = time.time()
        result_df = merge_results(whisper_df, diarization_df)
        merge_time = time.time() - start_time
        st.session_state.processing_times["Merge"] = merge_time
        st.session_state.result_df = result_df
        logger.info("Результаты объединены.")

    except Exception as e:
        st.error(f"Произошла ошибка: {e}")
        logger.error(f"Произошла ошибка: {e}", exc_info=True)

if st.session_state.result_df is not None:
    st.write("Результаты:")
    st.dataframe(st.session_state.result_df)

    # Сохранение результатов
    result_csv = st.session_state.result_df.to_csv(index=False).encode("utf-8")
    download_button = st.download_button(
        label="Скачать результаты в CSV",
        data=result_csv,
        file_name="results.csv",
        mime="text/csv",
    )
    logger.info("Результаты успешно объединены и сохранены.")

    # Удаление загруженного файла после скачивания CSV
    if download_button:
        os.remove(st.session_state.audio_path)
        logger.info(
            f"Файл {st.session_state.audio_path} был удален после скачивания CSV."
        )
        st.session_state.uploaded_file = None
        st.session_state.result_df = None
        st.session_state.audio_path = None

# Отображение времени обработки
if "processing_times" in st.session_state and st.session_state.processing_times:
    st.write("Время обработки:")
    for process, duration in st.session_state.processing_times.items():
        st.write(f"{process}: {duration:.2f} секунд")
