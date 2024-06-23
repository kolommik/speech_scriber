import os
import streamlit as st
from backend.whisper_processing import process_whisper
from backend.diarization_processing import process_diarization
from backend.merge_results import merge_results

st.title("Speech Scriber")

uploaded_file = st.file_uploader("Загрузите MP3 файл", type=["mp3"])

if uploaded_file is not None:
    # Создание директории 'data', если она не существует
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    audio_path = os.path.join(data_dir, uploaded_file.name)
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Файл загружен. Начинаем обработку...")

    # Параллельная обработка
    try:
        with st.spinner("Обработка через Whisper и Pyannote..."):
            whisper_df = process_whisper(audio_path)
            diarization_df = process_diarization(audio_path)

        st.write("Обработка завершена. Объединение результатов...")

        # Объединение результатов
        result_df = merge_results(whisper_df, diarization_df)

        st.write("Результаты:")
        st.dataframe(result_df)

        # Сохранение результатов
        result_csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Скачать результаты в CSV",
            data=result_csv,
            file_name="results.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Произошла ошибка: {e}")
