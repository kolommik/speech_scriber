import os
import streamlit as st
from logging_config import logger

# Инициализация сессионного состояния
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

st.title("File Uploader")

uploaded_file = st.file_uploader("Загрузите файл", type=None)

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    # Создание директории 'temp_data', если она не существует
    temp_data_dir = "temp_data"
    if not os.path.exists(temp_data_dir):
        os.makedirs(temp_data_dir)

    file_path = os.path.join(temp_data_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write(f"Файл загружен и сохранен в {file_path}")
    logger.info(f"Файл {uploaded_file.name} загружен и сохранен в {file_path}")
