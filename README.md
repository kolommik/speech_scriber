# speech_scriber

## Описание

Проект `speech_scriber` предназначен для распознавания аудиозаписей диалогов из mp3 файлов и проведения диаризации. Для распознавания речи используется библиотека `openai-whisper`, а для диаризации - `pyannote-audio`. Веб-интерфейс реализован с использованием `streamlit`.

## Установка

### Шаг 1: Установка Poetry

Для управления зависимостями и виртуальной средой используется `Poetry`. Установите его, следуя официальной документации: [Poetry Installation](https://python-poetry.org/docs/#installation).

### Шаг 2: Клонирование репозитория

Клонируйте репозиторий проекта на ваш локальный компьютер:

```sh
git clone https://github.com/kolommik/speech_scriber.git
cd speech_scriber
```

### Шаг 3: Установка зависимостей

Установите все зависимости, указанные в pyproject.toml, с помощью команды:

```sh
poetry install
```

### Шаг 4: Установка GPU версии PyTorch

По умолчанию Poetry установит версию torch для CPU. Для использования GPU необходимо вручную установить соответствующую версию torch.

```sh
nvcc --version
```

Посмотреть статус GPU устройств

```sh
nvidia-smi
```

Выберите команду в зависимости от вашей видеокарты:

#### Для CUDA 11.7

```sh
poetry run pip install torch==2.3.1+cu117 torchvision==0.18.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Для CUDA 11.8

```sh
poetry run pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Для CUDA 12.1

```sh
poetry run pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Для других версий CUDA или ROCm

Выбираем с сайта [https://pytorch.org/get-started/locally/] конфигурацию для установки PyTorch.

#### Проверьте что GPU версия установлена

```sh
poetry run python app\check_cuda.py
```

Если версия CUDA установлена, то выдаст сообщение со списком CUDA видеокарт.

### Запуск

Для запуска веб-интерфейса выполните следующую команду:

```sh
poetry run streamlit run app/main.py
```
