# local-llm

Локальный LLM-сервер для macOS (Apple Silicon) на базе [MLX](https://github.com/ml-explore/mlx) с OpenAI-совместимым API.

Скачивайте MLX-модели с Hugging Face, запускайте их полностью на устройстве и общайтесь через `POST /v1/chat/completions` (включая SSE-стриминг) — без облака и без собственных API-ключей.

> План проекта и заметки по дизайну: [docs/PLAN.md](docs/PLAN.md).

## Требования

- macOS на Apple Silicon (M1/M2/M3/M4)
- Python >= 3.12
- Достаточно свободной unified memory для скачиваемой модели (например, ~2 ГБ для модели 0.6B bf16)

## Быстрый старт

```bash
# 1. Создать .venv и установить пакет (editable) + dev-зависимости
make setup

# 2. (Опционально) Токен Hugging Face — нужен только для gated/приватных моделей
export HF_TOKEN=hf_...          # либо положите его в .env (см. Конфигурация)

# 3. Скачать MLX-модель в ./models
make download MODEL=mlx-community/Qwen3-0.6B-MLX-bf16

# 4. Указать серверу, какую модель загружать
export LLM_MODEL=Qwen3-0.6B-MLX-bf16

# 5. Запустить OpenAI-совместимый сервер
make run
```

Сервер слушает `http://127.0.0.1:8000` по умолчанию, загружает модель в память до приёма запросов и печатает стартовый баннер с путём к модели и параметрами генерации.

Небольшая dev/smoke-модель (~1.2 ГБ): `mlx-community/Qwen3-0.6B-MLX-bf16`.

## Конфигурация

Все настройки задаются переменными окружения (или файлом `.env` в корне проекта — он загружается автоматически). Редактировать конфигурационные файлы не нужно.

| Переменная | По умолчанию | Описание |
|---|---|---|
| `LLM_HOST` | `127.0.0.1` | Адрес, на который биндится сервер |
| `LLM_PORT` | `8000` | Порт, на котором слушает сервер |
| `LLM_MODEL` | — (обязательна для `run`) | Локальное имя модели (папка в `LLM_MODELS_DIR`) или HF repo id |
| `LLM_MODELS_DIR` | `./models` | Каталог, куда сохраняются скачанные модели |
| `LLM_MAX_TOKENS` | `512` | Максимальное число генерируемых токенов |
| `LLM_TEMPERATURE` | `0.7` | Температура сэмплирования (0–2) |
| `LLM_TOP_P` | `1.0` | Top-p (nucleus) сэмплирование (0–1) |
| `HF_TOKEN` | — (опционально) | Токен доступа Hugging Face; нужен только для gated/приватных моделей |

**Про `HF_TOKEN`:**

- Опционально — публичные модели скачиваются анонимно и без него.
- Если задан, он передаётся явно в SDK Hugging Face (`snapshot_download(..., token=...)`) при скачивании и листинге.
- Никогда не логируется и не печатается CLI/сервером.
- Рекомендуемые источники: `export HF_TOKEN=hf_...` в шелле или файл `.env` (читается при старте).

Некорректные значения (например, порт или temperature вне диапазона) приводят к быстрому падению на старте с сообщением, в котором названа виноватая переменная.

## Управление моделями

```bash
# Скачать MLX-модель в LLM_MODELS_DIR (идемпотентно)
make download MODEL=mlx-community/Qwen3-0.6B-MLX-bf16

# Список скачанных моделей с размерами и путями
make list

# Удалить скачанную модель по имени (как показывает `list`)
make delete MODEL=Qwen3-0.6B-MLX-bf16
```

Поддерживаются только модели в MLX-формате (`mlx-community/*` и другие MLX-репозитории на Hugging Face).

## Запуск сервера

```bash
export LLM_MODEL=Qwen3-0.6B-MLX-bf16
make run
```

- Биндится на `LLM_HOST:LLM_PORT` (по умолчанию `127.0.0.1:8000`).
- Модель загружается в память **до** того, как сервер начинает принимать запросы.
- Запрос на генерацию обслуживается по одному за раз (unified memory в MLX).
- Ошибки старта (нет модели, неверный порт, нехватка памяти) завершают процесс с ненулевым кодом и подсказкой, что делать.

## API

### `POST /v1/chat/completions`

OpenAI-совместимые chat completions. Тело запроса:

| Поле | Тип | Обязательно | Описание |
|---|---|---|---|
| `model` | string | да | Имя модели (должно совпадать с загруженной) |
| `messages` | array | да | `[{ "role": "system"\|"user"\|"assistant"\|"tool", "content": "..." }, ...]`, минимум один |
| `temperature` | number | нет | 0–2; переопределяет `LLM_TEMPERATURE` для этого запроса |
| `max_tokens` | integer | нет | >= 1; переопределяет `LLM_MAX_TOKENS` |
| `top_p` | number | нет | 0–1; переопределяет `LLM_TOP_P` |
| `stream` | boolean | нет | по умолчанию `false`; `true` — SSE-стриминг |

Неизвестные поля игнорируются. Ответ (без стриминга):

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "Qwen3-0.6B-MLX-bf16",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "..." },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46 }
}
```

### Стриминг (`stream: true`)

Возвращает `Content-Type: text/event-stream`. Каждое событие — `data: {...}\n\n` с объектом `chat.completion.chunk`, содержащим инкрементальный `delta`; поток завершается `data: [DONE]`. Usage (если доступен) передаётся в финальном чанке перед `[DONE]`.

### Ошибки

Ошибки оформлены в формате OpenAI: `{"error": {"message", "type", "code"}}`.

| Статус | Значение |
|---|---|
| `400` | Некорректное тело запроса (провалена валидация) |
| `404` | Запрошенная модель не загружена |
| `500` | Сбой генерации (сервер продолжает работать) |

## Примеры curl

Обычный запрос (без стриминга):

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B-MLX-bf16",
    "messages": [
      { "role": "system", "content": "You are a helpful assistant." },
      { "role": "user", "content": "Say hello in one short sentence." }
    ],
    "max_tokens": 64,
    "temperature": 0.7
  }'
```

Запрос со стримингом (SSE):

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B-MLX-bf16",
    "messages": [
      { "role": "user", "content": "Count from 1 to 5." }
    ],
    "stream": true
  }'
```

(`-N` отключает буферизацию curl, чтобы токены появлялись по мере генерации.)

## Make-цели

| Цель | Описание |
|---|---|
| `make setup` | Создать `.venv` (Python >= 3.12) и установить пакет в editable-режиме с dev-зависимостями |
| `make download MODEL=<repo>` | Скачать MLX-модель с Hugging Face в `LLM_MODELS_DIR` |
| `make list` | Показать скачанные модели с размерами и путями |
| `make delete MODEL=<name>` | Удалить скачанную модель |
| `make run` | Запустить сервер с моделью из `LLM_MODEL` |
| `make test` | Запустить тесты |
| `make coverage` | Запустить тесты с покрытием; падает ниже 80% |
| `make clean` | Удалить `.venv`, кэши и артефакты сборки |
| `make help` | Показать доступные цели |

## Разработка

```bash
make test        # pytest
make coverage    # pytest --cov с --cov-fail-under=80
```

Автотесты работают на моках и не требуют реальной модели.

## Лицензия

Подробности о лицензии — в репозитории.
