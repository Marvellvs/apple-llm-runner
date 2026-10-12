# local-llm

Локальный LLM-сервер для macOS (Apple Silicon) на базе [MLX](https://github.com/ml-explore/mlx) с OpenAI-совместимым API.

Скачивайте MLX-модели с Hugging Face, запускайте их полностью на устройстве и общайтесь через `POST /v1/chat/completions` (включая SSE-стриминг) или `POST /v1/responses` (OpenAI Responses API: стриминг, reasoning, function calling, сохранённые ответы, фон) — без облака и без собственных API-ключей.

> План проекта и заметки по дизайну: [docs/PLAN.md](docs/PLAN.md).
> Заметки о совместимости Responses API: [docs/RESPONSES_API.md](docs/RESPONSES_API.md).

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
| `LLM_RESPONSES_STORE_MAX` | `100` | Максимум объектов Responses API в памяти (>= 1) |
| `LLM_RESPONSES_STORE_TTL` | `3600` | Время жизни сохранённого ответа, секунды (>= 1) |
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
| `400` | Некорректное тело запроса (валидация, неподдерживаемая комбинация или элемент `input`) |
| `404` | Запрошенная модель не загружена, либо id ответа неизвестен/истёк |
| `429` | Слишком много одновременно выполняющихся фоновых запросов |
| `500` | Сбой генерации (сервер продолжает работать) |

### `POST /v1/responses`

OpenAI Responses API для AI coding harness (opencode, Kilo Code, клиенты на AI SDK). Тот же MLX-движок, та же конфигурация и та же очередь генерации, что и у `/v1/chat/completions`.

| Поле | Тип | Обязательно | Описание |
|---|---|---|---|
| `model` | string | да | Имя модели (должно совпадать с загруженной) |
| `input` | string \| array | да | Строка пользователя или элементы: `message` (роли `system`/`developer`/`user`/`assistant`), `function_call`, `function_call_output`, `reasoning` |
| `instructions` | string | нет | Системная инструкция; не дублируется при повторе с `previous_response_id` |
| `temperature`, `top_p` | number | нет | Переопределяют `LLM_TEMPERATURE` / `LLM_TOP_P` |
| `max_output_tokens` | integer | нет | Переопределяет `LLM_MAX_TOKENS`; при достижении лимита ответ получает статус `incomplete` |
| `stream` | boolean | нет | `true` — типизированные SSE-события (`response.created` … `response.completed` … `data: [DONE]`) |
| `store` | boolean | нет | По умолчанию `true`: ответ доступен через `GET /v1/responses/{id}`, пока помещается в in-memory хранилище |
| `previous_response_id` | string | нет | Продолжение сохранённого ответа; неизвестный/истёкший id — `404` |
| `tools`, `tool_choice` | array / string | нет | Function tools; только для моделей, чей chat template поддерживает инструменты |
| `reasoning` | object | нет | `{"effort": "none"}` отключает thinking (если модель/шаблон понимает этот ключ); остальные значения — справочные и промпт не меняют |
| `text` | object | нет | Поддерживается только `{"format": {"type": "text"}}`, иначе `400` |
| `background` | boolean | нет | Сразу вернуть `in_progress` и догенерировать в фоне; нельзя вместе со `stream` |

Неизвестные поля игнорируются. Ответ — Response object: `id` с префиксом `resp_`, `object: "response"`, `created_at`, `status`, `error`, `incomplete_details`, `model`, `output`, `usage`, `instructions`, `tools`, `tool_choice`, `temperature`, `top_p`, `max_output_tokens`, `background`, `previous_response_id`, `store`. Статусы: `in_progress`, `completed`, `incomplete`, `failed`.

Элементы `output`: `message` (контент `output_text`), `reasoning` (summary, если модель реально печатает маркеры мышления) и `function_call` (`call_id`, `name`, `arguments`). Reasoning и вызовы инструментов отдаются только тогда, когда они есть в выводе модели, — они никогда не имитируются.

#### События стриминга

При `stream: true` ответ имеет тип `text/event-stream`; каждое событие несёт и имя `event:`, и поле `type`, поток завершается `data: [DONE]`:

```
response.created -> response.in_progress
  message:    output_item.added -> content_part.added -> output_text.delta*
              -> output_text.done -> content_part.done -> output_item.done
  reasoning:  output_item.added -> reasoning_summary_part.added
              -> reasoning_summary_text.delta* -> ...done -> ...part.done -> output_item.done
  function:   output_item.added -> function_call_arguments.delta
              -> function_call_arguments.done -> output_item.done
response.completed (с usage) | response.incomplete | error
data: [DONE]
```

#### Сохранённые ответы и фон

`GET /v1/responses/{response_id}` возвращает сохранённый Response object (`404`, если ответ неизвестен или истёк). Хранилище **только в памяти**: ограничено `LLM_RESPONSES_STORE_MAX` записями и временем жизни `LLM_RESPONSES_STORE_TTL`; **после перезапуска сервера история теряется**.

Фоновые генерации выполняются отдельными рабочими потоками через ту же очередь (одновременно идёт не более одной генерации); одновременно незавершённых фоновых ответов — не более 8, дальше запрос отклоняется с `429`.

#### Известные ограничения

- Function calling требует модели, чей токенизатор/шаблон объявляет поддержку инструментов; иначе запрос с `tools` возвращает `400` с кодом `unsupported_tools`.
- Reasoning выделяется только если модель объявляет маркеры мышления; `reasoning.effort: "none"` отключает thinking, остальные значения промпт не меняют.
- Только текст: изображения/файлы в `input` не поддерживаются, `text.format` кроме `text` отклоняется.
- Usage берётся из движка; `output_tokens_details.reasoning_tokens` считается локально, если движок отдаёт токенизатор.

Подробнее: [docs/RESPONSES_API.md](docs/RESPONSES_API.md).

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

Responses API, обычный запрос:

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B-MLX-bf16",
    "instructions": "You are a helpful assistant.",
    "input": "Say hello in one short sentence.",
    "max_output_tokens": 64
  }'
```

Responses API, стриминг с описанием инструмента:

```bash
curl -N http://127.0.0.1:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B-MLX-bf16",
    "input": "What is the weather in Paris?",
    "stream": true,
    "tools": [
      {
        "type": "function",
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
          "type": "object",
          "properties": { "city": { "type": "string" } },
          "required": ["city"]
        }
      }
    ]
  }'
```

Получить сохранённый ответ (или опросить фоновый):

```bash
curl http://127.0.0.1:8000/v1/responses/resp_0123456789abcdef
```

Запустить фоновую генерацию и опросить её:

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{ "model": "Qwen3-0.6B-MLX-bf16", "input": "Write a haiku.", "background": true }'
```

## Make-цели

| Цель | Описание |
|---|---|
| `make setup` | Создать `.venv` (Python >= 3.12) и установить пакет в editable-режиме с dev-зависимостями |
| `make download MODEL=<repo>` | Скачать MLX-модель с Hugging Face в `LLM_MODELS_DIR` |
| `make list` | Показать скачанные модели с размерами и путями |
| `make delete MODEL=<name>` | Удалить скачанную модель |
| `make run` | Запустить сервер с моделью из `LLM_MODEL` |
| `make binary` | Собрать `dist/local-llm` — один самодостаточный исполняемый файл (Python на целевой машине не нужен) |
| `make binary-dir` | Собрать то же самое как каталог `dist/local-llm-dir/` — стартует без распаковки |
| `make test` | Запустить тесты |
| `make coverage` | Запустить тесты с покрытием; падает ниже 80% |
| `make clean` | Удалить `.venv`, кэши и артефакты сборки |
| `make help` | Показать доступные цели |

## Один бинарник

`make binary` упаковывает CLI, FastAPI-сервер и все зависимости (CPython, MLX,
Metal-ядра MLX, FastAPI, transformers) в один исполняемый файл ≈131 МиБ —
`dist/local-llm`. На целевой машине нужен только Apple Silicon Mac с macOS 11+:
ни Python, ни virtualenv.

Бинарник ведёт себя как `python -m local_llm`:

```bash
./dist/local-llm --help                 # serve / download / list / delete
LLM_MODEL=Qwen3-0.6B-MLX-bf16 ./dist/local-llm serve
```

Веса моделей намеренно не встраиваются: каждая весит 1+ ГБ, а однофайловая сборка
распаковывается во временный каталог при каждом запуске (≈9 с против ≈0.27 с у
варианта-каталога). Держите их в `LLM_MODELS_DIR` (по умолчанию `./models`) или
скачайте тем же бинарником:

```bash
./dist/local-llm download mlx-community/Qwen3-0.6B-MLX-bf16
```

`make binary-dir` собирает то же приложение каталогом `dist/local-llm-dir/`.
Переключатели сборки (встраивание модели, своё имя, one-file/one-dir) описаны в
[packaging/README.md](packaging/README.md).

## Разработка

```bash
make test        # pytest
make coverage    # pytest --cov с --cov-fail-under=80
```

Автотесты работают на моках и не требуют реальной модели.

## Лицензия

Подробности о лицензии — в репозитории.
