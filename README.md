# local-llm

Local LLM server for macOS (Apple Silicon) powered by [MLX](https://github.com/ml-explore/mlx) with an OpenAI-compatible API.

Download MLX models from Hugging Face, run them fully on-device, and talk to them through `POST /v1/chat/completions` (including SSE streaming) or `POST /v1/responses` (OpenAI Responses API, including streaming, reasoning, function calling, stored responses and background mode) — no cloud, no API keys of your own to manage.

> Project plan and design notes: [docs/PLAN.md](docs/PLAN.md).
> Responses API compatibility notes: [docs/RESPONSES_API.md](docs/RESPONSES_API.md).

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python >= 3.12
- Enough free unified memory for the model you download (e.g. ~2 GB free for a 0.6B bf16 model)

## Quick start

```bash
# 1. Create .venv and install the package (editable) + dev deps
make setup

# 2. (Optional) Hugging Face token — only needed for gated/private models
export HF_TOKEN=hf_...          # or put it in a .env file (see Configuration)

# 3. Download an MLX model into ./models
make download MODEL=mlx-community/Qwen3-0.6B-MLX-bf16

# 4. Tell the server which model to load
export LLM_MODEL=Qwen3-0.6B-MLX-bf16

# 5. Start the OpenAI-compatible server
make run
```

The server listens on `http://127.0.0.1:8000` by default, loads the model into memory before accepting requests, and logs a startup banner with the model path and generation parameters.

A small dev/smoke model (~1.2 GB): `mlx-community/Qwen3-0.6B-MLX-bf16`.

## Configuration

All settings come from environment variables (or a `.env` file in the project root — loaded automatically). No config files to edit.

| Variable | Default | Description |
|---|---|---|
| `LLM_HOST` | `127.0.0.1` | Address the server binds to |
| `LLM_PORT` | `8000` | Port the server listens on |
| `LLM_MODEL` | — (required for `run`) | Local model name (folder under `LLM_MODELS_DIR`) or HF repo id |
| `LLM_MODELS_DIR` | `./models` | Directory where downloaded models are stored |
| `LLM_MAX_TOKENS` | `512` | Maximum number of tokens to generate |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature (0–2) |
| `LLM_TOP_P` | `1.0` | Top-p (nucleus) sampling (0–1) |
| `LLM_RESPONSES_STORE_MAX` | `100` | Maximum number of Responses API objects kept in memory (>= 1) |
| `LLM_RESPONSES_STORE_TTL` | `3600` | Lifetime of a stored response in seconds (>= 1) |
| `HF_TOKEN` | — (optional) | Hugging Face access token; required only for gated/private models |

**About `HF_TOKEN`:**

- Optional — public models download anonymously without it.
- When set, it is passed explicitly to the Hugging Face SDK (`snapshot_download(..., token=...)`) for downloads and listing.
- Never logged or printed by the CLI/server.
- Recommended sources: `export HF_TOKEN=hf_...` in your shell, or a `.env` file (picked up at startup).

Invalid values (e.g. an out-of-range port or temperature) fail fast at startup with a message naming the offending variable.

## Managing models

```bash
# Download an MLX model into LLM_MODELS_DIR (idempotent)
make download MODEL=mlx-community/Qwen3-0.6B-MLX-bf16

# List downloaded models with sizes and paths
make list

# Delete a downloaded model by name (as shown by `list`)
make delete MODEL=Qwen3-0.6B-MLX-bf16
```

Only MLX-format models are supported (`mlx-community/*` and other MLX repos on Hugging Face).

## Running the server

```bash
export LLM_MODEL=Qwen3-0.6B-MLX-bf16
make run
```

- Binds to `LLM_HOST:LLM_PORT` (default `127.0.0.1:8000`).
- The model is loaded into memory **before** the server starts accepting requests.
- One generation request is served at a time (MLX unified memory).
- Startup errors (missing model, invalid port, out of memory) exit non-zero with an actionable hint.

## API

### `POST /v1/chat/completions`

OpenAI-compatible chat completions. Request body:

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | yes | Model name (must be the loaded model) |
| `messages` | array | yes | `[{ "role": "system"\|"user"\|"assistant"\|"tool", "content": "..." }, ...]`, at least one |
| `temperature` | number | no | 0–2; overrides `LLM_TEMPERATURE` for this request |
| `max_tokens` | integer | no | >= 1; overrides `LLM_MAX_TOKENS` |
| `top_p` | number | no | 0–1; overrides `LLM_TOP_P` |
| `stream` | boolean | no | `false` by default; `true` for SSE streaming |

Unknown fields are ignored. Response (non-streaming):

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

### Streaming (`stream: true`)

Returns `Content-Type: text/event-stream`. Each event is `data: {...}\n\n` with a `chat.completion.chunk` object containing an incremental `delta`; the stream ends with `data: [DONE]`. Usage (when available) is delivered in a final chunk before `[DONE]`.

### Errors

Errors follow the OpenAI envelope: `{"error": {"message", "type", "code"}}`.

| Status | Meaning |
|---|---|
| `400` | Invalid request body (validation failed, unsupported combination, unsupported input item) |
| `404` | Requested model is not loaded, or a Response id is unknown/expired |
| `429` | Too many background responses are already running |
| `500` | Generation failed (server keeps running) |

### `POST /v1/responses`

OpenAI Responses API for AI coding harnesses (opencode, Kilo Code, AI SDK based clients). Same MLX engine, same configuration and the same generation queue as `/v1/chat/completions`.

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | yes | Model name (must be the loaded model) |
| `input` | string \| array | yes | A user string, or items: `message` (roles `system`/`developer`/`user`/`assistant`), `function_call`, `function_call_output`, `reasoning` |
| `instructions` | string | no | Prepended as a system instruction (not duplicated when repeated with `previous_response_id`) |
| `temperature` / `top_p` | number | no | Override `LLM_TEMPERATURE` / `LLM_TOP_P` for this request |
| `max_output_tokens` | integer | no | Overrides `LLM_MAX_TOKENS`; reaching it ends the response as `incomplete` |
| `stream` | boolean | no | `true` for typed SSE events (`response.created` … `response.completed` … `data: [DONE]`) |
| `store` | boolean | no | Default `true`: keep the response retrievable through `GET /v1/responses/{id}` while it fits the in-memory store |
| `previous_response_id` | string | no | Continue from a stored response; unknown/expired id -> `404` |
| `tools` / `tool_choice` | array / string | no | Function tools (`{"type":"function","name","description","parameters"}`); only for models whose chat template supports tools |
| `reasoning` | object | no | `{"effort": "none"}` disables thinking (models/templates that support the keyword); other values are advisory and do not change the prompt |
| `text` | object | no | Only `{"format": {"type": "text"}}` is supported; other formats return `400` |
| `background` | boolean | no | Return `in_progress` immediately and finish the generation in the background; poll with `GET`. Cannot be combined with `stream` |

Unknown fields are ignored. Response: a Response object (`id` with the `resp_` prefix, `object: "response"`, `created_at`, `status`, `error`, `incomplete_details`, `model`, `output`, `usage`, `instructions`, `tools`, `tool_choice`, `temperature`, `top_p`, `max_output_tokens`, `background`, `previous_response_id`, `store`). Statuses: `in_progress`, `completed`, `incomplete`, `failed`.

`output` items: `message` (`output_text` content), `reasoning` (summary text, when the model emits its thinking markers) and `function_call` (`call_id`, `name`, `arguments`). Reasoning and function calling are only reported when the model output actually contains them — they are never imitated.

#### Streaming events

With `stream: true` the response is `text/event-stream`; every event carries both an `event:` name and a `type` field, and the stream ends with `data: [DONE]`:

```
response.created -> response.in_progress
  message:    response.output_item.added -> response.content_part.added -> response.output_text.delta*
              -> response.output_text.done -> response.content_part.done -> response.output_item.done
  reasoning:  response.output_item.added -> response.reasoning_summary_part.added
              -> response.reasoning_summary_text.delta* -> ...done -> ...part.done -> response.output_item.done
  function:   response.output_item.added -> response.function_call_arguments.delta
              -> response.function_call_arguments.done -> response.output_item.done
response.completed (with usage) | response.incomplete | error
data: [DONE]
```

#### Stored responses and background mode

`GET /v1/responses/{response_id}` returns the stored Response object (`404` when unknown or expired). The store is **in-memory only**: it is bounded by `LLM_RESPONSES_STORE_MAX` entries with a `LLM_RESPONSES_STORE_TTL` lifetime, and **history is lost when the server restarts**.

Background generations run on their own worker threads through the same generation queue (at most one generation at a time), and at most 8 background responses may be pending at once — beyond that the request is rejected with `429`.

#### Known limitations

- Function calling requires a model whose tokenizer/template advertises tool support (mlx-lm `has_tool_calling`); otherwise a request with `tools` returns `400` with code `unsupported_tools`.
- Reasoning is reported only when the loaded model declares thinking markers; `reasoning.effort: "none"` disables thinking, other effort values do not change the prompt.
- `tools` with images/files in `input` is not supported (text parts only), `text.format` other than `text` is rejected.
- Usage comes from the engine; `output_tokens_details.reasoning_tokens` is counted locally when the engine exposes a tokenizer.

More details and the verified contract notes: [docs/RESPONSES_API.md](docs/RESPONSES_API.md).

## curl examples

Plain (non-streaming) request:

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

Streaming request (SSE):

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

(`-N` disables curl buffering so tokens appear as they are generated.)

Responses API, plain (non-streaming) request:

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

Responses API, streaming with a tool definition:

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

Retrieve a stored response (or poll a background one):

```bash
curl http://127.0.0.1:8000/v1/responses/resp_0123456789abcdef
```

Start a background generation and poll it:

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{ "model": "Qwen3-0.6B-MLX-bf16", "input": "Write a haiku.", "background": true }'
```

## Make targets

| Target | Description |
|---|---|
| `make setup` | Create `.venv` (Python >= 3.12) and install the package in editable mode with dev deps |
| `make download MODEL=<repo>` | Download an MLX model from Hugging Face into `LLM_MODELS_DIR` |
| `make list` | List downloaded models with their sizes and paths |
| `make delete MODEL=<name>` | Delete a downloaded model |
| `make run` | Start the server with the model from `LLM_MODEL` |
| `make test` | Run the test suite |
| `make coverage` | Run tests with coverage; fails below 80% |
| `make clean` | Remove `.venv`, caches and build artifacts |
| `make help` | List available targets |

## Development

```bash
make test        # pytest
make coverage    # pytest --cov with --cov-fail-under=80
```

Autotests run on mocks and do not require a real model.

## License

See the repository for license details.
