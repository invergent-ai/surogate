# Quickstart: Native Serving

This starts Surogate's native OpenAI-compatible inference server with a single model.
Fastest path: run directly from CLI overrides (no YAML file).

## 1) Start the server (no YAML file)

```bash
surogate serve \
  --model Qwen/Qwen3-0.6B \
  --model_id qwen3-local \
  --port 8000 \
  --api_key my-secret-key \
  --max_num_seqs 64 \
  --max_model_len 4096
```

With `uv`:

```bash
uv run surogate serve \
  --model Qwen/Qwen3-0.6B \
  --model_id qwen3-local \
  --port 8000 \
  --api_key my-secret-key \
  --max_num_seqs 64 \
  --max_model_len 4096
```

## 2) Verify the server

Health check:

```bash
curl http://localhost:8000/health
```

List models:

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer my-secret-key"
```

## 3) Generate text

### Chat Completions

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-local",
    "messages": [
      {"role": "system", "content": "You are concise."},
      {"role": "user", "content": "Write one sentence about inference throughput."}
    ],
    "max_tokens": 80
  }'
```

### Completions

```bash
curl http://localhost:8000/v1/completions \
  -H "Authorization: Bearer my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-local",
    "prompt": "Surogate serving in one line:",
    "max_tokens": 40
  }'
```

## 4) Optional: use a YAML config

Create `serve.yaml`:

```yaml
model: Qwen/Qwen3-0.6B
model_id: qwen3-local

host: 0.0.0.0
port: 8000
api_key: my-secret-key

gpus: 1
dtype: bf16
max_num_seqs: 64
max_num_batched_tokens: 4096
gpu_memory_utilization: 0.9

max_gen_len: 512
temperature: 0.7
top_p: 0.95
```

Start:

```bash
surogate serve serve.yaml
```

Or:

```bash
uv run surogate serve serve.yaml
```

## Notes

- If `api_key` is set, all `/v1/*` endpoints require `Authorization: Bearer <api_key>`.
- Main endpoints: `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`.
- Request-level fields (`max_tokens`, `temperature`, `top_p`, etc.) can override config defaults.

## See also

- [CLI reference](../reference/cli.md)
- [Config reference](../reference/config.md)
- [Back to docs index](../index.mdx)
