# Amber Manifest

Amber manifest for the tau2 purple agent (customer service agent).

## Validate

```sh
docker run --rm -v "$PWD":/work -w /work ghcr.io/rdi-foundation/amber-cli:main check amber/amber-manifest.json5
```

## Generate Docker Compose

```sh
docker run --rm -v "$PWD":/work -w /work ghcr.io/rdi-foundation/amber-cli:main compile amber/amber-manifest.json5 \
  --docker-compose amber/docker-compose.yaml
```

## Configuration

| Config Key | Required | Description |
|---|---|---|
| `agent_llm` | Yes | LLM model in litellm format (e.g. `openai/gpt-4o-mini`) |
| `openai_api_key` | If using OpenAI | OpenAI API key |
| `gemini_api_key` | If using Gemini | Google Gemini API key |
| `deepseek_api_key` | If using DeepSeek | DeepSeek API key |
