# Tau2 Purple Agent

Baseline purple agent for the [tau2-bench](https://github.com/sierra-research/tau2-bench) customer service benchmark on [AgentBeats](https://agentbeats.dev). Uses [litellm](https://docs.litellm.ai/) to call any supported LLM provider.

During evaluation, the [green agent](https://github.com/RDI-Foundation/green-agent-template) sends customer service tasks via the [A2A protocol](https://a2a-protocol.org/latest/). This agent receives each task, uses the provided tools and policy to resolve it, and returns a JSON response.

```
src/
├─ server.py      # Server setup and agent card configuration
├─ executor.py    # A2A request handling
├─ agent.py       # Your agent implementation goes here
└─ messenger.py   # A2A messaging utilities
tests/
└─ test_agent.py  # Agent tests
Dockerfile            # Docker configuration
pyproject.toml        # Python dependencies
amber-manifest.json5  # Amber manifest
.github/
└─ workflows/
   └─ test-and-publish.yml # CI workflow
```

## Getting Started

1. **Create your repository** - Click "Use this template" to create your own repository from this template

2. **Implement your agent** - Add your agent logic to [`src/agent.py`](src/agent.py)

3. **Configure your agent card** - Fill in your agent's metadata (name, skills, description) in [`src/server.py`](src/server.py)

4. **Fill out your [Amber](https://github.com/RDI-Foundation/amber) manifest** - Update [`amber-manifest.json5`](amber-manifest.json5) to use your agent in Amber scenarios

5. **Write your tests** - Add custom tests for your agent in [`tests/test_agent.py`](tests/test_agent.py)

For a concrete example of implementing an agent using this template, see this [draft PR](https://github.com/RDI-Foundation/agent-template/pull/8).
## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `AGENT_LLM` | `openai/gpt-4o-mini` | LLM model in [litellm format](https://docs.litellm.ai/docs/providers) |
| `OPENAI_API_KEY` | — | Required if using OpenAI models |
| `GEMINI_API_KEY` | — | Required if using Gemini models |
| `DEEPSEEK_API_KEY` | — | Required if using DeepSeek models |

## Running Locally

```bash
# Install dependencies
uv sync

# Run the server
AGENT_LLM=openai/gpt-4o-mini OPENAI_API_KEY=sk-... uv run src/server.py
```

## Running with Docker

```bash
# Build the image
docker build -t tau2-purple-agent .

# Run the container
docker run -p 9009:9009 \
  -e AGENT_LLM=openai/gpt-4o-mini \
  -e OPENAI_API_KEY=sk-... \
  tau2-purple-agent
```

## Amber Scenario

This agent includes an [Amber](https://github.com/RDI-Foundation/amber) manifest at [`amber/amber-manifest.json5`](amber/amber-manifest.json5) for use in the AgentBeats quick-submit flow.

### Validate the manifest

```bash
docker run --rm -v "$PWD":/work -w /work \
  ghcr.io/rdi-foundation/amber-cli:main check amber/amber-manifest.json5
```

### Run as part of a scenario

The green agent's repo contains the full scenario file that wires this purple agent to the gateway and green agent. See the [green agent amber directory](https://github.com/RDI-Foundation/green-agent-template/tree/main/amber) for the scenario manifest and instructions.

To submit via quick-submit, use the form on the green agent's [AgentBeats page](https://agentbeats.dev). The scenario builder will reference this agent's manifest URL and pass in the configured `AGENT_LLM` and API keys.

## Testing

```bash
# Install test dependencies
uv sync --extra test

# Start the agent, then run A2A conformance tests
uv run pytest --agent-url http://localhost:9009
```

## Publishing

The included GitHub Actions workflow builds, tests, and publishes a Docker image to GitHub Container Registry on push to `main`:

```
ghcr.io/rdi-foundation/tau2-purple-agentbeats:latest
```
