# Tau2 Purple Agent

Baseline purple agent for the [tau2-bench](https://github.com/sierra-research/tau2-bench) customer service benchmark on [AgentBeats](https://agentbeats.dev). Uses the [OpenAI Python SDK](https://github.com/openai/openai-python) against OpenAI-compatible APIs, so you can switch providers through model name, API base URL, and API key.

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
| `AGENT_LLM` | `openai/gpt-4o-mini` | Model name for an OpenAI-compatible provider. Both `openai/gpt-4o-mini` and raw model IDs are accepted. |
| `OPENAI_API_BASE` | empty | Optional OpenAI-compatible base URL, useful for providers like NVIDIA |
| `OPENAI_API_KEY` | — | Required if using OpenAI models |

## Agent Behavior

This repo now implements a lightweight `custom` purple-agent scaffold instead of a one-shot baseline:

- The first tau2 turn is parsed into `policy`, `tools`, and initial user messages.
- The agent keeps compact per-conversation state: confirmed facts, pending questions, blocked reasons, and recent transcript.
- Each turn is routed through a decision controller that chooses exactly one next action:
  - a tool call, or
  - `respond` with a short clarification / completion message.
- The final outward response is always normalized back into the tau2 JSON action shape:

```json
{"name": "respond", "arguments": {"content": "Please confirm the phone number on the account."}}
```

The controller is explicitly optimized for:

- policy-first behavior,
- clarification-before-commit,
- shared-control awareness,
- conservative claims about completed actions.

The controller's internal tuning is fixed in code, so the external runtime configuration stays minimal for AgentBeats submissions.

## Running Locally

```bash
# Install dependencies
uv sync

# Run the server
AGENT_LLM=openai/gpt-4o-mini OPENAI_API_KEY=sk-... uv run src/server.py
```

You can also use a local `.env` file. `src/server.py` now calls `load_dotenv()`, so gitignored secrets are loaded automatically on startup.

Example for an OpenAI-compatible NVIDIA endpoint:

```bash
AGENT_LLM=openai/gpt-oss-120b \
OPENAI_API_BASE=https://integrate.api.nvidia.com/v1 \
OPENAI_API_KEY=nvapi-... \
uv run src/server.py
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

This agent includes an [Amber](https://github.com/RDI-Foundation/amber) manifest at [`amber-manifest.json5`](amber-manifest.json5) for use in the AgentBeats quick-submit flow.

### Validate the manifest

```bash
docker run --rm -v "$PWD":/work -w /work \
  ghcr.io/rdi-foundation/amber-cli:main check amber-manifest.json5
```

### Run as part of a scenario

The green agent's repo contains the full scenario file that wires this purple agent to the gateway and green agent. See the [green agent amber directory](https://github.com/RDI-Foundation/green-agent-template/tree/main/amber) for the scenario manifest and instructions.

To submit via quick-submit, use the form on the green agent's [AgentBeats page](https://agentbeats.dev). The scenario builder will reference this agent's manifest URL and pass in the configured `AGENT_LLM` and API keys.

For this repo's purple-agent manifest, the only external runtime fields you need in Quick Submit are:

- `AGENT_LLM`
- `OPENAI_API_BASE`
- `OPENAI_API_KEY`

## Testing

```bash
# Install test dependencies
uv sync --extra test

# Start the agent, then run A2A conformance tests
uv run pytest --agent-url http://localhost:9009
```

The test suite now includes:

- A2A card / message conformance checks
- tau2-style contract parsing checks
- scenario tests for clarification, policy blocking, valid tool calls, and malformed controller output

## Evaluation Notes

`tau2-bench` leaderboard submissions distinguish between `standard` and `custom` scaffolds. Because this agent now uses a dedicated decision controller and custom prompts/control flow, you should document it as a `custom` submission when preparing leaderboard metadata. See the public tau2 leaderboard submission guide for the exact schema and verification rules.

## Publishing

The included GitHub Actions workflow builds, tests, and publishes a Docker image to GitHub Container Registry on push to `main`:

```
ghcr.io/rdi-foundation/tau2-purple-agentbeats:latest
```
