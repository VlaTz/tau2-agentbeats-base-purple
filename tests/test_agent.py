import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import agent as purple_agent


# A2A validation helpers - adapted from https://github.com/a2aproject/a2a-inspector/blob/main/backend/validators.py

def validate_agent_card(card_data: dict[str, Any]) -> list[str]:
    """Validate the structure and fields of an agent card."""
    errors: list[str] = []

    # Use a frozenset for efficient checking and to indicate immutability.
    required_fields = frozenset(
        [
            'name',
            'description',
            'url',
            'version',
            'capabilities',
            'defaultInputModes',
            'defaultOutputModes',
            'skills',
        ]
    )

    # Check for the presence of all required fields
    for field in required_fields:
        if field not in card_data:
            errors.append(f"Required field is missing: '{field}'.")

    # Check if 'url' is an absolute URL (basic check)
    if 'url' in card_data and not (
        card_data['url'].startswith('http://')
        or card_data['url'].startswith('https://')
    ):
        errors.append(
            "Field 'url' must be an absolute URL starting with http:// or https://."
        )

    # Check if capabilities is a dictionary
    if 'capabilities' in card_data and not isinstance(
        card_data['capabilities'], dict
    ):
        errors.append("Field 'capabilities' must be an object.")

    # Check if defaultInputModes and defaultOutputModes are arrays of strings
    for field in ['defaultInputModes', 'defaultOutputModes']:
        if field in card_data:
            if not isinstance(card_data[field], list):
                errors.append(f"Field '{field}' must be an array of strings.")
            elif not all(isinstance(item, str) for item in card_data[field]):
                errors.append(f"All items in '{field}' must be strings.")

    # Check skills array
    if 'skills' in card_data:
        if not isinstance(card_data['skills'], list):
            errors.append(
                "Field 'skills' must be an array of AgentSkill objects."
            )
        elif not card_data['skills']:
            errors.append(
                "Field 'skills' array is empty. Agent must have at least one skill if it performs actions."
            )

    return errors


def _validate_task(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'id' not in data:
        errors.append("Task object missing required field: 'id'.")
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append("Task object missing required field: 'status.state'.")
    return errors


def _validate_status_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append(
            "StatusUpdate object missing required field: 'status.state'."
        )
    return errors


def _validate_artifact_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'artifact' not in data:
        errors.append(
            "ArtifactUpdate object missing required field: 'artifact'."
        )
    elif (
        'parts' not in data.get('artifact', {})
        or not isinstance(data.get('artifact', {}).get('parts'), list)
        or not data.get('artifact', {}).get('parts')
    ):
        errors.append("Artifact object must have a non-empty 'parts' array.")
    return errors


def _validate_message(data: dict[str, Any]) -> list[str]:
    errors = []
    if (
        'parts' not in data
        or not isinstance(data.get('parts'), list)
        or not data.get('parts')
    ):
        errors.append("Message object must have a non-empty 'parts' array.")
    if 'role' not in data or data.get('role') != 'agent':
        errors.append("Message from agent must have 'role' set to 'agent'.")
    return errors


def validate_event(data: dict[str, Any]) -> list[str]:
    """Validate an incoming event from the agent based on its kind."""
    if 'kind' not in data:
        return ["Response from agent is missing required 'kind' field."]

    kind = data.get('kind')
    validators = {
        'task': _validate_task,
        'status-update': _validate_status_update,
        'artifact-update': _validate_artifact_update,
        'message': _validate_message,
    }

    validator = validators.get(str(kind))
    if validator:
        return validator(data)

    return [f"Unknown message kind received: '{kind}'."]


# A2A messaging helpers

async def send_text_message(text: str, url: str, context_id: str | None = None, streaming: bool = False):
    async with httpx.AsyncClient(timeout=10) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=text))],
            message_id=uuid4().hex,
            context_id=context_id,
        )

        events = [event async for event in client.send_message(msg)]

    return events


# A2A conformance tests

def test_agent_card(agent):
    """Validate agent card structure and required fields."""
    response = httpx.get(f"{agent}/.well-known/agent-card.json")
    assert response.status_code == 200, "Agent card endpoint must return 200"

    card_data = response.json()
    errors = validate_agent_card(card_data)

    assert not errors, f"Agent card validation failed:\n" + "\n".join(errors)

@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_message(agent, streaming):
    """Test that agent returns valid A2A message format."""
    events = await send_text_message("Hello", agent, streaming=streaming)

    all_errors = []
    for event in events:
        match event:
            case Message() as msg:
                errors = validate_event(msg.model_dump())
                all_errors.extend(errors)

            case (task, update):
                errors = validate_event(task.model_dump())
                all_errors.extend(errors)
                if update:
                    errors = validate_event(update.model_dump())
                    all_errors.extend(errors)

            case _:
                pytest.fail(f"Unexpected event type: {type(event)}")

    assert events, "Agent should respond with at least one event"
    assert not all_errors, f"Message validation failed:\n" + "\n".join(all_errors)


class DummyUpdater:
    def __init__(self):
        self.artifacts = []

    async def add_artifact(self, parts, name):
        self.artifacts.append({"parts": parts, "name": name})

    def last_text(self) -> str:
        return self.artifacts[-1]["parts"][0].root.text


class FakeChoice:
    def __init__(self, content: str):
        self.message = type("FakeMessage", (), {"content": content})()


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [FakeChoice(content)]


def make_message(text: str, context_id: str = "ctx-test") -> Message:
    return Message(
        kind="message",
        role=Role.user,
        parts=[Part(root=TextPart(text=text))],
        message_id=uuid4().hex,
        context_id=context_id,
    )


def sample_tau2_prompt() -> str:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_account",
                "description": "Look up the customer account after identity verification.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phone_number": {"type": "string"},
                    },
                    "required": ["phone_number"],
                },
            },
        }
    ]
    respond_tool = {
        "type": "function",
        "function": {
            "name": "respond",
            "description": "Respond directly to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                },
                "required": ["content"],
            },
        },
    }
    return f"""You are helping with telecom support.
- Never change plans before verifying the customer identity.
- Ask for clarification before any irreversible action.

Here's a list of tools you can use (you can use at most one tool at a time):
{json.dumps(tools, indent=2)}

Additionally, you can respond with the following call:
{json.dumps(respond_tool, indent=2)}

Please respond in JSON format.
The JSON should contain:
- "name": the tool call function name.
- "arguments": the arguments for the tool call.

Now here are the user messages:
I want to change my mobile plan today.
"""


def test_parse_benchmark_contract_extracts_policy_tools_and_messages():
    contract = purple_agent.parse_benchmark_contract(sample_tau2_prompt())

    assert "Never change plans" in contract.policy
    assert contract.tool_names == ["lookup_account"]
    assert contract.initial_user_messages == ["I want to change my mobile plan today."]


def test_get_domain_hints_telecom_from_sample_policy():
    contract = purple_agent.parse_benchmark_contract(sample_tau2_prompt())
    hints = purple_agent.get_domain_hints(contract.policy)
    assert "telecom" in hints.lower()
    assert "roaming" in hints.lower()


def test_build_controller_messages_includes_domain_hints():
    state = purple_agent.ConversationState(context_id="ctx")
    state.contract.policy = "Retail Agent Policy — handle orders carefully."
    msgs = purple_agent.build_controller_messages(state, "Hello")
    outer = json.loads(msgs[1]["content"])
    assert "domain_hints" in outer
    assert "retail" in outer["domain_hints"].lower()
    assert "authentication" in outer["domain_hints"].lower()


@pytest.mark.asyncio
async def test_agent_returns_clarification_action_for_ambiguous_request(monkeypatch):
    async def fake_call_llm_with_retry(**kwargs):
        return FakeResponse(
            json.dumps(
                {
                    "mode": "clarify",
                    "reasoning_summary": "Need account verification before plan changes.",
                    "policy_assessment": {
                        "status": "uncertain",
                        "reason": "Missing verified account identifier.",
                    },
                    "selected_action": {
                        "name": "respond",
                        "arguments": {
                            "content": "Please confirm the phone number on the account before I proceed."
                        },
                    },
                    "reply_to_user": "Please confirm the phone number on the account before I proceed.",
                    "state_update": {
                        "pending_questions": ["What is the phone number on the account?"],
                    },
                    "confidence": 0.74,
                }
            )
        )

    monkeypatch.setattr(purple_agent, "call_llm_with_retry", fake_call_llm_with_retry)

    agent = purple_agent.Agent()
    updater = DummyUpdater()

    await agent.run(make_message(sample_tau2_prompt()), updater)

    payload = json.loads(updater.last_text())
    assert payload["name"] == "respond"
    assert "phone number" in payload["arguments"]["content"].lower()


@pytest.mark.asyncio
async def test_agent_blocks_policy_violating_tool_call(monkeypatch):
    async def fake_call_llm_with_retry(**kwargs):
        return FakeResponse(
            json.dumps(
                {
                    "mode": "act",
                    "reasoning_summary": "A tool exists, but policy blocks action before verification.",
                    "policy_assessment": {
                        "status": "blocked",
                        "reason": "Identity verification is still missing.",
                    },
                    "selected_action": {
                        "name": "lookup_account",
                        "arguments": {"phone_number": "555-0100"},
                    },
                    "reply_to_user": "I need to verify your identity before I can make any account changes.",
                    "state_update": {
                        "blocked_reasons": ["identity_verification_missing"],
                    },
                    "confidence": 0.89,
                }
            )
        )

    monkeypatch.setattr(purple_agent, "call_llm_with_retry", fake_call_llm_with_retry)

    agent = purple_agent.Agent()
    updater = DummyUpdater()

    await agent.run(make_message(sample_tau2_prompt()), updater)

    payload = json.loads(updater.last_text())
    assert payload["name"] == "respond"
    assert "verify" in payload["arguments"]["content"].lower()


@pytest.mark.asyncio
async def test_agent_keeps_valid_tool_action(monkeypatch):
    async def fake_call_llm_with_retry(**kwargs):
        return FakeResponse(
            json.dumps(
                {
                    "mode": "act",
                    "reasoning_summary": "The verified phone number allows an account lookup.",
                    "policy_assessment": {
                        "status": "allowed",
                        "reason": "Lookup is safe after verification.",
                    },
                    "selected_action": {
                        "name": "lookup_account",
                        "arguments": {"phone_number": "555-0100"},
                    },
                    "reply_to_user": "",
                    "state_update": {
                        "confirmed_facts": ["Customer confirmed phone number 555-0100."],
                    },
                    "confidence": 0.93,
                }
            )
        )

    monkeypatch.setattr(purple_agent, "call_llm_with_retry", fake_call_llm_with_retry)

    agent = purple_agent.Agent()
    updater = DummyUpdater()

    await agent.run(
        make_message(
            sample_tau2_prompt().rstrip() + "\nCustomer already confirmed the phone number is 555-0100.\n"
        ),
        updater,
    )

    payload = json.loads(updater.last_text())
    assert payload["name"] == "lookup_account"
    assert payload["arguments"] == {"phone_number": "555-0100"}


@pytest.mark.asyncio
async def test_agent_accepts_slim_controller_schema(monkeypatch):
    async def fake_call_llm_with_retry(**kwargs):
        return FakeResponse(
            json.dumps(
                {
                    "mode": "clarify",
                    "policy_status": "uncertain",
                    "policy_reason": "Missing verified account identifier.",
                    "selected_action": {
                        "name": "respond",
                        "arguments": {
                            "content": "Please confirm the phone number on the account before I proceed."
                        },
                    },
                    "reply_to_user": "Please confirm the phone number on the account before I proceed.",
                    "state_update": {
                        "pending_questions": ["What is the phone number on the account?"],
                    },
                }
            )
        )

    monkeypatch.setattr(purple_agent, "call_llm_with_retry", fake_call_llm_with_retry)

    agent = purple_agent.Agent()
    updater = DummyUpdater()

    await agent.run(make_message(sample_tau2_prompt()), updater)

    payload = json.loads(updater.last_text())
    assert payload["name"] == "respond"
    assert "phone number" in payload["arguments"]["content"].lower()


@pytest.mark.asyncio
async def test_agent_repairs_invalid_controller_output(monkeypatch):
    responses = iter(
        [
            FakeResponse("not valid json"),
            FakeResponse(
                json.dumps(
                    {
                        "mode": "clarify",
                        "reasoning_summary": "Repair step recovered a valid clarification.",
                        "policy_assessment": {
                            "status": "uncertain",
                            "reason": "The prior output was malformed, so the safe action is to clarify.",
                        },
                        "selected_action": {
                            "name": "respond",
                            "arguments": {
                                "content": "Please confirm the phone number on the account before I proceed."
                            },
                        },
                        "reply_to_user": "Please confirm the phone number on the account before I proceed.",
                        "state_update": {
                            "pending_questions": ["What is the phone number on the account?"],
                        },
                        "confidence": 0.61,
                    }
                )
            ),
        ]
    )
    seen_messages = []

    async def fake_call_llm_with_retry(**kwargs):
        seen_messages.append(kwargs["messages"])
        return next(responses)

    monkeypatch.setattr(purple_agent, "call_llm_with_retry", fake_call_llm_with_retry)

    agent = purple_agent.Agent()
    updater = DummyUpdater()

    await agent.run(make_message(sample_tau2_prompt()), updater)

    payload = json.loads(updater.last_text())
    assert payload["name"] == "respond"
    assert "phone number" in payload["arguments"]["content"].lower()
    assert len(seen_messages) == 2
    assert "malformed_controller_output" in seen_messages[1][1]["content"]


@pytest.mark.asyncio
async def test_agent_falls_back_to_safe_response_on_invalid_controller_output(monkeypatch):
    responses = iter(
        [
            FakeResponse("not valid json"),
            FakeResponse("still not valid json"),
        ]
    )

    async def fake_call_llm_with_retry(**kwargs):
        return next(responses)

    monkeypatch.setattr(purple_agent, "call_llm_with_retry", fake_call_llm_with_retry)

    agent = purple_agent.Agent()
    updater = DummyUpdater()

    await agent.run(make_message(sample_tau2_prompt()), updater)

    payload = json.loads(updater.last_text())
    assert payload["name"] == "respond"
    assert "internal issue" in payload["arguments"]["content"].lower()
