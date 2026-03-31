import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TextPart
from a2a.utils import get_message_text

LITELLM_IMPORT_ERROR: ImportError | None = None
try:
    from litellm import completion as litellm_completion
    from litellm.exceptions import (
        APIConnectionError,
        RateLimitError,
        ServiceUnavailableError,
        Timeout,
    )
except ImportError as import_error:
    litellm_completion = None
    LITELLM_IMPORT_ERROR = import_error

    class ServiceUnavailableError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    class Timeout(Exception):
        pass

    class APIConnectionError(Exception):
        pass

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

RESPOND_ACTION_NAME = "respond"
TOOLS_MARKER = "Here's a list of tools you can use"
RESPOND_MARKER = "Additionally, you can respond with the following call:"
JSON_FORMAT_MARKER = "Please respond in JSON format."
USER_MESSAGES_MARKER = "Now here are the user messages:"

SAFE_CLARIFICATION_MESSAGE = (
    "I need one more detail before I can safely proceed. "
    "Please confirm the relevant account, order, device, or billing details."
)
SAFE_POLICY_MESSAGE = (
    "I cannot safely take that step under the current policy yet. "
    "Please confirm the missing details or provide an allowed alternative."
)
FAILURE_MESSAGE = (
    "I ran into an internal issue while deciding the next step. "
    "Please restate the most relevant details and I will continue carefully."
)

CONTROLLER_PROMPT = """You are a tau2-bench purple-agent controller.

Choose exactly one next action for the current turn.
Return a single JSON object with this structure:
{
  "mode": "clarify" | "act" | "finalize",
  "reasoning_summary": "short explanation",
  "policy_assessment": {
    "status": "allowed" | "blocked" | "uncertain",
    "reason": "short explanation"
  },
  "selected_action": {
    "name": "tool_name_or_respond",
    "arguments": {}
  },
  "reply_to_user": "short user-facing message when selected_action.name is respond",
  "state_update": {
    "confirmed_facts": ["fact"],
    "pending_questions": ["question"],
    "completed_actions": ["completed action proven by transcript"],
    "blocked_reasons": ["reason"]
  },
  "confidence": 0.0
}

Rules:
- Use exactly one action.
- selected_action.name must be either one of the provided tools or "respond".
- If critical identifiers, consent, or policy permission are missing, choose mode="clarify" and selected_action.name="respond".
- If the policy seems to block the step, do not call a tool; explain briefly to the user instead.
- Never claim that a tool action succeeded unless the transcript explicitly confirms it.
- Distinguish what the agent can do itself from what the user must do in the shared system.
- Keep user-facing responses short and actionable.
- When the task is already solved, use mode="finalize" with selected_action.name="respond".
"""

RETRYABLE_EXCEPTIONS = (
    ServiceUnavailableError,
    RateLimitError,
    Timeout,
    APIConnectionError,
)


@dataclass
class BenchmarkContract:
    policy: str = ""
    tools: list[dict[str, Any]] = field(default_factory=list)
    initial_user_messages: list[str] = field(default_factory=list)
    raw_prompt: str = ""

    @property
    def tool_names(self) -> list[str]:
        names: list[str] = []
        for tool in self.tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if isinstance(function, dict):
                name = function.get("name")
            else:
                name = tool.get("name")
            if isinstance(name, str) and name and name not in names:
                names.append(name)
        return names


@dataclass
class ConversationState:
    context_id: str
    contract: BenchmarkContract = field(default_factory=BenchmarkContract)
    transcript: list[dict[str, str]] = field(default_factory=list)
    confirmed_facts: list[str] = field(default_factory=list)
    pending_questions: list[str] = field(default_factory=list)
    completed_actions: list[str] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)
    turn_count: int = 0


def _safe_json_action(content: str) -> dict[str, Any]:
    return {
        "name": RESPOND_ACTION_NAME,
        "arguments": {"content": content},
    }


def _extract_between(text: str, start_marker: str, end_marker: str | None) -> str | None:
    start = text.find(start_marker)
    if start == -1:
        return None

    start += len(start_marker)
    remainder = text[start:]
    if end_marker is not None:
        end = remainder.find(end_marker)
        if end != -1:
            remainder = remainder[:end]

    return remainder.strip()


def extract_first_json_value(text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("No JSON value found", text, 0)


def parse_benchmark_contract(user_input: str) -> BenchmarkContract:
    contract = BenchmarkContract(raw_prompt=user_input)

    policy_end = user_input.find(TOOLS_MARKER)
    if policy_end != -1:
        contract.policy = user_input[:policy_end].strip()

    tools_block = _extract_between(user_input, TOOLS_MARKER, RESPOND_MARKER)
    if tools_block:
        try:
            tools_value = extract_first_json_value(tools_block)
            if isinstance(tools_value, list):
                contract.tools = [tool for tool in tools_value if isinstance(tool, dict)]
            elif isinstance(tools_value, dict):
                contract.tools = [tools_value]
        except json.JSONDecodeError:
            logger.warning("Could not parse tools block from initial tau2 prompt")

    if USER_MESSAGES_MARKER in user_input:
        messages_block = user_input.split(USER_MESSAGES_MARKER, 1)[1].strip()
        lines = [line.strip() for line in messages_block.splitlines() if line.strip()]
        contract.initial_user_messages = lines or ([messages_block] if messages_block else [])
    elif user_input.strip():
        contract.initial_user_messages = [user_input.strip()]

    return contract


def _remember_items(target: list[str], new_items: list[str], *, limit: int = 12) -> list[str]:
    for item in new_items:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if cleaned and cleaned not in target:
            target.append(cleaned)
    if len(target) > limit:
        del target[:-limit]
    return target


def _format_action_for_memory(action: dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=True, sort_keys=True)


def _looks_like_initial_tau2_prompt(user_input: str) -> bool:
    return TOOLS_MARKER in user_input and JSON_FORMAT_MARKER in user_input


def build_controller_messages(state: ConversationState, current_input: str) -> list[dict[str, str]]:
    payload = {
        "contract": {
            "policy": state.contract.policy,
            "available_tool_names": state.contract.tool_names,
            "tools": state.contract.tools,
            "response_action_name": RESPOND_ACTION_NAME,
        },
        "state": {
            "turn_count": state.turn_count,
            "confirmed_facts": state.confirmed_facts,
            "pending_questions": state.pending_questions,
            "completed_actions": state.completed_actions,
            "blocked_reasons": state.blocked_reasons,
        },
        "recent_transcript": state.transcript[-10:],
        "current_input": current_input,
    }
    return [
        {"role": "system", "content": CONTROLLER_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=True, indent=2)},
    ]


async def call_llm_with_retry(
    *,
    messages: list[dict[str, str]],
    model: str,
    response_format: dict[str, Any],
    temperature: float | None,
    reasoning_effort: str | None,
    max_output_tokens: int | None,
    max_retries: int = 5,
    backoff_base: int = 2,
):
    if litellm_completion is None:
        raise RuntimeError("litellm is not installed in the current environment") from LITELLM_IMPORT_ERROR

    request_kwargs: dict[str, Any] = {
        "messages": messages,
        "model": model,
        "response_format": response_format,
    }
    if temperature is not None:
        request_kwargs["temperature"] = temperature
    if reasoning_effort:
        request_kwargs["reasoning_effort"] = reasoning_effort
    if max_output_tokens:
        request_kwargs["max_tokens"] = max_output_tokens

    for attempt in range(1, max_retries + 1):
        try:
            response = await asyncio.to_thread(litellm_completion, **request_kwargs)
            if attempt > 1:
                logger.info("LLM call succeeded on attempt %s", attempt)
            return response
        except RETRYABLE_EXCEPTIONS as error:
            if attempt >= max_retries:
                logger.error("LLM call failed after %s attempts", max_retries)
                raise

            backoff_seconds = backoff_base ** attempt
            logger.warning(
                "LLM call failed (attempt %s/%s): %s: %s",
                attempt,
                max_retries,
                type(error).__name__,
                str(error)[:120],
            )
            logger.info("Retrying in %ss", backoff_seconds)
            await asyncio.sleep(backoff_seconds)


def parse_controller_output(raw_content: str) -> dict[str, Any]:
    parsed = extract_first_json_value(raw_content)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed)}")
    return parsed


def normalize_controller_output(
    controller_output: dict[str, Any],
    state: ConversationState,
) -> dict[str, Any]:
    policy_assessment = controller_output.get("policy_assessment")
    if not isinstance(policy_assessment, dict):
        policy_assessment = {}

    selected_action = controller_output.get("selected_action")
    if not isinstance(selected_action, dict):
        selected_action = {}

    # Accept the outward tau2 shape as a degraded fallback.
    if not selected_action and isinstance(controller_output.get("name"), str):
        selected_action = {
            "name": controller_output.get("name"),
            "arguments": controller_output.get("arguments", {}),
        }

    mode = controller_output.get("mode", "clarify")
    if mode not in {"clarify", "act", "finalize"}:
        mode = "clarify"

    policy_status = policy_assessment.get("status", "uncertain")
    if policy_status not in {"allowed", "blocked", "uncertain"}:
        policy_status = "uncertain"

    action_name = selected_action.get("name")
    action_args = selected_action.get("arguments")
    if not isinstance(action_args, dict):
        action_args = {}

    reply_to_user = controller_output.get("reply_to_user")
    if not isinstance(reply_to_user, str):
        reply_to_user = ""

    state_update = controller_output.get("state_update")
    if not isinstance(state_update, dict):
        state_update = {}

    if (
        mode in {"clarify", "finalize"}
        and action_name != RESPOND_ACTION_NAME
        and reply_to_user
    ):
        action_name = RESPOND_ACTION_NAME
        action_args = {"content": reply_to_user}

    if policy_status == "blocked":
        final_action = _safe_json_action(reply_to_user or SAFE_POLICY_MESSAGE)
    elif action_name == RESPOND_ACTION_NAME:
        final_action = _safe_json_action(
            action_args.get("content") or reply_to_user or SAFE_CLARIFICATION_MESSAGE
        )
    elif isinstance(action_name, str) and action_name in state.contract.tool_names:
        final_action = {"name": action_name, "arguments": action_args}
    else:
        final_action = _safe_json_action(reply_to_user or SAFE_CLARIFICATION_MESSAGE)

    confidence = controller_output.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "mode": mode,
        "reasoning_summary": str(controller_output.get("reasoning_summary", "")),
        "policy_status": policy_status,
        "policy_reason": str(policy_assessment.get("reason", "")),
        "state_update": state_update,
        "confidence": max(0.0, min(confidence, 1.0)),
        "final_action": final_action,
    }


def apply_state_update(
    state: ConversationState,
    normalized_output: dict[str, Any],
    assistant_content: str,
) -> None:
    state_update = normalized_output.get("state_update", {})

    _remember_items(
        state.confirmed_facts,
        state_update.get("confirmed_facts", []),
    )
    _remember_items(
        state.pending_questions,
        state_update.get("pending_questions", []),
        limit=6,
    )
    _remember_items(
        state.completed_actions,
        state_update.get("completed_actions", []),
    )
    _remember_items(
        state.blocked_reasons,
        state_update.get("blocked_reasons", []),
        limit=6,
    )

    if normalized_output["mode"] == "clarify":
        response_text = normalized_output["final_action"]["arguments"]["content"]
        _remember_items(state.pending_questions, [response_text], limit=6)

    state.transcript.append({"role": "assistant", "content": assistant_content})
    if len(state.transcript) > 20:
        del state.transcript[:-20]

    state.turn_count += 1


class Agent:
    def __init__(self):
        self.model = os.getenv("AGENT_LLM", "openai/gpt-4o-mini")
        self.max_retries = int(os.getenv("AGENT_LLM_MAX_RETRIES", "5"))
        self.backoff_base = int(os.getenv("AGENT_LLM_BACKOFF_BASE", "2"))
        self.temperature = float(os.getenv("AGENT_TEMPERATURE", "0.2"))
        self.reasoning_effort = os.getenv("AGENT_REASONING_EFFORT", "").strip() or None
        self.max_output_tokens = int(os.getenv("AGENT_MAX_OUTPUT_TOKENS", "600"))
        self.state: ConversationState | None = None

        logger.info("Purple agent initialized with model: %s", self.model)
        logger.info(
            "Retry config: max_retries=%s, backoff_base=%s",
            self.max_retries,
            self.backoff_base,
        )

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        user_input = get_message_text(message).strip()
        context_id = message.context_id or "default"

        if self.state is None:
            self.state = ConversationState(context_id=context_id)

        state = self.state
        logger.info("Received message for context %s", context_id)
        logger.debug("User input length: %s chars", len(user_input))

        current_input = user_input
        if state.turn_count == 0 and _looks_like_initial_tau2_prompt(user_input):
            state.contract = parse_benchmark_contract(user_input)
            initial_messages = (
                state.contract.initial_user_messages or [user_input]
            )
            for entry in initial_messages:
                state.transcript.append({"role": "user", "content": entry})
            current_input = initial_messages[-1]
            logger.info(
                "Parsed initial tau2 contract with %s tools",
                len(state.contract.tool_names),
            )
        else:
            state.transcript.append({"role": "user", "content": user_input})
            if len(state.transcript) > 20:
                del state.transcript[:-20]

        llm_messages = build_controller_messages(state, current_input)
        logger.info("Calling LLM %s with %s controller messages", self.model, len(llm_messages))

        try:
            response = await call_llm_with_retry(
                messages=llm_messages,
                model=self.model,
                response_format={"type": "json_object"},
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort,
                max_output_tokens=self.max_output_tokens,
                max_retries=self.max_retries,
                backoff_base=self.backoff_base,
            )
            raw_content = response.choices[0].message.content or ""
            controller_output = parse_controller_output(raw_content)
            normalized_output = normalize_controller_output(controller_output, state)
        except Exception as error:
            logger.error("Controller failed: %s: %s", type(error).__name__, error)
            logger.exception("Full traceback:")
            normalized_output = {
                "mode": "clarify",
                "reasoning_summary": "fallback_after_error",
                "policy_status": "uncertain",
                "policy_reason": "controller error",
                "state_update": {"blocked_reasons": ["controller_error"]},
                "confidence": 0.0,
                "final_action": _safe_json_action(FAILURE_MESSAGE),
            }

        assistant_content = json.dumps(
            normalized_output["final_action"],
            ensure_ascii=True,
        )
        apply_state_update(state, normalized_output, assistant_content)

        logger.info(
            "Selected mode=%s policy=%s action=%s confidence=%.2f",
            normalized_output["mode"],
            normalized_output["policy_status"],
            _format_action_for_memory(normalized_output["final_action"]),
            normalized_output["confidence"],
        )

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=assistant_content))],
            name="Response",
        )
