import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TextPart
from a2a.utils import get_message_text
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

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
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 600
RECENT_TRANSCRIPT_LIMIT = 16

CONTROLLER_PROMPT = """You are a tau2-bench purple-agent controller.

Choose exactly one safe next action for the current turn.

Priority:
- Read `domain_hints` in the user JSON first when it applies to this scenario (typical tau2 edge cases).
- Then follow the scenario policy in `contract.policy` and the tools in `contract.tools`. If `domain_hints` and the scenario policy conflict, the scenario policy wins.

Workflow:
- Identify the user's real goal and the specific object involved.
- Re-read domain_hints, the provided policy, and tool schema before deciding.
- Use tools to verify facts when the action depends on system data.
- If required consent, verification, or identifiers are missing, ask a short clarification question instead of acting.
- Once all policy conditions are satisfied and the next tool call is clear, take that action immediately.

Common mistakes to avoid:
- Do not claim that any tool action succeeded unless the transcript explicitly proves it.
- Do not invent policy rules, exceptions, ids, or completed actions.
- Do not ask the user for internal identifiers or information that should be obtained via tools.
- Do not delay with extra explanation once the correct next tool call is clear.
- Do not offer disallowed workarounds when the policy blocks the request.

Efficiency rules:
- Prefer one decisive next step over broad exploration.
- Ask only for details that the user can realistically provide.
- If there is exactly one safe next tool call, emit it directly.

Return a single JSON object with this structure:
{
  "mode": "clarify" | "act" | "finalize",
  "policy_status": "allowed" | "blocked" | "uncertain",
  "policy_reason": "short explanation",
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
  }
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

CONTROLLER_REPAIR_PROMPT = """You repair malformed controller outputs.

Return exactly one JSON object and nothing else.
Preserve the original intent when possible.
If some fields are missing, infer the safest values.
The JSON object must use this structure:
{
  "mode": "clarify" | "act" | "finalize",
  "policy_status": "allowed" | "blocked" | "uncertain",
  "policy_reason": "short explanation",
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
  }
}
"""

DOMAIN_HINTS_AIRLINE = """
Tau2 airline domain — common edge cases (apply when relevant; scenario policy wins if conflict):

- Cabin vs itinerary: policy often allows ALL reservations (including basic economy) to change cabin without changing flights. "Cannot modify basic economy" usually means itinerary/date/flight-number changes, NOT cabin upgrades/downgrades.
- Cabin must match across ALL segments and ALL passengers; refuse split-cabin or per-leg-only changes; explain clearly.
- Origin and destination cities cannot be changed — offer cancel + rebook if user wants different O/D.
- Transfer to human only when truly outside tools; membership disputes or frustration are not automatic transfers — verify records first.
- Cancellation: only when policy allows (e.g. 24h booking, airline-cancelled flight, business class, or insurance+health/weather as policy states). Past flights cannot be cancelled. Multiple reservations: check each.
- Payments: flight changes often need ONE gift card OR credit card; certificates may be invalid for flight changes; for new bookings follow policy on mixing methods.
- Pricing cabin changes: use flight search tools for new cabin prices across all passengers/segments; compare to paid amount; do not use flight status for prices.
- Free bags: use membership + cabin rules from policy; extra bags per policy pricing.
- Cheapest economy vs basic economy are different; search direct then one-stop as needed.
- Multi-reservation: get user details for all ids, then each reservation details; evaluate separately.
""".strip()

DOMAIN_HINTS_RETAIL = """
Tau2 retail domain — common edge cases (apply when relevant; scenario policy wins if conflict):

- Authentication: verify identity as policy requires (e.g. email or name+zip) before sensitive actions — even if user gives a user id.
- modify_pending / exchange_delivered: often ONE call per order — gather ALL line items first; confirm with user before calling.
- Return vs exchange: different tools; match user intent; exchange is same product type variant, not arbitrary product swaps.
- Payment changes: single replacement method; gift card must cover full total if switching to it; refunds per policy (original method or existing gift card).
- Cancel: only pending orders; allowed reasons per policy (e.g. mistake / no longer needed).
- Multi-order: get_order_details per order; check pending vs delivered before choosing cancel vs return/exchange.
- Totals across orders: use calculate tool when available instead of mental math.
""".strip()

DOMAIN_HINTS_TELECOM = """
Tau2 telecom domain — common edge cases (apply when relevant; scenario policy wins if conflict):

- Identify customer before actions (phone, id, or name+dob per policy).
- Technical support: follow troubleshooting order; try relevant steps before escalating.
- Roaming: "enable" account permission vs "toggle" on device may be different tools — use both when policy requires.
- Data refuel: max per request and confirm price before applying.
- Suspension: lift only when policy allows (e.g. bills paid); contract end date may block resume even after payment.
- Overdue bills: follow policy flow (request → user confirms → pay → verify paid).
- Plan changes: list options, confirm price, then apply.
""".strip()

DOMAIN_HINTS_DEFAULT = """
General tau2 discipline (apply when relevant; scenario policy wins if conflict):

- Escalate to human only when truly outside tools; if policy forbids an action, deny clearly.
- Verify with tools before asserting facts or confirming success.
- Confirm destructive or irreversible steps with the user when policy requires.
""".strip()


def get_domain_hints(policy_text: str) -> str:
    """Select compact domain hints from policy text (tau2-bench style)."""
    text_lower = (policy_text or "").lower()
    if "airline agent policy" in text_lower or "airline agent" in text_lower:
        return DOMAIN_HINTS_AIRLINE
    if "retail agent policy" in text_lower or "retail agent" in text_lower:
        return DOMAIN_HINTS_RETAIL
    if "telecom agent policy" in text_lower or "telecom agent" in text_lower:
        return DOMAIN_HINTS_TELECOM
    if "airline" in text_lower and "policy" in text_lower:
        return DOMAIN_HINTS_AIRLINE
    if "retail" in text_lower and "policy" in text_lower:
        return DOMAIN_HINTS_RETAIL
    if "telecom" in text_lower:
        return DOMAIN_HINTS_TELECOM
    return DOMAIN_HINTS_DEFAULT


RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
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
        "domain_hints": get_domain_hints(state.contract.policy),
        "contract": {
            "policy": state.contract.policy,
            "available_tool_names": state.contract.tool_names,
            "tools": state.contract.tools,
            "response_action_name": RESPOND_ACTION_NAME,
        },
        "decision_reminders": [
            "Follow the provided policy exactly.",
            "Use one safe next action only.",
            "Clarify before acting when required details are missing.",
            "Do not claim success for tool actions that have not happened yet.",
        ],
        "state": {
            "turn_count": state.turn_count,
            "confirmed_facts": state.confirmed_facts,
            "pending_questions": state.pending_questions,
            "completed_actions": state.completed_actions,
            "blocked_reasons": state.blocked_reasons,
        },
        "recent_transcript": state.transcript[-RECENT_TRANSCRIPT_LIMIT:],
        "current_input": current_input,
    }
    return [
        {"role": "system", "content": CONTROLLER_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=True, indent=2)},
    ]


def build_controller_repair_messages(raw_content: str) -> list[dict[str, str]]:
    payload = {
        "malformed_controller_output": raw_content,
        "instructions": (
            "Repair the malformed output into one valid JSON object that matches the required schema. "
            "Return JSON only."
        ),
    }
    return [
        {"role": "system", "content": CONTROLLER_REPAIR_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=True, indent=2)},
    ]


def _openai_http_timeout_seconds() -> float:
    """Max wait for a single chat.completions call (slow OSS / long prompts)."""
    raw = os.getenv("AGENT_LLM_HTTP_TIMEOUT", "").strip()
    if raw:
        return float(raw)
    return 600.0


def create_openai_client() -> OpenAI:
    # Strip: secrets often pick up trailing newlines; breaks auth in subtle ways.
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    timeout = _openai_http_timeout_seconds()
    # No base_url: use OpenAI SDK default (https://api.openai.com/v1).
    return OpenAI(api_key=api_key, timeout=timeout)


def request_openai_completion(
    *,
    messages: list[dict[str, str]],
    model: str,
    response_format: dict[str, Any],
    temperature: float | None,
    max_output_tokens: int | None,
):
    client = create_openai_client()

    request_kwargs: dict[str, Any] = {
        "messages": messages,
        "model": model,
        "response_format": response_format,
    }
    if temperature is not None:
        request_kwargs["temperature"] = temperature
    if max_output_tokens:
        request_kwargs["max_tokens"] = max_output_tokens

    return client.chat.completions.create(**request_kwargs)


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
    if reasoning_effort:
        logger.info("Ignoring reasoning_effort=%s because OpenAI-compatible chat completions do not use it here", reasoning_effort)

    for attempt in range(1, max_retries + 1):
        try:
            response = await asyncio.to_thread(
                request_openai_completion,
                messages=messages,
                model=model,
                response_format=response_format,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
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


async def parse_or_repair_controller_output(
    *,
    raw_content: str,
    model: str,
    temperature: float | None,
    reasoning_effort: str | None,
    max_output_tokens: int | None,
    max_retries: int,
    backoff_base: int,
) -> dict[str, Any]:
    try:
        return parse_controller_output(raw_content)
    except (json.JSONDecodeError, ValueError) as initial_error:
        logger.warning(
            "Primary controller output was not valid JSON (%s). Trying repair step.",
            type(initial_error).__name__,
        )
        repair_response = await call_llm_with_retry(
            messages=build_controller_repair_messages(raw_content),
            model=model,
            response_format={"type": "json_object"},
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            backoff_base=backoff_base,
        )
        repaired_content = repair_response.choices[0].message.content or ""
        return parse_controller_output(repaired_content)


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

    policy_status = controller_output.get("policy_status", policy_assessment.get("status", "uncertain"))
    if policy_status not in {"allowed", "blocked", "uncertain"}:
        policy_status = "uncertain"

    policy_reason = controller_output.get("policy_reason", policy_assessment.get("reason", ""))
    if not isinstance(policy_reason, str):
        policy_reason = ""

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
        "policy_reason": policy_reason,
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
        self.temperature = DEFAULT_TEMPERATURE
        self.reasoning_effort = None
        self.max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS
        self.state: ConversationState | None = None

        logger.info("Purple agent initialized with model: %s", self.model)
        logger.info(
            "Retry config: max_retries=%s, backoff_base=%s, llm_http_timeout_s=%s",
            self.max_retries,
            self.backoff_base,
            _openai_http_timeout_seconds(),
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
            controller_output = await parse_or_repair_controller_output(
                raw_content=raw_content,
                model=self.model,
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort,
                max_output_tokens=self.max_output_tokens,
                max_retries=self.max_retries,
                backoff_base=self.backoff_base,
            )
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
