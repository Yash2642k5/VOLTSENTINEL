from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .decision_engine import QuotaExceededError, _attempt_json_repair, _strip_code_fence

MAX_TOOL_TURNS = 3
MAX_PARSE_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 1.0


class ToolLoopError(Exception):
    """Raised internally when a model turn can't be parsed even after
    repair. Always caught inside run_tool_loop — never escapes to the
    caller, since one bad turn shouldn't crash the whole conversation."""
    pass


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, str]
    fn: Callable[..., Any]


def build_tools_block(tools: Dict[str, Tool]) -> str:
    lines = []
    for tool in tools.values():
        params = ", ".join(f"{name} ({desc})" for name, desc in tool.parameters.items())
        lines.append(f"- {tool.name}({params or 'no arguments'}): {tool.description}")
    return "\n".join(lines)


def _parse_turn(raw_text: str) -> Dict[str, Any]:
    text = _strip_code_fence(raw_text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        repaired = _attempt_json_repair(text)
        if repaired is None:
            raise ToolLoopError(f"could not parse model output as JSON: {raw_text[:200]!r}")
        data = json.loads(repaired)

    if not isinstance(data, dict):
        raise ToolLoopError("model turn is not a JSON object")
    if "tool_call" not in data and "final_answer" not in data:
        raise ToolLoopError("model turn has neither 'tool_call' nor 'final_answer'")
    return data


def _execute_tool(tools: Dict[str, Tool], name: Any, arguments: Optional[Dict[str, Any]]) -> Any:
    tool = tools.get(name)
    if tool is None:
        return {"error": f"unknown tool {name!r}. Available tools: {', '.join(tools) or 'none'}"}
    try:
        return tool.fn(**(arguments or {}))
    except TypeError as e:
        return {"error": f"bad arguments for '{name}': {e}"}
    except Exception as e:  # a tool must never crash the whole conversation
        return {"error": f"tool '{name}' failed: {e}"}


def _quota_error_result(error: QuotaExceededError) -> Dict[str, Any]:
    delay = error.retry_delay_seconds
    wait_hint = f" Please try again in about {int(delay)}s." if delay else " Please wait a bit and try again."
    return {
        "text": f"I've hit the API's rate limit for this minute.{wait_hint}",
        "chart": None,
        "error": "quota_exceeded",
    }


def run_tool_loop(
    client,
    tools: Dict[str, Tool],
    transcript: List[str],
    max_tool_turns: int = MAX_TOOL_TURNS,
    retry_backoff_seconds: float = RETRY_BACKOFF_SECONDS,
) -> Dict[str, Any]:
    preamble = "Available tools:\n" + (build_tools_block(tools) or "(none)")

    tool_calls_made = 0
    force_final = False
    last_error = "unknown error"

    while True:
        if force_final:
            suffix = (
                "\n\nYou have reached the tool-call limit for this turn. Respond now with "
                "a final_answer JSON object using only the information already gathered above."
            )
        else:
            suffix = "\n\nYour next turn (a single JSON object, nothing else):"

        prompt = preamble + "\n\n" + "\n\n".join(transcript) + suffix

        parsed = None
        quota_error: Optional[QuotaExceededError] = None
        for attempt in range(MAX_PARSE_ATTEMPTS):
            try:
                raw = client.generate(prompt)
                parsed = _parse_turn(raw)
                break
            except QuotaExceededError as e:
                quota_error = e
                break
            except ToolLoopError as e:
                last_error = str(e)
            except Exception as e:
                last_error = f"API error: {e}"
            if attempt < MAX_PARSE_ATTEMPTS - 1:
                time.sleep(retry_backoff_seconds)

        if quota_error is not None:
            return _quota_error_result(quota_error)

        if parsed is None:
            return {"text": f"I couldn't complete that request ({last_error}).",
                    "chart": None, "error": last_error}

        if "final_answer" in parsed:
            return parsed["final_answer"]

        if force_final:
            return {"text": "I gathered some information but couldn't finish reasoning in time.",
                    "chart": None, "error": "max_tool_turns exceeded"}

        call = parsed.get("tool_call") or {}
        name = call.get("name")
        arguments = call.get("arguments") or {}
        result = _execute_tool(tools, name, arguments)
        tool_calls_made += 1

        if (
            isinstance(result, dict)
            and isinstance(result.get("error"), str)
            and ("429" in result["error"] or "quota" in result["error"].lower())
        ):
            return {
                "text": (
                    "I've hit the API's rate limit for this minute while trying to look "
                    f"that up ({result['error']}). Please try again shortly."
                ),
                "chart": None,
                "error": "quota_exceeded",
            }

        transcript.append(f"Assistant called tool: {name}({json.dumps(arguments, default=str)})")
        transcript.append(f"Tool result for {name}: {json.dumps(result, default=str)}")

        if tool_calls_made >= max_tool_turns:
            force_final = True