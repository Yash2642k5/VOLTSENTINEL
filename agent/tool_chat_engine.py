"""
agent/tool_chat_engine.py

Generic ReAct-style tool-calling loop shared by every conversational
agent surface in VoltSentinel — agent/bi_chat_engine.py (built now) and
the planned incident/security chat (same loop, a different tool
registry that also exposes agent/actions.py's writes as *proposed*
tool calls, gated behind an explicit fleet-manager "Accept" step in the
UI — the "agentic IDE" pattern: propose a change, show a diff/preview,
only execute once the human clicks Accept).

Why hand-rolled JSON instead of Gemini's native function-calling API:
this reuses the exact JSON-schema + validate + repair + retry +
fallback pattern already proven out in agent/decision_engine.py
(including its json_repair-based recovery from a model that
almost-but-not-quite returns valid JSON), so it doesn't depend on
pinning an exact SDK version's tool-schema support, and it's fully
unit-testable with a scripted fake client — no network, matching this
project's existing test style (tests/test_decision_engine.py).

Protocol, one model turn at a time. The model must respond with EXACTLY
one JSON object:

    {"tool_call": {"name": "<tool>", "arguments": {...}}}
or
    {"final_answer": {...}}   <- shape is defined by the caller

We execute tool_call locally against the registry passed in, append the
result back into the running transcript as plain text, and call the
model again. The loop ends when the model emits final_answer, or when
max_tool_turns is hit — at which point we force one last turn asking
the model to answer with whatever it already has, rather than looping
forever or silently truncating.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .decision_engine import _attempt_json_repair, _strip_code_fence

MAX_TOOL_TURNS = 6
MAX_PARSE_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 1.0


class ToolLoopError(Exception):
    """Raised internally when a model turn can't be parsed even after
    repair. Always caught inside run_tool_loop — never escapes to the
    caller, since one bad turn shouldn't crash the whole conversation."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, str]  # param_name -> human-readable type/description
    fn: Callable[..., Any]      # called with exactly the kwargs the model supplies


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


def run_tool_loop(
    client,
    tools: Dict[str, Tool],
    transcript: List[str],
    max_tool_turns: int = MAX_TOOL_TURNS,
    retry_backoff_seconds: float = RETRY_BACKOFF_SECONDS,
) -> Dict[str, Any]:
    """client: anything with .generate(prompt: str) -> str (GeminiClient or
    a test double — same interface agent/decision_engine.py already uses).

    transcript: the running list of human-readable turn strings (the
    user's question, tool calls made, tool results returned, ...) — not
    role-tagged chat objects, since the client's .generate() takes one
    flat prompt string. The system prompt itself is expected to already
    be baked into the client (system_instruction=...), matching how
    GeminiClient is constructed elsewhere in this codebase — this
    function only adds the tool catalogue and the running transcript.

    Returns whatever dict the model supplied under "final_answer". Never
    raises: if every parse attempt fails, or the tool-turn budget is
    exhausted, returns a graceful {"text": ..., "error": ...}-shaped
    fallback instead."""
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
        for attempt in range(MAX_PARSE_ATTEMPTS):
            try:
                raw = client.generate(prompt)
                parsed = _parse_turn(raw)
                break
            except ToolLoopError as e:
                last_error = str(e)
            except Exception as e:
                last_error = f"API error: {e}"
            if attempt < MAX_PARSE_ATTEMPTS - 1:
                time.sleep(retry_backoff_seconds)

        if parsed is None:
            return {"text": f"I couldn't complete that request ({last_error}).",
                    "chart": None, "error": last_error}

        if "final_answer" in parsed:
            return parsed["final_answer"]

        if force_final:
            # The model ignored the instruction to wrap up — stop rather
            # than risk an infinite loop.
            return {"text": "I gathered some information but couldn't finish reasoning in time.",
                    "chart": None, "error": "max_tool_turns exceeded"}

        call = parsed.get("tool_call") or {}
        name = call.get("name")
        arguments = call.get("arguments") or {}
        result = _execute_tool(tools, name, arguments)
        tool_calls_made += 1

        transcript.append(f"Assistant called tool: {name}({json.dumps(arguments, default=str)})")
        transcript.append(f"Tool result for {name}: {json.dumps(result, default=str)}")

        if tool_calls_made >= max_tool_turns:
            force_final = True