"""
tests/test_tool_chat_engine.py

Validates agent/tool_chat_engine.py's generic ReAct-style loop: tool
dispatch, error handling for bad tool calls, the parse-repair-retry
path, and the forced-final-turn behaviour when a script tries to keep
calling tools past the budget. No network -- a scripted fake client
plays the model's part, same pattern as tests/test_decision_engine.py's
FakeGeminiClient.

Run from the project root:
    pytest tests/test_tool_chat_engine.py -v
"""

import json

import pytest

from agent.tool_chat_engine import Tool, ToolLoopError, _execute_tool, _parse_turn, run_tool_loop


class FakeClient:
    """responses: list of strings (or Exception instances) returned/raised
    in order, one per .generate() call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0
        self.prompts_seen = []

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.prompts_seen.append(prompt)
        if not self._responses:
            raise AssertionError("FakeClient.generate() called more times than scripted")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _echo_tool(value: str = "") -> dict:
    return {"echoed": value}


def _boom_tool() -> dict:
    raise RuntimeError("kaboom")


@pytest.fixture
def tools():
    return {
        "echo": Tool(name="echo", description="echoes back its input", parameters={"value": "str"}, fn=_echo_tool),
        "boom": Tool(name="boom", description="always raises", parameters={}, fn=_boom_tool),
    }


# ----------------------------------------------------------------------
# _parse_turn
# ----------------------------------------------------------------------
class TestParseTurn:
    def test_valid_tool_call_parses(self):
        data = _parse_turn('{"tool_call": {"name": "echo", "arguments": {"value": "hi"}}}')
        assert data["tool_call"]["name"] == "echo"

    def test_valid_final_answer_parses(self):
        data = _parse_turn('{"final_answer": {"text": "done"}}')
        assert data["final_answer"]["text"] == "done"

    def test_neither_key_raises(self):
        with pytest.raises(ToolLoopError):
            _parse_turn('{"something_else": 1}')

    def test_non_object_raises(self):
        with pytest.raises(ToolLoopError):
            _parse_turn("[1, 2, 3]")

    def test_garbage_raises(self):
        with pytest.raises(ToolLoopError):
            _parse_turn("not json at all")

    def test_code_fence_is_stripped(self):
        data = _parse_turn('```json\n{"final_answer": {"text": "hi"}}\n```')
        assert data["final_answer"]["text"] == "hi"


# ----------------------------------------------------------------------
# _execute_tool
# ----------------------------------------------------------------------
class TestExecuteTool:
    def test_known_tool_executes(self, tools):
        result = _execute_tool(tools, "echo", {"value": "hi"})
        assert result == {"echoed": "hi"}

    def test_unknown_tool_returns_error_not_raise(self, tools):
        result = _execute_tool(tools, "not_a_tool", {})
        assert "error" in result

    def test_bad_arguments_returns_error_not_raise(self, tools):
        result = _execute_tool(tools, "echo", {"unexpected_kwarg": 1})
        assert "error" in result

    def test_tool_exception_returns_error_not_raise(self, tools):
        result = _execute_tool(tools, "boom", {})
        assert "error" in result
        assert "kaboom" in result["error"]


# ----------------------------------------------------------------------
# run_tool_loop -- the full loop
# ----------------------------------------------------------------------
class TestRunToolLoop:
    def test_immediate_final_answer_no_tool_calls(self, tools):
        client = FakeClient(['{"final_answer": {"text": "hi there"}}'])
        result = run_tool_loop(client, tools, transcript=["Fleet manager asks: hello"])
        assert result == {"text": "hi there"}
        assert client.call_count == 1

    def test_one_tool_call_then_final_answer(self, tools):
        client = FakeClient([
            '{"tool_call": {"name": "echo", "arguments": {"value": "ping"}}}',
            '{"final_answer": {"text": "the tool said ping"}}',
        ])
        result = run_tool_loop(client, tools, transcript=["Fleet manager asks: echo ping"])
        assert result == {"text": "the tool said ping"}
        assert client.call_count == 2
        # tool result must have been appended into what the model saw next
        assert "echoed" in client.prompts_seen[1]

    def test_unparseable_response_falls_back_gracefully(self, tools):
        client = FakeClient(["nonsense", "still nonsense"])
        result = run_tool_loop(client, tools, transcript=["Fleet manager asks: x"])
        assert "text" in result
        assert result.get("error")

    def test_unknown_tool_name_is_reported_back_to_model_not_crashed(self, tools):
        client = FakeClient([
            '{"tool_call": {"name": "not_a_real_tool", "arguments": {}}}',
            '{"final_answer": {"text": "gave up gracefully"}}',
        ])
        result = run_tool_loop(client, tools, transcript=["Fleet manager asks: x"])
        assert result == {"text": "gave up gracefully"}
        assert "unknown tool" in client.prompts_seen[1]

    def test_exceeding_max_tool_turns_forces_final_and_stops(self, tools):
        # Script the model to keep calling the tool forever -- the loop must
        # force a final turn instead of looping without bound.
        responses = ['{"tool_call": {"name": "echo", "arguments": {"value": "x"}}}'] * 10
        client = FakeClient(responses)
        result = run_tool_loop(client, tools, transcript=["Fleet manager asks: x"], max_tool_turns=3)
        assert "text" in result
        assert result.get("error") == "max_tool_turns exceeded"
        # exactly max_tool_turns real tool calls + 1 forced-final attempt
        assert client.call_count == 4

    def test_recovers_from_one_malformed_turn_via_repair_or_retry(self, tools):
        # First response is garbage (not valid JSON, not repairable), second
        # succeeds -- the per-turn retry (MAX_PARSE_ATTEMPTS) should recover
        # without needing a whole new tool-call round trip.
        client = FakeClient([
            "not json at all",
            '{"final_answer": {"text": "recovered"}}',
        ])
        result = run_tool_loop(client, tools, transcript=["Fleet manager asks: x"])
        assert result == {"text": "recovered"}