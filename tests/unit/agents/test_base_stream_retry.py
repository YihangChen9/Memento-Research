"""Regression tests for transient-stream retry in BaseAgentRunner.

Bug: when the LLM provider closed the chunked-transfer connection mid-response
(``httpx.RemoteProtocolError: peer closed connection without sending complete
message body (incomplete chunked read)``), the exception propagated out of
``astream_events`` / ``ainvoke`` and failed the task. LangChain's
``ChatOpenAI(max_retries=...)`` only retries the initial request, not mid-stream
chunk failures.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest

from onemancompany.agents import base as base_mod


class _FakeAgent:
    """Minimal stand-in for a LangGraph compiled agent.

    On the first ``n_fail`` calls it raises a transient httpx error, then it
    yields ``events`` (for astream_events) or returns ``invoke_result``
    (for ainvoke).
    """

    def __init__(self, *, events: list[dict] | None = None, invoke_result: dict | None = None, n_fail: int = 1, exc: BaseException | None = None) -> None:
        self.events = events or []
        self.invoke_result = invoke_result or {"messages": []}
        self.n_fail = n_fail
        self.exc = exc or httpx.RemoteProtocolError("peer closed connection without sending complete message body (incomplete chunked read)")
        self.stream_calls = 0
        self.invoke_calls = 0

    def astream_events(self, *_a: Any, **_kw: Any):
        self.stream_calls += 1
        outer = self

        class _Gen:
            def __aiter__(self_inner):
                return self_inner

            async def __anext__(self_inner):
                if outer.stream_calls <= outer.n_fail:
                    raise outer.exc
                if not outer.events:
                    raise StopAsyncIteration
                return outer.events.pop(0)

        return _Gen()

    async def ainvoke(self, _payload: dict) -> dict:
        self.invoke_calls += 1
        if self.invoke_calls <= self.n_fail:
            raise self.exc
        return self.invoke_result


def _make_runner(agent: _FakeAgent) -> base_mod.BaseAgentRunner:
    runner = base_mod.BaseAgentRunner()
    runner.employee_id = "00015"
    runner._agent = agent
    # Skip the rebuild path — refresh_agent would try to instantiate a real LLM.
    runner._refresh_agent = lambda: None  # type: ignore[method-assign]
    runner._set_status = lambda _s: None  # type: ignore[method-assign]
    runner._build_full_prompt = lambda: "system prompt"  # type: ignore[method-assign]

    async def _noop_publish(*_a, **_kw):
        return None

    runner._publish = _noop_publish  # type: ignore[method-assign]
    return runner


def test_is_transient_network_error_classifies_httpx_remote_protocol():
    exc = httpx.RemoteProtocolError("peer closed connection without sending complete message body (incomplete chunked read)")
    assert base_mod._is_transient_network_error(exc)


def test_is_transient_network_error_classifies_by_message_when_wrapped():
    """An opaque RuntimeError wrapping the httpx message should still be detected."""
    exc = RuntimeError("LLM call failed: peer closed connection without sending complete message body")
    assert base_mod._is_transient_network_error(exc)


def test_is_transient_network_error_rejects_unrelated_errors():
    assert not base_mod._is_transient_network_error(ValueError("bad input"))


@pytest.mark.asyncio
async def test_run_streamed_retries_on_transient_remote_protocol_error():
    """First astream_events attempt raises; second attempt succeeds."""
    agent = _FakeAgent(events=[], n_fail=1)
    runner = _make_runner(agent)

    captured: list[tuple[str, Any]] = []
    def on_log(kind: str, payload: Any) -> None:
        captured.append((kind, payload))

    # Speed the test up by zeroing the backoff.
    with patch.object(base_mod, "_LLM_STREAM_RETRY_DELAYS", (0.0, 0.0)):
        result = await runner.run_streamed("do the thing", on_log=on_log)

    assert agent.stream_calls == 2  # 1 failure + 1 success
    assert result == ""  # no events delivered; just verifying retry path completes


@pytest.mark.asyncio
async def test_run_streamed_reraises_after_exhausting_retries():
    agent = _FakeAgent(events=[], n_fail=10)  # always fail
    runner = _make_runner(agent)

    with patch.object(base_mod, "_LLM_STREAM_RETRY_DELAYS", (0.0, 0.0)):
        with pytest.raises(httpx.RemoteProtocolError):
            await runner.run_streamed("do the thing", on_log=lambda *_a: None)

    assert agent.stream_calls == base_mod._LLM_STREAM_RETRY_ATTEMPTS


@pytest.mark.asyncio
async def test_run_streamed_does_not_retry_non_transient_errors():
    agent = _FakeAgent(events=[], n_fail=10, exc=ValueError("schema mismatch"))
    runner = _make_runner(agent)

    with patch.object(base_mod, "_LLM_STREAM_RETRY_DELAYS", (0.0, 0.0)):
        with pytest.raises(ValueError):
            await runner.run_streamed("do the thing", on_log=lambda *_a: None)

    assert agent.stream_calls == 1  # no retries for non-transient errors


# ---------------------------------------------------------------------------
# Empty-final-turn wrap-up (P0): the agent does real tool work but its final
# turn has no text (hit recursion_limit mid-loop, or a tool-call-only final
# message). Instead of shipping a 14-char "Executed: bash" stub — which the
# pipeline reads as a failure and burns retries on at EVERY stage (6a/6b/
# critic/paper-writer) — make ONE tool-free wrap-up call so the agent states
# its result in text.
# ---------------------------------------------------------------------------

class _Out:
    """Minimal stand-in for a LangChain AIMessage from a stream event."""
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.response_metadata = {}
        self.usage_metadata = None


class _FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    async def ainvoke(self, _msgs):
        self.calls += 1
        return _Out(self.reply)


def _empty_final_turn_events():
    return [
        {"event": "on_chat_model_end",
         "data": {"output": _Out("", [{"name": "bash", "args": {"command": "python -m bench"}}])}},
        {"event": "on_tool_end", "name": "bash",
         "data": {"output": "=== RESULT_JSON: {\"accuracy\": 0.8} ==="}},
    ]


@pytest.mark.asyncio
async def test_empty_final_turn_triggers_wrapup_not_stub():
    agent = _FakeAgent(events=_empty_final_turn_events(), n_fail=0)
    runner = _make_runner(agent)
    fake = _FakeLLM("Final result: the benchmark ran, accuracy 0.8 (RESULT_JSON).")
    with patch.object(base_mod, "make_llm", lambda _eid: fake):
        result = await runner.run_streamed("run the benchmark", on_log=lambda *_a: None)
    assert fake.calls == 1, "must make exactly one wrap-up call"
    assert "accuracy 0.8" in result
    assert not result.startswith("Executed:"), "must NOT ship the 14-char stub"


@pytest.mark.asyncio
async def test_wrapup_empty_falls_back_to_stub():
    """If the wrap-up call ALSO yields no text, fall back to the old synthesis
    (no regression — never worse than before)."""
    agent = _FakeAgent(events=_empty_final_turn_events(), n_fail=0)
    runner = _make_runner(agent)
    with patch.object(base_mod, "make_llm", lambda _eid: _FakeLLM("")):
        result = await runner.run_streamed("run the benchmark", on_log=lambda *_a: None)
    assert result.startswith("Executed:")
    assert "bash" in result


@pytest.mark.asyncio
async def test_nonempty_final_turn_skips_wrapup():
    """Regression: when the agent DID return final text, no wrap-up call is
    made and that text is returned verbatim."""
    events = [
        {"event": "on_chat_model_end",
         "data": {"output": _Out("Decision: PASS. Confidence 0.9.", [])}},
    ]
    agent = _FakeAgent(events=events, n_fail=0)
    runner = _make_runner(agent)
    fake = _FakeLLM("SHOULD NOT BE CALLED")
    with patch.object(base_mod, "make_llm", lambda _eid: fake):
        result = await runner.run_streamed("grade it", on_log=lambda *_a: None)
    assert fake.calls == 0
    assert result == "Decision: PASS. Confidence 0.9."
