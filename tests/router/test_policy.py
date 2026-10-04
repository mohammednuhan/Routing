from __future__ import annotations

import contextlib
import http.client
import io
import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from router.config import load_config
from router.decisions import DB_ENV_VAR, MESSAGES_PATH, DecisionLog
from router.policy import decide, Decision
from router.proxy import LOOPBACK_HOST, ProxySettings, create_server
from router.signals import Signals

SECRET_PHRASE = "purple-orchid-trombone-7731"
MODEL_LOW = "TODO_MODEL_TIER_LOW"
MODEL_MID = "TODO_MODEL_TIER_MID"
MODEL_HIGH = "TODO_MODEL_TIER_HIGH"


def make_config():
    return load_config()


def test_escalate_tool_errors():
    c = make_config()
    s = Signals(consecutive_tool_errors=2)
    d = decide(s, MODEL_MID, None, c)
    assert d.action == "SWITCH"
    assert d.target_model == MODEL_HIGH
    assert "ESCALATE_TOOL_ERRORS" in (d.reason_codes or [])


def test_escalate_repeated_tools():
    c = make_config()
    s = Signals(repeated_tool_call_count=3)
    d = decide(s, MODEL_MID, None, c)
    assert d.action == "SWITCH"
    assert d.target_model == MODEL_HIGH
    assert "ESCALATE_REPEATED_TOOLS" in (d.reason_codes or [])


def test_rule_order():
    c = make_config()
    s = Signals(consecutive_tool_errors=5, repeated_tool_call_count=5)
    d = decide(s, MODEL_MID, None, c)
    assert d.action == "SWITCH"
    assert "ESCALATE_TOOL_ERRORS" in (d.reason_codes or [])


def test_none_signals_never_switch():
    c = make_config()
    s = Signals(consecutive_tool_errors=None, context_tokens_estimate=None)
    d = decide(s, MODEL_MID, None, c)
    assert d.action == "STAY"
    assert "SIGNAL_UNKNOWN" in (d.reason_codes or [])


def test_already_highest():
    c = make_config()
    s = Signals(consecutive_tool_errors=10)
    d = decide(s, MODEL_HIGH, None, c)
    assert d.action == "STAY"
    assert "ALREADY_HIGHEST" in (d.reason_codes or [])


def test_downgrade_small_context():
    c = make_config()
    s = Signals(context_tokens_estimate=1000, turn_index=1, consecutive_tool_errors=0)
    d = decide(s, MODEL_MID, None, c)
    assert d.action == "SWITCH"
    assert d.target_model == MODEL_LOW
    assert "DOWNGRADE_SMALL_CONTEXT" in (d.reason_codes or [])


def test_already_lowest():
    c = make_config()
    s = Signals(context_tokens_estimate=10, turn_index=0, consecutive_tool_errors=0)
    d = decide(s, MODEL_LOW, None, c)
    assert d.action == "STAY"
    assert "ALREADY_LOWEST" in (d.reason_codes or [])


def test_unknown_model():
    c = make_config()
    s = Signals(consecutive_tool_errors=100)
    d = decide(s, "UNKNOWN", None, c)
    assert d.action == "STAY"
    assert "UNKNOWN_MODEL" in (d.reason_codes or [])


def test_illegal_effort_pair():
    c = make_config()
    s = Signals(consecutive_tool_errors=2)
    d = decide(s, MODEL_MID, "NOT_LEGAL", c)
    assert d.action == "STAY"
    assert "ILLEGAL_PAIR" in (d.reason_codes or [])


def test_mode_off():
    c2 = make_config()
    object.__setattr__(c2, "mode", "off")
    s = Signals(consecutive_tool_errors=100)
    d = decide(s, MODEL_MID, None, c2)
    assert d.action == "STAY"
    assert "MODE_OFF" in (d.reason_codes or [])


def test_determinism():
    c = make_config()
    s = Signals(consecutive_tool_errors=2)
    d1 = decide(s, MODEL_MID, None, c)
    d2 = decide(s, MODEL_MID, None, c)
    assert d1 == d2


@dataclass
class Recorded:
    method: str
    path: str
    body: bytes


class RecordingUpstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "recording-upstream"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _handle(self) -> None:
        raw_length = self.headers.get("Content-Length")
        length = int(raw_length) if raw_length else 0
        body = self.rfile.read(length) if length else b""
        self.server.recorded.append(Recorded(self.command, self.path, body))
        payload = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    do_GET = _handle
    do_POST = _handle


@dataclass
class Harness:
    proxy_port: int
    recorded: list[Recorded]


@contextlib.contextmanager
def running_proxy(db_path: Path | None) -> Iterator[Harness]:
    upstream = ThreadingHTTPServer((LOOPBACK_HOST, 0), RecordingUpstream)
    upstream.daemon_threads = True
    upstream.recorded = []  # type: ignore[attr-defined]
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    decisions = DecisionLog(db_path) if db_path is not None else None
    cfg = load_config()
    server = create_server(
        ProxySettings(
            listen_port=0,
            upstream=f"http://{LOOPBACK_HOST}:{upstream.server_address[1]}",
            mode="shadow",
            decisions=decisions,
            config=cfg,
        )
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Harness(
            proxy_port=server.server_address[1],
            recorded=upstream.recorded,  # type: ignore[attr-defined]
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)


@contextlib.contextmanager
def db_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[DecisionLog]:
    monkeypatch.setenv(DB_ENV_VAR, str(tmp_path / "router.sqlite3"))
    yield DecisionLog.from_env()


def post(port: int, path: str, body: bytes) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection(LOOPBACK_HOST, port, timeout=5.0)
    try:
        connection.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def wait_for_rows(log: DecisionLog, expected: int, timeout: float = 5.0) -> int:
    import time

    deadline = time.monotonic() + timeout
    count = log.count()
    while count < expected and time.monotonic() < deadline:
        time.sleep(0.01)
        count = log.count()
    return count


def test_proxy_switch_applied_zero_and_bytes_identical(tmp_path, monkeypatch):
    body = json.dumps({
        "model": MODEL_MID,
        "messages": [
            {"role": "assistant", "content": [
                {"type": "tool_result", "is_error": True},
                {"type": "tool_result", "is_error": True},
            ]}
        ]
    }).encode()
    with db_at(tmp_path, monkeypatch) as log, running_proxy(log.db_path) as harness:
        status, payload = post(harness.proxy_port, MESSAGES_PATH, body)
        assert status == 200
        assert json.loads(payload) == {"ok": True}
        assert wait_for_rows(log, 1) == 1
        row = log.recent(1)[0]
        assert row.applied == 0
        assert row.action == "SWITCH"
        assert harness.recorded[-1].body == body


def _two_tool_errors_with_effort(effort: str) -> bytes:
    return json.dumps({
        "model": MODEL_MID,
        "effort": effort,
        "messages": [
            {"role": "assistant", "content": [
                {"type": "tool_result", "is_error": True},
                {"type": "tool_result", "is_error": True},
            ]}
        ]
    }).encode()


def test_proxy_carries_requested_effort_into_decision(tmp_path, monkeypatch):
    effort = "TODO_EFFORT_MEDIUM"
    body = _two_tool_errors_with_effort(effort)
    with db_at(tmp_path, monkeypatch) as log, running_proxy(log.db_path) as harness:
        status, _ = post(harness.proxy_port, MESSAGES_PATH, body)
        assert status == 200
        assert wait_for_rows(log, 1) == 1
        row = log.recent(1)[0]
        assert row.action == "SWITCH"
        assert row.chosen_model == MODEL_HIGH
        assert row.chosen_effort == effort
        assert row.applied == 0
        assert harness.recorded[-1].body == body


def test_proxy_illegal_effort_stays(tmp_path, monkeypatch):
    body = _two_tool_errors_with_effort("NOT_LEGAL")
    with db_at(tmp_path, monkeypatch) as log, running_proxy(log.db_path) as harness:
        status, _ = post(harness.proxy_port, MESSAGES_PATH, body)
        assert status == 200
        assert wait_for_rows(log, 1) == 1
        row = log.recent(1)[0]
        assert row.action == "STAY"
        assert "ILLEGAL_PAIR" in row.reason_codes
        assert row.applied == 0
        assert harness.recorded[-1].body == body
