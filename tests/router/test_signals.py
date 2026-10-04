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

from router.decisions import DB_ENV_VAR, MESSAGES_PATH, DecisionLog
from router.proxy import LOOPBACK_HOST, ProxySettings, create_server
from router.signals import Signals, compute_signals

SECRET_PHRASE = "purple-orchid-trombone-7731"
MODEL = "TODO_MODEL_TIER_MID"
CLIENT_TIMEOUT = 5.0


def messages_body(model: str = MODEL, phrase: str = SECRET_PHRASE) -> bytes:
    return json.dumps(
        {
            "model": model,
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": f"please summarise this: {phrase}"},
            ],
        }
    ).encode()


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
    upstream_port: int
    recorded: list[Recorded]


@contextlib.contextmanager
def running_proxy(db_path: Path | None) -> Iterator[Harness]:
    upstream = ThreadingHTTPServer((LOOPBACK_HOST, 0), RecordingUpstream)
    upstream.daemon_threads = True
    upstream.recorded = []  # type: ignore[attr-defined]
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    decisions = DecisionLog(db_path) if db_path is not None else None
    server = create_server(
        ProxySettings(
            listen_port=0,
            upstream=f"http://{LOOPBACK_HOST}:{upstream.server_address[1]}",
            mode="shadow",
            decisions=decisions,
        )
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Harness(
            proxy_port=server.server_address[1],
            upstream_port=upstream.server_address[1],
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
    connection = http.client.HTTPConnection(LOOPBACK_HOST, port, timeout=CLIENT_TIMEOUT)
    try:
        connection.request(
            "POST",
            path,
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def test_empty_conversation():
    body = {"messages": []}
    s = compute_signals(body)
    assert s.message_count == 0
    assert s.turn_index == 0


def test_string_content_messages():
    body = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]}
    s = compute_signals(body)
    assert s.message_count == 2
    assert s.turn_index == 1


def test_tool_result_errors_in_a_row():
    body = {
        "messages": [
            {"role": "assistant", "content": [
                {"type": "tool_use", "name": "edit"},
                {"type": "tool_result", "is_error": True},
                {"type": "tool_result", "is_error": True},
            ]}
        ]
    }
    s = compute_signals(body)
    assert s.tool_result_count == 2
    assert s.last_tool_result_is_error is True
    assert s.consecutive_tool_errors == 2


def test_error_followed_by_success_consecutive_zero():
    body = {
        "messages": [
            {"role": "assistant", "content": [
                {"type": "tool_result", "is_error": True},
                {"type": "tool_result", "is_error": False},
            ]}
        ]
    }
    s = compute_signals(body)
    assert s.consecutive_tool_errors == 0
    assert s.last_tool_result_is_error is False


def test_repeated_tool_names():
    body = {
        "messages": [
            {"role": "assistant", "content": [
                {"type": "tool_use", "name": "edit"},
                {"type": "tool_use", "name": "edit"},
                {"type": "tool_use", "name": "edit"},
            ]}
        ]
    }
    s = compute_signals(body)
    assert s.repeated_tool_call_count == 3


def test_thinking_enabled_disabled_absent():
    assert compute_signals({"messages": [], "thinking": {"type": "enabled"}}).has_thinking_enabled is True
    assert compute_signals({"messages": [], "thinking": {"type": "adaptive"}}).has_thinking_enabled is True
    assert compute_signals({"messages": [], "thinking": {"type": "disabled"}}).has_thinking_enabled is False
    assert compute_signals({"messages": []}).has_thinking_enabled is False


def test_effort_present_absent():
    assert compute_signals({"messages": [], "effort": "high"}).requested_effort_if_present == "high"
    assert compute_signals({"messages": [], "output_config": {"effort": "medium"}}).requested_effort_if_present == "medium"
    assert compute_signals({"messages": []}).requested_effort_if_present is None


def test_malformed_bodies_none_no_exception():
    assert compute_signals(None) == Signals()
    assert compute_signals("bad") == Signals()


def wait_for_rows(log: DecisionLog, expected: int, timeout: float = 5.0) -> int:
    import time

    deadline = time.monotonic() + timeout
    count = log.count()
    while count < expected and time.monotonic() < deadline:
        time.sleep(0.01)
        count = log.count()
    return count


def test_phrase_not_in_signals_repr_or_db_or_log(tmp_path, monkeypatch):
    body = messages_body()
    with db_at(tmp_path, monkeypatch) as log, running_proxy(log.db_path) as harness:
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            status, _ = post(harness.proxy_port, MESSAGES_PATH, body)
            assert status == 200
        assert wait_for_rows(log, 1) == 1
        db_bytes = log.db_path.read_bytes()
        logs = captured.getvalue()
        assert SECRET_PHRASE.encode() not in db_bytes
        assert SECRET_PHRASE not in logs
        row = log.recent(1)[0]
        assert SECRET_PHRASE not in str(row.signal_values)


def test_signal_values_filled_for_messages_request(tmp_path, monkeypatch):
    with db_at(tmp_path, monkeypatch) as log, running_proxy(log.db_path) as harness:
        assert post(harness.proxy_port, MESSAGES_PATH, messages_body())[0] == 200
        assert wait_for_rows(log, 1) == 1
        row = log.recent(1)[0]
        assert isinstance(row.signal_values, dict)
        assert row.signal_values
