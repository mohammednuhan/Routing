from __future__ import annotations

import contextlib
import http.client
import io
import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from router.decisions import (
    DB_ENV_VAR,
    MESSAGES_PATH,
    UNKNOWN_SESSION,
    DecisionLog,
    forbidden_columns,
)
from router.proxy import LOOPBACK_HOST, ProxySettings, create_server

#: A phrase that must never reach the database or any log line.
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
                {"role": "assistant", "content": "sure"},
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
    thread: threading.Thread


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
            thread=thread,
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
            headers={"Content-Type": "application/json", "x-api-key": "fake-key-DO-NOT-LOG"},
        )
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def get(port: int, path: str) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection(LOOPBACK_HOST, port, timeout=CLIENT_TIMEOUT)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def wait_for_rows(log: DecisionLog, expected: int, timeout: float = 5.0) -> int:
    """The row is written after the response is flushed, so it can trail it."""
    deadline = time.monotonic() + timeout
    count = log.count()
    while count < expected and time.monotonic() < deadline:
        time.sleep(0.01)
        count = log.count()
    return count


def wait_for_text(stream: io.StringIO, fragment: str, timeout: float = 5.0) -> str:
    """The proxy logs after the response is flushed, so wait for the line."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = stream.getvalue()
        if fragment in text:
            return text
        time.sleep(0.01)
    return stream.getvalue()


def test_one_row_per_messages_post_and_none_for_other_paths(tmp_path, monkeypatch):
    with db_at(tmp_path, monkeypatch) as log, running_proxy(log.db_path) as harness:
        logged = "TODO_MODEL_TIER_LOW"
        assert post(harness.proxy_port, MESSAGES_PATH, messages_body(model=logged))[0] == 200
        assert (
            post(
                harness.proxy_port,
                f"{MESSAGES_PATH}?beta=true",
                messages_body(model="TODO_MODEL_TIER_HIGH"),
            )[0]
            == 200
        )
        assert get(harness.proxy_port, MESSAGES_PATH)[0] == 200
        assert post(harness.proxy_port, "/v1/complete", messages_body(model="MODEL_OTHER_1"))[0] == 200
        assert post(harness.proxy_port, "/other", messages_body(model="MODEL_OTHER_2"))[0] == 200

        assert wait_for_rows(log, 2) == 2
        rows = log.recent(10)
        assert sorted(row.requested_model or "" for row in rows) == [
            "TODO_MODEL_TIER_HIGH",
            "TODO_MODEL_TIER_LOW",
        ]
        assert all(row.requested_model != "MODEL_OTHER_1" for row in rows)
        assert all(row.requested_model != "MODEL_OTHER_2" for row in rows)


def test_requested_and_chosen_model_are_captured(tmp_path, monkeypatch):
    with db_at(tmp_path, monkeypatch) as log, running_proxy(log.db_path) as harness:
        assert post(harness.proxy_port, MESSAGES_PATH, messages_body(model=MODEL))[0] == 200
        assert wait_for_rows(log, 1) == 1

        row = log.recent(1)[0]
        assert row.requested_model == MODEL
        assert row.chosen_model == MODEL
        assert row.chosen_effort is None
        assert row.action == "STAY"
        assert row.applied == 0
        assert row.mode == "shadow"
        assert row.reason_codes == ["PASSTHROUGH"]
        assert isinstance(row.signal_values, dict)
        assert row.error is None
        assert row.timestamp.endswith("Z")


def test_prompt_text_never_reaches_the_database_or_the_log(tmp_path, monkeypatch):
    with db_at(tmp_path, monkeypatch) as log, running_proxy(log.db_path) as harness:
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            assert post(harness.proxy_port, MESSAGES_PATH, messages_body())[0] == 200
            assert wait_for_rows(log, 1) == 1
            logs = wait_for_text(captured, "POST /v1/messages 200")

        database_bytes = log.db_path.read_bytes()
        assert database_bytes, "the database file was not written"
        assert SECRET_PHRASE.encode() not in database_bytes
        assert log.salt_path.read_bytes()
        assert SECRET_PHRASE.encode() not in log.salt_path.read_bytes()
        assert SECRET_PHRASE not in logs

        row = log.recent(1)[0]
        assert len(row.session_hint) == 16
        assert row.session_hint != UNKNOWN_SESSION
        assert all(character in "0123456789abcdef" for character in row.session_hint)


def test_forwarded_body_bytes_are_identical_to_received_bytes(tmp_path, monkeypatch):
    body = messages_body()
    with db_at(tmp_path, monkeypatch) as log, running_proxy(log.db_path) as harness:
        assert post(harness.proxy_port, MESSAGES_PATH, body)[0] == 200
        assert wait_for_rows(log, 1) == 1

        forwarded = harness.recorded[-1]
        assert forwarded.method == "POST"
        assert forwarded.path == MESSAGES_PATH
        assert forwarded.body == body


def test_database_failure_does_not_break_the_request(tmp_path, monkeypatch):
    blocker = tmp_path / "not-a-directory"
    blocker.write_bytes(b"this is a file, not a directory")
    monkeypatch.setenv(DB_ENV_VAR, str(blocker / "router.sqlite3"))
    log = DecisionLog.from_env()

    with running_proxy(log.db_path) as harness:
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            status, payload = post(harness.proxy_port, MESSAGES_PATH, messages_body())
            logs = wait_for_text(captured, "decision log write failed")

        assert status == 200
        assert json.loads(payload) == {"ok": True}
        assert harness.recorded[-1].body == messages_body()

        warnings = [
            line for line in logs.splitlines() if "decision log write failed" in line
        ]
        assert len(warnings) == 1
        assert SECRET_PHRASE not in logs

        assert post(harness.proxy_port, MESSAGES_PATH, messages_body())[0] == 200
        assert harness.thread.is_alive()


def test_same_first_message_gives_same_hint_and_increasing_index(tmp_path, monkeypatch):
    with db_at(tmp_path, monkeypatch) as log, running_proxy(log.db_path) as harness:
        for _ in range(3):
            assert post(harness.proxy_port, MESSAGES_PATH, messages_body())[0] == 200
        assert wait_for_rows(log, 3) == 3

        assert post(harness.proxy_port, MESSAGES_PATH, messages_body(phrase="other"))[0] == 200
        assert wait_for_rows(log, 4) == 4

        newest_first = log.recent(10)
        assert len(newest_first) == 4

        by_hint: dict[str, list[int]] = {}
        for row in newest_first:
            by_hint.setdefault(row.session_hint, []).append(row.request_index_in_session)

        assert len(by_hint) == 2
        for hint, seen in by_hint.items():
            assert hint != UNKNOWN_SESSION
            assert sorted(seen) == list(range(1, len(seen) + 1))


def test_unparsable_body_is_forwarded_and_the_problem_is_recorded(tmp_path, monkeypatch):
    body = b"{not json at all"
    with db_at(tmp_path, monkeypatch) as log, running_proxy(log.db_path) as harness:
        assert post(harness.proxy_port, MESSAGES_PATH, body)[0] == 200
        assert wait_for_rows(log, 1) == 1

        row = log.recent(1)[0]
        assert row.error == "request_body_not_json"
        assert row.requested_model is None
        assert row.chosen_model is None
        assert row.session_hint == UNKNOWN_SESSION
        assert harness.recorded[-1].body == body


def test_schema_has_no_content_columns(tmp_path, monkeypatch):
    with db_at(tmp_path, monkeypatch) as log, running_proxy(log.db_path) as harness:
        assert post(harness.proxy_port, MESSAGES_PATH, messages_body())[0] == 200
        assert wait_for_rows(log, 1) == 1

        columns = log.column_names()
        assert forbidden_columns(columns) == []
        assert columns == [
            "decision_id",
            "timestamp",
            "session_hint",
            "request_index_in_session",
            "requested_model",
            "chosen_model",
            "chosen_effort",
            "mode",
            "reason_codes",
            "signal_values",
            "action",
            "applied",
            "error",
        ]
