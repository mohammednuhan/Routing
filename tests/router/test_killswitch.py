"""Kill switch, circuit breaker and per-session switch cap.

Three separate guards, all of which can only stop the router from acting:

* the kill switch, checked on every request, from an env var or a flag file;
* the circuit breaker, which opens after too many internal router errors;
* the per-session switch cap, a safety check like dwell and hysteresis.

Everything here runs through the same in-process harness the other proxy tests
use, plus direct calls into the pure functions. No server is started by hand and
the router is never run as a subprocess. The breaker takes an injected clock, so
its cooldown is tested without sleeping.
"""
from __future__ import annotations

import contextlib
import http.client
import json
import re
import threading
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from router import proxy as proxy_module
from router.breaker import (
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_THRESHOLD,
    INTERNAL_ERROR_CLASSES,
    INTERNAL_ERROR_REASONS,
    REASON_CIRCUIT_OPEN,
    CircuitBreaker,
    is_internal_error,
)
from router.config import (
    ConfigError,
    RouterConfig,
    load_config,
    parse_listen,
)
from router.decisions import DB_ENV_VAR, MESSAGES_PATH, DecisionLog
from router.killswitch import (
    KILL_SWITCH_ENV,
    KILL_SWITCH_FILENAME,
    REASON_KILL_SWITCH,
    KillSwitch,
    kill_switch_path,
    read_kill_switch,
    write_kill_switch,
)
from router.policy import Decision
from router.proxy import LOOPBACK_HOST, ProxyError, ProxySettings, create_server
from router.safety import (
    BLOCKED_DWELL,
    BLOCKED_SWITCH_CAP,
    apply_safety,
    is_blocked,
)
from router.signals import Signals
from router.state import UP, SessionState

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_CONFIG_PATH = REPO_ROOT / "tools" / "sample-config-ACTIVE-test.yaml"
DUMMY_CONFIG_PATH = REPO_ROOT / "tools" / "sample-config-DUMMY-prices.yaml"

MODEL_LOW = "test-low"
MODEL_MID = "test-mid"
MODEL_HIGH = "test-high"

#: A phrase that must never reach the database or any log line.
SECRET_PHRASE = "purple-orchid-trombone-7731"

CLIENT_TIMEOUT = 5.0

#: One session for every request in a session test: the hint is derived from the
#: first user message, so the text must not change.
SESSION_TEXT = "one long session"

#: Dummy benefit USD 0.001 covers a rebuild below ~289 tokens, so bodies are kept
#: well under 1156 bytes and an escalation is never quietly blocked on cost.
CROSSOVER_BYTES = 289 * 4


def active_config() -> RouterConfig:
    return load_config(ACTIVE_CONFIG_PATH)


def dummy_shadow_config() -> RouterConfig:
    return load_config(DUMMY_CONFIG_PATH)


def session_config(**policy_overrides: Any) -> RouterConfig:
    """The active config with a policy tuned for a many-switch session."""
    return replace(active_config(), policy=replace(active_config().policy, **policy_overrides))


def switch_body(model: str = MODEL_MID) -> bytes:
    """A body that escalates and is small enough for the dummy benefit to cover.

    The first user message is the same in every body, so requests that share it
    are one session.
    """
    body = json.dumps(
        {
            "model": model,
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": SESSION_TEXT},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "t1", "name": "search"}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "is_error": True},
                        {"type": "tool_result", "tool_use_id": "t2", "is_error": True},
                    ],
                },
            ],
        }
    ).encode()
    assert len(body) < CROSSOVER_BYTES, "switch_body must stay cheap enough to be allowed"
    return body


def stay_body(model: str = MODEL_MID) -> bytes:
    """A body no rule acts on: one tool error, no repeat, one turn."""
    return json.dumps(
        {
            "model": model,
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": SESSION_TEXT},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "t1", "name": "search"}],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": True}],
                },
            ],
        }
    ).encode()


@dataclass
class Recorded:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes


class RecordingUpstream(BaseHTTPRequestHandler):
    """Records what arrived and replies with a fixed JSON body."""

    protocol_version = "HTTP/1.1"
    server_version = "recording-upstream"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _handle(self) -> None:
        raw_length = self.headers.get("Content-Length")
        length = int(raw_length) if raw_length else 0
        body = self.rfile.read(length) if length else b""
        self.server.recorded.append(  # type: ignore[attr-defined]
            Recorded(
                method=self.command,
                path=self.path,
                headers={name.lower(): value for name, value in self.headers.items()},
                body=body,
            )
        )
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
    db_path: Path

    @property
    def forwarded(self) -> Recorded:
        assert self.recorded, "the upstream received nothing"
        return self.recorded[-1]

    @property
    def flag_path(self) -> Path:
        return kill_switch_path(self.db_path)


@contextlib.contextmanager
def running_proxy(
    config: RouterConfig,
    db_path: Path,
    breaker: CircuitBreaker | None = None,
    state: Any = "unset",
    upstream: str | None = None,
) -> Iterator[Harness]:
    from router.state import SessionStore

    server_up: ThreadingHTTPServer | None = None
    upstream_thread: threading.Thread | None = None
    if upstream is None:
        server_up = ThreadingHTTPServer((LOOPBACK_HOST, 0), RecordingUpstream)
        server_up.daemon_threads = True
        server_up.recorded = []  # type: ignore[attr-defined]
        upstream_thread = threading.Thread(target=server_up.serve_forever, daemon=True)
        upstream_thread.start()
        upstream = f"http://{LOOPBACK_HOST}:{server_up.server_address[1]}"

    decisions = DecisionLog(db_path)
    settings = ProxySettings(
        listen_port=0,
        upstream=upstream,
        mode=config.mode,
        decisions=decisions,
        config=config,
        state=SessionStore() if state == "unset" else state,
    )
    if breaker is not None:
        settings = replace(settings, breaker=breaker)
    server = create_server(settings)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    recorded: list[Recorded] = getattr(server_up, "recorded", [])
    try:
        yield Harness(
            proxy_port=server.server_address[1],
            recorded=recorded,
            db_path=db_path,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if server_up is not None:
            server_up.shutdown()
            server_up.server_close()
        if upstream_thread is not None:
            upstream_thread.join(timeout=5)


def closed_port() -> str:
    """An upstream URL on a loopback port that nothing is listening on."""
    import socket

    with socket.socket() as probe:
        probe.bind((LOOPBACK_HOST, 0))
        port = int(probe.getsockname()[1])
    return f"http://{LOOPBACK_HOST}:{port}"


def log_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DecisionLog:
    monkeypatch.setenv(DB_ENV_VAR, str(tmp_path / "router.sqlite3"))
    return DecisionLog.from_env()


def post_once(port: int, body: bytes) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection(LOOPBACK_HOST, port, timeout=CLIENT_TIMEOUT)
    try:
        connection.request(
            "POST",
            MESSAGES_PATH,
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = response.read()
        return (
            response.status,
            {name.lower(): value for name, value in response.getheaders()},
            payload,
        )
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


class FakeClock:
    """A monotonic clock the test moves by hand."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# --- the kill switch, read as a value ---------------------------------------


def test_the_switch_is_off_when_nothing_is_set(tmp_path):
    switch = read_kill_switch(tmp_path / "router.sqlite3", env={})

    assert switch.on is False
    assert switch.label == "OFF"
    assert switch.reasons == ()


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", " 1 ", "ON"])
def test_the_env_var_turns_it_on(tmp_path, value):
    switch = read_kill_switch(tmp_path / "router.sqlite3", env={KILL_SWITCH_ENV: value})

    assert switch.on is True
    assert KILL_SWITCH_ENV in switch.detail


@pytest.mark.parametrize("value", ["0", "false", "", "  ", "maybe"])
def test_any_other_env_value_leaves_it_off(tmp_path, value):
    switch = read_kill_switch(tmp_path / "router.sqlite3", env={KILL_SWITCH_ENV: value})

    assert switch.on is False


def test_the_flag_file_turns_it_on(tmp_path):
    db_path = tmp_path / "router.sqlite3"
    write_kill_switch(db_path, True)

    switch = read_kill_switch(db_path, env={})

    assert switch.on is True
    assert KILL_SWITCH_FILENAME in switch.detail


def test_the_flag_file_sits_beside_the_database(tmp_path):
    db_path = tmp_path / "nested" / "router.sqlite3"

    path = write_kill_switch(db_path, True)

    assert path == db_path.parent / KILL_SWITCH_FILENAME
    assert path.is_file()
    assert read_kill_switch(db_path, env={}).on is True


def test_the_default_database_decides_where_the_flag_file_lives(tmp_path, monkeypatch):
    monkeypatch.setenv(DB_ENV_VAR, str(tmp_path / "router.sqlite3"))

    path = write_kill_switch(None, True)

    assert path == tmp_path / KILL_SWITCH_FILENAME
    assert read_kill_switch(env={}).on is True

    write_kill_switch(None, False)
    assert read_kill_switch(env={}).on is False


def test_the_flag_file_and_the_env_var_are_independent(tmp_path):
    db_path = tmp_path / "router.sqlite3"

    write_kill_switch(db_path, True)
    assert read_kill_switch(db_path, env={}).on is True

    write_kill_switch(db_path, False)
    assert read_kill_switch(db_path, env={}).on is False
    assert read_kill_switch(db_path, env={KILL_SWITCH_ENV: "1"}).on is True


def test_removing_a_flag_file_that_is_not_there_is_not_an_error(tmp_path):
    path = write_kill_switch(tmp_path / "router.sqlite3", False)

    assert not path.exists()
    assert read_kill_switch(tmp_path / "router.sqlite3", env={}).on is False


def test_both_reasons_are_reported_when_both_are_set(tmp_path):
    db_path = tmp_path / "router.sqlite3"
    write_kill_switch(db_path, True)

    switch = read_kill_switch(db_path, env={KILL_SWITCH_ENV: "1"})

    assert switch.on is True
    assert len(switch.reasons) == 2


# --- the kill switch, through the proxy ------------------------------------


def test_the_env_var_forwards_everything_untouched(tmp_path, monkeypatch):
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    log = log_at(tmp_path, monkeypatch)
    body = switch_body()

    with running_proxy(active_config(), log.db_path) as harness:
        monkeypatch.setenv(KILL_SWITCH_ENV, "1")
        status, headers, payload = post_once(harness.proxy_port, body)

        assert status == 200
        assert json.loads(payload) == {"ok": True}
        assert wait_for_rows(log, 1) == 1

        assert harness.forwarded.body == body
        assert "x-tamias-routed" not in headers

        row = log.recent(1)[0]
        assert row.reason_codes == [REASON_KILL_SWITCH]
        assert row.action == "STAY"
        assert row.applied == 0
        assert row.mode == "off"


def test_the_env_var_is_seen_again_when_it_is_cleared(tmp_path, monkeypatch):
    """No restart: the same proxy reads the environment on every request."""
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    log = log_at(tmp_path, monkeypatch)
    body = switch_body()

    with running_proxy(active_config(), log.db_path) as harness:
        monkeypatch.setenv(KILL_SWITCH_ENV, "1")
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 1) == 1
        assert log.recent(1)[0].reason_codes == [REASON_KILL_SWITCH]

        monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 2) == 2

        assert json.loads(harness.forwarded.body)["model"] == MODEL_HIGH
        row = log.recent(1)[0]
        assert row.reason_codes == ["ESCALATE_TOOL_ERRORS"]
        assert REASON_KILL_SWITCH not in row.reason_codes
        assert row.applied == 1


def test_the_flag_file_turns_it_off_and_back_on_without_a_restart(tmp_path, monkeypatch):
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    log = log_at(tmp_path, monkeypatch)
    body = switch_body()

    with running_proxy(active_config(), log.db_path) as harness:
        # off
        harness.flag_path.write_text("off\n", encoding="utf-8")
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 1) == 1
        assert harness.forwarded.body == body
        assert log.recent(1)[0].reason_codes == [REASON_KILL_SWITCH]
        assert log.recent(1)[0].applied == 0

        # on again
        harness.flag_path.unlink()
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 2) == 2
        assert json.loads(harness.forwarded.body)["model"] == MODEL_HIGH
        assert log.recent(1)[0].applied == 1


def test_the_flag_file_is_written_by_the_same_helper_the_cli_uses(tmp_path, monkeypatch):
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    log = log_at(tmp_path, monkeypatch)
    body = switch_body()

    with running_proxy(active_config(), log.db_path) as harness:
        write_kill_switch(log.db_path, True)
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 1) == 1
        assert harness.forwarded.body == body
        assert log.recent(1)[0].reason_codes == [REASON_KILL_SWITCH]


def test_the_switch_overrides_active_mode(tmp_path, monkeypatch):
    """The control case first: without the switch, this body IS rewritten."""
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    log = log_at(tmp_path, monkeypatch)
    body = switch_body()

    with running_proxy(active_config(), log.db_path) as harness:
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 1) == 1
        assert json.loads(harness.forwarded.body)["model"] == MODEL_HIGH
        assert log.recent(1)[0].applied == 1

        monkeypatch.setenv(KILL_SWITCH_ENV, "1")
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 2) == 2

        assert harness.forwarded.body == body
        row = log.recent(1)[0]
        assert row.action == "STAY"
        assert row.applied == 0
        assert row.requested_model == MODEL_MID
        assert row.chosen_model == MODEL_MID
        assert row.reason_codes == [REASON_KILL_SWITCH]


def test_the_switch_leaves_a_body_that_needed_no_switch_alone_too(tmp_path, monkeypatch):
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    log = log_at(tmp_path, monkeypatch)
    body = stay_body()

    with running_proxy(active_config(), log.db_path) as harness:
        monkeypatch.setenv(KILL_SWITCH_ENV, "1")
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 1) == 1

        assert harness.forwarded.body == body
        row = log.recent(1)[0]
        assert row.applied == 0
        assert row.reason_codes == [REASON_KILL_SWITCH]


def test_nothing_changes_while_the_switch_is_off(tmp_path, monkeypatch):
    """The guard is inert when it is off: rows, models and bytes are as before."""
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(active_config(), log.db_path) as harness:
        assert not harness.flag_path.exists()

        post_once(harness.proxy_port, switch_body())
        assert wait_for_rows(log, 1) == 1
        switched = log.recent(1)[0]
        assert switched.applied == 1
        assert switched.reason_codes == ["ESCALATE_TOOL_ERRORS"]
        assert switched.mode == "active"

        # A different first user message is a different session, so nothing
        # approved above can hold this one: it must still reach NO_RULE_MATCHED.
        elsewhere = json.loads(stay_body().decode())
        elsewhere["messages"][0]["content"] = "a different session entirely"

        post_once(harness.proxy_port, json.dumps(elsewhere).encode())
        assert wait_for_rows(log, 2) == 2
        stayed = log.recent(1)[0]
        assert stayed.applied == 0
        assert stayed.reason_codes == ["NO_RULE_MATCHED"]
        assert REASON_KILL_SWITCH not in stayed.reason_codes
        assert REASON_CIRCUIT_OPEN not in stayed.reason_codes


def test_the_switch_still_logs_metadata_and_no_request_text(tmp_path, monkeypatch):
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    log = log_at(tmp_path, monkeypatch)
    body = switch_body()

    with running_proxy(active_config(), log.db_path) as harness:
        monkeypatch.setenv(KILL_SWITCH_ENV, "1")
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 1) == 1

        assert SECRET_PHRASE.encode() not in log.db_path.read_bytes()
        row = log.recent(1)[0]
        assert row.requested_model == MODEL_MID
        assert row.session_hint not in (None, "")


def test_a_switch_on_one_install_leaves_another_alone(tmp_path, monkeypatch):
    """A temporary ROUTER_DB isolates the flag file."""
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    first = log_at(tmp_path, monkeypatch)
    second = DecisionLog(tmp_path / "other" / "router.sqlite3")

    write_kill_switch(first.db_path, True)

    assert read_kill_switch(first.db_path, env={}).on is True
    assert read_kill_switch(second.db_path, env={}).on is False


# --- the circuit breaker, as a value ---------------------------------------


def test_the_breaker_defaults_match_the_config_defaults():
    assert DEFAULT_THRESHOLD == 3
    assert DEFAULT_COOLDOWN_SECONDS == 60


def test_the_breaker_starts_closed():
    breaker = CircuitBreaker(threshold=3, cooldown_seconds=60, clock=FakeClock())

    assert breaker.is_open() is False
    assert breaker.consecutive == 0


def test_the_breaker_opens_on_the_kth_consecutive_error():
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=3, cooldown_seconds=60, clock=clock)

    assert breaker.record(True) is False
    assert breaker.record(True) is False
    assert breaker.record(True) is True
    assert breaker.is_open() is True


def test_a_clean_request_resets_the_count():
    breaker = CircuitBreaker(threshold=3, cooldown_seconds=60, clock=FakeClock())

    breaker.record(True)
    breaker.record(True)
    breaker.record(False)
    assert breaker.consecutive == 0

    breaker.record(True)
    breaker.record(True)
    assert breaker.is_open() is False
    assert breaker.record(True) is True


def test_it_stays_open_for_the_whole_cooldown():
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=1, cooldown_seconds=60, clock=clock)
    breaker.record(True)

    clock.advance(59.999)
    assert breaker.is_open() is True
    clock.advance(0.001)
    assert breaker.is_open() is False


def test_reaching_the_threshold_twice_needs_two_full_runs():
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=2, cooldown_seconds=60, clock=clock)

    breaker.record(True)
    assert breaker.record(True) is True
    clock.advance(60)
    assert breaker.is_open() is False

    breaker.record(True)
    assert breaker.is_open() is False, "the count restarts after a cooldown"
    assert breaker.record(True) is True


def test_the_breaker_rejects_nonsense_settings():
    with pytest.raises(ValueError):
        CircuitBreaker(threshold=0)
    with pytest.raises(ValueError):
        CircuitBreaker(cooldown_seconds=0)


@pytest.mark.parametrize("error", sorted(INTERNAL_ERROR_CLASSES))
def test_the_listed_error_classes_are_internal(error):
    assert is_internal_error(error, None) is True


@pytest.mark.parametrize("reason", sorted(INTERNAL_ERROR_REASONS))
def test_the_listed_reason_codes_are_internal(reason):
    assert is_internal_error(None, [reason]) is True


@pytest.mark.parametrize("error", ["upstream_unreachable", "upstream_timeout", "router_plan_failed"])
def test_transport_errors_are_not_internal(error):
    assert is_internal_error(error, None) is False


def test_a_clean_row_is_not_internal():
    assert is_internal_error(None, ["ESCALATE_TOOL_ERRORS"]) is False


def test_every_proxy_gets_a_breaker_built_from_the_config():
    config = session_config(breaker_error_threshold=7, breaker_cooldown_seconds=9)
    settings = ProxySettings(listen_port=0, upstream="http://127.0.0.1:1", config=config)

    server = create_server(settings)
    try:
        assert settings.breaker is None, "the caller's settings are not mutated"
        assert server.settings.breaker is not None
        assert server.settings.breaker.threshold == 7
        assert server.settings.breaker.cooldown_seconds == 9
    finally:
        server.server_close()


def test_a_breaker_passed_in_is_the_one_used():
    breaker = CircuitBreaker(threshold=2, cooldown_seconds=5, clock=FakeClock())
    settings = ProxySettings(listen_port=0, upstream="http://127.0.0.1:1", breaker=breaker)

    server = create_server(settings)
    try:
        assert server.settings.breaker is breaker
    finally:
        server.server_close()


# --- the circuit breaker, through the proxy --------------------------------


def break_signals(*args: Any, **kwargs: Any) -> Signals:
    raise RuntimeError("signals exploded")


def test_the_breaker_opens_after_k_internal_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(proxy_module, "compute_signals", break_signals)
    log = log_at(tmp_path, monkeypatch)
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=2, cooldown_seconds=60, clock=clock)
    body = switch_body()

    with running_proxy(active_config(), log.db_path, breaker=breaker) as harness:
        post_once(harness.proxy_port, body)
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 2) == 2

        assert breaker.is_open() is True
        assert [row.error for row in log.recent(2)] == ["signals_failed", "signals_failed"]
        assert log.recent(2)[0].reason_codes == ["PASSTHROUGH"]

        # the third request is passed through without a decision
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 3) == 3

        assert harness.forwarded.body == body
        row = log.recent(1)[0]
        assert row.reason_codes == [REASON_CIRCUIT_OPEN]
        assert row.action == "STAY"
        assert row.applied == 0
        assert row.mode == "off"
        assert row.error is None


def test_the_open_circuit_stays_open_for_the_cooldown(tmp_path, monkeypatch):
    monkeypatch.setattr(proxy_module, "compute_signals", break_signals)
    log = log_at(tmp_path, monkeypatch)
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=1, cooldown_seconds=60, clock=clock)
    body = switch_body()

    with running_proxy(active_config(), log.db_path, breaker=breaker) as harness:
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 1) == 1
        assert breaker.is_open() is True

        for index in range(2, 5):
            clock.advance(10.0)
            post_once(harness.proxy_port, body)
            assert wait_for_rows(log, index) == index
            assert log.recent(1)[0].reason_codes == [REASON_CIRCUIT_OPEN]

        clock.advance(9.0)
        assert breaker.is_open() is True
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 5) == 5
        assert log.recent(1)[0].reason_codes == [REASON_CIRCUIT_OPEN]


def test_the_circuit_closes_after_the_cooldown_and_routing_resumes(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=1, cooldown_seconds=60, clock=clock)
    body = switch_body()

    healthy = proxy_module.compute_signals
    monkeypatch.setattr(proxy_module, "compute_signals", break_signals)

    with running_proxy(active_config(), log.db_path, breaker=breaker) as harness:
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 1) == 1
        assert breaker.is_open() is True

        # the router is healthy again, but only after the cooldown
        monkeypatch.setattr(proxy_module, "compute_signals", healthy)
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 2) == 2
        assert log.recent(1)[0].reason_codes == [REASON_CIRCUIT_OPEN]

        clock.advance(60.0)
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 3) == 3

        assert json.loads(harness.forwarded.body)["model"] == MODEL_HIGH
        row = log.recent(1)[0]
        assert row.applied == 1
        assert row.reason_codes == ["ESCALATE_TOOL_ERRORS"]
        assert breaker.is_open() is False
        assert breaker.consecutive == 0


def test_a_clean_request_resets_the_breaker_count(tmp_path, monkeypatch):
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    log = log_at(tmp_path, monkeypatch)
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=3, cooldown_seconds=60, clock=clock)
    healthy = proxy_module.compute_signals

    with running_proxy(session_config(dwell_requests=1), log.db_path, breaker=breaker) as harness:
        for _ in range(2):
            monkeypatch.setattr(proxy_module, "compute_signals", break_signals)
            post_once(harness.proxy_port, switch_body())
        assert breaker.consecutive == 2
        assert breaker.is_open() is False

        monkeypatch.setattr(proxy_module, "compute_signals", healthy)
        post_once(harness.proxy_port, stay_body())
        assert wait_for_rows(log, 3) == 3
        assert breaker.consecutive == 0

        for _ in range(2):
            monkeypatch.setattr(proxy_module, "compute_signals", break_signals)
            post_once(harness.proxy_port, switch_body())
        assert breaker.consecutive == 2, "two errors after a reset must not open it"
        assert breaker.is_open() is False


def test_the_breaker_does_not_count_upstream_failures(tmp_path, monkeypatch):
    """A dead upstream is the router working correctly, not the router failing."""
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    log = log_at(tmp_path, monkeypatch)
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=2, cooldown_seconds=60, clock=clock)

    with running_proxy(
        session_config(dwell_requests=1), log.db_path, breaker=breaker, upstream=closed_port()
    ) as harness:
        for index in range(1, 4):
            status, _, _ = post_once(harness.proxy_port, switch_body())
            assert status == 502
            assert wait_for_rows(log, index) == index

        assert [row.error for row in log.recent(3)] == ["upstream_unreachable"] * 3
        assert breaker.consecutive == 0
        assert breaker.is_open() is False
        newest = log.recent(1)[0]
        assert is_internal_error(newest.error, newest.reason_codes) is False


def test_safety_errors_count_towards_the_breaker(tmp_path, monkeypatch):
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    log = log_at(tmp_path, monkeypatch)
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=2, cooldown_seconds=60, clock=clock)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("safety exploded")

    monkeypatch.setattr(proxy_module, "apply_safety", boom)

    with running_proxy(active_config(), log.db_path, breaker=breaker) as harness:
        post_once(harness.proxy_port, switch_body())
        post_once(harness.proxy_port, switch_body())
        assert wait_for_rows(log, 2) == 2

        assert is_internal_error("safety_failed", ["SAFETY_ERROR"]) is True
        assert breaker.is_open() is True
        assert log.recent(2)[1].reason_codes == ["SAFETY_ERROR"]


def test_policy_errors_count_towards_the_breaker(tmp_path, monkeypatch):
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    log = log_at(tmp_path, monkeypatch)
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=1, cooldown_seconds=60, clock=clock)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("policy exploded")

    monkeypatch.setattr(proxy_module, "decide", boom)

    with running_proxy(active_config(), log.db_path, breaker=breaker) as harness:
        post_once(harness.proxy_port, switch_body())
        assert wait_for_rows(log, 1) == 1

        row = log.recent(1)[0]
        assert row.error == "policy_failed"
        assert breaker.is_open() is True


def test_rewrite_failures_count_towards_the_breaker(tmp_path, monkeypatch):
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    log = log_at(tmp_path, monkeypatch)
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=3, cooldown_seconds=60, clock=clock)

    def boom(*args: Any, **kwargs: Any) -> bytes:
        raise RuntimeError("rewrite exploded")

    monkeypatch.setattr(proxy_module, "rewrite_model", boom)

    with running_proxy(
        session_config(dwell_requests=1), log.db_path, breaker=breaker
    ) as harness:
        for index in range(3):
            post_once(harness.proxy_port, switch_body())
            assert wait_for_rows(log, index + 1) == index + 1

        # recent() is newest first, so index 2 is the first of the three.
        assert log.recent(3)[2].error == "rewrite_failed"
        assert [row.error for row in log.recent(3)] == ["rewrite_failed"] * 3
        assert breaker.is_open() is True


# --- the per-session switch cap, as a value ---------------------------------


def switch_decision() -> Decision:
    return Decision(
        action="SWITCH",
        target_model=MODEL_HIGH,
        reason_codes=["ESCALATE_TOOL_ERRORS"],
        direction=UP,
    )


def cheap_signals() -> Signals:
    return Signals(
        context_tokens_estimate=100,
        tool_use_pending=False,
        in_tool_loop=False,
        consecutive_tool_errors=2,
    )


def test_the_default_cap_is_ten():
    from router.config import DEFAULT_MAX_SWITCHES_PER_SESSION

    assert DEFAULT_MAX_SWITCHES_PER_SESSION == 10
    assert dummy_shadow_config().policy.max_switches_per_session == 10
    assert active_config().policy.max_switches_per_session == 10


def test_a_switch_below_the_cap_is_allowed():
    config = session_config(max_switches_per_session=3)
    state = SessionState(requests_seen=1, switch_count=2)

    final, new_state = apply_safety(switch_decision(), cheap_signals(), state, config)

    assert final.action == "SWITCH"
    assert new_state.switch_count == 3


def test_the_cap_blocks_the_next_switch():
    config = session_config(max_switches_per_session=3)
    state = SessionState(requests_seen=1, switch_count=3)

    final, _ = apply_safety(switch_decision(), cheap_signals(), state, config)

    assert final.action == "STAY"
    assert BLOCKED_SWITCH_CAP in final.reason_codes
    assert is_blocked(final.reason_codes)


def test_the_cap_keeps_the_reason_the_policy_layer_gave():
    config = session_config(max_switches_per_session=1)
    state = SessionState(requests_seen=1, switch_count=1)

    final, _ = apply_safety(switch_decision(), cheap_signals(), state, config)

    assert final.reason_codes == ["ESCALATE_TOOL_ERRORS", BLOCKED_SWITCH_CAP]


def test_dwell_is_checked_before_the_cap():
    """A request that is both too early and over the cap reports dwell."""
    config = session_config(dwell_requests=5, max_switches_per_session=1)
    state = SessionState(
        requests_seen=3,
        last_switch_request_index=2,
        last_switch_direction=UP,
        switch_count=1,
    )

    final, _ = apply_safety(switch_decision(), cheap_signals(), state, config)

    assert final.action == "STAY"
    assert BLOCKED_DWELL in final.reason_codes
    assert BLOCKED_SWITCH_CAP not in final.reason_codes


def test_the_cap_only_counts_allowed_switches():
    config = session_config(max_switches_per_session=2)
    blocked_state = SessionState(requests_seen=1, switch_count=2)

    for decision in (
        Decision(action="STAY", target_model=MODEL_MID, reason_codes=["NO_RULE_MATCHED"]),
        replace(switch_decision(), target_model=MODEL_HIGH),
    ):
        _, unchanged = apply_safety(decision, cheap_signals(), blocked_state, config)
        assert unchanged.switch_count == 2


def test_recorded_switch_counts_one_more():
    state = SessionState(requests_seen=7, switch_count=4)

    recorded = state.recorded_switch(UP)

    assert recorded.switch_count == 5
    assert state.switch_count == 4, "the original state is untouched"


def test_a_fresh_session_has_made_no_switches():
    assert SessionState().switch_count == 0


# --- the per-session switch cap, through the proxy -------------------------


def test_the_cap_blocks_the_switch_after_the_configured_number(tmp_path, monkeypatch):
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    log = log_at(tmp_path, monkeypatch)
    config = session_config(dwell_requests=1, max_switches_per_session=2)
    body = switch_body(MODEL_LOW)

    with running_proxy(config, log.db_path) as harness:
        for index in range(1, 4):
            post_once(harness.proxy_port, body)
            assert wait_for_rows(log, index) == index

        rows = log.recent(3)  # newest first
        assert [row.applied for row in rows] == [0, 1, 1]
        assert [row.action for row in rows] == ["STAY", "SWITCH", "SWITCH"]
        assert BLOCKED_SWITCH_CAP in rows[0].reason_codes

        # the blocked request kept the model the client asked for
        assert json.loads(harness.forwarded.body)["model"] == MODEL_LOW


def test_the_cap_counts_per_session_not_globally(tmp_path, monkeypatch):
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    log = log_at(tmp_path, monkeypatch)
    config = session_config(dwell_requests=1, max_switches_per_session=1)
    first = switch_body(MODEL_LOW)

    other = json.loads(first.decode())
    other["messages"] = json.loads(first.decode())["messages"]
    other["messages"][0]["content"] = "a different session entirely"
    second = json.dumps(other).encode()

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, first)
        assert wait_for_rows(log, 1) == 1
        post_once(harness.proxy_port, first)
        assert wait_for_rows(log, 2) == 2

        post_once(harness.proxy_port, second)
        assert wait_for_rows(log, 3) == 3

        rows = log.recent(3)
        assert rows[0].applied == 1
        assert rows[1].applied == 0
        assert BLOCKED_SWITCH_CAP in rows[1].reason_codes
        assert rows[2].applied == 1, "a second session has its own budget"
        assert rows[0].session_hint != rows[2].session_hint


def test_a_blocked_cap_row_is_not_an_internal_error(tmp_path, monkeypatch):
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    log = log_at(tmp_path, monkeypatch)
    config = session_config(dwell_requests=1, max_switches_per_session=1)

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, switch_body(MODEL_LOW))
        post_once(harness.proxy_port, switch_body(MODEL_LOW))
        assert wait_for_rows(log, 2) == 2

        row = log.recent(1)[0]
        assert row.error is None
        assert is_internal_error(row.error, row.reason_codes) is False


# --- config validation ------------------------------------------------------


def test_the_new_policy_keys_load_from_the_shipped_config():
    policy = load_config().policy

    assert policy is not None
    assert policy.breaker_error_threshold == 3
    assert policy.breaker_cooldown_seconds == 60
    assert policy.max_switches_per_session == 10


@pytest.mark.parametrize(
    "overrides",
    [
        {"breaker_error_threshold": 5, "breaker_cooldown_seconds": 10, "max_switches_per_session": 2},
    ],
)
def test_the_new_policy_keys_are_read_when_present(tmp_path, overrides):
    import yaml

    data = yaml.safe_load(ACTIVE_CONFIG_PATH.read_text(encoding="utf-8"))
    data["policy"].update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    policy = load_config(path).policy

    assert policy is not None
    assert policy.breaker_error_threshold == overrides["breaker_error_threshold"]
    assert policy.breaker_cooldown_seconds == overrides["breaker_cooldown_seconds"]
    assert policy.max_switches_per_session == overrides["max_switches_per_session"]


@pytest.mark.parametrize(
    "key", ["breaker_error_threshold", "breaker_cooldown_seconds", "max_switches_per_session"]
)
@pytest.mark.parametrize("value", [0, -1, "3", 1.5, True, None])
def test_the_new_policy_keys_must_be_positive_integers(tmp_path, key, value):
    import yaml

    data = yaml.safe_load(ACTIVE_CONFIG_PATH.read_text(encoding="utf-8"))
    data["policy"][key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigError, match=key):
        load_config(path)


def test_an_unknown_policy_key_is_still_rejected(tmp_path):
    import yaml

    data = yaml.safe_load(ACTIVE_CONFIG_PATH.read_text(encoding="utf-8"))
    data["policy"]["breaker_cooldown_secondz"] = 60
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown policy key"):
        load_config(path)


# --- the listen host is still loopback only --------------------------------


@pytest.mark.parametrize(
    "host", ["0.0.0.0", "localhost", "192.168.1.10", "::", "example.internal", ""]
)
def test_no_host_but_loopback_is_ever_bound(host):
    """Confirms the existing guards, without restating their implementation."""
    with pytest.raises(ProxyError, match="refusing to listen"):
        create_server(ProxySettings(listen_host=host, listen_port=0, upstream="http://127.0.0.1:1"))

    with pytest.raises(ConfigError, match="listen host must be 127.0.0.1"):
        parse_listen(f"{host}:8787" if host else ":8787")


def test_the_kill_switch_does_not_relax_the_listen_host(monkeypatch):
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    try:
        with pytest.raises(ProxyError, match="refusing to listen"):
            create_server(
                ProxySettings(listen_host="0.0.0.0", listen_port=0, upstream="http://127.0.0.1:1")
            )
    finally:
        monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)


def test_a_shipped_config_on_the_loopback_still_loads():
    assert parse_listen(load_config().listen) == (LOOPBACK_HOST, 8787)


# --- the cli ---------------------------------------------------------------


def test_cli_off_creates_the_flag_file_and_on_removes_it(tmp_path, monkeypatch, capsys):
    from router.cli import main

    monkeypatch.setenv(DB_ENV_VAR, str(tmp_path / "router.sqlite3"))
    flag = kill_switch_path(DecisionLog.from_env().db_path)

    assert main(["off"]) == 0
    assert flag.is_file()
    assert capsys.readouterr().out.strip().endswith(f"kill switch ON ({flag})")

    assert main(["on"]) == 0
    assert not flag.exists()
    assert "kill switch OFF" in capsys.readouterr().out


def test_cli_status_reports_the_switch_and_why(tmp_path, monkeypatch, capsys):
    from router.cli import main

    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    monkeypatch.setenv(DB_ENV_VAR, str(tmp_path / "router.sqlite3"))

    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"^kill_switch\s+: OFF", out, re.MULTILINE)

    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"^kill_switch\s+: ON", out, re.MULTILINE)
    assert KILL_SWITCH_ENV in out


def test_cli_status_reports_the_flag_file(tmp_path, monkeypatch, capsys):
    from router.cli import main

    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    monkeypatch.setenv(DB_ENV_VAR, str(tmp_path / "router.sqlite3"))

    main(["off"])
    capsys.readouterr()
    assert main(["status"]) == 0

    out = capsys.readouterr().out
    assert re.search(r"^kill_switch\s+: ON", out, re.MULTILINE)
    assert KILL_SWITCH_FILENAME in out


def test_cli_log_labels_kill_and_circuit_rows(tmp_path, monkeypatch, capsys):
    from router.cli import main

    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    log = log_at(tmp_path, monkeypatch)
    body = switch_body()

    with running_proxy(active_config(), log.db_path) as harness:
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 1) == 1

        monkeypatch.setenv(KILL_SWITCH_ENV, "1")
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 2) == 2

    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    assert main(["log", "--last", "5"]) == 0

    out = capsys.readouterr().out
    lines = out.splitlines()
    assert REASON_KILL_SWITCH in lines[1]
    assert "kill switch" in lines[1]
    assert REASON_CIRCUIT_OPEN not in out


def test_cli_log_labels_an_open_circuit_row(tmp_path, monkeypatch, capsys):
    from router.cli import format_rows, main

    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    log = log_at(tmp_path, monkeypatch)
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=1, cooldown_seconds=60, clock=clock)

    monkeypatch.setattr(proxy_module, "compute_signals", break_signals)
    with running_proxy(active_config(), log.db_path, breaker=breaker) as harness:
        post_once(harness.proxy_port, switch_body())
        assert wait_for_rows(log, 1) == 1
        post_once(harness.proxy_port, switch_body())
        assert wait_for_rows(log, 2) == 2

    out = format_rows(log.recent(2))
    assert REASON_CIRCUIT_OPEN in out
    assert "circuit open" in out

    assert main(["log", "--last", "2"]) == 0
    assert REASON_CIRCUIT_OPEN in capsys.readouterr().out


def test_format_kill_switch_words():
    from router.cli import format_kill_switch

    assert format_kill_switch(None) == "OFF"
    assert format_kill_switch(KillSwitch(on=False)) == "OFF"
    assert format_kill_switch(KillSwitch(on=True, reasons=("x",))) == "ON (x)"