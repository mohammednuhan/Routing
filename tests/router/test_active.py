"""ACTIVE-mode tests.

Active mode is the only mode in which the router changes a request. Every test
here drives the proxy through the same in-process harness the other proxy tests
use, and asserts on the bytes that reached the upstream, on the row that was
recorded, and on the response headers. Nothing here starts a server of its own or
runs the router as a subprocess.

Config comes from `tools/sample-config-ACTIVE-test.yaml`: `mode: active`, the
non-placeholder ids `test-low`/`test-mid`/`test-high`, and dummy prices. The
shipped config keeps its `TODO_` ids and null prices and stays in shadow, so no
test here can change a request that a real install would send.
"""
from __future__ import annotations

import contextlib
import http.client
import json
import threading
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from router.cli import ACTIVE_BANNER, PLACEHOLDER_PREFIX, active_start_blocker
from router.config import RouterConfig, load_config
from router.decisions import DB_ENV_VAR, MESSAGES_PATH, DecisionLog
from router.proxy import (
    LOOPBACK_HOST,
    REWRITE_FAILED,
    REWRITE_SKIPPED_ENCODING,
    REWRITE_SKIPPED_NOT_JSON,
    ROUTED_HEADER,
    ProxySettings,
    create_server,
    rewrite_model,
)
from router.safety import BLOCKED_COST, is_blocked
from router.state import SessionStore

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_CONFIG_PATH = REPO_ROOT / "tools" / "sample-config-ACTIVE-test.yaml"
DUMMY_CONFIG_PATH = REPO_ROOT / "tools" / "sample-config-DUMMY-prices.yaml"

MODEL_LOW = "test-low"
MODEL_MID = "test-mid"
MODEL_HIGH = "test-high"

#: A phrase that must never reach the database or any log line.
SECRET_PHRASE = "purple-orchid-trombone-7731"

CLIENT_TIMEOUT = 5.0

#: Dummy write 3.75, dummy read 0.30, dummy benefit 0.001: the benefit covers a
#: rebuild below 0.001 * 1_000_000 / 3.45 = 289 tokens of context. The bodies
#: here are deliberately small so an escalation is affordable and the switch is
#: not quietly blocked on cost instead.
Crossover_BYTES = 289 * 4

#: Distinguishes "no store argument" from an explicit `state=None`.
UNSET = object()


def active_config() -> RouterConfig:
    return load_config(ACTIVE_CONFIG_PATH)


def dummy_shadow_config() -> RouterConfig:
    """The shadow config with dummy prices, for the mode-contrast tests."""
    return load_config(DUMMY_CONFIG_PATH)


def switch_body(model: str = MODEL_MID, phrase: str = SECRET_PHRASE) -> bytes:
    """A body that escalates: two tool errors in a row, small enough to afford.

    It is built and then asserted to be under the cost crossover, so a test
    about rewriting cannot pass because the switch was blocked.
    """
    body = json.dumps(
        {
            "model": model,
            "max_tokens": 16,
            "system": [{"type": "text", "text": phrase}],
            "messages": [
                {"role": "user", "content": "start"},
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
        },
        indent=1,
    ).encode()
    assert len(body) < Crossover_BYTES, "switch_body must stay cheap enough to be allowed"
    return body


def stay_body(model: str = MODEL_MID, phrase: str = SECRET_PHRASE) -> bytes:
    """A body no rule acts on: one tool error, no repeat, one turn.

    `escalate_consecutive_errors` is 2, so a single error is not an escalation,
    and the downgrade rule requires zero errors. Nothing matches, so the router
    must STAY. `block_in_tool_loop` is false, so ending inside a tool loop does
    not turn this into a blocked switch.
    """
    return json.dumps(
        {
            "model": model,
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": f"hello: {phrase}"},
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
    """Records what arrived, replies with a fixed JSON body."""

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

    @property
    def forwarded(self) -> Recorded:
        assert self.recorded, "the upstream received nothing"
        return self.recorded[-1]


@contextlib.contextmanager
def running_proxy(
    config: RouterConfig,
    db_path: Path,
    state: SessionStore | None | object = UNSET,
) -> Iterator[Harness]:
    upstream = ThreadingHTTPServer((LOOPBACK_HOST, 0), RecordingUpstream)
    upstream.daemon_threads = True
    upstream.recorded = []  # type: ignore[attr-defined]
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    decisions = DecisionLog(db_path)
    server = create_server(
        ProxySettings(
            listen_port=0,
            upstream=f"http://{LOOPBACK_HOST}:{upstream.server_address[1]}",
            mode=config.mode,
            decisions=decisions,
            config=config,
            state=SessionStore() if state is UNSET else state,  # type: ignore[arg-type]
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


def post_once(
    port: int, body: bytes, headers: dict[str, str] | None = None
) -> tuple[int, dict[str, str], bytes]:
    """One request; returns `(status, headers, body)` with headers lowercased."""
    connection = http.client.HTTPConnection(LOOPBACK_HOST, port, timeout=CLIENT_TIMEOUT)
    try:
        send = {"Content-Type": "application/json"}
        send.update(headers or {})
        connection.request("POST", MESSAGES_PATH, body=body, headers=send)
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


def log_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DecisionLog:
    monkeypatch.setenv(DB_ENV_VAR, str(tmp_path / "router.sqlite3"))
    return DecisionLog.from_env()


# --- the rewrite itself -----------------------------------------------------


def test_rewrite_replaces_only_the_model_value():
    body = b'{"model":"test-mid","max_tokens":16,"messages":[]}'
    rewritten = rewrite_model(body, "test-high")

    assert rewritten == b'{"model":"test-high","max_tokens":16,"messages":[]}'


def test_rewrite_preserves_key_order_whitespace_and_every_other_byte():
    body = (
        b'{\n  "max_tokens": 16,\n  "model": "test-mid",\n'
        b'  "messages": [{"role":"user","content":"a  b\\tc"}]\n}\n'
    )
    rewritten = rewrite_model(body, "test-high")

    assert rewritten == (
        b'{\n  "max_tokens": 16,\n  "model": "test-high",\n'
        b'  "messages": [{"role":"user","content":"a  b\\tc"}]\n}\n'
    )


def test_rewrite_touches_a_nested_model_field_never():
    body = b'{"outer":{"model":"keep-me"},"model":"test-mid"}'
    rewritten = rewrite_model(body, "test-high")

    assert b'"model":"keep-me"' in rewritten
    assert json.loads(rewritten)["model"] == "test-high"
    assert json.loads(rewritten)["outer"]["model"] == "keep-me"


def test_rewrite_leaves_non_ascii_text_exactly_as_sent():
    body = '{"model":"test-mid","note":"café — ünïcode ✓"}'.encode()
    rewritten = rewrite_model(body, "test-high")

    assert json.loads(rewritten)["note"] == "café — ünïcode ✓"
    assert rewritten.count(b"caf\xc3\xa9") == 1
    assert rewritten.startswith(b'{"model":"test-high",')


@pytest.mark.parametrize(
    "body",
    [
        b"not json at all",
        b"[1, 2, 3]",
        b'"a string"',
        b"12345",
        b"null",
        b'{"max_tokens": 16}',
        b'{"model" 3}',
        b'{"model": }',
        b"",
    ],
)
def test_rewrite_refuses_anything_that_is_not_a_json_object_with_a_model(body: bytes):
    from router.proxy import NotAJSONObject

    with pytest.raises(NotAJSONObject):
        rewrite_model(body, "test-high")


# --- active mode rewrites, shadow and off do not ---------------------------


def test_active_mode_rewrites_the_model_and_applies_one(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    body = switch_body()

    with running_proxy(active_config(), log.db_path) as harness:
        status, headers, payload = post_once(harness.proxy_port, body)
        assert status == 200
        assert json.loads(payload) == {"ok": True}
        assert wait_for_rows(log, 1) == 1

        forwarded = harness.forwarded
        assert json.loads(forwarded.body)["model"] == MODEL_HIGH
        assert forwarded.headers["content-length"] == str(len(forwarded.body))
        assert headers[ROUTED_HEADER.lower()] == f"{MODEL_MID}->{MODEL_HIGH}"

        row = log.recent(1)[0]
        assert row.action == "SWITCH"
        assert row.applied == 1
        assert row.requested_model == MODEL_MID
        assert row.chosen_model == MODEL_HIGH
        assert row.mode == "active"
        assert not is_blocked(row.reason_codes)


def test_active_mode_preserves_every_byte_other_than_the_model(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    body = switch_body()

    with running_proxy(active_config(), log.db_path) as harness:
        post_once(harness.proxy_port, body)
        forwarded = harness.forwarded.body

        # Byte for byte the same request, with only the model value swapped.
        assert len(forwarded) == len(body) - len(MODEL_MID) + len(MODEL_HIGH)
        assert forwarded.count(b"\n") == body.count(b"\n")
        assert forwarded.startswith(body[: body.index(b'"model"')])
        assert forwarded.endswith(body[body.index(b",", body.index(b'"model"')) :])


def test_active_mode_stay_forwards_the_body_untouched(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    body = stay_body()

    with running_proxy(active_config(), log.db_path) as harness:
        status, headers, _ = post_once(harness.proxy_port, body)
        assert status == 200
        assert wait_for_rows(log, 1) == 1

        assert harness.forwarded.body == body
        assert ROUTED_HEADER.lower() not in headers

        row = log.recent(1)[0]
        assert row.action == "STAY"
        assert row.applied == 0
        assert row.chosen_model == MODEL_MID


def test_active_mode_never_rewrites_the_effort(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    body = json.dumps(
        {
            "model": MODEL_MID,
            "effort": "test-effort-high",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "start"},
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

    with running_proxy(active_config(), log.db_path) as harness:
        post_once(harness.proxy_port, body)
        forwarded = json.loads(harness.forwarded.body)

        assert forwarded["model"] == MODEL_HIGH
        assert forwarded["effort"] == "test-effort-high"
        assert wait_for_rows(log, 1) == 1
        assert log.recent(1)[0].chosen_effort == "test-effort-high"


def test_shadow_mode_forwards_the_same_body_even_when_it_would_switch(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = dummy_shadow_config()
    body = switch_body(model=config.default_model)

    with running_proxy(config, log.db_path) as harness:
        status, headers, _ = post_once(harness.proxy_port, body)
        assert status == 200
        assert wait_for_rows(log, 1) == 1

        assert harness.forwarded.body == body
        assert ROUTED_HEADER.lower() not in headers

        row = log.recent(1)[0]
        assert row.mode == "shadow"
        assert row.action == "SWITCH"
        assert row.applied == 0


def test_off_mode_forwards_everything_untouched(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    body = switch_body()
    config = replace(active_config(), mode="off")

    with running_proxy(config, log.db_path) as harness:
        status, _, _ = post_once(harness.proxy_port, body)
        assert status == 200
        assert wait_for_rows(log, 1) == 1

        assert harness.forwarded.body == body
        row = log.recent(1)[0]
        assert row.mode == "off"
        assert row.action == "STAY"
        assert row.applied == 0


# --- a blocked switch is a STAY in active mode too --------------------------


def test_active_mode_applies_nothing_when_safety_blocks(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = active_config()
    expensive = json.dumps(
        {
            "model": MODEL_MID,
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "x" * 4_000},
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

    with running_proxy(config, log.db_path) as harness:
        status, headers, _ = post_once(harness.proxy_port, expensive)
        assert status == 200
        assert wait_for_rows(log, 1) == 1

        assert harness.forwarded.body == expensive
        assert ROUTED_HEADER.lower() not in headers

        row = log.recent(1)[0]
        assert row.action == "STAY"
        assert row.applied == 0
        assert is_blocked(row.reason_codes)
        assert BLOCKED_COST in row.reason_codes
        assert row.chosen_model == MODEL_MID


# --- every rewrite failure forwards the original bytes ----------------------


def test_content_encoding_blocks_the_rewrite_and_forwards_the_original(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    body = switch_body()

    with running_proxy(active_config(), log.db_path) as harness:
        status, headers, _ = post_once(
            harness.proxy_port, body, headers={"Content-Encoding": "gzip"}
        )
        assert status == 200
        assert wait_for_rows(log, 1) == 1

        assert harness.forwarded.body == body
        assert ROUTED_HEADER.lower() not in headers

        row = log.recent(1)[0]
        assert row.error == REWRITE_SKIPPED_ENCODING
        assert row.applied == 0
        assert row.chosen_model == MODEL_HIGH


def test_identity_content_encoding_still_allows_the_rewrite(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    body = switch_body()

    with running_proxy(active_config(), log.db_path) as harness:
        status, _, _ = post_once(
            harness.proxy_port, body, headers={"Content-Encoding": "identity"}
        )
        assert status == 200
        assert wait_for_rows(log, 1) == 1

        assert json.loads(harness.forwarded.body)["model"] == MODEL_HIGH
        assert log.recent(1)[0].applied == 1


def test_a_json_array_is_forwarded_untouched(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    body = b'[{"model":"test-mid"}]'

    with running_proxy(active_config(), log.db_path) as harness:
        status, _, _ = post_once(harness.proxy_port, body)
        assert status == 200
        assert wait_for_rows(log, 1) == 1

        assert harness.forwarded.body == body
        row = log.recent(1)[0]
        assert row.applied == 0
        assert row.error == REWRITE_SKIPPED_NOT_JSON


def test_a_json_object_without_a_model_is_forwarded_untouched(tmp_path, monkeypatch):
    """The policy can still switch on the configured default, but there is no
    top-level `model` to replace, so the original bytes go upstream."""
    log = log_at(tmp_path, monkeypatch)
    body = json.dumps(
        {
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "start"},
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

    with running_proxy(active_config(), log.db_path) as harness:
        status, _, _ = post_once(harness.proxy_port, body)
        assert status == 200
        assert wait_for_rows(log, 1) == 1

        assert harness.forwarded.body == body
        row = log.recent(1)[0]
        assert row.action == "SWITCH"
        assert row.applied == 0
        assert row.error == REWRITE_SKIPPED_NOT_JSON
        assert row.chosen_model == MODEL_HIGH


def test_a_body_that_is_not_json_at_all_is_forwarded_untouched(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    body = b"this is not json"

    with running_proxy(active_config(), log.db_path) as harness:
        status, _, _ = post_once(harness.proxy_port, body)
        assert status == 200
        assert wait_for_rows(log, 1) == 1

        assert harness.forwarded.body == body
        row = log.recent(1)[0]
        assert row.applied == 0
        assert row.error == REWRITE_SKIPPED_NOT_JSON


def test_a_rewrite_failure_forwards_the_original_and_still_answers(tmp_path, monkeypatch):
    """An unexpected failure inside the rewrite is `rewrite_failed`, not a 500."""
    from router import proxy as proxy_module

    log = log_at(tmp_path, monkeypatch)
    body = switch_body()

    def boom(*args: Any, **kwargs: Any) -> bytes:
        raise RuntimeError("rewrite exploded")

    monkeypatch.setattr(proxy_module, "rewrite_model", boom)

    with running_proxy(active_config(), log.db_path) as harness:
        status, headers, payload = post_once(harness.proxy_port, body)
        assert status == 200
        assert json.loads(payload) == {"ok": True}
        assert wait_for_rows(log, 1) == 1

        assert harness.forwarded.body == body
        assert ROUTED_HEADER.lower() not in headers

        row = log.recent(1)[0]
        assert row.error == REWRITE_FAILED
        assert row.applied == 0


# --- the routed header ------------------------------------------------------


def test_routed_header_is_absent_when_the_config_does_not_ask_for_it(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = replace(active_config(), routed_header=False)
    body = switch_body()

    with running_proxy(config, log.db_path) as harness:
        status, headers, _ = post_once(harness.proxy_port, body)
        assert status == 200
        assert json.loads(harness.forwarded.body)["model"] == MODEL_HIGH
        assert ROUTED_HEADER.lower() not in headers


def test_routed_header_is_absent_when_nothing_was_rewritten(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = replace(active_config(), routed_header=True)

    with running_proxy(config, log.db_path) as harness:
        _, headers, _ = post_once(harness.proxy_port, stay_body())
        assert ROUTED_HEADER.lower() not in headers


def test_routed_header_names_both_ends_of_the_hop(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    body = switch_body(model=MODEL_LOW)

    with running_proxy(active_config(), log.db_path) as harness:
        _, headers, _ = post_once(harness.proxy_port, body)
        assert headers[ROUTED_HEADER.lower()] == f"{MODEL_LOW}->{MODEL_MID}"


def test_shadow_mode_never_sends_the_routed_header(tmp_path, monkeypatch):
    """The flag is on in the active sample config; shadow still sends nothing."""
    log = log_at(tmp_path, monkeypatch)
    config = replace(dummy_shadow_config(), routed_header=True)

    with running_proxy(config, log.db_path) as harness:
        _, headers, _ = post_once(harness.proxy_port, switch_body(MODEL_MID))
        assert ROUTED_HEADER.lower() not in headers


# --- the startup guard ------------------------------------------------------


def test_active_start_is_refused_while_a_model_id_is_a_placeholder():
    real = active_config()
    placeholder = replace(real.models[0], id=f"{PLACEHOLDER_PREFIX}MODEL_TIER_LOW")
    config = replace(
        real,
        models=(placeholder, *real.models[1:]),
        default_model=placeholder.id,
    )

    blocker = active_start_blocker(config)

    assert blocker is not None
    assert PLACEHOLDER_PREFIX in blocker


def test_shipped_config_still_loads_in_shadow_despite_its_placeholder_ids():
    from router.config import load_config as load_shipped

    config = load_shipped()

    assert config.mode == "shadow"
    assert active_start_blocker(config) is None


def test_active_start_is_allowed_when_every_id_is_real():
    assert active_start_blocker(active_config()) is None


def test_start_prints_the_active_banner_when_it_is_allowed(capsys, monkeypatch):
    from router import cli as cli_module
    from router.cli import main

    served: list[object] = []

    def fake_serve(settings: Any) -> None:
        served.append(settings)

    monkeypatch.setattr(cli_module, "serve", fake_serve)

    code = main(["--config", str(ACTIVE_CONFIG_PATH), "start"])

    assert code == 0
    assert ACTIVE_BANNER in capsys.readouterr().out
    assert served, "the proxy was never started"


def test_start_prints_no_active_banner_in_shadow(capsys, monkeypatch):
    from router import cli as cli_module
    from router.cli import main

    monkeypatch.setattr(cli_module, "serve", lambda settings: None)

    code = main(["--config", str(DUMMY_CONFIG_PATH), "start"])

    assert code == 0
    assert ACTIVE_BANNER not in capsys.readouterr().out


def test_start_refuses_active_with_placeholders_and_exits_non_zero(capsys, tmp_path):
    from router.cli import main

    path = tmp_path / "active-with-placeholder.yaml"
    path.write_text(
        "\n".join(
            [
                "listen: 127.0.0.1:8787",
                "upstream: https://api.anthropic.com",
                "mode: active",
                "models:",
                f"  - id: {PLACEHOLDER_PREFIX}MODEL_TIER_LOW",
                "    legal_efforts: [a]",
                "    cost_tier: low",
                f"default_model: {PLACEHOLDER_PREFIX}MODEL_TIER_LOW",
                "default_effort: a",
            ]
        ),
        encoding="utf-8",
    )

    code = main(["--config", str(path), "start"])

    assert code != 0
    assert "active" in capsys.readouterr().err


# --- metadata only ----------------------------------------------------------


def test_no_request_text_reaches_the_database_or_the_log(tmp_path, monkeypatch):
    import io

    log = log_at(tmp_path, monkeypatch)
    body = switch_body()

    with running_proxy(active_config(), log.db_path) as harness:
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            post_once(harness.proxy_port, body)
            assert wait_for_rows(log, 1) == 1

        assert SECRET_PHRASE.encode() not in log.db_path.read_bytes()
        assert SECRET_PHRASE not in captured.getvalue()


def test_the_request_still_reaches_the_upstream_when_the_log_cannot_be_written(
    tmp_path, monkeypatch
):
    blocker = tmp_path / "not-a-directory"
    blocker.write_bytes(b"this is a file, not a directory")
    monkeypatch.setenv(DB_ENV_VAR, str(blocker / "router.sqlite3"))
    log = DecisionLog.from_env()

    with running_proxy(active_config(), log.db_path) as harness:
        status, _, payload = post_once(harness.proxy_port, switch_body())

        assert status == 200
        assert json.loads(payload) == {"ok": True}
        assert json.loads(harness.forwarded.body)["model"] == MODEL_HIGH


def test_a_rewrite_never_targets_a_model_outside_the_config(tmp_path, monkeypatch):
    """Rule 7: an illegal target is forwarded as-is and never sent upstream."""
    from router import proxy as proxy_module

    log = log_at(tmp_path, monkeypatch)
    body = switch_body()
    original = proxy_module._is_legal_target
    monkeypatch.setattr(
        proxy_module,
        "_is_legal_target",
        lambda config, model: False if model == MODEL_HIGH else original(config, model),
    )

    with running_proxy(active_config(), log.db_path) as harness:
        status, _, _ = post_once(harness.proxy_port, body)
        assert status == 200
        assert wait_for_rows(log, 1) == 1

        assert harness.forwarded.body == body
        row = log.recent(1)[0]
        assert row.applied == 0
        assert row.error == REWRITE_FAILED


def test_active_mode_without_a_state_store_never_rewrites(tmp_path, monkeypatch):
    """No store means no session, so no safety, so no switch is applied."""
    log = log_at(tmp_path, monkeypatch)
    body = switch_body()

    with running_proxy(active_config(), log.db_path, state=None) as harness:
        status, _, _ = post_once(harness.proxy_port, body)
        assert status == 200
        assert wait_for_rows(log, 1) == 1

        assert harness.forwarded.body == body
        row = log.recent(1)[0]
        assert row.action == "SWITCH"
        assert row.applied == 0


# --- the mock upstream understands /v1/messages ----------------------------


@contextlib.contextmanager
def running_mock_upstream() -> Iterator[int]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "mock_upstream", REPO_ROOT / "tools" / "mock_upstream.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    server = ThreadingHTTPServer((LOOPBACK_HOST, 0), module.MockUpstreamHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_mock_upstream_reports_the_model_it_received():
    with running_mock_upstream() as port:
        body = switch_body()
        status, _, payload = post_once(port, body, headers={"Authorization": "Bearer x"})
        assert status == 200

        report = json.loads(payload)
        assert report["ok"] is True
        assert report["model"] == MODEL_MID
        assert report["received_bytes"] == len(body)
        assert report["content_length"] == str(len(body))
        assert report["messages"] == 3
        assert SECRET_PHRASE.encode() not in payload


def test_mock_upstream_survives_a_body_that_is_not_json():
    """Answered 200, byte count intact, and nothing parsed out of the body."""
    with running_mock_upstream() as port:
        status, _, payload = post_once(port, b"not json")

        assert status == 200
        report = json.loads(payload)
        assert report["ok"] is True
        assert report["received_bytes"] == 8
        assert report["content_length"] == "8"
        assert report["json"] is False
        # Anything only a parsed body could produce must be absent: the mock
        # upstream must not have guessed a model, an effort, framing or a count.
        for absent in ("model", "effort", "stream", "messages"):
            assert absent not in report, f"reported {absent} from a body it could not parse"