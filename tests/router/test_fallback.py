"""Same-tier fallback tests.

A fallback is the one change the router makes that no rule proposed: the upstream
answered with a status that means "not right now", and the request is sent again
onto the next configured model sharing the tier of the one that just failed.

Every test here drives the same in-process harness the other proxy tests use and
asserts on the bytes that reached the upstream, on the response the client got,
and on the row that was recorded. Nothing here starts a server of its own or runs
the router as a subprocess, and nothing contacts a real upstream.

Config is `tools/sample-config-ACTIVE-test.yaml` with its models replaced by
models that share a tier, because one model per tier has nothing to fall back to.
Prices are the dummy numbers from that file; the ids are invented.
"""
from __future__ import annotations

import contextlib
import http.client
import json
import threading
import time
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml

from router.breaker import REASON_CIRCUIT_OPEN, CircuitBreaker
from router.config import (
    DEFAULT_FALLBACK_MAX_ATTEMPTS,
    DEFAULT_FALLBACK_STATUSES,
    ConfigError,
    ModelSpec,
    PolicySpec,
    RouterConfig,
    load_config,
)
from router.decisions import DB_ENV_VAR, MESSAGES_PATH, DecisionLog
from router.killswitch import REASON_KILL_SWITCH, write_kill_switch
from router.proxy import (
    FALLBACK_ATTEMPTS_SIGNAL_KEY,
    FALLBACK_FROM_SIGNAL_KEY,
    LOOPBACK_HOST,
    REASON_FALLBACK_NEXT_IN_TIER,
    ROUTED_HEADER,
    FallbackSettings,
    ProxySettings,
    _retry_body,
    _discard,
    create_server,
    fallback_settings,
    forwarded_model,
    next_in_cost_tier,
)
from router.state import SessionStore

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_CONFIG_PATH = REPO_ROOT / "tools" / "sample-config-ACTIVE-test.yaml"

#: Three models in one tier, so a fallback has somewhere to go, and one model
#: alone in another tier, so a fallback must never reach it.
MODEL_A = "fb-a"
MODEL_B = "fb-b"
MODEL_C = "fb-c"
SOLO_TIER = "fb-solo-tier"

#: An id shaped the way real gateway ids are: a slash and a colon inside it. A
#: retry must carry it as one id and never read it as two.
MODEL_PUNCT = "org/team:fb-model-1"

#: A phrase that must never reach the database or any log line, in the request
#: that gets retried and in the request that does not.
SECRET_PHRASE = "purple-orchid-trombone-7731"

CLIENT_TIMEOUT = 5.0

OPENAI_PATH = "/v1/chat/completions"

DUMMY_EFFORTS = ("test-effort-low", "test-effort-medium", "test-effort-high")


def _model(model_id: str, tier: str) -> ModelSpec:
    """A model with the sample file's dummy prices, in `tier`."""
    return ModelSpec(
        id=model_id,
        legal_efforts=DUMMY_EFFORTS,
        cost_tier=tier,
        input_per_million=3.0,
        output_per_million=15.0,
        cache_write_per_million=3.75,
        cache_read_per_million=0.30,
    )


TIER_MODELS: tuple[ModelSpec, ...] = (
    _model(MODEL_A, "mid"),
    _model(MODEL_B, "mid"),
    _model(MODEL_C, "mid"),
    _model(SOLO_TIER, "high"),
)


def fallback_config(
    *,
    mode: str = "active",
    enabled: bool = True,
    statuses: tuple[int, ...] = DEFAULT_FALLBACK_STATUSES,
    max_attempts: int = 1,
    models: tuple[ModelSpec, ...] = TIER_MODELS,
    routed_header: bool = True,
) -> RouterConfig:
    """The sample config, pointed at the models the fallback tests need."""
    base = load_config(ACTIVE_CONFIG_PATH)
    assert base.policy is not None
    policy = replace(
        base.policy,
        fallback_enabled=enabled,
        fallback_statuses=statuses,
        fallback_max_attempts=max_attempts,
    )
    return replace(base, mode=mode, models=models, policy=policy, routed_header=routed_header)


def messages_body(model: str = MODEL_A, phrase: str = SECRET_PHRASE) -> bytes:
    """A Messages request no rule acts on, so the plan is a STAY in active mode.

    One plain user turn: no tool error to repeat, nothing to escalate on and
    nothing to downgrade, so the only thing that can move the request is the
    upstream's answer.
    """
    return json.dumps(
        {
            "model": model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": f"hello: {phrase}"}],
        }
    ).encode()


def chat_body(model: str = MODEL_A, phrase: str = SECRET_PHRASE) -> bytes:
    """The same request in OpenAI chat-completions format."""
    return json.dumps(
        {
            "model": model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": f"hello: {phrase}"}],
        }
    ).encode()


def messages_reply(model: str = MODEL_B, input_tokens: int = 11, output_tokens: int = 22) -> bytes:
    """A Messages response carrying usage, so a usage row can be checked."""
    return json.dumps(
        {
            "id": "msg_fb_1",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": "hi"}],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 4,
            },
        }
    ).encode()


@dataclass
class Recorded:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes

    @property
    def model(self) -> str:
        return json.loads(self.body)["model"]


class ScriptedUpstream(BaseHTTPRequestHandler):
    """Answers with the statuses the test scripted, in order.

    The last entry repeats once the script runs out, so a test that expects fewer
    upstream requests than it scripted still gets a well-formed answer instead of
    a hang.
    """

    protocol_version = "HTTP/1.1"
    server_version = "scripted-upstream"
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
        index = len(self.server.recorded) - 1  # type: ignore[attr-defined]
        script: list[tuple[int, bytes]] = self.server.script  # type: ignore[attr-defined]
        status, payload = script[min(index, len(script) - 1)]
        self.send_response(status)
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
    def models(self) -> list[str]:
        return [entry.model for entry in self.recorded]

    @property
    def last(self) -> Recorded:
        assert self.recorded, "the upstream received nothing"
        return self.recorded[-1]


@contextlib.contextmanager
def running_proxy(
    config: RouterConfig,
    db_path: Path,
    script: list[tuple[int, bytes]],
    *,
    breaker: CircuitBreaker | None = None,
    state: SessionStore | None = None,
) -> Iterator[Harness]:
    upstream = ThreadingHTTPServer((LOOPBACK_HOST, 0), ScriptedUpstream)
    upstream.daemon_threads = True
    upstream.recorded = []  # type: ignore[attr-defined]
    upstream.script = script  # type: ignore[attr-defined]
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
            state=state if state is not None else SessionStore(),
            breaker=breaker,
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
    port: int,
    body: bytes,
    path: str = MESSAGES_PATH,
    method: str = "POST",
) -> tuple[int, dict[str, str], bytes]:
    """One request; returns `(status, headers, body)` with headers lowercased."""
    connection = http.client.HTTPConnection(LOOPBACK_HOST, port, timeout=CLIENT_TIMEOUT)
    try:
        connection.request(
            method,
            path,
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


def log_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DecisionLog:
    monkeypatch.setenv(DB_ENV_VAR, str(tmp_path / "router.sqlite3"))
    return DecisionLog.from_env()


def wait_for(log: DecisionLog, expected: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while log.count() < expected and time.monotonic() < deadline:
        time.sleep(0.01)


def wait_for_usage(log: DecisionLog, expected: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while log.usage_count() < expected and time.monotonic() < deadline:
        time.sleep(0.01)


# --- the retry itself --------------------------------------------------------


def test_a_retryable_status_sends_the_request_again_in_the_same_tier(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = fallback_config()

    with running_proxy(config, log.db_path, [(503, b'{"error":"busy"}'), (200, b'{"ok":true}')]) as harness:
        status, headers, payload = post_once(harness.proxy_port, messages_body())

        assert status == 200
        assert payload == b'{"ok":true}'
        assert harness.models == [MODEL_A, MODEL_B]
        assert headers[ROUTED_HEADER.lower()] == f"{MODEL_A}->{MODEL_B}"


def test_the_retry_carries_the_same_request_with_only_the_model_replaced(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    body = messages_body()

    with running_proxy(
        fallback_config(),
        log.db_path,
        [(503, b'{"error":"busy"}'), (200, b'{"ok":true}')],
    ) as harness:
        post_once(harness.proxy_port, body)

        first, second = harness.recorded
        # Every byte but the model value, in both attempts.
        assert first.body == body
        assert json.loads(second.body)["model"] == MODEL_B
        assert json.loads(second.body)["messages"] == json.loads(body)["messages"]
        assert SECRET_PHRASE.encode() in second.body
        assert len(second.body) == len(body) - len(MODEL_A) + len(MODEL_B)
        # And the framing each attempt sent describes those exact bytes.
        assert first.headers["content-length"] == str(len(first.body))
        assert second.headers["content-length"] == str(len(second.body))


def test_the_fallback_row_names_the_model_that_answered_and_the_one_that_failed(
    tmp_path, monkeypatch
):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(
        fallback_config(),
        log.db_path,
        [(503, b'{"error":"busy"}'), (200, b'{"ok":true}')],
    ) as harness:
        post_once(harness.proxy_port, messages_body())
        wait_for(log, 1)

        row = log.recent(1)[0]
        assert REASON_FALLBACK_NEXT_IN_TIER in row.reason_codes
        assert row.applied == 1
        assert row.requested_model == MODEL_A
        assert row.chosen_model == MODEL_B
        assert row.signal_values[FALLBACK_FROM_SIGNAL_KEY] == MODEL_A
        assert row.signal_values[FALLBACK_ATTEMPTS_SIGNAL_KEY] == 1
        assert harness.models == [MODEL_A, MODEL_B]


def test_one_row_and_one_usage_row_for_the_whole_client_request(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(
        fallback_config(max_attempts=2),
        log.db_path,
        [(503, b'{"error":"busy"}'), (429, b'{"error":"slow down"}'), (200, b'{"ok":true}')],
    ) as harness:
        post_once(harness.proxy_port, messages_body())
        wait_for(log, 1)
        wait_for_usage(log, 1)

        assert harness.models == [MODEL_A, MODEL_B, MODEL_C]
        assert log.count() == 1
        row = log.recent(1)[0]
        assert row.chosen_model == MODEL_C
        assert row.signal_values[FALLBACK_ATTEMPTS_SIGNAL_KEY] == 2


def test_the_tier_order_wraps_around_at_the_end_of_the_config(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(
        fallback_config(max_attempts=2),
        log.db_path,
        [(503, b'{"error":"busy"}'), (503, b'{"error":"busy"}'), (200, b'{"ok":true}')],
    ) as harness:
        status, _, _ = post_once(harness.proxy_port, messages_body(model=MODEL_C))

        assert status == 200
        assert harness.models == [MODEL_C, MODEL_A, MODEL_B]


def test_a_model_already_tried_is_not_tried_again(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    models = (
        _model(MODEL_A, "mid"),
        _model(MODEL_B, "mid"),
        _model(MODEL_C, "mid"),
        _model("fb-extra", "mid"),
        _model(SOLO_TIER, "high"),
    )

    with running_proxy(
        fallback_config(max_attempts=2, models=models),
        log.db_path,
        [(503, b'{"error":"busy"}'), (503, b'{"error":"busy"}'), (200, b'{"ok":true}')],
    ) as harness:
        post_once(harness.proxy_port, messages_body(model=MODEL_B))

        assert harness.models == [MODEL_B, MODEL_C, "fb-extra"]


def test_the_budget_of_one_retries_once_and_relays_the_last_answer(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(
        fallback_config(max_attempts=1),
        log.db_path,
        [(503, b'{"error":"one"}'), (503, b'{"error":"two"}'), (200, b'{"ok":true}')],
    ) as harness:
        status, _, payload = post_once(harness.proxy_port, messages_body())

        assert status == 503
        assert payload == b'{"error":"two"}'
        assert harness.models == [MODEL_A, MODEL_B]
        wait_for(log, 1)
        row = log.recent(1)[0]
        assert row.signal_values[FALLBACK_ATTEMPTS_SIGNAL_KEY] == 1
        assert row.chosen_model == MODEL_B


def test_a_tier_with_no_other_model_is_never_fallen_back_to(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    models = (_model(MODEL_A, "mid"), _model(SOLO_TIER, "high"))

    with running_proxy(
        fallback_config(models=models),
        log.db_path,
        [(503, b'{"error":"busy"}'), (200, b'{"ok":true}')],
    ) as harness:
        status, _, payload = post_once(harness.proxy_port, messages_body())

        assert status == 503
        assert payload == b'{"error":"busy"}'
        assert harness.models == [MODEL_A]
        wait_for(log, 1)
        row = log.recent(1)[0]
        assert REASON_FALLBACK_NEXT_IN_TIER not in row.reason_codes
        assert FALLBACK_FROM_SIGNAL_KEY not in row.signal_values
        assert row.applied == 0


def test_a_model_outside_the_config_is_never_fallen_back_to(tmp_path, monkeypatch):
    """The model the client named is not in the config, so it has no tier."""
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(
        fallback_config(),
        log.db_path,
        [(503, b'{"error":"busy"}'), (200, b'{"ok":true}')],
    ) as harness:
        status, _, _ = post_once(harness.proxy_port, messages_body(model="not-in-the-config"))

        assert status == 503
        assert harness.models == ["not-in-the-config"]


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_every_default_status_is_retried(status: int, tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(
        fallback_config(),
        log.db_path,
        [(status, b'{"error":"busy"}'), (200, b'{"ok":true}')],
    ) as harness:
        client_status, _, _ = post_once(harness.proxy_port, messages_body())

        assert client_status == 200
        assert harness.models == [MODEL_A, MODEL_B]


@pytest.mark.parametrize("status", [200, 400, 401, 403, 404, 409, 418, 429 + 100, 501, 505])
def test_no_other_status_is_retried(status: int, tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(
        fallback_config(),
        log.db_path,
        [(status, b'{"answer":true}'), (200, b'{"ok":true}')],
    ) as harness:
        client_status, _, payload = post_once(harness.proxy_port, messages_body())

        assert client_status == status
        assert payload == b'{"answer":true}'
        assert harness.models == [MODEL_A]


def test_only_the_configured_statuses_are_retried(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(
        fallback_config(statuses=(429,)),
        log.db_path,
        [(503, b'{"error":"busy"}'), (200, b'{"ok":true}')],
    ) as harness:
        status, _, _ = post_once(harness.proxy_port, messages_body())

        assert status == 503
        assert harness.models == [MODEL_A]


def test_an_empty_status_list_makes_nothing_retryable(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(
        fallback_config(statuses=()),
        log.db_path,
        [(503, b'{"error":"busy"}'), (200, b'{"ok":true}')],
    ) as harness:
        status, _, _ = post_once(harness.proxy_port, messages_body())

        assert status == 503
        assert harness.models == [MODEL_A]


def test_a_successful_first_attempt_is_never_retried(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(
        fallback_config(),
        log.db_path,
        [(200, b'{"ok":true}'), (503, b'{"error":"busy"}')],
    ) as harness:
        status, _, _ = post_once(harness.proxy_port, messages_body())

        assert status == 200
        assert harness.models == [MODEL_A]
        wait_for(log, 1)
        row = log.recent(1)[0]
        assert REASON_FALLBACK_NEXT_IN_TIER not in row.reason_codes
        assert FALLBACK_FROM_SIGNAL_KEY not in row.signal_values
        assert row.applied == 0


def test_a_body_with_no_top_level_model_is_relayed_unchanged(tmp_path, monkeypatch):
    """Nothing to rewrite means nothing to retry: the answer is passed through."""
    log = log_at(tmp_path, monkeypatch)
    body = b'{"max_tokens":16,"messages":[{"role":"user","content":"hi"}]}'

    with running_proxy(
        fallback_config(),
        log.db_path,
        [(503, b'{"error":"busy"}'), (200, b'{"ok":true}')],
    ) as harness:
        status, _, payload = post_once(harness.proxy_port, body)

        assert status == 503
        assert payload == b'{"error":"busy"}'
        assert [entry.body for entry in harness.recorded] == [body]


def test_an_id_with_a_slash_and_a_colon_is_one_model_and_not_two(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    models = (_model(MODEL_A, "mid"), _model(MODEL_PUNCT, "mid"), _model(SOLO_TIER, "high"))

    with running_proxy(
        fallback_config(models=models),
        log.db_path,
        [(503, b'{"error":"busy"}'), (200, b'{"ok":true}')],
    ) as harness:
        status, _, _ = post_once(harness.proxy_port, messages_body())

        assert status == 200
        assert harness.models == [MODEL_A, MODEL_PUNCT]
        assert harness.recorded[-1].headers["content-length"] == str(
            len(harness.recorded[-1].body)
        )


def test_falling_back_from_an_id_with_a_slash_and_a_colon(tmp_path, monkeypatch):
    """The id is the model, so the next model in the tier follows it, not a part."""
    log = log_at(tmp_path, monkeypatch)
    models = (_model(MODEL_A, "mid"), _model(MODEL_PUNCT, "mid"), _model(MODEL_C, "mid"))

    with running_proxy(
        fallback_config(models=models),
        log.db_path,
        [(503, b'{"error":"busy"}'), (200, b'{"ok":true}')],
    ) as harness:
        status, _, _ = post_once(harness.proxy_port, messages_body(model=MODEL_PUNCT))

        assert status == 200
        assert harness.models == [MODEL_PUNCT, MODEL_C]


def test_the_openai_format_falls_back_the_same_way(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(
        fallback_config(),
        log.db_path,
        [(503, b'{"error":{"message":"busy"}}'), (200, b'{"ok":true}')],
    ) as harness:
        status, _, _ = post_once(harness.proxy_port, chat_body(), path=OPENAI_PATH)

        assert status == 200
        assert harness.models == [MODEL_A, MODEL_B]
        wait_for(log, 1)
        assert log.recent(1)[0].signal_values["api_format"] == "openai"


def test_usage_is_read_from_the_response_that_was_relayed(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    failure = json.dumps({"error": "busy", "usage": {"input_tokens": 999, "output_tokens": 999}})
    success = messages_reply(model=MODEL_B, input_tokens=11, output_tokens=22)

    with running_proxy(
        fallback_config(),
        log.db_path,
        [(503, failure.encode()), (200, success)],
    ) as harness:
        status, _, payload = post_once(harness.proxy_port, messages_body())

        assert status == 200
        assert payload == success
        wait_for_usage(log, 1)
        row = log.usage_rows(1)[0]
        assert row.input_tokens == 11
        assert row.output_tokens == 22
        assert row.model_reported == MODEL_B
        assert row.decision_id == log.recent(1)[0].decision_id


def test_the_secret_phrase_reaches_no_row_after_a_retry(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(
        fallback_config(),
        log.db_path,
        [(503, b'{"error":"busy"}'), (200, b'{"ok":true}')],
    ) as harness:
        post_once(harness.proxy_port, messages_body())
        wait_for(log, 1)

        assert SECRET_PHRASE in harness.recorded[-1].body.decode()
        dump = json.dumps(
            [vars(row) for row in log.recent(10)] + [vars(row) for row in log.usage_rows(10)],
            default=str,
        )
        assert SECRET_PHRASE not in dump


# --- what stops a retry ------------------------------------------------------


def test_shadow_mode_never_retries(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(
        fallback_config(mode="shadow"),
        log.db_path,
        [(503, b'{"error":"busy"}'), (200, b'{"ok":true}')],
    ) as harness:
        status, _, _ = post_once(harness.proxy_port, messages_body())

        assert status == 503
        assert harness.models == [MODEL_A]
        wait_for(log, 1)
        row = log.recent(1)[0]
        assert row.mode == "shadow"
        assert REASON_FALLBACK_NEXT_IN_TIER not in row.reason_codes
        assert row.applied == 0


def test_mode_off_never_retries(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(
        fallback_config(mode="off"),
        log.db_path,
        [(503, b'{"error":"busy"}'), (200, b'{"ok":true}')],
    ) as harness:
        status, _, _ = post_once(harness.proxy_port, messages_body())

        assert status == 503
        assert harness.models == [MODEL_A]


def test_the_feature_off_never_retries(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(
        fallback_config(enabled=False),
        log.db_path,
        [(503, b'{"error":"busy"}'), (200, b'{"ok":true}')],
    ) as harness:
        status, _, _ = post_once(harness.proxy_port, messages_body())

        assert status == 503
        assert harness.models == [MODEL_A]


def test_the_kill_switch_passes_the_request_through_unchanged(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    log.count()
    write_kill_switch(log.db_path, True)

    with running_proxy(
        fallback_config(),
        log.db_path,
        [(503, b'{"error":"busy"}'), (200, b'{"ok":true}')],
    ) as harness:
        status, _, _ = post_once(harness.proxy_port, messages_body())

        assert status == 503
        assert harness.models == [MODEL_A]
        wait_for(log, 1)
        row = log.recent(1)[0]
        assert row.reason_codes == [REASON_KILL_SWITCH]
        assert REASON_FALLBACK_NEXT_IN_TIER not in row.reason_codes
        assert FALLBACK_FROM_SIGNAL_KEY not in row.signal_values


def test_an_open_circuit_passes_the_request_through_unchanged(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    breaker = CircuitBreaker(threshold=1, cooldown_seconds=60)
    assert breaker.record(True) is True
    assert breaker.is_open() is True

    with running_proxy(
        fallback_config(),
        log.db_path,
        [(503, b'{"error":"busy"}'), (200, b'{"ok":true}')],
        breaker=breaker,
    ) as harness:
        status, _, _ = post_once(harness.proxy_port, messages_body())

        assert status == 503
        assert harness.models == [MODEL_A]
        wait_for(log, 1)
        row = log.recent(1)[0]
        assert row.reason_codes == [REASON_CIRCUIT_OPEN]
        assert REASON_FALLBACK_NEXT_IN_TIER not in row.reason_codes


def test_a_request_the_router_failed_to_evaluate_is_not_retried(tmp_path, monkeypatch):
    """A router error passes the request through unchanged, retried or not."""
    log = log_at(tmp_path, monkeypatch)

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the policy blew up")

    monkeypatch.setattr("router.proxy.decide", explode)

    with running_proxy(
        fallback_config(),
        log.db_path,
        [(503, b'{"error":"busy"}'), (200, b'{"ok":true}')],
    ) as harness:
        status, _, _ = post_once(harness.proxy_port, messages_body())

        assert status == 503
        assert harness.models == [MODEL_A]
        wait_for(log, 1)
        row = log.recent(1)[0]
        assert row.error == "policy_failed"
        assert REASON_FALLBACK_NEXT_IN_TIER not in row.reason_codes
        assert FALLBACK_FROM_SIGNAL_KEY not in row.signal_values


def test_a_request_that_is_not_a_post_decision_is_not_retried(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(
        fallback_config(),
        log.db_path,
        [(503, b'{"error":"busy"}'), (200, b'{"ok":true}')],
    ) as harness:
        status, _, _ = post_once(harness.proxy_port, b"", path=MESSAGES_PATH, method="GET")

        assert status == 503
        assert len(harness.recorded) == 1
        assert log.count() == 0


def test_an_unrecognised_path_is_not_retried(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(
        fallback_config(),
        log.db_path,
        [(503, b'{"error":"busy"}'), (200, b'{"ok":true}')],
    ) as harness:
        status, _, _ = post_once(harness.proxy_port, messages_body(), path="/v1/other")

        assert status == 503
        assert len(harness.recorded) == 1
        assert log.count() == 0


# --- the helpers, without a server -------------------------------------------


def test_next_in_cost_tier_never_leaves_the_tier():
    config = fallback_config()

    assert next_in_cost_tier(config, MODEL_A, [MODEL_A]) == MODEL_B
    assert next_in_cost_tier(config, MODEL_C, [MODEL_C]) == MODEL_A
    assert next_in_cost_tier(config, MODEL_A, [MODEL_A, MODEL_B]) == MODEL_C
    assert next_in_cost_tier(config, SOLO_TIER, [SOLO_TIER]) is None
    assert SOLO_TIER not in (
        next_in_cost_tier(config, MODEL_A, [MODEL_A]),
        next_in_cost_tier(config, MODEL_B, [MODEL_B]),
    )


def test_next_in_cost_tier_refuses_a_model_the_config_does_not_list():
    assert next_in_cost_tier(fallback_config(), "not-in-the-config", []) is None


def test_next_in_cost_tier_without_a_config():
    assert next_in_cost_tier(None, MODEL_A, []) is None
    assert next_in_cost_tier(object(), MODEL_A, []) is None


def test_forwarded_model_reads_the_bytes_not_the_plan():
    assert forwarded_model(messages_body(MODEL_C)) == MODEL_C
    assert forwarded_model(messages_body(MODEL_PUNCT)) == MODEL_PUNCT
    assert forwarded_model(b'{"outer":{"model":"nested"}}') is None
    assert forwarded_model(b'{"model":7}') is None
    assert forwarded_model(b"not json") is None
    assert forwarded_model(b'{"model":"\xff\xfe"}') is None


def test_retry_body_refuses_a_body_it_cannot_rewrite():
    assert _retry_body(messages_body(), MODEL_B) is not None
    assert _retry_body(b'{"outer":{"model":"nested"}}', MODEL_B) is None
    assert _retry_body(b"not json", MODEL_B) is None
    assert _retry_body(b"", MODEL_B) is None


def test_a_discarded_response_is_closed_and_never_read():
    """The body of a status the router replaces is closed, not relayed and not read."""

    class Stub:
        def __init__(self) -> None:
            self.closed = False
            self.reads = 0

        def close(self) -> None:
            self.closed = True

        def read(self, size: int = -1) -> bytes:
            self.reads += 1
            raise AssertionError("a discarded response must never be read")

    stub = Stub()
    _discard(stub)  # type: ignore[arg-type]

    assert stub.closed is True
    assert stub.reads == 0


def test_a_close_that_fails_is_not_an_error():
    class Stub:
        def close(self) -> None:
            raise OSError("already gone")

    _discard(Stub())  # type: ignore[arg-type]


def test_the_shipped_defaults_are_off_and_one_attempt():
    assert FallbackSettings() == FallbackSettings(
        enabled=False,
        statuses=DEFAULT_FALLBACK_STATUSES,
        max_attempts=DEFAULT_FALLBACK_MAX_ATTEMPTS,
    )
    assert DEFAULT_FALLBACK_STATUSES == (429, 502, 503, 504)
    assert DEFAULT_FALLBACK_MAX_ATTEMPTS == 1


@pytest.mark.parametrize("config", [None, object()])
def test_a_config_with_no_policy_falls_back_to_the_shipped_defaults(config: Any):
    assert fallback_settings(config) == FallbackSettings()


def test_settings_are_read_from_the_policy_and_nothing_else():
    base = fallback_config()

    assert fallback_settings(base) == FallbackSettings(
        enabled=True, statuses=DEFAULT_FALLBACK_STATUSES, max_attempts=1
    )
    assert fallback_settings(fallback_config(max_attempts=3)).max_attempts == 3
    assert fallback_settings(fallback_config(statuses=(500,))).statuses == (500,)
    assert fallback_settings(fallback_config(enabled=False)).enabled is False


@pytest.mark.parametrize(
    "override",
    [
        {"fallback_enabled": "true"},
        {"fallback_enabled": 1},
        {"fallback_statuses": 429},
        {"fallback_statuses": ["429"]},
        {"fallback_statuses": [429.0]},
        {"fallback_statuses": [True]},
        {"fallback_statuses": [99]},
        {"fallback_statuses": [600]},
        {"fallback_max_attempts": 0},
        {"fallback_max_attempts": -1},
        {"fallback_max_attempts": "2"},
        {"fallback_max_attempts": 1.0},
    ],
)
def test_a_config_the_loader_would_refuse_is_not_built_by_the_helper(override: dict[str, Any]):
    """A policy field is not revalidated here; it is not trusted either."""
    config = fallback_config()
    assert config.policy is not None
    broken = replace(config.policy, **override)

    settings = fallback_settings(replace(config, policy=broken))

    if "fallback_enabled" in override:
        assert settings.enabled is False
    elif "fallback_statuses" in override:
        assert settings.statuses == DEFAULT_FALLBACK_STATUSES
    else:
        assert settings.max_attempts == DEFAULT_FALLBACK_MAX_ATTEMPTS


# --- the config loader -------------------------------------------------------


def _policy_config(tmp_path: Path, **policy: Any) -> Path:
    """A config file carrying `policy` on top of the sample's own block.

    Built from the sample rather than hand-written, so the fixture cannot drift
    from the shape the loader accepts. It is written into `tmp_path`: the shipped
    `router/config.yaml` is hand-edited and never dumped (Rule 9).
    """
    data = yaml.safe_load(ACTIVE_CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert isinstance(data["policy"], dict)
    data["policy"].update(policy)
    path = tmp_path / "fallback-policy.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_the_defaults_apply_when_the_keys_are_absent():
    config = load_config(ACTIVE_CONFIG_PATH)

    assert config.policy is not None
    assert config.policy.fallback_enabled is False
    assert config.policy.fallback_statuses == (429, 502, 503, 504)
    assert config.policy.fallback_max_attempts == 1


def test_the_shipped_config_keeps_the_feature_off():
    config = load_config(REPO_ROOT / "router" / "config.yaml")

    assert config.policy is not None
    assert config.policy.fallback_enabled is False
    assert config.policy.fallback_statuses == (429, 502, 503, 504)
    assert config.policy.fallback_max_attempts == 1


def test_the_loader_reads_the_three_keys(tmp_path):
    config = load_config(
        _policy_config(
            tmp_path,
            fallback_enabled=True,
            fallback_statuses=[500, 599],
            fallback_max_attempts=4,
        )
    )

    assert config.policy is not None
    assert config.policy.fallback_enabled is True
    assert config.policy.fallback_statuses == (500, 599)
    assert config.policy.fallback_max_attempts == 4


def test_an_empty_status_list_is_legal(tmp_path):
    config = load_config(_policy_config(tmp_path, fallback_statuses=[]))

    assert config.policy is not None
    assert config.policy.fallback_statuses == ()


@pytest.mark.parametrize(
    "policy",
    [
        {"fallback_enabled": "yes"},
        {"fallback_enabled": 1},
        {"fallback_enabled": None},
        {"fallback_statuses": 503},
        {"fallback_statuses": ["503"]},
        {"fallback_statuses": [503.5]},
        {"fallback_statuses": [True]},
        {"fallback_statuses": [None]},
        {"fallback_statuses": [0]},
        {"fallback_statuses": [99]},
        {"fallback_statuses": [600]},
        {"fallback_statuses": [503, 700]},
        {"fallback_max_attempts": 0},
        {"fallback_max_attempts": -2},
        {"fallback_max_attempts": "3"},
        {"fallback_max_attempts": None},
        {"fallback_max_attempts": 2.5},
        {"fallback_max_attempts": True},
        {"fallback_fallback": True},
    ],
)
def test_the_loader_refuses_an_illegal_fallback_value(tmp_path, policy: dict[str, Any]):
    with pytest.raises(ConfigError):
        load_config(_policy_config(tmp_path, **policy))


def test_a_policy_spec_built_by_hand_defaults_the_same_way():
    policy = PolicySpec(
        escalate_consecutive_errors=2,
        escalate_repeated_tool_calls=3,
        downgrade_enabled=True,
        downgrade_max_context_tokens=2000,
        downgrade_max_turn_index=1,
        dwell_requests=5,
        hysteresis_requests=3,
    )

    assert policy.fallback_enabled is False
    assert policy.fallback_statuses == (429, 502, 503, 504)
    assert policy.fallback_max_attempts == 1
