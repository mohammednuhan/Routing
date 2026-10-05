from __future__ import annotations

import contextlib
import copy
import http.client
import io
import json
import re
import threading
from contextlib import ExitStack
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml

from router import cli, proxy as proxy_module
from router.classifier import (
    CHEAP_KW,
    LONG_PROMPT,
    MANY_FILES,
    QUESTION_ONLY,
    STRONG_KW,
    VERY_LONG_PROMPT,
    TaskClass,
    classify_prompt,
    latest_prompt_text,
)
from router.config import (
    DEFAULT_CLASSIFIER_CHEAP,
    DEFAULT_CLASSIFIER_CHEAP_CAP,
    DEFAULT_CLASSIFIER_QUESTION_POINTS,
    DEFAULT_CLASSIFIER_STRONG,
    DEFAULT_CLASSIFIER_STRONG_CAP,
    ClassifierPoints,
    ClassifierSpec,
    ConfigError,
    load_config,
)
from router.decisions import DB_ENV_VAR, MESSAGES_PATH, DecisionLog
from router.policy import decide
from router.proxy import CLASSIFIER_FAILED, LOOPBACK_HOST, ProxySettings, create_server
from router.signals import Signals, compute_signals

SECRET_PHRASE = "purple-orchid-trombone-7731"

MODEL_LOW = "TODO_MODEL_TIER_LOW"
MODEL_MID = "TODO_MODEL_TIER_MID"
MODEL_HIGH = "TODO_MODEL_TIER_HIGH"

EFFORT_LOW = "TODO_EFFORT_LOW"
EFFORT_MEDIUM = "TODO_EFFORT_MEDIUM"
EFFORT_HIGH = "TODO_EFFORT_HIGH"

STRONG_WORDS = ("refactor", "debug", "migrate")
CHEAP_WORDS = ("typo", "lint")


# --- fixtures -------------------------------------------------------------


def enabled_spec(**overrides: Any) -> ClassifierSpec:
    """A classifier with short, unmistakable keywords and the shipped weights."""
    base = ClassifierSpec(
        enabled=True,
        strong_keywords=STRONG_WORDS,
        cheap_keywords=CHEAP_WORDS,
    )
    return replace(base, **overrides)


def config_with(spec: ClassifierSpec | None, **overrides: Any):
    """The shipped config with `classifier` replaced, keeping every comment."""
    return replace(load_config(), classifier=spec, **overrides)


@pytest.fixture()
def on():
    return config_with(enabled_spec())


def body_with(text: str) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": text}]}


def score_of(text: str, spec: ClassifierSpec) -> int:
    result = classify_prompt(body_with(text), config_with(spec))
    assert result is not None
    return result.score


# --- individual rules -----------------------------------------------------


def test_a_prompt_with_no_rule_scores_the_base():
    assert score_of("do the thing", enabled_spec()) == 50


def test_one_strong_keyword_adds_its_weight():
    assert score_of("please refactor this", enabled_spec()) == 50 + 25


def test_two_strong_keywords_add_twice_the_weight():
    assert score_of("refactor and debug", enabled_spec()) == 50 + 25 * 2


def test_strong_weight_is_capped():
    # Three keywords would be +75 uncapped; the cap holds the total at +50.
    result = classify_prompt(
        body_with("refactor debug migrate"), config_with(enabled_spec())
    )
    assert result is not None
    assert result.score == 50 + DEFAULT_CLASSIFIER_STRONG_CAP == 100


def test_the_strong_cap_default_is_fifty():
    assert DEFAULT_CLASSIFIER_STRONG == 25
    assert DEFAULT_CLASSIFIER_STRONG_CAP == 50


def test_a_repeated_keyword_counts_once():
    assert score_of("refactor refactor refactor", enabled_spec()) == 50 + 25


def test_one_cheap_keyword_removes_its_weight():
    assert score_of("fix the typo", enabled_spec()) == 50 - 20


def test_the_cheap_weight_is_capped():
    # Only two cheap keywords exist, so the cap needs a third to be reached.
    spec = enabled_spec(cheap_keywords=("typo", "lint", "wording"))
    assert score_of("typo lint wording", spec) == 50 + DEFAULT_CLASSIFIER_CHEAP_CAP


def test_the_cheap_defaults_are_negative():
    assert DEFAULT_CLASSIFIER_CHEAP == -20
    assert DEFAULT_CLASSIFIER_CHEAP_CAP == -40


def test_the_length_rules_fire_at_their_boundaries():
    spec = enabled_spec()
    assert score_of("x" * 200, spec) == 50, "exactly 200 is not over 200"
    assert score_of("x" * 201, spec) == 50 + 10
    assert score_of("x" * 600, spec) == 50 + 10, "exactly 600 is not over 600"
    assert score_of("x" * 601, spec) == 50 + 10 + 20


def test_a_very_long_prompt_keeps_both_length_reasons():
    result = classify_prompt(body_with("x" * 700), config_with(enabled_spec()))
    assert result is not None
    assert result.reason_codes == (LONG_PROMPT, VERY_LONG_PROMPT)


def test_three_or_more_file_like_tokens():
    spec = enabled_spec()
    assert score_of("look at a.py and b.py", spec) == 50, "two is not three"
    assert score_of("look at a.py b.py c.py", spec) == 50 + 15


def test_a_version_number_is_not_a_file_name():
    result = classify_prompt(
        body_with("upgrade to v1.2.3 today"), config_with(enabled_spec())
    )
    assert result is not None
    assert MANY_FILES not in result.reason_codes


def test_a_question_only_prompt_loses_points():
    spec = enabled_spec()
    assert score_of("what does this do?", spec) == 50 + DEFAULT_CLASSIFIER_QUESTION_POINTS
    assert score_of("fix it. what does this do?", spec) == 50, "not a single question"


def test_the_score_is_clamped_to_zero_and_a_hundred():
    hot = enabled_spec()
    assert score_of("refactor debug migrate " + "x" * 700, hot) == 100

    # The cheapest a prompt can score is base (50) - cheap_cap (40) - question
    # (10) = 0, which is exactly the floor: the rules never push it below.
    cold = enabled_spec(
        strong_keywords=(),
        cheap_keywords=("typo", "lint", "wording", "spell", "rename", "comment"),
    )
    assert score_of("typo lint wording spell rename comment what?", cold) == 0


# --- reason codes ---------------------------------------------------------


def test_keyword_reasons_carry_counts_not_words():
    result = classify_prompt(
        body_with("refactor and debug please"), config_with(enabled_spec())
    )
    assert result is not None
    assert result.reason_codes == (f"{STRONG_KW}:2",)
    for code in result.reason_codes:
        assert not any(word in code for word in (*STRONG_WORDS, *CHEAP_WORDS))


def test_cheap_and_strong_reasons_can_both_appear():
    spec = enabled_spec()
    result = classify_prompt(body_with("refactor the typo"), config_with(spec))
    assert result is not None
    assert result.reason_codes == (f"{STRONG_KW}:1", f"{CHEAP_KW}:1")


def test_a_clean_prompt_reports_no_reason():
    result = classify_prompt(body_with("do the thing"), config_with(enabled_spec()))
    assert result is not None
    assert result.reason_codes == ()


def test_the_question_code_appears_with_its_points():
    result = classify_prompt(body_with("why?"), config_with(enabled_spec()))
    assert result is not None
    assert QUESTION_ONLY in result.reason_codes
    assert result.score == 40


def test_the_result_holds_only_numbers_and_codes():
    result = classify_prompt(
        body_with(f"refactor {SECRET_PHRASE}"), config_with(enabled_spec())
    )
    assert isinstance(result, TaskClass)
    assert SECRET_PHRASE not in repr(result)
    assert SECRET_PHRASE not in json.dumps(result.score)
    assert SECRET_PHRASE not in " ".join(result.reason_codes)


# --- tier thresholds ------------------------------------------------------


def test_below_cheap_below_is_low():
    spec = enabled_spec()
    # 50 - 20 = 30, which is below cheap_below (35).
    assert score_of("typo", spec) == 30


def test_the_cheap_below_boundary_is_exclusive():
    spec = enabled_spec()
    assert spec.cheap_below == 35
    assert classify_prompt(body_with("typo"), config_with(spec)).tier == "low"

    # A prompt scoring exactly cheap_below is not below it.
    at_boundary = enabled_spec(cheap_below=30)
    assert score_of("typo", at_boundary) == 30
    assert classify_prompt(body_with("typo"), config_with(at_boundary)).tier == "mid"


def test_the_strong_from_boundary_is_inclusive():
    spec = enabled_spec()
    assert spec.strong_from == 65
    # 50 + 15 for three file names is exactly 65.
    result = classify_prompt(body_with("a.py b.py c.py"), config_with(spec))
    assert result is not None
    assert result.score == 65
    assert result.tier == "high"


def test_between_the_two_thresholds_is_mid():
    result = classify_prompt(body_with("do the thing"), config_with(enabled_spec()))
    assert result is not None
    assert result.tier == "mid"


def test_the_tier_names_are_the_config_cost_tiers():
    tiers = [spec.cost_tier for spec in load_config().models]
    assert tiers == ["low", "mid", "high"]


# --- finding the latest human prompt --------------------------------------


def test_a_plain_string_message_is_read():
    assert latest_prompt_text(body_with("hello")) == "hello"


def test_a_text_block_is_read():
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]}
    assert latest_prompt_text(body) == "hello"


def test_several_text_blocks_are_joined():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "refactor"},
                    {"type": "text", "text": "the parser"},
                ],
            }
        ]
    }
    assert latest_prompt_text(body) == "refactor\nthe parser"


def test_the_latest_human_prompt_wins():
    body = {
        "messages": [
            {"role": "user", "content": "the first prompt"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "the last prompt"},
        ]
    }
    assert latest_prompt_text(body) == "the last prompt"


def test_a_tool_result_only_message_is_skipped():
    """During a tool loop the last user message is tool output, not a prompt."""
    body = {
        "messages": [
            {"role": "user", "content": "refactor the parser"},
            {"role": "assistant", "content": [{"type": "tool_use", "name": "read"}]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "refactor debug migrate"},
                    {"type": "tool_result", "content": "refactor debug migrate"},
                ],
            },
        ]
    }
    assert latest_prompt_text(body) == "refactor the parser"
    result = classify_prompt(body, config_with(enabled_spec()))
    assert result is not None
    # The human prompt only: one strong keyword, not four.
    assert result.reason_codes == (f"{STRONG_KW}:1",)


def test_a_message_mixing_a_result_and_text_keeps_the_text():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "tool noise"},
                    {"type": "text", "text": "the real prompt"},
                ],
            }
        ]
    }
    assert latest_prompt_text(body) == "the real prompt"


def test_an_assistant_message_is_never_the_prompt():
    body = {
        "messages": [
            {"role": "user", "content": "the prompt"},
            {"role": "assistant", "content": "refactor debug migrate"},
        ]
    }
    assert latest_prompt_text(body) == "the prompt"


def test_a_whitespace_only_message_is_not_a_prompt():
    body = {"messages": [{"role": "user", "content": "   \n  "}]}
    assert latest_prompt_text(body) is None
    assert classify_prompt(body, config_with(enabled_spec())) is None


# --- None cases -----------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        None,
        "not json at all",
        [],
        {},
        {"messages": []},
        {"messages": "not a list"},
        {"messages": [{"role": "assistant", "content": "only the assistant"}]},
        {"messages": [{"role": "user", "content": []}]},
        {"messages": [{"role": "user"}]},
        {"messages": ["not even a message"]},
    ],
)
def test_no_human_prompt_classifies_to_none(body):
    assert classify_prompt(body, config_with(enabled_spec())) is None


def test_a_disabled_section_classifies_to_none():
    off = config_with(enabled_spec(enabled=False))
    assert classify_prompt(body_with("refactor debug migrate"), off) is None


def test_an_absent_section_classifies_to_none():
    assert classify_prompt(body_with("refactor debug migrate"), config_with(None)) is None


def test_the_shipped_config_has_the_classifier_disabled():
    spec = load_config().classifier
    assert spec is not None
    assert spec.enabled is False


# --- read-only on the body ------------------------------------------------


def test_the_parsed_body_is_not_modified(on):
    body = body_with("refactor the parser, a.py and b.py")
    before = copy.deepcopy(body)
    classify_prompt(body, on)
    assert body == before


# --- the config section ---------------------------------------------------


def test_the_shipped_section_round_trips():
    loaded = load_config()
    spec = loaded.classifier
    assert spec is not None
    assert spec.enabled is False
    assert spec.strong_keywords and spec.cheap_keywords
    assert spec.cheap_below == 35
    assert spec.strong_from == 65
    assert spec.points == ClassifierPoints()


def test_the_shipped_config_keeps_its_comments():
    text = DEFAULT_CONFIG_TEXT
    assert "# tamias-router configuration." in text
    assert "PLACEHOLDER" in text
    assert "policy:" in text
    assert "classifier:" in text


DEFAULT_CONFIG_TEXT = (
    Path(__file__).resolve().parents[2] / "router" / "config.yaml"
).read_text(encoding="utf-8")


def write_config(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def base_data() -> dict[str, Any]:
    data = yaml.safe_load(DEFAULT_CONFIG_TEXT)
    data["classifier"] = {"enabled": True}
    return data


def test_a_minimal_section_is_loadable(tmp_path):
    loaded = load_config(write_config(tmp_path, base_data()))

    assert loaded.classifier is not None
    assert loaded.classifier.enabled is True
    assert loaded.classifier.strong_keywords == ()
    assert loaded.classifier.points == ClassifierPoints()


def test_an_absent_section_is_none(tmp_path):
    data = base_data()
    del data["classifier"]
    assert load_config(write_config(tmp_path, data)).classifier is None


def test_a_null_section_is_none(tmp_path):
    data = base_data()
    data["classifier"] = None
    assert load_config(write_config(tmp_path, data)).classifier is None


def test_a_full_section_loads_every_weight(tmp_path):
    data = base_data()
    data["classifier"] = {
        "enabled": True,
        "strong_keywords": ["refactor", "debug"],
        "cheap_keywords": ["typo"],
        "points": {"base": 10, "strong": 5, "many_files": 2},
        "cheap_below": 20,
        "strong_from": 40,
    }
    loaded = load_config(write_config(tmp_path, data))

    spec = loaded.classifier
    assert spec is not None
    assert spec.strong_keywords == ("refactor", "debug")
    assert spec.cheap_keywords == ("typo",)
    assert spec.points.base == 10
    assert spec.points.strong == 5
    assert spec.points.many_files == 2
    assert spec.points.cheap == DEFAULT_CLASSIFIER_CHEAP, "an absent weight keeps its default"
    assert spec.cheap_below == 20
    assert spec.strong_from == 40


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ({"enabled": "yes"}, "'enabled' must be boolean"),
        ({"enabled": 1}, "'enabled' must be boolean"),
        ({"strong_keywords": "refactor"}, "must be a list of strings"),
        ({"strong_keywords": ["refactor", 7]}, re.escape("strong_keywords[1]")),
        ({"strong_keywords": [""]}, re.escape("strong_keywords[0]")),
        ({"cheap_keywords": {"a": 1}}, "must be a list of strings"),
        ({"points": [50]}, "'points' must be a mapping"),
        ({"points": {"weight": 3}}, "unknown classifier points key 'weight'"),
        ({"points": {"base": "50"}}, "points 'base' must be an integer"),
        ({"points": {"base": True}}, "points 'base' must be an integer"),
        ({"cheap_below": "35"}, "'cheap_below' must be an integer"),
        ({"cheap_below": -1}, "'cheap_below' must be in 0..100"),
        ({"strong_from": 101}, "'strong_from' must be in 0..100"),
        ({"cheap_below": 70, "strong_from": 40}, "must not exceed"),
        ({"rule_weight": 1}, "unknown classifier key 'rule_weight'"),
    ],
)
def test_a_bad_section_is_rejected(tmp_path, section, message):
    data = base_data()
    data["classifier"] = section
    with pytest.raises(ConfigError, match=message):
        load_config(write_config(tmp_path, data))


def test_a_section_that_is_not_a_mapping_is_rejected(tmp_path):
    data = base_data()
    data["classifier"] = ["enabled"]
    with pytest.raises(ConfigError, match="'classifier' must be a mapping"):
        load_config(write_config(tmp_path, data))


# --- disabled changes nothing ---------------------------------------------


def test_a_disabled_classifier_sets_no_signal():
    signals = Signals()
    assert signals.classifier_score is None
    assert signals.classifier_tier is None
    assert signals.to_dict()["classifier_score"] is None
    assert signals.to_dict()["classifier_tier"] is None


def test_a_disabled_classifier_does_not_change_the_downgrade_decision():
    """Rule (d) is untouched: with no classifier the old path still answers."""
    off = config_with(enabled_spec(enabled=False))
    signals = Signals(
        context_tokens_estimate=1000, turn_index=1, consecutive_tool_errors=0
    )
    decision = decide(signals, MODEL_MID, None, off)

    assert decision.action == "SWITCH"
    assert decision.target_model == MODEL_LOW
    assert decision.reason_codes == ["DOWNGRADE_SMALL_CONTEXT"]


def test_a_disabled_classifier_leaves_compute_signals_untouched():
    signals = compute_signals({"messages": [{"role": "user", "content": "refactor"}]})
    assert signals.classifier_score is None
    assert signals.classifier_tier is None


# --- the policy rule ------------------------------------------------------


def classifier_signals(tier: str, **overrides: Any) -> Signals:
    return replace(
        Signals(
            context_tokens_estimate=1000,
            turn_index=1,
            consecutive_tool_errors=0,
            classifier_score=70,
            classifier_tier=tier,
        ),
        **overrides,
    )


def test_a_high_tier_prompt_switches_up(on):
    decision = decide(classifier_signals("high"), MODEL_MID, None, on)

    assert decision.action == "SWITCH"
    assert decision.target_model == MODEL_HIGH
    assert decision.reason_codes == ["CLASSIFIER_HIGH"]
    assert decision.direction == "up"


def test_a_low_tier_prompt_switches_down(on):
    decision = decide(classifier_signals("low"), MODEL_MID, None, on)

    assert decision.action == "SWITCH"
    assert decision.target_model == MODEL_LOW
    assert decision.reason_codes == ["CLASSIFIER_LOW"]
    assert decision.direction == "down"


def test_a_mid_tier_prompt_names_its_reason(on):
    decision = decide(classifier_signals("mid"), MODEL_HIGH, None, on)

    assert decision.action == "SWITCH"
    assert decision.target_model == MODEL_MID
    assert decision.reason_codes == ["CLASSIFIER_MID"]
    assert decision.direction == "down"


@pytest.mark.parametrize("tier", ["low", "mid", "high"])
def test_matching_the_tier_stays(on, tier):
    requested = {"low": MODEL_LOW, "mid": MODEL_MID, "high": MODEL_HIGH}[tier]
    decision = decide(classifier_signals(tier), requested, None, on)

    assert decision.action == "STAY"
    assert decision.target_model == requested
    assert decision.reason_codes == ["CLASSIFIER_MATCHES_TIER"]
    assert decision.direction is None


def test_the_target_is_the_first_model_of_that_tier_in_config_order():
    models = (
        replace(load_config().models[0], id="FIRST_LOW"),
        replace(load_config().models[0], id="SECOND_LOW"),
        load_config().models[1],
        load_config().models[2],
    )
    config = replace(
        load_config(), models=models, classifier=ClassifierSpec(enabled=True)
    )
    decision = decide(classifier_signals("low"), MODEL_MID, None, config)

    assert decision.target_model == "FIRST_LOW"


def test_the_requested_effort_is_carried_into_the_switch(on):
    decision = decide(classifier_signals("high"), MODEL_MID, EFFORT_LOW, on)

    assert decision.action == "SWITCH"
    assert decision.target_model == MODEL_HIGH
    assert decision.target_effort == EFFORT_LOW


def test_an_illegal_effort_pair_stays(on):
    decision = decide(classifier_signals("high"), MODEL_MID, "NOT_LEGAL", on)

    assert decision.action == "STAY"
    assert decision.target_model == MODEL_MID
    assert decision.reason_codes == ["ILLEGAL_PAIR"]


def test_a_tier_no_model_carries_stays(on):
    """Rule 7: no model is invented to satisfy a tier the config cannot supply."""
    models = (load_config().models[0], load_config().models[1])
    config = replace(load_config(), models=models, classifier=ClassifierSpec(enabled=True))

    decision = decide(classifier_signals("high"), MODEL_LOW, None, config)

    assert decision.action == "STAY"
    assert decision.target_model == MODEL_LOW
    assert decision.reason_codes == ["CLASSIFIER_NO_TARGET"]


def test_tool_errors_beat_the_classifier(on):
    decision = decide(
        classifier_signals("low", consecutive_tool_errors=5), MODEL_MID, None, on
    )

    assert decision.reason_codes == ["ESCALATE_TOOL_ERRORS"]
    assert decision.target_model == MODEL_HIGH


def test_repeated_tool_calls_beat_the_classifier(on):
    decision = decide(
        classifier_signals("low", repeated_tool_call_count=5), MODEL_MID, None, on
    )

    assert decision.reason_codes == ["ESCALATE_REPEATED_TOOLS"]
    assert decision.target_model == MODEL_HIGH


def test_a_disabled_classifier_leaves_the_downgrade_rule_in_place():
    off = config_with(enabled_spec(enabled=False))
    decision = decide(classifier_signals("high"), MODEL_MID, None, off)

    assert decision.reason_codes == ["DOWNGRADE_SMALL_CONTEXT"]
    assert decision.target_model == MODEL_LOW


def test_a_none_tier_leaves_the_downgrade_rule_in_place(on):
    decision = decide(
        Signals(context_tokens_estimate=1000, turn_index=1, consecutive_tool_errors=0),
        MODEL_MID,
        None,
        on,
    )

    assert decision.reason_codes == ["DOWNGRADE_SMALL_CONTEXT"]
    assert decision.target_model == MODEL_LOW


def test_mode_off_still_wins(on):
    off = config_with(enabled_spec(), mode="off")
    decision = decide(classifier_signals("high"), MODEL_MID, None, off)

    assert decision.reason_codes == ["MODE_OFF"]


def test_an_unknown_model_is_never_routed(on):
    decision = decide(classifier_signals("high"), "NOT_A_MODEL", None, on)

    assert decision.action == "STAY"
    assert decision.reason_codes == ["UNKNOWN_MODEL"]


def test_the_classifier_decision_is_deterministic(on):
    first = decide(classifier_signals("high"), MODEL_MID, None, on)
    second = decide(classifier_signals("high"), MODEL_MID, None, on)

    assert first == second


# --- the proxy ------------------------------------------------------------


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
def running_proxy(config: Any, db_path: Path | None) -> Iterator[Harness]:
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
            config=config,
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
        connection.request(
            "POST", path, body=body, headers={"Content-Type": "application/json"}
        )
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


def wait_for_stderr(captured: io.StringIO, timeout: float = 5.0) -> bool:
    """Wait until the handler has written its one request line.

    The line is written after the response has been relayed, so it can land a
    moment after the client has its answer. Waiting for it is what keeps the
    "the phrase is not in the log" assertions from passing on an empty capture.
    """
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if captured.getvalue():
            return True
        time.sleep(0.01)
    return False


def messages(model: str, text: str) -> bytes:
    return json.dumps({"model": model, "messages": [{"role": "user", "content": text}]}).encode()


def test_the_proxy_fills_the_classifier_signals(tmp_path, monkeypatch):
    body = messages(MODEL_MID, "refactor the parser, a.py and b.py")
    with db_at(tmp_path, monkeypatch) as log, running_proxy(config_with(enabled_spec()), log.db_path) as harness:
        assert post(harness.proxy_port, MESSAGES_PATH, body)[0] == 200
        assert wait_for_rows(log, 1) == 1

        row = log.recent(1)[0]
        assert row.signal_values["classifier_tier"] == "high"
        assert isinstance(row.signal_values["classifier_score"], int)
        assert row.reason_codes == ["CLASSIFIER_HIGH"]
        assert row.action == "SWITCH"
        assert row.chosen_model == MODEL_HIGH
        assert row.applied == 0, "shadow mode never rewrites"
        assert harness.recorded[-1].body == body, "the forwarded bytes are untouched"


def test_the_proxy_records_none_when_the_classifier_is_disabled(tmp_path, monkeypatch):
    body = messages(MODEL_MID, "refactor debug migrate")
    config = config_with(enabled_spec(enabled=False))
    with db_at(tmp_path, monkeypatch) as log, running_proxy(config, log.db_path) as harness:
        assert post(harness.proxy_port, MESSAGES_PATH, body)[0] == 200
        assert wait_for_rows(log, 1) == 1

        row = log.recent(1)[0]
        assert row.signal_values["classifier_score"] is None
        assert row.signal_values["classifier_tier"] is None
        assert row.reason_codes == ["DOWNGRADE_SMALL_CONTEXT"]
        assert harness.recorded[-1].body == body


def test_the_latest_human_prompt_is_classified_during_a_tool_loop(tmp_path, monkeypatch):
    body = json.dumps(
        {
            "model": MODEL_MID,
            "messages": [
                {"role": "user", "content": "refactor the parser, a.py and b.py"},
                {"role": "assistant", "content": [{"type": "tool_use", "name": "read"}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "content": "typo lint typo lint"},
                    ],
                },
            ],
        }
    ).encode()
    with db_at(tmp_path, monkeypatch) as log, running_proxy(config_with(enabled_spec()), log.db_path) as harness:
        assert post(harness.proxy_port, MESSAGES_PATH, body)[0] == 200
        assert wait_for_rows(log, 1) == 1

        row = log.recent(1)[0]
        assert row.reason_codes == ["CLASSIFIER_HIGH"], "the tool output is not the prompt"
        assert harness.recorded[-1].body == body


def break_classifier(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("classifier exploded")


def test_a_classifier_failure_never_affects_forwarding(tmp_path, monkeypatch):
    monkeypatch.setattr(proxy_module, "classify_prompt", break_classifier)
    body = json.dumps(
        {
            "model": MODEL_MID,
            "messages": [
                {"role": "user", "content": f"please refactor {SECRET_PHRASE}"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_result", "is_error": True},
                        {"type": "tool_result", "is_error": True},
                    ],
                },
            ],
        }
    ).encode()
    config = config_with(enabled_spec())
    with db_at(tmp_path, monkeypatch) as log, running_proxy(config, log.db_path) as harness:
        status, payload = post(harness.proxy_port, MESSAGES_PATH, body)

    assert status == 200, "the request still gets its upstream response"
    assert json.loads(payload) == {"ok": True}
    assert wait_for_rows(log, 1) == 1
    row = log.recent(1)[0]
    assert row.error == CLASSIFIER_FAILED
    assert row.signal_values["classifier_score"] is None
    assert row.signal_values["classifier_tier"] is None
    # The rest of the router is untouched: the tool errors still escalate.
    assert row.action == "SWITCH"
    assert row.reason_codes == ["ESCALATE_TOOL_ERRORS"]
    assert row.applied == 0
    assert harness.recorded[-1].body == body, "the original bytes went upstream"


def test_a_classifier_failure_never_raises_to_the_client(tmp_path, monkeypatch):
    monkeypatch.setattr(proxy_module, "classify_prompt", break_classifier)
    body = messages(MODEL_MID, f"please refactor {SECRET_PHRASE}")
    config = config_with(enabled_spec())
    with db_at(tmp_path, monkeypatch) as log, running_proxy(config, log.db_path) as harness:
        status, _ = post(harness.proxy_port, MESSAGES_PATH, body)

        assert status == 200
        assert wait_for_rows(log, 1) == 1
        assert log.recent(1)[0].error == CLASSIFIER_FAILED
        assert harness.recorded[-1].body == body


# --- the secret phrase never leaves ---------------------------------------


def secret_body(model: str = MODEL_MID) -> bytes:
    return messages(model, f"please refactor the parser {SECRET_PHRASE} a.py b.py c.py")


@pytest.fixture()
def secret_capture(tmp_path, monkeypatch) -> dict[str, Any]:
    """Run one secret-bearing request through the proxy, keeping every output."""
    captured = io.StringIO()
    config = config_with(enabled_spec())
    with ExitStack() as stack:
        log = stack.enter_context(db_at(tmp_path, monkeypatch))
        harness = stack.enter_context(running_proxy(config, log.db_path))
        with contextlib.redirect_stderr(captured):
            status, _ = post(harness.proxy_port, MESSAGES_PATH, secret_body())
            assert status == 200
            assert wait_for_rows(log, 1) == 1
            assert wait_for_stderr(captured), "the proxy logs one line per request"
        yield {
            "log": log,
            "row": log.recent(1)[0],
            "stderr": captured.getvalue(),
            "forwarded": harness.recorded[-1].body,
        }


def test_the_secret_phrase_is_not_in_signal_values(secret_capture):
    row = secret_capture["row"]
    assert row.signal_values, "the classifier did fill signals in"
    assert row.signal_values["classifier_tier"] == "high"
    assert SECRET_PHRASE not in json.dumps(row.signal_values)


def test_the_secret_phrase_is_not_in_the_database_bytes(secret_capture):
    db_bytes = secret_capture["log"].db_path.read_bytes()
    assert db_bytes, "a row was really written"
    assert SECRET_PHRASE.encode() not in db_bytes


def test_the_secret_phrase_is_not_in_the_captured_log(secret_capture):
    captured = secret_capture["stderr"]
    assert captured, "the proxy really logged its request line"
    assert SECRET_PHRASE not in captured


def test_the_secret_phrase_is_not_in_the_signals_repr():
    signals = proxy_module.with_classifier(
        {"messages": [{"role": "user", "content": f"refactor {SECRET_PHRASE}"}]},
        compute_signals({"messages": [{"role": "user", "content": "hi"}]}),
        config_with(enabled_spec()),
    )
    assert signals is not None
    assert signals.classifier_tier == "high"
    assert SECRET_PHRASE not in repr(signals)


def test_the_secret_phrase_is_not_in_the_reason_codes(secret_capture):
    row = secret_capture["row"]
    assert SECRET_PHRASE not in " ".join(row.reason_codes)
    assert SECRET_PHRASE not in json.dumps(row.reason_codes)


def test_the_forwarded_body_is_the_clients_own_bytes(secret_capture):
    assert secret_capture["forwarded"] == secret_body()


# --- the CLI dry run ------------------------------------------------------


def test_classify_prints_the_score_tier_and_reasons(tmp_path, capsys):
    data = base_data()
    data["classifier"] = {"enabled": True, "strong_keywords": ["refactor"]}
    path = write_config(tmp_path, data)

    code = cli.main(["--config", str(path), "classify", "please refactor the parser"])

    assert code == 0
    out = capsys.readouterr().out
    assert "75" in out
    assert "high" in out
    assert f"{STRONG_KW}:1" in out


def test_classify_never_prints_the_prompt_or_a_matched_word(tmp_path, capsys):
    data = base_data()
    data["classifier"] = {"enabled": True, "strong_keywords": ["refactor"]}
    path = write_config(tmp_path, data)

    cli.main(["--config", str(path), "classify", f"please refactor {SECRET_PHRASE}"])

    printed = capsys.readouterr()
    assert SECRET_PHRASE not in printed.out
    assert SECRET_PHRASE not in printed.err
    assert "refactor" not in printed.out
    assert "refactor" not in printed.err


def test_classify_stores_nothing(tmp_path, monkeypatch):
    data = base_data()
    data["classifier"] = {"enabled": True, "strong_keywords": ["refactor"]}
    path = write_config(tmp_path, data)
    db = tmp_path / "router.sqlite3"
    monkeypatch.setenv(DB_ENV_VAR, str(db))

    cli.main(["--config", str(path), "classify", "please refactor the parser"])

    assert not db.exists(), "a dry run must not open the decision log"


def test_classify_refuses_while_the_section_is_disabled(capsys):
    code = cli.main(["classify", "please refactor the parser"])

    assert code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "disabled" in captured.err


def test_classify_reports_a_prompt_with_no_text(tmp_path, monkeypatch, capsys):
    """An empty prompt is a legal string, but it is not a human prompt."""
    data = base_data()
    data["classifier"] = {"enabled": True}
    path = write_config(tmp_path, data)

    code = cli.main(["--config", str(path), "classify", "   "])

    assert code == 1
    assert "no human prompt" in capsys.readouterr().err