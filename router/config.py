"""Load and validate router/config.yaml.

Validation is strict and fails closed: anything the config file does not
establish is an error, never a default and never a guess. This module makes no
network calls and logs nothing.

Rules that govern this package are in `router/AGENTS.md`.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

Mode = Literal["shadow", "active", "off"]

#: The only modes the router accepts.
LEGAL_MODES: tuple[str, ...] = ("shadow", "active", "off")

#: The mode the router starts in. Shadow observes and changes nothing.
DEFAULT_MODE: Mode = "shadow"

#: The router may listen on the IPv4 loopback interface and nothing else. Any
#: other host is rejected at load time rather than warned about.
LEGAL_LISTEN_HOST: str = "127.0.0.1"

#: The config file shipped inside the package.
DEFAULT_CONFIG_PATH: Path = Path(__file__).with_name("config.yaml")

#: Consecutive internal router errors that open the circuit breaker.
DEFAULT_BREAKER_ERROR_THRESHOLD = 3

#: Seconds the breaker stays open before the router is allowed to try again.
DEFAULT_BREAKER_COOLDOWN_SECONDS = 60

#: Switches one session may make before the cap blocks the next one.
DEFAULT_MAX_SWITCHES_PER_SESSION = 10

#: Requests one model hold may serve before it is released.
DEFAULT_HOLD_MAX_REQUESTS = 50

# --- Classifier defaults -------------------------------------------------
#
# The classifier is additive and off by default. Every number below is a plain
# rule weight, not a measurement: it is read in `router/classifier.py` and
# nothing else, and none of them can cause a request to change on their own.

#: Score a prompt starts from before any rule is applied.
DEFAULT_CLASSIFIER_BASE = 50

#: Added per distinct strong keyword, never more than `strong_cap` in total.
DEFAULT_CLASSIFIER_STRONG = 25
DEFAULT_CLASSIFIER_STRONG_CAP = 50

#: Added per distinct cheap keyword. Negative, and never below `cheap_cap`.
DEFAULT_CLASSIFIER_CHEAP = -20
DEFAULT_CLASSIFIER_CHEAP_CAP = -40

#: Length thresholds, in characters, and the points each one is worth.
DEFAULT_CLASSIFIER_LONG_CHARS = 200
DEFAULT_CLASSIFIER_LONG_POINTS = 10
DEFAULT_CLASSIFIER_VERY_LONG_CHARS = 600
DEFAULT_CLASSIFIER_VERY_LONG_POINTS = 20

#: How many file-like tokens make a prompt a file-shaped task, and its points.
DEFAULT_CLASSIFIER_MANY_FILES = 3
DEFAULT_CLASSIFIER_MANY_FILES_POINTS = 15

#: Points removed when the whole prompt is a single question.
DEFAULT_CLASSIFIER_QUESTION_POINTS = -10

#: Score at or above which a prompt is tier `high`.
DEFAULT_CLASSIFIER_STRONG_FROM = 65

#: Score below which a prompt is tier `low`. Between the two it is `mid`.
DEFAULT_CLASSIFIER_CHEAP_BELOW = 35

#: The tier names a score maps to. They are the same `cost_tier` labels the
#: models carry, so a classified tier names a tier a model can already have.
LEGAL_CLASSIFIER_TIERS: tuple[str, ...] = ("low", "mid", "high")

#: Keys the optional `classifier` section accepts. Anything else is rejected, so
#: a misspelled key cannot silently disable a rule somebody believes is running.
_CLASSIFIER_KEYS: frozenset[str] = frozenset(
    {
        "enabled",
        "strong_keywords",
        "cheap_keywords",
        "points",
        "cheap_below",
        "strong_from",
    }
)

#: Keys the `classifier.points` block accepts. Each is optional and falls back to
#: the `DEFAULT_CLASSIFIER_*` constant of the same name.
_CLASSIFIER_POINT_KEYS: tuple[str, ...] = (
    "base",
    "strong",
    "strong_cap",
    "cheap",
    "cheap_cap",
    "long_chars",
    "long_points",
    "very_long_chars",
    "very_long_points",
    "many_files",
    "many_files_points",
    "question_points",
)

#: A score is clamped to this range by `router/classifier.py`, so a tier
#: boundary outside it could never be reached. Declared here too, because the
#: boundaries are validated at load time and config must not import the module
#: that reads it.
MIN_CLASSIFIER_SCORE = 0
MAX_CLASSIFIER_SCORE = 100


class ConfigError(Exception):
    """The config file is missing, unreadable, malformed, or illegal."""


@dataclass(frozen=True)
class ModelSpec:
    """One legal model: its id, the efforts legal for it, and its cost tier.

    Every price is optional and defaults to None. None means "unknown", never
    zero: a switch whose cost cannot be computed is blocked, not allowed, and a
    response whose cost cannot be computed is reported as unknown rather than as
    free. A price of 0.0 is a real price - a model that is genuinely free - and
    is kept as 0.0, never turned into None.
    """

    id: str
    legal_efforts: tuple[str, ...]
    cost_tier: str
    input_per_million: float | None = None
    output_per_million: float | None = None
    cache_write_per_million: float | None = None
    cache_read_per_million: float | None = None


@dataclass(frozen=True)
class PolicySpec:
    """Policy parameters.

    The fields below `hysteresis_requests` govern the safety layer, which can
    only BLOCK a switch. None of them can cause one.
    """

    escalate_consecutive_errors: int
    escalate_repeated_tool_calls: int
    downgrade_enabled: bool
    downgrade_max_context_tokens: int
    downgrade_max_turn_index: int
    dwell_requests: int = 5
    hysteresis_requests: int = 3
    safety_margin_usd: float = 0.0
    escalate_benefit_usd: float | None = None
    downgrade_benefit_usd: float | None = None
    cost_check_enabled: bool = True
    block_in_tool_loop: bool = False
    breaker_error_threshold: int = DEFAULT_BREAKER_ERROR_THRESHOLD
    breaker_cooldown_seconds: int = DEFAULT_BREAKER_COOLDOWN_SECONDS
    max_switches_per_session: int = DEFAULT_MAX_SWITCHES_PER_SESSION
    hold_max_requests: int = DEFAULT_HOLD_MAX_REQUESTS


@dataclass(frozen=True)
class ClassifierPoints:
    """Rule weights for the classifier. Integers, never guesses.

    `cheap` and `cheap_cap` are negative on purpose: a cheap keyword pushes the
    score down, and `cheap_cap` is the floor that keeps a prompt full of them
    from reaching zero by arithmetic alone.
    """

    base: int = DEFAULT_CLASSIFIER_BASE
    strong: int = DEFAULT_CLASSIFIER_STRONG
    strong_cap: int = DEFAULT_CLASSIFIER_STRONG_CAP
    cheap: int = DEFAULT_CLASSIFIER_CHEAP
    cheap_cap: int = DEFAULT_CLASSIFIER_CHEAP_CAP
    long_chars: int = DEFAULT_CLASSIFIER_LONG_CHARS
    long_points: int = DEFAULT_CLASSIFIER_LONG_POINTS
    very_long_chars: int = DEFAULT_CLASSIFIER_VERY_LONG_CHARS
    very_long_points: int = DEFAULT_CLASSIFIER_VERY_LONG_POINTS
    many_files: int = DEFAULT_CLASSIFIER_MANY_FILES
    many_files_points: int = DEFAULT_CLASSIFIER_MANY_FILES_POINTS
    question_points: int = DEFAULT_CLASSIFIER_QUESTION_POINTS


@dataclass(frozen=True)
class ClassifierSpec:
    """The optional `classifier` section. Absent means disabled, never enabled."""

    enabled: bool = False
    strong_keywords: tuple[str, ...] = ()
    cheap_keywords: tuple[str, ...] = ()
    points: ClassifierPoints = ClassifierPoints()
    cheap_below: int = DEFAULT_CLASSIFIER_CHEAP_BELOW
    strong_from: int = DEFAULT_CLASSIFIER_STRONG_FROM


@dataclass(frozen=True)
class RouterConfig:
    """A validated router configuration."""

    listen_host: str
    listen_port: int
    upstream: str
    mode: Mode
    models: tuple[ModelSpec, ...]
    default_model: str
    default_effort: str
    policy: PolicySpec | None = None
    classifier: ClassifierSpec | None = None
    routed_header: bool = False
    source: Path | None = None

    @property
    def listen(self) -> str:
        """The listen address as `host:port`."""
        return f"{self.listen_host}:{self.listen_port}"

    @property
    def model_ids(self) -> tuple[str, ...]:
        """Every legal model id, in config order."""
        return tuple(spec.id for spec in self.models)

    def is_legal(self, model: str, effort: str) -> bool:
        """True only when `model` is in the config and `effort` is legal for it."""
        for spec in self.models:
            if spec.id == model:
                return effort in spec.legal_efforts
        return False


def parse_listen(value: str) -> tuple[str, int]:
    """Split `host:port` and enforce loopback-only.

    Anything other than 127.0.0.1 is an error: a non-loopback listen address
    would expose the router beyond this machine.
    """
    text = value.strip()
    host, separator, port_text = text.rpartition(":")
    if not separator:
        raise ConfigError(f"listen must be 'host:port', got {value!r}")
    if host != LEGAL_LISTEN_HOST:
        raise ConfigError(
            f"listen host must be {LEGAL_LISTEN_HOST}, got {host!r} "
            "(the router listens on loopback only)"
        )
    try:
        port = int(port_text, 10)
    except ValueError:
        raise ConfigError(f"listen port must be an integer, got {port_text!r}") from None
    if not 1 <= port <= 65535:
        raise ConfigError(f"listen port must be in 1..65535, got {port}")
    return host, port


def load_config(path: Path | None = None) -> RouterConfig:
    """Load and validate the router config.

    `path` defaults to the `config.yaml` shipped inside the package. Raises
    `ConfigError` with a message naming the offending key and value.
    """
    source = DEFAULT_CONFIG_PATH if path is None else Path(path)
    where = str(source)

    data = _require_mapping(_parse_yaml(_read_text(source), where), f"{where}: top level")

    listen_host, listen_port = parse_listen(_require_str(data, "listen", where))
    upstream = _require_str(data, "upstream", where)
    mode = _parse_mode(_require_key(data, "mode", where))
    models = _parse_models(_require_key(data, "models", where))
    default_model = _require_str(data, "default_model", where)
    default_effort = _require_str(data, "default_effort", where)
    routed_header = _parse_flag(data, "routed_header", where)

    policy = _parse_policy(data, where)
    classifier = _parse_classifier(data, where)

    by_id = {spec.id: spec for spec in models}
    if default_model not in by_id:
        raise ConfigError(
            f"default_model {default_model!r} is not in models "
            f"(legal ids: {', '.join(by_id)})"
        )
    spec = by_id[default_model]
    if default_effort not in spec.legal_efforts:
        raise ConfigError(
            f"default_effort {default_effort!r} is not legal for model "
            f"{default_model!r} (legal efforts: {', '.join(spec.legal_efforts)})"
        )

    return RouterConfig(
        listen_host=listen_host,
        listen_port=listen_port,
        upstream=upstream,
        mode=mode,
        models=models,
        default_model=default_model,
        default_effort=default_effort,
        policy=policy,
        classifier=classifier,
        routed_header=routed_header,
        source=source,
    )


def load_config_or_exit(path: Path | None = None) -> RouterConfig:
    """Load the config, or print a clear error to stderr and exit non-zero."""
    try:
        return load_config(path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _read_text(source: Path) -> str:
    try:
        return source.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {source}") from None
    except OSError as exc:
        raise ConfigError(f"cannot read config file {source}: {exc}") from exc


def _parse_yaml(text: str, where: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{where}: cannot parse YAML: {exc}") from exc


def _require_mapping(data: Any, where: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{where} must be a mapping, got {type(data).__name__}")
    return data


def _require_key(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{where} is missing required key {key!r}")
    return mapping[key]


def _require_str(mapping: dict[str, Any], key: str, where: str) -> str:
    value = _require_key(mapping, key, where)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} key {key!r} must be a non-empty string, got {value!r}")
    return value


def _parse_flag(data: dict[str, Any], key: str, where: str) -> bool:
    """An optional top-level boolean. Absent means false, never true."""
    if key not in data:
        return False
    value = data[key]
    if not isinstance(value, bool):
        raise ConfigError(f"{where} key {key!r} must be boolean, got {value!r}")
    return value


def _parse_mode(value: Any) -> Mode:
    if not isinstance(value, str) or value not in LEGAL_MODES:
        raise ConfigError(f"unknown mode {value!r} (legal modes: {', '.join(LEGAL_MODES)})")
    return cast(Mode, value)


def _parse_models(value: Any) -> tuple[ModelSpec, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError("'models' must be a non-empty list")
    models = tuple(_parse_model(raw, index) for index, raw in enumerate(value))
    seen: set[str] = set()
    for spec in models:
        if spec.id in seen:
            raise ConfigError(f"duplicate model id {spec.id!r} in 'models'")
        seen.add(spec.id)
    return models


def _parse_model(raw: Any, index: int) -> ModelSpec:
    where = f"models[{index}]"
    data = _require_mapping(raw, where)
    model_id = _require_str(data, "id", where)
    cost_tier = _require_str(data, "cost_tier", where)

    raw_efforts = _require_key(data, "legal_efforts", where)
    if not isinstance(raw_efforts, list) or not raw_efforts:
        raise ConfigError(f"{where} key 'legal_efforts' must be a non-empty list")

    efforts: list[str] = []
    for position, effort in enumerate(raw_efforts):
        if not isinstance(effort, str) or not effort.strip():
            raise ConfigError(
                f"{where} legal_efforts[{position}] must be a non-empty string, "
                f"got {effort!r}"
            )
        if effort not in efforts:
            efforts.append(effort)

    return ModelSpec(
        id=model_id,
        legal_efforts=tuple(efforts),
        cost_tier=cost_tier,
        input_per_million=_parse_price(data, "input_per_million", where),
        output_per_million=_parse_price(data, "output_per_million", where),
        cache_write_per_million=_parse_price(data, "cache_write_per_million", where),
        cache_read_per_million=_parse_price(data, "cache_read_per_million", where),
    )


def _parse_price(data: dict[str, Any], key: str, where: str) -> float | None:
    """A per-million price: a non-negative number, or null for "unknown".

    Absent and null both mean unknown. A string is rejected rather than parsed,
    so a price can never be a number that nobody verified. Zero is a price, not
    an absence: a model nobody pays for is free, and reporting that as "unknown"
    would be wrong in the opposite direction.
    """
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} key {key!r} must be a number or null, got {value!r}")
    if value < 0:
        raise ConfigError(f"{where} key {key!r} must not be negative, got {value!r}")
    return float(value)


def _parse_policy(data: dict[str, Any], where: str) -> PolicySpec | None:
    if "policy" not in data:
        return None
    policy = data["policy"]
    if not isinstance(policy, dict):
        raise ConfigError(f"{where} key 'policy' must be a mapping")
    known = {
        "escalate_consecutive_errors",
        "escalate_repeated_tool_calls",
        "downgrade_enabled",
        "downgrade_max_context_tokens",
        "downgrade_max_turn_index",
        "dwell_requests",
        "hysteresis_requests",
        "safety_margin_usd",
        "escalate_benefit_usd",
        "downgrade_benefit_usd",
        "cost_check_enabled",
        "block_in_tool_loop",
        "breaker_error_threshold",
        "breaker_cooldown_seconds",
        "max_switches_per_session",
        "hold_max_requests",
    }
    for k in policy:
        if k not in known:
            raise ConfigError(f"unknown policy key {k!r}")

    def pos_int(key: str) -> int:
        v = policy.get(key)
        if not isinstance(v, int) or isinstance(v, bool):
            raise ConfigError(f"policy {key!r} must be a positive integer, got {v!r}")
        if v <= 0:
            raise ConfigError(f"policy {key!r} must be a positive integer, got {v!r}")
        return v

    def pos_int_or(key: str, default: int) -> int:
        """A positive integer, or `default` when the key is absent.

        Absent means the shipped default, so a config written before a key
        existed keeps loading rather than being rejected.
        """
        if key not in policy:
            return default
        return pos_int(key)

    def flag(key: str, default: bool) -> bool:
        v = policy.get(key, default)
        if not isinstance(v, bool):
            raise ConfigError(f"policy {key!r} must be boolean, got {v!r}")
        return v

    def amount(key: str, default: float | None) -> float | None:
        v = policy.get(key, default)
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ConfigError(f"policy {key!r} must be a number or null, got {v!r}")
        if v < 0:
            raise ConfigError(f"policy {key!r} must not be negative, got {v!r}")
        return float(v)

    downgrade_enabled = policy.get("downgrade_enabled")
    if not isinstance(downgrade_enabled, bool):
        raise ConfigError(f"policy 'downgrade_enabled' must be boolean, got {downgrade_enabled!r}")
    return PolicySpec(
        escalate_consecutive_errors=pos_int("escalate_consecutive_errors"),
        escalate_repeated_tool_calls=pos_int("escalate_repeated_tool_calls"),
        downgrade_enabled=downgrade_enabled,
        downgrade_max_context_tokens=pos_int("downgrade_max_context_tokens"),
        downgrade_max_turn_index=pos_int("downgrade_max_turn_index"),
        dwell_requests=pos_int("dwell_requests"),
        hysteresis_requests=pos_int("hysteresis_requests"),
        safety_margin_usd=float(amount("safety_margin_usd", 0.0) or 0.0),
        escalate_benefit_usd=amount("escalate_benefit_usd", None),
        downgrade_benefit_usd=amount("downgrade_benefit_usd", None),
        cost_check_enabled=flag("cost_check_enabled", True),
        block_in_tool_loop=flag("block_in_tool_loop", False),
        breaker_error_threshold=pos_int_or(
            "breaker_error_threshold", DEFAULT_BREAKER_ERROR_THRESHOLD
        ),
        breaker_cooldown_seconds=pos_int_or(
            "breaker_cooldown_seconds", DEFAULT_BREAKER_COOLDOWN_SECONDS
        ),
        max_switches_per_session=pos_int_or(
            "max_switches_per_session", DEFAULT_MAX_SWITCHES_PER_SESSION
        ),
        hold_max_requests=pos_int_or(
            "hold_max_requests", DEFAULT_HOLD_MAX_REQUESTS
        ),
    )


def _parse_classifier(data: dict[str, Any], where: str) -> ClassifierSpec | None:
    """The optional `classifier` section, or None when there is not one.

    Absent means disabled, never enabled: `ClassifierSpec.enabled` defaults to
    False and nothing here can turn it on without the file saying so. While the
    section is disabled `router/classifier.py` returns None for every prompt, no
    signal is set and no decision changes. Rule 2.

    Every key is validated rather than coerced. A rule weight nobody verified is
    worse than no rule at all, so a wrong type is an error and never a default.
    """
    if "classifier" not in data:
        return None
    section = data["classifier"]
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ConfigError(f"{where} key 'classifier' must be a mapping")

    for key in section:
        if key not in _CLASSIFIER_KEYS:
            raise ConfigError(f"unknown classifier key {key!r}")

    enabled = section.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError(f"classifier 'enabled' must be boolean, got {enabled!r}")

    cheap_below = _classifier_int(
        section, "cheap_below", DEFAULT_CLASSIFIER_CHEAP_BELOW, MIN_CLASSIFIER_SCORE, MAX_CLASSIFIER_SCORE
    )
    strong_from = _classifier_int(
        section, "strong_from", DEFAULT_CLASSIFIER_STRONG_FROM, MIN_CLASSIFIER_SCORE, MAX_CLASSIFIER_SCORE
    )
    if cheap_below > strong_from:
        raise ConfigError(
            f"classifier 'cheap_below' ({cheap_below}) must not exceed "
            f"'strong_from' ({strong_from}): the tier boundaries would overlap"
        )

    return ClassifierSpec(
        enabled=enabled,
        strong_keywords=_keyword_list(section, "strong_keywords"),
        cheap_keywords=_keyword_list(section, "cheap_keywords"),
        points=_parse_classifier_points(section.get("points"), where),
        cheap_below=cheap_below,
        strong_from=strong_from,
    )


def _keyword_list(section: dict[str, Any], key: str) -> tuple[str, ...]:
    """A keyword list, or empty when the key is absent.

    An empty keyword is rejected rather than ignored: it would match every
    prompt, and a rule that always fires is not a rule anybody tuned.
    """
    raw = section.get(key)
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ConfigError(f"classifier {key!r} must be a list of strings, got {raw!r}")
    words: list[str] = []
    for position, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(
                f"classifier {key}[{position}] must be a non-empty string, got {item!r}"
            )
        words.append(item)
    return tuple(words)


def _classifier_int(
    section: dict[str, Any], key: str, default: int, least: int, most: int
) -> int:
    """An integer inside a range, or the shipped default when the key is absent."""
    if key not in section:
        return default
    value = section[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"classifier {key!r} must be an integer, got {value!r}")
    if not least <= value <= most:
        raise ConfigError(
            f"classifier {key!r} must be in {least}..{most}, got {value}"
        )
    return value


def _parse_classifier_points(raw: Any, where: str) -> ClassifierPoints:
    """The rule weights, or the shipped defaults when the block is absent."""
    if raw is None:
        return ClassifierPoints()
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} key 'classifier' -> 'points' must be a mapping")
    for key in raw:
        if key not in _CLASSIFIER_POINT_KEYS:
            raise ConfigError(f"unknown classifier points key {key!r}")

    def weight(key: str, default: int) -> int:
        if key not in raw:
            return default
        value = raw[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(
                f"classifier points {key!r} must be an integer, got {value!r}"
            )
        return value

    return ClassifierPoints(
        base=weight("base", DEFAULT_CLASSIFIER_BASE),
        strong=weight("strong", DEFAULT_CLASSIFIER_STRONG),
        strong_cap=weight("strong_cap", DEFAULT_CLASSIFIER_STRONG_CAP),
        cheap=weight("cheap", DEFAULT_CLASSIFIER_CHEAP),
        cheap_cap=weight("cheap_cap", DEFAULT_CLASSIFIER_CHEAP_CAP),
        long_chars=weight("long_chars", DEFAULT_CLASSIFIER_LONG_CHARS),
        long_points=weight("long_points", DEFAULT_CLASSIFIER_LONG_POINTS),
        very_long_chars=weight("very_long_chars", DEFAULT_CLASSIFIER_VERY_LONG_CHARS),
        very_long_points=weight("very_long_points", DEFAULT_CLASSIFIER_VERY_LONG_POINTS),
        many_files=weight("many_files", DEFAULT_CLASSIFIER_MANY_FILES),
        many_files_points=weight("many_files_points", DEFAULT_CLASSIFIER_MANY_FILES_POINTS),
        question_points=weight("question_points", DEFAULT_CLASSIFIER_QUESTION_POINTS),
    )
