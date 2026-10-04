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


class ConfigError(Exception):
    """The config file is missing, unreadable, malformed, or illegal."""


@dataclass(frozen=True)
class ModelSpec:
    """One legal model: its id, the efforts legal for it, and its cost tier."""

    id: str
    legal_efforts: tuple[str, ...]
    cost_tier: str


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

    return ModelSpec(id=model_id, legal_efforts=tuple(efforts), cost_tier=cost_tier)