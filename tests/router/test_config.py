from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from router import config
from router.config import ConfigError

_DELETE = object()

BASE: dict[str, Any] = {
    "listen": "127.0.0.1:8787",
    "upstream": "https://api.anthropic.com",
    "mode": "shadow",
    "models": [
        {
            "id": "TODO_MODEL_TIER_LOW",
            "legal_efforts": ["TODO_EFFORT_LOW", "TODO_EFFORT_HIGH"],
            "cost_tier": "low",
        },
        {
            "id": "TODO_MODEL_TIER_HIGH",
            "legal_efforts": ["TODO_EFFORT_HIGH"],
            "cost_tier": "high",
        },
    ],
    "default_model": "TODO_MODEL_TIER_LOW",
    "default_effort": "TODO_EFFORT_LOW",
}


def write_config(tmp_path: Path, **overrides: Any) -> Path:
    """Write a config built from BASE with `overrides` applied."""
    data = deepcopy(BASE)
    for key, value in overrides.items():
        if value is _DELETE:
            data.pop(key, None)
        else:
            data[key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


# --- valid configs -------------------------------------------------------


def test_packaged_config_loads():
    loaded = config.load_config()

    assert loaded.source == config.DEFAULT_CONFIG_PATH
    assert loaded.mode == "shadow"
    assert loaded.listen == "127.0.0.1:8787"
    assert loaded.listen_host == "127.0.0.1"
    assert loaded.upstream == "https://api.anthropic.com"
    assert len(loaded.models) == 3
    assert loaded.default_model in loaded.model_ids


def test_packaged_config_defaults_are_legal():
    loaded = config.load_config()

    assert loaded.is_legal(loaded.default_model, loaded.default_effort)


def test_packaged_config_starts_in_shadow_mode():
    assert config.load_config().mode == config.DEFAULT_MODE == "shadow"


def test_packaged_config_uses_placeholder_ids_only():
    for model_id in config.load_config().model_ids:
        assert model_id.startswith("TODO_"), model_id


def test_valid_config_loads(tmp_path):
    loaded = config.load_config(write_config(tmp_path))

    assert loaded.mode == "shadow"
    assert loaded.listen == "127.0.0.1:8787"
    assert loaded.upstream == "https://api.anthropic.com"
    assert loaded.model_ids == ("TODO_MODEL_TIER_LOW", "TODO_MODEL_TIER_HIGH")
    assert loaded.default_model == "TODO_MODEL_TIER_LOW"
    assert loaded.default_effort == "TODO_EFFORT_LOW"


def test_every_mode_is_accepted(tmp_path):
    for mode in config.LEGAL_MODES:
        loaded = config.load_config(write_config(tmp_path, mode=mode))
        assert loaded.mode == mode


def test_model_cost_tier_is_recorded(tmp_path):
    loaded = config.load_config(write_config(tmp_path))

    assert [spec.cost_tier for spec in loaded.models] == ["low", "high"]


def test_model_not_in_config_is_not_legal(tmp_path):
    loaded = config.load_config(write_config(tmp_path))

    assert loaded.is_legal("TODO_MODEL_TIER_LOW", "TODO_EFFORT_LOW")
    assert not loaded.is_legal("TODO_MODEL_TIER_MISSING", "TODO_EFFORT_LOW")
    assert not loaded.is_legal("TODO_MODEL_TIER_HIGH", "TODO_EFFORT_LOW")


# --- unknown mode --------------------------------------------------------


@pytest.mark.parametrize("mode", ["agressive", "SHADOW", "disabled", "", 3, None])
def test_unknown_mode_rejected(tmp_path, mode):
    path = write_config(tmp_path, mode=mode)

    with pytest.raises(ConfigError, match=re.escape(f"unknown mode {mode!r}")):
        config.load_config(path)


def test_unknown_mode_message_lists_legal_modes(tmp_path):
    path = write_config(tmp_path, mode="agressive")

    with pytest.raises(ConfigError, match=re.escape("legal modes: shadow, active, off")):
        config.load_config(path)


def test_missing_mode_rejected(tmp_path):
    path = write_config(tmp_path, mode=_DELETE)

    with pytest.raises(ConfigError, match=re.escape("missing required key 'mode'")):
        config.load_config(path)


# --- empty model list ----------------------------------------------------


@pytest.mark.parametrize("models", [[], None, {}, "TODO_MODEL_TIER_LOW"])
def test_empty_model_list_rejected(tmp_path, models):
    path = write_config(tmp_path, models=models)

    with pytest.raises(ConfigError, match=re.escape("'models' must be a non-empty list")):
        config.load_config(path)


def test_missing_models_rejected(tmp_path):
    path = write_config(tmp_path, models=_DELETE)

    with pytest.raises(ConfigError, match=re.escape("missing required key 'models'")):
        config.load_config(path)


def test_model_without_legal_efforts_rejected(tmp_path):
    path = write_config(
        tmp_path,
        models=[{"id": "TODO_MODEL_TIER_LOW", "legal_efforts": [], "cost_tier": "low"}],
    )

    with pytest.raises(ConfigError, match=re.escape("'legal_efforts' must be a non-empty list")):
        config.load_config(path)


def test_duplicate_model_id_rejected(tmp_path):
    duplicate = deepcopy(BASE["models"][0])
    path = write_config(tmp_path, models=[BASE["models"][0], duplicate])

    with pytest.raises(
        ConfigError, match=re.escape(f"duplicate model id {duplicate['id']!r}")
    ):
        config.load_config(path)


# --- default_model not in the list ---------------------------------------


def test_default_model_not_in_list_rejected(tmp_path):
    path = write_config(tmp_path, default_model="TODO_MODEL_TIER_MID")

    with pytest.raises(ConfigError, match=re.escape("default_model 'TODO_MODEL_TIER_MID'")) as err:
        config.load_config(path)

    assert "is not in models" in str(err.value)
    assert "TODO_MODEL_TIER_LOW" in str(err.value)


def test_default_model_may_not_be_invented(tmp_path):
    path = write_config(tmp_path, default_model="claude-some-model-2099")

    with pytest.raises(ConfigError, match=re.escape("is not in models")):
        config.load_config(path)


def test_missing_default_model_rejected(tmp_path):
    path = write_config(tmp_path, default_model=_DELETE)

    with pytest.raises(ConfigError, match=re.escape("missing required key 'default_model'")):
        config.load_config(path)


# --- default_effort not legal for that model -----------------------------


def test_default_effort_not_legal_rejected(tmp_path):
    path = write_config(tmp_path, default_effort="TODO_EFFORT_EXTREME")

    with pytest.raises(ConfigError, match=re.escape("default_effort 'TODO_EFFORT_EXTREME'")) as err:
        config.load_config(path)

    message = str(err.value)
    assert "is not legal for model 'TODO_MODEL_TIER_LOW'" in message
    assert "TODO_EFFORT_LOW" in message


def test_default_effort_legal_for_another_model_only_rejected(tmp_path):
    narrow = {
        "id": "TODO_MODEL_TIER_LOW",
        "legal_efforts": ["TODO_EFFORT_LOW"],
        "cost_tier": "low",
    }
    path = write_config(
        tmp_path,
        models=[narrow, BASE["models"][1]],
        default_effort="TODO_EFFORT_HIGH",
    )

    with pytest.raises(ConfigError, match=re.escape("is not legal for model")) as err:
        config.load_config(path)

    assert "(legal efforts: TODO_EFFORT_LOW)" in str(err.value)


def test_missing_default_effort_rejected(tmp_path):
    path = write_config(tmp_path, default_effort=_DELETE)

    with pytest.raises(ConfigError, match=re.escape("missing required key 'default_effort'")):
        config.load_config(path)


# --- listen address ------------------------------------------------------


@pytest.mark.parametrize(
    ("listen", "host"),
    [
        ("0.0.0.0:8787", "0.0.0.0"),
        ("192.168.1.10:8787", "192.168.1.10"),
        ("localhost:8787", "localhost"),
        ("[::1]:8787", "[::1]"),
        ("::1:8787", "::1"),
        ("127.0.0.2:8787", "127.0.0.2"),
        ("example.internal:8787", "example.internal"),
    ],
)
def test_non_loopback_listen_rejected(tmp_path, listen, host):
    path = write_config(tmp_path, listen=listen)

    with pytest.raises(
        ConfigError, match=re.escape(f"listen host must be 127.0.0.1, got {host!r}")
    ):
        config.load_config(path)


def test_listen_without_port_rejected(tmp_path):
    path = write_config(tmp_path, listen="127.0.0.1")

    with pytest.raises(ConfigError, match=re.escape("listen must be 'host:port'")):
        config.load_config(path)


@pytest.mark.parametrize("listen", ["127.0.0.1:0", "127.0.0.1:70000"])
def test_listen_port_out_of_range_rejected(tmp_path, listen):
    path = write_config(tmp_path, listen=listen)

    with pytest.raises(ConfigError, match="listen port must be in 1..65535"):
        config.load_config(path)


def test_listen_port_must_be_an_integer(tmp_path):
    path = write_config(tmp_path, listen="127.0.0.1:http")

    with pytest.raises(ConfigError, match="listen port must be an integer"):
        config.load_config(path)


def test_missing_listen_rejected(tmp_path):
    path = write_config(tmp_path, listen=_DELETE)

    with pytest.raises(ConfigError, match=re.escape("missing required key 'listen'")):
        config.load_config(path)


# --- malformed input -----------------------------------------------------


def test_missing_file_rejected(tmp_path):
    path = tmp_path / "absent.yaml"

    with pytest.raises(ConfigError, match=re.escape(f"config file not found: {path}")):
        config.load_config(path)


def test_malformed_yaml_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("mode: shadow\n  listen: [unclosed\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="cannot parse YAML"):
        config.load_config(path)


def test_empty_file_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ConfigError, match="must be a mapping"):
        config.load_config(path)


def test_top_level_must_be_a_mapping(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- mode: shadow\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="must be a mapping, got list"):
        config.load_config(path)


@pytest.mark.parametrize("field", ["listen", "upstream", "default_model", "default_effort"])
def test_empty_string_field_rejected(tmp_path, field):
    path = write_config(tmp_path, **{field: ""})

    with pytest.raises(ConfigError, match=re.escape(f"key {field!r} must be a non-empty string")):
        config.load_config(path)


# --- error reporting -----------------------------------------------------


def test_load_config_or_exit_returns_config(tmp_path):
    loaded = config.load_config_or_exit(write_config(tmp_path))

    assert loaded.mode == "shadow"


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"mode": "agressive"}, "unknown mode"),
        ({"models": []}, "'models' must be a non-empty list"),
        ({"default_model": "TODO_MODEL_TIER_MID"}, "is not in models"),
        ({"default_effort": "TODO_EFFORT_EXTREME"}, "is not legal for model"),
        ({"listen": "0.0.0.0:8787"}, "listen host must be 127.0.0.1"),
    ],
)
def test_load_config_or_exit_exits_non_zero(tmp_path, capsys, overrides, expected):
    path = write_config(tmp_path, **overrides)

    with pytest.raises(SystemExit) as err:
        config.load_config_or_exit(path)

    assert err.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("config error: ")
    assert expected in captured.err
