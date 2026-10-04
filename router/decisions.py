"""Metadata-only decision log for the Tamias router.

Standard library `sqlite3` only. The log records decisions: what was requested,
what was chosen, why, and what the router did. It never records request or
response content. There is deliberately no column for prompt text, source code,
tool arguments, assistant text or headers, and none may be added - see
`FORBIDDEN_COLUMN_TOKENS`, which the tests check against the live schema.

A session hint is a truncated, salted SHA-256 of the first user message. The
text is read in memory to compute the digest and is then discarded; only the
16-character digest is stored. The salt is random per install and lives beside
the database, so a hint cannot be correlated across installs and cannot be
recovered by hashing a guessed message against a leaked digest.

Path: `~/.tamias/router.sqlite3`, overridable with the `ROUTER_DB` env var.

Nothing here touches the filesystem until a read or a write happens, and no
constructor raises. A logging failure must never break or delay a proxied
request, so every failure is raised to the caller at the moment of use and is
the caller's to absorb.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

#: Environment variable that overrides the database location.
DB_ENV_VAR = "ROUTER_DB"

#: Default install-local location. Never inside the source tree.
DEFAULT_DB_PATH = Path.home() / ".tamias" / "router.sqlite3"

#: The random per-install salt, stored beside the database.
SALT_FILENAME = "router.salt"

#: Only this route is logged. Everything else is forwarded and not recorded.
MESSAGES_PATH = "/v1/messages"

#: Used when no session hint can be computed.
UNKNOWN_SESSION = "unknown"

#: Hex characters kept from the session digest.
SESSION_HINT_LENGTH = 16

#: Milliseconds sqlite waits for a lock before giving up.
BUSY_TIMEOUT_MS = 250

#: Words that must never appear in a column name. Rule 1. Matched per
#: underscore-separated word, so `reason_codes` is allowed and `prompt_text` is
#: not. Check the live schema with `forbidden_columns`.
FORBIDDEN_COLUMN_WORDS = frozenset(
    {
        "prompt",
        "prompts",
        "code",
        "source",
        "tool",
        "argument",
        "arguments",
        "assistant",
        "header",
        "headers",
        "body",
        "content",
        "text",
        "message",
        "messages",
        "transcript",
    }
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS router_decisions (
    decision_id              INTEGER PRIMARY KEY,
    timestamp                TEXT    NOT NULL,
    session_hint             TEXT    NOT NULL,
    request_index_in_session INTEGER NOT NULL,
    requested_model          TEXT,
    chosen_model             TEXT,
    chosen_effort            TEXT,
    mode                     TEXT    NOT NULL,
    reason_codes             TEXT    NOT NULL,
    signal_values            TEXT    NOT NULL,
    action                   TEXT    NOT NULL CHECK (action IN ('STAY', 'SWITCH')),
    applied                  INTEGER NOT NULL CHECK (applied IN (0, 1)),
    error                    TEXT
);

CREATE INDEX IF NOT EXISTS router_decisions_session
    ON router_decisions (session_hint, request_index_in_session);
"""

_COLUMNS = (
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
)

_INSERT = f"""
INSERT INTO router_decisions (
    timestamp, session_hint, request_index_in_session, requested_model,
    chosen_model, chosen_effort, mode, reason_codes, signal_values,
    action, applied, error
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def utc_timestamp() -> str:
    """UTC, ISO-8601, second resolution, explicit `Z`."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class RequestMetadata:
    """What may be extracted from a request body. Nothing else is kept."""

    model: str | None = None
    first_user_text: str | None = None
    error: str | None = None


def read_request_metadata(body: bytes) -> RequestMetadata:
    """Read a request body for metadata only. Never raises, never retains.

    The body is decoded, the `model` field and the first user message are read,
    and the decoded object is dropped when this returns. The caller still holds
    the original bytes and forwards them untouched.
    """
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return RequestMetadata(error="request_body_not_json")
    if not isinstance(payload, dict):
        return RequestMetadata(error="request_body_not_an_object")

    model = payload.get("model")
    if model is None:
        chosen_model = None
        model_error = None
    elif isinstance(model, str):
        chosen_model = model
        model_error = None
    else:
        chosen_model = None
        model_error = "model_field_not_text"

    return RequestMetadata(
        model=chosen_model,
        first_user_text=first_user_message_text(payload),
        error=model_error,
    )


def first_user_message_text(payload: dict[str, Any]) -> str | None:
    """The first user message's text, or None. Held in memory only."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _content_text(message.get("content"))
        if text:
            return text
    return None


def _content_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        joined = "".join(parts)
        return joined or None
    return None


def session_hint_for(first_user_text: str | None, salt: bytes) -> str:
    """A short, salted, truncated digest. The text is not recoverable from it."""
    if not first_user_text:
        return UNKNOWN_SESSION
    digest = hashlib.sha256(salt + first_user_text.encode("utf-8")).digest()
    return digest.hex()[:SESSION_HINT_LENGTH]


@dataclass(frozen=True)
class DecisionRecord:
    """One row to append. Metadata only."""

    session_hint: str
    requested_model: str | None
    chosen_model: str | None
    mode: str
    chosen_effort: str | None = None
    reason_codes: list[str] = field(default_factory=lambda: ["PASSTHROUGH"])
    signal_values: dict[str, Any] = field(default_factory=dict)
    action: str = "STAY"
    applied: int = 0
    error: str | None = None
    request_index_in_session: int | None = None
    timestamp: str | None = None


@dataclass(frozen=True)
class DecisionRow:
    """One row as read back."""

    decision_id: int
    timestamp: str
    session_hint: str
    request_index_in_session: int
    requested_model: str | None
    chosen_model: str | None
    chosen_effort: str | None
    mode: str
    reason_codes: list[str]
    signal_values: dict[str, Any]
    action: str
    applied: int
    error: str | None


class DecisionLog:
    """Append-only decision log. Metadata only.

    Construction performs no I/O, so an unwritable or missing location cannot
    stop the proxy from starting.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_db_path()
        self._salt: bytes | None = None
        self._schema_ready = False

    @classmethod
    def from_env(cls) -> DecisionLog:
        return cls(default_db_path())

    @property
    def salt_path(self) -> Path:
        return self.db_path.parent / SALT_FILENAME

    def session_hint(self, first_user_text: str | None) -> str:
        """The session hint, or "unknown" if it cannot be computed."""
        if not first_user_text:
            return UNKNOWN_SESSION
        try:
            salt = self._load_or_create_salt()
        except OSError:
            return UNKNOWN_SESSION
        return session_hint_for(first_user_text, salt)

    def record(self, decision: DecisionRecord) -> int:
        """Append one decision. Returns its `decision_id`.

        Raises whatever sqlite or the filesystem raises; the caller decides
        whether that is survivable.
        """
        with self._connect() as connection:
            if decision.request_index_in_session is not None:
                cursor = connection.execute(
                    _INSERT, self._row_values(decision, decision.request_index_in_session)
                )
                return int(cursor.lastrowid or 0)

            with _immediate_transaction(connection):
                index = self._next_index(connection, decision.session_hint)
                cursor = connection.execute(_INSERT, self._row_values(decision, index))
            return int(cursor.lastrowid or 0)

    def _row_values(self, decision: DecisionRecord, index: int) -> tuple[Any, ...]:
        return (
            decision.timestamp or utc_timestamp(),
            decision.session_hint,
            index,
            decision.requested_model,
            decision.chosen_model,
            decision.chosen_effort,
            decision.mode,
            json.dumps(list(decision.reason_codes)),
            json.dumps(dict(decision.signal_values)),
            decision.action,
            int(decision.applied),
            decision.error,
        )

    def recent(self, limit: int = 10) -> list[DecisionRow]:
        """The newest rows first."""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM router_decisions "
                "ORDER BY decision_id DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [self._to_row(row) for row in rows]

    def count(self, session_hint: str | None = None) -> int:
        """Total rows, or rows for one session."""
        with self._connect() as connection:
            if session_hint is None:
                row = connection.execute("SELECT COUNT(*) FROM router_decisions").fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) FROM router_decisions WHERE session_hint = ?",
                    (session_hint,),
                ).fetchone()
        return int(row[0])

    def column_names(self) -> list[str]:
        """The live column names, for verifying Rule 1 against reality."""
        with self._connect() as connection:
            rows = connection.execute("PRAGMA table_info(router_decisions)").fetchall()
        return [str(row[1]) for row in rows]

    def _to_row(self, row: tuple[Any, ...]) -> DecisionRow:
        values = dict(zip(_COLUMNS, row, strict=True))
        return DecisionRow(
            decision_id=int(values["decision_id"]),
            timestamp=str(values["timestamp"]),
            session_hint=str(values["session_hint"]),
            request_index_in_session=int(values["request_index_in_session"]),
            requested_model=values["requested_model"],
            chosen_model=values["chosen_model"],
            chosen_effort=values["chosen_effort"],
            mode=str(values["mode"]),
            reason_codes=_load_json_list(values["reason_codes"]),
            signal_values=_load_json_object(values["signal_values"]),
            action=str(values["action"]),
            applied=int(values["applied"]),
            error=values["error"],
        )

    def _next_index(self, connection: sqlite3.Connection, hint: str) -> int:
        row = connection.execute(
            "SELECT COUNT(*) FROM router_decisions WHERE session_hint = ?", (hint,)
        ).fetchone()
        return int(row[0]) + 1

    def _load_or_create_salt(self) -> bytes:
        if self._salt is not None:
            return self._salt
        try:
            existing = self.salt_path.read_bytes()
        except OSError:
            existing = b""
        if existing:
            self._salt = existing
            return existing

        generated = secrets.token_bytes(32)
        self.salt_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = os.open(self.salt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            self._salt = self.salt_path.read_bytes()
            return self._salt
        try:
            os.write(handle, generated)
        finally:
            os.close(handle)
        self._salt = generated
        return generated

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None)
        try:
            connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
            if not self._schema_ready:
                connection.executescript(SCHEMA)
                self._schema_ready = True
            yield connection
        finally:
            connection.close()


@contextlib.contextmanager
def _immediate_transaction(connection: sqlite3.Connection) -> Iterator[None]:
    """Serialize a read-then-write so concurrent handlers cannot both count the
    same rows and assign the same `request_index_in_session`."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        with contextlib.suppress(sqlite3.Error):
            connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")


def forbidden_columns(column_names: Iterable[str]) -> list[str]:
    """Any column name that would hold request or response content (Rule 1)."""
    return [
        name
        for name in column_names
        if FORBIDDEN_COLUMN_WORDS.intersection(name.lower().split("_"))
    ]


def default_db_path() -> Path:
    """`ROUTER_DB` when set, otherwise `~/.tamias/router.sqlite3`."""
    override = os.environ.get(DB_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return DEFAULT_DB_PATH


def _load_json_list(raw: Any) -> list[str]:
    try:
        parsed = json.loads(str(raw))
    except ValueError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _load_json_object(raw: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(str(raw))
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
