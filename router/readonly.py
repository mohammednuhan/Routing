"""Read-only SQLite access, for the commands that must never write.

`report` and `ledger-info` only look. Neither may create a database, change a
row, or leave a journal behind, so every connection is opened through
`sqlite3.connect` with a `file:` URI carrying `mode=ro`. SQLite refuses to
create, and refuses to write to, a database opened that way, and refuses to
open a missing file at all: read-only is enforced by SQLite, not by discipline
in the callers.

The read-only path also has no schema to apply. The writer's `CREATE TABLE IF
NOT EXISTS` is skipped entirely, so pointing a report at a fresh path can never
bring a database into existence.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

#: SQLite's read-only URI parameter. The only mode this module opens anything.
READ_ONLY_MODE = "mode=ro"

#: Milliseconds sqlite waits for a lock before giving up. Long enough to wait
#: out a proxy writing at the same moment, short enough that a report cannot
#: hang a terminal.
BUSY_TIMEOUT_MS = 250


class ReadOnlyError(Exception):
    """A database could not be opened for reading, or could not be read."""


def read_only_uri(path: Path | str) -> str:
    """The absolute `file:` URI that opens `path` read-only.

    `Path.as_uri` percent-encodes the path, which matters on Windows: a temp
    directory under a user profile routinely contains a space.
    """
    return f"{Path(path).expanduser().resolve().as_uri()}?{READ_ONLY_MODE}"


@contextmanager
def open_read_only(path: Path | str) -> Iterator[sqlite3.Connection]:
    """Yield a read-only connection to `path`, or raise `ReadOnlyError`.

    Nothing is created: not the file, not its parent directory, not a journal,
    not a salt file beside it.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise ReadOnlyError(f"no such database file: {resolved}")

    uri = read_only_uri(resolved)
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=BUSY_TIMEOUT_MS / 1000)
    except sqlite3.Error as exc:
        raise ReadOnlyError(f"cannot open {resolved} read-only: {exc}") from exc

    try:
        connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        yield connection
    except sqlite3.Error as exc:
        raise ReadOnlyError(f"cannot read {resolved}: {exc}") from exc
    finally:
        connection.close()