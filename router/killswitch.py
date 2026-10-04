"""The kill switch: a check on every request that stops the router changing anything.

Two independent switches turn it on, and either one alone is enough:

* the environment variable ``TAMIAS_ROUTER_OFF`` set to an on-value;
* a flag file named ``router.off`` in the folder that holds the decision
  database, so a test that points ``ROUTER_DB`` at a temporary directory gets a
  kill switch of its own and cannot disturb a real install.

Both are read on every request rather than cached, so dropping the flag file
next to a running router takes effect immediately and no restart is needed.
While the switch is on the router behaves exactly as it does in ``off`` mode:
the request is forwarded byte for byte, nothing is rewritten, `applied` is 0,
and the row that is logged carries `KILL_SWITCH`.

The flag file is what `tamias-router off` creates and `tamias-router on`
removes. The environment variable is for supervisors and one-off shells; it
outranks nothing, because either one on its own is enough to stop the router.

Nothing here stores request content. Rule 1.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .decisions import default_db_path

#: Environment variable that turns the kill switch on.
KILL_SWITCH_ENV = "TAMIAS_ROUTER_OFF"

#: Values of that variable that count as "on". Anything else, unset included,
#: means off. `TAMIAS_ROUTER_OFF=1` is the documented form; the rest are
#: accepted because a kill switch that silently ignores a human's `true` is
#: worse than one that accepts more spellings than it needs.
ON_VALUES = frozenset({"1", "true", "yes", "on"})

#: Flag file name. It sits beside the decision database so that a temporary
#: `ROUTER_DB` isolates the switch as well as the log.
KILL_SWITCH_FILENAME = "router.off"

#: Reason code recorded on every row written while the switch is on.
REASON_KILL_SWITCH = "KILL_SWITCH"

#: What `tamias-router off` writes into the flag file. The presence of the file
#: is the signal; the contents exist only so the file is not mistaken for an
#: empty accident.
FLAG_CONTENTS = "off\n"


@dataclass(frozen=True)
class KillSwitch:
    """The switch's state for one request, and why it is in that state."""

    on: bool
    reasons: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return "ON" if self.on else "OFF"

    @property
    def detail(self) -> str:
        """Why the switch is where it is, for `status`."""
        if self.reasons:
            return ", ".join(self.reasons)
        return "no env var and no flag file"


def kill_switch_path(db_path: Path | str | None) -> Path:
    """Where the flag file lives: in the folder holding the decision database.

    With no path given, the decision database is the one `ROUTER_DB` names, or
    the default one.
    """
    target = default_db_path() if db_path is None else db_path
    return Path(target).parent / KILL_SWITCH_FILENAME


def read_kill_switch(
    db_path: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> KillSwitch:
    """Read the switch now. Called once per request; nothing is cached."""
    environ = os.environ if env is None else env
    reasons: list[str] = []

    raw = (environ.get(KILL_SWITCH_ENV) or "").strip().lower()
    if raw in ON_VALUES:
        reasons.append(f"{KILL_SWITCH_ENV}={raw}")

    path = kill_switch_path(db_path)
    try:
        present = path.is_file()
    except OSError:
        # An unreadable path is not evidence that the switch is on, and it is
        # not evidence that it is off either: the env var above still counts,
        # and `status` reports what it could actually see.
        present = False
    if present:
        reasons.append(f"flag file {path}")

    return KillSwitch(on=bool(reasons), reasons=tuple(reasons))


def write_kill_switch(db_path: Path | str | None, on: bool) -> Path:
    """Create or remove the flag file. Raises OSError if the filesystem objects.

    Creating the parent folder is allowed, because that is what the decision
    database itself does on first use.
    """
    path = kill_switch_path(db_path)
    if on:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(FLAG_CONTENTS, encoding="utf-8")
        return path
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    return path