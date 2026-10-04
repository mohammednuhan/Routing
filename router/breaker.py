"""The circuit breaker: after too many internal router errors, stop deciding.

The router is supposed to fail quietly — a bad config, an unknown model, a
decode error or a bug all leave the request untouched. This module is what stops
that quiet failure from repeating on every request. It counts consecutive
requests that failed *inside the router*, and once the count reaches the
configured threshold it stops the router from deciding anything for a cooldown,
then lets it try again.

Only internal errors count. An upstream that is down, a 500 from the provider
and a malformed request are all things the router handled correctly, so none of
them open the circuit. The error classes that do count are listed in
`INTERNAL_ERROR_CLASSES`, with the matching reason codes in
`INTERNAL_ERROR_REASONS`.

While the circuit is open a request is handled exactly as `off` mode handles
one: forwarded byte for byte, nothing rewritten, `applied` 0, and the row
recorded with `CIRCUIT_OPEN`. Open requests are not evaluated by the router at
all, so they neither raise nor clear the count; the count is what the next
evaluation after the cooldown starts from, and any request it evaluates without
an internal error resets it to zero.

The clock is injectable so the cooldown can be tested without sleeping.

This module holds no request content. Rule 1.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Iterable

#: Error classes in the decision log that mean the router itself failed.
INTERNAL_ERROR_CLASSES = frozenset(
    {
        "signals_failed",
        "policy_failed",
        "safety_failed",
        "rewrite_failed",
    }
)

#: Reason codes that mean the same thing, for the layers that record a reason
#: rather than an error class.
INTERNAL_ERROR_REASONS = frozenset({"SAFETY_ERROR", "POLICY_ERROR"})

#: Reason code recorded on every row written while the circuit is open.
REASON_CIRCUIT_OPEN = "CIRCUIT_OPEN"

#: Used when the config carries no policy block to read a threshold from.
DEFAULT_THRESHOLD = 3

#: Seconds the circuit stays open before the router is allowed to try again.
DEFAULT_COOLDOWN_SECONDS = 60


def is_internal_error(
    error: str | None, reason_codes: Iterable[str] | None = None
) -> bool:
    """True when a recorded row shows the router failed on this request."""
    if error in INTERNAL_ERROR_CLASSES:
        return True
    return any(code in INTERNAL_ERROR_REASONS for code in (reason_codes or ()))


class CircuitBreaker:
    """Consecutive-error counter with an open/closed state and a cooldown.

    One instance is shared by every request thread, so all mutation happens
    under one lock.
    """

    def __init__(
        self,
        threshold: int = DEFAULT_THRESHOLD,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if threshold < 1:
            raise ValueError(f"threshold must be at least 1, got {threshold}")
        if cooldown_seconds < 1:
            raise ValueError(f"cooldown_seconds must be at least 1, got {cooldown_seconds}")
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._consecutive = 0
        self._opened_at: float | None = None

    @property
    def consecutive(self) -> int:
        """Consecutive internal errors counted so far."""
        with self._lock:
            return self._consecutive

    @property
    def opened_at(self) -> float | None:
        """When the circuit last opened, or None while it is closed."""
        with self._lock:
            return self._opened_at

    def is_open(self) -> bool:
        """Whether requests must be passed through without a decision.

        Reading the state closes a circuit whose cooldown has expired, so the
        next request is decided normally again.
        """
        with self._lock:
            return self._open_locked()

    def record(self, internal_error: bool) -> bool:
        """Note one request the router evaluated. Returns whether it is now open.

        A request with no internal error resets the count to zero, which is
        what stops two failures an hour apart from ever opening the circuit.
        """
        with self._lock:
            already_open = self._open_locked()
            if internal_error:
                self._consecutive += 1
                if not already_open and self._consecutive >= self.threshold:
                    self._opened_at = self._clock()
            else:
                self._consecutive = 0
            return self._open_locked()

    def reset(self) -> None:
        """Close the circuit and forget the count."""
        with self._lock:
            self._consecutive = 0
            self._opened_at = None

    def _open_locked(self) -> bool:
        if self._opened_at is None:
            return False
        if self._clock() - self._opened_at >= self.cooldown_seconds:
            self._opened_at = None
            self._consecutive = 0
            return False
        return True