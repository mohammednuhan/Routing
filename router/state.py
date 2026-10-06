"""In-memory per-session state for the safety layer.

Safety needs to remember, per session, how many requests it has seen, when it
last allowed a switch, which way that switch went, and which model that switch
chose. That is a small amount of metadata and it is held here, in memory, for
the life of the process.

`held_model` is the model an allowed switch picked, kept so that the requests
which follow it - which arrive naming the ORIGINAL model, because the client
never learns it was moved - can stay on it. `held_from_model` is the model the
client is still asking for, `held_requests` counts the requests the hold has
served, and `held_prompt_count` is how many human prompts the held request's
conversation carried, so a later request can tell the continuation of that one
prompt from a new one. All four are a model id, a model id, a counter and a
count. Rule 1 applies to them exactly as it applies to the rest: there is no
field here for a message, a prompt, tool arguments or a header, and none may
be added.

It is deliberately NOT persisted. A restart forgets every session, which can
only make the router more willing to switch, never less; the safety layer fails
open to STAY on its own checks but never to "assume a switch happened".

Thread safety: a `ThreadingHTTPServer` runs one handler thread per connection,
so every mutation happens under one lock and every value handed out is a copy.
A caller mutating what it was given can never corrupt another thread's view.

Bounded: at most `MAX_SESSIONS` sessions are retained. The least recently
observed session is evicted when the bound is reached, so the memory a
long-running router uses is a function of the bound and not of how many
distinct sessions it has ever seen.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, replace

#: Most sessions retained at once. Older entries are evicted past this.
MAX_SESSIONS: int = 1000

#: The only two directions a switch can go.
UP = "up"
DOWN = "down"


@dataclass(frozen=True)
class SessionState:
    """What safety remembers about one session. Counters and labels only."""

    requests_seen: int = 0
    last_switch_request_index: int | None = None
    last_switch_direction: str | None = None
    opposite_streak: int = 0
    switch_count: int = 0
    held_model: str | None = None
    held_from_model: str | None = None
    held_requests: int = 0
    held_prompt_count: int | None = None

    @property
    def requests_since_last_switch(self) -> int | None:
        """Requests since the last allowed switch, or None if there was none."""
        if self.last_switch_request_index is None:
            return None
        return self.requests_seen - self.last_switch_request_index

    def recorded_switch(self, direction: str) -> SessionState:
        """A copy that records an allowed switch in `direction`.

        `switch_count` only ever counts switches that were allowed and applied,
        so the per-session cap counts changes the router actually made.
        """
        return replace(
            self,
            last_switch_request_index=self.requests_seen,
            last_switch_direction=direction,
            opposite_streak=0,
            switch_count=self.switch_count + 1,
        )

    def recorded_hold(
        self, target_model: str, from_model: str, prompt_count: int | None = None
    ) -> SessionState:
        """A copy that holds `target_model` on behalf of a switch from `from_model`.

        Recorded next to an allowed switch, and nowhere else: a hold is the
        memory of a switch that was approved, not a decision of its own. It
        never touches `switch_count`, `last_switch_request_index` or
        `last_switch_direction`, because serving the hold is not a new switch
        and must not look like one to dwell or to the per-session cap.

        `prompt_count` is the request's `human_prompt_count`, stored so a later
        request can tell a tool-loop continuation of the same human prompt from
        a new one. It is a count, or None when that request's count could not
        be determined. A later request whose own count is known to be higher
        releases the hold rather than being served from it.
        """
        return replace(
            self,
            held_model=target_model,
            held_from_model=from_model,
            held_requests=0,
            held_prompt_count=prompt_count,
        )

    def released_hold(self) -> SessionState:
        """A copy with no hold. The counters go with it."""
        return replace(
            self,
            held_model=None,
            held_from_model=None,
            held_requests=0,
            held_prompt_count=None,
        )

    @property
    def has_hold(self) -> bool:
        return self.held_model is not None


class SessionStore:
    """A bounded, thread-safe map from session hint to `SessionState`."""

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        if max_sessions < 1:
            raise ValueError(f"max_sessions must be at least 1, got {max_sessions}")
        self._max_sessions = max_sessions
        self._lock = threading.Lock()
        self._states: OrderedDict[str, SessionState] = OrderedDict()

    @property
    def max_sessions(self) -> int:
        return self._max_sessions

    def __len__(self) -> int:
        with self._lock:
            return len(self._states)

    def __contains__(self, session_hint: object) -> bool:
        with self._lock:
            return session_hint in self._states

    def session_hints(self) -> list[str]:
        """Every retained hint, least recently observed first."""
        with self._lock:
            return list(self._states)

    def observe(self, session_hint: str) -> SessionState:
        """Count this request and return the state to reason about.

        The returned value is the stored state; treat it as read-only and hand
        back any change through `replace`.
        """
        with self._lock:
            state = self._states.get(session_hint)
            if state is None:
                state = SessionState()
            counted = replace(state, requests_seen=state.requests_seen + 1)
            self._states[session_hint] = counted
            self._states.move_to_end(session_hint)
            self._evict_locked()
            return counted

    def replace(self, session_hint: str, state: SessionState) -> SessionState:
        """Store `state` as the current state for `session_hint`."""
        with self._lock:
            self._states[session_hint] = state
            self._states.move_to_end(session_hint)
            self._evict_locked()
            return state

    def get(self, session_hint: str) -> SessionState:
        """The stored state without counting a request. Unknown means empty."""
        with self._lock:
            return self._states.get(session_hint, SessionState())

    def clear(self) -> None:
        with self._lock:
            self._states.clear()

    def _evict_locked(self) -> None:
        while len(self._states) > self._max_sessions:
            self._states.popitem(last=False)