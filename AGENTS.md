# AGENTS.md

Operating rules for this repository. Read this before writing code.

The same rules appear in `router/AGENTS.md`, scoped to the `router` package.
Keep the two copies identical when either changes.

---

## Router rules (`tamias-router`)

These govern the `router/` package and the `tamias-router` CLI.

### Rule 1 — Metadata only

The router never stores, and must never be extended to store:

* prompt text;
* source-code text;
* tool arguments;
* assistant text;
* any other request or response content.

Metadata only: model id, effort value, cost tier, token counts, timings,
flags, hashes, and error class. A field may be recorded because it exists; its
content may not be.

### Rule 2 — The default action is STAY

The default is to pass the request through unchanged. A router that has not
decided, has not measured, or is not confident must return the request
untouched.

Changing `(model, effort)` is an action that must be justified, never a
fallback and never a default.

### Rule 3 — Every router error passes the request through unchanged

Any failure inside the router — bad config, unknown model, upstream timeout,
decode error, bug — results in the request being passed through unchanged.

The router must never drop a request, never return a partial response, and
never answer on the client's behalf because it failed. Errors are counted and
reported as metadata; they never alter the response.

### Rule 4 — Listen on 127.0.0.1 only

The router binds `127.0.0.1` and nothing else. A listen address whose host is
not `127.0.0.1` is rejected when the config is loaded.

No binding to `0.0.0.0`, `::`, a LAN address, a hostname, or a public
interface, under any mode.

### Rule 5 — Never log API keys or Authorization headers

API keys, `Authorization` headers, cookies, and bearer tokens are never
written to a log, a report, an error message, or a crash trace.

If a header must be inspected to forward it, inspect it in memory only. Log
the header *name* if needed, never its value.

### Rule 6 — Start in shadow mode

The shipped default is `mode: shadow`. Shadow mode observes and reports
without changing any request.

Moving to `active` is a deliberate act, never a default and never something a
fresh install, a restart, or a fallback can select on its own.

### Rule 7 — Only models and efforts from the config are legal

A model is legal only if its `id` appears in `router/config.yaml`, and an
effort is legal only if it is listed in that model's `legal_efforts`.

The router never invents, interpolates, or guesses a model id or an effort
value, and never falls back to one that is not in the config. Model ids in the
shipped config are placeholders; replacing them requires a verified id with a
source and an access date.

### Rule 8 — Definition of done

The full test suite is green. Never report done with failing tests.

A change that leaves any test failing is not finished, not "done pending
review", and not "done except one known failure". Fix the cause or report the
blocker plainly. A test that cannot be made to pass is raised as a question
with the exact assertion quoted, never edited away.

### Rule 9 — `config.yaml` edits are additive

`config.yaml` edits are additive; never regenerate it programmatically, and
comments must survive.

The comments in `router/config.yaml` carry the placeholder warnings that Rule 7
depends on. Round-tripping the file through a YAML dumper such as
`yaml.safe_dump` silently discards every one of them, so the file must be edited
by hand: add or adjust the specific lines that need to change and leave every
other line, comment included, exactly as it was.

---

## Scope

This repository currently contains the router configuration and CLI only. No
proxy is implemented: `tamias-router start` prints `proxy not implemented yet`
and exits. Do not add an HTTP server, an upstream client, or request forwarding
as a side effect of another change.