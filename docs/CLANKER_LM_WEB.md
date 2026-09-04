# Clanker-LM private web workbench

The Clanker-LM browser workbench is live at
**[https://bazzite.tail85f65f.ts.net:8444/](https://bazzite.tail85f65f.ts.net:8444/)**.
It is a private Tailnet service, not a public website. Access requires membership
in the Tailnet and the exact allowlisted Tailscale login
`jerrymares@gmail.com`.

## Deployment map

```text
allowlisted Tailnet browser
  -> Tailscale Serve HTTPS :8444 (tailnet only; no Funnel)
  -> http://127.0.0.1:8765
  -> clanker-lm-web.service
  -> one Starlette/Uvicorn process
  -> one isolated Clanker-LM runtime per browser session
```

| Boundary | Live value |
| --- | --- |
| Browser URL | `https://bazzite.tail85f65f.ts.net:8444/` |
| Exposure | Tailnet only through Tailscale Serve; this UI has no Funnel/public route |
| Identity | Exact `Tailscale-User-Login` allowlist: `jerrymares@gmail.com` |
| Local listener | `127.0.0.1:8765` |
| Process | `clanker-lm-web.service` systemd user service |
| Application | Starlette on one Uvicorn worker, with access logs disabled |

The process refuses a non-loopback bind. Deployed mode also requires an exact
browser origin and at least one allowlisted Tailscale identity. The health route
returns only `{"status":"ok"}`; every other request requires the allowlisted
identity, and mutations must match the configured origin.

## Using the workbench

Enter a fact, question, or resolver request in the composer. Each answer carries
a six-field evidence rail: **Answer**, **Truth**, **Source**, **Certainty**,
**Memory**, and **VADUG**. Memory is isolated by browser session. Use **Export
state** to save an inspectable Clanker-LM snapshot or **Reset session** to close
that runtime and start with empty memory.

The browser receives only an opaque session identifier. The cookie is Secure,
HttpOnly, SameSite=Strict, and scoped to `/`. Session state lives in the server
process; restarting the service clears unexported sessions.

## Security and resource boundaries

- The app listens only on loopback; Tailscale Serve is the sole remote path.
- Port 8444 is marked `tailnet only`. Do not configure this workbench with
  `tailscale funnel`.
- Deployed requests require the exact allowlisted Tailscale login. From a
  same-site or cross-site source, only a user-activated top-level document
  navigation may bootstrap a session. Iframes, subresources, fetch/CORS, and
  non-user cross-site attempts fail closed without allocating a runtime.
  Mutations also require the exact configured origin.
- Content Security Policy and defensive browser headers apply to success and
  error responses. UI text is inserted through safe DOM text APIs.
- Message bodies are not written to application logs, and Uvicorn access logs
  are disabled.
- The live service allows at most 64 active sessions. Defaults also enforce a
  30-minute idle timeout, 30 requests per 60 seconds per session, 200 turns per
  session, 4 KiB messages, 16 KiB request bodies, 64 KiB chat responses, and
  8 MiB exports.
- Capacity exhaustion rejects a new session instead of evicting an active one.
  Oversized input, output, and exports fail closed.

These controls reduce the attack and resource surface; they do not make the
workbench suitable for public internet exposure.

## Run and inspect

The deployed instance is managed by a user service:

```bash
systemctl --user daemon-reload
systemctl --user enable --now clanker-lm-web.service
systemctl --user status clanker-lm-web.service --no-pager
```

The repository declares web dependencies separately. For a fresh environment:

```bash
python -m pip install -e '.[web]'
```

The live service is equivalent to this bounded application launch, with the
public origin, allowlist, and session cap supplied explicitly by its unit:

```bash
python -m clanker_lm web \
  --host 127.0.0.1 \
  --port 8765 \
  --deployed \
  --public-origin https://bazzite.tail85f65f.ts.net:8444 \
  --allow-user jerrymares@gmail.com \
  --max-sessions 64 \
  --log-level warning
```

Do not start that command beside the active service; both would compete for the
same loopback port.

## Verify

```bash
systemctl --user is-enabled clanker-lm-web.service
systemctl --user is-active clanker-lm-web.service
curl -fsS http://127.0.0.1:8765/healthz
curl -fsS https://bazzite.tail85f65f.ts.net:8444/healthz
tailscale serve status
journalctl --user -u clanker-lm-web.service -n 100 --no-pager
```

The two health checks should return `{"status":"ok"}`. In
`tailscale serve status`, inspect the specific 8444 entry; it must read
`tailnet only` and proxy to `http://127.0.0.1:8765`. The host may carry unrelated
Serve or Funnel routes, so a global status label is not evidence about this
workbench's route.

Functional verification must use a browser signed into the Tailnet as the
allowlisted login. Store a fact, ask for it in the same session, inspect the
six fields in the evidence rail (**Answer**, **Truth**, **Source**,
**Certainty**, **Memory**, and **VADUG**), export the snapshot, and reset the
session.

## Live link correction

After deployment, Jerry reported that opening the workbench link from ACL was
blocked by the original cross-site bootstrap rule. Commit `6f3c1bf` corrected
that boundary: an authenticated, user-activated, top-level document navigation
may bootstrap from a same-site or cross-site link, while iframe, subresource,
fetch/CORS, and non-user cross-site requests remain denied without allocating a
session. The updated service was restarted and Jerry's live ACL-link retest
opened the workbench successfully.

This fix landed after the independent pre-rebase reviewer returned ACCEPT. The
live retest and exact-tree tests prove the correction works in the exercised
paths; final exact-head independent review has not yet been recorded.

## Restart and rollback

Restart the application without changing the Tailnet route:

```bash
systemctl --user restart clanker-lm-web.service
systemctl --user status clanker-lm-web.service --no-pager
```

The reversible deployment rollback is to disable and stop the backend:

```bash
systemctl --user disable --now clanker-lm-web.service
```

Restore it with:

```bash
systemctl --user enable --now clanker-lm-web.service
```

Stopping the backend makes the workbench unavailable while leaving unrelated
Tailscale routes untouched. Do **not** use `tailscale serve reset` on this shared
host: it would remove routes outside this workbench. If the dormant 8444 handler
itself must be removed, change only that handler through the host's approved
Tailscale configuration workflow.

## Validation receipt

The current exact deployment tree passed:

```text
88 web tests passed
2,694 full-suite tests passed, 2 expected xfails
29/29 deterministic benchmark turns
compile, JavaScript syntax, and diff checks clean
engine/ and clanker_engine.py unchanged
```

An independent web reviewer returned **ACCEPT** before the rebase. The rebased
implementation later received the `6f3c1bf` live-link correction. The current
exact tree passed the dedicated web suite, full suite, benchmark, compile,
JavaScript syntax, diff, and engine-boundary checks; these results do not claim
that a reviewer inspected the final exact head.
