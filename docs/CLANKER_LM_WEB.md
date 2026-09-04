# Clanker-LM private web workbench

The Clanker-LM browser workbench is live at
**[https://bazzite.tail85f65f.ts.net:8444/](https://bazzite.tail85f65f.ts.net:8444/)**.
It is a private Tailnet service, not a public website. Access requires membership
in the Tailnet and the exact allowlisted Tailscale login
`jerrymares@gmail.com`.

The current live service is the reviewed PR #106 baseline. Issue #112's
changelog and build-identity contract are present only on its branch until the
post-merge release-row update, reviewed deployment, and live probes described
below. Nothing in this guide claims that branch work is already live.

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
| Build identity | Full installed commit supplied as `CLANKER_LM_BUILD_COMMIT` |

The process refuses a non-loopback bind. Deployed mode also requires an exact
browser origin, at least one allowlisted Tailscale identity, and a full lowercase
40-character build commit that is not the local all-zero sentinel. The health
route returns only `{"status":"ok"}`; every implemented application resource
and API operation requires the allowlisted identity, and mutations must match
the configured origin. Unmatched paths and method mismatches may return
framework `404`/`405` responses without allocating a session or exposing
application data.

## Using the workbench

Enter a fact, question, or resolver request in the composer. Each answer carries
a six-field evidence rail: **Answer**, **Truth**, **Source**, **Certainty**,
**Memory**, and **VADUG**. Memory is isolated by browser session. Use **Export
state** to save an inspectable Clanker-LM snapshot or **Reset session** to close
that runtime and start with empty memory.

After #112's gated deployment, use **Changelog** in the masthead to open the keyboard- and mobile-accessible
shipped-change record. The **Running artifact** specimen shows the runtime
package and separately injected build commit. Each ordered milestone card shows
its own package and merge commit, explains the capability in plain language,
links to repository evidence, and states its known limitations. The first
record is the reviewed and merged
[PR #106](https://github.com/deucebucket/clanker/pull/106) baseline at commit
`9ae77f072f8afda0b1d2b757ab492757cabff0f8`; it does not claim #107 or any
later roadmap work as shipped.

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
- The changelog fetches only the authenticated, same-origin `/api/releases`
  endpoint. Its source is the bounded packaged `releases.json` file; application
  startup rejects an unknown schema, out-of-order dates, version/commit drift,
  mismatched `pr-N`/pull-request evidence, mismatched milestone/commit evidence,
  non-repository evidence URL, unpinned deployment URL, or private-content field.
  Dynamic copy is rendered with `textContent`, never HTML insertion.
- `deployed_build_commit` is composed into `/api/releases` from `WebConfig`; it
  is not stored in `releases.json`. This prevents the ledger from pretending its
  own not-yet-created commit is the running build. CI separately asks GitHub to
  verify that every milestone PR is merged at its recorded commit; the runtime
  performs no network lookup.
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
public origin, allowlist, build identity, and session cap supplied explicitly by
its unit. Set the environment value from the exact installed artifact, never
from the release ledger:

```bash
export CLANKER_LM_BUILD_COMMIT="$(git -C /opt/clanker-lm rev-parse HEAD)"
test "${#CLANKER_LM_BUILD_COMMIT}" -eq 40
python -m clanker_lm web \
  --host 127.0.0.1 \
  --port 8765 \
  --deployed \
  --build-commit "$CLANKER_LM_BUILD_COMMIT" \
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

On the trusted host, verify the configured API identity against the checked-out
artifact independently of the milestone feed:

```bash
CLANKER_LM_EXPECTED_BUILD="$(git -C /opt/clanker-lm rev-parse HEAD)"
CLANKER_LM_RELEASE_JSON="$(curl -fsS \
  -H 'Tailscale-User-Login: jerrymares@gmail.com' \
  http://127.0.0.1:8765/api/releases)"
CLANKER_LM_REPORTED_BUILD="$(python -c \
  'import json,sys; print(json.load(sys.stdin)["deployed_build_commit"])' \
  <<<"$CLANKER_LM_RELEASE_JSON")"
test "$CLANKER_LM_REPORTED_BUILD" = "$CLANKER_LM_EXPECTED_BUILD"
python -m clanker_lm.web_release_verify
```

That equality is the runtime-build gate. The verifier's separate success proves
that each `pr-N` milestone exists, is merged, and names GitHub's merge commit.
Neither fact substitutes for the other.

Functional verification must use a browser signed into the Tailnet as the
allowlisted login. Store a fact, ask for it in the same session, inspect the
six fields in the evidence rail (**Answer**, **Truth**, **Source**,
**Certainty**, **Memory**, and **VADUG**), export the snapshot, and reset the
session. Open **Changelog** by keyboard and pointer; confirm **Runtime build**
matches `CLANKER_LM_EXPECTED_BUILD`, while the PR #106 card separately reads
milestone commit `9ae77f072f8afda0b1d2b757ab492757cabff0f8` and **Live · private
Tailnet**. Close it with the visible button and with Escape, and repeat at a
narrow mobile viewport without horizontal clipping.

## Maintaining the shipped feed

The source of truth is
`clanker_lm/web_assets/releases.json`, ordered newest first. Add an entry only
after the milestone is merged, independently reviewed, and deployed. The new
top entry and `latest_shipped_release` must agree exactly on release ID, package
version, and full 40-character milestone commit. Every `pr-N` row must contain
the exact `/pull/N` evidence URL and an evidence URL for its exact milestone
commit. Run `python -m clanker_lm.web_release_verify` to prove through GitHub
that the PR is merged at that commit. Use only evidence links below the
`deucebucket/clanker` GitHub repository and keep the direct deployment URL pinned exactly as
`https://bazzite.tail85f65f.ts.net:8444/`.

Do not copy ACL, browser, session, prompt, response, or raw transcript content
into the feed. A development branch and an accepted PR are not a shipped entry;
deployment verification is the final gate. The server deliberately fails at
startup instead of presenting a feed whose newest record disagrees with the
running package version or milestone identity. Never add
`deployed_build_commit` to this file; deployed configuration supplies it.

For #112 specifically, the current pre-merge feed must remain PR #106-only.
After #112 merges, add a PR #112 row in a follow-up reviewed commit using the
actual #112 merge commit and matching evidence links. Set
`CLANKER_LM_BUILD_COMMIT` to the full SHA of the follow-up artifact being
deployed, run the GitHub verifier and all release gates, deploy, then prove the
API/UI build equality above. Do not close #112 or describe it as shipped before
that post-merge row, deployment, and live verification are complete.

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

The final exact-head release gate returned **APPROVE** for
`780d77b4673aa45a692fc5a1f8af144a41f09fd0`, including automated and independent
review, before PR #106 was squash-merged as
`9ae77f072f8afda0b1d2b757ab492757cabff0f8`.

Issue #112's current branch validation receipt is recorded in
`docs/CLANKER_LM_DEVLOG.md`. It remains branch evidence only and does not claim
that #112 is merged or live.
