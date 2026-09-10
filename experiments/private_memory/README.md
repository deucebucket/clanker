# Private memory compartments (opt-in service boundary)

Issue #145; related #42/#125/#127. This adapter calls the existing ClankerLM.
It is not another generator and stores no response templates. It is excluded
from production wheel discovery and is NOT connected to the anonymous web route.

## Identity contract

The storage key is derived from `(trusted principal, compartment)` using a
private application key. A principal is an opaque authenticated account ID,
not a display name, a claimed identity inside chat, or an arbitrary request
parameter. Two local `jordan_1` records are distinct when their namespaces
are distinct. Names are labels; private references include the namespace.

One principal can use separate personal/work compartments. Reusing the same
principal and compartment across a service restart resumes that record. This
slice serializes an entire conversation state per compartment; it does not
merge concurrent independent threads or provide cross-user sharing grants.
Within-compartment same-name person disambiguation remains a parser/resolver
problem. Matching names never create cross-compartment SAME_AS edges.

Private snapshots include facts, entity associations, participant appraisals,
VADUGWI state, dialogue obligations, lexical hypotheses, counts and receipts.
A shared immutable language pack may be read by many runtimes; mutable stores
and learned overlays must not be shared. The service creates/restores a fresh
in-memory runtime for every operation and refuses an injected file database.
Trusted factories must supply genuinely fresh runtimes, not reusable objects.

## Minimal trusted-server use

```python
from experiments.private_memory import PrivateMemoryService

service = PrivateMemoryService(
    "/path/owned/by/application/private-state",
    secret=server_key,  # At least 32 bytes, provisioned privately; no default key.
    configuration_id="reviewed-native-config-v1",
)
result = service.process(authenticated_principal, message, compartment="personal")
```

`authenticated_principal` must come from separately reviewed server-side
identity resolution. This module has no login provider and cannot determine
whether a caller is lying about a principal. Do not expose its parameter as
an unauthenticated HTTP user selector. Possession of a namespace or record ID
is not authorization. Default anonymous sessions remain a separate path.

The directory must be application-owned and mode 0700. Files are 0600;
no-follow opens and regular/single-link checks reject inappropriate files.
Ancestors of the configured directory and the OS/application account are
trusted. This is local POSIX storage, not distributed/NFS coordination.

## Operations

- `process`: restore private state, call ClankerLM.process, atomically commit
  the successful generation. A failed operation preserves the previous file.
- `entities` / `entity`: enumerate and resolve namespace-qualified records.
- `graph`: render only that runtime's actual graph; reject foreign roots.
  A native runtime without MemoryWeb reports that absence explicitly. Top-level
  graph node and edge IDs are qualified; nested evidence IDs remain local to
  the envelope's declared namespace.
- `lexicon`: inspect only this private overlay.
- `export_snapshot` / `restore_snapshot`: authenticate namespace, software,
  configuration, complete state and generation before materialization.
- `reset`: write empty state at a higher generation; do not affect another
  compartment. A stale backup cannot undo this while the durable generation
  record remains. Reset does not erase external backups or export copies.
- `verify_response`: authenticate a historical response envelope's scope and
  configuration. This is not a freshness, factual-truth or semantic-quality check.

There is no automatic export to a shared vocabulary, no global learning
promotion, no implicit import of unsigned old snapshots, and no silent key or
software migration. A service upgrade must handle those migrations explicitly.

## Integrity, atomicity, and limits

The HMAC authenticates content to a holder of the server key. It does NOT
provide encryption. Someone who can read the files can read the private data;
someone with the server key can authenticate modified data. Protect the host,
key, backups and exports accordingly. An attacker controlling the storage can
roll back/delete the generation record itself; there is no external anti-
rollback journal or public cryptographic attestation in this slice.

Every public data operation derives scope from the trusted principal before
lookup. Snapshot validation occurs before runtime construction. Copying another
scope's file, changing a namespace, or recomputing an unkeyed hash does not
produce a valid envelope. Full snapshots/receipts are private, not safe metrics.

Per-compartment file locks bound lock waiting and serialize local worker
updates. Different compartments use different locks. A fresh restored runtime
and its SQLite connection remain inside that operation. Writes use temporary
files, fsync and atomic replacement. Disk/OS failures after replacement can
still leave caller uncertainty about whether a write completed; request-ID
idempotency is not implemented. Maximum input size, turns and snapshot bytes
are bounded; account/directory quotas and rate limits belong to the service
host. Graph depth/node limits reuse the existing runtime controls.

## What was actually tested

- 40 native-main-compatible tests; 12 additional delivered graph/decoder tests.
- Different users with the same local name, object words and opposite appraisals.
- Different meanings of glorp and separate pending definitions/usage counts.
- Foreign references/exports rejected, tampering, stale restore, reset, bad
  configuration/key, unsigned legacy payloads and factory-scope mistakes.
- Threaded interleaving plus six independent Python worker processes writing
  one compartment; bounded lock waits and file restrictions.
- An actual two-user, 200-call test matches uninterrupted per-user reference
  runtimes, including all 100 calls after service restart. No generated output
  is replaced by an answer template or imported from another session.

The native public subset does not require the separately delivered decoder.
Its actual native test run passes 40/40 against the production source from
29e2c5f... (same production bytes as main 9cf8690...). The graph/decoder tests
are run only on delivered bd744f4... plus this service. Do not present those
figures as default-main graph behavior or public-browser authentication.

The transcript and measurements use original authored examples, a small
unchanged count pack, real V8 and CPython 3.13.5. They are not an independent
security audit, 100-user load test, emotional-outcome evaluation, or claim of
unrestricted identity resolution. Same-person sharing and linked-user consent
are deliberately not inferred.
