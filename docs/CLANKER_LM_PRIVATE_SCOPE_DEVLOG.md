# Private scope: change/development log

## 2026-09-10 — priority identity boundary, issue #145

The owner asked whether each user's Jordan was separate. The executed round-six
baseline showed that fresh runtimes keep different facts despite both issuing
local `jordan_1`. It also showed `learning_scope=A/B` did NOT isolate one shared
LanguageStore, and ordinary snapshots had no authenticated expected-owner check.
Existing anonymous web sessions construct separate runtimes; this is not a claim
that the live app had one global Jordan or leaked data between those sessions.

Added an opt-in principal/compartment service over the existing runtime, scoped
external references, authenticated complete snapshots/responses, private local
file permissions, bounded locking, atomic committed generations, stale-restore
checks, reset, scoped lexical/graph reads, and independent worker tests. A server
must resolve the principal; no login or live-route integration is claimed.
No engine/, clanker_lm/ or evaluation/ production source changed in this round.

The first 40 native-compatible boundary tests passed; all 52 native/integrated
checks pass after response receipts were also bound to runtime/config identity.
An initial multiprocessing-pool test worked alone but an aggregate run timed
out in this environment. The final process test launches six independent bounded
stdlib-only Python workers; no unbounded pool or hidden skipped assertion is
used. Two ordinary aggregate attempts timed out; a native-startup attempt also
timed out later in the existing suite. None is claimed as a complete pass.

The interleaved-user school run passes 49/49 checks; 200/200 scoped turns match
uninterrupted references, including 100 after restart. Original count pack and
real V8 are used. See delivered JSON for measured latency, state sizes and memory.
The new code does not change the word decoder, semantic rules or emotional math.

Native publication is a separate opt-in service/tooling PR directly over main,
not another stacked parser patch. The integrated local branch retains the exact
same service. Default public authentication, same-user multiple-Jordan reference
disambiguation, sharing grants, snapshot migrations, distributed locking,
encryption/key rotation, request-ID replay and whole-directory quotas remain
explicit work. Matching labels alone never authorize a merge or lookup.
