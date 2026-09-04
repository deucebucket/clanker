# Finite attributed content complements

Clanker-LM represents one level of finite declarative content after licensed
speech, belief, knowledge, and perception predicates as two events plus a typed
`ContentRelation`.

```text
Sarah said that John left.

matrix event:  SAY(agent=Sarah)
content event: LEAVE(agent=John), discourse_role=content, source=attributed
relation:      REPORTED(matrix, content, source=Sarah, marker=that)
```

The same representation supports structurally licensed zero complementizers:

```text
Sarah thinks John left.
Sarah told me John left.
```

Supported initial relation types are `REPORTED`, `BELIEVED`, `KNOWN`, and
`PERCEIVED`. The relation retains the matrix predicate, source entity, marker,
predicate family, certainty, diagnostics, and stable matrix/content event IDs.

## Truth boundary

Attributed content is stored but excluded from ordinary event matching by
default. Therefore:

```text
Sarah said that John left.
Did John leave?
```

returns source-qualified uncertainty rather than `TRUE`. A direct user assertion
of the proposition is stored as independent evidence and can answer the truth
question normally. Contradictory attributed speakers remain distinct relations
and do not become an unqualified global contradiction.

Questions such as `What did Sarah say?`, `What did Sarah think?`, and
`What did Sarah notice?` traverse the content relation and realize the answer
with explicit attribution.

## Conservative boundaries

- A non-`tell` matrix predicate does not accept a substantive direct object
  before complementizer `that`; this protects noun-relative clauses such as
  `reported the idea that changed everything`.
- Zero complementizers require a licensed matrix predicate and an independently
  finite right clause.
- Nested finite content beyond one level produces a typed
  `ContentAttachmentAmbiguity` and suppresses unsafe durable storage.
- Relative clauses, ordinary direct objects, and unresolved source pronouns do
  not silently become content relations.

The implementation remains compositional and does not introduce stored response
sentences or changes to the V8 engine.
