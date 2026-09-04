# Selected infinitival complements: control, raising, and truth boundaries

Clanker-LM represents one reviewed level of selected `to`-infinitival content
as two event frames plus a typed `InfinitivalRelation`. The matrix event is a
direct user assertion. The embedded event is retained as nonassertive content
whose subject is licensed by the matrix predicate.

```text
Sarah told John to call Mary.

matrix event:      TELL(agent=Sarah, patient=John)
embedded event:    CALL(agent=John, patient=Mary)
relation:          OBJECT_CONTROL
content status:    DIRECTED
controller:        John
embedded source:   attributed
entailed complete: false
```

No stored response sentence is introduced by this layer. Answers are composed
from the matrix event, embedded event, relation metadata, morphology, atomic
lexemes, and the ordinary VADUGWI candidate-ranking path.

## Supported relation classes

### Subject control

The matrix subject controls the unexpressed embedded subject.

```text
Sarah plans to leave.
Sarah intends to call Mary.
Sarah hopes to win.
Sarah wants to leave.
```

Initial content states are `PLANNED`, `INTENDED`, `HOPED`, and `DESIRED`.

### Object control

A licensed matrix object controls the embedded subject.

```text
Sarah told John to leave.
Sarah asked John to call Mary.
Sarah wants John to leave.
```

Initial content states are `DIRECTED`, `REQUESTED`, and `DESIRED`. A predicate
that requires object control fails explicitly when no controller is present.
A subject-control-only predicate fails explicitly when an object controller is
inserted.

### Raising

The surface matrix subject is shared with the embedded predicate, but the
matrix predicate contributes evidential status rather than an additional
volitional actor role.

```text
Sarah seems to know the answer.
Sarah appears to be tired.
```

Raising content is stored as `EVIDENTIAL` and is not promoted to an unqualified
fact.

## Purpose-adjunct boundary

Movement-purpose syntax remains distinct:

```text
Sarah went to buy groceries.
```

This remains one `GO` event with a `purpose` argument. It does not create a
selected infinitival relation. Predicate/form pairing is closed over the
reviewed catalog; the presence of `to` alone is never sufficient.

## Independent polarity scopes

Matrix and embedded negation are represented separately:

```text
Sarah did not plan to leave.
matrix polarity:   negative
embedded polarity: positive
relation licensed: false

Sarah planned not to leave.
matrix polarity:   positive
embedded polarity: negative
relation licensed: true
```

Infinitival polar questions carry both values. Consequently:

```text
Sarah planned not to leave.
Did Sarah plan to leave?
-> FALSE

Did Sarah plan not to leave?
-> TRUE
```

This prevents embedded negation from being mistaken for negation of the matrix
plan, desire, request, command, or appearance.

## Completion is never inferred

Every relation in this slice has `entailed=false`. The following assertion:

```text
Sarah planned to leave.
```

supports questions about the plan:

```text
What did Sarah plan?
Who planned to leave?
Did Sarah plan to leave?
```

It does not prove the embedded event:

```text
Did Sarah leave?
-> Sarah planned to leave. I do not know whether Sarah left.
```

The same boundary applies to desires, intentions, hopes, requests, directives,
and appearances. A later direct user assertion of completion remains ordinary
first-party evidence and outranks the nonentailed infinitival context.

## Memory and persistence

`ConversationMemory` binds parser event indices to stable IDs and validates:

- matrix predicate identity;
- source entity licensing;
- subject- versus object-controller licensing;
- embedded subject/controller identity;
- embedded `infinitive` discourse role;
- attributed provenance;
- independent matrix and embedded polarity;
- matrix-polarity license state;
- the hard `entailed=false` invariant.

Runtime and memory snapshot version 4 stores infinitival relations. Versions
1–3 remain loadable; missing infinitival fields are migrated to an empty
relation set rather than reconstructed speculatively.

## Explicit limits

This delivery intentionally supports one selected infinitival level. It fails
closed for:

- nested selected infinitives;
- finite attributed content inside an infinitive;
- an infinitive inside finite attributed content;
- selected infinitives stacked with relative clauses, appositives,
  subordination, or coordination;
- unsupported controller noun-phrase structures;
- predicate/form combinations absent from the reviewed catalog.

Failures produce typed `InfinitivalAttachmentAmbiguity` records with source
surfaces, candidate boundaries, candidate relation types, stable IDs, and
diagnostics. They do not silently flatten content into a direct object or
invent a controller.

## Validation

The dedicated suite contains 259 tests, including 200 generated conformance
cases spanning present and past subject control, object control, and raising.
It also covers polarity pairs, truth-boundary Q&A, multiple controllers,
purpose-adjunct contrast, snapshot compatibility, deduplication, staged-layer
failures, and serialization. The full repository suite must remain green and
no file under `engine/` is modified.
