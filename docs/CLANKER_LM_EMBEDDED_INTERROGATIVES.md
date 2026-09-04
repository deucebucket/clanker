# Embedded Interrogative Content

Clanker-LM represents a reported or requested question as a matrix event, a
nonassertive inner question event, and a typed relation between them. The inner
question is not promoted into a direct user question or a factual assertion.

## Representation

```text
Sarah asked who called.

ASK(agent=Sarah), source=user
CALL(agent=?agent), source=attributed, discourse_role=interrogative
EMBEDDED_WH(
  status=asked,
  matrix=ASK,
  question=CALL,
  requested_role=agent,
  marker=who,
  licensed=true
)
```

For polar content:

```text
John wondered whether Mary left.

WONDER(experiencer=John), source=user
LEAVE(agent=Mary), source=attributed, discourse_role=interrogative
EMBEDDED_POLAR(status=wondered, marker=whether, licensed=true)
```

The matrix and inner events retain independent tense, polarity, modality,
certainty, source, and identity. `licensed` follows matrix polarity. A negated
matrix event records the mentioned question while withholding positive evidence
that the source actually asked, knew, remembered, or wondered it.

## Supported matrix predicates

| Predicate | Status | Family | Recipient allowed | Imperative answer request |
|---|---|---|---:|---:|
| `ask` | asked | questioning | yes | no |
| `wonder` | wondered | uncertain cognition | no | no |
| `know` | known | knowledge | no | no |
| `remember` | remembered | memory | no | no |
| `discover` | discovered | discovery | no | no |
| `determine` | discovered | determination | no | no |
| `tell` | requested | answer request | yes | yes |

Only reviewed matrix predicates enter this path. A WH-looking modifier after an
ordinary object is not enough to create a question relation.

## Supported inner questions

- subject and object `who`/`whom`;
- subject and object `what`;
- `when`, `where`, `why`, and `how`;
- bounded `which` selection;
- bounded `whose` possessed-subject questions;
- polar `whether` and `if`.

One embedded question level is supported. Competing markers or multiple
question-taking matrix predicates produce a typed
`EmbeddedInterrogativeAttachmentAmbiguity` and suppress unsafe storage.

## Truth boundaries

Mentioning a question never supplies its answer:

```text
Sarah asked who called.
Who called?
-> I don't know the agent.
```

The attribution itself is answerable:

```text
Who asked who called?
-> Sarah asked who called.

What did Sarah ask?
-> Sarah asked who called.
```

Polar attribution preserves the entire inner proposition:

```text
Did John wonder whether Mary left?
-> Yes. John wondered whether Mary left.
```

But the inner fact remains unknown unless separately asserted:

```text
Did Mary leave?
-> I don't know whether Mary left.
```

Matrix and inner negation remain separate:

```text
Sarah did not ask whether Mary left.
```

records a negated `ASK` relation and a positive mentioned `LEAVE` proposition.
Neither establishes that Mary left.

```text
Sarah asked whether Mary did not leave.
```

records a positive `ASK` relation and a negative mentioned `LEAVE` proposition.
It still does not establish that Mary did not leave.

Positive and negated attributions for the same source and question produce an
explicit conflict rather than an arbitrary winner.

## Direct answer requests

An imperative answer request retains its outer command while passing the inner
question to the ordinary evidence binder:

```text
John went to Chicago.
Tell me where John went.
-> John went to Chicago.
```

Without evidence:

```text
Tell me where John went.
-> I don't know where John went.
```

The command itself cannot manufacture the destination.

## Outer epistemic questions

An outer question remains distinct from the inner information request:

```text
Do you remember when the meeting starts?
```

is represented as an outer polar `REMEMBER` question containing an inner
`WHEN` frame. If the inner frame can be answered from stored evidence, the
outer answer is true and may include the fact. Otherwise the answer is false
and explicitly states that the inner answer is unknown.

## Memory and matching

Inner question events use:

```text
source=attributed
discourse_role=interrogative
```

`ConversationMemory.match_events()` excludes them by default. A diagnostic or
attribution-aware caller must explicitly request
`include_interrogative_content=True`. This prevents a reported question from
entering ordinary factual Q&A.

Memory snapshot version 5 stores relation IDs, matrix and question event IDs,
source identity, marker, question kind, requested role, focus, certainty,
license, and direct-request status. Versions 1 through 4 remain loadable and
initialize the new relation collection as empty.

## Realization

Responses are composed from the matrix event and the inner `QuestionFrame`.
The realizer does not replay the original sentence and does not store a
complete response template. It inflects the matrix predicate, realizes any
recipient, renders the embedded operator in indirect word order, preserves
negation, and validates the resulting candidate against the answer contract.

Examples:

```text
Sarah + ask[past] + who + call[past] + .
-> Sarah asked who called.

John + wonder[past] + whether + Mary + leave[past] + .
-> John wondered whether Mary left.
```

## Deliberate limits

- one embedded interrogative level;
- reviewed matrix-predicate catalog only;
- no free indirect discourse;
- no unrestricted ellipsis or implicit source recovery;
- no automatic answer assertion from `know` or `discover` when the answer value
  is absent;
- ambiguous relative/question and recipient/question attachments fail closed.
