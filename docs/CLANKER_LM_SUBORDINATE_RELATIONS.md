# Clanker-LM Typed Subordinate Relations

Issue #49 extends the deterministic parser with one finite subordinate relation
per independently coordinated assertion.

## Representation

A `ClauseRelation` records:

- the main and subordinate event indices used by the parser;
- stable event IDs after insertion into conversation memory;
- the connector surface;
- a typed relation kind;
- direction (`main_to_subordinate`, `subordinate_to_main`, `symmetric`, or
  `unresolved`);
- certainty and ambiguity candidates;
- deterministic diagnostics.

Conversation-memory snapshot version 2 stores these relations. Version-1
snapshots remain readable and load with an empty relation list.

## Resolved connectors

- `because` → cause
- `when` → temporal coincidence
- `before`, `after`, `until` → directed temporal relation
- `if` → condition
- `unless` → exception condition
- `although`, `though`, `even though` → concession

## Contextually resolved connectors

- `since` resolves temporally only when its subordinate clause contains an
  explicit time anchor; otherwise it remains cause/time ambiguous.
- `while` resolves as temporal overlap when both clauses are progressive;
  otherwise it remains overlap/concession ambiguous.
- `so that` resolves as purpose when the subordinate clause is modal, as result
  when a reviewed change-of-state cue is present, and otherwise remains
  purpose/result ambiguous.

Ambiguity is retained as data. The parser does not select a convenient relation
merely to complete a frame.

## Boundaries

Temporal prepositional phrases such as `after Monday` or `before dinner` are not
subordinate clauses because the material following the marker is not
independently finite. Relative clauses, complement clauses, and nested
subordination remain separate slices of #36.
