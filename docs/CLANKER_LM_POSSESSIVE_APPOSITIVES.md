# Possessive relatives and appositive identity links

Issue #84 adds two bounded deterministic structures without changing the V8 engine.

## Possessive relatives

`The man whose car broke called me` creates:

- a matrix `call` event;
- a modifier `break` event;
- one `EntityModifierRelation` with `gap_role=possessor`;
- a possessed car entity whose `owner_id` is the man entity.

The transformed internal alias is resolved before generic noun heuristics, so the modifier event and relation always reference the same possessed entity. `whose` questions can bind the owner directly from that explicit metadata.

## Appositives

`Sarah, my supervisor, called` and `My supervisor Sarah called` create one canonical Sarah entity plus an `AppositiveRelation`. The relation records both surfaces, whether the apposition is restrictive or nonrestrictive, its role/identity/description type, certainty, and diagnostics.

An appositive alias is accepted only when it is type-compatible and does not already identify another entity. Conflicting roles or aliases produce `AppositiveAttachmentAmbiguity` and suppress durable assertion storage rather than silently merging identities.

No completed response sentence is stored by this feature.
