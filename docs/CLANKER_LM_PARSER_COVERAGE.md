# Clanker-LM Deterministic Parser Coverage

This document records the explicitly supported grammar added through issue #48,
the first bounded implementation slice of #36.

## Finite morphology

The parser recognizes reviewed irregular forms and regular inflections only when
one of the following supplies evidence:

1. the surface form has an explicit reviewed lemma;
2. deterministic morphology maps it to a known predicate;
3. an auxiliary or modal licenses an otherwise unknown predicate.

This means `died` maps to `die`, `tried` maps to `try`, and `planned` maps to
`plan`. A bare invented token such as `florbed` does not become a verb merely
because it ends in `-ed`; `did florb` remains parseable because the auxiliary
provides independent grammatical evidence.

## Independently finite coordination

The assertion parser splits semicolons and the coordinators `and`, `but`, `yet`,
`or`, and resultative `so` only when both sides contain their own finite
predicate and explicit subject. The first event receives `discourse_role=main`;
subsequent primary events receive `discourse_role=coordinate`. The connector is
recorded in parser diagnostics.

The parser intentionally does not split:

- compound subjects (`Sarah and Mary opened the door`);
- compound objects (`bread and milk`);
- shared-subject/gapping continuations (`Sarah opened the door and closed the window`);
- subordinate `so that` clauses.

Those forms require separate deterministic transformations and remain tracked
under #36 rather than being guessed in this slice.
