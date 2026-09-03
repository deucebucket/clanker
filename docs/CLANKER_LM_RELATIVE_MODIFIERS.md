# Clanker-LM Typed Relative Modifiers

Issue #53 adds one finite restrictive or nonrestrictive entity modifier per
independently coordinated assertion.

## Representation

An `EntityModifierRelation` stores:

- the stable head entity ID;
- the modifier event index and stored event ID;
- the relative marker (`who`, `whom`, `whose`, `which`, or `that`);
- the missing semantic role (`agent`, `patient`, or `possessor`);
- restrictive versus nonrestrictive status;
- certainty and deterministic diagnostics;
- the possessed entity ID for `whose` constructions.

The modifier clause is also stored as an ordinary event with
`discourse_role=modifier`. It can therefore answer later deterministic
questions without flattening the modifier into the main event.

## Identity

Restrictive generic descriptions use a deterministic modifier-scoped identity.
`the woman who called Sarah` and `the woman who called Mary` remain distinct,
while the shared alias `woman` later produces an explicit ambiguity. Proper-name
nonrestrictive modifiers reuse the existing entity.

## Supported boundaries

This slice covers one finite relative marker and subject, object, or possessive
gaps. Multiple relative markers produce an explicit
`ModifierAttachmentAmbiguity` instead of a first-match attachment. Abstract
content heads such as `the fact that ...` and matrix complementizer `that`
remain deferred to issue #54.

Conversation-memory snapshot version 3 stores modifier relations. Version-1 and
version-2 snapshots remain readable with an empty modifier list.
