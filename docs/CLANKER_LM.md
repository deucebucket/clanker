# Clanker-LM 0.2

Clanker-LM is an adaptive deterministic semantic conversation runtime built
around the existing Clanker V8 VADUGWI engine.

It does not predict the next token. It does not store completed response
sentences. It does not retrieve canned replies. A turn is parsed into symbolic
state, resolved against evidence, compiled into one or more grammatical
compositions, and evaluated through Clanker's A+B=C state transition.

```text
language
  -> entities and event frames
  -> typed question / semantic command
  -> evidence binding or explicit unknown
  -> response act and target VADUGWI state
  -> abstract grammar + atomic lexemes
  -> semantically validated candidates
  -> Clanker state-transition scoring
  -> deterministic reply
```

## The no-template invariant

The language database is deliberately unable to store a response sentence.

`atoms` accepts one surface token per row and has a SQL check rejecting
whitespace. Examples are `what`, `does`, `sorry`, `because`, `?`, and `.`.

`grammar_rules` contains only abstract symbols:

```text
reply:answer -> DECLARATIVE_CLAUSE
clause:declarative -> SUBJECT FINITE_PREDICATE COMPLEMENTS ADJUNCTS
phrase:nominal -> DETERMINER MODIFIERS HEAD
```

There is no `template` column and there are no legacy `constructions`,
`construction_slots`, or `graph_edges` tables. `LanguageStore` verifies this
invariant when it opens. Tests also inspect the live SQLite schema and every
stored atom.

Dynamic facts are realized by generic morphology and semantic-role ordering.
For example, the runtime does not store:

```text
Your sister bought the used Honda yesterday.
```

It stores an event:

```text
BUY(
  agent=sister_1,
  patient=honda_1,
  time=yesterday,
  tense=past
)
```

The realizer determines the subject, inflects `buy -> bought`, realizes the
bound entity and temporal adjunct, and applies punctuation. The generated
candidate carries an inspectable `semantic_plan` and the identifiers of every
atomic database lexeme used.

Unknown answers and clarification probes obey the same rule. `What does glorp
mean?` is assembled from a question operator, auxiliary, dynamic term,
definition predicate, and punctuation atom. It is not a stored exception.

## Semantic representation

The core structures are:

- `Entity`: canonical identity, aliases, relation, gender, number, attributes,
  salience, and mention history.
- `SemanticRef`: entity, literal, event, or typed variable.
- `EventFrame`: predicate, semantic arguments, tense, aspect, polarity,
  modality, source, certainty, and discourse role.
- `QuestionFrame`: the proposition plus the exact role left open by the
  question.
- `AnswerContract`: evidence-backed obligations and forbidden claims for the
  response.

A question is treated as a nearly completed proposition:

```text
What did she buy?

BUY(
  agent=sister_1,
  patient=?patient,
  tense=past
)
```

Only a fact preserving the fixed roles and truth conditions may bind the open
slot. The runtime distinguishes direct answers, true, false, unknown,
contradictory evidence, missing references, ambiguous references, multiple
valid bindings, unsupported structures, lexical probes, and learned lexical
senses.

## Truth boundary

Semantic validity is a hard lock. Affect scoring cannot rescue a false or
unsupported candidate.

```text
valid(response) =
    answers_requested_slot
    AND preserves_entity_bindings
    AND preserves_polarity
    AND is_supported_by_evidence
    AND obeys_reference_resolution
    AND obeys_contextual_gates
```

Only valid candidates proceed to VADUGWI ranking.

Absence of evidence is not converted to false. If memory says Sarah bought a
car, that does not prove Mary did not buy it. Explicit positive and negative
facts produce a conflict state rather than an arbitrary winner.

## Active lexical learning

Unknown vocabulary is learned through versioned hypotheses, not by editing the
canonical Clanker vocabulary in place.

When a high-confidence unknown occupies an important semantic position:

```text
That movie was glorp.
```

Clanker-LM records the occurrence and composes one focused probe:

```text
What does glorp mean?
```

A useful answer becomes evidence:

```text
Negative, like disappointing and overhyped.
```

The learner extracts a semantic class, polarity, contextual VADUGWI signal,
part of speech, and scope. It then recomputes the sense hypothesis and returns a
composed confirmation such as:

```text
Glorp means negative evaluation.
```

### Clarification budget

Only one unknown term is probed at a time. A low-information explanation is
still saved, but it does not receive artificial confidence. The next probe is
reduced to one discriminating dimension:

```text
Polarity?
Meaning?
Intense?
Example?
```

After the configured probe budget, the evidence remains stored as provisional
rather than trapping the user in an interrogation loop.

### Ongoing correction

Learning continues after promotion. Every later use of a learned word adds a
low-weight context observation. The learner removes the target word, scores the
surrounding context, and uses that independent signal to update the hypothesis.

Every update:

1. preserves the original evidence;
2. creates a new sense version;
3. re-runs old contexts through the current sense set;
4. records the assigned sense and interpreted vector;
5. lowers, shifts, splits, or promotes hypotheses according to accumulated
   evidence.

Conflicting positive and negative definitions can create conditioned senses
rather than averaging incompatible meanings into a neutral entry.

Learned vocabulary lives in the adaptive overlay tables:

```text
learned_terms
learned_senses
lexical_evidence
```

Canonical atoms and learned senses therefore remain physically and logically
separate.

## Semantic command resolvers

Some questions request a live operation rather than conversational memory.
`ResolverRegistry` routes those questions to deterministic providers and turns
the result into an ordinary `AnswerContract`.

Included resolvers:

- current time, with IANA timezone and common-place aliases;
- current date;
- bounded arithmetic using a safe AST evaluator.

Example:

```text
What time is it in Tokyo?
  -> CURRENT_TIME(location=Asia/Tokyo)
  -> system clock observation
  -> BE(subject=current time in Tokyo, value=1:05 AM)
  -> compositional realization
```

Live observations record source, certainty, observation time, and expiration.
The runtime never learns that a transient clock value is permanently true.
Arithmetic accepts numbers and bounded arithmetic operators only; names,
function calls, attributes, code execution, unbounded powers, and non-finite
results are rejected.

Additional providers can implement the resolver protocol and register behind
the same boundary. Weather, document retrieval, databases, and APIs can return
typed observations without changing generation.

## Multi-turn transition learning

Sentence scoring describes an utterance. Conversation modeling requires the
transition between turns.

Each generated response stores only:

```text
hash(input)
hash(response)
incoming dialogue act
response dialogue act
input VADUGWI
state before the turn
target state
response VADUGWI
predicted post-response state
next observed state
signed transition residual
```

No input or response sentence is written to the trajectory tables.

When the next user turn arrives, the previous transition is finalized. Similar
contexts accumulate a signed mean residual and observed success rate. If a
response type systematically lands below or above the predicted axis values,
the next target is corrected in the opposite direction. Corrections are capped,
and critical safety floors cannot be learned away.

The context key includes:

- incoming dialogue act;
- response dialogue act;
- answer status;
- severity;
- collision masking state;
- quantized input and conversational vectors;
- active corpus profile.

This allows the controller to learn response physics for communication types,
not merely isolated emotional words.

## Dialogue and book trajectory profiles

`CorpusProfiler` extracts quoted or speaker-labelled dialogue and compiles each
turn to seven bytes of VADUGWI. It stores:

- the complete packed vector trajectory;
- centered signed deltas;
- centroid and per-axis variance;
- dialogue-act distribution;
- dialogue-act transitions;
- multi-resolution trajectory chunks;
- cryptographic fingerprints and source hashes.

It does not store quotations or book text.

When a profile is active, response targets are gently pulled toward the
profile's current phase and movement, not merely its global average. The phase
advances with conversation turns. High-severity and critical gates reduce the
profile strength so a dramatic corpus cannot override safe behavior.

This conditions affective contour and conversational movement. It does not
claim that seven emotional coordinates encode every property of prose style.
Exact diction, syntax, metaphor, and author identity require additional
non-textual feature channels if those are desired.

A known corpus can be matched by affect trajectory, delta sequence, dialogue
acts, and multi-resolution chunks. Exact source reconstruction is intentionally
out of scope: VADUGWI is lossy affective state, not a reversible text codec.

## SQLite layout

Static language and adaptive state share one inspectable SQLite connection but
remain separate table families.

```text
Atomic language
  atoms
  grammar_rules
  gate_rules

Lexical learning
  learned_terms
  learned_senses
  lexical_evidence

Live observations
  resolver_observations

Conversation calibration
  trajectory_turns
  transition_stats

Corpus profiles
  corpus_profiles
  trajectory_chunks
```

`LanguageStore.export_overlay()` serializes only adaptive tables. This allows a
reviewed static language seed to remain immutable while learned senses,
resolver history, trajectories, and corpus profiles move with a session
snapshot.

## Python API

```python
from clanker_lm import ClankerLM

with ClankerLM(default_timezone="America/Chicago") as runtime:
    print(runtime.process("That movie was glorp.").response)
    # What does glorp mean?

    print(runtime.process("Negative, like disappointing.").response)
    # Glorp means negative evaluation.

    print(runtime.process("What does glorp mean?").response)
    # Glorp means negative evaluation.

    print(runtime.process("What time is it in Tokyo?").response)
```

Compile and activate an affect-trajectory profile:

```python
text = open("public-domain-dialogue.txt", encoding="utf-8").read()

with ClankerLM() as runtime:
    profile = runtime.compile_corpus_profile(
        "dialogue-profile",
        text,
        activate=True,
    )
    result = runtime.process("My sister called me again.")
    print(result.trajectory["profile_adjustment"])
```

Inspect generation:

```python
result = runtime.process("Who bought the Honda?")
selected = next(item for item in result.candidates if item.text == result.response)

print(selected.semantic_plan)
print(selected.atom_ids)
print(selected.affect)
print(selected.predicted_state)
```

## Command line

```bash
python -m clanker_lm demo
python -m clanker_lm chat --memory session.json --trace
python -m clanker_lm once "What time is it in Tokyo?" --json
python -m clanker_lm parse "My sister bought a Honda."
python -m clanker_lm lexicon --memory session.json
python -m clanker_lm profile book.txt --name book-dialogue --memory session.json --activate
python -m clanker_lm profiles --memory session.json
python -m clanker_lm match quotation.txt --memory session.json
python -m clanker_lm tone off --memory session.json
python -m clanker_lm schema
```

## Validation

Run the complete suite:

```bash
python -m pytest -q
python benchmarks/clanker_lm_eval.py
```

The acceptance harness covers typed conversational slots, explicit false and
conflict states, pronoun probing, why/how/location roles, active lexical
learning, live semantic resolvers, and collision masking. The normal test suite
also checks template exclusion, safe arithmetic, snapshot persistence,
corrective sense splitting, prior-context reinterpretation, signed trajectory
learning, packed seven-byte vectors, and corpus matching.

## Extension boundary

New capability should enter through one of five typed seams:

```text
parser rule
semantic resolver
lexical evidence source
grammar/morphology rule
trajectory or corpus feature channel
```

None requires storing a completed response sentence or replacing the Clanker
VADUGWI kernel.
