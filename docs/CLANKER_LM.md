# Clanker-LM deterministic dialogue runtime

Clanker-LM is the symbolic language layer around the existing Clanker VADUGWI engine. It does not predict the next token. It parses an utterance into an inspectable semantic frame, binds questions against explicit evidence, constructs only proposition-preserving replies, and then lets Clanker's `A + B = C` state transition rank the semantically valid alternatives.

This implementation is the first complete vertical slice of the architecture described in the Clanker-LM scaffolding:

```text
raw text
  -> deterministic semantic parser
  -> entity/coreference memory
  -> statement fact or QuestionFrame
  -> evidence binding / AnswerContract
  -> contextual gates
  -> construction graph or answer realizer
  -> hard semantic validation
  -> Clanker VADUGWI back-solve
  -> response + auditable trace
```

## Non-negotiable separation of concerns

Clanker-LM keeps factual correctness and affective suitability as separate locks:

```text
semantic layer:  What is supported by evidence?
affective layer: Which supported wording best moves the conversation?
surface layer:   How is the verified proposition rendered grammatically?
```

Affective scoring never rescues a semantically invalid candidate. If all candidates violate the answer contract or a contextual gate, the runtime raises an error rather than selecting the least-wrong hallucination.

## Implemented modules

| Module | Responsibility |
|---|---|
| `clanker_lm/models.py` | Typed semantic, question, evidence, gating, affect, and trace objects |
| `clanker_lm/normalize.py` | Deterministic tokenization, morphology, relation vocabulary, register/severity markers |
| `clanker_lm/parser.py` | Predicate/argument frames, WH/polar/why/how classification, Yoda-order handling |
| `clanker_lm/memory.py` | Persistent entities, aliases, salience, pronoun resolution, fact unification |
| `clanker_lm/answers.py` | Typed-hole binding, four-state truth handling, surface realization, semantic hard gate |
| `clanker_lm/gating.py` | Severity/register pools, collision masking, response-act planning |
| `clanker_lm/constructions.py` | Deterministic traversal of the JSON construction graph |
| `clanker_lm/affect.py` | Real Clanker adapter, target policy, `A + B = C` candidate back-solving |
| `clanker_lm/persistence.py` | Normalized SQLite sessions, entities, aliases, facts, and running VADUGWI state |
| `clanker_lm/runtime.py` | End-to-end conversational runtime |

The runtime remains Python-standard-library-only. The existing project dependency on `pytest` is sufficient for development and testing.

## Question as a typed hole

The parser treats a question as a mostly complete proposition with one requested role.

```text
What did your sister buy yesterday?

BUY(
  agent=sister_1,
  theme=?,
  time=yesterday,
  tense=past
)
```

A stored fact can bind the hole:

```text
BUY(agent=sister_1, theme=used_honda_1, time=yesterday)
```

The answer is then a declarative realization of the completed frame:

```text
She bought a used Honda yesterday.
```

The same fact can answer different roles:

- `Who bought it?` -> `AGENT`
- `What did she buy?` -> `THEME`
- `When did she buy it?` -> `TIME`
- `Why did she buy it?` -> `MOTIVE`, then `PURPOSE`, then `CAUSE`
- `How did she buy it?` -> `METHOD`, then `MANNER`

## Explicit answer states

Clanker-LM does not collapse knowledge into yes/no. `AnswerContract.status` is one of:

- `ANSWERED`: a requested role has one supported binding;
- `TRUE`: a fact entails the polar proposition as asked;
- `FALSE`: a fact explicitly contradicts the polar proposition;
- `UNKNOWN`: neither entailment nor contradiction is stored;
- `PARTIAL_UNKNOWN`: the event is known, but the requested role is missing;
- `AMBIGUOUS`: multiple distinct bindings or contradictory facts remain;
- `NEEDS_CONTEXT`: a pronoun/reference cannot be resolved safely;
- `RHETORICAL`: the syntax is interrogative but the pragmatic act is not factual Q&A.

This is deliberately open-world. The absence of a fact is `UNKNOWN`, not `FALSE`.

## Why and how semantics

`why` is routed into semantic subtypes instead of one generic bucket:

```text
CAUSE          Why did the window break?
MOTIVE         Why did Sarah leave?
PURPOSE        Why did Sarah go to the store?
JUSTIFICATION  Why should I apologize?
EVIDENCE       Why do you believe that?
RHETORICAL     Why would anyone love me?
```

`how` is likewise typed:

```text
METHOD         How did she enter?
MANNER         How quickly did it happen?
PROCESS        How does the engine calculate valence?
MECHANISM      How does the lock work?
DEGREE         How tall is Sarah?
QUANTITY       How many books did Sarah buy?
CONDITION      How is the engine doing?
SOCIAL         How are you?
```

The parser is intentionally bounded and inspectable. Unsupported grammar becomes an unknown frame or a context probe; it is not silently guessed into a fact.

## Coreference and the pronoun trap

Entities carry kind, gender, grammatical number, relation to the user, aliases, salience, and last-mentioned turn. A gendered pronoun must have exactly one compatible antecedent.

```text
She pissed me off again.
-> Who do you mean by she?
```

With two compatible antecedents:

```text
My sister saw my mother.
She left.
-> Do you mean your sister or your mother?
```

No unresolved statement is written into factual memory.

## Contextual gating and collision masking

The gating layer derives register and severity from both lexical/structural parsing and Clanker's VADUGWI reading. It then locks incompatible construction pools.

```text
My tummy hurts bruh.
low severity + casual
-> formal, clinical, and high-drama pools locked

Bruh, my mom is really sick.
casual delivery + high-severity familial content
-> collision_masking=True
-> humor, playfulness, minimization, and performative empathy locked
```

The second case acknowledges the severity without breaking the user's protective conversational distance.

## Construction graph

Supportive and social replies are selected from `clanker_lm/data/constructions.json`. The graph is not a random phrase list. Each construction declares:

- response act;
- required slots;
- compatible registers;
- compatible severity bands;
- semantic/contextual tags.

Traversal removes nodes that cannot satisfy the response plan before affective scoring begins.

Factual answers are generated directly from the verified semantic frame, then reparsed by a hard semantic validator before Clanker ranks them.

## Running it

One message:

```bash
python -m clanker_lm --once "My sister pissed me off again." --trace --strict-clanker
```

Persistent interactive session:

```bash
python -m clanker_lm \
  --db .clanker-lm.sqlite3 \
  --session demo \
  --strict-clanker \
  --trace
```

Python API:

```python
from clanker_lm import ClankerLM

with ClankerLM(
    session_id="demo",
    db_path=".clanker-lm.sqlite3",
    strict_clanker=True,
) as dialogue:
    print(dialogue.reply("My sister bought a used Honda yesterday."))
    print(dialogue.reply("What did she buy?"))
    print(dialogue.reply("Why did she buy it?"))
```

Expected dialogue:

```text
Got it.
She bought a used Honda yesterday.
You told me she bought a used Honda yesterday, but not why they chose to do it.
```

## Traceability

`ClankerLM.process()` returns a `TurnResult`, not only a string. `TurnResult.trace_dict()` includes:

- parsed speech act and frame;
- input and response VADUGWI vectors;
- resulting session state;
- register/severity/collision gates;
- answer status, role binding, certainty, and provenance;
- every candidate's score, semantic validity, rejection state, and selection state;
- facts written during the turn.

## Test coverage

The Clanker-LM tests exercise:

- statement predicate/argument extraction;
- subject and object WH questions;
- polar positive and negative propositions;
- cause/motive/purpose/justification/evidence routing;
- method/process/manner/degree/quantity routing;
- missing-role and open-world unknown behavior;
- contradiction and multiple-binding ambiguity;
- pronoun absence and pronoun ambiguity;
- normal versus Yoda-style word order;
- quantity realization and double-object transfer frames;
- contextual gate locking and collision masking;
- hard semantic candidate rejection;
- deterministic repeated runs;
- SQLite restart persistence and session isolation;
- integration with the real Clanker VADUGWI engine.

Run the project suite:

```bash
python -m pytest engine/tests/ -q
```

## Current boundary

This is a complete deterministic conversational-Q&A vertical slice, not a claim to cover unrestricted English or open-world knowledge. It currently answers from explicitly stored conversation facts or facts injected through `ClankerLM.add_fact()`. Additional domains plug into the same seam:

```text
QuestionFrame -> evidence adapter -> AnswerContract
```

A document retriever, API, calculator, rules engine, or knowledge graph can provide facts without changing the parser, semantic hard gate, Clanker state transition, or surface realizer.
