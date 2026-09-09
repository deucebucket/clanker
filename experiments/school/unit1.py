"""Original authored development scenarios, never production response templates.

The world variables below supply expected roles independently of Clanker's
parser. Changing names is transfer stress, not independent source evidence.
"""
from .core import Check as C, Episode, Step as S, digest

WORLDS = (
    ("Sarah", "John", "sister", "she", "car", "bike", "yesterday"),
    ("Mary", "Daniel", "brother", "he", "camera", "phone", "today"),
    ("Anna", "Paul", "mother", "she", "book", "coat", "yesterday"),
    ("Jordan", "Alex", "father", "he", "bag", "radio", "today"),
    ("Morgan", "Casey", "sister", "she", "key", "book", "yesterday"),
    ("Taylor", "Robin", "brother", "he", "coat", "camera", "today"),
)


def unit_one(*, phase: str = "development") -> tuple[Episode, ...]:
    if phase not in {"development", "transfer"}:
        raise ValueError("phase must be development or transfer")
    worlds = WORLDS[:3] if phase == "development" else WORLDS[3:]
    episodes = []
    for a, b, relation, pronoun, obj, other, when in worlds:
        answer = (C("status", "answered"),)
        own_fact = f"My {relation} borrowed my {obj} {when}."
        fact = f"{a} borrowed a {obj} from {b} {when}."
        recipes = {
            "identity": (
                S(f"{a} called {b}."),
                S(f"Who called {b}?", answer + (C("role", a.lower(), "agent"),
                                                C("role", b.lower(), "patient"))),
            ),
            "perspective": (
                S(own_fact),
                S(f"What did my {relation} borrow?", answer + (C("owner", "user", "patient"),
                                                               C("perspective", obj))),
                S(f"When did {pronoun} borrow it?", answer + (C("role", when, "time"),
                                                             C("perspective", obj))),
            ),
            "time_attachment": (
                S(fact, (C("role", when, "time"),)),
                S(f"When did {a} borrow the {obj}?", answer + (C("role", when, "time"),)),
            ),
            "borrow_roles": (
                S(fact),
                S(f"Who did {a} borrow the {obj} from?", answer + (C("role", b.lower(), "source"),
                                                                  C("role", a.lower(), "agent"),
                                                                  C("role", obj, "patient"),
                                                                  C("requested_role", "source"),
                                                                  C("value", b.lower()))),
            ),
            "non_entailment": (
                S(fact),
                S(f"Does {a} own the {obj}?", (C("status", "unknown"),)),
            ),
            "event_continuity": (
                S(fact),
                S("The meeting is on Monday."),
                S(f"{b} called {a}."),
                S(f"{a} returned the {obj} today."),
                S(f"Who borrowed the {obj}?", answer + (C("role", a.lower(), "agent"),
                                                        C("same_entity", "borrowed_object", "patient"))),
                S(f"Who returned the {obj}?", answer + (C("role", a.lower(), "agent"),
                                                        C("same_entity", "borrowed_object", "patient"))),
            ),
            "correction": (
                S(own_fact),
                S(f"Actually, {pronoun} borrowed my {other}, not my {obj}."),
                S("The meeting is on Monday."),
                S(f"What did my {relation} borrow?", answer + (C("role", other, "patient"),
                                                              C("perspective", other))),
            ),
            "dialogue_obligations": (
                S(f"My {relation} pissed me off again."),
                S(own_fact),
                S(f"{pronoun.capitalize()} returned it late."),
                S("I was angry.", (C("no_question", "answered explanation"),)),
                S("Can you stop asking what happened?", (C("status", "acknowledged"),)),
                S("Thanks for listening.", (C("status", "acknowledged"), C("no_question", "gratitude"))),
                S("Bye.", (C("status", "acknowledged"), C("no_question", "closure"))),
            ),
            "appraisal_separation": (
                S(f"{a} is angry."),
                S("I am calm.", (C("appraisal", "user"), C("appraisal", a.lower()))),
            ),
        }
        for skill, steps in recipes.items():
            # Canonical inputs/expectations, NOT a mutable label, define context.
            context = digest([skill, [(s.text, [(c.kind, c.expected, c.role) for c in s.checks])
                                       for s in steps]])
            episodes.append(Episode(f"{phase}:{a.lower()}:{skill}", skill, context, steps))
    return tuple(episodes)
