"""Existing answer selection policy data; no learned or private session state."""

from __future__ import annotations
from typing import Dict, Tuple
from ..model import HowKind, QuestionKind, WhyKind

class PolicyComponent:
    """Declared selection tables, preserved without tuning."""

    WHY_FALLBACKS: Dict[WhyKind, Tuple[str, ...]] = {
        WhyKind.CAUSE: ("cause", "motive", "purpose", "justification", "evidence"),
        WhyKind.MOTIVE: ("motive", "cause", "purpose"),
        WhyKind.PURPOSE: ("purpose", "motive", "cause"),
        WhyKind.JUSTIFICATION: ("justification", "cause", "motive"),
        WhyKind.EVIDENCE: ("evidence", "cause", "justification"),
        WhyKind.UNKNOWN: ("cause", "motive", "purpose", "justification", "evidence"),
    }

    HOW_FALLBACKS: Dict[HowKind, Tuple[str, ...]] = {
        HowKind.METHOD: ("method", "manner", "process", "mechanism"),
        HowKind.MANNER: ("manner", "method"),
        HowKind.MECHANISM: ("mechanism", "cause", "method", "process"),
        HowKind.PROCESS: ("process", "method", "mechanism"),
        HowKind.DEGREE: ("value", "quantity"),
        HowKind.QUANTITY: ("quantity",),
        HowKind.STATE: ("value", "state"),
        HowKind.UNKNOWN: ("method", "manner", "process", "mechanism", "value"),
    }

    QUESTION_ROLE_FALLBACKS: Dict[QuestionKind, Dict[str, Tuple[str, ...]]] = {
        QuestionKind.WHERE: {
            "location": ("location", "destination", "goal"),
            "destination": ("destination", "location", "goal"),
            "source": ("source",),
        },
        QuestionKind.WHEN: {"time": ("time",)},
        QuestionKind.WHOSE: {"possessor": ("possessor", "owner")},
    }

    EMBEDDED_INTERROGATIVE_QUERY_PREDICATES = {
        "ask", "wonder", "know", "remember", "discover", "determine", "tell",
    }

    EPISTEMIC_SELF_QUERY_PREDICATES = {"know", "remember"}

    CONTENT_QUERY_PREDICATES = {
        "say", "tell", "report", "claim",
        "think", "believe", "know",
        "notice", "hear",
    }

    INFINITIVAL_QUERY_PREDICATES = {
        "plan", "intend", "hope", "want", "tell", "ask", "seem", "appear",
    }

    GERUND_QUERY_PREDICATES = {
        "enjoy", "avoid", "start", "begin", "stop", "keep", "continue",
        "see", "watch", "hear", "notice",
    }
