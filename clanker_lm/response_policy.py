"""Deterministic response-act planning for Clanker-LM.

This layer decides *what communicative operation is warranted* before surface
realization or VADUGWI candidate ranking.  It prevents a mathematically
attractive candidate from changing a neutral factual assertion into an
empathic intervention, while retaining hard severity and truth gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Set

from . import lexicon
from .model import AffectReading, AnswerStatus, ParseResult, SpeechAct


@dataclass(frozen=True)
class ResponsePolicyDecision:
    """Closed response-act decision consumed by contextual gating."""

    response_act: str
    severity: str
    register: str
    max_sentences: int
    rationale: Sequence[str] = field(default_factory=tuple)


class ResponseActPlanner:
    """Select a response act from semantics, evidence state, and affect.

    Surface wording is deliberately outside this class.  The planner emits a
    typed act and an inspectable rationale; the realizer may only generate
    candidates within that act class.
    """

    SEVERITY_TO_LEVEL = {"low": 0, "moderate": 1, "high": 2, "critical": 3}
    LEVEL_TO_SEVERITY = {value: key for key, value in SEVERITY_TO_LEVEL.items()}

    CRITICAL_STRUCTURES: Set[str] = {
        "FAREWELL",
        "METHOD_ACQUISITION",
        "SELF_REMOVAL",
        "NO_EXIT",
        "CRISIS",
    }

    LOSS_WORDS: Set[str] = {
        "died",
        "dead",
        "death",
        "funeral",
        "killed",
        "grief",
        "grieving",
        "mourning",
    }

    ACTIVE_DANGER_WORDS: Set[str] = {
        "bleeding",
        "danger",
        "dying",
        "emergency",
        "hospital",
        "injured",
        "missing",
        "overdose",
        "overdosed",
        "suicide",
        "suicidal",
        "unsafe",
    }

    SERIOUS_WORDS: Set[str] = {
        "cancer",
        "hospitalized",
        "hurt",
        "ill",
        "injured",
        "missing",
        "overdose",
        "sick",
    }

    NEGATIVE_WORDS: Set[str] = {
        "afraid",
        "angry",
        "awful",
        "bad",
        "betrayed",
        "cheated",
        "crying",
        "depressed",
        "disappointed",
        "failed",
        "fear",
        "frustrated",
        "hate",
        "hated",
        "helpless",
        "lonely",
        "mad",
        "miserable",
        "pissed",
        "sad",
        "scared",
        "terrible",
        "upset",
        "worried",
    }

    POSITIVE_WORDS: Set[str] = {
        "accepted",
        "awesome",
        "celebrate",
        "celebrating",
        "excited",
        "good",
        "graduated",
        "great",
        "happy",
        "love",
        "loved",
        "passed",
        "proud",
        "promoted",
        "promotion",
        "safe",
        "won",
    }

    LOSS_PREDICATES: Set[str] = {"die", "kill"}
    SERIOUS_PREDICATES: Set[str] = {
        "bleed",
        "hospitalize",
        "injure",
        "overdose",
    }
    NEGATIVE_PREDICATES: Set[str] = {
        "anger",
        "betray",
        "cry",
        "fail",
        "fear",
        "hate",
        "hurt",
        "lose",
        "upset",
        "worry",
    }
    POSITIVE_PREDICATES: Set[str] = {
        "celebrate",
        "graduate",
        "love",
        "pass",
        "promote",
        "win",
    }

    # These predicates describe ordinary propositions unless an explicit
    # affective marker, severe argument, critical structure, or polarity cue
    # says otherwise.  The list is intentionally conservative: an unknown
    # predicate does not automatically receive the neutral override.
    NEUTRAL_FACT_PREDICATES: Set[str] = {
        "arrive",
        "be",
        "build",
        "buy",
        "call",
        "close",
        "finish",
        "give",
        "go",
        "have",
        "leave",
        "live",
        "make",
        "meet",
        "move",
        "open",
        "own",
        "put",
        "read",
        "record",
        "say",
        "schedule",
        "sell",
        "send",
        "start",
        "take",
        "tell",
        "visit",
        "work",
        "write",
    }

    REFERENCE_STATUSES: Set[AnswerStatus] = {
        AnswerStatus.MISSING_REFERENCE,
        AnswerStatus.AMBIGUOUS_REFERENCE,
        AnswerStatus.MULTIPLE_MATCHES,
        AnswerStatus.LEXICAL_PROBE,
    }

    def plan(
        self,
        text: str,
        parse: ParseResult,
        affect: AffectReading,
        *,
        answer_status: Optional[AnswerStatus],
        initial_severity: str,
        initial_register: str,
    ) -> ResponsePolicyDecision:
        """Return one legal response-act class and its semantic severity."""

        rationale: List[str] = []
        severity_level = self.SEVERITY_TO_LEVEL.get(initial_severity, 0)
        tokens = lexicon.tokenize(text, include_punctuation=False)
        word_sequence = tuple(token.norm for token in tokens)
        words = set(word_sequence)
        predicates = {event.predicate.lower() for event in parse.events}
        structures = {str(item) for item in affect.structures}

        if answer_status in self.REFERENCE_STATUSES or parse.unresolved:
            rationale.append("unresolved or ambiguous information requires a probe")
            return self._decision("probe", severity_level, initial_register, 1, rationale)

        # A syntactic question owns the reply class.  Truth and evidence status
        # refine the answer, but cannot route a non-question assertion into the
        # answer path merely because its contract is UNKNOWN or UNSUPPORTED.
        if parse.speech_act == SpeechAct.ASK or parse.question is not None:
            if parse.question and parse.question.social_convention:
                rationale.append("conventional social question routes to social response")
                return self._decision("social", severity_level, initial_register, 1, rationale)
            rationale.append("question preserves answer-or-explicit-unknown contract")
            return self._decision("answer", severity_level, initial_register, 2, rationale)

        if parse.speech_act in {SpeechAct.GREET, SpeechAct.SOCIAL}:
            rationale.append("social convention routes to social response")
            return self._decision("social", severity_level, initial_register, 1, rationale)

        critical = bool(structures & self.CRITICAL_STRUCTURES) or severity_level >= 3
        not_safe = any(
            word_sequence[index : index + 2] == ("not", "safe")
            for index in range(max(0, len(word_sequence) - 1))
        )
        loss = bool(words & self.LOSS_WORDS) or bool(predicates & self.LOSS_PREDICATES)
        active_danger = not_safe or bool(words & self.ACTIVE_DANGER_WORDS) or bool(
            predicates & self.SERIOUS_PREDICATES
        )
        serious = bool(words & self.SERIOUS_WORDS)
        explicit_negative = bool(words & self.NEGATIVE_WORDS) or bool(
            predicates & self.NEGATIVE_PREDICATES
        )
        explicit_positive = bool(words & self.POSITIVE_WORDS) or bool(
            predicates & self.POSITIVE_PREDICATES
        )
        semantically_neutral = bool(predicates) and predicates.issubset(
            self.NEUTRAL_FACT_PREDICATES
        )

        if critical:
            rationale.append("critical structure or coordinate fixes a safety-check act")
            return self._decision("safety_probe", 3, initial_register, 1, rationale)

        # Completed loss is checked before the broader serious-content class so
        # the system can acknowledge grief without mechanically interrogating.
        if loss:
            rationale.append("completed loss calls for empathy without compulsory interrogation")
            return self._decision(
                "empathic_acknowledge",
                max(severity_level, 2),
                initial_register,
                1,
                rationale,
            )

        if active_danger or serious:
            rationale.append("explicit severe semantic content warrants one grounded follow-up")
            return self._decision(
                "serious_followup",
                max(severity_level, 2),
                initial_register,
                2,
                rationale,
            )

        if explicit_negative and not explicit_positive:
            rationale.append("explicit negative disclosure warrants validation and elaboration")
            return self._decision(
                "empathic_followup",
                max(severity_level, 1),
                initial_register,
                2,
                rationale,
            )

        if explicit_positive and not explicit_negative:
            rationale.append("explicit positive outcome warrants positive acknowledgment")
            return self._decision("positive_acknowledge", 0, "positive", 1, rationale)

        # Explicit lexical loss, danger, and affect remain available even when
        # the parser has not yet produced a complete event frame.  Only after
        # those high-value semantic cues are exhausted may an incomplete
        # non-question fall back to a minimal acknowledgment.
        if parse.speech_act != SpeechAct.ASSERT or not parse.events:
            rationale.append("incomplete non-question has no explicit affective cue")
            return self._decision("neutral_acknowledge", severity_level, initial_register, 1, rationale)

        # Literal neutral event semantics outrank an incidental moderate VADUGWI
        # boundary.  High/critical affect remains cautionary unless the event is
        # explicitly known to be an ordinary fact and no severe structure fired.
        if semantically_neutral and severity_level <= 1:
            rationale.append("closed assertion uses an ordinary factual predicate")
            if severity_level == 1:
                rationale.append("semantic neutrality resets incidental moderate affect")
            return self._decision("neutral_acknowledge", 0, initial_register, 1, rationale)

        if severity_level >= 2:
            rationale.append("high affect without a safe neutral override preserves caution")
            return self._decision("serious_followup", severity_level, initial_register, 2, rationale)

        if severity_level == 1:
            rationale.append("moderate affect on an unclassified predicate warrants gentle elaboration")
            return self._decision("empathic_followup", 1, initial_register, 2, rationale)

        rationale.append("closed assertion contains no explicit affective or severe semantics")
        return self._decision("neutral_acknowledge", 0, initial_register, 1, rationale)

    def _decision(
        self,
        response_act: str,
        severity_level: int,
        register: str,
        max_sentences: int,
        rationale: Iterable[str],
    ) -> ResponsePolicyDecision:
        level = max(0, min(3, int(severity_level)))
        return ResponsePolicyDecision(
            response_act=response_act,
            severity=self.LEVEL_TO_SEVERITY[level],
            register=register,
            max_sentences=max(1, int(max_sentences)),
            rationale=tuple(rationale),
        )
