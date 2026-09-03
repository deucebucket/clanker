"""Evidence binding and deterministic answer realization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .memory import ConversationMemory
from .models import (
    AnswerContract,
    AnswerStatus,
    EntityKind,
    Fact,
    Provenance,
    QuestionFamily,
    QuestionFrame,
    ResponseCandidate,
    RoleValue,
    SemanticRole,
    ValueKind,
)
from .normalize import MOTION_VERBS, TRANSFER_VERBS, normalize_alias, past_tense, third_person


class AnswerEngine:
    """Bind a question's typed hole against facts with explicit failure states."""

    def __init__(self, memory: ConversationMemory) -> None:
        self.memory = memory

    def answer(self, question: QuestionFrame) -> AnswerContract:
        if question.unresolved_reference is not None:
            return AnswerContract(
                status=AnswerStatus.NEEDS_CONTEXT,
                question=question,
                certainty=255,
                explanation=question.unresolved_reference.reason,
            )
        if question.rhetorical:
            return AnswerContract(
                status=AnswerStatus.RHETORICAL,
                question=question,
                certainty=220,
                explanation="syntactic question classified as a pragmatic disclosure",
            )
        if question.family == QuestionFamily.SOCIAL:
            return AnswerContract(
                status=AnswerStatus.ANSWERED,
                question=question,
                certainty=255,
                provenance=Provenance.VERIFIED,
                explanation="conventional social check-in",
            )

        matches = self.memory.query(question)
        if question.family == QuestionFamily.POLAR:
            return self._answer_polar(question, matches)
        return self._answer_open_slot(question, matches)

    def _answer_polar(
        self, question: QuestionFrame, matches: Sequence[Fact]
    ) -> AnswerContract:
        agreeing = [
            fact for fact in matches if fact.frame.polarity == question.frame.polarity
        ]
        opposing = [
            fact for fact in matches if fact.frame.polarity != question.frame.polarity
        ]
        if agreeing and opposing:
            return AnswerContract(
                status=AnswerStatus.AMBIGUOUS,
                question=question,
                matching_facts=tuple(matches),
                certainty=min(fact.certainty for fact in matches),
                explanation="both positive and negative facts match the proposition",
            )
        if agreeing:
            selected = agreeing[0]
            return AnswerContract(
                status=AnswerStatus.TRUE,
                question=question,
                matching_facts=tuple(matches),
                selected_fact=selected,
                truth_value=True,
                certainty=selected.certainty,
                provenance=selected.provenance,
                explanation="a stored fact entails the proposition as asked",
            )
        if opposing:
            selected = opposing[0]
            return AnswerContract(
                status=AnswerStatus.FALSE,
                question=question,
                matching_facts=tuple(matches),
                selected_fact=selected,
                truth_value=False,
                certainty=selected.certainty,
                provenance=selected.provenance,
                explanation="a stored fact explicitly contradicts the proposition",
            )
        return AnswerContract(
            status=AnswerStatus.UNKNOWN,
            question=question,
            matching_facts=(),
            certainty=0,
            explanation="no stored fact entails or contradicts the proposition",
        )

    def _answer_open_slot(
        self, question: QuestionFrame, matches: Sequence[Fact]
    ) -> AnswerContract:
        if not matches:
            return AnswerContract(
                status=AnswerStatus.UNKNOWN,
                question=question,
                certainty=0,
                explanation="no fact matches the known portion of the question",
            )

        bindings: List[tuple[SemanticRole, RoleValue, Fact]] = []
        for requested_role in question.requested_roles:
            for fact in matches:
                value = fact.frame.roles.get(requested_role)
                if value is not None:
                    bindings.append((requested_role, value, fact))
            if bindings:
                # Requested roles are ordered by semantic preference.  For WHY,
                # motive outranks purpose, which outranks raw cause.
                break

        if not bindings:
            return AnswerContract(
                status=AnswerStatus.PARTIAL_UNKNOWN,
                question=question,
                matching_facts=tuple(matches),
                selected_fact=matches[0],
                certainty=matches[0].certainty,
                provenance=matches[0].provenance,
                explanation="the event is known but the requested role is unbound",
            )

        unique: List[tuple[SemanticRole, RoleValue, Fact]] = []
        for binding in bindings:
            if not any(
                binding[0] == existing[0]
                and self.memory.role_values_match(binding[1], existing[1])
                for existing in unique
            ):
                unique.append(binding)
        if len(unique) > 1:
            return AnswerContract(
                status=AnswerStatus.AMBIGUOUS,
                question=question,
                matching_facts=tuple(matches),
                certainty=min(item[2].certainty for item in unique),
                explanation="multiple distinct values bind the requested role",
            )

        role, value, fact = unique[0]
        return AnswerContract(
            status=AnswerStatus.ANSWERED,
            question=question,
            matching_facts=tuple(matches),
            selected_fact=fact,
            bound_role=role,
            bound_value=value,
            certainty=fact.certainty,
            provenance=fact.provenance,
            explanation="the requested role was bound from a matching fact",
        )


class SurfaceRealizer:
    """Turn a verified proposition into grammatical declarative English."""

    def __init__(self, memory: ConversationMemory) -> None:
        self.memory = memory

    def fact_sentence(
        self,
        fact: Fact,
        *,
        focus: Optional[SemanticRole] = None,
        prefer_subject_pronoun: bool = True,
        prefer_object_pronoun: bool = False,
    ) -> str:
        frame = fact.frame
        agent = frame.roles.get(SemanticRole.AGENT)
        if agent is None:
            return self._finish(frame.surface or "That happened")
        subject = self._entity_phrase(
            agent,
            subject=True,
            prefer_pronoun=(prefer_subject_pronoun and focus != SemanticRole.AGENT),
            definite=False,
        )

        if frame.predicate == "be":
            predicate = self._realize_copula(
                subject,
                agent,
                frame.tense,
                frame.polarity,
            )
            complement = self._copular_complement(frame, focus=focus)
            return self._finish(f"{subject} {predicate}{(' ' + complement) if complement else ''}")

        verb, particle = self._realize_verb(
            frame.predicate,
            frame.tense,
            frame.polarity,
            agent,
        )
        pieces = [subject, verb]

        object_role = (
            SemanticRole.PATIENT
            if SemanticRole.PATIENT in frame.roles
            else SemanticRole.THEME
        )
        obj = frame.roles.get(object_role)
        if obj is not None:
            quantity = frame.roles.get(SemanticRole.QUANTITY)
            if quantity is not None:
                pieces.append(quantity.display)
            if quantity is not None and obj.kind == ValueKind.ENTITY:
                entity = self.memory.entities.get(obj.value)
                object_text = entity.display_name if entity is not None else obj.display
            else:
                object_text = self._entity_phrase(
                    obj,
                    subject=False,
                    prefer_pronoun=(
                        prefer_object_pronoun
                        and focus != object_role
                        and quantity is None
                    ),
                    definite=(focus != object_role and quantity is None),
                )
            pieces.append(object_text)
        if particle:
            pieces.append(particle)

        recipient = frame.roles.get(SemanticRole.RECIPIENT)
        if recipient is not None:
            pieces.extend(
                [
                    "to",
                    self._entity_phrase(
                        recipient,
                        subject=False,
                        prefer_pronoun=False,
                        definite=True,
                    ),
                ]
            )

        location = frame.roles.get(SemanticRole.LOCATION)
        if location is not None:
            location_text = self._value_text(location, definite=True)
            if frame.predicate == "enter":
                pieces.append(location_text)
            elif frame.predicate in MOTION_VERBS:
                pieces.extend(["to", location_text])
            else:
                pieces.extend(["at", location_text])

        method = frame.roles.get(SemanticRole.METHOD)
        if method is not None:
            method_text = method.display.strip()
            if method_text.lower().split()[0] not in {"by", "with", "using", "via", "through"}:
                method_text = f"by {method_text}"
            pieces.append(method_text)

        purpose = frame.roles.get(SemanticRole.PURPOSE)
        motive = frame.roles.get(SemanticRole.MOTIVE)
        cause = frame.roles.get(SemanticRole.CAUSE)
        justification = frame.roles.get(SemanticRole.JUSTIFICATION)
        if motive is not None:
            pieces.append(f"because {motive.display}")
        elif purpose is not None:
            purpose_text = purpose.display.strip()
            if purpose_text.lower().startswith("to "):
                pieces.append(purpose_text)
            else:
                pieces.append(f"to {purpose_text}")
        elif cause is not None:
            pieces.append(f"because {cause.display}")
        elif justification is not None:
            pieces.append(f"because {justification.display}")

        time = frame.roles.get(SemanticRole.TIME)
        if time is not None:
            pieces.append(time.display)
        if frame.repeated and not any(
            normalize_alias(piece) in {"again", "still", "always"} for piece in pieces
        ):
            pieces.append("again")
        return self._finish(" ".join(piece for piece in pieces if piece))

    def proposition_fragment(self, frame) -> str:
        dummy = Fact(fact_id="question", frame=frame, certainty=0)
        sentence = self.fact_sentence(
            dummy,
            prefer_subject_pronoun=True,
            prefer_object_pronoun=True,
        )
        return sentence[:-1].lower() if sentence.endswith(".") else sentence.lower()

    def _realize_copula(
        self, subject: str, agent: RoleValue, tense: str, polarity: bool
    ) -> str:
        plural = self._is_plural(agent)
        lower = subject.lower()
        if tense == "past":
            verb = "were" if plural or lower == "you" else "was"
        else:
            if lower == "i":
                verb = "am"
            elif plural or lower == "you":
                verb = "are"
            else:
                verb = "is"
        return verb if polarity else f"{verb} not"

    def _realize_verb(
        self,
        predicate: str,
        tense: str,
        polarity: bool,
        agent: RoleValue,
    ) -> tuple[str, Optional[str]]:
        particle: Optional[str] = None
        base = predicate
        if "_" in predicate:
            base, particle = predicate.split("_", 1)
        if tense == "future":
            return ((f"will {base}" if polarity else f"will not {base}"), particle)
        if tense == "past":
            return ((past_tense(base) if polarity else f"did not {base}"), particle)
        plural = self._is_plural(agent)
        entity = self.memory.entities.get(agent.value) if agent.kind == ValueKind.ENTITY else None
        first_or_second = agent.value in {"user", "assistant"}
        if polarity:
            return ((base if plural or first_or_second else third_person(base)), particle)
        auxiliary = "do" if plural or first_or_second else "does"
        return (f"{auxiliary} not {base}", particle)

    def _copular_complement(
        self, frame, *, focus: Optional[SemanticRole]
    ) -> str:
        for role in (
            SemanticRole.ATTRIBUTE,
            SemanticRole.VALUE,
            SemanticRole.LOCATION,
            SemanticRole.TIME,
        ):
            value = frame.roles.get(role)
            if value is None:
                continue
            if role == SemanticRole.LOCATION:
                return f"at {self._value_text(value, definite=True)}"
            return value.display
        return ""

    def _entity_phrase(
        self,
        value: RoleValue,
        *,
        subject: bool,
        prefer_pronoun: bool,
        definite: bool,
    ) -> str:
        if value.kind != ValueKind.ENTITY:
            return value.display
        return self.memory.describe_entity(
            value.value,
            subject=subject,
            prefer_pronoun=prefer_pronoun,
            definite=definite,
        )

    def _value_text(self, value: RoleValue, *, definite: bool) -> str:
        if value.kind == ValueKind.ENTITY:
            return self.memory.describe_entity(
                value.value,
                subject=False,
                prefer_pronoun=False,
                definite=definite,
            )
        return value.display

    def _is_plural(self, value: RoleValue) -> bool:
        if value.kind != ValueKind.ENTITY:
            return False
        entity = self.memory.entities.get(value.value)
        return bool(entity and entity.number.value == "plural")

    @staticmethod
    def _finish(text: str) -> str:
        text = " ".join(text.split()).strip()
        if not text:
            return "I don't know."
        return text[0].upper() + text[1:] + ("" if text[-1] in ".!?" else ".")


class AnswerRealizer:
    """Generate only proposition-preserving response candidates."""

    def __init__(self, memory: ConversationMemory) -> None:
        self.memory = memory
        self.surface = SurfaceRealizer(memory)

    def candidates(self, contract: AnswerContract) -> Tuple[ResponseCandidate, ...]:
        status = contract.status
        if status == AnswerStatus.NEEDS_CONTEXT:
            return self._context_probe_candidates(contract)
        if status == AnswerStatus.RHETORICAL:
            return (
                ResponseCandidate(
                    "rhetorical-support-1",
                    "It sounds like that question is really about how you see yourself. What happened?",
                    ("support", "inquiry", "no_claim"),
                    "rhetorical",
                ),
                ResponseCandidate(
                    "rhetorical-support-2",
                    "That sounds painful, not like a question with a factual answer. What brought this up?",
                    ("support", "inquiry", "no_claim"),
                    "rhetorical",
                ),
            )
        if contract.question.family == QuestionFamily.SOCIAL:
            return (
                ResponseCandidate(
                    "social-checkin-1",
                    "I'm here and ready to help. How are you doing?",
                    ("social", "inquiry"),
                    "social",
                ),
                ResponseCandidate(
                    "social-checkin-2",
                    "I'm doing all right and paying attention. How are you?",
                    ("social", "inquiry"),
                    "social",
                ),
            )
        if status == AnswerStatus.ANSWERED:
            return self._answered_candidates(contract)
        if status in {AnswerStatus.TRUE, AnswerStatus.FALSE}:
            return self._polar_candidates(contract)
        if status == AnswerStatus.PARTIAL_UNKNOWN:
            return self._partial_unknown_candidates(contract)
        if status == AnswerStatus.AMBIGUOUS:
            return (
                ResponseCandidate(
                    "ambiguous-1",
                    "I found more than one matching answer, so I need a little more detail.",
                    ("clarify", "uncertain", "no_claim"),
                    "ambiguous",
                ),
                ResponseCandidate(
                    "ambiguous-2",
                    "There is more than one possible answer in the stored facts. Which event do you mean?",
                    ("clarify", "uncertain", "no_claim"),
                    "ambiguous",
                ),
            )
        return self._unknown_candidates(contract)

    def _answered_candidates(
        self, contract: AnswerContract
    ) -> Tuple[ResponseCandidate, ...]:
        fact = contract.selected_fact
        role = contract.bound_role
        value = contract.bound_value
        if fact is None or role is None or value is None:
            return self._unknown_candidates(contract)
        full = self.surface.fact_sentence(
            fact,
            focus=role,
            prefer_subject_pronoun=True,
            prefer_object_pronoun=role not in {SemanticRole.THEME, SemanticRole.PATIENT},
        )
        compact = self._compact_binding(role, value)
        candidates = [
            ResponseCandidate(
                f"fact-full-{fact.fact_id}-{role.value}",
                full,
                ("factual", "direct", "supported"),
                f"{fact.fact_id}:{role.value}:{value.value}",
            )
        ]
        if compact and normalize_alias(compact) != normalize_alias(full):
            candidates.append(
                ResponseCandidate(
                    f"fact-compact-{fact.fact_id}-{role.value}",
                    compact,
                    ("factual", "concise", "supported"),
                    f"{fact.fact_id}:{role.value}:{value.value}",
                )
            )
        return tuple(candidates)

    def _compact_binding(self, role: SemanticRole, value: RoleValue) -> str:
        if value.kind == ValueKind.ENTITY:
            text = self.memory.describe_entity(
                value.value,
                subject=role == SemanticRole.AGENT,
                prefer_pronoun=False,
                definite=False,
            )
        else:
            text = value.display
        if role in {
            SemanticRole.CAUSE,
            SemanticRole.MOTIVE,
            SemanticRole.JUSTIFICATION,
            SemanticRole.EVIDENCE,
        }:
            return self.surface._finish(f"Because {text}")
        if role == SemanticRole.PURPOSE:
            return self.surface._finish(text if text.lower().startswith("to ") else f"To {text}")
        return self.surface._finish(text)

    def _polar_candidates(self, contract: AnswerContract) -> Tuple[ResponseCandidate, ...]:
        fact = contract.selected_fact
        if fact is None:
            return self._unknown_candidates(contract)
        yes = contract.status == AnswerStatus.TRUE
        fact_text = self.surface.fact_sentence(
            fact,
            prefer_subject_pronoun=True,
            prefer_object_pronoun=True,
        )
        prefix = "Yes" if yes else "No"
        direct = f"{prefix}. {fact_text}"
        concise = f"{prefix}."
        signature = f"polar:{fact.fact_id}:{str(yes).lower()}"
        return (
            ResponseCandidate(
                f"polar-full-{fact.fact_id}",
                direct,
                ("factual", "direct", "supported", "polar"),
                signature,
            ),
            ResponseCandidate(
                f"polar-compact-{fact.fact_id}",
                concise,
                ("factual", "concise", "supported", "polar"),
                signature,
            ),
        )

    def _partial_unknown_candidates(
        self, contract: AnswerContract
    ) -> Tuple[ResponseCandidate, ...]:
        fact = contract.selected_fact
        requested = contract.question.requested_roles[0] if contract.question.requested_roles else None
        label = self._role_question_word(requested)
        if fact is None:
            return self._unknown_candidates(contract)
        known = self.surface.fact_sentence(
            fact,
            prefer_subject_pronoun=True,
            prefer_object_pronoun=True,
        ).rstrip(".")
        return (
            ResponseCandidate(
                "partial-unknown-1",
                f"You told me {known[0].lower() + known[1:]}, but not {label}.",
                ("factual", "uncertain", "no_invention"),
                f"partial:{fact.fact_id}:{label}",
            ),
            ResponseCandidate(
                "partial-unknown-2",
                f"I know that {known[0].lower() + known[1:]}. I don't know {label}.",
                ("factual", "uncertain", "no_invention"),
                f"partial:{fact.fact_id}:{label}",
            ),
        )

    def _unknown_candidates(self, contract: AnswerContract) -> Tuple[ResponseCandidate, ...]:
        question = contract.question
        if question.family == QuestionFamily.POLAR:
            proposition = self.surface.proposition_fragment(question.frame)
            text = f"I don't have enough information to say whether {proposition}."
        elif question.requested_roles:
            label = self._role_question_word(question.requested_roles[0])
            text = f"I don't have a stored fact that tells me {label}."
        else:
            text = "I don't have enough information to answer that."
        return (
            ResponseCandidate(
                "unknown-1",
                text,
                ("uncertain", "no_claim", "no_invention"),
                "unknown",
            ),
            ResponseCandidate(
                "unknown-2",
                "I don't know from the information currently stored.",
                ("uncertain", "no_claim", "no_invention", "concise"),
                "unknown",
            ),
        )

    def _context_probe_candidates(
        self, contract: AnswerContract
    ) -> Tuple[ResponseCandidate, ...]:
        unresolved = contract.question.unresolved_reference
        assert unresolved is not None
        surface = unresolved.surface.lower()
        if unresolved.compatible_entity_ids:
            descriptions = [
                self.memory.describe_entity(entity_id, prefer_pronoun=False)
                for entity_id in unresolved.compatible_entity_ids[:3]
                if entity_id in self.memory.entities
            ]
            if len(descriptions) >= 2:
                options = " or ".join(descriptions)
                text = f"Do you mean {options}?"
            else:
                text = f"Which {surface} do you mean?"
        elif surface in {"it", "this", "that"}:
            text = f"What do you mean by {surface}?"
        else:
            text = f"Who do you mean by {surface}?"
        return (
            ResponseCandidate(
                "context-probe-1",
                text,
                ("clarify", "context_probe", "no_claim"),
                f"probe:{surface}",
            ),
        )

    @staticmethod
    def _role_question_word(role: Optional[SemanticRole]) -> str:
        mapping = {
            SemanticRole.AGENT: "who did it",
            SemanticRole.PATIENT: "who or what was affected",
            SemanticRole.THEME: "what it was",
            SemanticRole.RECIPIENT: "who received it",
            SemanticRole.LOCATION: "where it happened",
            SemanticRole.TIME: "when it happened",
            SemanticRole.CAUSE: "why it happened",
            SemanticRole.MOTIVE: "why they chose to do it",
            SemanticRole.PURPOSE: "what it was meant to accomplish",
            SemanticRole.JUSTIFICATION: "what justifies it",
            SemanticRole.EVIDENCE: "what evidence supports it",
            SemanticRole.METHOD: "how it was done",
            SemanticRole.MANNER: "how it happened",
            SemanticRole.PROCESS: "how the process works",
            SemanticRole.MECHANISM: "what mechanism makes it work",
            SemanticRole.DEGREE: "the requested degree",
            SemanticRole.QUANTITY: "the requested quantity",
            SemanticRole.ATTRIBUTE: "what its state or attribute is",
            SemanticRole.VALUE: "what it is",
        }
        return mapping.get(role, "the missing information")


class SemanticValidator:
    """Hard semantic gate applied before affective candidate scoring.

    Affect may rank semantically valid alternatives, but it is never allowed to
    rescue a candidate that invents a binding, hides uncertainty, or reverses
    the truth value of a polar answer.
    """

    _UNCERTAINTY_MARKERS = (
        "don't", "do not", "not enough", "enough information", "unknown",
        "not stored", "currently stored", "need more", "cannot tell",
    )
    _PARTIAL_MARKERS = (
        "but not", "don't know", "do not know", "not told", "not provided",
        "missing", "haven't said", "have not said",
    )
    _AMBIGUITY_MARKERS = (
        "more than one", "multiple", "conflict", "ambiguous", "which ",
        "two possible",
    )
    _PROBE_MARKERS = (
        "who do you mean", "what do you mean", "which ", "do you mean",
    )

    def validate(
        self, candidate: ResponseCandidate, contract: Optional[AnswerContract]
    ) -> tuple[bool, Tuple[str, ...]]:
        if contract is None:
            return True, ("non-factual response plan",)

        status = contract.status
        text = candidate.text.lower().strip()
        normalized = normalize_alias(text)
        reasons: List[str] = []

        if status == AnswerStatus.TRUE and not text.startswith("yes"):
            reasons.append("true polar answer must begin with yes")
        if status == AnswerStatus.FALSE and not text.startswith("no"):
            reasons.append("false polar answer must begin with no")

        if status == AnswerStatus.UNKNOWN and not any(
            marker in text for marker in self._UNCERTAINTY_MARKERS
        ):
            reasons.append("unknown answer lacks an uncertainty marker")
        if status == AnswerStatus.PARTIAL_UNKNOWN and not any(
            marker in text for marker in self._PARTIAL_MARKERS
        ):
            reasons.append("partially known answer asserts the missing role")
        if status == AnswerStatus.AMBIGUOUS and not any(
            marker in text for marker in self._AMBIGUITY_MARKERS
        ):
            reasons.append("ambiguous answer does not expose the ambiguity")
        if status == AnswerStatus.NEEDS_CONTEXT:
            if not text.endswith("?") or not any(
                marker in text for marker in self._PROBE_MARKERS
            ):
                reasons.append("unresolved reference requires an explicit context probe")

        if status == AnswerStatus.ANSWERED and contract.bound_value is not None:
            value = contract.bound_value
            signature = self._entity_signature(value)
            if signature and signature not in normalized:
                reasons.append("candidate omits the verified bound value")

        # A full polar candidate may name the evidence; a concise yes/no is also
        # valid.  It may not, however, state the opposite prefix.
        if status == AnswerStatus.TRUE and text.startswith("no"):
            reasons.append("candidate reverses true proposition")
        if status == AnswerStatus.FALSE and text.startswith("yes"):
            reasons.append("candidate reverses false proposition")

        return (not reasons), tuple(reasons or ["semantic contract preserved"])

    @staticmethod
    def _entity_signature(value: RoleValue) -> str:
        return normalize_alias(value.display)

