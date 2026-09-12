"""Events responsibilities; extracted without changing decision rules."""

from __future__ import annotations
import re
from typing import Dict, Iterable, Optional, Tuple
from .. import lexicon
from ..model import EntityKind, EventFrame, GrammaticalNumber, RefKind, SemanticRef

class EventsComponent:
    """Events behavior composed by the stable public SurfaceRealizer API."""

    def render_event(
        self,
        event: EventFrame,
        *,
        polarity: Optional[bool] = None,
        omit_roles: Iterable[str] = (),
        omit_variables: bool = True,
        capitalize: bool = False,
    ) -> str:
        args = {
            role: value
            for role, value in event.arguments.items()
            if role not in set(omit_roles) and not (omit_variables and value.is_variable)
        }
        positive = event.polarity if polarity is None else polarity

        if event.predicate == "attribute":
            subject = args.get("subject")
            value = args.get("value")
            attribute = args.get("attribute")
            if subject and value:
                text = f"{self.render_ref(subject, case='subject')} {self._be_form(event, subject, positive)} {self.render_ref(value)}"
            elif subject and attribute:
                text = f"{self.render_ref(subject, case='subject')} has a known {self.render_ref(attribute)}"
            else:
                text = "the attribute is known"
            return self._finish_clause(text, capitalize)

        subject_role, subject_ref = self._select_subject(event, args)
        if not subject_ref:
            return self._finish_clause(event.raw_text.strip().rstrip(".?!") or event.predicate, capitalize)
        subject_text = self.render_ref(subject_ref, case="subject", definite=True)

        if event.predicate == "be":
            complement = self._render_copular_complement(args)
            verb = (
                self._verb_form("be", event, subject_ref, positive)
                if event.aspect in {"progressive", "perfect_progressive"}
                else self._be_form(event, subject_ref, positive)
            )
            text = f"{subject_text} {verb} {complement}".strip()
            return self._finish_clause(text, capitalize)

        if event.predicate == "own":
            patient = args.get("patient")
            verb = self._verb_form("have", event, subject_ref, positive)
            # Body parts and ordinary possessed objects retain their natural
            # determiner; previously mentioned standalone things are definite.
            definite = bool(patient and self._should_be_definite(patient))
            text = f"{subject_text} {verb} {self.render_ref(patient, definite=definite) if patient else ''}".strip()
            return self._finish_clause(text, capitalize)

        verb = self._verb_form(event.predicate, event, subject_ref, positive)
        pieces = [subject_text, verb]

        # Passive event: patient selected as subject and a separate agent exists.
        if subject_role == "patient" and "agent" in args and event.predicate not in lexicon.UNACCUSATIVE_VERBS:
            aux = self._be_form(event, subject_ref, positive=True)
            participle = lexicon.participle_form(event.predicate)
            if not positive:
                aux = self._negative_be(event, subject_ref)
            pieces = [subject_text, aux, participle, "by", self.render_ref(args["agent"], case="object")]
        else:
            patient = args.get("patient") or args.get("state")
            if patient and patient != subject_ref:
                if args.get("quantity") and patient.kind == RefKind.ENTITY:
                    quantity_text = self.render_ref(args["quantity"])
                    entity = self.memory.get_entity(patient.key)
                    patient_text = entity.canonical_name if entity else self.render_ref(patient, case="object", definite=False)
                    patient_text = re.sub(r"^(?:a|an|the)\s+", "", patient_text, flags=re.IGNORECASE)
                    # A parser may preserve the original surface (for example
                    # ``three cars``) while also storing a typed quantity slot.
                    # Realization uses the entity's canonical head so the
                    # quantity is expressed exactly once.
                    patient_text = re.sub(
                        r"^(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|\d+(?:\.\d+)?)\s+)+",
                        "",
                        patient_text,
                        flags=re.IGNORECASE,
                    )
                    pieces.append(f"{quantity_text} {patient_text}")
                else:
                    pieces.append(self.render_ref(patient, case="object", definite=self._should_be_definite(patient)))
            recipient = args.get("recipient")
            if recipient:
                pieces.extend(["to", self.render_ref(recipient, case="object", definite=True)])

        adjunct_order = [
            ("destination", "to"),
            ("source", "from"),
            ("location", "at"),
            ("time", ""),
            ("method", "by"),
            ("manner", ""),
            ("purpose", "to"),
            ("cause", "because"),
            ("motive", "because"),
            ("justification", "because"),
            ("evidence", "because"),
            ("topic", "about"),
        ]
        used_reason = False
        for role, prefix in adjunct_order:
            value = args.get(role)
            if not value:
                continue
            if role in {"cause", "motive", "justification", "evidence"}:
                if used_reason:
                    continue
                used_reason = True
            phrase = self.render_ref(value, case="object", definite=role in {"location", "destination", "source"})
            stored_preposition = args.get(f"{role}_preposition")
            if stored_preposition:
                prefix = self.render_ref(stored_preposition)
            if role == "time":
                phrase = phrase if stored_preposition else self._render_time_phrase(phrase)
            if role == "purpose" and phrase.lower().startswith("to "):
                pieces.append(phrase)
            elif prefix:
                pieces.extend([prefix, phrase])
            else:
                pieces.append(phrase)

        quantity = args.get("quantity")
        if quantity and not args.get("patient"):
            pieces.append(self.render_ref(quantity))

        text = " ".join(piece for piece in pieces if piece).strip()
        return self._finish_clause(text, capitalize)

    def _select_subject(self, event: EventFrame, args: Dict[str, SemanticRef]) -> Tuple[str, Optional[SemanticRef]]:
        for role in ("agent", "subject", "experiencer", "possessor"):
            if role in args:
                return role, args[role]
        if "patient" in args:
            return "patient", args["patient"]
        return "", None

    def _render_copular_complement(self, args: Dict[str, SemanticRef]) -> str:
        if "location" in args:
            prep = self.render_ref(args.get("location_preposition")) or "at"
            return f"{prep} {self.render_ref(args['location'], definite=True)}"
        if "time" in args:
            prep = self.render_ref(args.get("time_preposition"))
            phrase = self.render_ref(args["time"])
            return f"{prep} {phrase}".strip() if prep else self._render_time_phrase(phrase)
        if "value" in args:
            return self.render_ref(args["value"], definite=False)
        if "state" in args:
            return self.render_ref(args["state"], definite=False)
        return ""

    def _verb_form(self, predicate: str, event: EventFrame, subject: SemanticRef, positive: bool) -> str:
        entity = self.memory.get_entity(subject.key) if subject.kind == RefKind.ENTITY else None
        plural = bool(entity and entity.number == GrammaticalNumber.PLURAL)
        first_person = subject.kind == RefKind.ENTITY and subject.key == "assistant"
        second_person = subject.kind == RefKind.ENTITY and subject.key == "user"
        third_singular = not plural and not first_person and not second_person

        if event.aspect == "perfect_progressive":
            gerund = lexicon.gerund_form(predicate)
            negation = "" if positive else " not"
            if event.tense == "future":
                modal = event.modality or "will"
                auxiliary = f"{modal}{negation} have been"
            elif event.modality:
                auxiliary = f"{event.modality}{negation} have been"
            elif event.tense == "past":
                auxiliary = f"had{negation} been"
            else:
                have = "has" if third_singular else "have"
                auxiliary = f"{have}{negation} been"
            return f"{auxiliary} {gerund}"

        if event.aspect == "progressive":
            gerund = lexicon.gerund_form(predicate)
            if event.modality:
                auxiliary = f"{event.modality}{'' if positive else ' not'} be"
            elif event.tense == "future":
                auxiliary = f"will{'' if positive else ' not'} be"
            else:
                auxiliary = (
                    self._be_form(event, subject, True)
                    if positive
                    else self._negative_be(event, subject)
                )
            return f"{auxiliary} {gerund}"

        if event.aspect == "perfect":
            participle = lexicon.participle_form(predicate)
            if event.tense == "future":
                auxiliary = "will have"
            elif event.tense == "past":
                auxiliary = "had"
            else:
                auxiliary = "has" if third_singular else "have"
            return f"{auxiliary}{'' if positive else ' not'} {participle}"

        if positive:
            if event.modality:
                return f"{event.modality} {predicate}"
            if event.tense == "past":
                return lexicon.past_form(predicate, plural_subject=plural)
            if event.tense == "future":
                return f"will {predicate}"
            return lexicon.present_form(
                predicate,
                third_person_singular=third_singular,
                plural_subject=plural,
                first_person=first_person,
            )

        if event.modality:
            return f"{event.modality} not {predicate}"
        if predicate == "be":
            return self._negative_be(event, subject)
        if event.tense == "past":
            return f"did not {predicate}"
        if event.tense == "future":
            return f"will not {predicate}"
        auxiliary = "does" if third_singular else "do"
        return f"{auxiliary} not {predicate}"

    def _be_form(self, event: EventFrame, subject: SemanticRef, positive: bool) -> str:
        if not positive:
            return self._negative_be(event, subject)
        entity = self.memory.get_entity(subject.key) if subject.kind == RefKind.ENTITY else None
        plural = bool(entity and entity.number == GrammaticalNumber.PLURAL)
        first_person = subject.kind == RefKind.ENTITY and subject.key == "assistant"
        second_person = subject.kind == RefKind.ENTITY and subject.key == "user"
        if event.tense == "past":
            return "were" if plural or second_person else "was"
        if event.tense == "future":
            return "will be"
        if first_person:
            return "am"
        if plural or second_person:
            return "are"
        return "is"

    def _negative_be(self, event: EventFrame, subject: SemanticRef) -> str:
        base = self._be_form(event.copy(polarity=True), subject, True)
        if base == "will be":
            return "will not be"
        return f"{base} not"

    @staticmethod
    def _render_time_phrase(phrase: str) -> str:
        lower = phrase.lower().strip()
        if not lower or lower.startswith(("on ", "at ", "in ", "before ", "after ", "during ", "since ", "until ")):
            return phrase
        first = lower.split()[0]
        if first in lexicon.DAYS or first in lexicon.MONTHS:
            return "on " + phrase
        if re.fullmatch(r"\d{1,2}(:\d{2})?(am|pm)?", lower.replace(" ", "")):
            return "at " + phrase
        return phrase

    def _predicate_tail_for_question(self, event: EventFrame) -> str:
        args = dict(event.arguments)
        args.pop("agent", None)
        args.pop("subject", None)
        temp = event.copy(arguments={"agent": SemanticRef.literal("someone", "someone", EntityKind.PERSON), **args})
        rendered = self.render_event(temp, capitalize=False)
        if rendered.startswith("someone "):
            return rendered[len("someone ") :]
        return rendered
