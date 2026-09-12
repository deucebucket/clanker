"""Wh Questions recognition and transformation. Shared state is passed explicitly as memory."""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
from .. import lexicon
from ..memory import ConversationMemory
from ..model import EntityKind, EventFrame, HowKind, QuestionFrame, QuestionKind, SemanticRef, UnresolvedReference, WhyKind
from .types import NPResult, ClauseResult

class WhQuestionsRules:
    """Stateless wh questions transformations composed by SemanticParser."""

    def _parse_who(
        self,
        items: Sequence[lexicon.Token],
        raw: str,
        memory: ConversationMemory,
    ) -> Tuple[Optional[QuestionFrame], List[UnresolvedReference], List[str], List[str]]:
        rest = list(items[1:])
        if not rest:
            return None, [], [], ["who question lacks predicate"]

        if rest[0].norm in lexicon.AUXILIARIES:
            aux = rest[0]
            body = rest[1:]
            main_idx = self._find_main_verb(body, start=0)
            if main_idx < 0 and aux.norm in lexicon.AUX_DO and body:
                # ``Who did the work?`` uses *did* as the lexical predicate,
                # not do-support.  The WH term is its subject.
                declarative = [self._variable_token("agent"), aux] + body
                clause = self._parse_clause(declarative, raw, memory)
                requested_role = "agent"
                if not clause.event:
                    return None, clause.unresolved, clause.entities, clause.diagnostics + ["failed lexical-do who parse"]
                clause.event.arguments[requested_role] = SemanticRef.variable(requested_role, EntityKind.PERSON)
                frame = QuestionFrame(
                    kind=QuestionKind.WHO,
                    event=clause.event,
                    requested_role=requested_role,
                    answer_type=EntityKind.PERSON,
                    raw_text=raw,
                    unresolved=clause.unresolved,
                )
                return frame, clause.unresolved, clause.entities, clause.diagnostics + ["who requests agent of lexical do"]
            if main_idx < 0:
                return None, [], [], ["who object question lacks main verb"]
            subject = body[:main_idx]
            verb_and_tail = body[main_idx:]
            requested_role = "patient"
            synthetic = self._variable_token("patient")
            if verb_and_tail and verb_and_tail[-1].norm in {"to", "for"}:
                prep = verb_and_tail[-1]
                verb_and_tail = verb_and_tail[:-1] + [prep, self._variable_token("recipient")]
                requested_role = "recipient"
            elif verb_and_tail and verb_and_tail[-1].norm in {"with", "by"}:
                prep = verb_and_tail[-1]
                verb_and_tail = verb_and_tail[:-1] + [prep, self._variable_token("method")]
                requested_role = "method"
            else:
                insertion = self._object_insertion_index(verb_and_tail)
                verb_and_tail = verb_and_tail[:insertion] + [synthetic] + verb_and_tail[insertion:]
            declarative = subject + [aux] + verb_and_tail
            clause = self._parse_clause(declarative, raw, memory)
        else:
            # Who bought the car?  The WH term is the grammatical subject.
            declarative = [self._variable_token("agent")] + rest
            clause = self._parse_clause(declarative, raw, memory)
            requested_role = "agent"
            if clause.event:
                variable_roles = clause.event.variable_roles()
                if variable_roles:
                    requested_role = variable_roles[0]

        if not clause.event:
            return None, clause.unresolved, clause.entities, clause.diagnostics + ["failed who proposition parse"]
        clause.event.arguments[requested_role] = SemanticRef.variable(requested_role, EntityKind.PERSON)
        frame = QuestionFrame(
            kind=QuestionKind.WHO,
            event=clause.event,
            requested_role=requested_role,
            answer_type=EntityKind.PERSON,
            raw_text=raw,
            unresolved=clause.unresolved,
        )
        return frame, clause.unresolved, clause.entities, clause.diagnostics + [f"who requests {requested_role}"]

    def _parse_whose(
        self,
        items: Sequence[lexicon.Token],
        raw: str,
        memory: ConversationMemory,
    ) -> Tuple[Optional[QuestionFrame], List[UnresolvedReference], List[str], List[str]]:
        # Whose car is this? / Whose car did Sarah buy?
        rest = list(items[1:])
        if not rest:
            return None, [], [], ["whose question lacks possessed noun"]
        aux_idx = next((idx for idx, token in enumerate(rest) if token.norm in lexicon.AUXILIARIES), -1)
        possessed_tokens = rest[:aux_idx] if aux_idx > 0 else rest[:1]
        possessed_kind = lexicon.classify_unknown_noun(
            [token.norm for token in possessed_tokens]
        )
        possessed_surface = self._surface(possessed_tokens)
        possessed_alias = memory.normalize_alias(possessed_surface)
        existing = memory.find_by_alias(possessed_alias, possessed_kind)
        if existing.resolved and existing.entity:
            entity = existing.entity
            memory.mention(entity.entity_id, "patient")
            possessed = NPResult(
                entity.to_ref(possessed_surface),
                entity_ids=[entity.entity_id],
                surface=possessed_surface,
            )
        elif existing.status == "ambiguous":
            possessed = NPResult(
                None,
                [
                    memory.unresolved_from_resolution(
                        possessed_surface,
                        existing,
                        possessed_kind,
                    )
                ],
                surface=possessed_surface,
            )
        else:
            possessed = self._parse_np(
                possessed_tokens,
                memory,
                expected_kind=possessed_kind,
                role_hint="patient",
            )
        unresolved = list(possessed.unresolved)
        entities = list(possessed.entity_ids)
        args: Dict[str, SemanticRef] = {
            "possessor": SemanticRef.variable("possessor", EntityKind.PERSON),
        }
        if possessed.ref:
            args["patient"] = possessed.ref
        event = EventFrame("own", args, raw_text=raw)
        frame = QuestionFrame(
            kind=QuestionKind.WHOSE,
            event=event,
            requested_role="possessor",
            answer_type=EntityKind.PERSON,
            raw_text=raw,
            unresolved=unresolved,
            focus_surface=possessed.surface,
        )
        return frame, unresolved, entities, ["whose mapped to ownership proposition"]

    def _parse_what(
        self,
        items: Sequence[lexicon.Token],
        raw: str,
        memory: ConversationMemory,
    ) -> Tuple[Optional[QuestionFrame], List[UnresolvedReference], List[str], List[str]]:
        rest = list(items[1:])
        if not rest:
            return None, [], [], ["what question lacks predicate"]

        # What color/age/name is the car?
        if rest[0].norm in lexicon.ATTRIBUTE_NOUNS:
            attribute = rest[0].norm
            tail = rest[1:]
            if tail and tail[0].norm in lexicon.COPULAS:
                subject_tokens = tail[1:]
                subject = self._parse_np(subject_tokens, memory, role_hint="subject")
                args: Dict[str, SemanticRef] = {
                    "attribute": SemanticRef.literal(attribute, attribute, EntityKind.ABSTRACT),
                    "value": SemanticRef.variable("value", EntityKind.ABSTRACT),
                }
                if subject.ref:
                    args["subject"] = subject.ref
                event = EventFrame("attribute", args, raw_text=raw)
                frame = QuestionFrame(
                    kind=QuestionKind.WHAT,
                    event=event,
                    requested_role="value",
                    answer_type=EntityKind.ABSTRACT,
                    raw_text=raw,
                    unresolved=subject.unresolved,
                    focus_surface=attribute,
                )
                return frame, subject.unresolved, subject.entity_ids, [f"attribute question: {attribute}"]

        if rest[0].norm in lexicon.AUXILIARIES:
            aux = rest[0]
            body = rest[1:]
            main_idx = self._find_main_verb(body, start=0)
            if main_idx < 0:
                # What is Sarah? -> Sarah is ?value.
                if aux.norm in lexicon.COPULAS and body:
                    declarative = body + [aux, self._variable_token("value")]
                    clause = self._parse_clause(declarative, raw, memory)
                    requested_role = "value"
                else:
                    return None, [], [], ["what object question lacks main verb"]
            else:
                subject = body[:main_idx]
                verb_and_tail = body[main_idx:]
                if lexicon.lemma(verb_and_tail[0].norm) == "do" and len(verb_and_tail) == 1:
                    # What did Sarah do? -> ask for Sarah's event/predicate.
                    subject_np = self._parse_np(subject, memory, expected_kind=EntityKind.PERSON, role_hint="agent")
                    args = {"agent": subject_np.ref} if subject_np.ref else {}
                    event = EventFrame("*", args, raw_text=raw)
                    frame = QuestionFrame(
                        kind=QuestionKind.WHAT_HAPPENED,
                        event=event,
                        requested_role="event",
                        answer_type=EntityKind.EVENT,
                        raw_text=raw,
                        unresolved=subject_np.unresolved,
                    )
                    return frame, subject_np.unresolved, subject_np.entity_ids, ["what-did-do event query"]
                insertion = self._object_insertion_index(verb_and_tail)
                declarative = subject + [aux] + verb_and_tail[:insertion] + [self._variable_token("patient")] + verb_and_tail[insertion:]
                clause = self._parse_clause(declarative, raw, memory)
                requested_role = "patient"
        else:
            # What broke? / What caused the outage?
            declarative = [self._variable_token("agent")] + rest
            clause = self._parse_clause(declarative, raw, memory)
            requested_role = "agent"
            if clause.event and clause.event.variable_roles():
                requested_role = clause.event.variable_roles()[0]

        if not clause.event:
            return None, clause.unresolved, clause.entities, clause.diagnostics + ["failed what proposition parse"]
        answer_type = EntityKind.THING
        clause.event.arguments[requested_role] = SemanticRef.variable(requested_role, answer_type)
        frame = QuestionFrame(
            kind=QuestionKind.WHAT,
            event=clause.event,
            requested_role=requested_role,
            answer_type=answer_type,
            raw_text=raw,
            unresolved=clause.unresolved,
        )
        return frame, clause.unresolved, clause.entities, clause.diagnostics + [f"what requests {requested_role}"]

    def _parse_which(
        self,
        items: Sequence[lexicon.Token],
        raw: str,
        memory: ConversationMemory,
    ) -> Tuple[Optional[QuestionFrame], List[UnresolvedReference], List[str], List[str]]:
        rest = list(items[1:])
        if not rest:
            return None, [], [], ["which question lacks selection class"]
        aux_idx = next((idx for idx, token in enumerate(rest) if token.norm in lexicon.AUXILIARIES), -1)
        focus_tokens = rest[:aux_idx] if aux_idx > 0 else rest[:1]
        focus_surface = " ".join(token.text for token in focus_tokens)
        if aux_idx < 0:
            return None, [], [], ["which question lacks auxiliary"]
        aux = rest[aux_idx]
        body = rest[aux_idx + 1 :]
        main_idx = self._find_main_verb(body, start=0)
        if main_idx < 0:
            return None, [], [], ["which question lacks main verb"]
        declarative = body[:main_idx] + [aux] + body[main_idx:main_idx + 1] + [self._variable_token("patient")] + body[main_idx + 1 :]
        clause = self._parse_clause(declarative, raw, memory)
        if not clause.event:
            return None, clause.unresolved, clause.entities, clause.diagnostics
        clause.event.arguments["patient"] = SemanticRef.variable("patient", EntityKind.THING)
        frame = QuestionFrame(
            kind=QuestionKind.WHICH,
            event=clause.event,
            requested_role="patient",
            answer_type=EntityKind.THING,
            raw_text=raw,
            unresolved=clause.unresolved,
            focus_surface=focus_surface,
        )
        return frame, clause.unresolved, clause.entities, clause.diagnostics + [f"selection class={focus_surface}"]

    def _parse_adverbial_question(
        self,
        items: Sequence[lexicon.Token],
        raw: str,
        memory: ConversationMemory,
    ) -> Tuple[Optional[QuestionFrame], List[UnresolvedReference], List[str], List[str]]:
        qword = items[0].norm
        rest = list(items[1:])
        diagnostics: List[str] = []

        # How many/much + noun + auxiliary ...
        if qword == "how" and rest and rest[0].norm in {"many", "much"}:
            kind = QuestionKind.HOW_MANY if rest[0].norm == "many" else QuestionKind.HOW_MUCH
            body = rest[1:]
            aux_idx = next((idx for idx, token in enumerate(body) if token.norm in lexicon.AUXILIARIES), -1)
            if aux_idx < 0:
                return None, [], [], ["quantity question lacks auxiliary"]
            object_class = body[:aux_idx]
            aux = body[aux_idx]
            remainder = body[aux_idx + 1 :]
            main_idx = self._find_main_verb(remainder, start=0)
            if main_idx < 0:
                return None, [], [], ["quantity question lacks main verb"]
            declarative = remainder[:main_idx] + [aux] + remainder[main_idx:main_idx + 1] + object_class + remainder[main_idx + 1 :]
            clause = self._parse_clause(declarative, raw, memory)
            if not clause.event:
                return None, clause.unresolved, clause.entities, clause.diagnostics
            clause.event.arguments["quantity"] = SemanticRef.variable("quantity", EntityKind.ABSTRACT)
            frame = QuestionFrame(
                kind=kind,
                event=clause.event,
                requested_role="quantity",
                answer_type=EntityKind.ABSTRACT,
                how_kind=HowKind.QUANTITY,
                raw_text=raw,
                unresolved=clause.unresolved,
                focus_surface=" ".join(token.text for token in object_class),
            )
            return frame, clause.unresolved, clause.entities, clause.diagnostics + ["quantity query"]

        # How tall/old/fast is X?
        if qword == "how" and rest and rest[0].norm in lexicon.ADJECTIVE_DIMENSIONS:
            adjective = rest[0].norm
            dimension = lexicon.ADJECTIVE_DIMENSIONS[adjective]
            tail = rest[1:]
            if tail and tail[0].norm in lexicon.COPULAS:
                subject = self._parse_np(tail[1:], memory, role_hint="subject")
                args: Dict[str, SemanticRef] = {
                    "attribute": SemanticRef.literal(dimension, dimension),
                    "value": SemanticRef.variable("value", EntityKind.ABSTRACT),
                }
                if subject.ref:
                    args["subject"] = subject.ref
                event = EventFrame("attribute", args, raw_text=raw)
                frame = QuestionFrame(
                    kind=QuestionKind.HOW,
                    event=event,
                    requested_role="value",
                    answer_type=EntityKind.ABSTRACT,
                    how_kind=HowKind.DEGREE,
                    raw_text=raw,
                    unresolved=subject.unresolved,
                    focus_surface=adjective,
                )
                return frame, subject.unresolved, subject.entity_ids, [f"degree dimension={dimension}"]

        role_map = {"when": "time", "where": "location", "why": "cause", "how": "method"}
        kind_map = {
            "when": QuestionKind.WHEN,
            "where": QuestionKind.WHERE,
            "why": QuestionKind.WHY,
            "how": QuestionKind.HOW,
        }
        answer_type_map = {
            "when": EntityKind.TIME,
            "where": EntityKind.PLACE,
            "why": EntityKind.ABSTRACT,
            "how": EntityKind.ABSTRACT,
        }
        requested_role = role_map[qword]
        trailing_preposition = rest[-1].norm if rest and rest[-1].norm in lexicon.PREPOSITIONS else None
        if qword == "where" and trailing_preposition in lexicon.SOURCE_PREPOSITIONS:
            requested_role = "source"
        elif qword == "where" and trailing_preposition in lexicon.DIRECTION_PREPOSITIONS:
            requested_role = "destination"

        if rest and rest[0].norm in lexicon.COPULAS:
            aux = rest[0]
            subject_tokens = rest[1:]
            subject = self._parse_np(subject_tokens, memory, role_hint="subject")
            args: Dict[str, SemanticRef] = {}
            if subject.ref:
                args["subject"] = subject.ref
            args[requested_role if qword in {"when", "where"} else "value"] = SemanticRef.variable(
                requested_role if qword in {"when", "where"} else "value",
                answer_type_map[qword],
            )
            event = EventFrame("be", args, tense=lexicon.detect_tense(aux.norm), raw_text=raw)
            clause = ClauseResult(event, subject.unresolved, subject.entity_ids, ["copular adverbial question"])
            if qword == "how":
                requested_role = "value"
        else:
            if not rest:
                return None, [], [], [f"{qword} question lacks proposition"]
            aux = rest[0] if rest[0].norm in lexicon.AUXILIARIES else None
            body = rest[1:] if aux else rest
            main_idx = self._find_main_verb(body, start=0)
            if main_idx < 0:
                return None, [], [], [f"{qword} question lacks main verb"]
            if aux:
                declarative = body[:main_idx] + [aux] + body[main_idx:]
            else:
                declarative = body
            clause = self._parse_clause(declarative, raw, memory)
            if clause.event:
                if qword == "where" and trailing_preposition is None and clause.event.predicate in lexicon.MOVEMENT_VERBS:
                    requested_role = "destination"
                # Remove any provisional adverbial variable before installing
                # the semantically specific one (location, destination, or
                # source).
                for provisional in ("location", "destination", "source"):
                    value = clause.event.arguments.get(provisional)
                    if value is not None and value.is_variable:
                        clause.event.arguments.pop(provisional, None)
                clause.event.arguments[requested_role] = SemanticRef.variable(requested_role, answer_type_map[qword])

        if not clause.event:
            return None, clause.unresolved, clause.entities, clause.diagnostics

        why_kind = WhyKind.UNKNOWN
        how_kind = HowKind.UNKNOWN
        if qword == "why":
            why_kind = self._classify_why(clause.event)
            requested_role = why_kind.value if why_kind in {WhyKind.CAUSE, WhyKind.MOTIVE, WhyKind.PURPOSE, WhyKind.JUSTIFICATION, WhyKind.EVIDENCE} else "cause"
            clause.event.arguments.pop("cause", None)
            clause.event.arguments[requested_role] = SemanticRef.variable(requested_role, EntityKind.ABSTRACT)
            diagnostics.append(f"why subtype={why_kind.value}")
        elif qword == "how":
            how_kind = self._classify_how(clause.event)
            requested_role = "value" if clause.event.predicate == "be" else how_kind.value
            if requested_role not in clause.event.arguments or not clause.event.arguments[requested_role].is_variable:
                clause.event.arguments.pop("method", None)
                clause.event.arguments[requested_role] = SemanticRef.variable(requested_role, EntityKind.ABSTRACT)
            diagnostics.append(f"how subtype={how_kind.value}")

        frame = QuestionFrame(
            kind=kind_map[qword],
            event=clause.event,
            requested_role=requested_role,
            answer_type=answer_type_map[qword],
            why_kind=why_kind,
            how_kind=how_kind,
            raw_text=raw,
            unresolved=clause.unresolved,
        )
        return frame, clause.unresolved, clause.entities, clause.diagnostics + diagnostics + [f"{qword} requests {requested_role}"]
