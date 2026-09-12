"""Clauses recognition and transformation. Shared state is passed explicitly as memory."""

from __future__ import annotations
from typing import Dict, List, Sequence
from .. import lexicon
from ..memory import ConversationMemory
from ..model import EntityKind, EventFrame, SemanticRef, SourceKind, UnresolvedReference
from .types import ClauseResult

class ClausesRules:
    """Stateless clauses transformations composed by SemanticParser."""

    def _clause_has_resolved_subject(
        self,
        tokens: Sequence[lexicon.Token],
        event: EventFrame,
    ) -> bool:
        """Return whether the surface clause resolved its grammatical subject."""

        items = [
            token for token in tokens if token.norm not in lexicon.PUNCTUATION
        ]
        main_items, _cause_items = self._split_cause(items)
        verb_index = self._find_main_verb(main_items)
        if verb_index < 0:
            return False
        predicate = lexicon.lemma(main_items[verb_index].norm)
        auxiliary_tokens = [
            token.norm
            for token in main_items[:verb_index]
            if token.norm in lexicon.AUXILIARIES
        ]
        passive = (
            predicate != "be"
            and any(word in lexicon.COPULAS for word in auxiliary_tokens)
            and any(
                token.norm == "by" for token in main_items[verb_index + 1 :]
            )
        )
        return self._subject_role(predicate, passive) in event.arguments

    def _parse_clause(
        self,
        tokens: Sequence[lexicon.Token],
        raw: str,
        memory: ConversationMemory,
    ) -> ClauseResult:
        items = [token for token in tokens if token.norm not in lexicon.PUNCTUATION]
        if not items:
            return ClauseResult(None, diagnostics=["empty clause"])

        diagnostics: List[str] = []
        unresolved: List[UnresolvedReference] = []
        entities: List[str] = []

        # Split explicit causal clause before finding the main predicate.
        main_items, cause_items = self._split_cause(items)
        purpose_items: List[lexicon.Token] = []

        verb_idx = self._find_main_verb(main_items)
        if verb_idx < 0:
            return ClauseResult(None, diagnostics=["no main predicate found"])

        main_token = main_items[verb_idx]
        predicate = lexicon.lemma(main_token.norm)
        auxiliary_tokens = [token.norm for token in main_items[:verb_idx] if token.norm in lexicon.AUXILIARIES]
        modality = next((word for word in auxiliary_tokens if word in lexicon.MODALS), None)
        polarity = not any(
            token.norm in lexicon.NEGATORS for token in main_items
        )
        tense = lexicon.detect_tense(main_token.norm, auxiliary_tokens[0] if auxiliary_tokens else None)
        if "will" in auxiliary_tokens or "shall" in auxiliary_tokens:
            tense = "future"
        perfect = any(word in lexicon.AUX_HAVE for word in auxiliary_tokens)
        progressive = (
            (
                main_token.norm.endswith("ing")
                and any(word in lexicon.COPULAS for word in auxiliary_tokens)
            )
            or "being" in auxiliary_tokens
        )
        if perfect and progressive:
            aspect = "perfect_progressive"
        elif perfect:
            aspect = "perfect"
        elif progressive:
            aspect = "progressive"
        else:
            aspect = "simple"
        if perfect and tense != "future":
            tense = "past" if "had" in auxiliary_tokens else "present"
        elif progressive and tense != "future":
            if any(word in {"was", "were"} for word in auxiliary_tokens):
                tense = "past"
            else:
                tense = "present"

        # Passive voice: "The coat was bought by Sarah".
        passive = (
            predicate != "be"
            and any(word in lexicon.COPULAS for word in auxiliary_tokens)
            and any(token.norm == "by" for token in main_items[verb_idx + 1 :])
        )

        subject_tokens = [
            token for token in main_items[:verb_idx]
            if token.norm not in lexicon.AUXILIARIES
            and token.norm not in lexicon.NEGATORS
            and token.norm not in lexicon.INTENSIFIERS
        ]
        if passive:
            subject_expected = EntityKind.THING
        else:
            subject_expected = EntityKind.PERSON if predicate in lexicon.VOLITIONAL_VERBS | lexicon.COMMUNICATION_VERBS else EntityKind.UNKNOWN
        subject = self._parse_np(subject_tokens, memory, expected_kind=subject_expected, role_hint="subject")
        unresolved.extend(subject.unresolved)
        entities.extend(subject.entity_ids)

        args: Dict[str, SemanticRef] = {}
        subject_role = self._subject_role(predicate, passive)
        if subject.ref:
            args[subject_role] = subject.ref
        if subject.quantity:
            args["quantity"] = subject.quantity

        if predicate == "be":
            complement_tokens = [token for token in main_items[verb_idx + 1 :] if token.norm not in lexicon.NEGATORS]
            self._parse_copular_complement(complement_tokens, args, memory, unresolved, entities, diagnostics)
        else:
            tail = list(main_items[verb_idx + 1 :])
            # Remove leading negator after the main verb.
            tail = [token for token in tail if token.norm != "not"]

            # Normalize separable phrasal predicates while preserving the
            # argument between verb and particle: ``pissed me off`` becomes
            # ANGER(agent, patient), not a patient named ``me off``.
            for particle_index, token in enumerate(list(tail)):
                mapped = lexicon.PHRASAL_VERBS.get((predicate, token.norm))
                if mapped:
                    predicate = mapped
                    tail.pop(particle_index)
                    diagnostics.append(f"phrasal predicate normalized via {token.norm}")
                    break

            # Extract purpose infinitive after the first object/destination.
            purpose_index = self._find_purpose_index(tail, predicate)
            if purpose_index >= 0:
                purpose_items = tail[purpose_index + 1 :]
                tail = tail[:purpose_index]
            self._parse_predicate_tail(
                predicate,
                tail,
                args,
                memory,
                unresolved,
                entities,
                diagnostics,
                passive=passive,
            )
            if predicate in {"belong", "have", "possess"}:
                predicate = "own"
                diagnostics.append("normalized possession predicate as ownership")

        if cause_items:
            cause_surface = self._surface(cause_items)
            cause_key = lexicon.normalize_phrase(token.norm for token in cause_items)
            cause_role = "motive" if predicate in lexicon.VOLITIONAL_VERBS else "cause"
            args[cause_role] = SemanticRef.literal(cause_key, cause_surface, EntityKind.ABSTRACT)
            # Preserve generic cause as a fallback anchor as well.
            if cause_role != "cause":
                args.setdefault("cause", SemanticRef.literal(cause_key, cause_surface, EntityKind.ABSTRACT))
            diagnostics.append(f"explicit {cause_role} clause")

        if purpose_items:
            surface = self._surface(purpose_items)
            key = lexicon.normalize_phrase(token.norm for token in purpose_items)
            args["purpose"] = SemanticRef.literal(key, surface, EntityKind.ABSTRACT)
            diagnostics.append("purpose infinitive")

        event = EventFrame(
            predicate=predicate,
            arguments=args,
            tense=tense,
            aspect=aspect,
            polarity=polarity,
            modality=modality,
            raw_text=raw,
            source=SourceKind.USER,
            certainty=230,
            turn_index=memory.turn_index,
        )
        diagnostics.append(
            f"frame predicate={predicate} roles={','.join(sorted(args)) or 'none'} tense={tense} polarity={polarity}"
        )
        return ClauseResult(event, unresolved, list(dict.fromkeys(entities)), diagnostics)

    def _find_main_verb(self, tokens: Sequence[lexicon.Token], start: int = 0) -> int:
        words = [token.norm for token in tokens]
        for idx in range(start, len(tokens)):
            word = words[idx]
            previous = words[idx - 1] if idx > 0 else None
            following = words[idx + 1] if idx + 1 < len(words) else None
            if word in lexicon.NEGATORS:
                continue
            if word in lexicon.AUXILIARIES:
                next_idx = idx + 1
                while next_idx < len(words) and words[next_idx] in lexicon.NEGATORS | lexicon.INTENSIFIERS:
                    next_idx += 1

                # Copula is the predicate unless followed by a recognizable
                # participle/verb (passive/progressive construction).
                if word in lexicon.COPULAS:
                    if next_idx < len(words) and lexicon.is_probable_verb(words[next_idx], word):
                        continue
                    return idx

                # HAVE and DO are also ordinary lexical predicates.  Only
                # skip them when a following verb proves auxiliary use:
                # ``has bought`` / ``did buy``.  This keeps ``has blue eyes``
                # and ``did the work`` from selecting a noun/adjective as the
                # main predicate.
                if word in lexicon.AUX_HAVE | lexicon.AUX_DO:
                    if next_idx < len(words) and lexicon.is_probable_verb(words[next_idx], word):
                        continue
                    return idx
                # Modals are always auxiliaries in this grammar.
                continue
            if word.startswith("__var_"):
                continue
            if lexicon.is_probable_verb(word, previous, following):
                return idx
        return -1

    @staticmethod
    def _subject_role(predicate: str, passive: bool) -> str:
        if passive:
            return "patient"
        if predicate == "be":
            return "subject"
        if predicate == "belong":
            return "patient"
        if predicate in lexicon.UNACCUSATIVE_VERBS:
            return "patient"
        if predicate in lexicon.POSSESSION_VERBS:
            return "possessor"
        if predicate == "feel":
            return "experiencer"
        return "agent"

    def _parse_copular_complement(
        self,
        tokens: Sequence[lexicon.Token],
        args: Dict[str, SemanticRef],
        memory: ConversationMemory,
        unresolved: List[UnresolvedReference],
        entities: List[str],
        diagnostics: List[str],
    ) -> None:
        if not tokens:
            return
        first = tokens[0].norm
        if first.startswith("__var_"):
            role = first[len("__var_") : -2]
            args[role] = SemanticRef.variable(role)
            return
        if first in lexicon.LOCATION_PREPOSITIONS:
            phrase = list(tokens[1:])
            phrase_words = [token.norm for token in phrase]
            if lexicon.is_time_phrase(phrase_words) or (first == "at" and lexicon.is_clock_phrase(phrase_words)):
                args["time"] = SemanticRef.literal(lexicon.normalize_phrase(token.norm for token in phrase), self._surface(phrase), EntityKind.TIME)
                args["time_preposition"] = SemanticRef.literal(first, first)
                diagnostics.append("copular time complement")
            else:
                location = self._parse_np(phrase, memory, expected_kind=EntityKind.PLACE, role_hint="location", preposition=first)
                unresolved.extend(location.unresolved)
                entities.extend(location.entity_ids)
                if location.ref:
                    args["location"] = location.ref
                    args["location_preposition"] = SemanticRef.literal(first, first)
                diagnostics.append("copular location complement")
            return
        if first in lexicon.TIME_PREPOSITIONS and (
            lexicon.is_time_phrase([token.norm for token in tokens[1:]])
            or (first == "at" and lexicon.is_clock_phrase([token.norm for token in tokens[1:]]))
        ):
            args["time"] = SemanticRef.literal(
                lexicon.normalize_phrase(token.norm for token in tokens[1:]),
                self._surface(tokens[1:]),
                EntityKind.TIME,
            )
            args["time_preposition"] = SemanticRef.literal(first, first)
            return
        # Copular values are literals by default.  Named people remain entities.
        if len(tokens) == 1 and (tokens[0].text[:1].isupper() or tokens[0].norm in lexicon.FEMALE_NAMES | lexicon.MALE_NAMES):
            value = self._parse_np(tokens, memory, expected_kind=EntityKind.PERSON, role_hint="value")
            unresolved.extend(value.unresolved)
            entities.extend(value.entity_ids)
            if value.ref:
                args["value"] = value.ref
        else:
            surface = self._surface(tokens)
            key = lexicon.normalize_phrase(token.norm for token in tokens if token.norm not in lexicon.INTENSIFIERS)
            args["value"] = SemanticRef.literal(key, surface, EntityKind.ABSTRACT)
            # Store common adjective dimensions as explicit attributes too.
            adjective = next((token.norm for token in tokens if token.norm in lexicon.ADJECTIVE_DIMENSIONS), None)
            if adjective:
                args["attribute"] = SemanticRef.literal(lexicon.ADJECTIVE_DIMENSIONS[adjective])
            else:
                color = next((token.norm for token in tokens if token.norm in lexicon.COLORS), None)
                if color:
                    args["attribute"] = SemanticRef.literal("color", "color")
