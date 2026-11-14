"""Predicate Roles recognition and transformation. Shared state is passed explicitly as memory."""

from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
from .. import lexicon
from ..memory import ConversationMemory
from ..model import EntityKind, RefKind, SemanticRef, UnresolvedReference

class PredicateRolesRules:
    """Stateless predicate roles transformations composed by SemanticParser."""

    def _parse_predicate_tail(
        self,
        predicate: str,
        tokens: Sequence[lexicon.Token],
        args: Dict[str, SemanticRef],
        memory: ConversationMemory,
        unresolved: List[UnresolvedReference],
        entities: List[str],
        diagnostics: List[str],
        *,
        passive: bool,
    ) -> None:
        chunks = self._chunk_tail(tokens)
        direct_tokens = chunks.pop("direct", [])

        # Remove temporal material from the direct object tail.
        direct_tokens, direct_time = self._extract_trailing_time(direct_tokens)
        if direct_time:
            args["time"] = SemanticRef.literal(
                lexicon.normalize_phrase(token.norm for token in direct_time),
                self._surface(direct_time),
                EntityKind.TIME,
            )

        if predicate in lexicon.DITRANSITIVE_VERBS and direct_tokens:
            recipient_tokens, theme_tokens = self._split_ditransitive(direct_tokens)
            if recipient_tokens and theme_tokens:
                recipient = self._parse_np(recipient_tokens, memory, expected_kind=EntityKind.PERSON, role_hint="recipient")
                theme = self._parse_np(theme_tokens, memory, expected_kind=EntityKind.THING, role_hint="patient")
                unresolved.extend(recipient.unresolved + theme.unresolved)
                entities.extend(recipient.entity_ids + theme.entity_ids)
                if recipient.ref:
                    args["recipient"] = recipient.ref
                if theme.ref:
                    args["patient"] = theme.ref
                if theme.quantity:
                    args["quantity"] = theme.quantity
            else:
                self._attach_direct_object(predicate, direct_tokens, args, memory, unresolved, entities)
        elif direct_tokens:
            self._attach_direct_object(predicate, direct_tokens, args, memory, unresolved, entities)

        for prep, phrase in chunks.items():
            if not phrase:
                continue
            role = self._preposition_role(prep, predicate, phrase, passive)
            expected = {
                "agent": EntityKind.PERSON,
                "possessor": EntityKind.PERSON,
                "recipient": EntityKind.PERSON,
                "location": EntityKind.PLACE,
                "destination": EntityKind.PLACE,
                "source": EntityKind.PLACE,
                "time": EntityKind.TIME,
            }.get(role, EntityKind.UNKNOWN)
            if role in {"location", "destination", "source", "time", "method", "manner"}:
                args[f"{role}_preposition"] = SemanticRef.literal(prep, prep)
            if role == "time" or (role in {"method", "manner", "purpose", "topic"} and not self._looks_entity_phrase(phrase)):
                args[role] = SemanticRef.literal(
                    lexicon.normalize_phrase(token.norm for token in phrase),
                    self._surface(phrase),
                    expected if expected != EntityKind.UNKNOWN else EntityKind.ABSTRACT,
                )
                continue
            value = self._parse_np(phrase, memory, expected_kind=expected, role_hint=role, preposition=prep)
            unresolved.extend(value.unresolved)
            entities.extend(value.entity_ids)
            if value.ref:
                args[role] = value.ref
            if value.quantity:
                args.setdefault("quantity", value.quantity)

        diagnostics.append(f"tail chunks={','.join(chunks.keys()) or 'none'}")

    def _attach_direct_object(
        self,
        predicate: str,
        tokens: Sequence[lexicon.Token],
        args: Dict[str, SemanticRef],
        memory: ConversationMemory,
        unresolved: List[UnresolvedReference],
        entities: List[str],
    ) -> None:
        person_patients = {
            "meet", "call", "text", "marry", "help", "love", "hate", "tell",
            "upset", "hurt", "hit", "teach", "see", "hear", "ask", "answer",
        }
        thing_patients = {
            "buy", "purchase", "open", "close", "use", "unlock", "build",
            "create", "delete", "fix", "read", "write", "eat", "drink",
            "wear", "sell", "find", "take", "make", "break", "cut", "put",
        }
        if predicate in person_patients:
            expected = EntityKind.PERSON
        elif predicate in thing_patients:
            expected = EntityKind.THING
        else:
            expected = EntityKind.UNKNOWN
        object_result = self._parse_np(tokens, memory, expected_kind=expected, role_hint="patient")
        unresolved.extend(object_result.unresolved)
        entities.extend(object_result.entity_ids)
        if object_result.ref:
            role = "state" if predicate == "feel" else "patient"
            args[role] = object_result.ref
            if predicate in lexicon.POSSESSION_VERBS and object_result.ref.kind == RefKind.ENTITY:
                owner = args.get("possessor") or args.get("agent")
                entity = memory.get_entity(object_result.ref.key)
                if owner and owner.kind == RefKind.ENTITY and entity:
                    entity.owner_id = owner.key
                    entity.add_alias(f"{owner.key}:{entity.canonical_name}")
        if object_result.quantity:
            args["quantity"] = object_result.quantity

    @staticmethod
    def _chunk_tail(tokens: Sequence[lexicon.Token]) -> Dict[str, List[lexicon.Token]]:
        chunks: Dict[str, List[lexicon.Token]] = {"direct": []}
        current = "direct"
        for token in tokens:
            if token.norm in lexicon.PREPOSITIONS:
                current = token.norm
                if current in chunks:
                    # Preserve multiple identical prepositions by suffixing.
                    suffix = 2
                    while f"{current}#{suffix}" in chunks:
                        suffix += 1
                    current = f"{current}#{suffix}"
                chunks[current] = []
            else:
                chunks[current].append(token)
        # Normalize duplicate keys back to their preposition; later occurrence
        # wins only when the semantic role is the same.
        normalized: Dict[str, List[lexicon.Token]] = {"direct": chunks.get("direct", [])}
        for key, value in chunks.items():
            if key == "direct":
                continue
            normalized[key.split("#", 1)[0]] = value
        return normalized

    @staticmethod
    def _split_ditransitive(tokens: Sequence[lexicon.Token]) -> Tuple[List[lexicon.Token], List[lexicon.Token]]:
        if len(tokens) < 2:
            return [], list(tokens)
        # Recipient is normally a pronoun/name or one-token relation before an
        # article/quantity-led theme: "gave Mary a book", "gave her the key".
        for idx in range(1, len(tokens)):
            if tokens[idx].norm in lexicon.ARTICLES or tokens[idx].norm in lexicon.NUMBER_WORDS or tokens[idx].norm.isdigit():
                return list(tokens[:idx]), list(tokens[idx:])
        if tokens[0].norm in lexicon.PRONOUN_FEATURES or tokens[0].text[:1].isupper() or tokens[0].norm in lexicon.RELATIONS:
            return [tokens[0]], list(tokens[1:])
        return [], list(tokens)

    @staticmethod
    def _extract_trailing_time(tokens: Sequence[lexicon.Token]) -> Tuple[List[lexicon.Token], List[lexicon.Token]]:
        items = list(tokens)
        if not items:
            return items, []
        for idx, token in enumerate(items):
            if token.norm in lexicon.RELATIVE_TIMES or token.norm in lexicon.DAYS:
                return items[:idx], items[idx:]
            if token.norm in {"last", "next", "this"} and idx + 1 < len(items) and items[idx + 1].norm in lexicon.TIME_WORDS:
                return items[:idx], items[idx:]
        return items, []

    @staticmethod
    def _find_purpose_index(tokens: Sequence[lexicon.Token], predicate: str) -> int:
        seen_nonprep_to = 0
        for idx, token in enumerate(tokens[:-1]):
            if token.norm != "to":
                continue
            next_word = tokens[idx + 1].norm
            if lexicon.is_probable_verb(next_word, "to"):
                # For movement verbs the first "to school" is destination, but
                # "to buy milk" is purpose because the next token is a verb.
                return idx
            seen_nonprep_to += 1
        return -1

    @staticmethod
    def _split_cause(tokens: Sequence[lexicon.Token]) -> Tuple[List[lexicon.Token], List[lexicon.Token]]:
        norms = [token.norm for token in tokens]
        if "because" in norms:
            idx = norms.index("because")
            cause_start = idx + 1
            if cause_start < len(tokens) and tokens[cause_start].norm == "of":
                cause_start += 1
            return list(tokens[:idx]), list(tokens[cause_start:])
        if "due" in norms:
            idx = norms.index("due")
            if idx + 1 < len(tokens) and tokens[idx + 1].norm == "to":
                return list(tokens[:idx]), list(tokens[idx + 2 :])
        return list(tokens), []

    @staticmethod
    def _preposition_role(preposition: str, predicate: str, phrase: Sequence[lexicon.Token], passive: bool) -> str:
        words = [token.norm for token in phrase]
        if preposition == "by" and passive:
            return "agent"
        if preposition in lexicon.TIME_PREPOSITIONS and lexicon.is_time_phrase(words):
            return "time"
        if preposition in lexicon.LOCATION_PREPOSITIONS:
            return "location"
        if preposition in lexicon.DIRECTION_PREPOSITIONS:
            if predicate == "belong":
                return "possessor"
            if predicate in lexicon.DITRANSITIVE_VERBS:
                return "recipient"
            if predicate in lexicon.MOVEMENT_VERBS:
                return "destination"
            return "goal"
        if preposition in lexicon.SOURCE_PREPOSITIONS:
            return "source"
        if preposition in lexicon.METHOD_PREPOSITIONS:
            return "method"
        if preposition == "for":
            return "purpose" if any(lexicon.is_probable_verb(word) for word in words) else "beneficiary"
        if preposition == "about":
            return "topic"
        if preposition == "of":
            return "relation"
        return "adjunct"
