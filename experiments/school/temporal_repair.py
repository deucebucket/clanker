"""Opt-in parser candidate for #134, not a production grammar replacement.

Changes one attachment boundary through the existing SemanticParser interface.
No output sentence is rewritten. Source and time are separated before the
original NP parser sees the PP. Existing engine and sealed evaluator are intact.
"""
from __future__ import annotations

from clanker_lm import lexicon
from clanker_lm.model import EntityKind, SemanticRef, QuestionFrame, QuestionKind
from clanker_lm.parser import SemanticParser


class TemporalAttachmentAmbiguity(ValueError):
    """The narrow candidate will not decide incompatible temporal attachments."""


class TemporalSuffixParser(SemanticParser):
    DEICTICS = frozenset({"yesterday", "today", "tomorrow", "tonight"})
    PP_CANDIDATES = frozenset({"from", "to", "at", "in", "with", "by"})
    QUOTES = frozenset({'"', "'", "“", "”", "‘", "’"})

    def _parse_predicate_tail(self, predicate, tokens, args, memory, unresolved,
                              entities, diagnostics, *, passive):
        chunks = self._chunk_tail(tokens)
        last_prep = next(reversed(chunks), "direct")
        phrase = chunks.get(last_prep, [])
        suffix = None
        if (last_prep in self.PP_CANDIDATES and len(phrase) >= 2
                and phrase[-1].norm in self.DEICTICS
                and phrase[-1].text == phrase[-1].norm
                and not any(t.text in self.QUOTES for t in phrase)
                and not lexicon.is_time_phrase([t.norm for t in phrase[:-1]])
                and self._preposition_role(last_prep, predicate, phrase[:-1], passive) != "time"):
            suffix = phrase[-1]
        if suffix is None:
            return super()._parse_predicate_tail(
                predicate, tokens, args, memory, unresolved, entities, diagnostics,
                passive=passive)

        # Existing explicit/direct time evidence may agree, but is never
        # silently overwritten. Ambiguous cases need a broader typed policy.
        previous_times = []
        if "time" in args:
            previous_times.append(args["time"].key)
        _, direct_time = self._extract_trailing_time(chunks.get("direct", []))
        if direct_time:
            previous_times.append(lexicon.normalize_phrase(t.norm for t in direct_time))
        for prep, content in chunks.items():
            if prep not in {"direct", last_prep} and self._preposition_role(
                    prep, predicate, content, passive) == "time":
                previous_times.append(lexicon.normalize_phrase(t.norm for t in content))
        if any(value != suffix.norm for value in previous_times):
            raise TemporalAttachmentAmbiguity("conflicting trailing time evidence")

        shortened = [t for t in tokens if t is not suffix]
        super()._parse_predicate_tail(predicate, shortened, args, memory, unresolved,
                                     entities, diagnostics, passive=passive)
        if "time" in args and args["time"].key != suffix.norm:
            raise TemporalAttachmentAmbiguity("parser produced incompatible time evidence")
        args["time"] = SemanticRef.literal(suffix.norm, suffix.text, EntityKind.TIME)
        diagnostics.append("school-candidate: detached final PP deictic before entity resolution")

    def _parse_who(self, items, raw, memory):
        # Missing source-role branch in the native WH transformation. Keep
        # the direct object fixed; bind the question variable after 'from'.
        rest = list(items[1:])
        if (len(rest) < 4 or rest[0].norm not in lexicon.AUXILIARIES
                or rest[-1].norm not in lexicon.SOURCE_PREPOSITIONS):
            return super()._parse_who(items, raw, memory)
        auxiliary, body = rest[0], rest[1:]
        index = self._find_main_verb(body, start=0)
        if index <= 0:
            return super()._parse_who(items, raw, memory)
        declarative = body[:index] + [auxiliary] + body[index:] + [self._variable_token("source")]
        clause = self._parse_clause(declarative, raw, memory)
        if clause.event is None:
            return None, clause.unresolved, clause.entities, clause.diagnostics
        clause.event.arguments["source"] = SemanticRef.variable("source", EntityKind.PERSON)
        frame = QuestionFrame(kind=QuestionKind.WHO, event=clause.event,
                              requested_role="source", answer_type=EntityKind.PERSON,
                              raw_text=raw, unresolved=clause.unresolved)
        return frame, clause.unresolved, clause.entities, clause.diagnostics + [
            "school-candidate: WHO binds source without replacing the direct object"]
