"""Input orchestration. Construction bodies live in parsing.steps."""

from __future__ import annotations
from .. import lexicon
from ..memory import ConversationMemory
from ..model import ParseResult, SpeechAct, UnresolvedReference
from .context import ParseContext
from .steps import assemble

class InputPipeline:
    """Ingress, scope preflight, and explicit ordered stage execution."""

    def parse(self, text: str, memory: ConversationMemory) -> ParseResult:
        return self._parse_input(text, memory)

    def _parse_input(self, text: str, memory: ConversationMemory) -> ParseResult:
        raw = text.strip()
        if not raw:
            return ParseResult(SpeechAct.UNKNOWN, raw, diagnostics=["empty input"])

        tokens = lexicon.tokenize(raw, include_punctuation=True)
        normalized = " ".join(token.norm for token in tokens if token.norm not in lexicon.PUNCTUATION)
        # Retain semicolons until assertion segmentation.  Clause parsing
        # removes punctuation later, but deleting ``;`` here made the existing
        # semicolon branch unreachable.
        clean = [token for token in tokens if token.norm not in {".", "!", "?"}]
        clean = self._rewrite_yoda(clean)
        clean = lexicon.strip_discourse_prefix(clean)
        # A final casual vocative (``my tummy hurts, bruh``) controls register
        # but is not a semantic patient.  Keep it in raw text for Clanker's
        # affect/gating pass and remove it only from the proposition parser.
        while len(clean) > 1 and clean[-1].norm in {"bruh", "bro", "dude", "fam", "bestie"}:
            clean = clean[:-1]
        clean = [lexicon.Token(token.text, token.norm, idx) for idx, token in enumerate(clean)]
        words = [token.norm for token in clean if token.norm not in lexicon.PUNCTUATION]

        if not words:
            return ParseResult(SpeechAct.UNKNOWN, raw, normalized_text=normalized, diagnostics=["no lexical tokens"])

        social = self._detect_social(words)
        if social == "greeting":
            return ParseResult(
                speech_act=SpeechAct.GREET,
                raw_text=raw,
                normalized_text=normalized,
                diagnostics=[f"social convention: {social}"],
            )

        # A fronted finite subordinate clause can begin with a WH-shaped
        # marker (especially ``when``) without asking a question.  The comma
        # boundary and independently finite clauses provide stronger
        # structural evidence than the first token alone.
        fronted_subordinate = (
            not raw.rstrip().endswith("?")
            and self._split_subordinate_clause(clean) is not None
            and clean[0].norm in {"when", "while", "before", "after", "until", "since", "if", "unless", "although", "though"}
        )
        is_question = raw.rstrip().endswith("?") or (
            not fronted_subordinate
            and (
                words[0] in lexicon.QUESTION_WORDS
                or words[0] in lexicon.YES_NO_STARTERS
            )
        )
        if is_question:
            question, unresolved, entities, diagnostics = self._parse_question(clean, raw, memory)
            return ParseResult(
                speech_act=SpeechAct.ASK if question else SpeechAct.UNKNOWN,
                raw_text=raw,
                question=question,
                entities=entities,
                unresolved=unresolved,
                normalized_text=normalized,
                diagnostics=diagnostics,
            )

        # Coordination combined with a selected infinitive is intentionally
        # deferred until the relation layers can be composed without losing a
        # controller or attaching the infinitive to the wrong conjunct.  Run a
        # full-clause preflight before coordination splits the token stream.
        _preflight_split, preflight_ambiguity = (
            self._split_infinitival_clause(clean)
        )
        if (
            preflight_ambiguity is not None
            and "coordination" in preflight_ambiguity.reason
        ):
            unresolved = UnresolvedReference(
                surface=preflight_ambiguity.complement_surface,
                reason=preflight_ambiguity.reason,
            )
            return ParseResult(
                speech_act=SpeechAct.ASSERT,
                raw_text=raw,
                infinitival_ambiguities=[preflight_ambiguity],
                unresolved=[unresolved],
                normalized_text=normalized,
                diagnostics=[
                    *preflight_ambiguity.diagnostics,
                    "infinitival coordination requires staged parsing",
                ],
            )

        _gerund_preflight, gerund_preflight_ambiguity = (
            self._split_gerund_clause(clean)
        )
        if (
            gerund_preflight_ambiguity is not None
            and "coordination" in gerund_preflight_ambiguity.reason
        ):
            unresolved = UnresolvedReference(
                surface=gerund_preflight_ambiguity.complement_surface,
                reason=gerund_preflight_ambiguity.reason,
            )
            return ParseResult(
                speech_act=SpeechAct.ASSERT,
                raw_text=raw,
                gerund_ambiguities=[gerund_preflight_ambiguity],
                unresolved=[unresolved],
                normalized_text=normalized,
                diagnostics=[
                    *gerund_preflight_ambiguity.diagnostics,
                    "gerund coordination requires staged parsing",
                ],
            )

        ctx = ParseContext(raw=raw, normalized=normalized, memory=memory)
        for clause, connector in self._split_assertion_segments(clean):
            assemble(self, clause, connector, ctx)
        return ctx.result()
