"""Deterministic semantic parser for Clanker-LM.

The parser maps common conversational English into entity/event frames and
turns interrogatives into propositions containing a typed open slot.  It is a
rule system, not a hidden statistical parser: every transformation is exposed
through diagnostics and stable semantic roles.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import lexicon
from .memory import ConversationMemory, Resolution
from .model import (
    AppositiveAttachmentAmbiguity,
    AppositiveRelation,
    AppositiveRelationType,
    ClauseRelation,
    ClauseRelationDirection,
    ClauseRelationType,
    ContentAttachmentAmbiguity,
    ContentRelation,
    ContentRelationType,
    EntityKind,
    EntityModifierRelation,
    EventFrame,
    Gender,
    GrammaticalNumber,
    ModifierAttachmentAmbiguity,
    ModifierGapRole,
    ModifierRestriction,
    HowKind,
    InfinitivalAttachmentAmbiguity,
    InfinitivalContentStatus,
    InfinitivalRelation,
    InfinitivalRelationType,
    ParseResult,
    QuestionFrame,
    QuestionKind,
    RefKind,
    SemanticRef,
    SourceKind,
    SpeechAct,
    UnresolvedReference,
    WhyKind,
)


@dataclass
class NPResult:
    ref: Optional[SemanticRef]
    unresolved: List[UnresolvedReference] = field(default_factory=list)
    quantity: Optional[SemanticRef] = None
    entity_ids: List[str] = field(default_factory=list)
    surface: str = ""


@dataclass
class ClauseResult:
    event: Optional[EventFrame]
    unresolved: List[UnresolvedReference] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)


@dataclass
class ContentSplit:
    matrix_tokens: List[lexicon.Token]
    content_tokens: List[lexicon.Token]
    marker: str
    relation_type: ContentRelationType
    predicate_family: str
    certainty: int
    diagnostics: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class InfinitivalPredicateProfile:
    relation_type: InfinitivalRelationType
    content_status: InfinitivalContentStatus
    predicate_family: str
    certainty: int
    allows_object_controller: bool = False
    allows_subject_controller: bool = True


@dataclass
class InfinitivalSplit:
    matrix_tokens: List[lexicon.Token]
    complement_tokens: List[lexicon.Token]
    marker: str
    profile: InfinitivalPredicateProfile
    controller_role: str
    certainty: int
    diagnostics: List[str] = field(default_factory=list)


@dataclass
class RelativeSplit:
    main_tokens: List[lexicon.Token]
    modifier_tokens: List[lexicon.Token]
    head_entity_id: str
    marker: str
    gap_role: ModifierGapRole
    restriction: ModifierRestriction
    possessed_entity_id: str = ""
    certainty: int = 230
    diagnostics: List[str] = field(default_factory=list)


@dataclass
class AppositiveSplit:
    main_tokens: List[lexicon.Token]
    relation: AppositiveRelation
    entity_ids: List[str] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)


@dataclass
class SubordinateSplit:
    main_tokens: List[lexicon.Token]
    subordinate_tokens: List[lexicon.Token]
    marker: str
    relation_type: ClauseRelationType
    direction: ClauseRelationDirection
    candidate_types: List[ClauseRelationType] = field(default_factory=list)
    certainty: int = 230
    diagnostics: List[str] = field(default_factory=list)


class SemanticParser:
    """Conservative deterministic parser for the Clanker-LM vertical slice."""

    def parse(self, text: str, memory: ConversationMemory) -> ParseResult:
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

        segments = self._split_assertion_segments(clean)
        events: List[EventFrame] = []
        relations: List[ClauseRelation] = []
        modifiers: List[EntityModifierRelation] = []
        modifier_ambiguities: List[ModifierAttachmentAmbiguity] = []
        appositives: List[AppositiveRelation] = []
        appositive_ambiguities: List[AppositiveAttachmentAmbiguity] = []
        contents: List[ContentRelation] = []
        content_ambiguities: List[ContentAttachmentAmbiguity] = []
        infinitivals: List[InfinitivalRelation] = []
        infinitival_ambiguities: List[InfinitivalAttachmentAmbiguity] = []
        unresolved: List[UnresolvedReference] = []
        entities: List[str] = []
        diagnostics: List[str] = []
        primary_event_count = 0
        for clause, connector in segments:
            infinitival_split, infinitival_ambiguity = (
                self._split_infinitival_clause(clause)
            )
            if infinitival_ambiguity is not None:
                infinitival_ambiguities.append(infinitival_ambiguity)
                unresolved.append(
                    UnresolvedReference(
                        surface=infinitival_ambiguity.complement_surface,
                        reason=infinitival_ambiguity.reason,
                    )
                )
                diagnostics.extend(infinitival_ambiguity.diagnostics)
                diagnostics.append(
                    "infinitival complement boundary/controller remains ambiguous"
                )
                continue

            if infinitival_split is not None:
                matrix_result = self._parse_clause(
                    infinitival_split.matrix_tokens,
                    raw,
                    memory,
                )
                if matrix_result.event is not None:
                    source_entity_id = self._content_source_entity(
                        matrix_result.event
                    )
                    controller_entity_id = self._infinitival_controller_entity(
                        matrix_result.event,
                        infinitival_split.controller_role,
                    )
                    if source_entity_id and controller_entity_id:
                        internal_alias = memory.ensure_internal_alias(
                            controller_entity_id
                        )
                        embedded_tokens = [
                            lexicon.Token(
                                internal_alias,
                                internal_alias,
                                -1,
                            ),
                            *infinitival_split.complement_tokens,
                        ]
                        complement_result = self._parse_clause(
                            embedded_tokens,
                            raw,
                            memory,
                        )
                        if (
                            complement_result.event is not None
                            and not complement_result.unresolved
                        ):
                            matrix_result.event.discourse_role = (
                                "main"
                                if primary_event_count == 0
                                else "coordinate"
                            )
                            complement_result.event.discourse_role = "infinitive"
                            complement_result.event.source = SourceKind.ATTRIBUTED
                            complement_result.event.tense = "infinitive"
                            complement_result.event.certainty = min(
                                complement_result.event.certainty,
                                infinitival_split.certainty,
                            )
                            if connector is not None:
                                matrix_result.diagnostics.insert(
                                    0,
                                    f"coordinate connector={connector}",
                                )
                            primary_event_count += 1

                            matrix_index = len(events)
                            events.append(matrix_result.event)
                            complement_index = len(events)
                            events.append(complement_result.event)
                            infinitivals.append(
                                InfinitivalRelation(
                                    relation_type=(
                                        infinitival_split.profile.relation_type
                                    ),
                                    content_status=(
                                        infinitival_split.profile.content_status
                                    ),
                                    matrix_event_index=matrix_index,
                                    complement_event_index=complement_index,
                                    marker=infinitival_split.marker,
                                    matrix_predicate=(
                                        matrix_result.event.predicate
                                    ),
                                    source_entity_id=source_entity_id,
                                    controller_entity_id=controller_entity_id,
                                    embedded_subject_entity_id=(
                                        controller_entity_id
                                    ),
                                    predicate_family=(
                                        infinitival_split.profile.predicate_family
                                    ),
                                    certainty=infinitival_split.certainty,
                                    licensed=matrix_result.event.polarity,
                                    entailed=False,
                                    diagnostics=[
                                        *infinitival_split.diagnostics,
                                        (
                                            "positive matrix licenses the "
                                            "non-entailed complement relation"
                                            if matrix_result.event.polarity
                                            else "negated matrix does not "
                                            "license positive complement content"
                                        ),
                                    ],
                                )
                            )
                            unresolved.extend(matrix_result.unresolved)
                            entities.extend(matrix_result.entities)
                            entities.extend(complement_result.entities)
                            diagnostics.extend(matrix_result.diagnostics)
                            diagnostics.extend(complement_result.diagnostics)
                            diagnostics.extend(infinitival_split.diagnostics)
                            diagnostics.append(
                                "infinitival relation="
                                f"{infinitival_split.profile.relation_type.value} "
                                f"status={infinitival_split.profile.content_status.value} "
                                f"controller={controller_entity_id}"
                            )
                            continue

                        reason = (
                            "embedded infinitive could not be parsed with its "
                            "licensed controller"
                        )
                        ambiguity = self._infinitival_ambiguity(
                            clause,
                            [
                                len(infinitival_split.matrix_tokens)
                            ],
                            reason=reason,
                            candidate_types=[
                                infinitival_split.profile.relation_type
                            ],
                        )
                        infinitival_ambiguities.append(ambiguity)
                        unresolved.extend(complement_result.unresolved)
                        diagnostics.extend(complement_result.diagnostics)
                        diagnostics.extend(ambiguity.diagnostics)
                        continue

                    missing = (
                        "source"
                        if not source_entity_id
                        else "controller"
                    )
                    ambiguity = self._infinitival_ambiguity(
                        clause,
                        [len(infinitival_split.matrix_tokens)],
                        reason=(
                            f"matrix infinitival predicate lacks a resolved {missing} entity"
                        ),
                        candidate_types=[
                            infinitival_split.profile.relation_type
                        ],
                    )
                    infinitival_ambiguities.append(ambiguity)
                    unresolved.extend(matrix_result.unresolved)
                    diagnostics.extend(matrix_result.diagnostics)
                    diagnostics.extend(ambiguity.diagnostics)
                    continue

                diagnostics.extend(matrix_result.diagnostics)
                diagnostics.append(
                    f"infinitival split fallback marker={infinitival_split.marker}"
                )

            content_split, content_ambiguity = self._split_content_clause(clause)
            if content_ambiguity is not None:
                content_ambiguities.append(content_ambiguity)
                unresolved.append(
                    UnresolvedReference(
                        surface=content_ambiguity.content_surface,
                        reason=content_ambiguity.reason,
                    )
                )
                diagnostics.extend(content_ambiguity.diagnostics)
                diagnostics.append("content-clause boundary remains ambiguous")
                continue

            if content_split is not None:
                matrix_result = self._parse_clause(
                    content_split.matrix_tokens,
                    raw,
                    memory,
                )
                content_result = self._parse_clause(
                    content_split.content_tokens,
                    raw,
                    memory,
                )
                if matrix_result.event and content_result.event:
                    source_entity_id = self._content_source_entity(
                        matrix_result.event
                    )
                    if not source_entity_id:
                        ambiguity = ContentAttachmentAmbiguity(
                            matrix_surface=self._surface(content_split.matrix_tokens),
                            content_surface=self._surface(content_split.content_tokens),
                            clause_surface=self._surface(clause),
                            reason="matrix content predicate lacks a resolved source entity",
                            ambiguity_id=(
                                "content-"
                                + hashlib.sha256(
                                    " ".join(token.norm for token in clause).encode("utf-8")
                                ).hexdigest()[:16]
                            ),
                            diagnostics=[
                                "content source unresolved; durable assertion suppressed"
                            ],
                        )
                        content_ambiguities.append(ambiguity)
                        unresolved.extend(matrix_result.unresolved)
                        unresolved.extend(content_result.unresolved)
                        diagnostics.extend(ambiguity.diagnostics)
                        continue

                    matrix_result.event.discourse_role = (
                        "main" if primary_event_count == 0 else "coordinate"
                    )
                    content_result.event.discourse_role = "content"
                    content_result.event.source = SourceKind.ATTRIBUTED
                    content_result.event.certainty = min(
                        content_result.event.certainty,
                        content_split.certainty,
                    )
                    if connector is not None:
                        matrix_result.diagnostics.insert(
                            0,
                            f"coordinate connector={connector}",
                        )
                    primary_event_count += 1

                    matrix_index = len(events)
                    events.append(matrix_result.event)
                    content_index = len(events)
                    events.append(content_result.event)
                    contents.append(
                        ContentRelation(
                            relation_type=content_split.relation_type,
                            matrix_event_index=matrix_index,
                            content_event_index=content_index,
                            marker=content_split.marker,
                            matrix_predicate=matrix_result.event.predicate,
                            source_entity_id=source_entity_id,
                            predicate_family=content_split.predicate_family,
                            certainty=content_split.certainty,
                            attributed=matrix_result.event.polarity,
                            diagnostics=[
                                *content_split.diagnostics,
                                (
                                    "matrix predicate licenses attributed evidence"
                                    if matrix_result.event.polarity
                                    else "negated matrix predicate does not license attributed evidence"
                                ),
                            ],
                        )
                    )
                    unresolved.extend(matrix_result.unresolved)
                    unresolved.extend(content_result.unresolved)
                    entities.extend(matrix_result.entities)
                    entities.extend(content_result.entities)
                    diagnostics.extend(matrix_result.diagnostics)
                    diagnostics.extend(content_result.diagnostics)
                    diagnostics.extend(content_split.diagnostics)
                    diagnostics.append(
                        "content relation="
                        f"{content_split.relation_type.value} "
                        f"marker={content_split.marker} "
                        f"source={source_entity_id}"
                    )
                    continue

                diagnostics.extend(matrix_result.diagnostics)
                diagnostics.extend(content_result.diagnostics)
                diagnostics.append(
                    f"content split fallback marker={content_split.marker}"
                )

            relative_split, modifier_ambiguity = self._split_relative_clause(
                clause,
                raw,
                memory,
            )
            if modifier_ambiguity is not None:
                modifier_ambiguities.append(modifier_ambiguity)
                diagnostics.extend(modifier_ambiguity.diagnostics)
                diagnostics.append(
                    f"relative attachment ambiguous marker={modifier_ambiguity.marker}"
                )

            if relative_split is not None:
                main_result = self._parse_clause(
                    relative_split.main_tokens,
                    raw,
                    memory,
                )
                modifier_result = self._parse_clause(
                    relative_split.modifier_tokens,
                    raw,
                    memory,
                )
                if main_result.event and modifier_result.event:
                    main_result.event.discourse_role = (
                        "main" if primary_event_count == 0 else "coordinate"
                    )
                    modifier_result.event.discourse_role = "modifier"
                    if connector is not None:
                        main_result.diagnostics.insert(
                            0,
                            f"coordinate connector={connector}",
                        )
                    primary_event_count += 1

                    main_index = len(events)
                    events.append(main_result.event)
                    modifier_index = len(events)
                    events.append(modifier_result.event)
                    modifiers.append(
                        EntityModifierRelation(
                            head_entity_id=relative_split.head_entity_id,
                            modifier_event_index=modifier_index,
                            modifier_event_signature=self._event_signature_key(
                                modifier_result.event
                            ),
                            marker=relative_split.marker,
                            gap_role=relative_split.gap_role,
                            restriction=relative_split.restriction,
                            certainty=relative_split.certainty,
                            possessed_entity_id=relative_split.possessed_entity_id,
                            diagnostics=list(relative_split.diagnostics),
                        )
                    )
                    unresolved.extend(main_result.unresolved)
                    unresolved.extend(modifier_result.unresolved)
                    entities.extend(main_result.entities)
                    entities.extend(modifier_result.entities)
                    diagnostics.extend(main_result.diagnostics)
                    diagnostics.extend(modifier_result.diagnostics)
                    diagnostics.extend(relative_split.diagnostics)
                    diagnostics.append(
                        "entity modifier="
                        f"{relative_split.restriction.value} "
                        f"marker={relative_split.marker} "
                        f"gap={relative_split.gap_role.value}"
                    )
                    continue
                diagnostics.extend(main_result.diagnostics)
                diagnostics.extend(modifier_result.diagnostics)
                diagnostics.append(
                    f"relative split fallback marker={relative_split.marker}"
                )

            appositive_split, appositive_ambiguity = self._split_appositive_clause(
                clause,
                raw,
                memory,
            )
            if appositive_ambiguity is not None:
                appositive_ambiguities.append(appositive_ambiguity)
                unresolved.append(
                    UnresolvedReference(
                        surface=appositive_ambiguity.appositive_surface,
                        reason=appositive_ambiguity.reason,
                        candidates=list(appositive_ambiguity.candidate_entity_ids),
                    )
                )
                diagnostics.extend(appositive_ambiguity.diagnostics)
                diagnostics.append("appositive identity remains ambiguous")
                continue

            if appositive_split is not None:
                result = self._parse_clause(
                    appositive_split.main_tokens,
                    raw,
                    memory,
                )
                if result.event:
                    result.event.discourse_role = (
                        "main" if primary_event_count == 0 else "coordinate"
                    )
                    if connector is not None:
                        result.diagnostics.insert(
                            0,
                            f"coordinate connector={connector}",
                        )
                    primary_event_count += 1
                    events.append(result.event)
                    appositives.append(appositive_split.relation)
                    unresolved.extend(result.unresolved)
                    entities.extend(appositive_split.entity_ids)
                    entities.extend(result.entities)
                    diagnostics.extend(result.diagnostics)
                    diagnostics.extend(appositive_split.diagnostics)
                    diagnostics.append(
                        "appositive relation="
                        f"{appositive_split.relation.relation_type.value} "
                        f"restriction={appositive_split.relation.restriction.value}"
                    )
                    continue
                diagnostics.extend(result.diagnostics)
                diagnostics.append("appositive split fallback")

            subordinate_split = self._split_subordinate_clause(clause)
            if subordinate_split is not None:
                main_result = self._parse_clause(
                    subordinate_split.main_tokens,
                    raw,
                    memory,
                )
                subordinate_result = self._parse_clause(
                    subordinate_split.subordinate_tokens,
                    raw,
                    memory,
                )
                if main_result.event and subordinate_result.event:
                    main_result.event.discourse_role = (
                        "main" if primary_event_count == 0 else "coordinate"
                    )
                    subordinate_result.event.discourse_role = "subordinate"
                    if connector is not None:
                        main_result.diagnostics.insert(
                            0,
                            f"coordinate connector={connector}",
                        )
                    primary_event_count += 1

                    main_index = len(events)
                    events.append(main_result.event)
                    subordinate_index = len(events)
                    events.append(subordinate_result.event)

                    relation = ClauseRelation(
                        relation_type=subordinate_split.relation_type,
                        main_event_index=main_index,
                        subordinate_event_index=subordinate_index,
                        marker=subordinate_split.marker,
                        direction=subordinate_split.direction,
                        certainty=subordinate_split.certainty,
                        candidate_types=list(subordinate_split.candidate_types),
                        diagnostics=list(subordinate_split.diagnostics),
                    )
                    relations.append(relation)

                    subordinate_surface = self._surface(
                        subordinate_split.subordinate_tokens
                    )
                    subordinate_key = lexicon.normalize_phrase(
                        token.norm
                        for token in subordinate_split.subordinate_tokens
                    )
                    if relation.relation_type == ClauseRelationType.CAUSE:
                        cause_role = (
                            "motive"
                            if main_result.event.predicate in lexicon.VOLITIONAL_VERBS
                            else "cause"
                        )
                        cause_ref = SemanticRef.literal(
                            subordinate_key,
                            subordinate_surface,
                            EntityKind.ABSTRACT,
                        )
                        main_result.event.arguments[cause_role] = cause_ref
                        main_result.event.arguments.setdefault("cause", cause_ref)
                    elif relation.relation_type == ClauseRelationType.PURPOSE:
                        main_result.event.arguments["purpose"] = SemanticRef.literal(
                            subordinate_key,
                            subordinate_surface,
                            EntityKind.ABSTRACT,
                        )

                    unresolved.extend(main_result.unresolved)
                    unresolved.extend(subordinate_result.unresolved)
                    entities.extend(main_result.entities)
                    entities.extend(subordinate_result.entities)
                    diagnostics.extend(main_result.diagnostics)
                    diagnostics.extend(subordinate_result.diagnostics)
                    diagnostics.extend(subordinate_split.diagnostics)
                    diagnostics.append(
                        "clause relation="
                        f"{relation.relation_type.value} "
                        f"marker={relation.marker} "
                        f"direction={relation.direction.value}"
                    )
                    continue

                diagnostics.extend(main_result.diagnostics)
                diagnostics.extend(subordinate_result.diagnostics)
                diagnostics.append(
                    f"subordinate split fallback marker={subordinate_split.marker}"
                )

            result = self._parse_clause(clause, raw, memory)
            if result.event:
                result.event.discourse_role = (
                    "main" if primary_event_count == 0 else "coordinate"
                )
                if connector is not None:
                    result.diagnostics.insert(0, f"coordinate connector={connector}")
                primary_event_count += 1
                events.append(result.event)
            unresolved.extend(result.unresolved)
            entities.extend(result.entities)
            diagnostics.extend(result.diagnostics)
        return ParseResult(
            speech_act=SpeechAct.ASSERT if events else SpeechAct.UNKNOWN,
            raw_text=raw,
            events=events,
            relations=relations,
            modifiers=modifiers,
            modifier_ambiguities=modifier_ambiguities,
            appositives=appositives,
            appositive_ambiguities=appositive_ambiguities,
            contents=contents,
            content_ambiguities=content_ambiguities,
            infinitivals=infinitivals,
            infinitival_ambiguities=infinitival_ambiguities,
            entities=list(dict.fromkeys(entities)),
            unresolved=unresolved,
            normalized_text=normalized,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Top-level transformations
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_social(words: Sequence[str]) -> Optional[str]:
        compact = tuple(word for word in words if word not in {",", ":"})
        for pattern, name in lexicon.SOCIAL_QUESTIONS.items():
            if compact[: len(pattern)] == pattern and len(compact) <= len(pattern) + 2:
                return name
        if len(compact) <= 3 and compact[0] in lexicon.GREETINGS:
            return "greeting"
        return None

    def _split_assertion_segments(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> List[Tuple[List[lexicon.Token], Optional[str]]]:
        """Split only independently finite coordinated assertions.

        A conjunction is not enough evidence by itself: compound subjects,
        compound objects, and shared-subject/gapping coordination remain in one
        clause.  Each returned connector belongs to the clause on its right.
        """

        segments: List[Tuple[List[lexicon.Token], Optional[str]]] = []
        current: List[lexicon.Token] = []
        incoming_connector: Optional[str] = None
        index = 0
        while index < len(tokens):
            token = tokens[index]
            norm = token.norm
            connector: Optional[str] = None
            right_start = index + 1

            if norm == ";":
                connector = ";"
            elif norm in {"and", "but", "yet", "or", "so"}:
                # ``so that`` introduces a subordinate purpose/result clause,
                # not the independently finite resultative coordination covered
                # by this slice.
                if norm == "so" and right_start < len(tokens) and tokens[right_start].norm == "that":
                    current.append(token)
                    index += 1
                    continue
                if norm == "and" and right_start < len(tokens) and tokens[right_start].norm == "then":
                    connector = "and then"
                    right_start += 1
                else:
                    connector = norm

            if connector is not None:
                right = self._immediate_right_clause(tokens, right_start)
                if self._is_independently_finite(current) and self._is_independently_finite(right):
                    # Preserve commas until subordinate-clause analysis;
                    # ``_parse_clause`` removes punctuation after structural
                    # boundaries have consumed it.
                    cleaned_current = [
                        item for item in current if item.norm != ";"
                    ]
                    if cleaned_current:
                        segments.append((cleaned_current, incoming_connector))
                    current = []
                    incoming_connector = connector
                    index = right_start
                    continue

            current.append(token)
            index += 1

        cleaned_current = [item for item in current if item.norm != ";"]
        if cleaned_current:
            segments.append((cleaned_current, incoming_connector))
        return segments or [(list(tokens), None)]

    def _split_assertion_clauses(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> List[List[lexicon.Token]]:
        """Compatibility wrapper returning token lists without connectors."""

        return [clause for clause, _connector in self._split_assertion_segments(tokens)]

    @staticmethod
    def _immediate_right_clause(
        tokens: Sequence[lexicon.Token],
        start: int,
    ) -> List[lexicon.Token]:
        """Return lookahead only through the next coordination boundary.

        Without this bound, an early compound-object ``and`` could borrow a
        finite verb from a later ``but John left`` clause and split at the
        wrong connector.
        """

        end = len(tokens)
        for index in range(start, len(tokens)):
            norm = tokens[index].norm
            if norm == ";":
                end = index
                break
            if norm in {"and", "but", "yet", "or", "so"}:
                if (
                    norm == "so"
                    and index + 1 < len(tokens)
                    and tokens[index + 1].norm == "that"
                ):
                    continue
                end = index
                break
        return [
            item
            for item in tokens[start:end]
            if item.norm not in lexicon.PUNCTUATION
        ]

    def _is_independently_finite(self, tokens: Sequence[lexicon.Token]) -> bool:
        """Return whether a segment contains a finite predicate and subject."""

        items = [item for item in tokens if item.norm not in lexicon.PUNCTUATION]
        verb_index = self._find_main_verb(items)
        if verb_index <= 0:
            # A verb-initial segment may be an imperative or shared-subject
            # continuation.  Those require their own deterministic slice and
            # are deliberately not split here.
            return False
        subject_tokens = [
            item
            for item in items[:verb_index]
            if item.norm not in lexicon.AUXILIARIES
            and item.norm not in lexicon.NEGATORS
            and item.norm not in lexicon.INTENSIFIERS
            and item.norm not in {",", ";"}
        ]
        return bool(subject_tokens)

    INFINITIVAL_PREDICATES: Dict[str, InfinitivalPredicateProfile] = {
        "plan": InfinitivalPredicateProfile(
            InfinitivalRelationType.SUBJECT_CONTROL,
            InfinitivalContentStatus.PLANNED,
            "planning",
            215,
        ),
        "intend": InfinitivalPredicateProfile(
            InfinitivalRelationType.SUBJECT_CONTROL,
            InfinitivalContentStatus.INTENDED,
            "intention",
            220,
        ),
        "hope": InfinitivalPredicateProfile(
            InfinitivalRelationType.SUBJECT_CONTROL,
            InfinitivalContentStatus.HOPED,
            "hope",
            190,
        ),
        "want": InfinitivalPredicateProfile(
            InfinitivalRelationType.SUBJECT_CONTROL,
            InfinitivalContentStatus.DESIRED,
            "desire",
            195,
            allows_object_controller=True,
        ),
        "tell": InfinitivalPredicateProfile(
            InfinitivalRelationType.OBJECT_CONTROL,
            InfinitivalContentStatus.DIRECTED,
            "directive",
            210,
            allows_object_controller=True,
            allows_subject_controller=False,
        ),
        "ask": InfinitivalPredicateProfile(
            InfinitivalRelationType.OBJECT_CONTROL,
            InfinitivalContentStatus.REQUESTED,
            "request",
            205,
            allows_object_controller=True,
            allows_subject_controller=False,
        ),
        "seem": InfinitivalPredicateProfile(
            InfinitivalRelationType.RAISING,
            InfinitivalContentStatus.EVIDENTIAL,
            "appearance",
            170,
        ),
        "appear": InfinitivalPredicateProfile(
            InfinitivalRelationType.RAISING,
            InfinitivalContentStatus.EVIDENTIAL,
            "appearance",
            175,
        ),
    }

    def _split_infinitival_clause(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> Tuple[
        Optional[InfinitivalSplit],
        Optional[InfinitivalAttachmentAmbiguity],
    ]:
        """Split one catalog-licensed selected ``to`` complement.

        Movement/purpose clauses deliberately remain in the pre-existing
        purpose-adjunct path.  This splitter activates only when a reviewed
        matrix predicate licenses control or raising.
        """

        items = [item for item in tokens if item.norm != ";"]

        # Relation layers can place another finite verb before the selected
        # infinitival matrix predicate (for example, a relative clause in
        # ``the woman who called plans to leave``).  Inspect every licensed
        # predicate that has a later infinitival boundary before relying on
        # the ordinary main-verb heuristic.  Until staged composition exists,
        # such inputs must fail closed instead of letting the earlier modifier
        # verb steal the entire frame.
        licensed_candidates: List[int] = []
        for candidate_index, token in enumerate(items):
            if lexicon.lemma(token.norm) not in self.INFINITIVAL_PREDICATES:
                continue
            if any(
                items[index].norm == "to"
                and index + 1 < len(items)
                and lexicon.is_probable_verb(items[index + 1].norm, "to")
                for index in range(candidate_index + 1, len(items) - 1)
            ):
                licensed_candidates.append(candidate_index)
        for candidate_index in licensed_candidates:
            layered = self._infinitival_layering_ambiguity(
                items,
                candidate_index,
            )
            if layered is not None:
                return None, layered

        verb_index = self._find_main_verb(items)
        if verb_index < 0:
            return None, None
        predicate = lexicon.lemma(items[verb_index].norm)
        profile = self.INFINITIVAL_PREDICATES.get(predicate)
        if profile is None:
            return None, None

        boundaries: List[int] = []
        for index in range(verb_index + 1, len(items) - 1):
            if items[index].norm != "to":
                continue
            next_index = index + 1
            while (
                next_index < len(items)
                and items[next_index].norm
                in lexicon.NEGATORS | lexicon.INTENSIFIERS
            ):
                next_index += 1
            if next_index >= len(items):
                continue
            next_word = items[next_index].norm
            if lexicon.is_probable_verb(next_word, "to"):
                boundaries.append(index)

        if not boundaries:
            return None, None
        if len(boundaries) > 1:
            return None, self._infinitival_ambiguity(
                items,
                boundaries,
                reason="nested infinitival content exceeds the configured depth",
                candidate_types=[profile.relation_type],
            )

        boundary = boundaries[0]
        matrix_tail = [
            item
            for item in items[verb_index + 1 : boundary]
            if item.norm not in lexicon.PUNCTUATION
        ]
        embedded_prefix: List[lexicon.Token] = []
        if matrix_tail and matrix_tail[-1].norm == "not":
            # ``plans not to leave`` and ``told John not to leave`` place
            # negation inside the infinitive rather than on the matrix event.
            embedded_prefix.append(matrix_tail.pop())

        has_object_controller = bool(matrix_tail)
        if has_object_controller:
            if not profile.allows_object_controller:
                return None, self._infinitival_ambiguity(
                    items,
                    [boundary],
                    reason=(
                        f"{predicate} does not license an explicit object "
                        "controller in this parser slice"
                    ),
                    candidate_types=[profile.relation_type],
                )
            if not self._plausible_controller_np(matrix_tail):
                return None, self._infinitival_ambiguity(
                    items,
                    [boundary],
                    reason="object-controller phrase is structurally unsupported",
                    candidate_types=[InfinitivalRelationType.OBJECT_CONTROL],
                )
            relation_type = InfinitivalRelationType.OBJECT_CONTROL
            controller_role = "object"
        else:
            if not profile.allows_subject_controller:
                return None, self._infinitival_ambiguity(
                    items,
                    [boundary],
                    reason=f"{predicate} requires an explicit object controller",
                    candidate_types=[InfinitivalRelationType.OBJECT_CONTROL],
                )
            relation_type = profile.relation_type
            controller_role = "subject"

        selected_profile = InfinitivalPredicateProfile(
            relation_type=relation_type,
            content_status=profile.content_status,
            predicate_family=profile.predicate_family,
            certainty=profile.certainty,
            allows_object_controller=profile.allows_object_controller,
            allows_subject_controller=profile.allows_subject_controller,
        )
        matrix = [
            item
            for item in items[:boundary]
            if item not in embedded_prefix and item.norm not in lexicon.PUNCTUATION
        ]
        complement = [
            *embedded_prefix,
            *[
                item
                for item in items[boundary + 1 :]
                if item.norm not in lexicon.PUNCTUATION
            ],
        ]
        if self._contains_nested_content_clause(complement):
            return None, self._infinitival_ambiguity(
                items,
                [boundary],
                reason=(
                    "infinitival content containing a finite content clause "
                    "exceeds the configured depth"
                ),
                candidate_types=[relation_type],
            )
        return (
            InfinitivalSplit(
                matrix_tokens=matrix,
                complement_tokens=complement,
                marker="to",
                profile=selected_profile,
                controller_role=controller_role,
                certainty=selected_profile.certainty,
                diagnostics=[
                    f"infinitival matrix predicate={predicate}",
                    f"infinitival relation={relation_type.value}",
                    f"infinitival status={profile.content_status.value}",
                    f"infinitival boundary={boundary}",
                ],
            ),
            None,
        )

    def _infinitival_layering_ambiguity(
        self,
        tokens: Sequence[lexicon.Token],
        matrix_verb_index: int,
    ) -> Optional[InfinitivalAttachmentAmbiguity]:
        norms = [item.norm for item in tokens]
        coordination_markers = [
            index
            for index, norm in enumerate(norms)
            if norm in {"and", "but", "yet", "or", "so"}
        ]
        if coordination_markers:
            return self._infinitival_ambiguity(
                tokens,
                coordination_markers,
                reason=(
                    "infinitival content combined with coordination requires "
                    "staged parsing"
                ),
            )
        relative_markers = [
            index
            for index, norm in enumerate(norms)
            if norm in {"who", "whom", "whose", "which"}
            or (norm == "that" and index < matrix_verb_index)
        ]
        if relative_markers:
            return self._infinitival_ambiguity(
                tokens,
                [matrix_verb_index + 1],
                reason=(
                    "infinitival content combined with a relative clause "
                    "requires staged parsing"
                ),
            )
        commas_before = [
            index for index in range(matrix_verb_index) if norms[index] == ","
        ]
        if len(commas_before) >= 2:
            return self._infinitival_ambiguity(
                tokens,
                [matrix_verb_index + 1],
                reason=(
                    "infinitival content combined with an appositive requires "
                    "staged parsing"
                ),
            )
        for index in range(len(tokens)):
            for marker in self.SUBORDINATE_MARKERS:
                if tuple(norms[index : index + len(marker)]) == marker:
                    return self._infinitival_ambiguity(
                        tokens,
                        [matrix_verb_index + 1],
                        reason=(
                            "infinitival content combined with a subordinate "
                            "clause requires staged parsing"
                        ),
                    )
        return None

    @staticmethod
    def _plausible_controller_np(
        tokens: Sequence[lexicon.Token],
    ) -> bool:
        if not tokens:
            return False
        norms = [item.norm for item in tokens]
        if any(item in lexicon.PREPOSITIONS for item in norms):
            return False
        if any(
            lexicon.is_probable_verb(
                item.norm,
                previous=(tokens[index - 1].norm if index else None),
                following=(
                    tokens[index + 1].norm
                    if index + 1 < len(tokens)
                    else None
                ),
            )
            for index, item in enumerate(tokens)
        ):
            return False
        return True

    @staticmethod
    def _infinitival_controller_entity(
        event: EventFrame,
        controller_role: str,
    ) -> str:
        roles = (
            ("patient", "recipient")
            if controller_role == "object"
            else ("agent", "experiencer", "subject", "possessor", "patient")
        )
        for role in roles:
            ref = event.arguments.get(role)
            if ref is not None and ref.kind == RefKind.ENTITY:
                return ref.key
        return ""

    def _infinitival_ambiguity(
        self,
        tokens: Sequence[lexicon.Token],
        boundaries: Sequence[int],
        *,
        reason: str,
        candidate_types: Sequence[InfinitivalRelationType] = (),
    ) -> InfinitivalAttachmentAmbiguity:
        first = boundaries[0] if boundaries else len(tokens)
        return InfinitivalAttachmentAmbiguity(
            matrix_surface=self._surface(tokens[:first]),
            complement_surface=self._surface(tokens[first:]),
            clause_surface=self._surface(tokens),
            reason=reason,
            candidate_boundaries=list(boundaries),
            candidate_relation_types=list(candidate_types),
            ambiguity_id=(
                "infinitive-"
                + hashlib.sha256(
                    " ".join(item.norm for item in tokens).encode("utf-8")
                ).hexdigest()[:16]
            ),
            diagnostics=[reason, "unsafe infinitival content suppressed"],
        )

    CONTENT_PREDICATES: Dict[str, Tuple[ContentRelationType, str, int]] = {
        "say": (ContentRelationType.REPORTED, "speech", 205),
        "tell": (ContentRelationType.REPORTED, "speech", 205),
        "report": (ContentRelationType.REPORTED, "speech", 215),
        "claim": (ContentRelationType.REPORTED, "speech", 190),
        "think": (ContentRelationType.BELIEVED, "belief", 185),
        "believe": (ContentRelationType.BELIEVED, "belief", 185),
        "know": (ContentRelationType.KNOWN, "knowledge", 210),
        "notice": (ContentRelationType.PERCEIVED, "perception", 200),
        "hear": (ContentRelationType.PERCEIVED, "perception", 195),
    }
    CONTENT_RECIPIENT_PREDICATES = {"tell"}

    def _split_content_clause(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> Tuple[Optional[ContentSplit], Optional[ContentAttachmentAmbiguity]]:
        """Split one licensed finite content complement conservatively.

        This bounded slice intentionally refuses to flatten a content clause
        together with another relation layer.  Relative modifiers, appositives,
        and finite subordination are each represented by their own typed
        structures elsewhere in the parser; until staged composition is
        implemented, combining them here would create false source identities
        or swallow an outer matrix predicate.
        """

        items = [item for item in tokens if item.norm not in {";"}]
        layered_ambiguity = self._content_layering_ambiguity(items)
        if layered_ambiguity is not None:
            return None, layered_ambiguity

        verb_index = self._find_main_verb(items)
        if verb_index < 0:
            return None, None
        predicate = lexicon.lemma(items[verb_index].norm)
        profile = self.CONTENT_PREDICATES.get(predicate)
        if profile is None:
            return None, None
        relation_type, family, certainty = profile

        explicit = [
            index
            for index in range(verb_index + 1, len(items))
            if items[index].norm == "that"
        ]
        if explicit:
            marker_index = explicit[0]
            if len(explicit) > 1:
                return None, self._content_ambiguity(
                    items,
                    [index + 1 for index in explicit],
                    reason="multiple complementizer candidates require nested parsing",
                )
            matrix_tail = [
                item
                for item in items[verb_index + 1 : marker_index]
                if item.norm not in lexicon.PUNCTUATION
            ]
            if matrix_tail and predicate not in self.CONTENT_RECIPIENT_PREDICATES:
                # A substantive noun phrase before ``that`` is more likely the
                # head of a relative clause (``reported the idea that ...``).
                return None, None
            content = [
                item
                for item in items[marker_index + 1 :]
                if item.norm not in lexicon.PUNCTUATION
            ]
            matrix = [
                item
                for item in items[:marker_index]
                if item.norm not in lexicon.PUNCTUATION
            ]
            if not self._is_independently_finite(content):
                return None, None
            if self._contains_nested_content_clause(content):
                return None, self._content_ambiguity(
                    items,
                    [marker_index + 1],
                    reason="nested finite content exceeds the configured depth",
                )
            if self._contains_selected_infinitive(content):
                return None, self._content_ambiguity(
                    items,
                    [marker_index + 1],
                    reason=(
                        "finite content containing an infinitival complement "
                        "exceeds the configured depth"
                    ),
                )
            return (
                ContentSplit(
                    matrix_tokens=matrix,
                    content_tokens=content,
                    marker="that",
                    relation_type=relation_type,
                    predicate_family=family,
                    certainty=certainty,
                    diagnostics=[
                        f"finite content predicate={predicate}",
                        "content marker=that",
                    ],
                ),
                None,
            )

        candidates: List[int] = []
        for boundary in range(verb_index + 1, len(items)):
            content = [
                item
                for item in items[boundary:]
                if item.norm not in lexicon.PUNCTUATION
            ]
            if not self._is_independently_finite(content):
                continue
            if not self._plausible_content_subject(content):
                continue
            matrix_tail = [
                item
                for item in items[verb_index + 1 : boundary]
                if item.norm not in lexicon.PUNCTUATION
            ]
            if matrix_tail and predicate not in self.CONTENT_RECIPIENT_PREDICATES:
                continue
            candidates.append(boundary)

        if not candidates:
            return None, None
        if predicate in self.CONTENT_RECIPIENT_PREDICATES:
            # ``tell`` licenses a recipient before zero-marked content. Prefer
            # the boundary that keeps one recipient NP with the matrix clause.
            recipient_candidates = [
                boundary
                for boundary in candidates
                if boundary > verb_index + 1
            ]
            if recipient_candidates:
                candidates = recipient_candidates
        if len(candidates) != 1:
            return None, self._content_ambiguity(
                items,
                candidates,
                reason="multiple finite zero-complementizer boundaries remain plausible",
            )

        boundary = candidates[0]
        matrix = [
            item for item in items[:boundary] if item.norm not in lexicon.PUNCTUATION
        ]
        content = [
            item for item in items[boundary:] if item.norm not in lexicon.PUNCTUATION
        ]
        if self._contains_nested_content_clause(content):
            return None, self._content_ambiguity(
                items,
                [boundary],
                reason="nested finite content exceeds the configured depth",
            )
        if self._contains_selected_infinitive(content):
            return None, self._content_ambiguity(
                items,
                [boundary],
                reason=(
                    "finite content containing an infinitival complement "
                    "exceeds the configured depth"
                ),
            )
        return (
            ContentSplit(
                matrix_tokens=matrix,
                content_tokens=content,
                marker="zero",
                relation_type=relation_type,
                predicate_family=family,
                certainty=max(0, certainty - 10),
                diagnostics=[
                    f"finite content predicate={predicate}",
                    "content marker=zero",
                    f"content boundary={boundary}",
                ],
            ),
            None,
        )

    def _content_layering_ambiguity(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> Optional[ContentAttachmentAmbiguity]:
        """Reject unsupported combinations of content and relation layers.

        The check scans for any licensed finite content predicate rather than
        only the first predicate.  That matters for forms such as ``The man
        who left said that John arrived`` and fronted subordinate clauses,
        where the first finite verb belongs to a different relation layer.
        """

        items = list(tokens)
        content_predicates: List[int] = []
        for index, token in enumerate(items):
            predicate = lexicon.lemma(token.norm)
            if predicate not in self.CONTENT_PREDICATES:
                continue
            previous = items[index - 1].norm if index > 0 else None
            if lexicon.is_probable_verb(token.norm, previous=previous):
                content_predicates.append(index)
        if not content_predicates:
            return None

        first_predicate = content_predicates[0]
        norms = [item.norm for item in items]
        relative_markers = [
            index
            for index, norm in enumerate(norms)
            if norm in {"who", "whom", "whose", "which"}
            or (norm == "that" and index < first_predicate)
        ]
        if relative_markers:
            return self._content_ambiguity(
                items,
                [first_predicate + 1],
                reason=(
                    "finite content combined with a relative clause exceeds "
                    "this parser slice"
                ),
            )

        # A paired comma before the content predicate is the structural shape
        # of the supported appositive layer (for example ``Sarah, my
        # supervisor, said ...``).  Parsing it as a flat matrix subject would
        # invent a composite identity, so fail closed until staged composition.
        commas_before = [
            index
            for index in range(first_predicate)
            if norms[index] == ","
        ]
        if len(commas_before) >= 2:
            return self._content_ambiguity(
                items,
                [first_predicate + 1],
                reason=(
                    "finite content combined with an appositive requires "
                    "staged parsing"
                ),
            )

        # Any reviewed subordinate marker in the same segment creates a second
        # finite relation layer.  Preserve it as an explicit ambiguity rather
        # than treating the embedded proposition as a direct object string.
        for index in range(len(items)):
            for marker in self.SUBORDINATE_MARKERS:
                if tuple(norms[index : index + len(marker)]) == marker:
                    return self._content_ambiguity(
                        items,
                        [first_predicate + 1],
                        reason=(
                            "finite content combined with a subordinate clause "
                            "requires staged parsing"
                        ),
                    )
        return None

    def _contains_nested_content_clause(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> bool:
        verb_index = self._find_main_verb(tokens)
        if verb_index < 0:
            return False
        predicate = lexicon.lemma(tokens[verb_index].norm)
        if predicate not in self.CONTENT_PREDICATES:
            return False
        for index in range(verb_index + 1, len(tokens)):
            if tokens[index].norm == "that":
                suffix = [
                    item
                    for item in tokens[index + 1 :]
                    if item.norm not in lexicon.PUNCTUATION
                ]
                if self._is_independently_finite(suffix):
                    return True
            suffix = [
                item
                for item in tokens[index:]
                if item.norm not in lexicon.PUNCTUATION
            ]
            if self._is_independently_finite(suffix):
                return True
        return False

    def _contains_selected_infinitive(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> bool:
        items = [item for item in tokens if item.norm not in lexicon.PUNCTUATION]
        verb_index = self._find_main_verb(items)
        if verb_index < 0:
            return False
        predicate = lexicon.lemma(items[verb_index].norm)
        if predicate not in self.INFINITIVAL_PREDICATES:
            return False
        for index in range(verb_index + 1, len(items) - 1):
            if items[index].norm != "to":
                continue
            next_index = index + 1
            while (
                next_index < len(items)
                and items[next_index].norm
                in lexicon.NEGATORS | lexicon.INTENSIFIERS
            ):
                next_index += 1
            if (
                next_index < len(items)
                and lexicon.is_probable_verb(items[next_index].norm, "to")
            ):
                return True
        return False

    def _content_ambiguity(
        self,
        tokens: Sequence[lexicon.Token],
        boundaries: Sequence[int],
        *,
        reason: str,
    ) -> ContentAttachmentAmbiguity:
        first = boundaries[0] if boundaries else len(tokens)
        return ContentAttachmentAmbiguity(
            matrix_surface=self._surface(tokens[:first]),
            content_surface=self._surface(tokens[first:]),
            clause_surface=self._surface(tokens),
            reason=reason,
            candidate_boundaries=list(boundaries),
            ambiguity_id=(
                "content-"
                + hashlib.sha256(
                    " ".join(token.norm for token in tokens).encode("utf-8")
                ).hexdigest()[:16]
            ),
            diagnostics=[reason, "unsafe attributed content suppressed"],
        )

    def _plausible_content_subject(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> bool:
        verb_index = self._find_main_verb(tokens)
        if verb_index <= 0:
            return False
        subject = [
            item
            for item in tokens[:verb_index]
            if item.norm not in lexicon.AUXILIARIES
            and item.norm not in lexicon.NEGATORS
            and item.norm not in lexicon.INTENSIFIERS
        ]
        if not subject:
            return False
        first = subject[0].norm
        if first in lexicon.PRONOUN_FEATURES and len(subject) > 1:
            return False
        return True

    @staticmethod
    def _content_source_entity(event: EventFrame) -> str:
        for role in ("agent", "experiencer", "subject", "source"):
            ref = event.arguments.get(role)
            if ref is not None and ref.kind == RefKind.ENTITY:
                return ref.key
        return ""

    RELATIVE_MARKERS = {"who", "whom", "whose", "which", "that"}
    ABSTRACT_COMPLEMENT_HEADS = {
        "belief", "claim", "fact", "hope", "idea", "news", "reason",
        "report", "story", "thought",
    }
    PERSON_RELATIVE_HEADS = {
        "adult", "boy", "child", "doctor", "driver", "girl", "man",
        "nurse", "person", "student", "teacher", "technician", "woman",
        "worker",
    }
    FEMALE_RELATIVE_HEADS = {"girl", "mother", "nurse", "sister", "woman"}
    MALE_RELATIVE_HEADS = {"boy", "brother", "father", "man"}
    APPOSITIVE_ROLE_NOUNS = {
        "advisor", "boss", "coach", "colleague", "coworker", "doctor",
        "friend", "manager", "mentor", "nurse", "partner", "professor",
        "supervisor", "teacher", "therapist", "worker",
    }

    def _split_relative_clause(
        self,
        tokens: Sequence[lexicon.Token],
        raw: str,
        memory: ConversationMemory,
    ) -> Tuple[Optional[RelativeSplit], Optional[ModifierAttachmentAmbiguity]]:
        """Split one finite relative modifier without attachment guessing."""

        items = list(tokens)
        marker_indices = [
            index
            for index, token in enumerate(items)
            if token.norm in self.RELATIVE_MARKERS
            and index > 0
            and not (
                token.norm == "that"
                and items[index - 1].norm == "so"
            )
        ]
        if not marker_indices:
            return None, None
        if len(marker_indices) > 1:
            surfaces = []
            for index in marker_indices:
                candidate = self._surface(
                    item for item in items[:index] if item.norm not in {",", ";"}
                )
                if candidate and candidate not in surfaces:
                    surfaces.append(candidate)
            ambiguity = ModifierAttachmentAmbiguity(
                marker=items[marker_indices[0]].norm,
                clause_surface=self._surface(items),
                candidate_head_surfaces=surfaces,
                reason="multiple finite relative markers require attachment resolution",
                ambiguity_id=(
                    "relative-"
                    + hashlib.sha256(" ".join(token.norm for token in items).encode("utf-8")).hexdigest()[:16]
                ),
                diagnostics=["multiple relative markers; durable assertion suppressed"],
            )
            return None, ambiguity

        marker_index = marker_indices[0]
        marker = items[marker_index].norm
        pre = list(items[:marker_index])
        if not pre:
            return None, None

        restriction = (
            ModifierRestriction.NONRESTRICTIVE
            if pre and pre[-1].norm == ","
            else ModifierRestriction.RESTRICTIVE
        )
        pre_without_comma = [item for item in pre if item.norm not in {",", ";"}]
        outer_verb_before = self._find_main_verb(pre_without_comma)

        closing_comma = -1
        if restriction == ModifierRestriction.NONRESTRICTIVE:
            closing_comma = next(
                (
                    index
                    for index in range(marker_index + 1, len(items))
                    if items[index].norm == ","
                ),
                -1,
            )
            if closing_comma < 0:
                return None, ModifierAttachmentAmbiguity(
                    marker=marker,
                    clause_surface=self._surface(items),
                    candidate_head_surfaces=[self._surface(pre_without_comma)],
                    reason="nonrestrictive relative clause lacks a closing comma",
                    ambiguity_id=(
                        "relative-"
                        + hashlib.sha256(" ".join(token.norm for token in items).encode("utf-8")).hexdigest()[:16]
                    ),
                    diagnostics=["unterminated nonrestrictive relative clause"],
                )
            relative_body = [
                item
                for item in items[marker_index + 1 : closing_comma]
                if item.norm not in {",", ";"}
            ]
            post_relative = [
                item
                for item in items[closing_comma + 1 :]
                if item.norm not in {",", ";"}
            ]
        else:
            relative_body = [
                item
                for item in items[marker_index + 1 :]
                if item.norm not in {",", ";"}
            ]
            post_relative = []

        main_tail: List[lexicon.Token] = []
        if outer_verb_before >= 0:
            head_start = outer_verb_before + 1
            if (
                head_start < len(pre_without_comma)
                and pre_without_comma[head_start].norm in lexicon.PREPOSITIONS
            ):
                head_start += 1
            head_tokens = pre_without_comma[head_start:]
            main_prefix = pre_without_comma[:head_start]
            if not head_tokens:
                return None, None
            if self._relative_head_looks_like_complement(head_tokens):
                return None, None
            if restriction == ModifierRestriction.RESTRICTIVE:
                main_tail = []
            else:
                main_tail = post_relative
        else:
            head_tokens = pre_without_comma
            main_prefix = []
            if self._relative_head_looks_like_complement(head_tokens):
                return None, None
            if restriction == ModifierRestriction.RESTRICTIVE:
                verb_indices = self._finite_verb_indices(relative_body)
                if len(verb_indices) < 2:
                    return None, None
                outer_boundary = verb_indices[-1]
                main_tail = relative_body[outer_boundary:]
                relative_body = relative_body[:outer_boundary]
            else:
                main_tail = post_relative

        if not head_tokens or not relative_body or not main_tail and not main_prefix:
            return None, None

        gap_role = self._relative_gap_role(marker, relative_body)
        if gap_role is None:
            return None, ModifierAttachmentAmbiguity(
                marker=marker,
                clause_surface=self._surface(items),
                candidate_head_surfaces=[self._surface(head_tokens)],
                reason="relative clause lacks a resolvable finite gap role",
                ambiguity_id=(
                    "relative-"
                    + hashlib.sha256(
                        " ".join(token.norm for token in items).encode("utf-8")
                    ).hexdigest()[:16]
                ),
                diagnostics=[
                    "relative gap unresolved; durable assertion suppressed"
                ],
            )
        signature = lexicon.normalize_phrase(
            [marker, gap_role.value]
            + [token.norm for token in relative_body]
        )
        head_surface = self._surface(head_tokens)
        kind, gender, number = self._relative_head_features(head_tokens)

        if restriction == ModifierRestriction.RESTRICTIVE:
            head_entity, internal_alias = memory.get_or_create_modified_entity(
                head_surface,
                signature,
                kind=kind,
                gender=gender,
                number=number,
            )
        else:
            head_np = self._parse_np(
                head_tokens,
                memory,
                expected_kind=kind,
                role_hint="subject",
            )
            if not head_np.ref or head_np.ref.kind != RefKind.ENTITY:
                return None, None
            head_entity = memory.get_entity(head_np.ref.key)
            if head_entity is None:
                return None, None
            internal_alias = memory.ensure_internal_alias(head_entity.entity_id)

        head_token = lexicon.Token(head_surface, internal_alias, -1)
        main_tokens = main_prefix + [head_token] + main_tail
        possessed_entity_id = ""

        if gap_role == ModifierGapRole.AGENT:
            modifier_tokens = [head_token] + relative_body
        elif gap_role == ModifierGapRole.PATIENT:
            verb_index = self._find_main_verb(relative_body)
            if verb_index < 0:
                return None, None
            verb_and_tail = list(relative_body[verb_index:])
            insertion = self._object_insertion_index(verb_and_tail)
            modifier_tokens = (
                list(relative_body[:verb_index])
                + verb_and_tail[:insertion]
                + [head_token]
                + verb_and_tail[insertion:]
            )
        else:
            verb_index = self._find_main_verb(relative_body)
            if verb_index <= 0:
                return None, None
            possessed_tokens = relative_body[:verb_index]
            possessed_surface = self._surface(possessed_tokens)
            possessed_kind = lexicon.classify_unknown_noun(
                [token.norm for token in possessed_tokens]
            )
            possessed, possessed_alias = memory.get_or_create_possessed_entity(
                head_entity.entity_id,
                possessed_surface,
                signature,
                kind=possessed_kind,
            )
            possessed_entity_id = possessed.entity_id
            possessed_token = lexicon.Token(
                possessed_surface,
                possessed_alias,
                -1,
            )
            modifier_tokens = [possessed_token] + list(relative_body[verb_index:])

        return (
            RelativeSplit(
                main_tokens=main_tokens,
                modifier_tokens=modifier_tokens,
                head_entity_id=head_entity.entity_id,
                marker=marker,
                gap_role=gap_role,
                restriction=restriction,
                possessed_entity_id=possessed_entity_id,
                diagnostics=[
                    f"relative marker={marker}",
                    f"relative restriction={restriction.value}",
                    f"relative gap={gap_role.value}",
                    f"relative head={head_surface}",
                ],
            ),
            None,
        )


    def _split_appositive_clause(
        self,
        tokens: Sequence[lexicon.Token],
        raw: str,
        memory: ConversationMemory,
    ) -> Tuple[Optional[AppositiveSplit], Optional[AppositiveAttachmentAmbiguity]]:
        """Resolve one explicit subject appositive with conservative evidence."""

        items = list(tokens)
        primary_tokens: List[lexicon.Token]
        appositive_tokens: List[lexicon.Token]
        tail: List[lexicon.Token]
        restriction: ModifierRestriction

        comma_indices = [
            index for index, token in enumerate(items) if token.norm == ","
        ]
        if len(comma_indices) >= 2:
            first, second = comma_indices[0], comma_indices[1]
            primary_tokens = [
                token for token in items[:first] if token.norm not in lexicon.PUNCTUATION
            ]
            appositive_tokens = [
                token for token in items[first + 1 : second]
                if token.norm not in lexicon.PUNCTUATION
            ]
            tail = [
                token for token in items[second + 1 :]
                if token.norm not in lexicon.PUNCTUATION
            ]
            restriction = ModifierRestriction.NONRESTRICTIVE
            if (
                not primary_tokens
                or not appositive_tokens
                or not tail
                or appositive_tokens[0].norm in self.RELATIVE_MARKERS
                or self._find_main_verb(primary_tokens) >= 0
                or self._find_main_verb(appositive_tokens) >= 0
                or self._find_main_verb(tail) < 0
            ):
                return None, None
        elif not comma_indices:
            verb_index = self._find_main_verb(items)
            if verb_index <= 2:
                return None, None
            subject = [
                token for token in items[:verb_index]
                if token.norm not in lexicon.PUNCTUATION
            ]
            name_start = next(
                (
                    index
                    for index, token in enumerate(subject[1:], start=1)
                    if self._looks_proper_name([token])
                    and any(
                        item.norm in lexicon.RELATIONS
                        or item.norm in self.APPOSITIVE_ROLE_NOUNS
                        for item in subject[:index]
                    )
                ),
                -1,
            )
            if name_start < 1:
                return None, None
            primary_tokens = subject[:name_start]
            appositive_tokens = subject[name_start:]
            tail = list(items[verb_index:])
            restriction = ModifierRestriction.RESTRICTIVE
        else:
            return None, None

        primary_surface = self._surface(primary_tokens)
        appositive_surface = self._surface(appositive_tokens)
        primary_is_name = self._looks_proper_name(primary_tokens)
        appositive_is_name = self._looks_proper_name(appositive_tokens)

        if appositive_is_name and not primary_is_name:
            canonical_tokens = appositive_tokens
            descriptor_tokens = primary_tokens
        else:
            canonical_tokens = primary_tokens
            descriptor_tokens = appositive_tokens

        relation_type, expected_kind, owner_id, role_name = self._appositive_profile(
            descriptor_tokens,
            descriptor_is_name=self._looks_proper_name(descriptor_tokens),
        )
        canonical = self._parse_np(
            canonical_tokens,
            memory,
            expected_kind=expected_kind,
            role_hint="subject",
        )
        if not canonical.ref or canonical.ref.kind != RefKind.ENTITY:
            candidates = [
                candidate
                for unresolved in canonical.unresolved
                for candidate in unresolved.candidates
            ]
            return None, self._appositive_ambiguity(
                items,
                primary_surface,
                appositive_surface,
                "appositive head could not be resolved",
                candidates,
            )

        entity = memory.get_entity(canonical.ref.key)
        if entity is None:
            return None, self._appositive_ambiguity(
                items,
                primary_surface,
                appositive_surface,
                "appositive head entity is unavailable",
                [],
            )

        descriptor_surface = self._surface(descriptor_tokens)
        binding = memory.bind_appositive_alias(
            entity.entity_id,
            descriptor_surface,
            relation_type=relation_type,
            expected_kind=expected_kind,
            role_owner_id=owner_id,
            role_name=role_name,
        )
        if not binding.resolved:
            return None, self._appositive_ambiguity(
                items,
                primary_surface,
                appositive_surface,
                binding.reason or "appositive identity is ambiguous",
                [candidate.entity_id for candidate in binding.candidates],
            )

        # Both explicit surfaces identify the same entity after safe binding.
        entity.add_alias(primary_surface)
        entity.add_alias(appositive_surface)
        internal_alias = memory.ensure_internal_alias(entity.entity_id)
        head_token = lexicon.Token(
            entity.canonical_name,
            internal_alias,
            -1,
        )
        relation = AppositiveRelation(
            head_entity_id=entity.entity_id,
            primary_surface=primary_surface,
            appositive_surface=appositive_surface,
            relation_type=relation_type,
            restriction=restriction,
            appositive_key=memory.normalize_alias(descriptor_surface),
            role_owner_id=owner_id,
            role_name=role_name,
            diagnostics=[
                f"appositive primary={primary_surface}",
                f"appositive value={appositive_surface}",
                f"appositive type={relation_type.value}",
            ],
        )
        return (
            AppositiveSplit(
                main_tokens=[head_token] + tail,
                relation=relation,
                entity_ids=[entity.entity_id],
                diagnostics=list(relation.diagnostics),
            ),
            None,
        )

    @staticmethod
    def _looks_proper_name(tokens: Sequence[lexicon.Token]) -> bool:
        content = [
            token
            for token in tokens
            if token.norm not in lexicon.DETERMINERS
            and token.norm not in lexicon.POSSESSIVES
            and token.norm not in lexicon.PUNCTUATION
        ]
        return bool(content) and all(
            token.text[:1].isupper()
            and token.norm not in lexicon.RELATIONS
            for token in content
        )

    def _appositive_profile(
        self,
        tokens: Sequence[lexicon.Token],
        *,
        descriptor_is_name: bool,
    ) -> Tuple[AppositiveRelationType, EntityKind, str, str]:
        norms = [
            token.norm
            for token in tokens
            if token.norm not in lexicon.PUNCTUATION
        ]
        if descriptor_is_name:
            return AppositiveRelationType.IDENTITY, EntityKind.PERSON, "", ""
        relation_word = next(
            (
                word
                for word in reversed(norms)
                if word in lexicon.RELATIONS
                or word in self.APPOSITIVE_ROLE_NOUNS
            ),
            "",
        )
        if relation_word:
            canonical, _gender, _number, kind = lexicon.relation_features(
                relation_word
            )
            owner_id = ""
            if norms and norms[0] == "my":
                owner_id = "user"
            elif norms and norms[0] == "your":
                owner_id = "assistant"
            return AppositiveRelationType.ROLE, kind, owner_id, canonical
        location_heads = {
            "city", "country", "county", "hospital", "office", "place",
            "school", "state", "store", "town", "village",
        }
        expected = (
            EntityKind.PLACE
            if any(word in location_heads for word in norms)
            else lexicon.classify_unknown_noun(norms)
        )
        return AppositiveRelationType.DESCRIPTION, expected, "", ""

    @staticmethod
    def _appositive_ambiguity(
        tokens: Sequence[lexicon.Token],
        primary_surface: str,
        appositive_surface: str,
        reason: str,
        candidate_ids: Sequence[str],
    ) -> AppositiveAttachmentAmbiguity:
        normalized = " ".join(token.norm for token in tokens)
        return AppositiveAttachmentAmbiguity(
            primary_surface=primary_surface,
            appositive_surface=appositive_surface,
            clause_surface=SemanticParser._surface(tokens),
            reason=reason,
            candidate_entity_ids=list(dict.fromkeys(candidate_ids)),
            ambiguity_id=(
                "appositive-"
                + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
            ),
            diagnostics=["appositive identity unresolved; durable assertion suppressed"],
        )

    @staticmethod
    def _event_signature_key(event: EventFrame) -> str:
        payload = repr(event.signature()).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:24]

    def _finite_verb_indices(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> List[int]:
        indices: List[int] = []
        for index, token in enumerate(tokens):
            previous = tokens[index - 1].norm if index > 0 else None
            if lexicon.is_probable_verb(token.norm, previous=previous):
                if token.norm not in lexicon.AUXILIARIES or index + 1 == len(tokens):
                    indices.append(index)
        return indices

    def _relative_gap_role(
        self,
        marker: str,
        body: Sequence[lexicon.Token],
    ) -> Optional[ModifierGapRole]:
        if marker == "whose":
            return ModifierGapRole.POSSESSOR
        if marker == "whom":
            return ModifierGapRole.PATIENT
        verb_index = self._find_main_verb(body)
        if verb_index < 0:
            return None
        if verb_index == 0:
            return ModifierGapRole.AGENT
        prefix = [token.norm for token in body[:verb_index]]
        if all(
            word in lexicon.AUXILIARIES
            or word in lexicon.NEGATORS
            or word in lexicon.INTENSIFIERS
            for word in prefix
        ):
            return ModifierGapRole.AGENT
        return ModifierGapRole.PATIENT

    def _relative_head_features(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> Tuple[EntityKind, Gender, GrammaticalNumber]:
        content = [
            token.norm
            for token in tokens
            if token.norm not in lexicon.DETERMINERS
            and token.norm not in {",", ";"}
        ]
        head = content[-1] if content else "entity"
        proper_name = bool(
            len(content) == 1
            and any(
                token.text[:1].isupper()
                for token in tokens
                if token.norm == head
            )
        )
        if proper_name or head in self.PERSON_RELATIVE_HEADS or head in lexicon.RELATIONS:
            kind = EntityKind.PERSON
        else:
            kind = lexicon.classify_unknown_noun(head)
        if head in self.FEMALE_RELATIVE_HEADS:
            gender = Gender.FEMALE
        elif head in self.MALE_RELATIVE_HEADS:
            gender = Gender.MALE
        else:
            gender = Gender.UNKNOWN
        number = (
            GrammaticalNumber.PLURAL
            if head.endswith("s") and not head.endswith("ss")
            else GrammaticalNumber.SINGULAR
        )
        return kind, gender, number

    def _relative_head_looks_like_complement(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> bool:
        content = [
            token.norm
            for token in tokens
            if token.norm not in lexicon.DETERMINERS
        ]
        return bool(content and content[-1] in self.ABSTRACT_COMPLEMENT_HEADS)

    SUBORDINATE_MARKERS: Tuple[Tuple[str, ...], ...] = (
        ("even", "though"),
        ("so", "that"),
        ("because",),
        ("when",),
        ("while",),
        ("before",),
        ("after",),
        ("until",),
        ("since",),
        ("if",),
        ("unless",),
        ("although",),
        ("though",),
    )

    def _split_subordinate_clause(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> Optional[SubordinateSplit]:
        """Split one finite subordinate clause without guessing attachment."""

        items = list(tokens)
        norms = [item.norm for item in items]
        marker_index = -1
        marker_tokens: Tuple[str, ...] = ()
        for index in range(len(items)):
            for candidate in self.SUBORDINATE_MARKERS:
                if tuple(norms[index : index + len(candidate)]) == candidate:
                    marker_index = index
                    marker_tokens = candidate
                    break
            if marker_index >= 0:
                break
        if marker_index < 0:
            return None

        marker = " ".join(marker_tokens)
        after_marker = marker_index + len(marker_tokens)
        if marker_index == 0:
            comma_index = next(
                (
                    index
                    for index in range(after_marker, len(items))
                    if items[index].norm == ","
                ),
                -1,
            )
            if comma_index < 0:
                return None
            subordinate_tokens = [
                item
                for item in items[after_marker:comma_index]
                if item.norm not in {",", ";"}
            ]
            main_tokens = [
                item
                for item in items[comma_index + 1 :]
                if item.norm not in {",", ";"}
            ]
            order = "subordinate-first"
        else:
            main_tokens = [
                item
                for item in items[:marker_index]
                if item.norm not in {",", ";"}
            ]
            subordinate_tokens = [
                item
                for item in items[after_marker:]
                if item.norm not in {",", ";"}
            ]
            order = "main-first"

        if not (
            self._is_independently_finite(main_tokens)
            and self._is_independently_finite(subordinate_tokens)
        ):
            return None

        relation_type, direction, candidates, certainty, rationale = (
            self._classify_subordinate_relation(
                marker,
                main_tokens,
                subordinate_tokens,
            )
        )
        return SubordinateSplit(
            main_tokens=main_tokens,
            subordinate_tokens=subordinate_tokens,
            marker=marker,
            relation_type=relation_type,
            direction=direction,
            candidate_types=candidates,
            certainty=certainty,
            diagnostics=[
                f"subordinate marker={marker}",
                f"subordinate order={order}",
                rationale,
            ],
        )

    def _classify_subordinate_relation(
        self,
        marker: str,
        main_tokens: Sequence[lexicon.Token],
        subordinate_tokens: Sequence[lexicon.Token],
    ) -> Tuple[
        ClauseRelationType,
        ClauseRelationDirection,
        List[ClauseRelationType],
        int,
        str,
    ]:
        main_words = [item.norm for item in main_tokens]
        subordinate_words = [item.norm for item in subordinate_tokens]

        resolved = {
            "because": (
                ClauseRelationType.CAUSE,
                ClauseRelationDirection.SUBORDINATE_TO_MAIN,
            ),
            "when": (
                ClauseRelationType.TEMPORAL_WHEN,
                ClauseRelationDirection.SYMMETRIC,
            ),
            "before": (
                ClauseRelationType.TEMPORAL_BEFORE,
                ClauseRelationDirection.MAIN_TO_SUBORDINATE,
            ),
            "after": (
                ClauseRelationType.TEMPORAL_AFTER,
                ClauseRelationDirection.MAIN_TO_SUBORDINATE,
            ),
            "until": (
                ClauseRelationType.TEMPORAL_UNTIL,
                ClauseRelationDirection.MAIN_TO_SUBORDINATE,
            ),
            "if": (
                ClauseRelationType.CONDITION,
                ClauseRelationDirection.SUBORDINATE_TO_MAIN,
            ),
            "unless": (
                ClauseRelationType.EXCEPTION_CONDITION,
                ClauseRelationDirection.SUBORDINATE_TO_MAIN,
            ),
            "although": (
                ClauseRelationType.CONCESSION,
                ClauseRelationDirection.SUBORDINATE_TO_MAIN,
            ),
            "though": (
                ClauseRelationType.CONCESSION,
                ClauseRelationDirection.SUBORDINATE_TO_MAIN,
            ),
            "even though": (
                ClauseRelationType.CONCESSION,
                ClauseRelationDirection.SUBORDINATE_TO_MAIN,
            ),
        }
        if marker in resolved:
            relation_type, direction = resolved[marker]
            return relation_type, direction, [], 230, "relation resolved by connector"

        if marker == "since":
            if any(word in lexicon.TIME_WORDS for word in subordinate_words):
                return (
                    ClauseRelationType.TEMPORAL_SINCE,
                    ClauseRelationDirection.MAIN_TO_SUBORDINATE,
                    [],
                    210,
                    "since resolved temporally by explicit time anchor",
                )
            return (
                ClauseRelationType.AMBIGUOUS,
                ClauseRelationDirection.UNRESOLVED,
                [ClauseRelationType.CAUSE, ClauseRelationType.TEMPORAL_SINCE],
                128,
                "since remains causally/temporally ambiguous",
            )

        if marker == "while":
            if self._contains_progressive(main_words) and self._contains_progressive(
                subordinate_words
            ):
                return (
                    ClauseRelationType.TEMPORAL_OVERLAP,
                    ClauseRelationDirection.SYMMETRIC,
                    [],
                    210,
                    "while resolved as temporal overlap from paired progressives",
                )
            return (
                ClauseRelationType.AMBIGUOUS,
                ClauseRelationDirection.UNRESOLVED,
                [
                    ClauseRelationType.TEMPORAL_OVERLAP,
                    ClauseRelationType.CONCESSION,
                ],
                128,
                "while remains temporal/concessive ambiguous",
            )

        if marker == "so that":
            if any(word in lexicon.MODALS for word in subordinate_words):
                return (
                    ClauseRelationType.PURPOSE,
                    ClauseRelationDirection.MAIN_TO_SUBORDINATE,
                    [],
                    220,
                    "so that resolved as purpose from subordinate modality",
                )
            if any(
                word in {"became", "got", "happened", "resulted"}
                for word in subordinate_words
            ):
                return (
                    ClauseRelationType.RESULT,
                    ClauseRelationDirection.MAIN_TO_SUBORDINATE,
                    [],
                    205,
                    "so that resolved as result from change-of-state cue",
                )
            return (
                ClauseRelationType.AMBIGUOUS,
                ClauseRelationDirection.UNRESOLVED,
                [ClauseRelationType.PURPOSE, ClauseRelationType.RESULT],
                128,
                "so that remains purpose/result ambiguous",
            )

        return (
            ClauseRelationType.AMBIGUOUS,
            ClauseRelationDirection.UNRESOLVED,
            [],
            96,
            "unclassified subordinate relation",
        )

    @staticmethod
    def _contains_progressive(words: Sequence[str]) -> bool:
        return any(word in lexicon.COPULAS for word in words) and any(
            word.endswith("ing") for word in words
        )

    @staticmethod
    def _rewrite_yoda(tokens: Sequence[lexicon.Token]) -> List[lexicon.Token]:
        """Normalize fronted-predicate forms such as ``Bought it, Sarah did``."""

        items = list(tokens)
        norms = [token.norm for token in items]
        if not items or norms[-1] not in {"did", "does", "will"}:
            return items
        if not lexicon.is_probable_verb(norms[0]):
            return items

        comma_index = norms.index(",") if "," in norms else -1
        if comma_index >= 0:
            front = [token for token in items[:comma_index] if token.norm not in lexicon.PUNCTUATION]
            subject = [token for token in items[comma_index + 1 : -1] if token.norm not in lexicon.PUNCTUATION]
        else:
            # Implicit comma: locate a late possessive/relation/proper-name chunk.
            subject_start = -1
            for idx in range(1, len(items) - 1):
                norm = items[idx].norm
                if norm in lexicon.POSSESSIVES or norm in lexicon.RELATIONS or items[idx].text[:1].isupper():
                    subject_start = idx
                    break
            if subject_start < 0:
                return items
            front = [token for token in items[:subject_start] if token.norm not in lexicon.PUNCTUATION]
            subject = [token for token in items[subject_start:-1] if token.norm not in lexicon.PUNCTUATION]
        if not front or not subject:
            return items
        rewritten = subject + front
        return [lexicon.Token(token.text, token.norm, idx) for idx, token in enumerate(rewritten)]

    # ------------------------------------------------------------------
    # Question parsing
    # ------------------------------------------------------------------

    def _parse_question(
        self,
        tokens: Sequence[lexicon.Token],
        raw: str,
        memory: ConversationMemory,
    ) -> Tuple[Optional[QuestionFrame], List[UnresolvedReference], List[str], List[str]]:
        items = [token for token in tokens if token.norm not in lexicon.PUNCTUATION]
        words = [token.norm for token in items]
        diagnostics: List[str] = []
        if not words:
            return None, [], [], ["empty question"]

        # Conventional social questions bypass literal query semantics.
        social = self._detect_social(words)
        if social:
            dummy = EventFrame("social", raw_text=raw, source=SourceKind.USER)
            frame = QuestionFrame(
                kind=QuestionKind.HOW if words[0] == "how" else QuestionKind.WHAT,
                event=dummy,
                raw_text=raw,
                social_convention=social,
            )
            return frame, [], [], [f"social convention: {social}"]

        if words[:2] == ["what", "happened"] or words[:3] == ["what", "has", "happened"]:
            event = EventFrame("*", {"event": SemanticRef.variable("event", EntityKind.EVENT)}, raw_text=raw)
            return QuestionFrame(
                kind=QuestionKind.WHAT_HAPPENED,
                event=event,
                requested_role="event",
                answer_type=EntityKind.EVENT,
                raw_text=raw,
            ), [], [], ["open event query"]

        def finalize_question(result):
            frame = result[0]
            if frame is not None:
                self._annotate_infinitival_question_scope(frame, items)
            return result

        first = words[0]
        if first in lexicon.YES_NO_STARTERS:
            return finalize_question(self._parse_yes_no(items, raw, memory))
        if first == "who" or first == "whom":
            return finalize_question(self._parse_who(items, raw, memory))
        if first == "whose":
            return finalize_question(self._parse_whose(items, raw, memory))
        if first == "what":
            return finalize_question(self._parse_what(items, raw, memory))
        if first == "which":
            return finalize_question(self._parse_which(items, raw, memory))
        if first in {"when", "where", "why", "how"}:
            return finalize_question(
                self._parse_adverbial_question(items, raw, memory)
            )
        return None, [], [], ["unrecognized interrogative form"]

    def _parse_yes_no(
        self,
        items: Sequence[lexicon.Token],
        raw: str,
        memory: ConversationMemory,
    ) -> Tuple[Optional[QuestionFrame], List[UnresolvedReference], List[str], List[str]]:
        aux = items[0]
        rest = list(items[1:])
        if not rest:
            return None, [], [], ["yes/no question lacks proposition"]
        main_idx = self._find_main_verb(rest, start=0)
        if aux.norm in lexicon.COPULAS:
            # Is Sarah a nurse? -> Sarah is a nurse.
            declarative = rest[:1] + [aux] + rest[1:]
        elif main_idx >= 0:
            # Did Sarah buy a car? -> Sarah did buy a car.
            declarative = rest[:main_idx] + [aux] + rest[main_idx:]
        else:
            declarative = rest + [aux]
        matrix_polarity, embedded_polarity = (
            self._question_infinitival_polarities(rest, main_idx)
        )
        clause = self._parse_clause(declarative, raw, memory)
        if not clause.event:
            return None, clause.unresolved, clause.entities, clause.diagnostics + ["failed yes/no proposition parse"]
        if matrix_polarity is not None:
            clause.event.polarity = matrix_polarity
        frame = QuestionFrame(
            kind=QuestionKind.YES_NO,
            event=clause.event,
            raw_text=raw,
            unresolved=clause.unresolved,
            matrix_polarity=matrix_polarity,
            embedded_polarity=embedded_polarity,
        )
        return frame, clause.unresolved, clause.entities, clause.diagnostics + [f"yes/no auxiliary={aux.norm}"]

    def _annotate_infinitival_question_scope(
        self,
        frame: QuestionFrame,
        items: Sequence[lexicon.Token],
    ) -> None:
        """Attach two-level polarity to any licensed infinitival question."""

        predicate = frame.event.predicate
        if predicate not in self.INFINITIVAL_PREDICATES:
            return
        predicate_index = next(
            (
                index
                for index, token in enumerate(items)
                if lexicon.lemma(token.norm) == predicate
            ),
            -1,
        )
        if predicate_index < 0:
            return
        boundary = next(
            (
                index
                for index in range(predicate_index + 1, len(items) - 1)
                if items[index].norm == "to"
                and any(
                    lexicon.is_probable_verb(items[next_index].norm, "to")
                    for next_index in range(
                        index + 1,
                        min(len(items), index + 4),
                    )
                    if items[next_index].norm not in (
                        lexicon.NEGATORS | lexicon.INTENSIFIERS
                    )
                )
            ),
            -1,
        )
        if boundary < 0:
            return
        matrix_negative = any(
            token.norm in lexicon.NEGATORS
            for token in items[:predicate_index]
        )
        embedded_negative = any(
            token.norm in lexicon.NEGATORS
            for token in items[predicate_index + 1 :]
        )
        frame.matrix_polarity = not matrix_negative
        frame.embedded_polarity = not embedded_negative
        frame.event.polarity = frame.matrix_polarity

    def _question_infinitival_polarities(
        self,
        rest: Sequence[lexicon.Token],
        main_idx: int,
    ) -> Tuple[Optional[bool], Optional[bool]]:
        """Return matrix and embedded polarity for a selected infinitive.

        General clause parsing has one polarity bit, but a question such as
        ``Did Sarah plan not to leave?`` contains two independently scoped
        propositions.  Preserve that distinction for the infinitival answer
        contract instead of treating embedded negation as matrix negation.
        """

        if main_idx < 0 or main_idx >= len(rest):
            return None, None
        predicate = lexicon.lemma(rest[main_idx].norm)
        if predicate not in self.INFINITIVAL_PREDICATES:
            return None, None

        boundary = next(
            (
                index
                for index in range(main_idx + 1, len(rest) - 1)
                if rest[index].norm == "to"
                and any(
                    lexicon.is_probable_verb(rest[next_index].norm, "to")
                    for next_index in range(index + 1, min(len(rest), index + 4))
                    if rest[next_index].norm not in lexicon.NEGATORS
                    | lexicon.INTENSIFIERS
                )
            ),
            -1,
        )
        if boundary < 0:
            return None, None

        matrix_negative = any(
            token.norm in lexicon.NEGATORS
            for token in rest[:main_idx]
        )
        embedded_negative = any(
            token.norm in lexicon.NEGATORS
            for token in rest[main_idx + 1 :]
        )
        return not matrix_negative, not embedded_negative

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

    @staticmethod
    def _classify_why(event: EventFrame) -> WhyKind:
        if event.modality in {"should", "must", "ought", "need"}:
            return WhyKind.JUSTIFICATION
        if event.predicate in {"think", "believe", "know", "claim", "say"}:
            return WhyKind.EVIDENCE
        if event.predicate in lexicon.PHYSICAL_EVENT_VERBS:
            return WhyKind.CAUSE
        if event.predicate in lexicon.PURPOSE_LIKELY_VERBS:
            return WhyKind.PURPOSE
        if event.predicate in lexicon.VOLITIONAL_VERBS:
            return WhyKind.MOTIVE
        return WhyKind.CAUSE

    @staticmethod
    def _classify_how(event: EventFrame) -> HowKind:
        if event.predicate in lexicon.PROCESS_VERBS:
            return HowKind.PROCESS
        if event.predicate in lexicon.PHYSICAL_EVENT_VERBS:
            return HowKind.MECHANISM
        if event.predicate == "be":
            return HowKind.STATE
        return HowKind.METHOD

    @staticmethod
    def _variable_token(role: str) -> lexicon.Token:
        return lexicon.Token(f"?{role}", f"__var_{role}__", -1)

    @staticmethod
    def _object_insertion_index(verb_and_tail: Sequence[lexicon.Token]) -> int:
        """Place an object variable before temporal/prepositional adjuncts."""

        if not verb_and_tail:
            return 0
        for idx in range(1, len(verb_and_tail)):
            norm = verb_and_tail[idx].norm
            if norm in lexicon.PREPOSITIONS or norm in lexicon.TIME_WORDS:
                return idx
        return len(verb_and_tail)

    # ------------------------------------------------------------------
    # Declarative clause parser
    # ------------------------------------------------------------------

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
        polarity = "not" not in [token.norm for token in main_items]
        tense = lexicon.detect_tense(main_token.norm, auxiliary_tokens[0] if auxiliary_tokens else None)
        if "will" in auxiliary_tokens or "shall" in auxiliary_tokens:
            tense = "future"
        aspect = "perfect" if any(word in lexicon.AUX_HAVE for word in auxiliary_tokens) else "simple"
        if aspect == "perfect" and tense != "future":
            tense = "past" if "had" in auxiliary_tokens else "present"

        # Passive voice: "The coat was bought by Sarah".
        passive = (
            predicate != "be"
            and any(word in lexicon.COPULAS for word in auxiliary_tokens)
            and any(token.norm == "by" for token in main_items[verb_idx + 1 :])
        )

        subject_tokens = [
            token for token in main_items[:verb_idx]
            if token.norm not in lexicon.AUXILIARIES and token.norm not in lexicon.NEGATORS
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

    # ------------------------------------------------------------------
    # Noun phrase/entity resolution
    # ------------------------------------------------------------------

    def _parse_np(
        self,
        tokens: Sequence[lexicon.Token],
        memory: ConversationMemory,
        *,
        expected_kind: EntityKind = EntityKind.UNKNOWN,
        role_hint: str = "other",
        preposition: Optional[str] = None,
    ) -> NPResult:
        items = [token for token in tokens if token.norm not in lexicon.PUNCTUATION and token.norm not in lexicon.INTENSIFIERS]
        if not items:
            return NPResult(None)
        surface = self._surface(items)
        norms = [token.norm for token in items]

        # Typed variable inserted by interrogative transformation.
        if len(items) == 1 and norms[0].startswith("__var_") and norms[0].endswith("__"):
            role = norms[0][len("__var_") : -2]
            return NPResult(SemanticRef.variable(role, expected_kind), surface=surface)

        # Private aliases inserted by structural transformations bind an
        # already-created entity before ordinary possessive/name heuristics.
        if len(items) == 1 and norms[0].startswith(
            ("entityref-", "modified-", "possessed-")
        ):
            resolution = memory.find_by_alias(norms[0], EntityKind.UNKNOWN)
            if resolution.resolved and resolution.entity:
                entity = resolution.entity
                memory.mention(entity.entity_id, role_hint)
                return NPResult(
                    entity.to_ref(surface),
                    entity_ids=[entity.entity_id],
                    surface=surface,
                )
            unresolved = memory.unresolved_from_resolution(
                surface,
                resolution,
                expected_kind,
            )
            return NPResult(None, [unresolved], surface=surface)

        # Numeric determiner/quantity: "three cars", "$50".
        quantity: Optional[SemanticRef] = None
        quantity_prefix: List[str] = []
        for word in norms:
            if word.isdigit() or word.replace(".", "", 1).isdigit() or (word in lexicon.NUMBER_WORDS and word not in lexicon.ARTICLES) or word[:1] in "$€£":
                quantity_prefix.append(word)
            else:
                break
        if quantity_prefix:
            parsed = lexicon.parse_number(quantity_prefix)
            if parsed is not None:
                key = str(int(parsed)) if isinstance(parsed, float) and parsed.is_integer() else str(parsed)
                quantity = SemanticRef.literal(key, " ".join(quantity_prefix), EntityKind.ABSTRACT)
                items = items[len(quantity_prefix) :]
                norms = norms[len(quantity_prefix) :]
                if not items:
                    return NPResult(quantity, quantity=quantity, surface=surface)

        # Fixed participant or third-person pronoun.
        if len(items) == 1 and norms[0] in lexicon.PRONOUN_FEATURES:
            resolution = memory.resolve_pronoun(norms[0], expected_kind)
            if resolution.resolved:
                assert resolution.entity is not None
                memory.mention(resolution.entity.entity_id, role_hint)
                return NPResult(resolution.entity.to_ref(surface), quantity=quantity, entity_ids=[resolution.entity.entity_id], surface=surface)
            unresolved = memory.unresolved_from_resolution(surface, resolution, expected_kind)
            return NPResult(None, [unresolved], quantity=quantity, surface=surface)

        # Genitive noun phrase: ``Sarah's eyes`` / ``John's car``.
        # The tokenizer intentionally keeps apostrophe-s attached, so resolve
        # the owner here and key the possessed entity by owner + head phrase.
        if len(items) >= 2 and (norms[0].endswith("'s") or norms[0].endswith("’s")):
            owner_surface = items[0].text[:-2]
            owner_norm = norms[0][:-2]
            owner_resolution = memory.find_by_alias(owner_norm, EntityKind.PERSON)
            if owner_resolution.resolved and owner_resolution.entity:
                owner = owner_resolution.entity
            else:
                owner = memory.get_or_create_named_entity(
                    owner_surface,
                    kind=EntityKind.PERSON,
                    gender=lexicon.infer_name_gender(owner_norm),
                    aliases=[owner_norm],
                    role_salience=0.4,
                )
            possessed_items = list(items[1:])
            possessed_norms = [token.norm for token in possessed_items if token.norm not in lexicon.ARTICLES]
            canonical = self._surface([token for token in possessed_items if token.norm not in lexicon.ARTICLES])
            alias = f"{owner.entity_id}:{lexicon.normalize_phrase(possessed_norms)}"
            existing = memory.find_by_alias(alias, expected_kind)
            if existing.resolved and existing.entity:
                entity = existing.entity
            else:
                kind = expected_kind if expected_kind != EntityKind.UNKNOWN else lexicon.classify_unknown_noun(possessed_norms, preposition=preposition)
                entity = memory.get_or_create_named_entity(
                    canonical,
                    kind=kind,
                    aliases=[surface, alias, lexicon.normalize_phrase(possessed_norms)],
                    role_salience=0.4,
                )
                entity.owner_id = owner.entity_id
            self._apply_np_attributes(entity, possessed_norms)
            return NPResult(entity.to_ref(surface), quantity=quantity, entity_ids=[owner.entity_id, entity.entity_id], surface=surface)

        # Possessive relationship: my sister / your mom / his brother.
        if len(items) >= 2 and norms[0] in lexicon.POSSESSIVES and norms[-1] in lexicon.RELATIONS:
            owner_id: Optional[str] = None
            if norms[0] == "my":
                owner_id = "user"
            elif norms[0] == "your":
                owner_id = "assistant"
            elif norms[0] in {"his", "her", "their"}:
                pronoun = {"his": "he", "her": "she", "their": "they"}[norms[0]]
                owner_resolution = memory.resolve_pronoun(pronoun, EntityKind.PERSON)
                if owner_resolution.resolved and owner_resolution.entity:
                    owner_id = owner_resolution.entity.entity_id
                else:
                    unresolved = memory.unresolved_from_resolution(norms[0], owner_resolution, EntityKind.PERSON)
                    return NPResult(None, [unresolved], quantity=quantity, surface=surface)
            if owner_id:
                entity = memory.get_or_create_relation(owner_id, norms[-1], surface=surface, role_salience=0.4)
                return NPResult(entity.to_ref(surface), quantity=quantity, entity_ids=[entity.entity_id], surface=surface)

        # Possessive common object: my car / her phone.  Resolve owner and create
        # an object entity keyed by owner + head noun.
        if len(items) >= 2 and norms[0] in lexicon.POSSESSIVES:
            owner_id: Optional[str] = None
            if norms[0] == "my":
                owner_id = "user"
            elif norms[0] == "your":
                owner_id = "assistant"
            else:
                pronoun = {"his": "he", "her": "she", "their": "they", "our": "we", "its": "it"}.get(norms[0])
                if pronoun:
                    owner_resolution = memory.resolve_pronoun(pronoun)
                    if owner_resolution.resolved and owner_resolution.entity:
                        owner_id = owner_resolution.entity.entity_id
                    else:
                        unresolved = memory.unresolved_from_resolution(norms[0], owner_resolution, EntityKind.UNKNOWN)
                        return NPResult(None, [unresolved], quantity=quantity, surface=surface)
            canonical = " ".join(norms[1:])
            alias = f"{owner_id}:{canonical}" if owner_id else canonical
            existing = memory.find_by_alias(alias, expected_kind)
            if existing.resolved and existing.entity:
                entity = existing.entity
            else:
                kind = expected_kind if expected_kind != EntityKind.UNKNOWN else lexicon.classify_unknown_noun(norms[1:], preposition=preposition)
                entity = memory.get_or_create_named_entity(canonical, kind=kind, aliases=[surface, alias], role_salience=0.4)
                entity.owner_id = owner_id
            return NPResult(entity.to_ref(surface), quantity=quantity, entity_ids=[entity.entity_id], surface=surface)

        # Remove articles for entity lookup/creation.
        content_items = [token for token in items if token.norm not in lexicon.ARTICLES]
        content_norms = [token.norm for token in content_items]
        if not content_items:
            return NPResult(None, quantity=quantity, surface=surface)
        normalized = lexicon.normalize_phrase(content_norms)

        # Standalone relation defaults to the user's relation in conversational
        # self-report unless an existing unowned relation is more salient.
        if content_norms[-1] in lexicon.RELATIONS and len(content_norms) <= 2:
            entity = memory.get_or_create_relation("user", content_norms[-1], surface=surface, role_salience=0.4)
            return NPResult(entity.to_ref(surface), quantity=quantity, entity_ids=[entity.entity_id], surface=surface)

        # Existing alias/coreference before creating a new entity.
        resolution = memory.find_by_alias(normalized, expected_kind)
        if resolution.resolved and resolution.entity:
            resolution.entity.add_alias(surface)
            self._apply_np_attributes(resolution.entity, content_norms)
            memory.mention(resolution.entity.entity_id, role_hint)
            return NPResult(resolution.entity.to_ref(surface), quantity=quantity, entity_ids=[resolution.entity.entity_id], surface=surface)
        if resolution.status == "ambiguous":
            unresolved = memory.unresolved_from_resolution(surface, resolution, expected_kind)
            return NPResult(None, [unresolved], quantity=quantity, surface=surface)

        # Proper names and known names are people; prepositional/location nouns
        # and ordinary noun phrases are typed conservatively.
        looks_name = (
            len(content_items) <= 3
            and any(token.text[:1].isupper() for token in content_items)
            and not all(token.index == 0 for token in content_items)
        ) or content_norms[0] in lexicon.FEMALE_NAMES | lexicon.MALE_NAMES
        if looks_name and expected_kind in {EntityKind.UNKNOWN, EntityKind.PERSON}:
            kind = EntityKind.PERSON
            gender = lexicon.infer_name_gender(content_norms[0])
        else:
            kind = expected_kind if expected_kind != EntityKind.UNKNOWN else lexicon.classify_unknown_noun(content_norms, preposition=preposition)
            gender = Gender.UNKNOWN
        number = (
            GrammaticalNumber.PLURAL
            if "and" in content_norms or (content_norms[-1].endswith("s") and content_norms[-1] not in {"news"})
            else GrammaticalNumber.SINGULAR
        )
        if "and" in content_norms:
            gender = Gender.NEUTRAL
        entity = memory.get_or_create_named_entity(
            self._surface(content_items),
            kind=kind,
            gender=gender,
            number=number,
            aliases=[surface, normalized],
            role_salience=0.4,
        )
        self._apply_np_attributes(entity, content_norms)
        return NPResult(entity.to_ref(surface), quantity=quantity, entity_ids=[entity.entity_id], surface=surface)

    @staticmethod
    def _apply_np_attributes(entity, words: Sequence[str]) -> None:
        """Extract safe, explicit modifier attributes from a noun phrase."""

        if not words:
            return
        for word in words[:-1]:
            if word in lexicon.COLORS:
                entity.attributes["color"] = word
            dimension = lexicon.ADJECTIVE_DIMENSIONS.get(word)
            if dimension:
                entity.attributes[dimension] = word

    @staticmethod
    def _looks_entity_phrase(tokens: Sequence[lexicon.Token]) -> bool:
        if not tokens:
            return False
        if len(tokens) == 1 and tokens[0].norm in lexicon.PRONOUN_FEATURES:
            return True
        return any(token.text[:1].isupper() for token in tokens) or tokens[-1].norm in lexicon.RELATIONS

    @staticmethod
    def _surface(tokens: Sequence[lexicon.Token]) -> str:
        return " ".join(token.text for token in tokens if token.norm not in lexicon.PUNCTUATION).strip()
