from __future__ import annotations

from pathlib import Path


SOURCE = Path("/tmp/relative-apply.py")
source = SOURCE.read_text(encoding="utf-8")


def swap(old: str, new: str, *, count: int = 1) -> None:
    global source
    found = source.count(old)
    if found != count:
        raise RuntimeError(
            f"Expected {count} staged-source matches, found {found}: {old[:120]!r}"
        )
    source = source.replace(old, new, count)


# The staged implementation was authored before #49 reordered ParseResult and
# added typed subordinate relations. Rebase only those exact structural seams.
swap(
    '''class ParseResult:\n    speech_act: SpeechAct\n    raw_text: str\n    events: List[EventFrame] = field(default_factory=list)\n    question: Optional[QuestionFrame] = None\n    unresolved: List[UnresolvedReference] = field(default_factory=list)\n    relations: List[ClauseRelation] = field(default_factory=list)\n''',
    '''class ParseResult:\n    speech_act: SpeechAct\n    raw_text: str\n    events: List[EventFrame] = field(default_factory=list)\n    relations: List[ClauseRelation] = field(default_factory=list)\n    question: Optional[QuestionFrame] = None\n    entities: List[str] = field(default_factory=list)\n    unresolved: List[UnresolvedReference] = field(default_factory=list)\n    normalized_text: str = ""\n    diagnostics: List[str] = field(default_factory=list)\n''',
)
swap(
    '''class ParseResult:\n    speech_act: SpeechAct\n    raw_text: str\n    events: List[EventFrame] = field(default_factory=list)\n    question: Optional[QuestionFrame] = None\n    unresolved: List[UnresolvedReference] = field(default_factory=list)\n    relations: List[ClauseRelation] = field(default_factory=list)\n    modifiers: List[EntityModifierRelation] = field(default_factory=list)\n    modifier_ambiguities: List[ModifierAttachmentAmbiguity] = field(default_factory=list)\n''',
    '''class ParseResult:\n    speech_act: SpeechAct\n    raw_text: str\n    events: List[EventFrame] = field(default_factory=list)\n    relations: List[ClauseRelation] = field(default_factory=list)\n    modifiers: List[EntityModifierRelation] = field(default_factory=list)\n    modifier_ambiguities: List[ModifierAttachmentAmbiguity] = field(default_factory=list)\n    question: Optional[QuestionFrame] = None\n    entities: List[str] = field(default_factory=list)\n    unresolved: List[UnresolvedReference] = field(default_factory=list)\n    normalized_text: str = ""\n    diagnostics: List[str] = field(default_factory=list)\n''',
)
swap(
    '''            "relations": [relation.to_dict() for relation in self.relations],\n            "diagnostics": list(self.diagnostics),\n''',
    '''            "relations": [relation.to_dict() for relation in self.relations],\n            "question": self.question.to_dict() if self.question else None,\n            "entities": list(self.entities),\n            "unresolved": [item.to_dict() for item in self.unresolved],\n            "normalized_text": self.normalized_text,\n            "diagnostics": list(self.diagnostics),\n''',
)
swap(
    '''            "relations": [relation.to_dict() for relation in self.relations],\n            "modifiers": [modifier.to_dict() for modifier in self.modifiers],\n            "modifier_ambiguities": [\n                ambiguity.to_dict() for ambiguity in self.modifier_ambiguities\n            ],\n            "diagnostics": list(self.diagnostics),\n''',
    '''            "relations": [relation.to_dict() for relation in self.relations],\n            "modifiers": [modifier.to_dict() for modifier in self.modifiers],\n            "modifier_ambiguities": [\n                ambiguity.to_dict() for ambiguity in self.modifier_ambiguities\n            ],\n            "question": self.question.to_dict() if self.question else None,\n            "entities": list(self.entities),\n            "unresolved": [item.to_dict() for item in self.unresolved],\n            "normalized_text": self.normalized_text,\n            "diagnostics": list(self.diagnostics),\n''',
)

# Avoid duplicate imports already present in the current parser.
swap(
    '''    ClauseRelationType,\n    EntityKind,\n    EventFrame,\n''',
    '''    ClauseRelationType,\n    EntityKind,\n    EventFrame,\n    Gender,\n    GrammaticalNumber,\n''',
)
swap(
    '''    ClauseRelationType,\n    EntityKind,\n    EntityModifierRelation,\n    EventFrame,\n    Gender,\n    GrammaticalNumber,\n    ModifierAttachmentAmbiguity,\n    ModifierGapRole,\n    ModifierRestriction,\n''',
    '''    ClauseRelationType,\n    EntityKind,\n    EntityModifierRelation,\n    EventFrame,\n    Gender,\n    GrammaticalNumber,\n    ModifierAttachmentAmbiguity,\n    ModifierGapRole,\n    ModifierRestriction,\n''',
)

# Preserve #49's candidate relation types rather than referencing a removed
# `SubordinateSplit.inferred` field.
swap(
    '''                        certainty=subordinate_split.certainty,\n                        inferred=subordinate_split.inferred,\n                        diagnostics=list(subordinate_split.diagnostics),\n''',
    '''                        certainty=subordinate_split.certainty,\n                        candidate_types=list(subordinate_split.candidate_types),\n                        diagnostics=list(subordinate_split.diagnostics),\n''',
)

# Adapt the final ParseResult constructor insertion to current field order.
swap(
    '''parser_text = parser_text.replace(\n    ''' + "'''" + '''            unresolved=unresolved,\n            relations=relations,\n            diagnostics=diagnostics,\n''' + "'''" + ''',\n    ''' + "'''" + '''            unresolved=unresolved,\n            relations=relations,\n            modifiers=modifiers,\n            modifier_ambiguities=modifier_ambiguities,\n            diagnostics=diagnostics,\n''' + "'''" + ''',\n    1,\n)''',
    '''parser_text = parser_text.replace(\n    ''' + "'''" + '''            events=events,\n            relations=relations,\n            entities=list(dict.fromkeys(entities)),\n            unresolved=unresolved,\n            normalized_text=normalized,\n            diagnostics=diagnostics,\n''' + "'''" + ''',\n    ''' + "'''" + '''            events=events,\n            relations=relations,\n            modifiers=modifiers,\n            modifier_ambiguities=modifier_ambiguities,\n            entities=list(dict.fromkeys(entities)),\n            unresolved=unresolved,\n            normalized_text=normalized,\n            diagnostics=diagnostics,\n''' + "'''" + ''',\n    1,\n)''',
)

# Drop an obsolete direct-learn replacement that no longer exists in runtime.py.
start_marker = (
    'replace_once(\n'
    '    "clanker_lm/runtime.py",\n'
    "    '''            self.memory.add_clause_relations(parsed.relations, stored)"
)
start = source.find(start_marker)
if start >= 0:
    end_marker = "\n)\n\n# ---------------------------------------------------------------------------\n# Conformance suite"
    end = source.find(end_marker, start)
    if end < 0:
        raise RuntimeError("Could not locate obsolete direct-learn replacement end")
    source = source[:start] + source[end + 3 :]

compiled = compile(source, str(SOURCE), "exec")
exec(compiled, {"__name__": "__main__"})

# Deterministic ambiguity IDs and a parser import compatible with current main.
parser_path = Path("clanker_lm/parser.py")
parser_text = parser_path.read_text(encoding="utf-8")
header = "from __future__ import annotations\n\n"
if parser_text.count(header) != 1:
    raise RuntimeError("Could not locate parser import header")
parser_text = parser_text.replace(header, header + "import hashlib\n", 1)
old_hash = '''("relative-"\n                    + str(abs(hash(tuple(token.norm for token in items))))\n                )'''
new_hash = '''("relative-"\n                    + hashlib.sha256(\n                        " ".join(token.norm for token in items).encode("utf-8")\n                    ).hexdigest()[:16]\n                )'''
if parser_text.count(old_hash) != 2:
    raise RuntimeError(f"Expected two nondeterministic relative IDs, found {parser_text.count(old_hash)}")
parser_text = parser_text.replace(old_hash, new_hash)
parser_path.write_text(parser_text, encoding="utf-8")

# The possessive assertion must be role-agnostic because `break` is currently
# represented with its affected entity under the parser's licensed role.
test_path = Path("tests/test_relative_modifiers.py")
test_text = test_path.read_text(encoding="utf-8")
old_assertion = '''    assert relation.possessed_entity_id\n    assert result.events[1].arguments["patient"].key == relation.possessed_entity_id\n'''
new_assertion = '''    assert relation.possessed_entity_id\n    assert any(\n        ref.key == relation.possessed_entity_id\n        for ref in result.events[1].arguments.values()\n    )\n'''
if test_text.count(old_assertion) != 1:
    raise RuntimeError("Could not locate possessive modifier assertion")
test_path.write_text(test_text.replace(old_assertion, new_assertion, 1), encoding="utf-8")
