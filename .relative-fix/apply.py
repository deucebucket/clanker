from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected one match in {path}, found {count}: {old[:100]!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "clanker_lm/parser.py",
    '''        marker_indices = [
            index
            for index, token in enumerate(items)
            if token.norm in self.RELATIVE_MARKERS and index > 0
        ]
''',
    '''        marker_indices = [
            index
            for index, token in enumerate(items)
            if token.norm in self.RELATIVE_MARKERS
            and index > 0
            and not (
                token.norm == "that"
                and items[index - 1].norm == "so"
            )
        ]
''',
)

replace_once(
    "clanker_lm/parser.py",
    '''        content = [
            token.norm
            for token in tokens
            if token.norm not in lexicon.ARTICLES
            and token.norm not in lexicon.DEMONSTRATIVES
            and token.norm not in {",", ";"}
        ]
        head = content[-1] if content else "entity"
        if head in self.PERSON_RELATIVE_HEADS or head in lexicon.RELATIONS:
            kind = EntityKind.PERSON
        else:
            kind = lexicon.classify_unknown_noun(head)
''',
    '''        content = [
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
''',
)

replace_once(
    "clanker_lm/parser.py",
    '''        content = [
            token.norm
            for token in tokens
            if token.norm not in lexicon.ARTICLES
            and token.norm not in lexicon.DEMONSTRATIVES
        ]
''',
    '''        content = [
            token.norm
            for token in tokens
            if token.norm not in lexicon.DETERMINERS
        ]
''',
)

replace_once(
    "clanker_lm/memory.py",
    '''        if len(direct) > 1:
            ranked = sorted(direct, key=lambda entity: (entity.last_mentioned_turn, entity.salience), reverse=True)
            if (
''',
    '''        if len(direct) > 1:
            ranked = sorted(direct, key=lambda entity: (entity.last_mentioned_turn, entity.salience), reverse=True)
            if all("modifier_signature" in entity.attributes for entity in ranked):
                return Resolution(
                    "ambiguous",
                    candidates=ranked,
                    reason="multiple modified entities share that generic alias",
                )
            if (
''',
)

replace_once(
    "tests/test_subordinate_relations.py",
    '''    assert restored.SNAPSHOT_VERSION == 2
''',
    '''    assert restored.SNAPSHOT_VERSION == ConversationMemory.SNAPSHOT_VERSION
''',
)
