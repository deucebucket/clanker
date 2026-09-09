"""Apply only the reviewed cessation seam to the pinned native parser.

The integrated delivery has additional continuation logic that is not in this
native parent. Do not copy that unrelated parser or relax patch verification.
"""
from pathlib import Path
import subprocess

path = Path('clanker_lm/parser.py')
blob = subprocess.check_output(['git', 'hash-object', str(path)], text=True).strip()
if blob != 'c84980500a0893441948767aea1759399bb30630':
    raise RuntimeError('native parser does not match the reviewed parent: ' + blob)
text = path.read_text(encoding='utf-8')
replacements = [
    ('from . import lexicon\n',
     'from . import lexicon\nfrom .cessation import scan_cessation, CESSATION_KIND, CESSATION_ROLE\n'),
    ('''        main_token = main_items[verb_idx]
        predicate = lexicon.lemma(main_token.norm)
        auxiliary_tokens = [token.norm for token in main_items[:verb_idx] if token.norm in lexicon.AUXILIARIES]
        modality = next((word for word in auxiliary_tokens if word in lexicon.MODALS), None)
        polarity = not any(
            token.norm in lexicon.NEGATORS for token in main_items
        )
''', '''        main_token = main_items[verb_idx]
        predicate = lexicon.lemma(main_token.norm)
        phase = scan_cessation(main_items, predicate, verb_idx)
        if phase.error:
            return ClauseResult(None, diagnostics=["unresolved state change: " + phase.error])
        if phase.marker:
            main_items = list(phase.tokens)
            verb_idx = next(i for i,t in enumerate(main_items) if t is main_token)
            diagnostics.append("typed state cessation: " + phase.marker)
        auxiliary_tokens = [token.norm for token in main_items[:verb_idx] if token.norm in lexicon.AUXILIARIES]
        modality = next((word for word in auxiliary_tokens if word in lexicon.MODALS), None)
        polarity = not (phase.marker or any(
            token.norm in lexicon.NEGATORS for token in main_items
        ))
'''),
    ('''        args: Dict[str, SemanticRef] = {}
        subject_role = self._subject_role(predicate, passive)
''', '''        args: Dict[str, SemanticRef] = {}
        if phase.marker:
            args[CESSATION_ROLE] = SemanticRef.literal(CESSATION_KIND, phase.marker, EntityKind.ABSTRACT)
        subject_role = self._subject_role(predicate, passive)
'''),
]
for old, new in replacements:
    if text.count(old) != 1:
        raise RuntimeError('expected exactly one native seam: ' + repr(old[:100]))
    text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')
