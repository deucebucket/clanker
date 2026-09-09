"""Apply only exact shared native seams; no integrated runtime is copied."""
from pathlib import Path
root=Path('.')
def change(file,old,new):
    p=root/file;s=p.read_text(encoding='utf-8')
    if s.count(old)!=1:raise RuntimeError('unmatched native seam: '+file+': '+repr(old[:80]))
    p.write_text(s.replace(old,new,1),encoding='utf-8')
change('clanker_lm/parser.py',
 'from .cessation import scan_cessation, CESSATION_KIND, CESSATION_ROLE\n',
 'from .cessation import scan_cessation, CESSATION_KIND, CESSATION_ROLE\nfrom .recurrence import scan_recurrence, RECURRENCE_ROLE, RECURRENCE_KIND\n')
change('clanker_lm/parser.py',
 '        auxiliary_tokens = [token.norm for token in main_items[:verb_idx] if token.norm in lexicon.AUXILIARIES]\n        modality = next((word for word in auxiliary_tokens if word in lexicon.MODALS), None)\n',
 '''        recurrence = scan_recurrence(main_items, predicate, verb_idx, cessation=bool(phase.marker))
        if recurrence.error:
            return ClauseResult(None, diagnostics=["unresolved state recurrence: " + recurrence.error])
        if recurrence.marker:
            main_items = list(recurrence.tokens)
            verb_idx = next(i for i,t in enumerate(main_items) if t is main_token)
            diagnostics.append("typed state recurrence: " + recurrence.marker)
        auxiliary_tokens = [token.norm for token in main_items[:verb_idx] if token.norm in lexicon.AUXILIARIES]
        modality = next((word for word in auxiliary_tokens if word in lexicon.MODALS), None)
''')
change('clanker_lm/parser.py',
 '        if phase.marker:\n            args[CESSATION_ROLE] = SemanticRef.literal(CESSATION_KIND, phase.marker, EntityKind.ABSTRACT)\n',
 '''        if phase.marker:
            args[CESSATION_ROLE] = SemanticRef.literal(CESSATION_KIND, phase.marker, EntityKind.ABSTRACT)
        if recurrence.marker:
            args[RECURRENCE_ROLE] = SemanticRef.literal(RECURRENCE_KIND, recurrence.marker, EntityKind.ABSTRACT)
''')
change('clanker_lm/memory.py','        # Exact repeated assertions update certainty/provenance rather than\n',
 '''        # State reports are time-ordered observations: merging a later report
        # into an earlier row would rewrite the positive/ceased/renewed sequence.
        # Ordinary durable facts still use the existing duplicate policy.
        from .recurrence import preserve_state_occurrence
        retain_observation = preserve_state_occurrence(stored)
        # Exact repeated assertions update certainty/provenance rather than
''')
change('clanker_lm/memory.py',
 '                and existing.source == stored.source\n            ):\n',
 '                and existing.source == stored.source\n                and not (retain_observation and existing.turn_index != stored.turn_index)\n            ):\n')
change('clanker_lm/state_scope.py','from .cessation import marker_of\n','from .cessation import marker_of\nfrom .recurrence import recurrence_of\n')
change('clanker_lm/state_scope.py','        marker_of(event)\n','        marker_of(event)\n        recurrence_of(event)\n')
change('clanker_lm/state_descriptions.py',
 'from .cessation import CESSATION_ROLE, marker_of, withdrawal_matches\n',
 'from .cessation import CESSATION_ROLE, marker_of, withdrawal_matches\nfrom .recurrence import RECURRENCE_ROLE, recurrence_of\n')
change('clanker_lm/state_descriptions.py','SCHEMA = "qualified-state-components-v2"','SCHEMA = "qualified-state-components-v3-recurrence"')
change('clanker_lm/state_descriptions.py','        marker_of(event)\n','        marker_of(event)\n        recurrence = recurrence_of(event)\n')
change('clanker_lm/state_descriptions.py',
 '{subject_roles[0], value_roles[0], "attribute", "time", CESSATION_ROLE}',
 '{subject_roles[0], value_roles[0], "attribute", "time", CESSATION_ROLE, RECURRENCE_ROLE}')
change('clanker_lm/state_descriptions.py','    if len(parts) > 1 and (not event.polarity or event.modality):\n',
 '    if len(parts) > 1 and (not event.polarity or event.modality or recurrence):\n')
change('clanker_lm/state_descriptions.py',
 '                    or row.get("state_change") != marker_of(e) or expected_id in seen):\n',
 '                    or row.get("state_change") != marker_of(e)\n                    or row.get("state_recurrence") != recurrence_of(e) or expected_id in seen):\n')
change('clanker_lm/state_descriptions.py',
 '    selected=[v for v in latest.values() if not is_who or v[1].report.polarity==q.event.polarity]\n',
 '''    selected=[v for v in latest.values() if not is_who or v[1].report.polarity==q.event.polarity]
    requested_recurrence = recurrence_of(query_event)
    if requested_recurrence:
        # An unqualified positive report does not prove renewed occurrence.
        # A current denial of the state is sufficient to deny its recurrence.
        selected = [v for v in selected if not v[1].report.polarity
                    or recurrence_of(v[1].event) == requested_recurrence]
''')
change('clanker_lm/state_descriptions.py',
 '                         "reporter_ids":["user"],"state_change":marker_of(e)})\n',
 '                         "reporter_ids":["user"],"state_change":marker_of(e),\n                         "state_recurrence":recurrence_of(e)})\n')
change('clanker_lm/state_descriptions.py','        elif CESSATION_ROLE in e.arguments:\n',
 '        elif CESSATION_ROLE in e.arguments or RECURRENCE_ROLE in e.arguments:\n')
change('clanker_lm/memory.py',
    '        for existing in self.events:\n            if (\n                existing.signature() == stored.signature()',
    '        for existing in self.events:\n            if (retain_observation and existing.event_id == stored.event_id\n                    and existing.turn_index != stored.turn_index):\n                raise ValueError("state observation identity cannot be reused across turns")\n            if (\n                existing.signature() == stored.signature()')
