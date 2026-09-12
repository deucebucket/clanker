"""Source-qualified views of observations in the EXISTING MemoryWeb.

Host-supplied origin sets group known dependencies; they do not certify source
independence, accuracy or authenticity. No new factual store or authority here.
"""
from copy import deepcopy
from .records import ALLOWED_SOURCES, Claim, Limits, SCHEMA
from ..activation import fingerprint, identifier

OBSERVATION = 'study_observation'
WITHDRAWAL = 'study_withdrawal'
REFERENCE_KINDS = frozenset({'concept', 'entity', 'value', 'study_term'})


def observation_id(scope, evidence_id):
    return 'study:' + fingerprint([scope, evidence_id])[:32]


def record_body(scope, claim, *, evidence_id, origins, source_kind, content_hash):
    identifier(scope); identifier(evidence_id)
    if not isinstance(claim, Claim): raise TypeError('typed claim required')
    if source_kind not in ALLOWED_SOURCES:
        raise ValueError('hypothesis/self-output/analogy is not external observation evidence')
    if (type(origins) is not tuple or not 1 <= len(origins) <= 16
            or len(set(origins)) != len(origins)):
        raise ValueError('declared source lineage required')
    for root in origins:
        identifier(root)
        if root.startswith(('hypothesis:', 'study-review:', 'self_output:')):
            raise ValueError('candidate output cannot become independent evidence')
    if (not isinstance(content_hash, str) or len(content_hash) != 64
            or any(c not in '0123456789abcdef' for c in content_hash)):
        raise ValueError('source content hash required')
    body = {'schema': SCHEMA, 'scope': scope, 'claim': claim.to_dict(),
            'evidence_id': evidence_id, 'origins': sorted(origins),
            'source_kind': source_kind, 'content_hash': content_hash}
    body['digest'] = fingerprint(body)
    return body


def validate_record(body, scope):
    if not isinstance(body, dict) or set(body) != {'schema', 'scope', 'claim', 'evidence_id',
                                                  'origins', 'source_kind', 'content_hash', 'digest'}:
        raise ValueError('invalid study evidence record')
    if body['scope'] != scope: raise PermissionError('study evidence outside authorized scope')
    expected = record_body(scope, Claim.from_dict(body['claim']), evidence_id=body['evidence_id'],
                          origins=tuple(body['origins']), source_kind=body['source_kind'],
                          content_hash=body['content_hash'])
    if body != expected: raise ValueError('study evidence identity changed')


class EvidenceView:
    def __init__(self, web, scope, limits=None):
        self.scope = scope
        limits = limits or Limits()
        records, withdrawals = {}, []
        for node in web.nodes.values():
            if node['kind'] == OBSERVATION:
                body = node.get('record')
                validate_record(body, scope)
                expected = observation_id(scope, body['evidence_id'])
                if node['id'] != expected: raise ValueError('study node identity changed')
                for key in Claim.from_dict(body['claim']).arguments:
                    if key not in web.nodes or web.nodes[key]['kind'] not in REFERENCE_KINDS:
                        raise ValueError('study claim has an invalid reference')
                if len(records) >= limits.max_rows: raise ValueError('study evidence budget exceeded')
                records[expected] = deepcopy(body)
            elif node['kind'] == WITHDRAWAL:
                body = node.get('record', {})
                if set(body) != {'target', 'target_digest', 'scope', 'reason_hash', 'digest'}:
                    raise ValueError('invalid withdrawal record')
                if body['scope'] != scope: raise PermissionError('withdrawal outside authorized scope')
                if body['digest'] != fingerprint({k:v for k,v in body.items() if k != 'digest'}):
                    raise ValueError('withdrawal changed')
                if node['id'] != 'withdraw:' + body['target']:
                    raise ValueError('withdrawal identity changed')
                withdrawals.append(body)
        retired = set()
        for body in withdrawals:
            if body['target'] not in records or records[body['target']]['digest'] != body['target_digest']:
                raise ValueError('withdrawal lost its original evidence')
            retired.add(body['target'])
        self.records = records
        self.active = {k:v for k,v in records.items() if k not in retired}
        self.generation = fingerprint([scope, sorted((k,v['digest']) for k,v in records.items()),
                                       sorted(b['digest'] for b in withdrawals)])


def origin_groups(rows):
    """Connected components of overlapping DECLARED source-root sets."""
    groups = []
    for row in sorted(rows, key=lambda r:r['evidence_id']):
        new = set(row['origins'])
        rest = []
        for old in groups:
            if old & new: new |= old
            else: rest.append(old)
        groups = rest + [new]
    # Repeat to handle groups bridged after an earlier group was visited.
    changed = True
    while changed:
        changed = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if groups[a] & groups[b]:
                    groups[a] |= groups.pop(b); changed = True; break
            if changed: break
    return sorted(sorted(g) for g in groups)
