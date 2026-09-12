"""Versioned equation evidence; never a table of completed language responses.

Equality entries are explicit ground compound->atom lessons. The proof engine
may substitute them anywhere the same typed expression occurs. Hashes detect
mutation, not teacher authenticity or truth; access is checked by the host.
"""
from __future__ import annotations
from dataclasses import asdict
import copy
from types import MappingProxyType
from .terms import SCHEMA, Operator, Term, canonical, digest, identifier, validate


class EquationLedger:
    MAX_ROWS = 256

    def __init__(self, scope: str, operators: tuple[Operator, ...]):
        self.scope = identifier(scope)
        if (type(operators) is not tuple or not 1 <= len(operators) <= 16
                or any(not isinstance(o, Operator) for o in operators)):
            raise ValueError('bounded typed operator catalog required')
        if len({o.key for o in operators}) != len(operators):
            raise ValueError('duplicate operator identity')
        self.operators = MappingProxyType({o.key: o for o in operators})
        self._rows: list[dict] = []
        self.frozen = False

    def _append(self, body: dict) -> dict:
        if self.frozen:
            raise ValueError('equation learning is frozen')
        if len(self._rows) >= self.MAX_ROWS:
            raise ValueError('equation ledger budget exceeded')
        row = {**body, 'scope': self.scope, 'sequence': len(self._rows)+1,
               'previous': self._rows[-1]['digest'] if self._rows else '0'*64}
        row['digest'] = digest(row)
        self._rows.append(row)
        return copy.deepcopy(row)

    def teach(self, lhs: Term, rhs: Term, *, evidence_id: str, source_id: str,
              source_hash: str, origin: str = 'external_lesson') -> dict:
        for x in (evidence_id, source_id):
            identifier(x)
        validate(lhs, self.operators); validate(rhs, self.operators)
        if lhs.atomic or not rhs.atomic or lhs.sort != rhs.sort:
            raise ValueError('ground reduction must preserve sort and end in an atom')
        if origin not in {'external_lesson', 'reviewed_seed'}:
            raise ValueError('a hypothesis, contextual link or self-output is not an equation premise')
        if (not isinstance(source_hash, str) or len(source_hash) != 64
                or any(c not in '0123456789abcdef' for c in source_hash)):
            raise ValueError('source digest required')
        body = dict(kind='equation', lhs=lhs.to_dict(), rhs=rhs.to_dict(), evidence_id=evidence_id,
                    source_id=source_id, source_hash=source_hash, origin=origin)
        for old in self._rows:
            if old['kind'] == 'equation' and old['evidence_id'] == evidence_id:
                if all(old[k] == body[k] for k in body):
                    return copy.deepcopy(old)
                raise ValueError('equation evidence ID cannot be rebound')
        return self._append(body)

    def withdraw(self, evidence_id: str, *, reason: str) -> dict:
        identifier(evidence_id)
        if not isinstance(reason, str) or not reason.strip() or len(reason.encode()) > 1024:
            raise ValueError('bounded withdrawal reason required')
        original = next((r for r in self._rows if r['kind'] == 'equation'
                         and r['evidence_id'] == evidence_id), None)
        if original is None:
            raise KeyError('unknown equation evidence')
        existing = next((r for r in self._rows if r['kind'] == 'withdraw'
                         and r['evidence_id'] == evidence_id), None)
        if existing:
            return copy.deepcopy(existing)
        return self._append(dict(kind='withdraw', evidence_id=evidence_id,
                                 original_digest=original['digest'], reason_digest=digest(reason)))

    def active(self) -> tuple[dict, ...]:
        retired = {r['evidence_id'] for r in self._rows if r['kind'] == 'withdraw'}
        return tuple(copy.deepcopy(r) for r in self._rows
                     if r['kind'] == 'equation' and r['evidence_id'] not in retired)

    @property
    def generation(self) -> str:
        return digest([SCHEMA, self.scope, [asdict(self.operators[k]) for k in sorted(self.operators)],
                       self._rows[-1]['digest'] if self._rows else None])

    def to_dict(self) -> dict:
        return {'schema': SCHEMA, 'scope': self.scope,
                'operators': [asdict(self.operators[k]) for k in sorted(self.operators)],
                'rows': copy.deepcopy(self._rows), 'frozen': self.frozen}

    @classmethod
    def from_dict(cls, data: dict) -> 'EquationLedger':
        if (not isinstance(data, dict) or set(data) != {'schema','scope','operators','rows','frozen'}
                or data['schema'] != SCHEMA or type(data['rows']) is not list
                or len(data['rows']) > cls.MAX_ROWS or type(data['frozen']) is not bool
                or type(data['operators']) is not list or not 1 <= len(data['operators']) <= 16):
            raise ValueError('invalid equation snapshot')
        obj = cls(data['scope'], tuple(Operator(**o) for o in data['operators']))
        for row in data['rows']:
            if not isinstance(row, dict):
                raise ValueError('invalid ledger row')
            body = {k:v for k,v in row.items() if k not in {'digest','scope','sequence','previous'}}
            if row.get('scope') != obj.scope or row.get('sequence') != len(obj._rows)+1:
                raise ValueError('equation scope/sequence mismatch')
            if row.get('previous') != (obj._rows[-1]['digest'] if obj._rows else '0'*64):
                raise ValueError('equation chain mismatch')
            if row.get('digest') != digest({k:v for k,v in row.items() if k != 'digest'}):
                raise ValueError('equation content mismatch')
            if body.get('kind') == 'equation':
                try:
                    result = obj.teach(Term.from_dict(body['lhs']), Term.from_dict(body['rhs']),
                        **{k:body[k] for k in ('evidence_id','source_id','source_hash','origin')})
                except (KeyError, TypeError) as exc:
                    raise ValueError('invalid equation row') from exc
                if result != row:
                    raise ValueError('equation duplicate or schema mismatch')
            elif body.get('kind') == 'withdraw':
                if set(body) != {'kind','evidence_id','original_digest','reason_digest'}:
                    raise ValueError('invalid withdrawal schema')
                source = next((r for r in obj.active() if r['evidence_id'] == body['evidence_id']), None)
                if not source or source['digest'] != body['original_digest']:
                    raise ValueError('withdrawal does not bind active source')
                rd = body['reason_digest']
                if not isinstance(rd, str) or len(rd) != 64 or any(c not in '0123456789abcdef' for c in rd):
                    raise ValueError('invalid reason digest')
                obj._append(body)
            else:
                raise ValueError('unknown ledger row kind')
        obj.frozen = data['frozen']
        return obj
