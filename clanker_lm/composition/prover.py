"""Bounded equality composition: ground substitution and licensed associativity.

No target calculator is called. Search is complete only within this oriented
compound-to-atom calculus, not general equational theorem proving. Strong
contrary lesson values are reported rather than resolved by frequency.
"""
from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass
from .ledger import EquationLedger
from .terms import SCHEMA, Term, canonical, digest, law_key, replace_at, validate, walk


@dataclass(frozen=True)
class Limits:
    max_states: int = 1024
    max_steps: int = 8192
    max_proof_depth: int = 32

    def __post_init__(self):
        for name, bound in [('max_states',4096),('max_steps',32768),('max_proof_depth',64)]:
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= bound:
                raise ValueError('invalid proof budget')


def derive(query: Term, ledger: EquationLedger, *, scope: str, limits: Limits | None = None) -> dict:
    if scope != ledger.scope:
        raise PermissionError('equation proof outside bound scope')
    cfg = Limits() if limits is None else limits
    if not isinstance(cfg, Limits):
        raise TypeError('typed limits required')
    validate(query, ledger.operators)
    equations = []
    values_by_lhs = {}
    for r in sorted(ledger.active(), key=lambda r:r['evidence_id']):
        lhs, rhs = Term.from_dict(r['lhs']), Term.from_dict(r['rhs'])
        key = law_key(lhs, ledger.operators)
        values_by_lhs.setdefault(key, set()).add(rhs)
        equations.append((lhs, rhs, r))
    by_exact = {}
    for lhs, rhs, row in equations:
        by_exact.setdefault(lhs, []).append((rhs, row))
    queue = deque([(query, ())])
    seen = {query}
    solutions = {}
    conflict_ids = set()
    stop_reasons = set()
    examined = proposals = 0
    exhausted = False
    while queue and not exhausted:
        current, proof = queue.popleft()
        examined += 1
        if current.atomic:
            solutions.setdefault(current, proof)
            continue
        for path, node in walk(current):
            if node.atomic:
                continue
            key = law_key(node, ledger.operators)
            if len(values_by_lhs.get(key, ())) > 1:
                conflict_ids.update(r['evidence_id'] for lhs, _, r in equations
                                    if law_key(lhs, ledger.operators) == key)
            candidates = []
            for rhs, row in by_exact.get(node, ()):
                candidates.append((rhs, {'rule':'substitute_equal', 'evidence_id':row['evidence_id']}))
            op = ledger.operators[node.symbol]
            if op.associative:
                left, right = node.arguments
                if left.arguments and left.symbol == node.symbol:
                    a,b = left.arguments
                    grouped = Term(node.sort, op.key, (a, Term(node.sort, op.key, (b,right))))
                    candidates.append((grouped, {'rule':'associate_right','operator':op.key,'law_version':op.version}))
                if right.arguments and right.symbol == node.symbol:
                    b,c = right.arguments
                    grouped = Term(node.sort, op.key, (Term(node.sort, op.key, (left,b)), c))
                    candidates.append((grouped, {'rule':'associate_left','operator':op.key,'law_version':op.version}))
            for replacement, details in candidates:
                if proposals >= cfg.max_steps:
                    stop_reasons.add('step_budget'); exhausted = True; break
                proposals += 1
                nxt = replace_at(current, path, replacement)
                if nxt in seen:
                    continue
                if len(proof) >= cfg.max_proof_depth:
                    stop_reasons.add('proof_depth'); continue
                if len(seen) >= cfg.max_states:
                    stop_reasons.add('state_budget'); exhausted = True; break
                try:
                    validate(nxt, ledger.operators)
                except ValueError:
                    stop_reasons.add('term_budget')
                    continue
                seen.add(nxt)
                step = {**details, 'path':list(path), 'before':current.to_dict(), 'after':nxt.to_dict()}
                queue.append((nxt, proof + (step,)))
            if exhausted:
                break
    # Search exhaustion does not certify that no opposing route exists.
    status = ('conflict' if conflict_ids or len(solutions)>1 else
              'incomplete' if stop_reasons else 'proved' if solutions else 'unknown')
    ordered = sorted(solutions, key=lambda t:canonical(t.to_dict()))
    selected = ordered[0] if status == 'proved' else None
    proof = list(solutions[selected]) if selected else []
    used = sorted({s['evidence_id'] for s in proof if s['rule']=='substitute_equal'})
    result = dict(schema=SCHEMA, scope=scope, query=query.to_dict(),
                  ledger_generation=ledger.generation, limits=asdict(cfg),
                  status=status, result=selected.to_dict() if selected else None,
                  proof=proof, supporting_evidence=used,
                  supporting_sources=sorted({r['source_id'] for _, _, r in equations if r['evidence_id'] in used}),
                  conflicting_evidence=sorted(conflict_ids),
                  candidates=[v.to_dict() for v in ordered],
                  complete_within_ground_calculus=not stop_reasons,
                  stop_reasons=sorted(stop_reasons),
                  work=dict(states_seen=len(seen), states_examined=examined, rewrite_candidates=proposals),
                  independent_support_not_increased=True, target_calculator_used=False)
    result['digest'] = digest(result)
    return result


def verify(receipt: dict, query: Term, ledger: EquationLedger, *, scope: str,
           limits: Limits | None = None) -> None:
    if not isinstance(receipt, dict) or canonical(receipt) != canonical(derive(query, ledger, scope=scope, limits=limits)):
        raise ValueError('proof does not reproduce from the active typed premises')
