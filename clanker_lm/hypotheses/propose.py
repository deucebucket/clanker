"""Bounded relational unification proposes claims; it never proves them."""
from dataclasses import asdict
from .records import Claim, Limits, ProposalRule, SCHEMA, instantiate, unify
from ..activation import fingerprint


def propose(view, rules, limits=None):
    limits = limits or Limits()
    rules = tuple(rules)
    if (not 1 <= len(rules) <= 16 or any(not isinstance(r, ProposalRule) for r in rules)
            or len({r.key for r in rules}) != len(rules)):
        raise ValueError('bounded unique proposal rules required')
    indexed = {}
    for key, row in sorted(view.active.items()):
        c = Claim.from_dict(row['claim'])
        indexed.setdefault((c.predicate, c.positive), []).append((key, row, c))
    checks, candidates, seen, stopped = 0, [], set(), None
    for rule in sorted(rules, key=lambda r:r.key):
        # DFS holds at most bounded stack entries. Rows and rule order are stable.
        stack = [(0, {}, (), None)]
        while stack:
            index, binding, evidence, conditions = stack.pop()
            if index == len(rule.premises):
                if any(binding[a] == binding[b] for a,b in rule.distinct): continue
                conclusion = instantiate(rule.conclusion, binding, conditions)
                if any(conclusion.to_dict() == view.active[k]['claim'] for k in evidence): continue
                body = {'schema': SCHEMA, 'scope': view.scope, 'rule': rule.key,
                        'rule_digest': rule.identity, 'claim': conclusion.to_dict(),
                        'bindings': dict(sorted(binding.items())),
                        'premises': [{'node': k, 'digest': view.active[k]['digest']} for k in evidence],
                        'obligations': list(rule.obligations), 'kind': 'hypothesis_not_proof'}
                body['id'] = 'hypothesis:' + fingerprint([view.scope, rule.identity, body['claim']])[:32]
                body['digest'] = fingerprint(body)
                if body['digest'] in seen: continue
                if len(candidates) >= limits.max_candidates:
                    stopped = 'candidate_budget'; break
                seen.add(body['digest']); candidates.append(body)
                continue
            pattern = rule.premises[index]
            for key, row, claim in reversed(indexed.get((pattern.predicate, pattern.positive), ())):
                if checks >= limits.max_checks:
                    stopped = 'match_budget'; break
                checks += 1
                if key in evidence or (conditions is not None and conditions != claim.conditions): continue
                next_binding = unify(pattern, claim, binding)
                if next_binding is None: continue
                if any(a in next_binding and b in next_binding and next_binding[a] == next_binding[b]
                       for a,b in rule.distinct): continue
                stack.append((index + 1, next_binding, evidence + (key,), claim.conditions))
            if stopped: break
        if stopped: break
    report = {'schema': SCHEMA, 'scope': view.scope, 'evidence_generation': view.generation,
              'rules': {r.key:r.identity for r in rules}, 'limits': asdict(limits),
              'status': 'incomplete' if stopped else 'complete_within_proposal_schemas',
              'stop_reason': stopped, 'match_checks': checks, 'candidates': candidates,
              'probability_estimated': False, 'facts_committed': False}
    report['digest'] = fingerprint(report)
    return report


def route_is_active(route, view, rule):
    """Recheck actual premise bindings, not merely a caller-provided hash."""
    if route.get('scope') != view.scope: raise PermissionError('candidate outside authorized scope')
    if (route.get('rule_digest') != rule.identity or route.get('rule') != rule.key
            or route.get('schema') != SCHEMA or route.get('kind') != 'hypothesis_not_proof'
            or route.get('digest') != fingerprint({k:v for k,v in route.items() if k!='digest'})
            or len(route.get('premises', [])) != len(rule.premises)):
        raise ValueError('candidate schema or rule identity changed')
    binding, conditions, seen, available = {}, None, set(), True
    for pattern, premise in zip(rule.premises, route['premises']):
        key = premise['node']
        if key not in view.records: raise ValueError('candidate source record missing')
        original = view.records[key]
        if original['digest'] != premise['digest']: raise ValueError('candidate source changed')
        if key not in view.active: available = False
        claim = Claim.from_dict(original['claim'])
        if key in seen or (conditions is not None and claim.conditions != conditions):
            raise ValueError('candidate premise scope changed')
        seen.add(key); conditions = claim.conditions
        binding = unify(pattern, claim, binding)
        if binding is None: raise ValueError('candidate premise does not match proposal schema')
        if any(Claim.from_dict(r['claim']) == claim.opposite() for r in view.active.values()):
            available = False
    target = instantiate(rule.conclusion, binding, conditions)
    if (binding != route['bindings'] or target.to_dict() != route['claim']
            or any(binding[a] == binding[b] for a,b in rule.distinct)
            or route['obligations'] != list(rule.obligations)
            or route['id'] != 'hypothesis:' + fingerprint([view.scope, rule.identity, route['claim']])[:32]):
        raise ValueError('candidate mapping, obligations or identity changed')
    return available
