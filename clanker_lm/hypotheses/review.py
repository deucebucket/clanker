"""Study revises support; resemblance and repetition never count as proof."""
from .records import Claim, SCHEMA
from .evidence import origin_groups
from .propose import route_is_active
from ..activation import fingerprint


def assess(routes, view, rules):
    rules = {r.key:r for r in rules}
    if not routes: raise ValueError('candidate needs a motivating route')
    target = Claim.from_dict(routes[0]['claim'])
    hid = routes[0]['id']
    active_routes = []
    for route in routes:
        if route['id'] != hid or Claim.from_dict(route['claim']) != target:
            raise ValueError('mixed candidate identities')
        if route['rule'] not in rules: raise ValueError('unknown proposal rule')
        if route_is_active(route, view, rules[route['rule']]): active_routes.append(route['digest'])
    supporting, opposing, adjacent = [], [], []
    for key, row in sorted(view.active.items()):
        claim = Claim.from_dict(row['claim'])
        if claim == target: supporting.append(key)
        elif claim == target.opposite(): opposing.append(key)
        elif set(claim.arguments) & set(target.arguments): adjacent.append(key)
    state = ('contested' if supporting and opposing else 'supported_by_recorded_evidence' if supporting
             else 'opposed_by_recorded_evidence' if opposing else 'proposed' if active_routes else 'stale')
    plus = origin_groups([view.active[k] for k in supporting])
    minus = origin_groups([view.active[k] for k in opposing])
    result = {'schema': SCHEMA, 'hypothesis_id': hid, 'scope': view.scope,
              'claim': target.to_dict(), 'status': state, 'active_routes': sorted(active_routes),
              'supporting_evidence': supporting, 'opposing_evidence': opposing,
              'adjacent_not_confirming': adjacent,
              'support_origin_groups': plus, 'opposition_origin_groups': minus,
              'origin_independence': 'host_declared_not_independently_verified',
              'probability': None, 'calibration': 'not_established',
              'is_proof': False, 'authorized_for_factual_inference': False,
              'support_obligations': ['independent_verification', 'seek_counterevidence'],
              'motivating_obligations': sorted({k for r in routes for k in r['obligations']})}
    # Identity binds the relevant projection, not number of repeated review calls.
    result['digest'] = fingerprint(result)
    return result


def verify_assessment(receipt, routes, view, rules):
    if receipt != assess(routes, view, rules):
        raise ValueError('hypothesis review does not reproduce against live evidence')
