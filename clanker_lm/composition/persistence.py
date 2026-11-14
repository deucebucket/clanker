"""Explicit paired checkpoint. Default runtime saves do NOT capture this addon.

Use only under the host's authorized scope; the checksum is not an access key
or a replacement for the existing authenticated compartment service. The addon
is not yet installed in that service's standard snapshot format.
"""
from __future__ import annotations
from dataclasses import asdict
from .ledger import EquationLedger
from .prover import Limits
from .resolver import bind
from .terms import digest

FORMAT = 'clanker-composition-session-v1'


def capture(runtime, provider) -> dict:
    if (runtime.learner.scope_id != provider.scope or provider.ledger.scope != provider.scope
            or provider not in runtime.resolvers._resolvers
            or provider._validated_generation != provider.ledger.generation):
        raise ValueError('runtime, provider and active equations are not bound')
    data = {'schema':FORMAT, 'scope':provider.scope, 'runtime':runtime.to_dict(),
            'equations':provider.ledger.to_dict(), 'limits':asdict(provider.limits)}
    data['digest'] = digest(data)
    return data


def restore(data: dict, *, expected_scope: str, clock=None):
    if not isinstance(data, dict) or set(data) != {'schema','scope','runtime','equations','limits','digest'}:
        raise ValueError('invalid paired composition checkpoint')
    if (data['scope'] != expected_scope or data['equations'].get('scope') != expected_scope
            or data['runtime'].get('learner', {}).get('scope_id') != expected_scope):
        raise PermissionError('checkpoint outside the authorized expected scope')
    if data['schema'] != FORMAT or data['digest'] != digest({k:v for k,v in data.items() if k!='digest'}):
        raise ValueError('checkpoint identity mismatch')
    ledger = EquationLedger.from_dict(data['equations'])
    limits = Limits(**data['limits'])
    from ..runtime import ClankerLM
    runtime = ClankerLM.from_dict(data['runtime'], clock=clock)
    try:
        provider = bind(runtime, ledger, limits=limits)
    except Exception:
        runtime.close()
        raise
    return runtime, provider
