"""Opt-in resolver bridge to the EXISTING runtime and atomic word decoder.

Use 'Derive ...' for premise-only reasoning; ordinary 'What is ...?' retains
its existing calculator route. Equations are taught through the explicit API.
No user claims can install operation laws. No default runtime/snapshot change.
"""
from __future__ import annotations
import re
from .ledger import EquationLedger
from .numeric import NATURAL_ADD, parse_addition, validate_numeric_lesson
from .prover import Limits, derive, verify
from .terms import Term, atom, digest
from ..model import AnswerContract, AnswerStatus, EntityKind, EventFrame, SemanticRef, SourceKind
from ..resolvers import ResolverOutcome


class CompositionResolver:
    name = 'premise_composition'
    PREFIX = re.compile(r'^\s*derive\s+(.+?)\s*[?.!]?\s*$', re.IGNORECASE)

    def __init__(self, ledger: EquationLedger, *, scope: str, limits: Limits | None = None):
        if scope != ledger.scope:
            raise PermissionError('composition ledger belongs to another scope')
        if dict(ledger.operators) != {NATURAL_ADD.key: NATURAL_ADD}:
            raise ValueError('numeric provider requires the reviewed exact-addition catalog')
        # Deserialized or externally supplied numeric lessons are checked at
        # binding time. The target proof never calls a numeric evaluator.
        for row in ledger.active():
            validate_numeric_lesson(Term.from_dict(row['lhs']), Term.from_dict(row['rhs']))
        self.ledger, self.scope = ledger, scope
        self.limits = limits or Limits()
        if not isinstance(self.limits, Limits):
            raise TypeError('typed limits required')
        self._validated_generation = ledger.generation

    def teach(self, expression: str, value: int, *, evidence_id: str, source_id: str):
        if self.ledger.generation != self._validated_generation:
            raise ValueError('equations changed outside the validated ingress')
        if type(value) is not int:
            raise ValueError('integer lesson result required')
        lhs, rhs = parse_addition(expression), atom('natural', str(value))
        validate_numeric_lesson(lhs, rhs)
        result = self.ledger.teach(lhs, rhs, evidence_id=evidence_id, source_id=source_id,
            source_hash=digest({'lhs':lhs.to_dict(),'rhs':rhs.to_dict(),'source':source_id}))
        self._validated_generation = self.ledger.generation
        return result

    def withdraw(self, evidence_id: str, *, reason: str):
        if self.ledger.generation != self._validated_generation:
            raise ValueError('equations changed outside the validated ingress')
        row = self.ledger.withdraw(evidence_id, reason=reason)
        self._validated_generation = self.ledger.generation
        return row

    def resolve(self, text, registry):
        match = self.PREFIX.fullmatch(text)
        if not match:
            return ResolverOutcome()
        if self.ledger.generation != self._validated_generation:
            raise ValueError('numeric premises changed without validation; rebind explicitly')
        expression = match.group(1).strip()
        metadata = {'command':'DERIVE_FROM_PREMISES','source':self.name,
                    'observed_at':registry.now_utc().isoformat(), 'target_calculator_used':False}
        try:
            term = parse_addition(expression)
        except ValueError as exc:
            return ResolverOutcome(True, AnswerContract(status=AnswerStatus.UNSUPPORTED,
                certainty=0, source=SourceKind.UNKNOWN, reason=str(exc), response_goal='clarify'),
                {**metadata,'error':'unsupported_typed_expression'})
        receipt = derive(term, self.ledger, scope=self.scope, limits=self.limits)
        verify(receipt, term, self.ledger, scope=self.scope, limits=self.limits)
        metadata['composition'] = receipt
        if receipt['status'] != 'proved':
            # Absence/budget statuses are explicit in the resolver receipt; the
            # legacy contract has no separate incomplete enumeration.
            return ResolverOutcome(True, AnswerContract(status=AnswerStatus.UNKNOWN,
                certainty=0, source=SourceKind.UNKNOWN,
                reason='no complete unconflicted derivation from the active premises',
                response_goal='answer', forbidden_claims=['invent_missing_fact'],
                required_slots={'composition_status':receipt['status']}), metadata)
        value = receipt['result']['symbol']
        proposition = EventFrame(predicate='be', arguments={
            'subject':SemanticRef.literal(expression, expression, EntityKind.ABSTRACT),
            'value':SemanticRef.literal(value, value, EntityKind.ABSTRACT)},
            source=SourceKind.EXTERNAL, certainty=255)
        return ResolverOutcome(True, AnswerContract(status=AnswerStatus.ANSWERED,
            proposition=proposition, values=[proposition.arguments['value']], source=SourceKind.EXTERNAL,
            certainty=255, reason='checked equality composition over validated numeric premises',
            response_goal='answer', required_slots={'requested_role':'value','composition_proof':receipt['digest']},
            forbidden_claims=[]), metadata)


def bind(runtime, ledger: EquationLedger, *, limits: Limits | None = None) -> CompositionResolver:
    """Host-owned explicit binding; a matching string is not authentication."""
    scope = runtime.learner.scope_id
    if ledger.scope != scope:
        raise PermissionError('runtime and equation scopes disagree')
    if any(r.name == CompositionResolver.name for r in runtime.resolvers._resolvers):
        raise ValueError('composition provider already installed')
    provider = CompositionResolver(ledger, scope=scope, limits=limits)
    runtime.resolvers.register(provider)
    return provider
