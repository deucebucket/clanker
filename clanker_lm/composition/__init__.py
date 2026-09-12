"""Optional typed ground-equality reasoning, distinct from relevance search."""
from .terms import Operator, Term, atom, apply
from .ledger import EquationLedger
from .prover import Limits, derive, verify

__all__ = ['Operator', 'Term', 'atom', 'apply', 'EquationLedger', 'Limits', 'derive', 'verify']
