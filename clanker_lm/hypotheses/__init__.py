"""Typed hypothesis proposals, never factual or identity proof.

Use .runtime.HypothesisLab where the integrated MemoryWeb is installed.
The portable matcher and evidence validator require no chat engine.
"""
from .records import Claim, Limits, Pattern, ProposalRule
from .propose import propose
from .review import assess

__all__ = ['Claim', 'Limits', 'Pattern', 'ProposalRule', 'propose', 'assess']
