"""Stable QuestionAnswerer API composed from responsibility-specific components.

This is a behavior-preserving decomposition, not a new generation path.
"""

from .answering.policy import PolicyComponent
from .answering.routing import RoutingComponent
from .answering.embedded import EmbeddedComponent
from .answering.gerund_queries import GerundQueriesComponent
from .answering.gerund_evidence import GerundEvidenceComponent
from .answering.infinitives import InfinitivesComponent
from .answering.attribution import AttributionComponent
from .answering.facts import FactsComponent
from .answering.matching import MatchingComponent


class QuestionAnswerer(
    PolicyComponent,
    RoutingComponent,
    EmbeddedComponent,
    GerundQueriesComponent,
    GerundEvidenceComponent,
    InfinitivesComponent,
    AttributionComponent,
    FactsComponent,
    MatchingComponent,
):
    """Bind a question's typed hole from explicit evidence in memory."""

__all__ = ('QuestionAnswerer',)
