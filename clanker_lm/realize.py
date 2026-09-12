"""Stable SurfaceRealizer API composed from responsibility-specific components.

This is a behavior-preserving decomposition, not a new generation path.
"""

from .realization.types import Part
from .realization.routing import RoutingComponent
from .realization.assembly import AssemblyComponent
from .realization.answers import AnswersComponent
from .realization.uncertainty import UncertaintyComponent
from .realization.conflicts import ConflictsComponent
from .realization.clarification import ClarificationComponent
from .realization.social import SocialComponent
from .realization.references import ReferencesComponent
from .realization.embedded import EmbeddedComponent
from .realization.complements import ComplementsComponent
from .realization.events import EventsComponent
from .realization.validation import ValidationComponent


class SurfaceRealizer(
    RoutingComponent,
    AssemblyComponent,
    AnswersComponent,
    UncertaintyComponent,
    ConflictsComponent,
    ClarificationComponent,
    SocialComponent,
    ReferencesComponent,
    EmbeddedComponent,
    ComplementsComponent,
    EventsComponent,
    ValidationComponent,
):
    """Compose contract-preserving replies without whole-sentence templates."""

__all__ = ('SurfaceRealizer', 'Part')
