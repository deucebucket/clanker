"""Clanker Pipeline — demo package (V2 archived to archive/v2_demo/).

Only V5-current files remain here. V2 engine modules live in archive/v2_demo/.
The engine/ package imports demo.zones and demo.shared — keep those.
"""

from .shared import VADUG, VADU, MetadataHeader, PersonalityVector
from .forces_curated import EMOTIONAL_VOCABULARY
from .idioms import IDIOMS
from .fuzzy import fuzzy_match
from .zones import ZONES

__all__ = [
    "VADUG", "VADU", "MetadataHeader", "PersonalityVector",
    "EMOTIONAL_VOCABULARY", "IDIOMS",
    "fuzzy_match",
    "ZONES",
]
