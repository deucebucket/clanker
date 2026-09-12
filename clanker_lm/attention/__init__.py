"""Optional, scoped relevance propagation; not a fact or authentication engine."""
from .types import WaveConfig
from .wave import WaveIndex

__all__ = ['WaveConfig', 'WaveIndex']
