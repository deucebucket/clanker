"""
Clanker-Lang Decoder — Reference implementation.

Decodes Clanker text-format scripts into any language via dictionary lookup.
"""

from .decoder import decode, ClankerDecoder
from .loader import DictionaryLoader

__version__ = "0.1.0"
__all__ = ["decode", "ClankerDecoder", "DictionaryLoader"]
