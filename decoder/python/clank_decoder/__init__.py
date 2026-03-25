"""
Clank-Lang Decoder — Reference implementation.

Decodes Clank text-format scripts into any language via dictionary lookup.
"""

from .decoder import decode, ClankDecoder
from .loader import DictionaryLoader

__version__ = "0.1.0"
__all__ = ["decode", "ClankDecoder", "DictionaryLoader"]
