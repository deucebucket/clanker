"""
Phin-Lang Decoder — Reference implementation.

Decodes Phin text-format scripts into any language via dictionary lookup.
"""

from .decoder import decode, PhinDecoder
from .loader import DictionaryLoader

__version__ = "0.1.0"
__all__ = ["decode", "PhinDecoder", "DictionaryLoader"]
