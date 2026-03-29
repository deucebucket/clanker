"""
Clanker-Lang Decoder — Reference implementation.

Decodes Clanker text-format scripts into any language via dictionary lookup.
Compiles/decompiles between .clank.txt text format and .clank binary bytecode.
"""

from .decoder import decode, ClankerDecoder
from .loader import DictionaryLoader
from .compiler import ClankerCompiler, ClankerDecompiler, CompilerError

__version__ = "0.1.0"
__all__ = [
    "decode",
    "ClankerDecoder",
    "DictionaryLoader",
    "ClankerCompiler",
    "ClankerDecompiler",
    "CompilerError",
]
