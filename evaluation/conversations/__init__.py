"""Versioned whole-conversation evaluation for Clanker-LM."""

from .corpus import (
    CORPUS_VERSION,
    CorpusIntegrityError,
    load_manifest,
    load_split,
    verify_corpus,
)

__all__ = [
    "CORPUS_VERSION",
    "CorpusIntegrityError",
    "load_manifest",
    "load_split",
    "verify_corpus",
]

