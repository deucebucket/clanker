"""Ordered construction dispatch, preserving the original precedence exactly."""
from __future__ import annotations
from typing import Optional, Sequence
from ... import lexicon
from ..context import ParseContext
from ..ports import ParserPort
from . import embedded, gerund, infinitival, content, relative, appositive, subordinate, ordinary

STAGES = (embedded.apply, gerund.apply, infinitival.apply, content.apply,
          relative.apply, appositive.apply, subordinate.apply, ordinary.apply)


def assemble(parser: ParserPort, clause: Sequence[lexicon.Token],
             connector: Optional[str], ctx: ParseContext) -> None:
    for stage in STAGES:
        if stage(parser, clause, connector, ctx):
            return
    raise RuntimeError("parser assembly pipeline has no terminal handler")
