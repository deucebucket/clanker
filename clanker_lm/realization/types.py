"""Local composition type aliases, not response content."""

from __future__ import annotations
from typing import Union
from ..database import Atom

Part = Union[str, Atom]
