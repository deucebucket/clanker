"""Affect data contracts; no runtime or persistence dependencies."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional
from .serialization import _json_safe


@dataclass(frozen=True)
class AffectVector:
    v: int = 128
    a: int = 128
    d: int = 128
    u: int = 0
    g: int = 128
    w: int = 128
    i: int = 128

    def __post_init__(self) -> None:
        for name in ("v", "a", "d", "u", "g", "w", "i"):
            value = int(getattr(self, name))
            object.__setattr__(self, name, max(0, min(255, value)))

    def to_dict(self) -> Dict[str, int]:
        return {name: int(getattr(self, name)) for name in ("v", "a", "d", "u", "g", "w", "i")}

    @classmethod
    def from_object(cls, value: Any) -> "AffectVector":
        return cls(**{name: int(getattr(value, name)) for name in ("v", "a", "d", "u", "g", "w", "i")})

    def distance(self, other: "AffectVector", weights: Optional[Mapping[str, float]] = None) -> float:
        axis_weights = {"v": 1.0, "a": 0.7, "d": 0.8, "u": 1.0, "g": 0.8, "w": 0.7, "i": 0.9}
        if weights:
            axis_weights.update(weights)
        total = 0.0
        divisor = 0.0
        for name, weight in axis_weights.items():
            delta = float(getattr(self, name) - getattr(other, name))
            total += weight * delta * delta
            divisor += weight
        return (total / max(divisor, 1e-9)) ** 0.5


@dataclass
class AffectReading:
    vector: AffectVector
    structures: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    backend: str = "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector": self.vector.to_dict(),
            "structures": list(self.structures),
            "roles": list(self.roles),
            "metadata": _json_safe(self.metadata),
            "backend": self.backend,
        }
