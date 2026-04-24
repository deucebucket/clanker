"""Clanker V5.5 — Structural Pattern Recognition Engine."""
from .shared import VADUG
from .pendulum import compute_vadug
from .solver import forward, solve_for_b_range, optimal_b_temperature, state_transition
from .battleship import triangulate, fire_probe, PROBES
from .word_classifier import classify_sentence, WordRole, ROLES
from .structures import StructureDetector, StructureMatch
from .zones import ZoneClassifier, ZONES
