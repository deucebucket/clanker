"""Clanker V9 — Equation Decomposition Engine."""
from .shared import VADUG, PersonalityVector
from .pipeline import compute_vadug
from .solver import forward, solve_for_b_range, optimal_b_temperature, state_transition
from .battleship import triangulate, fire_probe, PROBES
from .word_classifier import classify_sentence, WordRole, ROLES
from .structures import StructureDetector, StructureMatch
from .zones import ZoneClassifier, ZONES
from .crisis import CrisisTracker
from .anomaly import AnomalyDetector
from .force_flow import resolve_force_flow, ForceFlow
