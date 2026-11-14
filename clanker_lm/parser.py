"""Stable parser API and static composition. No grammar or persistent state lives here.

Recognition components are stateless; assembly uses one typed context per call.
Existing SemanticParser subclasses continue to override the same methods.
"""

from .parsing.pipeline import InputPipeline
from .parsing.profiles import PredicateProfiles
from .parsing.transforms import TransformsRules
from .parsing.embedded import EmbeddedRules
from .parsing.infinitives import InfinitivesRules
from .parsing.gerunds import GerundsRules
from .parsing.content import ContentRules
from .parsing.relatives import RelativesRules
from .parsing.appositives import AppositivesRules
from .parsing.subordinates import SubordinatesRules
from .parsing.question_routing import QuestionRoutingRules
from .parsing.wh_questions import WhQuestionsRules
from .parsing.clauses import ClausesRules
from .parsing.predicate_roles import PredicateRolesRules
from .parsing.nouns import NounsRules
from .parsing.types import (
    NPResult,
    ClauseResult,
    ContentSplit,
    EmbeddedInterrogativePredicateProfile,
    EmbeddedInterrogativeSplit,
    InfinitivalPredicateProfile,
    InfinitivalSplit,
    GerundPredicateProfile,
    GerundSplit,
    RelativeSplit,
    AppositiveSplit,
    SubordinateSplit,
)

class SemanticParser(
    InputPipeline, PredicateProfiles,
    TransformsRules,
    EmbeddedRules,
    InfinitivesRules,
    GerundsRules,
    ContentRules,
    RelativesRules,
    AppositivesRules,
    SubordinatesRules,
    QuestionRoutingRules,
    WhQuestionsRules,
    ClausesRules,
    PredicateRolesRules,
    NounsRules,
):
    """Conservative deterministic parser composed from focused components."""

__all__ = (
    'SemanticParser',
    'NPResult',
    'ClauseResult',
    'ContentSplit',
    'EmbeddedInterrogativePredicateProfile',
    'EmbeddedInterrogativeSplit',
    'InfinitivalPredicateProfile',
    'InfinitivalSplit',
    'GerundPredicateProfile',
    'GerundSplit',
    'RelativeSplit',
    'AppositiveSplit',
    'SubordinateSplit',
)
