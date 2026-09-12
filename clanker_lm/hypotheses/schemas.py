"""Authored PROPOSAL schemas, not sound inference laws or sentence templates."""
from .records import Pattern as P, ProposalRule

MATERIAL = ProposalRule('material_property_candidate_v1', (
    P('made_from', ('?product', '?material'), '?frame'),
    P('has_property', ('?product', '?property'), '?frame'),
), P('has_property', ('?material', '?property'), '?frame'),
    distinct=(('?product', '?material'),),
    obligations=('check_transformation_effect', 'validate_target_claim', 'seek_counterevidence'))

CATEGORY = ProposalRule('category_property_candidate_v1', (
    P('instance_of', ('?a', '?category'), '?frame'),
    P('has_property', ('?a', '?property'), '?frame'),
    P('instance_of', ('?b', '?category'), '?frame'),
), P('has_property', ('?b', '?property'), '?frame'),
    distinct=(('?a', '?b'),),
    obligations=('check_relevant_shared_features', 'validate_target_claim', 'seek_counterevidence'))

SITUATION = ProposalRule('resource_blockage_candidate_v1', (
    P('attempts', ('?a', '?task_a'), '?base'),
    P('requires', ('?task_a', '?resource_a'), '?base'),
    P('lacks', ('?a', '?resource_a'), '?base'),
    P('blocked', ('?a', '?task_a'), '?base'),
    P('attempts', ('?b', '?task_b'), '?target'),
    P('requires', ('?task_b', '?resource_b'), '?target'),
    P('lacks', ('?b', '?resource_b'), '?target'),
), P('blocked', ('?b', '?task_b'), '?target'),
    distinct=(('?base', '?target'), ('?a', '?b'), ('?task_a', '?task_b')),
    obligations=('check_other_routes', 'validate_target_claim', 'seek_counterevidence'))

RULES = (MATERIAL, CATEGORY, SITUATION)
