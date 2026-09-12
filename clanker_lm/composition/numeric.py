"""Strict numeric ingress; premise checking is separate from proof search.

This adapter accepts natural-number addition only. The core remains typed and
symbolic. The standard calculator is used to validate a NUMERIC LESSON, never
to compute a target requested through the composition path.
"""
from __future__ import annotations
import ast
from .terms import Operator, Term, atom, apply, digest, validate
from ..resolvers import SafeArithmetic

NATURAL_ADD = Operator('natural.add', 'natural', associative=True, version='exact-addition-v1')


def parse_addition(text: str) -> Term:
    if not isinstance(text, str) or not 1 <= len(text) <= 160:
        raise ValueError('bounded exact addition expression required')
    try:
        tree = ast.parse(text.strip(), mode='eval')
    except (SyntaxError, RecursionError) as exc:
        raise ValueError('invalid expression') from exc
    def build(node, depth=0):
        if depth > 16:
            raise ValueError('expression depth exceeded')
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return atom('natural', str(node.value))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return apply(NATURAL_ADD, build(node.left, depth+1), build(node.right, depth+1))
        raise ValueError('only exact natural literals and addition are licensed')
    term = build(tree.body)
    validate(term, {NATURAL_ADD.key: NATURAL_ADD})
    return term


def numeric_text(term: Term) -> str:
    validate(term, {NATURAL_ADD.key: NATURAL_ADD})
    if term.sort != 'natural':
        raise ValueError('not an exact natural expression')
    if term.atomic:
        return term.symbol
    left, right = term.arguments
    return '(' + numeric_text(left) + ' + ' + numeric_text(right) + ')'


def validate_numeric_lesson(lhs: Term, rhs: Term) -> None:
    if lhs.atomic or not rhs.atomic or rhs.sort != 'natural':
        raise ValueError('numeric equation must reduce to a natural value')
    if SafeArithmetic.evaluate(numeric_text(lhs)) != SafeArithmetic.evaluate(numeric_text(rhs)):
        raise ValueError('numeric lesson disagrees with exact arithmetic')
