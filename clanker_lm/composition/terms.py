"""Typed ground expressions. Domain laws are explicit host-reviewed inputs.

No arithmetic evaluator, natural-language reply, or graph-similarity inference
lives here. A sort may denote exact numbers or another closed symbolic domain;
using a unit/scope-specific sort does not itself license a physical aggregation.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Mapping

SCHEMA = 'clanker-ground-composition-v1'
ID = re.compile(r'[A-Za-z0-9_.:/-]{1,128}\Z')
MAX_TERM_NODES = 63
MAX_TERM_DEPTH = 16


def identifier(value: str) -> str:
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError('bounded atomic identifier required')
    return value


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class Operator:
    key: str
    sort: str
    associative: bool = False
    version: str = '1'

    def __post_init__(self):
        for value in (self.key, self.sort, self.version):
            identifier(value)
        if type(self.associative) is not bool:
            raise ValueError('associativity is an explicit reviewed law')


@dataclass(frozen=True)
class Term:
    sort: str
    symbol: str
    arguments: tuple['Term', ...] = ()

    def __post_init__(self):
        identifier(self.sort); identifier(self.symbol)
        if (type(self.arguments) is not tuple or len(self.arguments) not in (0, 2)
                or any(not isinstance(a, Term) or a.sort != self.sort for a in self.arguments)):
            raise ValueError('closed binary operation requires matching operand sorts')
        if self.sort == 'natural' and not self.arguments:
            if not re.fullmatch(r'0|[1-9][0-9]{0,8}', self.symbol):
                raise ValueError('natural atom requires a canonical bounded numeral')

    @property
    def atomic(self) -> bool:
        return not self.arguments

    def to_dict(self) -> dict:
        return {'sort': self.sort, 'symbol': self.symbol,
                'arguments': [a.to_dict() for a in self.arguments]}

    @classmethod
    def from_dict(cls, data: dict) -> 'Term':
        count = 0
        def load(row, depth):
            nonlocal count
            count += 1
            if count > MAX_TERM_NODES or depth > MAX_TERM_DEPTH:
                raise ValueError('expression budget exceeded')
            if (not isinstance(row, dict) or set(row) != {'sort', 'symbol', 'arguments'}
                    or type(row['arguments']) is not list or len(row['arguments']) not in (0, 2)):
                raise ValueError('invalid expression record')
            return cls(row['sort'], row['symbol'], tuple(load(x, depth+1) for x in row['arguments']))
        return load(data, 0)


def atom(sort: str, symbol: str) -> Term:
    return Term(sort, symbol)


def apply(op: Operator, left: Term, right: Term) -> Term:
    if not isinstance(op, Operator) or left.sort != op.sort or right.sort != op.sort:
        raise ValueError('operator and operands must share their declared sort')
    return Term(op.sort, op.key, (left, right))


def validate(term: Term, operators: Mapping[str, Operator]) -> None:
    count = 0
    def visit(node, depth):
        nonlocal count
        count += 1
        if not isinstance(node, Term) or count > MAX_TERM_NODES or depth > MAX_TERM_DEPTH:
            raise ValueError('expression budget exceeded')
        if node.arguments:
            op = operators.get(node.symbol)
            if not op or op.sort != node.sort:
                raise ValueError('unknown or inapplicable operator')
            for child in node.arguments:
                visit(child, depth+1)
    visit(term, 0)


def walk(term: Term, path: tuple[int, ...] = ()):
    yield path, term
    for i, child in enumerate(term.arguments):
        yield from walk(child, path + (i,))


def replace_at(term: Term, path: tuple[int, ...], value: Term) -> Term:
    if not path:
        if term.sort != value.sort:
            raise ValueError('substitution changes the sort')
        return value
    children = list(term.arguments)
    if not children or path[0] not in (0, 1):
        raise ValueError('invalid expression address')
    children[path[0]] = replace_at(children[path[0]], path[1:], value)
    return Term(term.sort, term.symbol, tuple(children))


def law_key(term: Term, operators: Mapping[str, Operator]) -> tuple:
    """Canonical key modulo ONLY declared associativity, not commutativity."""
    if term.atomic:
        return term.sort, 'atom', term.symbol
    op = operators[term.symbol]
    if not op.associative:
        return term.sort, op.key, tuple(law_key(a, operators) for a in term.arguments)
    leaves = []
    def flatten(node):
        if node.arguments and node.symbol == term.symbol:
            for a in node.arguments:
                flatten(a)
        else:
            leaves.append(law_key(node, operators))
    flatten(term)
    return term.sort, op.key, tuple(leaves)
