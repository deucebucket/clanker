"""Typed candidate claims. Labels, epistemic status and identity are distinct."""
from dataclasses import asdict, dataclass
from ..activation import identifier, fingerprint

SCHEMA = 'clanker-hypotheses-v1'
ALLOWED_SOURCES = frozenset({'user_report', 'document_report', 'test_observation'})


@dataclass(frozen=True)
class Claim:
    predicate: str
    arguments: tuple[str, ...]
    frame: str
    conditions: tuple[tuple[str, str], ...] = ()
    positive: bool = True

    def __post_init__(self):
        identifier(self.predicate); identifier(self.frame)
        if type(self.arguments) is not tuple or not 1 <= len(self.arguments) <= 4:
            raise ValueError('one to four bound arguments required')
        for arg in self.arguments: identifier(arg)
        if type(self.positive) is not bool:
            raise ValueError('boolean polarity required')
        if (type(self.conditions) is not tuple or len(self.conditions) > 8
                or any(type(c) is not tuple or len(c) != 2 for c in self.conditions)):
            raise ValueError('bounded typed conditions required')
        for key, value in self.conditions: identifier(key); identifier(value)
        if len({k for k, _ in self.conditions}) != len(self.conditions):
            raise ValueError('duplicate condition keys')
        object.__setattr__(self, 'conditions', tuple(sorted(self.conditions)))

    def to_dict(self):
        return {"predicate": self.predicate, "arguments": list(self.arguments), "frame": self.frame,
                "conditions": [list(x) for x in self.conditions], "positive": self.positive}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {'predicate', 'arguments', 'frame', 'conditions', 'positive'}:
            raise ValueError('invalid claim schema')
        return cls(data['predicate'], tuple(data['arguments']), data['frame'],
                   tuple(tuple(v) for v in data['conditions']), data['positive'])

    def opposite(self):
        return Claim(self.predicate, self.arguments, self.frame, self.conditions, not self.positive)


@dataclass(frozen=True)
class Pattern:
    predicate: str
    arguments: tuple[str, ...]
    frame: str
    positive: bool = True

    def __post_init__(self):
        identifier(self.predicate)
        if type(self.arguments) is not tuple or not 1 <= len(self.arguments) <= 4:
            raise ValueError('invalid pattern arguments')
        for value in self.arguments + (self.frame,):
            identifier(value)
            if value.startswith('?') and (len(value) < 2 or not value[1:].isidentifier()):
                raise ValueError('invalid pattern variable')
        if type(self.positive) is not bool: raise ValueError('invalid pattern polarity')


@dataclass(frozen=True)
class ProposalRule:
    key: str
    premises: tuple[Pattern, ...]
    conclusion: Pattern
    distinct: tuple[tuple[str, str], ...] = ()
    obligations: tuple[str, ...] = ('validate_target_claim', 'seek_counterevidence')

    def __post_init__(self):
        identifier(self.key)
        if (type(self.premises) is not tuple or not 1 <= len(self.premises) <= 8
                or any(not isinstance(p, Pattern) for p in self.premises)
                or not isinstance(self.conclusion, Pattern)):
            raise ValueError('bounded typed rule required')
        variables = {x for p in self.premises for x in p.arguments + (p.frame,) if x.startswith('?')}
        if any(x.startswith('?') and x not in variables for x in self.conclusion.arguments + (self.conclusion.frame,)):
            raise ValueError('unbound conclusion variable')
        if (type(self.distinct) is not tuple or any(type(p) is not tuple or len(p) != 2
                or any(x not in variables for x in p) for p in self.distinct)):
            raise ValueError('unbound distinct variables')
        if type(self.obligations) is not tuple or not 1 <= len(self.obligations) <= 8:
            raise ValueError('validation obligations required')
        for key in self.obligations: identifier(key)

    @property
    def identity(self):
        return fingerprint(asdict(self))


def unify(pattern, claim, binding):
    if (pattern.predicate != claim.predicate or pattern.positive != claim.positive
            or len(pattern.arguments) != len(claim.arguments)):
        return None
    out = dict(binding)
    for term, value in zip(pattern.arguments + (pattern.frame,), claim.arguments + (claim.frame,)):
        if term.startswith('?'):
            if term in out and out[term] != value: return None
            out[term] = value
        elif term != value:
            return None
    return out


def instantiate(pattern, bindings, conditions):
    def value(x): return bindings[x] if x.startswith('?') else x
    return Claim(pattern.predicate, tuple(value(x) for x in pattern.arguments),
                 value(pattern.frame), conditions, pattern.positive)


@dataclass(frozen=True)
class Limits:
    max_rows: int = 512
    max_checks: int = 4096
    max_candidates: int = 64

    def __post_init__(self):
        for key, high in (('max_rows', 512), ('max_checks', 65536), ('max_candidates', 128)):
            v = getattr(self, key)
            if type(v) is not int or not 1 <= v <= high: raise ValueError('invalid hypothesis work budget')
