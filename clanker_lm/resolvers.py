"""Deterministic semantic command resolvers.

Resolvers bind typed question slots to live or computed observations.  They do
not generate prose.  The observation is converted to an ``AnswerContract`` and
passes through the same compositional grammar and VADUGWI ranking as ordinary
conversation.
"""

from __future__ import annotations

import ast
import hashlib
import math
import operator
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any, Callable, Dict, Mapping, Optional, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .database import LanguageStore
from .model import (
    AnswerContract,
    AnswerStatus,
    EntityKind,
    EventFrame,
    SemanticRef,
    SourceKind,
)


@dataclass
class ResolverOutcome:
    handled: bool = False
    contract: Optional[AnswerContract] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticResolver(Protocol):
    name: str

    def resolve(self, text: str, registry: "ResolverRegistry") -> ResolverOutcome:
        ...


class ResolverRegistry:
    """Route a semantic command to a deterministic provider."""

    def __init__(
        self,
        store: LanguageStore,
        *,
        default_timezone: str = "America/Chicago",
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.store = store
        self.default_timezone = default_timezone
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._resolvers: list[SemanticResolver] = []
        self.register(ClockResolver())
        self.register(DateResolver())
        self.register(CalculatorResolver())

    def register(self, resolver: SemanticResolver) -> None:
        self._resolvers.append(resolver)

    def resolve(self, text: str) -> ResolverOutcome:
        for resolver in self._resolvers:
            outcome = resolver.resolve(text, self)
            if not outcome.handled:
                continue
            if outcome.contract is not None:
                outcome.contract.required_slots.setdefault("resolver_name", resolver.name)
            self._record(text, resolver.name, outcome)
            return outcome
        return ResolverOutcome()

    def now_utc(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _record(self, text: str, resolver_name: str, outcome: ResolverOutcome) -> None:
        contract = outcome.contract
        if contract is None:
            return
        observed_at = str(outcome.metadata.get("observed_at") or self.now_utc().isoformat())
        expires_at = outcome.metadata.get("expires_at")
        proposition = contract.proposition
        value = {
            "status": contract.status.value,
            "proposition": proposition.to_dict() if proposition else None,
            "values": [item.to_dict() for item in contract.values],
        }
        self.store.record_resolver_observation(
            request_hash=hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest(),
            resolver_name=resolver_name,
            predicate=proposition.predicate if proposition else resolver_name,
            value=value,
            source_name=str(outcome.metadata.get("source", resolver_name)),
            certainty=contract.certainty,
            observed_at=observed_at,
            expires_at=str(expires_at) if expires_at else None,
            metadata=outcome.metadata,
        )


_TIME_PATTERNS = (
    re.compile(
        r"^\s*(?:what\s+time\s+is\s+it|what(?:'s|\s+is)\s+the\s+(?:current\s+)?time|current\s+time)"
        r"(?:\s+(?:right\s+)?now)?(?:\s+in\s+(?P<place>[^?]+?))?\s*[?!.]*\s*$",
        re.IGNORECASE,
    ),
)
_DATE_PATTERNS = (
    re.compile(
        r"^\s*(?:what(?:'s|\s+is)\s+(?:today(?:'s)?\s+date|the\s+(?:current\s+)?date)|"
        r"what\s+date\s+is\s+it|today(?:'s)?\s+date|current\s+date)"
        r"(?:\s+in\s+(?P<place>[^?]+?))?\s*[?!.]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:what\s+day\s+is\s+it|what(?:'s|\s+is)\s+today)"
        r"(?:\s+in\s+(?P<place>[^?]+?))?\s*[?!.]*\s*$",
        re.IGNORECASE,
    ),
)


TIMEZONE_ALIASES: Mapping[str, str] = {
    "local": "America/Chicago",
    "here": "America/Chicago",
    "chicago": "America/Chicago",
    "central": "America/Chicago",
    "central time": "America/Chicago",
    "new york": "America/New_York",
    "eastern": "America/New_York",
    "eastern time": "America/New_York",
    "denver": "America/Denver",
    "mountain": "America/Denver",
    "mountain time": "America/Denver",
    "los angeles": "America/Los_Angeles",
    "pacific": "America/Los_Angeles",
    "pacific time": "America/Los_Angeles",
    "utc": "UTC",
    "gmt": "UTC",
    "london": "Europe/London",
    "paris": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "tokyo": "Asia/Tokyo",
    "japan": "Asia/Tokyo",
    "seoul": "Asia/Seoul",
    "beijing": "Asia/Shanghai",
    "shanghai": "Asia/Shanghai",
    "hong kong": "Asia/Hong_Kong",
    "singapore": "Asia/Singapore",
    "delhi": "Asia/Kolkata",
    "india": "Asia/Kolkata",
    "sydney": "Australia/Sydney",
    "melbourne": "Australia/Melbourne",
    "auckland": "Pacific/Auckland",
    "honolulu": "Pacific/Honolulu",
}


def _zone_for(place: Optional[str], default: str) -> tuple[Optional[ZoneInfo], str, str]:
    display = (place or "local").strip().rstrip(".?!")
    normalized = re.sub(r"\s+", " ", display.lower())
    zone_name = TIMEZONE_ALIASES.get(normalized)
    if place is None or normalized in {"local", "here"}:
        zone_name = default
    if zone_name is None:
        # Explicit IANA identifiers are accepted as deterministic commands.
        zone_name = display if "/" in display else ""
    try:
        zone = ZoneInfo(zone_name) if zone_name else None
    except ZoneInfoNotFoundError:
        zone = None
    return zone, display, zone_name


def _unknown_zone_contract(place: str) -> AnswerContract:
    return AnswerContract(
        status=AnswerStatus.UNKNOWN,
        certainty=255,
        source=SourceKind.EXTERNAL,
        reason="timezone name could not be resolved deterministically",
        response_goal="clarify",
        required_slots={"unknown_object": "timezone", "reference": place},
        forbidden_claims=["guess_timezone", "invent_current_time"],
    )


class ClockResolver:
    name = "system_clock"

    def resolve(self, text: str, registry: ResolverRegistry) -> ResolverOutcome:
        match = next((pattern.match(text) for pattern in _TIME_PATTERNS if pattern.match(text)), None)
        if not match:
            return ResolverOutcome()
        place = match.groupdict().get("place")
        zone, display, zone_name = _zone_for(place, registry.default_timezone)
        now = registry.now_utc()
        if zone is None:
            contract = _unknown_zone_contract(display)
            return ResolverOutcome(
                True,
                contract,
                {
                    "command": "CURRENT_TIME",
                    "source": self.name,
                    "observed_at": now.isoformat(),
                    "timezone_input": display,
                },
            )
        local = now.astimezone(zone)
        time_text = local.strftime("%I:%M %p").lstrip("0")
        location_text = display if place else zone_name
        subject_surface = "current time" if not place else f"current time in {location_text}"
        proposition = EventFrame(
            predicate="be",
            arguments={
                "subject": SemanticRef.literal(subject_surface, subject_surface, EntityKind.TIME),
                "value": SemanticRef.literal(time_text.lower(), time_text, EntityKind.TIME),
            },
            tense="present",
            source=SourceKind.EXTERNAL,
            certainty=255,
        )
        contract = AnswerContract(
            status=AnswerStatus.ANSWERED,
            proposition=proposition,
            values=[proposition.arguments["value"]],
            certainty=255,
            source=SourceKind.EXTERNAL,
            reason="live system clock observation",
            response_goal="answer",
            required_slots={"requested_role": "value"},
            forbidden_claims=["reuse_expired_time"],
        )
        return ResolverOutcome(
            True,
            contract,
            {
                "command": "CURRENT_TIME",
                "source": self.name,
                "observed_at": now.isoformat(),
                "expires_at": (now + timedelta(seconds=60)).isoformat(),
                "timezone": zone_name,
                "timezone_input": display,
                "utc_offset": local.strftime("%z"),
            },
        )


class DateResolver:
    name = "system_calendar"

    def resolve(self, text: str, registry: ResolverRegistry) -> ResolverOutcome:
        match = next((pattern.match(text) for pattern in _DATE_PATTERNS if pattern.match(text)), None)
        if not match:
            return ResolverOutcome()
        place = match.groupdict().get("place")
        zone, display, zone_name = _zone_for(place, registry.default_timezone)
        now = registry.now_utc()
        if zone is None:
            return ResolverOutcome(
                True,
                _unknown_zone_contract(display),
                {
                    "command": "CURRENT_DATE",
                    "source": self.name,
                    "observed_at": now.isoformat(),
                    "timezone_input": display,
                },
            )
        local = now.astimezone(zone)
        date_text = f"{local.strftime('%A, %B')} {local.day}, {local.year}"
        location_text = display if place else zone_name
        subject_surface = "current date" if not place else f"current date in {location_text}"
        proposition = EventFrame(
            predicate="be",
            arguments={
                "subject": SemanticRef.literal(subject_surface, subject_surface, EntityKind.TIME),
                "value": SemanticRef.literal(local.date().isoformat(), date_text, EntityKind.TIME),
            },
            tense="present",
            source=SourceKind.EXTERNAL,
            certainty=255,
        )
        midnight = (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return ResolverOutcome(
            True,
            AnswerContract(
                status=AnswerStatus.ANSWERED,
                proposition=proposition,
                values=[proposition.arguments["value"]],
                certainty=255,
                source=SourceKind.EXTERNAL,
                reason="live system calendar observation",
                response_goal="answer",
                required_slots={"requested_role": "value"},
                forbidden_claims=["reuse_expired_date"],
            ),
            {
                "command": "CURRENT_DATE",
                "source": self.name,
                "observed_at": now.isoformat(),
                "expires_at": midnight.astimezone(timezone.utc).isoformat(),
                "timezone": zone_name,
                "timezone_input": display,
            },
        )


_CALCULATE_PREFIX = re.compile(r"^\s*(?:calculate|compute|evaluate)\s+(?P<expression>.+?)\s*[?!.]*\s*$", re.IGNORECASE)
_WHAT_IS_EXPRESSION = re.compile(r"^\s*what(?:'s|\s+is)\s+(?P<expression>.+?)\s*\?\s*$", re.IGNORECASE)
_ALLOWED_EXPRESSION = re.compile(r"^[0-9eE.\s+\-*/%()^]+$")


class CalculatorResolver:
    name = "deterministic_calculator"

    def resolve(self, text: str, registry: ResolverRegistry) -> ResolverOutcome:
        match = _CALCULATE_PREFIX.match(text) or _WHAT_IS_EXPRESSION.match(text)
        if not match:
            return ResolverOutcome()
        expression = match.group("expression").strip().replace("^", "**")
        if not any(char.isdigit() for char in expression) or not _ALLOWED_EXPRESSION.fullmatch(expression):
            return ResolverOutcome()
        observed = registry.now_utc()
        try:
            result = SafeArithmetic.evaluate(expression)
        except (ValueError, ZeroDivisionError, InvalidOperation, OverflowError) as error:
            contract = AnswerContract(
                status=AnswerStatus.UNSUPPORTED,
                certainty=255,
                source=SourceKind.EXTERNAL,
                reason=f"calculator rejected expression: {type(error).__name__}",
                response_goal="clarify",
                required_slots={"unknown_object": "calculation"},
                forbidden_claims=["guess_calculation"],
            )
            return ResolverOutcome(
                True,
                contract,
                {
                    "command": "CALCULATE",
                    "source": self.name,
                    "observed_at": observed.isoformat(),
                    "expression_hash": hashlib.sha256(expression.encode("utf-8")).hexdigest(),
                    "error": type(error).__name__,
                },
            )
        formatted = SafeArithmetic.format(result)
        subject_surface = match.group("expression").strip()
        proposition = EventFrame(
            predicate="be",
            arguments={
                "subject": SemanticRef.literal(subject_surface, subject_surface, EntityKind.ABSTRACT),
                "value": SemanticRef.literal(formatted, formatted, EntityKind.ABSTRACT),
            },
            source=SourceKind.EXTERNAL,
            certainty=255,
        )
        return ResolverOutcome(
            True,
            AnswerContract(
                status=AnswerStatus.ANSWERED,
                proposition=proposition,
                values=[proposition.arguments["value"]],
                certainty=255,
                source=SourceKind.EXTERNAL,
                reason="deterministic arithmetic evaluation",
                response_goal="answer",
                required_slots={"requested_role": "value"},
                forbidden_claims=["execute_code", "guess_calculation"],
            ),
            {
                "command": "CALCULATE",
                "source": self.name,
                "observed_at": observed.isoformat(),
                "expression_hash": hashlib.sha256(expression.encode("utf-8")).hexdigest(),
                "result": formatted,
            },
        )


class SafeArithmetic:
    """Small, bounded arithmetic evaluator with no names, calls, or attributes."""

    MAX_SOURCE_LENGTH = 160
    MAX_DEPTH = 20
    MAX_ABS = Decimal("1e100")
    MAX_POWER = 12

    @classmethod
    def evaluate(cls, expression: str) -> Decimal:
        if len(expression) > cls.MAX_SOURCE_LENGTH:
            raise ValueError("expression is too long")
        node = ast.parse(expression, mode="eval")
        with localcontext() as context:
            context.prec = 50
            value = cls._eval(node.body, depth=0)
        if not value.is_finite() or abs(value) > cls.MAX_ABS:
            raise OverflowError("result is outside the configured range")
        return value

    @classmethod
    def _eval(cls, node: ast.AST, *, depth: int) -> Decimal:
        if depth > cls.MAX_DEPTH:
            raise ValueError("expression is too deep")
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return Decimal(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = cls._eval(node.operand, depth=depth + 1)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = cls._eval(node.left, depth=depth + 1)
            right = cls._eval(node.right, depth=depth + 1)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            elif isinstance(node.op, ast.Div):
                value = left / right
            elif isinstance(node.op, ast.FloorDiv):
                value = left // right
            elif isinstance(node.op, ast.Mod):
                value = left % right
            elif isinstance(node.op, ast.Pow):
                if right != right.to_integral_value() or abs(right) > cls.MAX_POWER:
                    raise ValueError("power must be a small integer")
                value = left ** int(right)
            else:
                raise ValueError("operator is not allowed")
            if not value.is_finite() or abs(value) > cls.MAX_ABS:
                raise OverflowError("intermediate result is outside the configured range")
            return value
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    @staticmethod
    def format(value: Decimal) -> str:
        if value == value.to_integral_value():
            return str(value.quantize(Decimal(1)))
        normalized = value.normalize()
        text = format(normalized, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text
