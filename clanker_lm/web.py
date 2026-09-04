"""Small, hardened ASGI surface for the deterministic Clanker-LM runtime.

The web extra is intentionally isolated in this module so importing
``clanker_lm`` never requires Starlette or Uvicorn.  Every request handler is
async and runtime calls stay on the server's single event-loop thread.
"""

from __future__ import annotations

import json
import mimetypes
import re
import secrets
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date
from importlib import resources
from typing import Any, Callable, Deque, Mapping, Optional
from urllib.parse import urlsplit

from starlette.applications import Starlette
from starlette.datastructures import MutableHeaders
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import __version__
from .runtime import ClankerLM


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
SESSION_COOKIE = "clanker_lm_session"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_ASSET_NAMES = ("index.html", "app.css", "app.js", "releases.json")
_RELEASE_FEED_MAX_BYTES = 64 * 1024
_RELEASE_FEED_KEYS = frozenset(
    {"schema_version", "latest_shipped_release", "releases"}
)
_RELEASE_KEYS = frozenset(
    {
        "release_id",
        "package_version",
        "milestone_commit",
        "date",
        "title",
        "capabilities",
        "evidence",
        "limitations",
        "deployment",
    }
)
_SHIPPED_RELEASE_KEYS = frozenset(
    {"release_id", "package_version", "milestone_commit"}
)
_EVIDENCE_KEYS = frozenset({"label", "url"})
_DEPLOYMENT_KEYS = frozenset({"state", "label", "detail", "url"})
_DEPLOYMENT_LABELS = {
    "live": "Live · private Tailnet",
    "pending": "Pending · live verification",
    "retired": "Retired · release history",
    "rolled_back": "Rolled back · release history",
}
_PRIVATE_FEED_KEYS = frozenset(
    {
        "attachment",
        "body",
        "message",
        "prompt",
        "raw",
        "receipt_token",
        "response",
        "session",
        "transcript",
    }
)
_PRIVATE_FEED_MARKERS = (
    "/acl ",
    "chatall:",
    "from_handle",
    "has_attachment",
    "message_id",
    "receipt_token",
)
_LIVE_WORKBENCH_URL = "https://bazzite.tail85f65f.ts.net:8444/"
_STAGING_RECEIPT_URL = (
    "https://github.com/deucebucket/clanker/issues/112#issuecomment-5539229707"
)
_PR113_STAGING_EVIDENCE_URLS = frozenset(
    {
        "https://github.com/deucebucket/clanker/pull/114",
        "https://github.com/deucebucket/clanker/commit/2d736961d7db0510711a7ac54eb39a458446f5ee",
        "https://github.com/deucebucket/clanker/actions/runs/33863653170",
        _STAGING_RECEIPT_URL,
    }
)
_LOCAL_BUILD_COMMIT = "0" * 40
_CONTENT_SECURITY_POLICY = (
    "default-src 'none'; base-uri 'none'; form-action 'self'; "
    "frame-ancestors 'none'; img-src 'self' data:; "
    "script-src 'self'; style-src 'self'; connect-src 'self'"
)


def _require_loopback_host(host: Any) -> None:
    if host not in _LOOPBACK_HOSTS:
        raise ValueError(
            "host must be 127.0.0.1, ::1, or localhost; "
            "use a trusted local reverse proxy for remote access"
        )


@dataclass(frozen=True)
class WebConfig:
    """Security and resource limits for one web process."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    public_origin: Optional[str] = None
    deployed: bool = False
    build_commit: Optional[str] = None
    allowed_users: tuple[str, ...] = ()
    cookie_name: str = SESSION_COOKIE
    session_idle_seconds: float = 30 * 60
    max_sessions: int = 128
    rate_limit: int = 30
    rate_window_seconds: float = 60.0
    max_turns: int = 200
    max_message_bytes: int = 4 * 1024
    max_body_bytes: int = 16 * 1024
    max_chat_response_bytes: int = 64 * 1024
    max_export_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host:
            raise ValueError("host must be a non-empty string")
        _require_loopback_host(self.host)
        self._positive_integer("port", self.port, maximum=65535)
        self._positive_integer("max_sessions", self.max_sessions)
        self._positive_integer("rate_limit", self.rate_limit)
        self._positive_integer("max_turns", self.max_turns)
        self._positive_integer("max_message_bytes", self.max_message_bytes)
        self._positive_integer("max_body_bytes", self.max_body_bytes)
        self._positive_integer("max_chat_response_bytes", self.max_chat_response_bytes)
        self._positive_integer("max_export_bytes", self.max_export_bytes)
        if self.max_body_bytes < self.max_message_bytes:
            raise ValueError("max_body_bytes must be at least max_message_bytes")
        for name, value in (
            ("session_idle_seconds", self.session_idle_seconds),
            ("rate_window_seconds", self.rate_window_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.cookie_name, str) or not self.cookie_name.isascii():
            raise ValueError("cookie_name must be ASCII")
        if not self.cookie_name or any(
            character
            not in "!#$%&'*+-.^_`|~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for character in self.cookie_name
        ):
            raise ValueError("cookie_name is not a valid cookie token")
        if self.public_origin is not None:
            if not isinstance(self.public_origin, str) or not self.public_origin:
                raise ValueError("public_origin must be a non-empty string")
            if any(
                character in self.public_origin for character in ("\n", "\r", "\t", " ")
            ):
                raise ValueError(
                    "public_origin must be an origin without whitespace or a path"
                )
            try:
                parsed_origin = urlsplit(self.public_origin)
                # Access validates malformed and out-of-range explicit ports.
                parsed_origin.port
            except ValueError as exc:
                raise ValueError("public_origin is not a valid origin") from exc
            if (
                parsed_origin.scheme not in {"http", "https"}
                or not parsed_origin.netloc
                or parsed_origin.hostname is None
                or parsed_origin.username is not None
                or parsed_origin.password is not None
            ):
                raise ValueError("public_origin must be an absolute HTTP(S) origin")
            if parsed_origin.path or parsed_origin.query or parsed_origin.fragment:
                raise ValueError(
                    "public_origin must not include a path, query, or fragment"
                )
        if not isinstance(self.allowed_users, tuple) or any(
            not isinstance(user, str) or not user for user in self.allowed_users
        ):
            raise ValueError("allowed_users must be a tuple of non-empty login strings")
        if len(set(self.allowed_users)) != len(self.allowed_users):
            raise ValueError("allowed_users must not contain duplicates")
        if self.build_commit is not None and (
            not isinstance(self.build_commit, str)
            or re.fullmatch(r"[0-9a-f]{40}", self.build_commit) is None
        ):
            raise ValueError("build_commit must be a full lowercase Git commit")
        if self.deployed:
            if self.public_origin is None:
                raise ValueError("deployed mode requires an explicit public_origin")
            if not self.allowed_users:
                raise ValueError("deployed mode requires at least one allowed user")
            if self.build_commit in {None, _LOCAL_BUILD_COMMIT}:
                raise ValueError("deployed mode requires an exact build_commit")
            assert parsed_origin is not None
            hostname = parsed_origin.hostname
            assert hostname is not None
            rendered_host = f"[{hostname}]" if ":" in hostname else hostname
            port = parsed_origin.port
            rendered_authority = (
                rendered_host
                if port is None or port == 443
                else f"{rendered_host}:{port}"
            )
            canonical_origin = f"https://{rendered_authority}"
            if hostname.endswith(".") or self.public_origin != canonical_origin:
                raise ValueError(
                    "deployed public_origin must be canonical HTTPS with a lowercase "
                    "host and no explicit default port"
                )

    @staticmethod
    def _positive_integer(
        name: str, value: Any, *, maximum: Optional[int] = None
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        if maximum is not None and value > maximum:
            raise ValueError(f"{name} must not exceed {maximum}")

    @property
    def allowed_origin(self) -> str:
        """The one exact origin accepted for state-changing requests."""

        host = (
            f"[{self.host}]"
            if ":" in self.host and not self.host.startswith("[")
            else self.host
        )
        return self.public_origin or f"http://{host}:{self.port}"

    @property
    def resolved_build_commit(self) -> str:
        """Return a visible commit identity, using zeros only outside deployment."""

        return self.build_commit or _LOCAL_BUILD_COMMIT


@dataclass
class WebSession:
    """One browser's runtime and bounded request accounting."""

    session_id: str
    runtime: ClankerLM
    created_at: float
    last_seen: float
    turns: int = 0
    requests: Deque[float] = field(default_factory=deque)


class SessionCapacityError(RuntimeError):
    """Raised when every bounded session slot belongs to an active browser."""


class SessionRegistry:
    """Event-loop-confined bounded registry with deterministic test seams."""

    def __init__(
        self,
        config: WebConfig,
        *,
        runtime_factory: Callable[[], ClankerLM] = ClankerLM,
        clock: Callable[[], float] = time.monotonic,
        token_factory: Callable[[], str] = lambda: secrets.token_urlsafe(32),
    ) -> None:
        self.config = config
        self._runtime_factory = runtime_factory
        self._clock = clock
        self._token_factory = token_factory
        self._sessions: "OrderedDict[str, WebSession]" = OrderedDict()

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    @property
    def session_ids(self) -> tuple[str, ...]:
        """Opaque identifiers, exposed only as a deterministic test seam."""

        return tuple(self._sessions)

    def get(self, session_id: Optional[str]) -> Optional[WebSession]:
        now = self._clock()
        self.expire_idle(now=now)
        if not session_id or len(session_id) > 128:
            return None
        session = self._sessions.get(session_id)
        if session is None:
            return None
        session.last_seen = now
        self._sessions.move_to_end(session_id)
        return session

    def get_or_create(self, session_id: Optional[str]) -> tuple[WebSession, bool]:
        session = self.get(session_id)
        if session is not None:
            return session, False

        now = self._clock()
        if len(self._sessions) >= self.config.max_sessions:
            raise SessionCapacityError("active session capacity reached")

        new_id = self._new_session_id()
        runtime = self._runtime_factory()
        session = WebSession(
            session_id=new_id,
            runtime=runtime,
            created_at=now,
            last_seen=now,
        )
        self._sessions[new_id] = session
        return session, True

    def admit_request(self, session: WebSession) -> bool:
        """Consume one request from this session's fixed-length sliding window."""

        now = self._clock()
        cutoff = now - self.config.rate_window_seconds
        while session.requests and session.requests[0] <= cutoff:
            session.requests.popleft()
        if len(session.requests) >= self.config.rate_limit:
            return False
        session.requests.append(now)
        return True

    def drop(self, session_id: Optional[str]) -> bool:
        if not session_id:
            return False
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        self._close_runtime(session.runtime)
        return True

    def expire_idle(self, *, now: Optional[float] = None) -> int:
        current = self._clock() if now is None else now
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if current - session.last_seen >= self.config.session_idle_seconds
        ]
        for session_id in expired:
            self.drop(session_id)
        return len(expired)

    def close_all(self) -> None:
        sessions = tuple(self._sessions.values())
        self._sessions.clear()
        for session in sessions:
            self._close_runtime(session.runtime)

    def _new_session_id(self) -> str:
        for _ in range(8):
            candidate = self._token_factory()
            if (
                isinstance(candidate, str)
                and 32 <= len(candidate) <= 128
                and candidate.isascii()
                and all(
                    character.isalnum() or character in "-_" for character in candidate
                )
                and candidate not in self._sessions
            ):
                return candidate
        raise RuntimeError("could not allocate a session")

    @staticmethod
    def _close_runtime(runtime: ClankerLM) -> None:
        try:
            runtime.close()
        except Exception:
            # Cleanup is best-effort and never turns a response into a leak of
            # backend exception details.
            pass


class _WebError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class _SecurityHeadersMiddleware:
    """Add fixed headers without BaseHTTPMiddleware or a worker thread."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_hardened(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["Content-Security-Policy"] = _CONTENT_SECURITY_POLICY
                headers["Cross-Origin-Opener-Policy"] = "same-origin"
                headers["Cross-Origin-Resource-Policy"] = "same-origin"
                headers["Permissions-Policy"] = (
                    "camera=(), microphone=(), geolocation=(), payment=(), usb=()"
                )
                headers["Referrer-Policy"] = "no-referrer"
                headers["X-Content-Type-Options"] = "nosniff"
                headers["X-Frame-Options"] = "DENY"
                headers["Cache-Control"] = "no-store"
            await send(message)

        await self.app(scope, receive, send_hardened)


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message}},
        status_code=status_code,
    )


def _limited_json_response(
    value: Any,
    *,
    max_bytes: int,
    overflow_message: str,
) -> Response:
    """Serialize JSON incrementally and fail before emitting an oversized body."""

    chunks: list[bytes] = []
    size = 0
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    try:
        for chunk in encoder.iterencode(value):
            encoded = chunk.encode("utf-8")
            size += len(encoded)
            if size > max_bytes:
                raise _WebError(507, "response_too_large", overflow_message)
            chunks.append(encoded)
    except (TypeError, ValueError) as exc:
        raise _WebError(
            500,
            "serialization_error",
            "The response could not be serialized.",
        ) from exc
    return Response(b"".join(chunks), media_type="application/json")


def _load_assets() -> Mapping[str, bytes]:
    root = resources.files("clanker_lm.web_assets")
    return {name: root.joinpath(name).read_bytes() for name in _ASSET_NAMES}


def _release_text(value: Any, *, name: str, maximum: int = 500) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > maximum
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise ValueError(f"release feed {name} must be bounded single-line text")
    return value


def _release_text_list(value: Any, *, name: str) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= 12:
        raise ValueError(f"release feed {name} must contain 1 to 12 items")
    return [
        _release_text(item, name=f"{name} item")
        for item in value
    ]


def _release_evidence_url(value: Any) -> str:
    url = _release_text(value, name="evidence URL", maximum=300)
    if url == _STAGING_RECEIPT_URL:
        return url
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parsed.query
        or parsed.fragment
        or not re.fullmatch(
            r"/deucebucket/clanker/(?:pull/\d+|commit/[0-9a-f]{40}|actions/runs/\d+)",
            parsed.path,
        )
    ):
        raise ValueError("release evidence URL is outside the repository allowlist")
    return url


def _reject_private_feed_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str) or key.lower() in _PRIVATE_FEED_KEYS:
                raise ValueError("release feed contains a private-content field")
            _reject_private_feed_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_private_feed_keys(child)
    elif isinstance(value, str):
        lowered = value.lower()
        if any(marker in lowered for marker in _PRIVATE_FEED_MARKERS):
            raise ValueError("release feed contains private conversation metadata")


def _validate_release_feed(value: Any) -> Mapping[str, Any]:
    """Validate the packaged release ledger before serving it."""

    if not isinstance(value, dict) or set(value) != _RELEASE_FEED_KEYS:
        raise ValueError("release feed has an unsupported top-level shape")
    _reject_private_feed_keys(value)
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("release feed has an unsupported schema version")

    shipped = value["latest_shipped_release"]
    if not isinstance(shipped, dict) or set(shipped) != _SHIPPED_RELEASE_KEYS:
        raise ValueError("release feed shipped milestone identity is malformed")
    shipped_id = _release_text(
        shipped["release_id"], name="shipped release ID", maximum=80
    )
    shipped_version = _release_text(
        shipped["package_version"], name="shipped package version", maximum=40
    )
    milestone_commit = _release_text(
        shipped["milestone_commit"], name="shipped milestone commit", maximum=40
    )
    if shipped_version != __version__:
        raise ValueError("release feed version does not match the running package")
    if not re.fullmatch(r"[0-9a-f]{40}", milestone_commit):
        raise ValueError("release feed milestone commit must be a full Git commit")

    releases = value["releases"]
    if not isinstance(releases, list) or not 1 <= len(releases) <= 100:
        raise ValueError("release feed must contain 1 to 100 releases")

    release_ids: set[str] = set()
    lifecycle_ranks: list[int] = []
    pending_dates: list[date] = []
    history_dates: list[date] = []
    live_count = 0
    for index, release in enumerate(releases):
        if not isinstance(release, dict) or set(release) != _RELEASE_KEYS:
            raise ValueError(f"release feed item {index} is malformed")
        release_id = _release_text(
            release["release_id"], name="release ID", maximum=80
        )
        if release_id in release_ids:
            raise ValueError("release feed release IDs must be unique")
        release_ids.add(release_id)
        version = _release_text(
            release["package_version"], name="package version", maximum=40
        )
        commit = _release_text(
            release["milestone_commit"], name="milestone commit", maximum=40
        )
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("release feed commits must be full Git commits")
        rendered_date = _release_text(release["date"], name="date", maximum=10)
        try:
            release_date = date.fromisoformat(rendered_date)
        except ValueError as exc:
            raise ValueError("release feed dates must be ISO calendar dates") from exc
        _release_text(release["title"], name="title", maximum=120)
        _release_text_list(release["capabilities"], name="capabilities")
        _release_text_list(release["limitations"], name="limitations")

        evidence = release["evidence"]
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 12:
            raise ValueError("release evidence must contain 1 to 12 links")
        evidence_urls: set[str] = set()
        for link in evidence:
            if not isinstance(link, dict) or set(link) != _EVIDENCE_KEYS:
                raise ValueError("release evidence link is malformed")
            _release_text(link["label"], name="evidence label", maximum=100)
            evidence_urls.add(_release_evidence_url(link["url"]))

        release_match = re.fullmatch(r"pr-(\d+)", release_id)
        if (
            release_match is None
            or release_match.group(1) == "0"
            or release_match.group(1).startswith("0")
        ):
            raise ValueError("release ID must identify one pull request as pr-N")
        pull_number = release_match.group(1)
        required_evidence = {
            f"https://github.com/deucebucket/clanker/pull/{pull_number}",
            f"https://github.com/deucebucket/clanker/commit/{commit}",
        }
        if not required_evidence <= evidence_urls:
            raise ValueError(
                "release ID and milestone commit require matching evidence links"
            )

        deployment = release["deployment"]
        if not isinstance(deployment, dict) or set(deployment) != _DEPLOYMENT_KEYS:
            raise ValueError("release deployment record is malformed")
        deployment_state = _release_text(
            deployment["state"], name="deployment state", maximum=20
        )
        if deployment_state not in _DEPLOYMENT_LABELS:
            raise ValueError("release deployment state is unsupported")
        deployment_label = _release_text(
            deployment["label"], name="deployment label", maximum=100
        )
        if deployment_label != _DEPLOYMENT_LABELS[deployment_state]:
            raise ValueError("release deployment label does not match its state")
        _release_text(deployment["detail"], name="deployment detail")
        deployment_url = _release_text(
            deployment["url"], name="deployment URL", maximum=200
        )
        if deployment_url != _LIVE_WORKBENCH_URL:
            raise ValueError("release deployment URL is not the pinned workbench")
        if (
            release_id == "pr-113"
            and not _PR113_STAGING_EVIDENCE_URLS <= evidence_urls
        ):
            raise ValueError("pr-113 release requires exact staging evidence")

        if deployment_state == "live":
            live_count += 1
            lifecycle_ranks.append(0)
        elif deployment_state == "pending":
            lifecycle_ranks.append(1)
            pending_dates.append(release_date)
        else:
            lifecycle_ranks.append(2)
            history_dates.append(release_date)

        if index == 0 and (
            release_id != shipped_id
            or version != shipped_version
            or commit != milestone_commit
            or deployment_state != "live"
        ):
            raise ValueError("current live release does not match shipped identity")

    if live_count != 1:
        raise ValueError("release feed must contain exactly one live release")
    if lifecycle_ranks != sorted(lifecycle_ranks):
        raise ValueError("release feed lifecycle groups are out of order")
    if pending_dates != sorted(pending_dates, reverse=True):
        raise ValueError("pending releases must be ordered newest first")
    if history_dates != sorted(history_dates, reverse=True):
        raise ValueError("release history must be ordered newest first")
    return value


def _load_release_feed(data: bytes) -> Mapping[str, Any]:
    if len(data) > _RELEASE_FEED_MAX_BYTES:
        raise ValueError("release feed exceeds its packaged byte limit")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("release feed is not valid UTF-8 JSON") from exc
    return _validate_release_feed(value)


async def _read_json(request: Request, *, max_bytes: int) -> Mapping[str, Any]:
    content_type = (
        request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    )
    if content_type != "application/json":
        raise _WebError(400, "invalid_request", "Expected an application/json request.")

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            parsed_length = int(content_length)
            if parsed_length < 0:
                raise ValueError
            if parsed_length > max_bytes:
                raise _WebError(
                    413, "request_too_large", "The request body is too large."
                )
        except ValueError as exc:
            raise _WebError(
                400, "invalid_request", "The request is malformed."
            ) from exc

    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > max_bytes:
            raise _WebError(413, "request_too_large", "The request body is too large.")
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _WebError(
            400, "invalid_request", "The request body is not valid JSON."
        ) from exc
    if not isinstance(value, dict):
        raise _WebError(400, "invalid_request", "The JSON body must be an object.")
    return value


async def _discard_bounded_body(request: Request, *, max_bytes: int) -> None:
    """Consume an unused request body without allowing an unbounded upload."""

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            parsed_length = int(content_length)
            if parsed_length < 0:
                raise ValueError
            if parsed_length > max_bytes:
                raise _WebError(
                    413, "request_too_large", "The request body is too large."
                )
        except ValueError as exc:
            raise _WebError(
                400, "invalid_request", "The request is malformed."
            ) from exc
    consumed = 0
    async for chunk in request.stream():
        consumed += len(chunk)
        if consumed > max_bytes:
            raise _WebError(413, "request_too_large", "The request body is too large.")


def create_app(
    *,
    config: Optional[WebConfig] = None,
    runtime_factory: Callable[[], ClankerLM] = ClankerLM,
    clock: Callable[[], float] = time.monotonic,
    token_factory: Callable[[], str] = lambda: secrets.token_urlsafe(32),
) -> Starlette:
    """Create one single-loop ASGI application.

    ``clock``, ``runtime_factory``, and ``token_factory`` make expiry, resource
    cleanup, and cookie behavior directly observable without sleeping or
    mocking global state.
    """

    resolved = config or WebConfig()
    registry = SessionRegistry(
        resolved,
        runtime_factory=runtime_factory,
        clock=clock,
        token_factory=token_factory,
    )
    assets = _load_assets()
    release_feed = _load_release_feed(assets["releases.json"])
    release_payload = {
        "schema_version": release_feed["schema_version"],
        "running_package_version": __version__,
        "deployed_build_commit": resolved.resolved_build_commit,
        "latest_shipped_release": release_feed["latest_shipped_release"],
        "releases": release_feed["releases"],
    }

    def authorize(request: Request, *, mutation: bool = False) -> None:
        if resolved.deployed:
            login = request.headers.get("tailscale-user-login")
            if login not in resolved.allowed_users:
                raise _WebError(403, "forbidden", "This user is not allowed.")
        if mutation and request.headers.get("origin") != resolved.allowed_origin:
            raise _WebError(
                403, "forbidden_origin", "The request origin is not allowed."
            )

    def authorize_bootstrap(request: Request) -> None:
        authorize(request)
        if not resolved.deployed:
            return
        fetch_site = request.headers.get("sec-fetch-site")
        if fetch_site in {None, "none", "same-origin"}:
            return
        user_navigation = (
            fetch_site in {"same-site", "cross-site"}
            and request.headers.get("sec-fetch-mode") == "navigate"
            and request.headers.get("sec-fetch-dest") == "document"
            and request.headers.get("sec-fetch-user") == "?1"
        )
        if not user_navigation:
            raise _WebError(
                403,
                "forbidden_site",
                "Cross-site session bootstrap is not allowed.",
            )

    def lookup(request: Request, *, create: bool) -> tuple[Optional[WebSession], bool]:
        session_id = request.cookies.get(resolved.cookie_name)
        if create:
            try:
                return registry.get_or_create(session_id)
            except SessionCapacityError as exc:
                raise _WebError(
                    503,
                    "session_capacity",
                    "The service has no available session slots. Try again later.",
                ) from exc
        return registry.get(session_id), False

    def attach_cookie(response: Response, session: WebSession) -> None:
        response.set_cookie(
            resolved.cookie_name,
            session.session_id,
            max_age=max(1, int(resolved.session_idle_seconds)),
            path="/",
            secure=True,
            httponly=True,
            samesite="strict",
        )

    def admit(session: WebSession) -> None:
        if not registry.admit_request(session):
            raise _WebError(
                429, "rate_limited", "Too many requests. Try again shortly."
            )

    async def index(request: Request) -> Response:
        authorize_bootstrap(request)
        session, _ = lookup(request, create=True)
        assert session is not None
        response = Response(assets["index.html"], media_type="text/html")
        attach_cookie(response, session)
        return response

    async def asset(request: Request) -> Response:
        authorize(request)
        name = request.path_params["name"]
        if name not in {"app.css", "app.js"}:
            return _error(404, "not_found", "The requested resource was not found.")
        media_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        return Response(assets[name], media_type=media_type)

    async def health(_: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def releases(request: Request) -> Response:
        authorize(request)
        return _limited_json_response(
            release_payload,
            max_bytes=_RELEASE_FEED_MAX_BYTES,
            overflow_message="The release feed is too large.",
        )

    async def chat(request: Request) -> Response:
        authorize(request, mutation=True)
        value = await _read_json(request, max_bytes=resolved.max_body_bytes)
        if set(value) != {"message"} or not isinstance(value.get("message"), str):
            raise _WebError(400, "invalid_request", "Provide exactly one text message.")
        message = value["message"]
        if not message.strip():
            raise _WebError(400, "invalid_message", "The message must not be empty.")
        if len(message.encode("utf-8")) > resolved.max_message_bytes:
            raise _WebError(413, "message_too_large", "The message is too large.")

        session, created = lookup(request, create=True)
        assert session is not None
        admit(session)
        if session.turns >= resolved.max_turns:
            raise _WebError(
                429,
                "turn_limit",
                "This session has reached its turn limit. Reset it to continue.",
            )
        session.turns += 1
        try:
            result = session.runtime.process(message)
        except Exception as exc:
            registry.drop(session.session_id)
            raise _WebError(
                500, "runtime_error", "The runtime could not process that message."
            ) from exc

        payload = {
            "response": result.response,
            "evidence": {
                "answer_status": result.contract.status.value,
                "source": result.contract.source.value,
                "memory_revision": result.memory_revision,
                "vadug": result.predicted_state.to_dict(),
                "truth": result.contract.truth.value,
                "certainty": result.contract.certainty,
            },
        }
        try:
            response = _limited_json_response(
                payload,
                max_bytes=resolved.max_chat_response_bytes,
                overflow_message="The generated response is too large.",
            )
        except _WebError:
            if created:
                registry.drop(session.session_id)
            raise
        attach_cookie(response, session)
        return response

    async def reset(request: Request) -> Response:
        authorize(request, mutation=True)
        await _discard_bounded_body(request, max_bytes=resolved.max_body_bytes)
        session, _ = lookup(request, create=False)
        if session is not None:
            admit(session)
            registry.drop(session.session_id)
        response = JSONResponse({"reset": True})
        response.delete_cookie(
            resolved.cookie_name,
            path="/",
            secure=True,
            httponly=True,
            samesite="strict",
        )
        return response

    async def export(request: Request) -> Response:
        authorize(request)
        session, _ = lookup(request, create=False)
        if session is None:
            raise _WebError(
                404, "session_not_found", "There is no active session to export."
            )
        admit(session)
        try:
            snapshot = session.runtime.to_dict()
        except Exception as exc:
            raise _WebError(
                500, "runtime_error", "The runtime could not export this session."
            ) from exc
        response = _limited_json_response(
            snapshot,
            max_bytes=resolved.max_export_bytes,
            overflow_message="The session is too large to export.",
        )
        response.headers["Content-Disposition"] = (
            'attachment; filename="clanker-lm-session.json"'
        )
        attach_cookie(response, session)
        return response

    @asynccontextmanager
    async def lifespan(_: Starlette):
        try:
            yield
        finally:
            registry.close_all()

    async def web_error_handler(_: Request, exc: _WebError) -> Response:
        return _error(exc.status_code, exc.code, exc.message)

    async def internal_error_handler(_: Request, _exc: Exception) -> Response:
        return _error(
            500,
            "internal_error",
            "The request could not be completed.",
        )

    app = Starlette(
        debug=False,
        lifespan=lifespan,
        middleware=[Middleware(_SecurityHeadersMiddleware)],
        routes=[
            Route("/", index, methods=["GET"]),
            Route("/assets/{name:str}", asset, methods=["GET"]),
            Route("/healthz", health, methods=["GET"]),
            Route("/api/releases", releases, methods=["GET"]),
            Route("/api/chat", chat, methods=["POST"]),
            Route("/api/reset", reset, methods=["POST"]),
            Route("/api/export", export, methods=["GET"]),
        ],
        exception_handlers={
            _WebError: web_error_handler,
            Exception: internal_error_handler,
        },
    )
    app.state.session_registry = registry
    app.state.web_config = resolved
    return app


def run_server(config: WebConfig, *, log_level: str = "info") -> None:
    """Run exactly one Uvicorn worker; intended for the lazy CLI adapter."""

    # Keep this defense at the launch boundary as well as WebConfig validation.
    # It protects callers that forged or mutated a frozen config object.
    _require_loopback_host(config.host)

    import uvicorn

    uvicorn.run(
        create_app(config=config),
        host=config.host,
        port=config.port,
        workers=1,
        access_log=False,
        log_level=log_level,
        proxy_headers=False,
        server_header=False,
    )


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "SESSION_COOKIE",
    "SessionRegistry",
    "WebConfig",
    "WebSession",
    "create_app",
    "run_server",
]
