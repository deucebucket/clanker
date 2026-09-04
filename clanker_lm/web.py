"""Small, hardened ASGI surface for the deterministic Clanker-LM runtime.

The web extra is intentionally isolated in this module so importing
``clanker_lm`` never requires Starlette or Uvicorn.  Every request handler is
async and runtime calls stay on the server's single event-loop thread.
"""

from __future__ import annotations

import json
import mimetypes
import secrets
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
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

from .runtime import ClankerLM


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
SESSION_COOKIE = "clanker_lm_session"
_ASSET_NAMES = ("index.html", "app.css", "app.js")
_CONTENT_SECURITY_POLICY = (
    "default-src 'none'; base-uri 'none'; form-action 'self'; "
    "frame-ancestors 'none'; img-src 'self' data:; "
    "script-src 'self'; style-src 'self'; connect-src 'self'"
)


@dataclass(frozen=True)
class WebConfig:
    """Security and resource limits for one web process."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    public_origin: Optional[str] = None
    deployed: bool = False
    allowed_users: tuple[str, ...] = ()
    cookie_name: str = SESSION_COOKIE
    session_idle_seconds: float = 30 * 60
    max_sessions: int = 128
    rate_limit: int = 30
    rate_window_seconds: float = 60.0
    max_turns: int = 200
    max_message_bytes: int = 4 * 1024
    max_body_bytes: int = 16 * 1024

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host:
            raise ValueError("host must be a non-empty string")
        self._positive_integer("port", self.port, maximum=65535)
        self._positive_integer("max_sessions", self.max_sessions)
        self._positive_integer("rate_limit", self.rate_limit)
        self._positive_integer("max_turns", self.max_turns)
        self._positive_integer("max_message_bytes", self.max_message_bytes)
        self._positive_integer("max_body_bytes", self.max_body_bytes)
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
        if self.deployed:
            if self.public_origin is None:
                raise ValueError("deployed mode requires an explicit public_origin")
            if not self.allowed_users:
                raise ValueError("deployed mode requires at least one allowed user")

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


@dataclass
class WebSession:
    """One browser's runtime and bounded request accounting."""

    session_id: str
    runtime: ClankerLM
    created_at: float
    last_seen: float
    turns: int = 0
    requests: Deque[float] = field(default_factory=deque)


class SessionRegistry:
    """Event-loop-confined LRU registry with deterministic test seams."""

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
        while len(self._sessions) >= self.config.max_sessions:
            _, oldest = self._sessions.popitem(last=False)
            self._close_runtime(oldest.runtime)

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


def _load_assets() -> Mapping[str, bytes]:
    root = resources.files("clanker_lm.web_assets")
    return {name: root.joinpath(name).read_bytes() for name in _ASSET_NAMES}


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

    def authorize(request: Request, *, mutation: bool = False) -> None:
        if resolved.deployed:
            login = request.headers.get("tailscale-user-login")
            if login not in resolved.allowed_users:
                raise _WebError(403, "forbidden", "This user is not allowed.")
        if mutation and request.headers.get("origin") != resolved.allowed_origin:
            raise _WebError(
                403, "forbidden_origin", "The request origin is not allowed."
            )

    def lookup(request: Request, *, create: bool) -> tuple[Optional[WebSession], bool]:
        session_id = request.cookies.get(resolved.cookie_name)
        if create:
            return registry.get_or_create(session_id)
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
        authorize(request)
        return Response(assets["index.html"], media_type="text/html")

    async def asset(request: Request) -> Response:
        authorize(request)
        name = request.path_params["name"]
        if name not in {"app.css", "app.js"}:
            return _error(404, "not_found", "The requested resource was not found.")
        media_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        return Response(assets[name], media_type=media_type)

    async def health(_: Request) -> Response:
        return JSONResponse({"status": "ok"})

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

        session, _ = lookup(request, create=True)
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

        response = JSONResponse(
            {
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
        )
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
        response = JSONResponse(snapshot)
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
