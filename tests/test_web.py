"""Adversarial acceptance tests for the private Clanker-LM web adapter.

These tests intentionally exercise the ASGI boundary with real runtimes.  The web
layer is responsible for keeping those runtimes isolated, bounded, and private;
mocking ``process()`` would hide the SQLite-thread and memory-leakage failures this
suite is meant to catch.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from starlette.testclient import TestClient

from clanker_lm import ClankerLM, HeuristicAffectBackend
from clanker_lm.web import WebConfig, create_app, run_server


PUBLIC_ORIGIN = "https://clanker.example.ts.net"
ALLOWED_LOGIN = "owner@example.com"
LOGIN_HEADER = "Tailscale-User-Login"


def _config(**logical_overrides: Any) -> WebConfig:
    """Construct the exact public deployed-mode configuration."""

    aliases = {
        "idle_seconds": "session_idle_seconds",
        "rate_requests": "rate_limit",
    }
    values: dict[str, Any] = {
        "public_origin": PUBLIC_ORIGIN,
        "deployed": True,
        "allowed_users": (ALLOWED_LOGIN,),
    }
    values.update({aliases.get(name, name): value for name, value in logical_overrides.items()})
    return WebConfig(**values)


def _headers(*, origin: str | None = PUBLIC_ORIGIN, login: str = ALLOWED_LOGIN) -> dict[str, str]:
    headers = {LOGIN_HEADER: login}
    if origin is not None:
        headers["Origin"] = origin
    return headers


def _client(app: Any) -> TestClient:
    # Secure cookies are deliberately ignored by an HTTP test origin.  Testing
    # against the configured HTTPS origin exercises actual browser semantics.
    return TestClient(app, base_url=PUBLIC_ORIGIN)


def _chat(
    client: TestClient,
    message: str,
    *,
    expected_status: int = 200,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    response = client.post(
        "/api/chat",
        json={"message": message},
        headers=headers or _headers(),
    )
    assert response.status_code == expected_status, response.text
    payload = response.json()
    if expected_status == 200:
        assert set(payload) >= {"response", "evidence"}
        assert set(payload["evidence"]) >= {
            "answer_status",
            "source",
            "memory_revision",
            "vadug",
        }
    else:
        assert set(payload) == {"error"}
        assert set(payload["error"]) >= {"code", "message"}
    return payload


class ManualClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class TrackedRuntime:
    runtime: ClankerLM
    close_calls: int = 0
    calls: list[str] = field(default_factory=list)

    def process(self, message: str) -> Any:
        self.calls.append(message)
        return self.runtime.process(message)

    def dumps(self, *, indent: int | None = 2) -> str:
        return self.runtime.dumps(indent=indent)

    def to_dict(self) -> dict[str, Any]:
        return self.runtime.to_dict()

    def close(self) -> None:
        self.close_calls += 1
        self.runtime.close()


class TrackingFactory:
    def __init__(self) -> None:
        self.instances: list[TrackedRuntime] = []

    def __call__(self) -> TrackedRuntime:
        runtime = TrackedRuntime(ClankerLM(affect_backend=HeuristicAffectBackend()))
        self.instances.append(runtime)
        return runtime


@dataclass
class LargeOutputRuntime(TrackedRuntime):
    chat_text: str = ""
    export_text: str = ""

    def process(self, message: str) -> Any:
        result = super().process(message)
        if self.chat_text:
            result.response = self.chat_text
        return result

    def to_dict(self) -> dict[str, Any]:
        if self.export_text:
            return {"oversized_private_snapshot": self.export_text}
        return super().to_dict()


class LargeOutputFactory(TrackingFactory):
    def __init__(self, *, chat_text: str = "", export_text: str = "") -> None:
        super().__init__()
        self.chat_text = chat_text
        self.export_text = export_text

    def __call__(self) -> LargeOutputRuntime:
        runtime = LargeOutputRuntime(
            ClankerLM(affect_backend=HeuristicAffectBackend()),
            chat_text=self.chat_text,
            export_text=self.export_text,
        )
        self.instances.append(runtime)
        return runtime


def _cookie_pair(client: TestClient) -> tuple[str, str]:
    cookies = list(client.cookies.items())
    assert len(cookies) == 1, cookies
    return cookies[0]


def test_test_extra_declares_an_explicit_bounded_httpx_requirement() -> None:
    pyproject = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    optional_dependencies = re.search(
        r"(?ms)^\[project\.optional-dependencies\]\s*(.*?)(?=^\[)",
        pyproject,
    )
    assert optional_dependencies is not None
    test_extra = re.search(
        r"(?ms)^test\s*=\s*\[(.*?)\]",
        optional_dependencies.group(1),
    )
    assert test_extra is not None
    requirements = re.findall(r'''["']([^"']+)["']''', test_extra.group(1))
    assert [requirement for requirement in requirements if requirement.startswith("httpx")] == [
        "httpx>=0.27,<1"
    ]


@pytest.mark.parametrize("deployed", [False, True], ids=["local", "deployed"])
@pytest.mark.parametrize(
    "host",
    [
        "0.0.0.0",
        "192.168.1.50",
        "10.0.0.8",
        "clanker.lan",
        "workstation",
        "127.0.0.2",
        "::",
    ],
)
def test_web_config_rejects_every_nonloopback_bind_host(
    host: str,
    deployed: bool,
) -> None:
    kwargs: dict[str, Any] = {"host": host, "deployed": deployed}
    if deployed:
        kwargs.update(
            public_origin=PUBLIC_ORIGIN,
            allowed_users=(ALLOWED_LOGIN,),
        )
    with pytest.raises(ValueError):
        WebConfig(**kwargs)


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
@pytest.mark.parametrize("deployed", [False, True], ids=["local", "deployed"])
def test_web_config_accepts_only_the_supported_loopback_spellings(
    host: str,
    deployed: bool,
) -> None:
    kwargs: dict[str, Any] = {"host": host, "deployed": deployed}
    if deployed:
        kwargs.update(
            public_origin=PUBLIC_ORIGIN,
            allowed_users=(ALLOWED_LOGIN,),
        )
    assert WebConfig(**kwargs).host == host


@pytest.mark.parametrize(
    "origin",
    [
        "http://clanker.example.ts.net",
        "HTTPS://clanker.example.ts.net",
        "https://CLANKER.example.ts.net",
        "https://clanker.example.ts.net.",
        "https://user@clanker.example.ts.net",
        "https://user:password@clanker.example.ts.net",
        "https://clanker.example.ts.net/",
        "https://clanker.example.ts.net/path",
        "https://clanker.example.ts.net?query=yes",
        "https://clanker.example.ts.net#fragment",
        "https://clanker.example.ts.net:443",
    ],
)
def test_deployed_public_origin_rejects_noncanonical_or_unsafe_forms(origin: str) -> None:
    with pytest.raises(ValueError):
        WebConfig(
            deployed=True,
            public_origin=origin,
            allowed_users=(ALLOWED_LOGIN,),
        )


@pytest.mark.parametrize(
    "origin",
    [
        "https://clanker.example.ts.net",
        "https://clanker.example.ts.net:8444",
    ],
)
def test_deployed_public_origin_accepts_canonical_https_and_nondefault_port(
    origin: str,
) -> None:
    config = WebConfig(
        deployed=True,
        public_origin=origin,
        allowed_users=(ALLOWED_LOGIN,),
    )
    assert config.allowed_origin == origin


def test_local_mode_may_retain_an_explicit_http_origin() -> None:
    config = WebConfig(public_origin="http://localhost:8765")
    assert config.allowed_origin == "http://localhost:8765"


def test_run_server_rechecks_loopback_before_calling_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = WebConfig()
    # Simulate a caller bypassing frozen-dataclass construction.  ``run_server``
    # is the final process-boundary guard and must not trust prior validation.
    object.__setattr__(config, "host", "0.0.0.0")
    uvicorn_calls: list[dict[str, Any]] = []

    def fake_run(*_args: Any, **kwargs: Any) -> None:
        uvicorn_calls.append(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    with pytest.raises(ValueError):
        run_server(config)
    assert uvicorn_calls == []


def test_index_bootstraps_one_secure_session_and_reuses_it() -> None:
    factory = TrackingFactory()
    app = create_app(config=_config(), runtime_factory=factory)
    with _client(app) as client:
        first = client.get("/", headers=_headers())
        first_cookie = _cookie_pair(client)
        second = client.get("/", headers=_headers())
        second_cookie = _cookie_pair(client)

        assert first.status_code == second.status_code == 200
        assert first_cookie == second_cookie
        assert len(factory.instances) == 1
        assert app.state.session_registry.active_count == 1

    set_cookie = first.headers["set-cookie"].lower()
    assert "secure" in set_cookie
    assert "httponly" in set_cookie
    assert "samesite=strict" in set_cookie
    assert factory.instances[0].close_calls == 1


@pytest.mark.parametrize("fetch_site", ["same-site", "cross-site", "other"])
def test_deployed_cross_site_shell_bootstrap_is_rejected_without_allocation(
    fetch_site: str,
) -> None:
    factory = TrackingFactory()
    app = create_app(config=_config(), runtime_factory=factory)
    headers = {
        LOGIN_HEADER: ALLOWED_LOGIN,
        "Sec-Fetch-Site": fetch_site,
    }
    with _client(app) as client:
        response = client.get("/", headers=headers)

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "forbidden_site"
        assert "set-cookie" not in response.headers
        assert list(client.cookies.items()) == []
        assert factory.instances == []
        assert app.state.session_registry.active_count == 0


@pytest.mark.parametrize("fetch_site", [None, "none", "same-origin"])
def test_direct_and_same_origin_shell_navigation_bootstrap_one_session(
    fetch_site: str | None,
) -> None:
    factory = TrackingFactory()
    app = create_app(config=_config(), runtime_factory=factory)
    headers = {LOGIN_HEADER: ALLOWED_LOGIN}
    if fetch_site is not None:
        headers["Sec-Fetch-Site"] = fetch_site

    with _client(app) as client:
        response = client.get("/", headers=headers)
        assert response.status_code == 200
        assert "set-cookie" in response.headers
        assert len(factory.instances) == 1
        assert app.state.session_registry.active_count == 1


def test_concurrent_first_turns_from_bootstrapped_cookie_share_one_ordered_runtime() -> None:
    fact = "The launch is on Monday."
    question = "When is the launch?"
    factory = TrackingFactory()
    app = create_app(config=_config(), runtime_factory=factory)

    async def exercise() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport,
                base_url=PUBLIC_ORIGIN,
                headers={LOGIN_HEADER: ALLOWED_LOGIN},
            ) as client:
                shell = await client.get("/")
                assert shell.status_code == 200
                assert "set-cookie" in shell.headers
                assert len(factory.instances) == 1

                first, second = await asyncio.gather(
                    client.post(
                        "/api/chat",
                        json={"message": fact},
                        headers={"Origin": PUBLIC_ORIGIN},
                    ),
                    client.post(
                        "/api/chat",
                        json={"message": question},
                        headers={"Origin": PUBLIC_ORIGIN},
                    ),
                )
                assert app.state.session_registry.active_count == 1
                return first, second

    first, second = asyncio.run(exercise())
    assert first.status_code == second.status_code == 200
    assert len(factory.instances) == 1
    assert factory.instances[0].calls == [fact, question]
    # Read-only questions retain the fact turn's revision; the answered result
    # plus the call trace proves the fact completed before the question ran.
    assert first.json()["evidence"]["memory_revision"] > 0
    assert (
        first.json()["evidence"]["memory_revision"]
        == second.json()["evidence"]["memory_revision"]
    )
    assert "monday" in second.json()["response"].lower()
    assert factory.instances[0].close_calls == 1


def test_fact_then_question_uses_the_same_cookie_session() -> None:
    app = create_app(config=_config())
    with _client(app) as client:
        _chat(client, "The launch is on Monday.")
        result = _chat(client, "When is the launch?")

    assert "monday" in result["response"].lower()
    assert result["evidence"]["answer_status"] == "answered"


def test_two_cookie_jars_have_distinct_sessions_and_no_fact_leakage() -> None:
    app = create_app(config=_config())
    with _client(app) as first, _client(app) as second:
        _chat(first, "My name is Alice.")
        _chat(second, "My name is Bob.")
        assert _cookie_pair(first) != _cookie_pair(second)

        _chat(first, "The launch is on Monday.")
        leaked = _chat(second, "When is the launch?")
        retained = _chat(first, "When is the launch?")

    assert "monday" not in leaked["response"].lower()
    assert leaked["evidence"]["answer_status"] != "answered"
    assert "monday" in retained["response"].lower()


def test_sequentially_interleaved_sessions_preserve_their_own_ordering() -> None:
    app = create_app(config=_config())
    with _client(app) as first, _client(app) as second:
        _chat(first, "The launch is on Monday.")
        _chat(second, "The concert is on Tuesday.")
        launch = _chat(first, "When is the launch?")
        concert = _chat(second, "When is the concert?")

    assert "monday" in launch["response"].lower()
    assert "tuesday" in concert["response"].lower()


def test_repeated_requests_do_not_move_sqlite_across_threads() -> None:
    app = create_app(config=_config())
    with _client(app) as client:
        payloads = [_chat(client, f"This is message {number}.") for number in range(12)]

    serialized = json.dumps(payloads).lower()
    assert "sqlite" not in serialized
    assert "thread" not in serialized


def test_reset_forgets_closes_and_rotates_the_session() -> None:
    factory = TrackingFactory()
    app = create_app(config=_config(), runtime_factory=factory)
    with _client(app) as client:
        _chat(client, "The launch is on Monday.")
        old_cookie = _cookie_pair(client)
        old_runtime = factory.instances[-1]

        reset = client.post("/api/reset", headers=_headers())
        assert reset.status_code == 200, reset.text
        assert reset.json() == {"reset": True}
        assert old_runtime.close_calls == 1

        forgotten = _chat(client, "When is the launch?")
        new_cookie = _cookie_pair(client)

    assert new_cookie != old_cookie
    assert len(factory.instances) == 2
    assert forgotten["evidence"]["answer_status"] != "answered"
    # Lifespan shutdown closes the replacement as well, without double closing.
    assert [runtime.close_calls for runtime in factory.instances] == [1, 1]


def test_export_is_a_reloadable_clanker_snapshot_with_session_state() -> None:
    app = create_app(config=_config())
    with _client(app) as client:
        _chat(client, "The launch is on Monday.")
        exported = client.get("/api/export", headers=_headers(origin=None))

    assert exported.status_code == 200, exported.text
    assert exported.headers["content-type"].startswith("application/json")
    restored = ClankerLM.loads(exported.text, affect_backend=HeuristicAffectBackend())
    try:
        result = restored.process("When is the launch?")
        assert "monday" in result.response.lower()
    finally:
        restored.close()


def test_oversized_chat_response_fails_closed_below_the_hard_byte_ceiling() -> None:
    limit = 1024
    secret = "private-chat-output:" + "é" * limit
    assert len(secret.encode("utf-8")) > limit
    factory = LargeOutputFactory(chat_text=secret)
    app = create_app(
        config=_config(max_chat_response_bytes=limit),
        runtime_factory=factory,
    )
    with _client(app) as client:
        response = client.post(
            "/api/chat",
            json={"message": "Hello."},
            headers=_headers(),
        )

    assert response.status_code == 507
    assert response.json()["error"]["code"] == "response_too_large"
    assert secret not in response.text
    assert "private-chat-output" not in response.text
    assert len(response.content) <= limit


def test_oversized_export_fails_closed_below_the_hard_byte_ceiling() -> None:
    limit = 1024
    secret = "private-export-output:" + "é" * limit
    assert len(secret.encode("utf-8")) > limit
    factory = LargeOutputFactory(export_text=secret)
    app = create_app(
        config=_config(max_export_bytes=limit),
        runtime_factory=factory,
    )
    with _client(app) as client:
        shell = client.get("/", headers=_headers())
        assert shell.status_code == 200
        response = client.get("/api/export", headers=_headers(origin=None))

    assert response.status_code == 507
    assert response.json()["error"]["code"] == "response_too_large"
    assert secret not in response.text
    assert "private-export-output" not in response.text
    assert len(response.content) <= limit


def test_idle_expiry_closes_runtime_and_forgets_its_memory() -> None:
    clock = ManualClock()
    factory = TrackingFactory()
    app = create_app(
        config=_config(idle_seconds=10.0),
        runtime_factory=factory,
        clock=clock,
    )
    with _client(app) as client:
        _chat(client, "The launch is on Monday.")
        expired = factory.instances[-1]
        clock.advance(10.01)
        result = _chat(client, "When is the launch?")

        assert expired.close_calls == 1
        assert len(factory.instances) == 2
        assert result["evidence"]["answer_status"] != "answered"


def test_full_active_capacity_rejects_new_session_without_evicting_or_forgetting() -> None:
    clock = ManualClock()
    factory = TrackingFactory()
    app = create_app(
        config=_config(idle_seconds=100.0, max_sessions=2),
        runtime_factory=factory,
        clock=clock,
    )
    with _client(app) as first, _client(app) as second, _client(app) as third:
        _chat(first, "The launch is on Monday.")
        first_runtime = factory.instances[-1]
        clock.advance(1.0)

        _chat(second, "The concert is on Tuesday.")
        second_runtime = factory.instances[-1]
        clock.advance(1.0)

        rejected = third.get("/", headers=_headers())
        assert rejected.status_code == 503
        assert rejected.json()["error"]["code"] == "session_capacity"
        assert "set-cookie" not in rejected.headers
        assert app.state.session_registry.active_count == 2
        assert len(factory.instances) == 2
        assert first_runtime.close_calls == 0
        assert second_runtime.close_calls == 0

        first_retained = _chat(first, "When is the launch?")
        second_retained = _chat(second, "When is the concert?")
        assert "monday" in first_retained["response"].lower()
        assert "tuesday" in second_retained["response"].lower()

    assert first_runtime.close_calls == 1
    assert second_runtime.close_calls == 1


@pytest.mark.parametrize(
    ("message", "limit"),
    [
        ("a" * 4097, 4096),
        ("é" * 2049, 4096),
    ],
)
def test_message_quota_counts_utf8_bytes(message: str, limit: int) -> None:
    app = create_app(config=_config(max_message_bytes=limit))
    with _client(app) as client:
        error = _chat(client, message, expected_status=413)
    assert error["error"]["code"] == "message_too_large"


def test_request_body_quota_precedes_json_schema_parsing() -> None:
    app = create_app(config=_config(max_body_bytes=16 * 1024))
    body = json.dumps({"message": "ok", "padding": "x" * (16 * 1024)}).encode()
    assert len(body) > 16 * 1024
    with _client(app) as client:
        response = client.post(
            "/api/chat",
            content=body,
            headers={**_headers(), "Content-Type": "application/json"},
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_message_and_body_quotas_accept_the_exact_byte_boundary() -> None:
    app = create_app(
        config=_config(max_message_bytes=4096, max_body_bytes=16 * 1024)
    )
    exact_message = "a" * 4096
    compact = b'{"message":"ok"}'
    exact_body = compact + b" " * (16 * 1024 - len(compact))
    assert len(exact_body) == 16 * 1024

    with _client(app) as client:
        accepted_message = client.post(
            "/api/chat",
            json={"message": exact_message},
            headers=_headers(),
        )
        accepted_body = client.post(
            "/api/chat",
            content=exact_body,
            headers={**_headers(), "Content-Type": "application/json"},
        )

    assert accepted_message.status_code == 200, accepted_message.text
    assert accepted_body.status_code == 200, accepted_body.text


def test_per_session_turn_quota_is_deterministic() -> None:
    app = create_app(config=_config(max_turns=1, rate_requests=100))
    with _client(app) as client:
        _chat(client, "Hello.")
        limited = _chat(client, "Hello again.", expected_status=429)
    assert limited["error"]["code"] == "turn_limit"


def test_rate_quota_resets_only_after_the_injected_window() -> None:
    clock = ManualClock()
    app = create_app(
        config=_config(max_turns=100, rate_requests=2, rate_window_seconds=60.0),
        clock=clock,
    )
    with _client(app) as client:
        _chat(client, "First.")
        _chat(client, "Second.")
        limited = _chat(client, "Third.", expected_status=429)
        clock.advance(60.01)
        recovered = _chat(client, "Fourth.")
    assert limited["error"]["code"] == "rate_limited"
    assert recovered["response"]


def test_deployed_mode_requires_an_exact_tailscale_login() -> None:
    app = create_app(config=_config())
    with _client(app) as client:
        missing = client.get("/")
        wrong = client.get("/", headers=_headers(login="Owner@example.com"))
        suffix = client.get("/", headers=_headers(login=f"{ALLOWED_LOGIN}.evil"))
        accepted = client.get("/", headers=_headers())

    for denied in (missing, wrong, suffix):
        assert denied.status_code == 403
        assert denied.json()["error"]["code"] == "forbidden"
        assert "Tailscale-User-Login" not in denied.text
    assert accepted.status_code == 200


def test_mutations_require_the_exact_configured_origin() -> None:
    app = create_app(config=_config())
    with _client(app) as client:
        missing = client.post(
            "/api/chat",
            json={"message": "Hello."},
            headers=_headers(origin=None),
        )
        wrong = client.post(
            "/api/chat",
            json={"message": "Hello."},
            headers=_headers(origin=f"{PUBLIC_ORIGIN}.evil"),
        )
        wrong_scheme = client.post(
            "/api/chat",
            json={"message": "Hello."},
            headers=_headers(origin=PUBLIC_ORIGIN.replace("https://", "http://")),
        )
        accepted = client.post(
            "/api/chat",
            json={"message": "Hello."},
            headers=_headers(),
        )
        reset_missing = client.post(
            "/api/reset",
            headers=_headers(origin=None),
        )
        reset_wrong = client.post(
            "/api/reset",
            headers=_headers(origin=f"{PUBLIC_ORIGIN}/"),
        )
        reset_accepted = client.post("/api/reset", headers=_headers())

    for denied in (missing, wrong, wrong_scheme, reset_missing, reset_wrong):
        assert denied.status_code == 403
        assert denied.json()["error"]["code"] == "forbidden_origin"
    assert accepted.status_code == 200
    assert reset_accepted.status_code == 200


def test_session_cookie_has_strict_browser_security_attributes() -> None:
    app = create_app(config=_config())
    with _client(app) as client:
        response = client.post(
            "/api/chat",
            json={"message": "Hello."},
            headers=_headers(),
        )

    assert response.status_code == 200
    cookie = response.headers["set-cookie"].lower()
    assert re.match(r"^[a-z0-9_-]+=[a-z0-9_-]+;", cookie)
    cookie_name, cookie_tail = cookie.split("=", 1)
    token = cookie_tail.split(";", 1)[0]
    assert cookie_name == app.state.web_config.cookie_name
    assert len(token) >= 32
    assert "secure" in cookie
    assert "httponly" in cookie
    assert "samesite=strict" in cookie
    assert "path=/" in cookie


def _assert_hardened(response: Any) -> None:
    headers = response.headers
    csp = headers["content-security-policy"].lower()
    assert "default-src 'none'" in csp
    assert "script-src 'self'" in csp
    assert "style-src 'self'" in csp
    assert "connect-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert headers["x-content-type-options"].lower() == "nosniff"
    assert headers["referrer-policy"].lower() == "no-referrer"
    assert headers["cache-control"].lower() == "no-store"
    assert headers["x-frame-options"].lower() == "deny"


def test_security_headers_cover_successes_and_fail_closed_errors() -> None:
    app = create_app(config=_config())
    with _client(app) as client:
        success = client.get("/", headers=_headers())
        denied = client.get("/")
        missing = client.get("/does-not-exist", headers=_headers())

    for response in (success, denied, missing):
        _assert_hardened(response)


@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        (b"{", "application/json"),
        (b"null", "application/json"),
        (b"[]", "application/json"),
        (b'{"message": 3}', "application/json"),
        (b'{"message": "ok"}', "text/plain"),
    ],
)
def test_malformed_chat_requests_fail_closed(body: bytes, content_type: str) -> None:
    app = create_app(config=_config())
    with _client(app) as client:
        response = client.post(
            "/api/chat",
            content=body,
            headers={**_headers(), "Content-Type": content_type},
        )
    assert response.status_code in {400, 415}
    assert set(response.json()) == {"error"}
    _assert_hardened(response)


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        ("GET", "/api/chat", 405),
        ("GET", "/api/reset", 405),
        ("POST", "/api/export", 405),
        ("PUT", "/", 405),
        ("GET", "/api/import", 404),
        ("GET", "/assets/%2e%2e/web.py", 404),
        ("GET", "/assets/%2fetc%2fpasswd", 404),
    ],
)
def test_unknown_methods_and_paths_fail_closed(method: str, path: str, expected: int) -> None:
    app = create_app(config=_config())
    with _client(app) as client:
        response = client.request(method, path, headers=_headers())
    assert response.status_code == expected
    assert "traceback" not in response.text.lower()
    _assert_hardened(response)


def test_health_is_minimal_public_and_does_not_create_or_leak_a_session() -> None:
    app = create_app(config=_config())
    with _client(app) as client:
        _chat(client, "The launch secret is cobalt-seven.")
        assert app.state.session_registry.active_count == 1
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "set-cookie" not in response.headers
    lowered = response.text.lower()
    assert "session" not in lowered
    assert "memory" not in lowered
    assert "cobalt-seven" not in lowered
    _assert_hardened(response)


def test_ui_uses_safe_dom_apis_local_assets_and_accessibility_hooks() -> None:
    app = create_app(config=_config())
    with _client(app) as client:
        html_response = client.get("/", headers=_headers())
        js_response = client.get("/assets/app.js", headers=_headers())
        css_response = client.get("/assets/app.css", headers=_headers())

    assert html_response.status_code == js_response.status_code == css_response.status_code == 200
    html = html_response.text
    js = js_response.text
    css = css_response.text
    combined = "\n".join((html, js))

    assert "innerHTML" not in combined
    assert "insertAdjacentHTML" not in combined
    assert "document.write" not in combined
    assert "textContent" in js
    assert not re.search(r'''(?:src|href)\s*=\s*["'](?:https?:)?//''', html, re.I)
    assert not re.search(r'''fetch\s*\(\s*["'](?:https?:)?//''', js, re.I)
    assert not re.search(r'''url\s*\(\s*["']?(?:https?:)?//''', css, re.I)
    assert not re.search(r"@import\b", css, re.I)
    assert not re.search(r"\son[a-z]+\s*=", html, re.I)

    scripts = re.findall(r"<script\b([^>]*)>(.*?)</script>", html, re.I | re.S)
    assert scripts
    assert all(re.search(r"\bsrc\s*=", attrs, re.I) and not body.strip() for attrs, body in scripts)

    lowered_html = html.lower()
    assert 'name="viewport"' in lowered_html
    assert "aria-live" in lowered_html
    assert "aria-label" in lowered_html or "<label" in lowered_html
    assert "@media" in css


def test_ui_has_one_logical_in_flight_guard_for_submit_keyboard_and_reset() -> None:
    app = create_app(config=_config())
    with _client(app) as client:
        response = client.get("/assets/app.js", headers=_headers())

    assert response.status_code == 200
    js = response.text
    assert re.search(r"\blet\s+requestInFlight\s*=\s*false\s*;", js)
    assert js.count("if (requestInFlight)") >= 2
    assert js.count("requestInFlight = true;") >= 2
    # Declaration plus independent submit/reset finally paths.
    assert js.count("requestInFlight = false;") >= 3
    assert re.search(
        r'message\.addEventListener\("keydown"[\s\S]+?submitMessage\(\)',
        js,
    )


def test_submitted_message_is_not_written_to_application_logs(caplog: pytest.LogCaptureFixture) -> None:
    marker = "private-message-8c7dbf"
    app = create_app(config=_config())
    with _client(app) as client:
        _chat(client, marker)
    assert marker not in caplog.text
