from __future__ import annotations

import copy
import json
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest

from clanker_lm import web as web_module
from clanker_lm.web_release_verify import verify_merged_releases


MERGE_COMMIT = "9ae77f072f8afda0b1d2b757ab492757cabff0f8"


class FakeResponse:
    def __init__(self, payload: Any) -> None:
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self._stream = BytesIO(data)

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


def _feed() -> dict[str, Any]:
    assets = web_module._load_assets()
    return json.loads(assets["releases.json"].decode("utf-8"))


def _merged_payload(*, number: int = 106, commit: str = MERGE_COMMIT) -> dict[str, Any]:
    return {
        "number": number,
        "merged_at": "2026-09-04T08:28:56Z",
        "merge_commit_sha": commit,
    }


def test_release_verifier_accepts_pr_merged_at_the_recorded_commit() -> None:
    requests: list[Any] = []

    def opener(request: Any, *, timeout: int) -> FakeResponse:
        requests.append((request, timeout))
        return FakeResponse(_merged_payload())

    verify_merged_releases(_feed(), opener=opener, token="test-token")
    assert len(requests) == 1
    request, timeout = requests[0]
    assert request.full_url.endswith("/pulls/106")
    assert request.headers["Authorization"] == "Bearer test-token"
    assert timeout == 15


def test_coherent_but_unmerged_pr_identity_fails_external_release_verification() -> None:
    feed = _feed()
    feed["latest_shipped_release"]["release_id"] = "pr-107"
    feed["releases"][0]["release_id"] = "pr-107"
    feed["releases"][0]["evidence"][0]["url"] = (
        "https://github.com/deucebucket/clanker/pull/107"
    )
    assert web_module._validate_release_feed(feed)

    def opener(_request: Any, *, timeout: int) -> FakeResponse:
        assert timeout == 15
        return FakeResponse(
            {"number": 107, "merged_at": None, "merge_commit_sha": None}
        )

    with pytest.raises(ValueError, match="not merged"):
        verify_merged_releases(feed, opener=opener)


def test_coherent_arbitrary_commit_fails_github_merge_commit_agreement() -> None:
    feed = _feed()
    invented = "e" * 40
    feed["latest_shipped_release"]["milestone_commit"] = invented
    feed["releases"][0]["milestone_commit"] = invented
    feed["releases"][0]["evidence"][1]["url"] = (
        f"https://github.com/deucebucket/clanker/commit/{invented}"
    )
    assert web_module._validate_release_feed(feed)

    with pytest.raises(ValueError, match="milestone commit disagrees"):
        verify_merged_releases(
            feed,
            opener=lambda *_args, **_kwargs: FakeResponse(_merged_payload()),
        )


def test_nonexistent_pr_fails_closed() -> None:
    feed = copy.deepcopy(_feed())
    feed["latest_shipped_release"]["release_id"] = "pr-999"
    feed["releases"][0]["release_id"] = "pr-999"
    feed["releases"][0]["evidence"][0]["url"] = (
        "https://github.com/deucebucket/clanker/pull/999"
    )
    assert web_module._validate_release_feed(feed)

    def opener(request: Any, *, timeout: int) -> FakeResponse:
        raise HTTPError(request.full_url, 404, "not found", {}, None)

    with pytest.raises(ValueError, match="could not verify pr-999"):
        verify_merged_releases(feed, opener=opener)


@pytest.mark.parametrize(
    "payload",
    [
        {"number": 999, "merged_at": "now", "merge_commit_sha": MERGE_COMMIT},
        {"number": 106, "merged_at": "now", "merge_commit_sha": "f" * 40},
        [],
        b"{",
        b"x" * (1024 * 1024 + 1),
    ],
)
def test_release_verifier_rejects_malformed_or_mismatched_api_results(
    payload: Any,
) -> None:
    with pytest.raises(ValueError):
        verify_merged_releases(
            _feed(),
            opener=lambda *_args, **_kwargs: FakeResponse(payload),
        )


def test_ci_runs_the_network_release_verifier() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "tests.yml"
    ).read_text(encoding="utf-8")
    assert "python -m clanker_lm.web_release_verify" in workflow
    assert "GITHUB_TOKEN: ${{ github.token }}" in workflow
    web_guide = (
        Path(__file__).resolve().parents[1] / "docs" / "CLANKER_LM_WEB.md"
    ).read_text(encoding="utf-8")
    assert "PR #112 row" not in web_guide
    assert "pr-112" not in web_guide.lower()
    assert "implementation PR for issue #112" in web_guide
    assert "actual number" in web_guide and "actual squash/merge SHA" in web_guide
