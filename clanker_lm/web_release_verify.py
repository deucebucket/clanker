"""Verify packaged web milestones against merged GitHub pull requests.

The running application deliberately stays network-free. CI and release
operators run this module before deployment so an internally coherent but
invented pull-request/commit pair cannot enter the shipped feed.
"""

from __future__ import annotations

import json
import os
from importlib import resources
from typing import Any, Callable, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .web import _load_release_feed


_API_ROOT = "https://api.github.com/repos/deucebucket/clanker/pulls"
_MAX_API_RESPONSE_BYTES = 1024 * 1024


def _read_response(response: Any) -> bytes:
    data = response.read(_MAX_API_RESPONSE_BYTES + 1)
    if len(data) > _MAX_API_RESPONSE_BYTES:
        raise ValueError("GitHub pull-request response exceeds the byte limit")
    return data


def verify_merged_releases(
    feed: Mapping[str, Any],
    *,
    opener: Callable[..., Any] = urlopen,
    token: Optional[str] = None,
) -> None:
    """Require every feed row to name a PR merged at its milestone commit."""

    releases = feed.get("releases")
    if not isinstance(releases, list) or not releases:
        raise ValueError("release feed has no milestones to verify")

    for release in releases:
        release_id = release.get("release_id") if isinstance(release, dict) else None
        if not isinstance(release_id, str) or not release_id.startswith("pr-"):
            raise ValueError("release feed has an invalid pull-request identity")
        pull_number = release_id.removeprefix("pr-")
        if (
            not pull_number.isdigit()
            or pull_number == "0"
            or pull_number.startswith("0")
        ):
            raise ValueError("release feed has an invalid pull-request number")
        milestone_commit = release.get("milestone_commit")
        if not isinstance(milestone_commit, str):
            raise ValueError("release feed has no milestone commit")

        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "clanker-web-release-verifier/1",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(f"{_API_ROOT}/{pull_number}", headers=headers)
        try:
            with opener(request, timeout=15) as response:
                payload = json.loads(_read_response(response).decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not verify {release_id} against GitHub") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"GitHub returned a malformed record for {release_id}")
        if payload.get("number") != int(pull_number):
            raise ValueError(f"GitHub pull-request number disagrees for {release_id}")
        if not payload.get("merged_at"):
            raise ValueError(f"{release_id} is not merged")
        if payload.get("merge_commit_sha") != milestone_commit:
            raise ValueError(f"{release_id} milestone commit disagrees with GitHub")


def main() -> int:
    data = resources.files("clanker_lm.web_assets").joinpath("releases.json").read_bytes()
    feed = _load_release_feed(data)
    verify_merged_releases(feed, token=os.environ.get("GITHUB_TOKEN"))
    print(f"verified {len(feed['releases'])} merged web release(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
