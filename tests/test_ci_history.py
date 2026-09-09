"""Keep provenance ancestry visible; reproduce the depth-one fetch regression."""
from __future__ import annotations

from pathlib import Path
import subprocess


def git(path: Path, *args: str, check: bool = True):
    return subprocess.run(["git", "-C", str(path), *args], text=True,
                          capture_output=True, check=check, timeout=15)


def test_depth_one_refetch_hides_parent_but_full_fetch_preserves_it(tmp_path: Path):
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-b", "main")
    git(origin, "config", "user.name", "History fixture")
    git(origin, "config", "user.email", "history@example.invalid")
    git(origin, "config", "commit.gpgsign", "false")
    for version in (1, 2):
        (origin / "item.txt").write_text(str(version), encoding="utf-8")
        git(origin, "add", "item.txt")
        git(origin, "commit", "-m", str(version))
    parent = git(origin, "rev-parse", "HEAD^").stdout.strip()
    for mode in ("shallow", "full"):
        checkout = tmp_path / mode
        git(tmp_path, "clone", origin.as_uri(), str(checkout))
        assert git(checkout, "rev-parse", "HEAD^").stdout.strip() == parent
        depth = ["--depth=1"] if mode == "shallow" else []
        git(checkout, "fetch", "--no-tags", *depth, "origin", "main")
        resolved = git(checkout, "rev-parse", "HEAD^", check=False)
        if mode == "shallow":
            assert resolved.returncode != 0
            assert git(checkout, "rev-parse", "--is-shallow-repository").stdout.strip() == "true"
        else:
            assert resolved.returncode == 0
            assert resolved.stdout.strip() == parent
            assert git(checkout, "rev-parse", "--is-shallow-repository").stdout.strip() == "false"


def test_workflow_preserves_full_history_and_keeps_all_provenance_gates():
    workflow = (Path(__file__).resolve().parent.parent / ".github/workflows/tests.yml").read_text()
    assert "fetch-depth: 0" in workflow
    assert "--depth=" not in workflow and "--depth " not in workflow
    assert "--is-shallow-repository" in workflow
    assert "verify-history --ref HEAD" in workflow
    assert 'verify-additive --base "$additive_base"' in workflow
    assert "python -m pytest -q" in workflow
