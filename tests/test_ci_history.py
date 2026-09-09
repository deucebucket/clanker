"""Regression guard: evaluation ancestry requires the full checkout history."""
from pathlib import Path
import shlex


def test_additive_ci_fetch_does_not_re_shallow_the_full_checkout():
    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/tests.yml").read_text()
    assert "fetch-depth: 0" in workflow
    commands = [shlex.split(line.strip()) for line in workflow.splitlines()
                if line.strip().startswith("git fetch ")]
    assert commands
    for command in commands:
        assert not any(arg.startswith(("--depth", "--shallow", "--deepen")) for arg in command), command
    assert 'test "$(git rev-parse --is-shallow-repository)" = false' in workflow
