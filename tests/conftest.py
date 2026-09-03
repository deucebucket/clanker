from __future__ import annotations

import pytest

from clanker_lm import ClankerLM, HeuristicAffectBackend


@pytest.fixture
def runtime() -> ClankerLM:
    instance = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        yield instance
    finally:
        instance.close()
