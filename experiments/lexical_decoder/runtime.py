"""Opt-in integration with the REAL ClankerLM turn loop, not a mock chatbot.

The legacy realizer seam now returns an unlexicalized frontier. The ranking
seam receives its already computed VADUGWI target and executes word search.
No production file, frozen evaluator, or default web route is modified.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from clanker_lm.affect import AffectController
from clanker_lm.model import AnswerStatus, ParseResult, SpeechAct
from clanker_lm.parser import SemanticParser
from clanker_lm.runtime import ClankerLM
from .affinity import AffinityStore, canonical, digest, tokens
from .decoder import DecoderConfig, Decoding, LexicalDecoder
from .grammar import DecodeError, Grammar, ResponsePlan


def software_hash() -> str:
    root = Path(__file__).resolve().parents[2]
    paths = []
    for folder in (root / "engine", root / "clanker_lm", Path(__file__).parent):
        paths.extend(p for p in folder.rglob("*") if p.is_file() and p.suffix in {".py", ".json"} and "__pycache__" not in p.parts)
    h = hashlib.sha256()
    for p in sorted(paths):
        name = str(p.relative_to(root)).encode()
        content = p.read_bytes()
        h.update(len(name).to_bytes(4, "big")); h.update(name)
        h.update(len(content).to_bytes(8, "big")); h.update(content)
    return h.hexdigest()


def development_pack() -> AffinityStore:
    payload = json.loads(Path(__file__).with_name("development_counts.json").read_text())
    return AffinityStore.from_payload(payload)


class DialogueParser(SemanticParser):
    """Two closed social acts; all content parsing delegates to the base parser."""
    convention: str | None = None

    def parse(self, text, memory):
        words = tuple(w for w in tokens(text) if w not in {".", "!", "?", ","})
        self.convention = None
        if words in {("thanks",), ("thank", "you")}:
            self.convention = "gratitude"
        elif words in {("bye",), ("goodbye",)}:
            self.convention = "closure"
        if self.convention:
            return ParseResult(SpeechAct.SOCIAL, text, diagnostics=["social_convention:" + self.convention])
        return super().parse(text, memory)


@dataclass(frozen=True)
class Frontier:
    plan: ResponsePlan
    gates: object
    context_hash: str


class PlanningRealizer:
    def __init__(self, runtime: ClankerLM):
        self.runtime = runtime

    def realize(self, contract, gates) -> Frontier:
        grammar = Grammar(self.runtime.memory, self.runtime.store)
        plan = grammar.plan(contract, gates, convention=self.runtime.parser.convention)
        if plan.coverage.startswith("declined"):
            contract.required_slots["decoder_original_status"] = contract.status.value
            contract.status = AnswerStatus.UNSUPPORTED
            contract.reason = plan.coverage
            contract.response_goal = "clarify"
            plan = replace(plan, contract_hash=digest(contract.to_dict()))
        senses = [{"normalized": t["normalized"], "senses": t["senses"]}
                  for t in self.runtime.learned_lexicon()]
        context = {"memory": self.runtime.memory.to_dict(), "senses": sorted(senses, key=lambda x: x["normalized"])}
        return Frontier(plan, gates, digest(context))


class FrontierController(AffectController):
    def __init__(self, runtime: ClankerLM, decoder: LexicalDecoder, code_hash: str):
        super().__init__(runtime.affect.backend)
        self.runtime = runtime
        self.decoder = decoder
        self.code_hash = code_hash
        self.last: Decoding | None = None
        self.replay_context: tuple | None = None

    def read_candidate(self, text):
        # Candidate text is read through the same learned lexical overlay as
        # input text; evaluating a candidate does not add learning evidence.
        return self.runtime.learner.apply_overlay(text, self.backend.analyze(text))

    def rank_candidates(self, candidates, observed, target):
        if not isinstance(candidates, Frontier):
            eligible = [c for c in candidates if c.semantic_valid]
            if not eligible:
                raise DecodeError("no_valid_candidate")
            return super().rank_candidates(eligible, observed, target)
        args = (candidates.plan, candidates.gates, observed, target)
        kw = {"read": self.read_candidate, "transition": self.backend.transition,
              "runtime_hash": self.code_hash, "context_hash": candidates.context_hash,
              "backend_name": self.backend.name}
        result = self.decoder.decode(*args, **kw)
        self.replay_context = (args, kw)
        self.last = result
        return result.selected, result.candidates


class ReceiptChat:
    """Own one isolated mutable conversation and one immutable lexical pack.

    The new path is explicitly experimental. Receipts may contain bound user
    words, so they are session-local; no chat or generated output is trained on.
    """

    def __init__(self, *, pack: AffinityStore | None = None, config: DecoderConfig | None = None,
                 affect_backend=None, clock: Callable | None = None):
        self._owns_pack = pack is None
        self.pack = pack or development_pack()
        self.config = config or DecoderConfig()
        self.code_hash = software_hash()
        self.clock = clock
        self.runtime = ClankerLM(affect_backend=affect_backend, clock=clock)
        self._has_response = False
        self._attach()

    def _attach(self):
        self.controller = FrontierController(self.runtime, LexicalDecoder(self.pack, self.config), self.code_hash)
        self.runtime.affect = self.controller
        self.runtime.realizer = PlanningRealizer(self.runtime)
        self.runtime.parser = DialogueParser()

    def process(self, text: str):
        if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 4096:
            raise DecodeError("input must be nonempty text of at most 4096 bytes")
        if self.runtime.memory.turn_index >= 200:
            raise DecodeError("session turn budget exceeded; start a new session")
        before = self.runtime.to_dict()
        before_flag = self._has_response
        backend = self.runtime.affect.backend
        try:
            # Base v6 omits last_result from snapshots. Carry the persisted
            # post-response prior explicitly rather than restarting from the
            # pre-response observation on the first resumed turn.
            if self._has_response and self.runtime.last_result is None:
                self.runtime.observed_state = self.runtime.predicted_state
            self.runtime.parser.convention = None
            result = self.runtime.process(text)
            if self.controller.last is None:
                raise DecodeError("decoder integration did not execute")
            self._has_response = True
            return result
        except Exception:
            # Database helper methods commit independently. Restore BOTH store
            # and symbolic memory from the pre-turn checkpoint on a failed turn.
            self.runtime.close()
            self.runtime = ClankerLM.from_dict(before, affect_backend=backend, clock=self.clock)
            self._has_response = before_flag
            self._attach()
            raise

    @property
    def receipt(self) -> dict:
        if self.controller.last is None:
            raise DecodeError("no successful decoded turn")
        return json.loads(canonical(self.controller.last.receipt))

    def replay_last(self) -> bool:
        if self.controller.last is None or self.controller.replay_context is None:
            raise DecodeError("no successful decoded turn")
        args, kw = self.controller.replay_context
        self.controller.decoder.replay(self.receipt, *args, **kw)
        return True

    def committed_chunks(self):
        if self.controller.last is None:
            raise DecodeError("no successful decoded turn")
        return self.controller.last.committed_chunks()

    def snapshot(self) -> dict:
        return {"schema": "lexical-chat-v1", "runtime": self.runtime.to_dict(),
                "pack_sha256": self.pack.pack_hash, "config": self.config.__dict__,
                "has_response": self._has_response, "runtime_sha256": self.code_hash}

    def restore(self, data: dict) -> None:
        if set(data) != {"schema", "runtime", "pack_sha256", "config", "has_response", "runtime_sha256"}:
            raise DecodeError("invalid chat snapshot fields")
        if data["schema"] != "lexical-chat-v1" or data["pack_sha256"] != self.pack.pack_hash or data["runtime_sha256"] != self.code_hash:
            raise DecodeError("snapshot pack/runtime/schema mismatch")
        if data["config"] != self.config.__dict__ or type(data["has_response"]) is not bool:
            raise DecodeError("snapshot decoding policy mismatch")
        new = ClankerLM.from_dict(data["runtime"], affect_backend=self.runtime.affect.backend, clock=self.clock)
        self.runtime.close()
        self.runtime = new
        self._has_response = data["has_response"]
        self._attach()

    def close(self):
        self.runtime.close()
        if self._owns_pack:
            self.pack.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
