"""Deterministic word-level search over executable semantic grammar slots."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from clanker_lm.model import AffectReading, AffectVector, CandidateResponse, GateDecision
from .affinity import AffinityStore, canonical, digest
from .grammar import DecodeError, GRAMMAR_VERSION, ResponsePlan, Word, compose


@dataclass(frozen=True)
class DecoderConfig:
    beam_width: int = 4
    max_nodes: int = 512
    max_tokens: int = 64
    lexical_weight: int = 1
    affect_weight: int = 24
    max_receipt_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        for key, upper in (("beam_width", 32), ("max_nodes", 4096), ("max_tokens", 64),
                           ("lexical_weight", 1000), ("affect_weight", 1000), ("max_receipt_bytes", 4_000_000)):
            value = getattr(self, key)
            lower = 0 if key.endswith("weight") else 1
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{key} must be an integer between {lower} and {upper}")


@dataclass(frozen=True)
class State:
    node: int
    ids: tuple[str, ...]
    words: tuple[str, ...]
    lexical_cost: int
    score: int = 0


@dataclass
class Decoding:
    selected: CandidateResponse
    candidates: list[CandidateResponse]
    receipt: dict
    words: tuple[str, ...]

    def committed_chunks(self):
        """Immediate delivery after full validation, NOT speculative streaming."""
        previous = ""
        for index in range(1, len(self.words) + 1):
            current = compose(self.words[:index])
            yield current[len(previous):]
            previous = current


class LexicalDecoder:
    def __init__(self, pack: AffinityStore, config: DecoderConfig | None = None):
        self.pack = pack
        self.config = config or DecoderConfig()

    def decode(self, plan: ResponsePlan, gates: GateDecision, observed: AffectVector,
               target: AffectVector, *, read: Callable[[str], AffectReading],
               transition: Callable[[AffectVector, AffectVector], AffectVector],
               runtime_hash: str, context_hash: str, backend_name: str) -> Decoding:
        cfg = self.config
        if not plan.slots or len(plan.slots) > cfg.max_tokens:
            raise DecodeError("token_budget: no complete derivation within configured limit")
        legal: list[list[Word]] = []
        narrowing = []
        for position, slot in enumerate(plan.slots):
            choices, decisions = slot.decisions(gates)
            if not choices:
                raise DecodeError(f"no_eligible_token: slot {position} ({slot.role})")
            if len({x.identity for x in choices}) != len(choices):
                raise DecodeError("duplicate word identities in grammatical slot")
            legal.append(choices)
            narrowing.append({"position": position, "role": slot.role, "concept": slot.concept,
                              "considered": len(decisions), "eligible": len(choices), "decisions": decisions})
        receipt = {
            "schema": "clanker-word-receipt-v1", "grammar_version": GRAMMAR_VERSION,
            "runtime_sha256": runtime_hash, "context_sha256": context_hash,
            "backend": backend_name, "plan_sha256": plan.hash,
            "contract_sha256": plan.contract_hash, "pack_sha256": self.pack.pack_hash,
            "source_manifest": dict(self.pack.manifest), "config": asdict(cfg),
            "observed": observed.to_dict(), "target": target.to_dict(),
            "gate_sha256": digest(gates.to_dict()), "act": plan.act, "coverage": plan.coverage,
            "grammar_rules_executed": list(plan.rules), "narrowing": narrowing,
            "nodes": [], "pruning": [],
            "affect_estimator": "lexicographic_legal_completion; estimated, not an admissible reachability bound",
            "search": "bounded beam; no global-optimality claim when pruned",
            "delivery": "complete_validated_response_before_first_chunk",
        }
        prediction_cache: dict[tuple[str, ...], tuple[AffectVector, AffectVector, int]] = {}
        def effect(words: tuple[str, ...]):
            if words not in prediction_cache:
                reading = read(compose(words))
                predicted = transition(observed, reading.vector)
                cost = round(predicted.distance(target) / 255.0 * 1_000_000)
                prediction_cache[words] = (reading.vector, predicted, cost)
            return prediction_cache[words]
        beam = [State(0, (), (), 0)]
        nodes = []
        next_id = 1
        for position, choices in enumerate(legal):
            expanded = []
            suffix = tuple(x[0] for x in legal[position + 1:])
            for parent in beam:
                if len(nodes) + len(choices) > cfg.max_nodes:
                    raise DecodeError("node_budget: incomplete search; no partial response emitted")
                probabilities = [self.pack.probability(parent.words, w.surface) for w in choices]
                remaining_mass = sum(p["probability"] for p in probabilities)
                for word, probability in zip(choices, probabilities):
                    words = parent.words + (word.surface,)
                    ids = parent.ids + (word.identity,)
                    lexical = parent.lexical_cost + probability["cost_micros"]
                    projected = words + tuple(w.surface for w in suffix)
                    vector, predicted, affect_cost = effect(projected)
                    components = {"lexical": cfg.lexical_weight * lexical,
                                  "affect": cfg.affect_weight * affect_cost}
                    score = sum(components.values())
                    node = {
                        "id": next_id, "parent": parent.node, "position": position,
                        "word_id": word.identity, "surface": word.surface,
                        "lexical": probability,
                        "conditional_probability_among_slot_choices": probability["probability"] / remaining_mass,
                        "lexical_prefix_cost_micros": lexical,
                        "lookahead_word_ids": [w.identity for w in suffix],
                        "projected_vector": vector.to_dict(), "predicted_state": predicted.to_dict(),
                        "axis_error": {a: getattr(predicted, a)-getattr(target, a) for a in target.to_dict()},
                        "affect_distance_micros": affect_cost,
                        "components": components, "total_cost": score,
                    }
                    nodes.append(node)
                    expanded.append(State(next_id, ids, words, lexical, score))
                    next_id += 1
            expanded.sort(key=lambda x: (x.score, x.ids))
            receipt["pruning"].append({"position": position,
                "retained": [x.node for x in expanded[:cfg.beam_width]],
                "pruned_search_budget_not_semantic_rejection": [x.node for x in expanded[cfg.beam_width:]]})
            beam = expanded[:cfg.beam_width]
        if not beam:
            raise DecodeError("no_complete_candidate")
        # Validate actual chosen IDs/surfaces against every executed slot; a
        # string containing one bound value is NOT evidence of correctness.
        for state in beam:
            if len(state.ids) != len(legal) or any(
                not any(w.identity == identity and w.surface == surface for w in choices)
                for identity, surface, choices in zip(state.ids, state.words, legal)
            ):
                raise DecodeError("derivation_mismatch")
        final = []
        for state in beam:
            vector, predicted, affect_cost = effect(state.words)
            final.append(CandidateResponse(
                compose(state.words), "lexical:" + digest(state.ids)[:16],
                semantic_valid=True, semantic_reason="all bound slots preserved by typed derivation; not arbitrary-English equivalence",
                affect=vector, predicted_state=predicted, affect_distance=predicted.distance(target),
                score=state.score / 1_000_000, atom_ids=list(state.ids),
                semantic_plan=["ACT:" + plan.act, "PLAN:" + plan.hash, "COVERAGE:" + plan.coverage],
            ))
        winner = beam[0]
        receipt.update({"nodes": nodes, "nodes_used": len(nodes), "prediction_calls": len(prediction_cache),
                        "stop_reason": "complete", "selected_node": winner.node,
                        "selected_word_ids": list(winner.ids), "selected_text": final[0].text,
                        "winning_cost": winner.score,
                        "runner_up_margin": beam[1].score-winner.score if len(beam)>1 else None,
                        "tie_break": "ascending (total integer cost, complete word-ID tuple)",
                        "final_eligible_nodes": [x.node for x in beam]})
        # Normalize tuple/list forms before publication or replay comparisons.
        import json
        receipt = json.loads(canonical(receipt))
        receipt["receipt_sha256"] = digest(receipt)
        if len(canonical(receipt).encode()) > cfg.max_receipt_bytes:
            raise DecodeError("receipt_budget: complete output withheld")
        return Decoding(final[0], final, receipt, winner.words)

    def replay(self, receipt: dict, *args, **kwargs) -> Decoding:
        if not isinstance(receipt, dict):
            raise DecodeError("receipt must be an object")
        supplied = dict(receipt)
        expected_digest = supplied.pop("receipt_sha256", None)
        if digest(supplied) != expected_digest:
            raise DecodeError("receipt content digest mismatch")
        reconstructed = self.decode(*args, **kwargs)
        if reconstructed.receipt != receipt:
            raise DecodeError("receipt does not reproduce with supplied state, runtime, grammar, or pack")
        return reconstructed
