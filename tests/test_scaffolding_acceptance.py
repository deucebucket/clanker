from __future__ import annotations

from clanker_lm import ClankerLM, HeuristicAffectBackend
from clanker_lm.model import AnswerStatus


def runtime():
    return ClankerLM(affect_backend=HeuristicAffectBackend())


def test_pronoun_trap_from_scaffolding_halts_and_probes():
    with runtime() as lm:
        result = lm.process("She pissed me off again.")
        assert result.contract.status == AnswerStatus.MISSING_REFERENCE
        assert "who do you mean" in result.response.lower()
        assert not lm.memory.events


def test_pronoun_trap_resolves_after_female_antecedent_exists():
    with runtime() as lm:
        lm.process("My sister called me.")
        result = lm.process("She pissed me off again.")
        assert result.contract.status == AnswerStatus.ACKNOWLEDGED
        event = lm.memory.events[-1]
        assert event.predicate == "anger"
        assert lm.memory.get_entity(event.arguments["agent"].key).relation == "sister"
        assert event.arguments["time"].key == "again"


def test_yoda_problem_preserves_actor_action_receiver_math():
    with runtime() as normal, runtime() as yoda:
        normal.process("My sister pissed me off.")
        yoda.process("Pissed me off, my sister did.")
        left = normal.memory.events[-1]
        right = yoda.memory.events[-1]
        assert left.predicate == right.predicate == "anger"
        assert normal.memory.get_entity(left.arguments["agent"].key).relation == "sister"
        assert yoda.memory.get_entity(right.arguments["agent"].key).relation == "sister"
        assert left.arguments["patient"].key == right.arguments["patient"].key == "user"


def test_contextual_gating_examples_from_scaffolding():
    with runtime() as low, runtime() as high:
        low_result = low.process("My tummy hurts bruh.")
        high_result = high.process("My mom is really sick.")
        assert low_result.gates.register == "casual"
        assert "formal" in low_result.gates.locked_pools
        assert "high_severity" in low_result.gates.locked_pools
        assert "humor" in high_result.gates.locked_pools
        assert "slang" in high_result.gates.locked_pools


def test_collision_masking_example_uses_casual_serious_pool():
    with runtime() as lm:
        result = lm.process("Bruh, my mom is really sick.")
        assert result.gates.masking
        assert result.candidates[0].construction_id == "compose.acknowledge.masked_serious"
        assert "ACT:SEVERITY_RECOGNITION" in result.candidates[0].semantic_plan
        assert "serious" in result.response.lower()


def test_question_is_completed_by_binding_its_typed_hole():
    with runtime() as lm:
        lm.process("My sister bought a used Honda yesterday.")
        result = lm.process("What did she buy?")
        assert result.contract.status == AnswerStatus.ANSWERED
        assert result.contract.required_slots["requested_role"] == "patient"
        assert result.contract.values[0].surface == "a used Honda"
        assert "honda" in result.response.lower()


def test_a_plus_b_solver_scores_supported_candidates_after_semantic_validation():
    with runtime() as lm:
        lm.process("Sarah bought the car.")
        result = lm.process("Did Sarah buy the car?")
        assert result.contract.status == AnswerStatus.TRUE
        assert result.candidates
        assert result.candidates[0].semantic_valid
        assert result.candidates[0].predicted_state is not None
        assert result.candidates[0].affect_distance >= 0
