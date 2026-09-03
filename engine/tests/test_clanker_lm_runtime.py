"""End-to-end deterministic dialogue tests using the dependency-free adapter."""

from clanker_lm import AnswerStatus, ClankerLM, RuleAffectAdapter


def _bot(**kwargs):
    return ClankerLM(affect=RuleAffectAdapter(), **kwargs)


def test_conversational_fact_can_be_queried_from_multiple_open_slots():
    bot = _bot()
    assert bot.reply("My sister bought a used Honda yesterday.") == "Got it."

    who = bot.process("Who bought the Honda?")
    what = bot.process("What did she buy?")
    when = bot.process("When did she buy it?")

    assert who.answer and who.answer.status == AnswerStatus.ANSWERED
    assert who.response == "Your sister bought it yesterday."
    assert what.answer and what.answer.status == AnswerStatus.ANSWERED
    assert what.response == "She bought a used Honda yesterday."
    assert when.answer and when.answer.status == AnswerStatus.ANSWERED
    assert when.response == "She bought it yesterday."


def test_fact_in_first_sentence_is_available_to_question_in_same_turn():
    bot = _bot()
    result = bot.process("My sister bought a red bike. What did she buy?")

    assert result.answer and result.answer.status == AnswerStatus.ANSWERED
    assert result.response == "She bought a red bike."
    assert result.stored_fact_ids


def test_missing_why_role_is_reported_without_invention():
    bot = _bot()
    bot.reply("My sister bought a used Honda yesterday.")
    result = bot.process("Why did she buy it?")

    assert result.answer and result.answer.status == AnswerStatus.PARTIAL_UNKNOWN
    assert "but not why" in result.response.lower()
    assert "because" not in result.response.lower()


def test_later_causal_fact_binds_the_why_slot():
    bot = _bot()
    bot.reply("My sister bought a used Honda yesterday.")
    bot.reply("She bought it because her old car broke down.")
    result = bot.process("Why did she buy it?")

    assert result.answer and result.answer.status == AnswerStatus.ANSWERED
    assert result.response == "She bought it because her old car broke down."


def test_unknown_how_does_not_repurpose_known_cause_as_method():
    bot = _bot()
    bot.reply("My sister bought a used Honda because her old car broke down.")
    result = bot.process("How did she buy it?")

    assert result.answer and result.answer.status == AnswerStatus.PARTIAL_UNKNOWN
    assert "but not how" in result.response.lower()


def test_polar_answers_require_explicit_entailment_or_contradiction():
    bot = _bot()
    bot.reply("My sister bought a used Honda yesterday.")

    true_result = bot.process("Did my sister buy it?")
    unknown_result = bot.process("Did my mother buy it?")
    bot.reply("My mother did not buy the Honda.")
    false_result = bot.process("Did my mother buy the Honda?")

    assert true_result.answer and true_result.answer.status == AnswerStatus.TRUE
    assert true_result.response.startswith("Yes.")
    assert unknown_result.answer and unknown_result.answer.status == AnswerStatus.UNKNOWN
    assert unknown_result.response.startswith("I don't")
    assert false_result.answer and false_result.answer.status == AnswerStatus.FALSE
    assert false_result.response == "No. She did not buy it."


def test_negative_polar_question_is_answered_against_negative_proposition():
    bot = _bot()
    bot.reply("My sister bought a used Honda yesterday.")
    result = bot.process("Did my sister not buy the Honda?")

    assert result.answer and result.answer.status == AnswerStatus.FALSE
    assert result.response.startswith("No.")
    assert "bought" in result.response


def test_pronoun_trap_halts_generation_and_requests_context():
    result = _bot().process("She pissed me off again.")

    assert result.answer is None
    assert result.parsed.unresolved_references
    assert result.response == "Who do you mean by she?"
    assert result.stored_fact_ids == ()


def test_ambiguous_pronoun_names_the_competing_antecedents():
    bot = _bot()
    bot.reply("My sister saw my mother.")
    result = bot.process("She left.")

    assert result.response == "Do you mean your sister or your mother?"
    assert not result.stored_fact_ids


def test_contextual_gating_handles_conflict_collision_masking_and_low_severity_pain():
    conflict = _bot().process("My sister pissed me off again.")
    masking = _bot().process("Bruh, my mom is really sick.")
    pain = _bot().process("My tummy hurts bruh.")

    assert conflict.response == "That sounds frustrating. What happened?"
    assert conflict.gates.collision_masking is False

    assert masking.gates.severity == "high"
    assert masking.gates.collision_masking is True
    assert {"humor", "playful", "minimization"}.issubset(masking.gates.locked_pools)
    assert masking.response == "That sounds serious. What happened?"

    assert pain.gates.severity == "low"
    assert pain.gates.register == "casual"
    assert "formal" in pain.gates.locked_pools
    assert pain.response == "That sucks. How bad does it hurt?"


def test_repeated_runs_are_deterministic():
    transcript = (
        "My sister bought a red bike yesterday.",
        "Who bought it?",
        "What did she buy?",
        "Why did she buy it?",
    )

    first = _bot()
    second = _bot()
    first_responses = [first.reply(item) for item in transcript]
    second_responses = [second.reply(item) for item in transcript]

    assert first_responses == second_responses


def test_quantity_and_double_object_transfer_questions_are_realized_grammatically():
    bot = _bot()
    bot.reply("Sarah bought three books.")
    quantity = bot.process("How many books did Sarah buy?")
    assert quantity.answer and quantity.answer.status == AnswerStatus.ANSWERED
    assert quantity.response == "Sarah bought three books."

    bot.reply("Sarah gave John the book.")
    recipient = bot.process("Who did Sarah give the book to?")
    theme = bot.process("What did Sarah give John?")
    agent = bot.process("Who gave John the book?")

    assert recipient.response == "Sarah gave it to John."
    assert theme.response == "Sarah gave the book to John."
    assert agent.response == "Sarah gave it to John."

def test_open_slot_queries_preserve_polarity_tense_and_modality():
    negative = _bot()
    negative.reply("Sarah did not buy a Honda.")

    positive_wh = negative.process("What did Sarah buy?")
    negative_wh = negative.process("What did Sarah not buy?")

    assert positive_wh.answer and positive_wh.answer.status == AnswerStatus.UNKNOWN
    assert negative_wh.answer and negative_wh.answer.status == AnswerStatus.ANSWERED
    assert negative_wh.response == "Sarah did not buy a Honda."

    temporal = _bot()
    temporal.reply("Sarah bought a Honda.")
    future = temporal.process("What will Sarah buy?")
    modal = temporal.process("What might Sarah buy?")

    assert future.answer and future.answer.status == AnswerStatus.UNKNOWN
    assert modal.answer and modal.answer.status == AnswerStatus.UNKNOWN


def test_normative_because_clause_binds_why_justification_slot():
    bot = _bot()
    bot.reply("I should apologize because my comment was hurtful.")
    result = bot.process("Why should I apologize?")

    assert result.answer and result.answer.status == AnswerStatus.ANSWERED
    assert result.answer.bound_role.value == "justification"
    assert result.answer.bound_value.display == "my comment was hurtful"
    assert "because my comment was hurtful" in result.response.lower()

