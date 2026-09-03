"""Parser-level tests for the deterministic Clanker-LM semantic IR."""

from clanker_lm.memory import ConversationMemory
from clanker_lm.models import (
    HowType,
    QuestionFamily,
    SemanticRole,
    SpeechAct,
    WhyType,
)
from clanker_lm.parser import SemanticParser


def _parser():
    memory = ConversationMemory(session_id="parser-tests")
    memory.begin_turn()
    return memory, SemanticParser(memory)


def test_statement_extracts_predicate_arguments_and_time():
    _, parser = _parser()
    parsed = parser.parse("My sister bought a used Honda yesterday.")

    assert parsed.speech_act == SpeechAct.STATEMENT
    assert parsed.frame is not None
    assert parsed.frame.predicate == "buy"
    assert parsed.frame.tense == "past"
    assert parsed.frame.role(SemanticRole.AGENT).display == "sister"
    assert parsed.frame.role(SemanticRole.THEME).display == "used Honda"
    assert parsed.frame.role(SemanticRole.TIME).display == "yesterday"


def test_yoda_order_maps_to_same_force_direction():
    _, parser = _parser()
    normal = parser.parse("My sister pissed me off again.")
    inverted = parser.parse("Pissed me off, my sister did.")

    assert normal.frame is not None and inverted.frame is not None
    assert normal.frame.predicate == inverted.frame.predicate == "piss_off"
    assert normal.frame.role(SemanticRole.AGENT).value == inverted.frame.role(SemanticRole.AGENT).value
    assert normal.frame.role(SemanticRole.PATIENT).value == inverted.frame.role(SemanticRole.PATIENT).value
    assert normal.frame.repeated is True


def test_unbound_gendered_pronoun_creates_explicit_probe_state():
    _, parser = _parser()
    parsed = parser.parse("She pissed me off again.")

    assert parsed.unresolved_references
    unresolved = parsed.unresolved_references[0]
    assert unresolved.surface == "she"
    assert unresolved.compatible_entity_ids == ()
    assert "no compatible antecedent" in unresolved.reason


def test_ambiguous_pronoun_retains_all_compatible_antecedents():
    memory, parser = _parser()
    parser.parse("My sister saw my mother.")
    parsed = parser.parse("She left.")

    assert parsed.unresolved_references
    unresolved = parsed.unresolved_references[0]
    assert len(unresolved.compatible_entity_ids) == 2
    names = {memory.entities[item].relation_to_user for item in unresolved.compatible_entity_ids}
    assert names == {"sister", "mother"}


def test_object_wh_question_opens_theme_slot():
    _, parser = _parser()
    parsed = parser.parse("What did Sarah buy?")

    assert parsed.question is not None
    assert parsed.question.family == QuestionFamily.WH
    assert parsed.question.requested_roles == (SemanticRole.THEME,)
    assert parsed.question.frame.predicate == "buy"
    assert parsed.question.frame.role(SemanticRole.AGENT).display == "Sarah"


def test_subject_wh_question_opens_agent_slot():
    _, parser = _parser()
    parsed = parser.parse("Who bought the Honda?")

    assert parsed.question is not None
    assert parsed.question.requested_roles == (SemanticRole.AGENT,)
    assert parsed.question.frame.predicate == "buy"
    assert parsed.question.frame.role(SemanticRole.THEME).display == "Honda"


def test_transfer_recipient_question_opens_recipient_slot():
    _, parser = _parser()
    parsed = parser.parse("Who did Sarah give the book to?")

    assert parsed.question is not None
    assert parsed.question.requested_roles == (SemanticRole.RECIPIENT,)
    assert parsed.question.frame.predicate == "give"
    assert parsed.question.frame.role(SemanticRole.THEME).display == "book"


def test_why_router_distinguishes_cause_motive_justification_and_evidence():
    _, parser = _parser()

    cause = parser.parse("Why did the window break?").question
    motive = parser.parse("Why did Sarah leave?").question
    justification = parser.parse("Why should I apologize?").question
    evidence = parser.parse("Why do you think that?").question

    assert cause and cause.why_type == WhyType.CAUSE
    assert cause.requested_roles == (SemanticRole.CAUSE,)
    assert motive and motive.why_type == WhyType.MOTIVE
    assert motive.requested_roles[:2] == (SemanticRole.MOTIVE, SemanticRole.PURPOSE)
    assert justification and justification.why_type == WhyType.JUSTIFICATION
    assert justification.requested_roles == (SemanticRole.JUSTIFICATION,)
    assert evidence and evidence.why_type == WhyType.EVIDENCE
    assert evidence.requested_roles[0] == SemanticRole.EVIDENCE


def test_how_router_distinguishes_method_process_degree_quantity_and_social():
    _, parser = _parser()

    method = parser.parse("How did Sarah enter?")
    process = parser.parse("How does the engine calculate valence?")
    degree = parser.parse("How tall is Sarah?")
    quantity = parser.parse("How many books did Sarah buy?")
    social = parser.parse("How are you?")

    assert method.question and method.question.how_type == HowType.METHOD
    assert method.question.requested_roles[:2] == (SemanticRole.METHOD, SemanticRole.MANNER)
    assert process.question and process.question.how_type == HowType.PROCESS
    assert SemanticRole.PROCESS in process.question.requested_roles
    assert degree.question and degree.question.how_type == HowType.DEGREE
    assert degree.question.requested_roles == (SemanticRole.DEGREE,)
    assert quantity.question and quantity.question.how_type == HowType.QUANTITY
    assert quantity.question.requested_roles == (SemanticRole.QUANTITY,)
    assert quantity.question.frame.predicate == "buy"
    assert quantity.question.frame.role(SemanticRole.THEME).display == "books"
    assert social.speech_act == SpeechAct.SOCIAL_CHECKIN
    assert social.question and social.question.family == QuestionFamily.SOCIAL


def test_negative_polar_question_preserves_requested_polarity():
    _, parser = _parser()
    parsed = parser.parse("Did my sister not buy the Honda?")

    assert parsed.question is not None
    assert parsed.question.family == QuestionFamily.POLAR
    assert parsed.question.frame.polarity is False
    assert parsed.question.frame.predicate == "buy"



def test_double_object_transfer_statement_and_subject_question_share_roles():
    _, parser = _parser()
    statement = parser.parse("Sarah gave John the book.")
    question = parser.parse("Who gave John the book?")

    assert statement.frame is not None and question.question is not None
    assert statement.frame.role(SemanticRole.RECIPIENT).display == "John"
    assert statement.frame.role(SemanticRole.THEME).display == "book"
    assert question.question.requested_roles == (SemanticRole.AGENT,)
    assert question.question.frame.role(SemanticRole.RECIPIENT).display == "John"
    assert question.question.frame.role(SemanticRole.THEME).display == "the book"
    assert question.question.frame.tense == "past"


def test_statement_splits_numeric_quantifier_from_object():
    _, parser = _parser()
    parsed = parser.parse("Sarah bought three books.")

    assert parsed.frame is not None
    assert parsed.frame.role(SemanticRole.QUANTITY).display == "three"
    assert parsed.frame.role(SemanticRole.THEME).display == "books"

def test_bare_past_copula_is_not_misexpanded_as_we_are():
    _, parser = _parser()
    parsed = parser.parse("Were they angry?")

    assert parsed.question is not None
    assert parsed.question.family == QuestionFamily.POLAR
    assert parsed.question.frame.predicate == "be"
    assert parsed.question.frame.tense == "past"
