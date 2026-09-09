"""Executable typed clause grammar for a deliberately bounded response slice.

Logical response acts select propositions, not ordered response sentences.
Clause operations determine order/auxiliaries/agreement; lexical slots retain
alternatives until the decoder executes. Complex facts fail coverage closed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable

from clanker_lm import lexicon
from clanker_lm.database import LanguageStore
from clanker_lm.memory import ConversationMemory
from clanker_lm.model import AnswerContract, AnswerStatus, GateDecision, RefKind
from clanker_lm.realize import SurfaceRealizer
from .affinity import TOKEN, digest, tokens

GRAMMAR_VERSION = "typed-clauses-v1"


class DecodeError(ValueError):
    """No supported complete, eligible derivation; never a partial reply."""


@dataclass(frozen=True)
class Word:
    identity: str
    surface: str
    concept: str
    pools: tuple[str, ...] = ()
    register: str = "any"


@dataclass(frozen=True)
class Slot:
    role: str
    concept: str
    options: tuple[Word, ...]

    def decisions(self, gates: GateDecision) -> tuple[list[Word], list[dict]]:
        eligible, decisions = [], []
        for word in sorted(self.options, key=lambda x: x.identity):
            reasons = []
            if word.concept != self.concept:
                reasons.append("semantic_concept")
            if set(word.pools) & set(gates.locked_pools):
                reasons.append("locked_pool")
            if gates.allowed_pools and word.pools and not set(word.pools) & set(gates.allowed_pools):
                reasons.append("pool_not_enabled")
            if word.register not in {"any", "neutral", gates.register}:
                reasons.append("register")
            if tokens(word.surface) != (word.surface.lower(),):
                reasons.append("not_one_token")
            decisions.append({"word_id": word.identity, "surface": word.surface,
                              "eligible": not reasons, "rejected_by": reasons})
            if not reasons:
                eligible.append(word)
        return eligible, decisions


@dataclass(frozen=True)
class Clause:
    subject: tuple[Slot, ...]
    predicates: tuple[Word, ...]
    predicate_concept: str
    complements: tuple[Slot, ...] = ()
    tense: str = "present"
    polarity: bool = True
    mode: str = "declarative"
    agreement: str = "third"
    modality: str | None = None


@dataclass(frozen=True)
class ResponsePlan:
    act: str
    contract_hash: str
    slots: tuple[Slot, ...]
    rules: tuple[str, ...]
    semantic_binding: dict
    coverage: str = "supported"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def hash(self) -> str:
        return digest(self.to_dict())


def fixed(surface: str, role: str) -> tuple[Slot, ...]:
    seq = tuple(TOKEN.findall(surface))
    if not seq or len(seq) > 24:
        raise DecodeError("bound reference is empty or exceeds token budget")
    return tuple(Slot(f"{role}:{i}", f"bound:{role}:{i}",
                      (Word(f"bound:{role}:{i}:{digest(word)[:12]}", word,
                            f"bound:{role}:{i}"),)) for i, word in enumerate(seq))


def atom(surface: str, role: str) -> Slot:
    result = fixed(surface, role)
    if len(result) != 1:
        raise DecodeError("grammar terminals must be one token")
    return result[0]


def predicate(lemma: str, concept: str | None = None) -> Word:
    if tokens(lemma) != (lemma.lower(),):
        raise DecodeError("predicate must be one token")
    return Word("predicate:" + lemma, lemma, concept or lemma)


class Grammar:
    def __init__(self, memory: ConversationMemory, store: LanguageStore):
        self.memory = memory
        self.store = store
        self.np = SurfaceRealizer(memory, store)

    def category(self, category: str, concept: str, *, evaluation: bool = False) -> Slot:
        options = []
        for a in self.store.atom_candidates(category):
            meaning = a.features.get("valence", concept) if evaluation else concept
            pools = ("casual", "slang") if a.register == "casual" else ("neutral",)
            if a.surface == "serious":
                pools = ("high_severity",)
                meaning = "serious"
            options.append(Word(a.atom_id, a.lemma, meaning, pools, a.register))
        # Supplemental single-word development lexemes; no learned/global claims.
        if category == "evaluation_state":
            options.extend([
                Word("dev.difficult", "difficult", "negative"),
                Word("dev.frustrating", "frustrating", "negative"),
                Word("dev.encouraging", "encouraging", "positive", ("neutral",)),
                Word("dev.serious_difficult", "difficult", "serious", ("casual_serious",)),
            ])
        if not options:
            raise DecodeError("missing lexical category: " + category)
        return Slot(category, concept, tuple(options))

    def clause(self, c: Clause) -> tuple[Slot, ...]:
        if c.tense not in {"past", "present"} or c.mode not in {"declarative", "polar", "wh_subject", "wh_object"}:
            raise DecodeError("unsupported grammatical mode")
        if not c.predicates or any(w.concept != c.predicate_concept for w in c.predicates):
            raise DecodeError("predicate alternatives must preserve the selected sense")
        is_be = all(w.surface == "be" for w in c.predicates)
        if any(w.surface == "be" for w in c.predicates) and not is_be:
            raise DecodeError("copular and lexical-verb alternatives cannot share a slot")
        question = c.mode in {"polar", "wh_object"}
        prefix = (atom("what", "wh:object"),) if c.mode == "wh_object" else ()
        if c.modality:
            if c.modality not in {"may", "might", "can", "could", "will", "would", "should", "must"}:
                raise DecodeError("unsupported modal")
            modal = atom(c.modality, "modality")
            body = ((modal,) + c.subject) if question else (c.subject + (modal,))
            if not c.polarity:
                body += (atom("not", "polarity"),)
            body += (Slot("finite:predicate", c.predicate_concept, c.predicates),)
        elif is_be:
            form = ("was" if c.agreement != "plural" else "were") if c.tense == "past" else {
                "first": "am", "plural": "are", "third": "is"}[c.agreement]
            finite = atom(form, "finite:be")
            body = ((finite,) + c.subject) if question else (c.subject + (finite,))
            if not c.polarity:
                body += (atom("not", "polarity"),)
        else:
            needs_aux = question or not c.polarity
            options = []
            for w in c.predicates:
                surface = w.surface if needs_aux else (
                    lexicon.past_form(w.surface) if c.tense == "past" else
                    lexicon.present_form(w.surface, third_person_singular=c.agreement == "third"))
                options.append(replace(w, surface=surface, identity=w.identity + ":" + surface))
            verb = Slot("finite:predicate", c.predicate_concept, tuple(options))
            if needs_aux:
                auxiliary = "did" if c.tense == "past" else ("does" if c.agreement == "third" else "do")
                aux = atom(auxiliary, "finite:aux")
                body = ((aux,) + c.subject) if question else (c.subject + (aux,))
                if not c.polarity:
                    body += (atom("not", "polarity"),)
                body += (verb,)
            else:
                body = c.subject + (verb,)
        return prefix + body + c.complements + (atom("?" if c.mode != "declarative" else ".", "punctuation"),)

    def ref(self, ref, role: str, *, subject: bool = False) -> tuple[Slot, ...]:
        if ref is None or ref.kind in {RefKind.EVENT, RefKind.VARIABLE}:
            raise DecodeError("event references and open variables require a later grammar slice")
        if ref.kind == RefKind.ENTITY and self.memory.get_entity(ref.key) is None:
            raise DecodeError("unresolved bound entity")
        surface = self.np.render_ref(ref, case="subject" if subject else "object", definite=False)
        if any(w in tokens(surface) for w in {".", "?", "!", ";", '"', "<", ">"}):
            raise DecodeError("bound phrase exceeds the declarative reference grammar")
        return fixed(surface, role)

    @staticmethod
    def agreement(subject: tuple[Slot, ...]) -> str:
        values = [s.options[0].surface.lower() for s in subject]
        if values == ["i"]:
            return "first"
        if values in (["you"], ["we"], ["they"]):
            return "plural"
        return "third"

    def event_clause(self, contract: AnswerContract) -> Clause:
        event = contract.proposition
        if event is None or event.aspect != "simple" or event.modality or event.discourse_role != "main":
            raise DecodeError("qualified/aspectual content requires a later grammar slice")
        if any(k in contract.required_slots for k in ("attributed", "gerund", "infinitival", "embedded_interrogative")):
            raise DecodeError("attribution must not be flattened")
        if contract.status == AnswerStatus.TRUE and not event.polarity:
            raise DecodeError("affirmation contradicts proposition polarity")
        if contract.status == AnswerStatus.FALSE and event.polarity:
            raise DecodeError("denial contradicts proposition polarity")
        tentative_definition = contract.source.value == "inferred" and event.predicate == "mean"
        if contract.source.value in {"attributed", "unknown"} or (contract.source.value == "inferred" and not tentative_definition):
            raise DecodeError("source qualification is not supported by this grammar slice")
        args = event.arguments
        allowed = {"agent", "subject", "patient", "value", "recipient", "location", "destination", "time", "instrument", "repeated"}
        if set(args) - allowed:
            raise DecodeError("unrealized semantic roles: " + ",".join(sorted(set(args) - allowed)))
        if "agent" in args and "subject" in args:
            raise DecodeError("ambiguous subject role")
        subject_ref = args.get("agent") or args.get("subject")
        subject = self.ref(subject_ref, "subject", subject=True)
        complements: tuple[Slot, ...] = ()
        for role in ("patient", "value", "recipient", "destination", "location", "instrument", "time", "repeated"):
            if role not in args:
                continue
            ref = args[role]
            content = self.ref(ref, role)
            marker = {"recipient": "to", "destination": "to", "instrument": "with"}.get(role)
            if role == "location":
                # Never guess a spatial relation not present in the binding.
                first = content[0].options[0].surface
                if first not in {"at", "in", "on", "to", "from", "near"}:
                    raise DecodeError("location lacks a bound preposition")
            if marker:
                complements += (atom(marker, "role-marker:" + role),)
            complements += content
        agreement = self.agreement(subject)
        if subject_ref.kind == RefKind.ENTITY:
            entity = self.memory.get_entity(subject_ref.key)
            if entity and entity.number.value == "plural":
                agreement = "plural"
        present = {(r.kind, r.key) for r in args.values()}
        if any((r.kind, r.key) not in present for r in contract.values):
            raise DecodeError("bound answer value is absent from proposition")
        return Clause(subject, (predicate(event.predicate),), event.predicate, complements,
                      event.tense, event.polarity, agreement=agreement, modality="may" if tentative_definition else None)

    def plan(self, contract: AnswerContract, gates: GateDecision, *, convention: str | None = None) -> ResponsePlan:
        status = contract.status
        rules: list[str] = []
        clauses: list[Clause] = []
        direct: tuple[Slot, ...] = ()
        coverage = "supported"
        binding = {"status": status.value, "source": contract.source.value,
                   "certainty": contract.certainty, "forbidden_claims": sorted(contract.forbidden_claims)}
        act = gates.response_act
        if status in {AnswerStatus.ANSWERED, AnswerStatus.TRUE, AnswerStatus.FALSE, AnswerStatus.LEXICAL_LEARNED} and contract.proposition is not None:
            try:
                clauses.append(self.event_clause(contract))
                binding["event"] = contract.proposition.to_dict()
                binding["values"] = [x.to_dict() for x in contract.values]
                if status in {AnswerStatus.TRUE, AnswerStatus.FALSE}:
                    category = "affirmation" if status == AnswerStatus.TRUE else "denial"
                    direct = (self.category(category, category), atom(".", "particle:punctuation"))
                act = "answer"
            except DecodeError as exc:
                coverage = "declined:" + str(exc)
        elif status == AnswerStatus.UNKNOWN:
            clauses.append(Clause(fixed("I", "speaker"), (predicate("know"),), "know",
                                  fixed("the answer", "knowledge-object"), polarity=False, agreement="first"))
            act = "uncertainty"
        elif status == AnswerStatus.CONFLICT:
            clauses.append(Clause(fixed("information", "conflict-subject"), (predicate("conflict"),), "conflict"))
            act = "conflict_disclosure"
        elif status == AnswerStatus.LEXICAL_PROBE:
            term = contract.required_slots.get("term", "")
            if not term or len(tokens(term)) != 1:
                raise DecodeError("lexical probe requires one explicit term")
            clauses.append(Clause(fixed(term, "unknown-word"), (predicate("mean"),), "mean", mode="wh_object"))
            act = "lexical_probe"
        elif status in {AnswerStatus.MISSING_REFERENCE, AnswerStatus.AMBIGUOUS_REFERENCE, AnswerStatus.MULTIPLE_MATCHES}:
            clauses.append(Clause(fixed("you", "addressee"), (predicate("mean"),), "mean", mode="wh_object", agreement="plural"))
            act = "reference_probe"
        elif status == AnswerStatus.ACKNOWLEDGED:
            if act == "social":
                binding["social_convention"] = convention or "greeting"
                if convention == "gratitude":
                    clauses.append(Clause(fixed("you", "addressee"), (predicate("be"),), "be",
                                          (Slot("social_complement", "welcome", (Word("social.welcome", "welcome", "welcome"),)),), agreement="plural"))
                elif convention == "closure":
                    direct = (Slot("farewell", "farewell", (Word("social.goodbye", "goodbye", "farewell"),
                               Word("social.bye", "bye", "farewell", ("casual",), "casual"))), atom(".", "farewell:punctuation"))
                else:
                    direct = (self.category("greeting", "greeting"), atom(".", "greeting:punctuation"))
            elif act == "neutral_acknowledge":
                verb = self.category("acknowledge_verb", "acknowledge")
                # GET and HEAR have different conventional objects; only the
                # explicitly compatible transitive senses share this clause.
                verbs = tuple(w for w in verb.options if w.surface in {"hear", "understand"})
                clauses.append(Clause(fixed("I", "speaker"), verbs, "acknowledge", fixed("you", "addressee"), agreement="first"))
            elif act in {"empathic_followup", "serious_followup", "positive_acknowledge"}:
                polarity = "positive" if act == "positive_acknowledge" else "serious" if act == "serious_followup" else "negative"
                adjective = self.category("evaluation_state", polarity, evaluation=True)
                clauses.append(Clause(fixed("that", "disclosed-content"), (predicate("sound"),), "sound", (adjective,)))
                if act.endswith("followup"):
                    clauses.append(Clause(fixed("what", "wh:subject"), (predicate("happen"),), "happen", tense="past", mode="wh_subject"))
            elif act == "empathic_acknowledge":
                clauses.append(Clause(fixed("I", "speaker"), (predicate("be"),), "be", (self.category("empathy_state", "sympathy"),), agreement="first"))
            elif act == "safety_probe":
                clauses.append(Clause(fixed("you", "addressee"), (predicate("be"),), "be", (self.category("safety_state", "safety"),), mode="polar", agreement="plural"))
            else:
                coverage = "declined:unimplemented response act"
        else:
            coverage = "declined:unsupported answer status"
        if coverage.startswith("declined"):
            # An explicitly recorded coverage decline is not a successful answer.
            direct = ()
            clauses = [Clause(fixed("you", "addressee"), (predicate("mean"),), "mean", mode="wh_object", agreement="plural")]
            act = "coverage_probe"
            binding["unsupported_contract_sha256"] = digest(contract.to_dict())
        count = len(clauses) + (1 if direct else 0)
        if count > gates.max_sentences:
            raise DecodeError("required discourse acts exceed sentence budget")
        slots = direct
        if direct:
            rules.append("UTTERANCE:INTERJECTION+PUNCTUATION")
        for clause in clauses:
            rules.append("CLAUSE:" + clause.mode + ":" + clause.tense + ":" + str(clause.polarity))
            slots += self.clause(clause)
        if not slots or len(slots) > 64:
            raise DecodeError("empty or oversized grammatical derivation")
        return ResponsePlan(act, digest(contract.to_dict()), slots, tuple(rules), binding, coverage)


def compose(words: Iterable[str]) -> str:
    text = ""
    capital = True
    for word in words:
        if word in {".", "?", "!", ",", ":", ";"}:
            text = text.rstrip() + word
            capital = word in {".", "?", "!"}
        else:
            if word == "i" or capital:
                word = word[:1].upper() + word[1:]
            text += (" " if text else "") + word
            capital = False
    return text
