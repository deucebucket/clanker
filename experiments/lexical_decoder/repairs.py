"""Bounded conversation repairs for the opt-in decoder, not the shipped kernel.

These adapters consume actual identity, parse and dialogue state. They store no
finished replies. The frozen evaluation/production tree is deliberately intact.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from clanker_lm import lexicon
from clanker_lm.gates import ContextGate
from clanker_lm.learning import LexicalLearner, SEMANTIC_HINTS, POSITIVE_WORDS, NEGATIVE_WORDS
from clanker_lm.model import RefKind, SpeechAct
from clanker_lm.parser import SemanticParser
from clanker_lm.realize import SurfaceRealizer
from .affinity import tokens
from .grammar import DecodeError


class PerspectiveRealizer(SurfaceRealizer):
    """Ordinary answer NPs derive possession from IDs, not input deixis.

    Quoted content is not routed through this helper; attributed/quoted clauses
    remain outside the experiment's supported grammar. No global text replace.
    """
    def render_ref(self, ref, *, case="object", definite=False, capitalize=False):
        entity = self.memory.get_entity(ref.key) if ref and ref.kind == RefKind.ENTITY else None
        if entity and entity.owner_id and entity.entity_id not in {"user", "assistant"}:
            if entity.owner_id == entity.entity_id:
                raise DecodeError("self-referential ownership")
            if entity.owner_id == "user":
                owner = "your"
            elif entity.owner_id == "assistant":
                owner = "my"
            else:
                parent = self.memory.get_entity(entity.owner_id)
                if parent is None:
                    raise DecodeError("possessor identity is missing")
                # Render known relational possessor via its identity as well.
                if parent.relation and parent.owner_id == "user":
                    name = "your " + parent.relation
                else:
                    name = parent.canonical_name
                owner = name + ("'" if name.endswith("s") else "'s")
            name = entity.relation or entity.canonical_name
            words = name.split()
            if words and words[0].lower() in lexicon.POSSESSIVES | {"a", "an", "the"}:
                words = words[1:]
            if not words:
                raise DecodeError("possessed entity lacks a nominal head")
            text = owner + " " + " ".join(words)
            return self._capitalize(text) if capitalize else text
        return super().render_ref(ref, case=case, definite=definite, capitalize=capitalize)


class DefinitionLearner(LexicalLearner):
    """Conservative definition scope: reject negated hints, not invert them.

    'Not bad' alone does not prove 'good'. A positive contrast clause may supply
    positive evidence. Full pragmatics/litotes remain explicitly unsupported.
    """
    @staticmethod
    def _infer_semantic_class(text):
        sequence = tokens(text)
        accepted = set()
        negated = False
        for index, word in enumerate(sequence):
            if word in {".", ",", ";", "!", "?", "but", "instead", "however"}:
                negated = False
                continue
            if word in {"not", "never", "no", "isn't", "isnt", "doesn't", "doesnt"}:
                negated = True
                continue
            if not negated:
                accepted.add(word)
        positive, negative = len(accepted & POSITIVE_WORDS), len(accepted & NEGATIVE_WORDS)
        polarity = 1 if positive > negative else -1 if negative > positive else 0
        ranked = [(len(accepted & hints), label) for label, hints in SEMANTIC_HINTS.items()]
        score, label = max(ranked, key=lambda item: (item[0], item[1]))
        return (label if score else "unknown"), polarity

    def _looks_like_definition_reply(self, text):
        sequence = tuple(w for w in tokens(text) if w not in {".", ",", "!", "?"})
        if not sequence:
            return False
        if sequence[0] in {"actually", "basically", "meaning", "like"}:
            sequence = sequence[1:]
        # Feature fragments are answers, not arbitrary short declaratives.
        hints = set().union(*SEMANTIC_HINTS.values())
        fragment_words = hints | {"like", "and", "or", "very", "really", "not", "just", "a", "an", "the", "in", "way", "basically", "kind", "of"}
        if sequence and sequence[0] in {"positive", "negative", "neutral", "intense", "descriptive"} and set(sequence) <= fragment_words:
            return True
        if sequence[:2] in {("not", "good"), ("not", "bad"), ("not", "positive"), ("not", "negative")}:
            return True
        if sequence[:2] in {("it", "means"), ("that", "means"), ("this", "means")}:
            return True
        if sequence[0] in lexicon.QUESTION_WORDS or sequence[0] in lexicon.YES_NO_STARTERS:
            return False
        # Speculative parsing uses a COPY: checking ownership may otherwise
        # mutate entity salience or create facts before we decide to learn.
        parsed = SemanticParser().parse(text, copy.deepcopy(self.memory))
        if parsed.events or parsed.question or parsed.speech_act in {SpeechAct.GREET, SpeechAct.SOCIAL, SpeechAct.COMMAND}:
            return False
        hints = set().union(*SEMANTIC_HINTS.values())
        return len(sequence) <= 12 and bool(set(sequence) & hints)


@dataclass
class DialogueState:
    """One bounded active episode and its actually answered follow-up.

    This is a dialogue bridge, not a complete associative/appraisal graph.
    """
    topic_entities: list[str] = field(default_factory=list)
    episode_events: list[str] = field(default_factory=list)
    pending_elaboration: bool = False
    elaboration_answered: bool = False
    suppress_elaboration: bool = False
    last_decision: str = "initial"
    corrections: list[dict] = field(default_factory=list)

    def to_dict(self):
        return {"topic_entities": list(self.topic_entities), "episode_events": list(self.episode_events),
                "pending_elaboration": self.pending_elaboration, "elaboration_answered": self.elaboration_answered,
                "suppress_elaboration": self.suppress_elaboration, "last_decision": self.last_decision,
                "corrections": copy.deepcopy(self.corrections)}

    @classmethod
    def from_dict(cls, data):
        expected = set(cls().to_dict())
        if not isinstance(data, dict) or set(data) != expected:
            raise DecodeError("invalid dialogue fields")
        for key, maximum in (("topic_entities", 32), ("episode_events", 200)):
            if not isinstance(data[key], list) or len(data[key]) > maximum or any(not isinstance(x, str) or len(x) > 128 for x in data[key]):
                raise DecodeError("invalid dialogue identities")
        for key in ("pending_elaboration", "elaboration_answered", "suppress_elaboration"):
            if type(data[key]) is not bool:
                raise DecodeError("invalid dialogue flag")
        if not isinstance(data["last_decision"], str) or len(data["last_decision"]) > 128:
            raise DecodeError("invalid dialogue decision")
        if not isinstance(data["corrections"], list) or len(data["corrections"]) > 200:
            raise DecodeError("invalid correction ledger")
        for record in data["corrections"]:
            if not isinstance(record, dict) or set(record) != {"old_event", "new_event", "source", "evidence_sha256", "inherited_roles"}:
                raise DecodeError("invalid correction record")
            if record["source"] != "user_explicit_correction" or record["inherited_roles"] not in ([], ["time"]):
                raise DecodeError("invalid correction provenance")
            if any(not isinstance(record[k], str) or not record[k] or len(record[k]) > 128 for k in ("old_event", "new_event", "evidence_sha256")):
                raise DecodeError("invalid correction binding")
        return cls(**copy.deepcopy(data))

    def constrain(self, parse, gates, memory):
        current = {ref.key for event in parse.events for role, ref in event.arguments.items()
                   if ref.kind == RefKind.ENTITY and ref.key not in {"user", "assistant"}
                   and memory.get_entity(ref.key) is not None
                   and (memory.get_entity(ref.key).kind.value == "person"
                        or (memory.get_entity(ref.key).kind.value == "thing"
                            and role == "patient" and event.predicate not in {"be", "feel"}))}
        old = set(self.topic_entities)
        # A newly named, disjoint participant/object begins another episode.
        if current and old and current.isdisjoint(old):
            self.pending_elaboration = False
            self.elaboration_answered = False
            self.episode_events = []
            old = set()
        if current:
            self.topic_entities = sorted(old | current)[:32]
        details = [event for event in parse.events if event.discourse_role == "main"
                   and event.predicate not in {"be", "feel", "anger", "hate", "worry", "fear"}]
        if self.pending_elaboration and details and (not old or bool(current & old)):
            self.pending_elaboration = False
            self.elaboration_answered = True
            self.last_decision = "elaboration_answered_by_asserted_event"
        for event in details:
            if event.event_id and event.event_id not in self.episode_events:
                self.episode_events.append(event.event_id)
        self.episode_events = self.episode_events[-200:]
        if gates.response_act in {"empathic_followup", "serious_followup"}:
            if self.suppress_elaboration or self.elaboration_answered:
                gates.response_act = "empathic_acknowledge"
                gates.max_sentences = 1
                reason = "user_disabled_elaboration" if self.suppress_elaboration else "episode_explanation_already_received"
                self.last_decision = reason
                gates.rationale.append("dialogue:" + reason)
        if "explicit_object_replacement" in parse.diagnostics and gates.severity != "critical":
            gates.response_act = "neutral_acknowledge"
            gates.max_sentences = 1
            gates.rationale.append("dialogue:explicit_object_correction")
        return gates

    def completed(self, act):
        if act in {"empathic_followup", "serious_followup"}:
            self.pending_elaboration = True
            self.elaboration_answered = False
            self.last_decision = "elaboration_requested"


class DialogueGate(ContextGate):
    def __init__(self, store, dialogue):
        super().__init__(store)
        self.dialogue = dialogue

    def decide(self, text, parse, affect, memory, *, answer_status=None):
        gate = super().decide(text, parse, affect, memory, answer_status=answer_status)
        return self.dialogue.constrain(parse, gate, memory)


class ActiveEvidenceAnswerer:
    """Query a view excluding explicitly superseded facts; keep source history.

    Only the closed replacement form below creates these exclusions. Ordinary
    conflicting assertions are still conflicts, never silently overwritten.
    """
    def __init__(self, dialogue):
        from clanker_lm.qa import QuestionAnswerer
        self.base = QuestionAnswerer()
        self.dialogue = dialogue

    def answer(self, question, memory):
        superseded = {x["old_event"] for x in self.dialogue.corrections}
        if not superseded:
            return self.base.answer(question, memory)
        view = copy.deepcopy(memory)
        view.events = [e for e in view.events if e.event_id not in superseded]
        result = self.base.answer(question, view)
        result.diagnostics.append("active_evidence_excludes:" + ",".join(sorted(superseded)))
        return result


def explicit_replacement(text, memory, dialogue):
    """One positive simple object's explicit replacement, only if unambiguous.

    Returns (parse, source-change) or None. No sentence is generated here.
    Unsupported or ambiguous correction forms abstain without changing memory.
    """
    import re
    from .affinity import digest
    from clanker_lm.model import ParseResult
    match = re.fullmatch(r"\s*(?:actually|correction),?\s+(.+?),\s*not\s+([^,;?!]+?)\.?\s*", text, re.I)
    if not match:
        return None
    def decline(reason):
        return ParseResult(SpeechAct.UNKNOWN, text, diagnostics=["correction_declined:"+reason]), None
    if any(c in text for c in {'"', '“', '”'}):
        return decline("quotation_scope")
    staged = copy.deepcopy(memory)
    parsed = SemanticParser().parse(match.group(1), staged)
    if parsed.speech_act != SpeechAct.ASSERT or len(parsed.events) != 1 or parsed.unresolved:
        return decline("not_one_closed_assertion")
    event = parsed.events[0]
    if event.discourse_role != "main" or not event.polarity or event.modality or event.aspect != "simple":
        return decline("unsupported_scope")
    if set(event.arguments) - {"agent", "patient", "time"} or not {"agent", "patient"} <= set(event.arguments):
        return decline("unsupported_roles")
    old = staged.find_by_alias(match.group(2).rstrip("."))
    if not old.resolved:
        return decline("old_object_unresolved")
    replacement = event.arguments["patient"]
    if replacement.kind != RefKind.ENTITY or replacement.key == old.entity.entity_id:
        return decline("replacement_must_change_object")
    superseded = {r["old_event"] for r in dialogue.corrections}
    possible = []
    for candidate in memory.events:
        args = candidate.arguments
        if (candidate.event_id not in superseded and candidate.discourse_role == "main" and candidate.polarity
                and candidate.predicate == event.predicate and candidate.tense == event.tense
                and candidate.aspect == event.aspect and candidate.modality == event.modality
                and set(args) <= {"agent", "patient", "time"}
                and args.get("agent") and args["agent"].key == event.arguments["agent"].key
                and args.get("patient") and args["patient"].key == old.entity.entity_id):
            if "time" in event.arguments and ("time" not in args or args["time"].key != event.arguments["time"].key):
                continue
            possible.append(candidate)
    if len(possible) != 1:
        return decline("antecedent_count_"+str(len(possible)))
    previous = possible[0]
    inherited = []
    if "time" not in event.arguments and "time" in previous.arguments:
        event.arguments["time"] = copy.deepcopy(previous.arguments["time"])
        inherited.append("time")
    # Parsing this supported simple clause only stages entity state/salience.
    # Keep the existing memory object so all runtime consumers share one owner.
    memory.entities = staged.entities
    memory._entity_counter = staged._entity_counter
    memory.revision = staged.revision
    parsed.raw_text = text
    event.raw_text = text
    parsed.diagnostics.append("explicit_object_replacement")
    return parsed, {"old_event": previous.event_id, "new_event": "", "source": "user_explicit_correction",
                    "evidence_sha256": digest(text), "inherited_roles": inherited}
