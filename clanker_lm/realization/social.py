"""Social responsibilities; extracted without changing decision rules."""

from __future__ import annotations
from typing import List
from .. import lexicon
from ..model import AnswerContract, CandidateResponse, EntityKind, EventFrame, GateDecision, SemanticRef
from .types import Part

class SocialComponent:
    """Social behavior composed by the stable public SurfaceRealizer API."""

    def _acknowledge(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        """Realize only candidates inside the planner-selected act class."""

        if contract.response_goal == "social" or gates.response_act == "social":
            return self._social(contract, gates)

        period = self._atom("punct.period")
        route_plan = self._rule_plan("reply:acknowledge")
        response_act = gates.response_act or "neutral_acknowledge"

        if response_act == "safety_probe" or gates.severity == "critical":
            return [
                self._candidate(
                    [*self._safe_question_parts()],
                    candidate_id="compose.acknowledge.safety_probe",
                    semantic_plan=[*route_plan, "RESPONSE_ACT:safety_probe", "ACT:SAFETY_CHECK"],
                    priority=140,
                )
            ]

        if response_act == "positive_acknowledge":
            positive = self.render_event(
                EventFrame(
                    "sound",
                    {
                        "subject": SemanticRef.literal(
                            "that", self._atom("demonstrative.that").surface
                        ),
                        "state": SemanticRef.literal(
                            "good", self._atom("evaluation.good").surface
                        ),
                    },
                ),
                capitalize=True,
            )
            return [
                self._candidate(
                    [positive, period],
                    candidate_id="compose.acknowledge.positive",
                    semantic_plan=[
                        *route_plan,
                        "RESPONSE_ACT:positive_acknowledge",
                        "ACT:POSITIVE_RECOGNITION",
                    ],
                    priority=128,
                )
            ]

        if response_act == "empathic_acknowledge":
            sorry = self.render_event(
                EventFrame(
                    "be",
                    {
                        "subject": SemanticRef.entity(
                            "assistant", "I", EntityKind.PERSON
                        ),
                        "value": SemanticRef.literal(
                            "sorry", self._atom("empathy.sorry").surface
                        ),
                    },
                ),
                capitalize=True,
            )
            rough = self.render_event(
                EventFrame(
                    "sound",
                    {
                        "subject": SemanticRef.literal(
                            "that", self._atom("demonstrative.that").surface
                        ),
                        "state": SemanticRef.literal(
                            "rough", self._atom("evaluation.rough").surface
                        ),
                    },
                ),
                capitalize=True,
            )
            return [
                self._candidate(
                    [sorry, period],
                    candidate_id="compose.acknowledge.loss.sorry",
                    semantic_plan=[
                        *route_plan,
                        "RESPONSE_ACT:empathic_acknowledge",
                        "ACT:EMPATHY",
                    ],
                    priority=132,
                ),
                self._candidate(
                    [rough, period],
                    candidate_id="compose.acknowledge.loss.rough",
                    semantic_plan=[
                        *route_plan,
                        "RESPONSE_ACT:empathic_acknowledge",
                        "ACT:VALIDATE",
                    ],
                    priority=124,
                ),
            ]

        if response_act == "serious_followup":
            sorry = self.render_event(
                EventFrame(
                    "be",
                    {
                        "subject": SemanticRef.entity(
                            "assistant", "I", EntityKind.PERSON
                        ),
                        "value": SemanticRef.literal(
                            "sorry", self._atom("empathy.sorry").surface
                        ),
                    },
                ),
                capitalize=True,
            )
            serious = self.render_event(
                EventFrame(
                    "sound",
                    {
                        "subject": SemanticRef.literal(
                            "that", self._atom("demonstrative.that").surface
                        ),
                        "state": SemanticRef.literal(
                            "serious", self._atom("evaluation.serious").surface
                        ),
                    },
                ),
                capitalize=True,
            )
            question = self._what_happened_question()
            candidates = [
                self._candidate(
                    [serious, period, *question],
                    candidate_id="compose.acknowledge.serious",
                    semantic_plan=[
                        *route_plan,
                        "RESPONSE_ACT:serious_followup",
                        "ACT:SEVERITY_RECOGNITION",
                        "ACT:OPEN_FOLLOWUP",
                    ],
                    priority=124,
                )
            ]
            if gates.masking or gates.register == "casual":
                candidates.insert(
                    0,
                    self._candidate(
                        [sorry, period, serious, period, *question],
                        candidate_id="compose.acknowledge.masked_serious",
                        semantic_plan=[
                            *route_plan,
                            "RESPONSE_ACT:serious_followup",
                            "ACT:EMPATHY",
                            "ACT:SEVERITY_RECOGNITION",
                            "ACT:OPEN_FOLLOWUP",
                        ],
                        priority=132,
                    ),
                )
            return candidates

        if response_act == "empathic_followup":
            rough = self.render_event(
                EventFrame(
                    "sound",
                    {
                        "subject": SemanticRef.literal(
                            "that", self._atom("demonstrative.that").surface
                        ),
                        "state": SemanticRef.literal(
                            "rough", self._atom("evaluation.rough").surface
                        ),
                    },
                ),
                capitalize=True,
            )
            return [
                self._candidate(
                    [rough, period, *self._what_happened_question()],
                    candidate_id="compose.acknowledge.rough",
                    semantic_plan=[
                        *route_plan,
                        "RESPONSE_ACT:empathic_followup",
                        "ACT:VALIDATE",
                        "ACT:OPEN_FOLLOWUP",
                    ],
                    priority=120,
                )
            ]

        # Neutral acknowledgment uses a deterministic discourse-cycle instead
        # of randomness.  Only one candidate is emitted, so affective ranking
        # cannot escape the selected response-act class or repeat one form on
        # every compatible turn.
        variant = max(0, self.memory.turn_index - 1) % 3
        if variant == 0:
            verb = self._atom("ack.understand")
            clause = self.render_event(
                EventFrame(
                    verb.lemma,
                    {"agent": SemanticRef.entity("assistant", "I", EntityKind.PERSON)},
                ),
                capitalize=True,
            )
            parts: List[Part] = [clause, period]
            candidate_id = "compose.acknowledge.neutral.understand"
            act_plan = "ACT:UNDERSTAND"
        elif variant == 1:
            verb = self._atom("ack.hear")
            clause = self.render_event(
                EventFrame(
                    verb.lemma,
                    {
                        "agent": SemanticRef.entity(
                            "assistant", "I", EntityKind.PERSON
                        ),
                        "patient": SemanticRef.entity(
                            "user", "you", EntityKind.PERSON
                        ),
                    },
                ),
                capitalize=True,
            )
            parts = [clause, period]
            candidate_id = "compose.acknowledge.neutral.hear"
            act_plan = "ACT:HEAR"
        else:
            parts = [
                self._atom("pronoun.i"),
                self._atom("ack.get"),
                self._atom("demonstrative.that"),
                period,
            ]
            candidate_id = "compose.acknowledge.neutral.get"
            act_plan = "ACT:GET"

        return [
            self._candidate(
                parts,
                candidate_id=candidate_id,
                semantic_plan=[
                    *route_plan,
                    "RESPONSE_ACT:neutral_acknowledge",
                    f"DISCOURSE_VARIANT:{variant}",
                    act_plan,
                ],
                priority=110,
            )
        ]

    def _social(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        convention = contract.question.social_convention if contract.question else contract.reason
        period = self._atom("punct.period")
        if convention == "wellbeing_check":
            return [
                self._candidate(
                    [
                        self._atom("pronoun.i"),
                        self._atom("aux.am"),
                        self._atom("state.ready"),
                        self._atom("prep.to"),
                        self._atom("process.work"),
                        period,
                    ],
                    candidate_id="compose.social.wellbeing",
                    semantic_plan=[*self._rule_plan("reply:social"), "STATE:READY", "PURPOSE:WORK"],
                    priority=110,
                )
            ]
        if convention == "activity_check":
            return [
                self._candidate(
                    [self._atom("pronoun.i"), self._atom("aux.am"), self._atom("state.working"), period],
                    candidate_id="compose.social.activity",
                    semantic_plan=[*self._rule_plan("reply:social"), "ACTIVITY:WORK"],
                    priority=110,
                )
            ]
        preferred = "social.hey" if gates.register == "casual" else "social.hello"
        return [
            self._candidate(
                [self._atom(preferred), period],
                candidate_id=f"compose.social.{preferred}",
                semantic_plan=[*self._rule_plan("reply:social"), "ACT:GREET"],
                priority=108,
            )
        ]

    def _what_happened_question(self) -> List[Part]:
        happen = self._atom("event.happen")
        return [self._atom("question.what"), lexicon.past_form(happen.lemma), self._atom("punct.question")]

    def _safe_question_parts(self) -> List[Part]:
        return [self._atom("aux.are"), self._atom("pronoun.you"), self._atom("evaluation.safe"), self._atom("punct.question")]
