"""Construction eligibility tables, shared by parser components."""

from __future__ import annotations
from typing import Dict, Tuple
from ..model import ContentRelationType, EmbeddedInterrogativeStatus, GerundContentStatus, GerundRelationType, InfinitivalContentStatus, InfinitivalRelationType
from .types import EmbeddedInterrogativePredicateProfile, InfinitivalPredicateProfile, GerundPredicateProfile

class PredicateProfiles:
    """Reviewed construction predicates; data only, no mutable session state."""

    EMBEDDED_WH_MARKERS = {
        "who", "whom", "what", "which", "whose",
        "when", "where", "why", "how",
    }

    EMBEDDED_POLAR_MARKERS = {"whether", "if"}

    EMBEDDED_INTERROGATIVE_PREDICATES: Dict[
        str,
        EmbeddedInterrogativePredicateProfile,
    ] = {
        "ask": EmbeddedInterrogativePredicateProfile(
            EmbeddedInterrogativeStatus.ASKED,
            "questioning",
            210,
            allows_recipient=True,
        ),
        "wonder": EmbeddedInterrogativePredicateProfile(
            EmbeddedInterrogativeStatus.WONDERED,
            "uncertain_cognition",
            190,
        ),
        "know": EmbeddedInterrogativePredicateProfile(
            EmbeddedInterrogativeStatus.KNOWN,
            "knowledge",
            210,
        ),
        "remember": EmbeddedInterrogativePredicateProfile(
            EmbeddedInterrogativeStatus.REMEMBERED,
            "memory",
            200,
        ),
        "discover": EmbeddedInterrogativePredicateProfile(
            EmbeddedInterrogativeStatus.DISCOVERED,
            "discovery",
            210,
        ),
        "determine": EmbeddedInterrogativePredicateProfile(
            EmbeddedInterrogativeStatus.DISCOVERED,
            "determination",
            205,
        ),
        "tell": EmbeddedInterrogativePredicateProfile(
            EmbeddedInterrogativeStatus.REQUESTED,
            "answer_request",
            215,
            allows_recipient=True,
            allows_direct_answer_request=True,
        ),
    }

    INFINITIVAL_PREDICATES: Dict[str, InfinitivalPredicateProfile] = {
        "plan": InfinitivalPredicateProfile(
            InfinitivalRelationType.SUBJECT_CONTROL,
            InfinitivalContentStatus.PLANNED,
            "planning",
            215,
        ),
        "intend": InfinitivalPredicateProfile(
            InfinitivalRelationType.SUBJECT_CONTROL,
            InfinitivalContentStatus.INTENDED,
            "intention",
            220,
        ),
        "hope": InfinitivalPredicateProfile(
            InfinitivalRelationType.SUBJECT_CONTROL,
            InfinitivalContentStatus.HOPED,
            "hope",
            190,
        ),
        "want": InfinitivalPredicateProfile(
            InfinitivalRelationType.SUBJECT_CONTROL,
            InfinitivalContentStatus.DESIRED,
            "desire",
            195,
            allows_object_controller=True,
        ),
        "tell": InfinitivalPredicateProfile(
            InfinitivalRelationType.OBJECT_CONTROL,
            InfinitivalContentStatus.DIRECTED,
            "directive",
            210,
            allows_object_controller=True,
            allows_subject_controller=False,
        ),
        "ask": InfinitivalPredicateProfile(
            InfinitivalRelationType.OBJECT_CONTROL,
            InfinitivalContentStatus.REQUESTED,
            "request",
            205,
            allows_object_controller=True,
            allows_subject_controller=False,
        ),
        "seem": InfinitivalPredicateProfile(
            InfinitivalRelationType.RAISING,
            InfinitivalContentStatus.EVIDENTIAL,
            "appearance",
            170,
        ),
        "appear": InfinitivalPredicateProfile(
            InfinitivalRelationType.RAISING,
            InfinitivalContentStatus.EVIDENTIAL,
            "appearance",
            175,
        ),
    }

    GERUND_PREDICATES: Dict[str, GerundPredicateProfile] = {
        "enjoy": GerundPredicateProfile(
            GerundRelationType.GERUND_CONTENT,
            GerundContentStatus.ENJOYED,
            "enjoyment",
            205,
        ),
        "avoid": GerundPredicateProfile(
            GerundRelationType.GERUND_CONTENT,
            GerundContentStatus.AVOIDED,
            "avoidance",
            210,
        ),
        "start": GerundPredicateProfile(
            GerundRelationType.ASPECTUAL_START,
            GerundContentStatus.BEGUN,
            "aspectual_onset",
            220,
            phase_entailing=True,
        ),
        "begin": GerundPredicateProfile(
            GerundRelationType.ASPECTUAL_START,
            GerundContentStatus.BEGUN,
            "aspectual_onset",
            220,
            phase_entailing=True,
        ),
        "stop": GerundPredicateProfile(
            GerundRelationType.ASPECTUAL_STOP,
            GerundContentStatus.STOPPED,
            "aspectual_cessation",
            220,
            phase_entailing=True,
        ),
        "keep": GerundPredicateProfile(
            GerundRelationType.ASPECTUAL_CONTINUATION,
            GerundContentStatus.CONTINUED,
            "aspectual_continuation",
            215,
            phase_entailing=True,
        ),
        "continue": GerundPredicateProfile(
            GerundRelationType.ASPECTUAL_CONTINUATION,
            GerundContentStatus.CONTINUED,
            "aspectual_continuation",
            215,
            phase_entailing=True,
        ),
        "see": GerundPredicateProfile(
            GerundRelationType.PERCEPTION_PARTICIPIAL,
            GerundContentStatus.PERCEIVED,
            "visual_perception",
            205,
            allows_object_controller=True,
            allows_subject_controller=False,
        ),
        "hear": GerundPredicateProfile(
            GerundRelationType.PERCEPTION_PARTICIPIAL,
            GerundContentStatus.PERCEIVED,
            "auditory_perception",
            200,
            allows_object_controller=True,
            allows_subject_controller=False,
        ),
        "watch": GerundPredicateProfile(
            GerundRelationType.PERCEPTION_PARTICIPIAL,
            GerundContentStatus.PERCEIVED,
            "visual_perception",
            205,
            allows_object_controller=True,
            allows_subject_controller=False,
        ),
        "notice": GerundPredicateProfile(
            GerundRelationType.PERCEPTION_PARTICIPIAL,
            GerundContentStatus.PERCEIVED,
            "perception",
            205,
            allows_object_controller=True,
            allows_subject_controller=False,
        ),
    }

    CONTENT_PREDICATES: Dict[str, Tuple[ContentRelationType, str, int]] = {
        "say": (ContentRelationType.REPORTED, "speech", 205),
        "tell": (ContentRelationType.REPORTED, "speech", 205),
        "report": (ContentRelationType.REPORTED, "speech", 215),
        "claim": (ContentRelationType.REPORTED, "speech", 190),
        "think": (ContentRelationType.BELIEVED, "belief", 185),
        "believe": (ContentRelationType.BELIEVED, "belief", 185),
        "know": (ContentRelationType.KNOWN, "knowledge", 210),
        "notice": (ContentRelationType.PERCEIVED, "perception", 200),
        "hear": (ContentRelationType.PERCEIVED, "perception", 195),
    }

    CONTENT_RECIPIENT_PREDICATES = {"tell"}

    RELATIVE_MARKERS = {"who", "whom", "whose", "which", "that"}

    ABSTRACT_COMPLEMENT_HEADS = {
        "belief", "claim", "fact", "hope", "idea", "news", "reason",
        "report", "story", "thought",
    }

    PERSON_RELATIVE_HEADS = {
        "adult", "boy", "child", "doctor", "driver", "girl", "man",
        "nurse", "person", "student", "teacher", "technician", "woman",
        "worker",
    }

    FEMALE_RELATIVE_HEADS = {"girl", "mother", "nurse", "sister", "woman"}

    MALE_RELATIVE_HEADS = {"boy", "brother", "father", "man"}

    APPOSITIVE_ROLE_NOUNS = {
        "advisor", "boss", "coach", "colleague", "coworker", "doctor",
        "friend", "manager", "mentor", "nurse", "partner", "professor",
        "supervisor", "teacher", "therapist", "worker",
    }

    SUBORDINATE_MARKERS: Tuple[Tuple[str, ...], ...] = (
        ("even", "though"),
        ("so", "that"),
        ("because",),
        ("when",),
        ("while",),
        ("before",),
        ("after",),
        ("until",),
        ("since",),
        ("if",),
        ("unless",),
        ("although",),
        ("though",),
    )
