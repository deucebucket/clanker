"""V4 Layer 1.5: Force Flow Resolver — WHO does WHAT to WHOM.

Every sentence has a force flow: an actor pushes force toward a target.
The direction determines how the impact lands.

  "I love my dog"     → Self --love(+)--> pet      (giving affection)
  "He hit me"         → Other --hit(-)--> Self      (victimization)
  "I hit him"         → Self --hit(-)--> Other      (aggression, D stays)
  "The medicine stopped working" → thing --stopped--> function (purpose severed)

The resolver identifies Subject-Verb-Object triples from word roles
and computes a force flow direction that modifies how the physics
loop applies word forces.

Force flow affects:
  - D (dominance): actor keeps/gains D, target loses D
  - W (self-worth): self-directed negative = W drops harder
  - V (valence): direction amplifies or dampens based on who is affected
"""

from dataclasses import dataclass
from typing import List, Optional

from .word_classifier import WordRole


@dataclass
class ForceFlow:
    """Resolved force flow for a sentence."""
    actor_idx: int = -1          # position of the actor (subject)
    actor_role: str = ""         # SELF_REF, OTHER_REF, RELATION_REF, or NEUTRAL (thing)
    force_idx: int = -1          # position of the main force word (verb)
    force_valence: int = 0       # dV of the force word
    target_idx: int = -1         # position of the target (object)
    target_role: str = ""        # SELF_REF, OTHER_REF, RELATION_REF, or NEUTRAL
    negated: bool = False        # is the force negated? (stopped, not, didn't)

    @property
    def self_is_actor(self) -> bool:
        return self.actor_role == "SELF_REF"

    @property
    def self_is_target(self) -> bool:
        return self.target_role == "SELF_REF"

    @property
    def other_acts_on_self(self) -> bool:
        """Other/relation acts on self — victimization pattern."""
        return (self.actor_role in ("OTHER_REF", "RELATION_REF")
                and self.target_role == "SELF_REF")

    @property
    def self_acts_on_self(self) -> bool:
        """Self acts on self — self-directed force."""
        return self.actor_role == "SELF_REF" and self.target_role == "SELF_REF"

    @property
    def effective_valence(self) -> int:
        """Force valence after negation."""
        if self.negated:
            return -self.force_valence
        return self.force_valence


# Roles that can be actors or targets (entities, not operators)
_ENTITY_ROLES = {"SELF_REF", "OTHER_REF", "RELATION_REF"}
_NEGATOR_ROLES = {"NEGATOR"}


def resolve_force_flow(roles: List[WordRole]) -> Optional[ForceFlow]:
    """Resolve the primary force flow (SVO) from classified word roles.

    Scanning strategy:
      1. Find the strongest force word (highest |dV| in vocabulary)
      2. Look LEFT for the nearest entity → actor (subject)
      3. Look RIGHT for the nearest entity → target (object)
      4. Check for negators between actor and force

    Returns ForceFlow or None if no clear SVO found.
    """
    if len(roles) < 1:
        return None

    # Find the strongest emotional force word
    # Check both role-assigned force AND vocabulary lookup (same as pendulum)
    from .vocabulary import VOCABULARY
    best_force_idx = -1
    best_force_strength = 0

    for i, wr in enumerate(roles):
        force = wr.force or VOCABULARY.get(wr.word)
        if force is not None:
            strength = abs(force[0])  # |dV|
            if strength > best_force_strength:
                best_force_strength = strength
                best_force_idx = i

    if best_force_idx == -1 or best_force_strength < 10:
        return None  # no meaningful force word

    # Use the resolved force for valence
    force_word = roles[best_force_idx]
    _resolved_force = force_word.force or VOCABULARY.get(force_word.word)

    force_word = roles[best_force_idx]

    # Look LEFT for actor (nearest entity before the force word)
    actor_idx = -1
    actor_role = ""
    for j in range(best_force_idx - 1, -1, -1):
        if roles[j].role in _ENTITY_ROLES:
            actor_idx = j
            actor_role = roles[j].role
            break

    # Look RIGHT for target (nearest entity after the force word)
    target_idx = -1
    target_role = ""
    for j in range(best_force_idx + 1, len(roles)):
        if roles[j].role in _ENTITY_ROLES:
            target_idx = j
            target_role = roles[j].role
            break

    # If no explicit target but actor is SELF_REF and force is self-directed
    # (e.g., "i am stupid"), self is both actor and target
    if target_idx == -1 and actor_role == "SELF_REF":
        target_idx = actor_idx
        target_role = "SELF_REF"

    # If no explicit target but actor is OTHER/RELATION AND the force is
    # strong enough, the implied target is SELF. The user is the default
    # gravitational center. "he lied" = he lied to ME. "he proposed" = to ME.
    # But "she laughed" alone is ambiguous -- don't imply target on weak forces.
    if (target_idx == -1
            and actor_role in ("OTHER_REF", "RELATION_REF")
            and best_force_strength >= 25):
        target_role = "SELF_REF"  # implied, no index

    # IMPERATIVE detection: strong force word + no actor + no target + short sentence
    # = bare command aimed at the listener. "Shut up" = USER → OTHER.
    # "Get out" = USER → OTHER. The speaker is the actor, the listener is the target.
    #
    # ALSO: command tokens (getout, shutup, fuckoff) with possessive SELF_REF
    # ("get out of MY way", "shut MY door") = SELF is authority, not target.
    # The possessive "my" after a command = ownership, not victimhood.
    _COMMAND_TOKENS = {"shutup", "getout", "fuckoff", "backoff", "pissoff"}
    force_word_text = roles[best_force_idx].word if best_force_idx >= 0 else ""
    is_command_token = force_word_text in _COMMAND_TOKENS

    if is_command_token:
        # Command token always = SELF commands OTHER, regardless of possessives
        actor_role = "SELF_REF"
        target_role = "OTHER_REF"
    elif (actor_idx == -1 and target_idx == -1
            and best_force_strength >= 25 and len(roles) <= 6):
        actor_role = "SELF_REF"
        target_role = "OTHER_REF"
    elif (actor_role == "" and target_role == ""
            and best_force_strength >= 30):
        actor_role = "SELF_REF"
        target_role = "OTHER_REF"

    # If no actor or target resolved (even implied), give up
    if actor_role == "" and target_role == "":
        return None

    # Check for negation: either a NEGATOR between actor and force,
    # OR the actor itself is a negating word (nobody, nothing, no one → resolved to "nobody")
    # "nobody hurt me" = negated actor + negative verb = positive outcome
    # "nobody loves me" = negated actor + positive verb = negative outcome
    _NEGATING_ACTORS = {"nobody", "nothing", "none", "noone"}
    # Search for negators between actor and force, AND before actor
    # "it WASNT my fault" = wasnt is before "my" (actor) but negates the whole predicate
    search_start = max(0, (actor_idx - 2) if actor_idx >= 0 else 0)
    negated = any(
        roles[j].role in _NEGATOR_ROLES
        for j in range(search_start, best_force_idx)
    )
    # Actor itself is a negation word
    if actor_idx >= 0 and roles[actor_idx].word in _NEGATING_ACTORS:
        negated = True

    return ForceFlow(
        actor_idx=actor_idx,
        actor_role=actor_role,
        force_idx=best_force_idx,
        force_valence=_resolved_force[0] if _resolved_force else 0,
        target_idx=target_idx,
        target_role=target_role,
        negated=negated,
    )


def compute_intent(flow: Optional[ForceFlow], roles=None) -> int:
    """Compute Intent (I) dimension from force flow.

    Intent = WHERE is the force aimed and WHY.
      0   = WITHDRAW (retreating, cutting ties, pulling away)
      64  = DEFLECT (avoiding, redirecting, not engaging)
      128 = NEUTRAL (informational, no directional intent)
      192 = CONNECT (reaching toward, building, repairing)
      255 = CONTROL (dominating, directing, commanding)

    The intent is determined by:
    1. The force valence (positive = connect/heal, negative = poison/attack)
    2. The direction (who is acting on whom)
    3. Structural cues (PULL_AWAY = withdraw, POWER = control)
    """
    if flow is None:
        return 128  # no flow detected = neutral intent

    ev = flow.effective_valence
    intent = 128  # start neutral

    # Positive force directed at OTHER/RELATION = CONNECT (healing potion → other)
    if ev > 0 and flow.self_is_actor and not flow.self_is_target:
        intent = 160 + min(ev // 4, 60)  # 160-220

    # Self-directed force: accountability, self-attack, self-affirm, or deflection.
    # Must check BEFORE generic positive-self, because negated accountability
    # (ev positive after flip) would otherwise read as self-affirmation.
    elif flow.self_acts_on_self:
        _ACCOUNTABILITY_WORDS = {"wrong", "sorry", "apologize", "fault", "mistake",
                                "responsibility", "owe", "messed", "screwed"}
        force_word = roles[flow.force_idx].word if roles and 0 <= flow.force_idx < len(roles) else ""
        raw_fv = flow.force_valence  # before negation

        if force_word in _ACCOUNTABILITY_WORDS and not flow.negated and raw_fv < 0:
            # "I was wrong" = accepting the hit = CONNECT/REPAIR intent
            intent = 170 + min(abs(raw_fv) // 5, 40)  # 170-210 = connect
        elif force_word in _ACCOUNTABILITY_WORDS and flow.negated:
            # "It wasn't my fault" = DEFLECTING the hit = DEFLECT
            intent = 80 + min(abs(raw_fv) // 5, 30)  # 80-110 = deflect
        elif raw_fv < 0 and not flow.negated:
            # "I hate myself" = self-destruction = WITHDRAW
            intent = 40 - min(abs(raw_fv) // 6, 30)  # 40-10 = withdraw
        elif raw_fv > 0:
            # "I am proud of myself" = self-affirm = CONNECT
            intent = 150 + min(abs(raw_fv) // 6, 40)  # 150-190 = connect

    # Negative force from OTHER onto SELF = being attacked (DEFLECT/WITHDRAW)
    elif ev < 0 and flow.other_acts_on_self:
        intent = 80 - min(abs(ev) // 4, 60)  # 80-20 = deflect→withdraw

    # Negative force SELF → OTHER: could be ATTACK/CONTROL or SELF-ASSESSMENT
    # "im a burden to everyone" = self-assessment (withdraw), not attack
    # "i hate you" = attack (control)
    # Distinguish: if the force word is self-descriptive (burden, problem, waste)
    # it's self-assessment/withdraw, not control
    elif ev < 0 and flow.self_is_actor and not flow.self_is_target:
        # Check if this is self-assessment (self describing self negatively TO others)
        _SELF_ASSESSMENT = {"burden", "problem", "waste", "mistake", "obstacle",
                           "nuisance", "hindrance", "liability", "deadweight",
                           "nothing", "worthless", "useless", "failure", "trash",
                           "garbage", "broken", "pathetic", "stupid", "weak"}
        force_word = roles[flow.force_idx].word if roles and 0 <= flow.force_idx < len(roles) else ""
        if force_word in _SELF_ASSESSMENT:
            intent = 40 - min(abs(ev) // 6, 30)  # withdraw -- self-assessment, not attack
        else:
            intent = 200 + min(abs(ev) // 4, 55)  # control -- attacking other

    # Positive force from OTHER = receiving (not self-initiated)
    elif ev > 0 and flow.other_acts_on_self:
        intent = 155 + min(ev // 5, 50)  # 155-205 = connect (receiving)

    # Check for structural withdraw cues
    if roles:
        has_pull_away = any(r.role == "PULL_AWAY" for r in roles)
        has_finality = any(r.role == "FINALITY" for r in roles)
        if has_pull_away or has_finality:
            intent = min(intent, 80)  # cap at deflect -- pulling away

    return max(0, min(255, intent))


def compute_flow_modifiers(flow: Optional[ForceFlow]) -> dict:
    """Compute VADUGWI modifiers from force flow direction.

    Returns dict with keys: v_mod, d_mod, w_mod (multipliers, 1.0 = no change).
    """
    if flow is None:
        return {"v_mod": 1.0, "d_mod": 1.0, "w_mod": 1.0}

    v_mod = 1.0
    d_mod = 1.0
    w_mod = 1.0

    ev = flow.effective_valence

    if flow.other_acts_on_self:
        # Other → negative → Self = victimization: amplify negative V, drop D
        if ev < 0:
            v_mod = 1.3   # negative hits harder when you're the target
            d_mod = 0.8   # D drops — you're being acted upon
            w_mod = 1.2   # self-worth takes a hit from being targeted
        # Other → positive → Self = receiving love/support
        elif ev > 0:
            v_mod = 1.1   # positive slightly amplified (being loved)
            w_mod = 0.9   # self-worth slightly boosted (dampens W loss)

    elif flow.self_acts_on_self:
        # Self → negative → Self = self-attack: W drops hard
        if ev < 0:
            w_mod = 1.5   # self-directed negative hits W 50% harder
            d_mod = 0.85  # D drops — attacking yourself
        # Self → positive → Self = self-affirmation
        elif ev > 0:
            w_mod = 0.7   # W boosted (dampens W loss, amplifies W gain)

    elif flow.self_is_actor and not flow.self_is_target:
        # Self → action → Other = agency: D preserved
        if ev < 0:
            d_mod = 1.1   # you have power (you're the one acting)
        elif ev > 0:
            d_mod = 1.05  # slight D boost for positive agency

    return {"v_mod": v_mod, "d_mod": d_mod, "w_mod": w_mod}
