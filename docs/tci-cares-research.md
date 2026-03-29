# TCI, CARE, and Psychological Frameworks for Emotional Modeling

Research compiled for Clanker-Lang VADUG integration.

---

## 1. TCI (Therapeutic Crisis Intervention) -- Cornell University

TCI is a crisis management protocol developed by Cornell University's Residential Child Care Project (RCCP), first funded in 1979 by the National Center on Child Abuse and Neglect. Now in Edition 7, it is the standard training system for residential child care facilities worldwide.

### 1.1 The Stress Model of Crisis

The central framework of TCI. It models emotional escalation as a **five-phase arc** over time, with stress level on the Y-axis and time on the X-axis.

#### Phase 1: Baseline (Pre-Crisis)

- The individual's normal functioning state when **not** stressed.
- Fundamental principle: you cannot detect escalation unless you know each individual's unique baseline.
- Baseline is NOT a single fixed point -- it varies per person and per day based on **setting conditions** (see below).

**VADUG mapping:**
- Each person has a characteristic VADUG "home position"
- V: personal neutral (not necessarily 128 -- a depressed person's baseline V might be 90)
- A: resting arousal level
- D: habitual sense of control
- U: ambient urgency (0 for most, higher for chronically stressed individuals)
- G: default gravity (grounded vs. already sinking)

#### Phase 2: Triggering Event

- An identifiable stressor that disrupts baseline functioning.
- Individuals already under high ambient stress are more likely to react strongly to triggers.
- **Setting conditions** amplify trigger sensitivity:
  - Environmental: noise, light, overstimulation, sensory overload
  - Social: conflict, perceived threat, rejection, unfamiliar people
  - Structural: unclear boundaries, broken routines, unmet expectations
  - Internal: hunger, fatigue, pain, medication effects

**VADUG mapping:**
- Sudden dV negative (valence drops)
- dA positive spike (arousal jumps)
- dD negative (sense of control drops)
- dU positive (urgency appears)
- dG negative (gravity pulls down -- "sinking feeling")

#### Phase 3: Escalation

- Observable signs: increased anxiety, agitation, verbal threats, pacing, clenched fists.
- **Critical property**: as behaviors increase in duration, frequency, or intensity, the likelihood of responding to intervention **decreases**.
- This is a narrowing funnel -- early intervention works, late intervention does not.
- The individual is losing rational control progressively.

**VADUG mapping:**
- V continues dropping (accelerating negative)
- A climbing steeply (approaching 200+)
- D collapsing (feeling increasingly helpless/trapped)
- U rising fast
- G plummeting (emotional free-fall)
- Momentum is high and increasing -- pendulum physics should show acceleration

#### Phase 4: Outburst (Crisis Peak)

- The individual may act out aggressively -- dangerous to self and others.
- Rational thought is largely offline.
- The primary goal shifts from de-escalation to **safety and containment**.
- This is the point of maximum arousal and minimum control.

**VADUG mapping:**
- V: very low (sub-40, extreme negative affect)
- A: maxed (240-255, fight-or-flight)
- D: near floor (0-30, total loss of control)
- U: maxed (255, everything is emergency)
- G: near floor (0-30, emotional collapse / crushing weight)

#### Phase 5: Recovery (De-escalation + Post-Crisis)

- After the outburst energy is spent, the individual begins returning toward baseline.
- **Two sub-phases**:
  - **Active recovery**: still emotionally raw, may re-escalate if provoked. Not yet ready for reasoning.
  - **Post-crisis stabilization**: can begin reflecting, reasoning, and learning. This is when the Life Space Interview happens.
- Recovery does NOT necessarily return to the original baseline. Repeated crises can shift baseline (see allostatic load, Section 4).

**VADUG mapping:**
- V slowly climbing back (but may settle lower than original baseline)
- A dropping (but often overshoots into hypoarousal -- emotional exhaustion)
- D gradually returning
- U dropping to near-zero
- G recovering toward grounded (128) but may remain heavy

### 1.2 The Four Questions

At every phase of a potential crisis, staff are trained to continuously ask themselves four assessment questions:

| # | Question | What It Assesses |
|---|----------|-----------------|
| 1 | **What am I feeling now?** | Staff self-awareness -- own emotional state, countertransference, activation level |
| 2 | **What does this young person feel, need, or want?** | Empathic reading of the child's internal state |
| 3 | **How is the environment affecting the situation?** | Setting conditions, sensory factors, audience effects |
| 4 | **How do I best respond?** | Intervention selection -- match response to phase |

**Clanker relevance**: These map to a dual-VADUG system -- one vector for the speaker, one for the listener/responder. The "environment" question maps to setting conditions that modify dark matter parameters.

### 1.3 Pain-Based Behavior

Core TCI principle: **a child's behavior is an expression of their needs, not their character.**

- "Pain-based behaviors" are caused or triggered by trauma.
- Aggression, withdrawal, and self-harm are expressions of emotional pain, not defiance.
- Key rule: "No child should be punished for behavior that is a result of pain -- either physical or emotional, as this inflicts pain on top of the pain they already feel."
- Punitive responses to pain-based behavior increase allostatic load and narrow the window of tolerance.

**Clanker relevance**: This maps to the concept that negative VADUG states are **signals**, not errors. The engine should not "correct" negative affect -- it should **read** it as information about unmet needs.

### 1.4 Behavior Support Techniques by Phase

| Phase | Techniques | Mechanism |
|-------|-----------|-----------|
| **Baseline** | Manage setting conditions, build relationship, maintain routines | Prevent triggering |
| **Triggering/Anxiety** | Prompting, redirection, caring gestures, hurdle help, proximity, environmental modification | Interrupt escalation early |
| **Escalation** | Crisis co-regulation: calm presence, simple language, low-stimulation environment, directive communication, space | Contain momentum |
| **Outburst** | Safety protocols, physical space, limit audience, document, maintain calm presence | Ride out the peak |
| **Recovery** | Reassurance, gradual normalization, validation, Life Space Interview | Teach and repair |

### 1.5 Crisis Co-Regulation

When a child begins escalating, a calm adult provides "emotional first aid":

- Deep breathing (models regulation)
- Calm body language (non-threatening posture)
- Minimal verbal responses (avoids cognitive overload)
- Active listening (reflects feelings)
- Time and space (allows natural de-escalation)
- Positive self-talk by the adult (manages own activation)

**Clanker relevance**: Co-regulation is a **VADUG targeting** operation. The responder aims to output a VADUG vector that pulls the escalating person's state back toward baseline. The response VADUG should be:
- V: moderately positive (not artificially cheerful)
- A: low (calm models calm)
- D: moderate (conveys confidence without dominance)
- U: low (signals "no emergency")
- G: grounded (128 -- stable center of gravity)

### 1.6 Life Space Interview (LSI) -- I ESCAPE

The post-crisis teaching tool. Conducted as close to the event as possible, by the adult who helped through the crisis.

| Step | Action | Purpose |
|------|--------|---------|
| **I** - Isolate | Move to a private, calm space | Remove audience/stimulation |
| **E** - Explore | Ask the young person to tell their story | Understand their perspective |
| **S** - Summarize | Reflect back their story and feelings | Validate, show understanding |
| **C** - Connect | Link feelings to behaviors | Build insight: "you felt X so you did Y" |
| **A** - Alternative | Discuss what could be done differently next time | Skill-building |
| **P** - Plan | Create a concrete plan for future situations | Actionable coping strategy |
| **E** - Enter | Return to routine | Normalize, move forward |

### 1.7 Non-Verbal Communication Ratios

TCI teaches that meaning is conveyed through:
- **Facial expression: 55%**
- **Tone of voice: 38%**
- **Words: 7%**

This is the Mehrabian ratio. Clanker's word-force engine currently operates only on the 7% channel. Tone and expression would require audio/visual input or metadata annotations.

---

## 2. CARE (Children And Residential Experiences) -- Cornell

CARE stands for **Children And Residential Experiences: Creating Conditions for Change**. It is a separate but related program from the same Cornell RCCP that developed TCI.

### 2.1 Six Evidence-Based Principles

| Principle | Description |
|-----------|-------------|
| **Developmentally focused** | Interventions match the child's developmental stage, not chronological age |
| **Family involved** | Family engagement is maintained even in residential placement |
| **Relationship based** | Therapeutic change happens through relationships, not programs |
| **Competence centered** | Build on strengths, not just treat deficits |
| **Trauma informed** | All behavior understood through the lens of trauma history |
| **Ecologically oriented** | The whole environment (peers, staff, structure) is the intervention |

### 2.2 Measured Outcomes

Research across 13 agencies in a four-year Duke Foundation study showed:
- **3-5% per month decrease** in incidents of aggression toward staff
- **3-5% per month decrease** in property destruction and running away
- **8-14% improvement** in children's perceptions of relationship quality with caregivers
- Decreases in psychotropic medication use
- Decreases in physical restraint use

### 2.3 Implementation Model

- 4-year partnership with RCCP
- Requires commitment from all levels: staff, supervisors, managers, administrators
- Listed as a **Promising Research Evidence based model** by the California Evidence-Based Clearinghouse for Child Welfare (CEBC)

**Clanker relevance**: CARE's ecological orientation maps to the concept that emotional processing is not just about the individual signal -- it is about the environment, relationships, and accumulated context. This supports the case for dark matter (background state) as a first-class parameter.

---

## 3. Residential/Group Home Emotional Pattern Prediction

### 3.1 Predictive Indicators for Escalation

Professionals in residential care track these categories of indicators:

**Temporal patterns:**
- Time of day (transitions, mealtimes, bedtime are high-risk)
- Day of week (Mondays after weekends with family, Fridays before)
- Seasonal patterns (holidays, anniversaries of traumatic events)
- Time since last crisis (refractory period vs. clustering)

**Individual history:**
- Previous crisis patterns (frequency, duration, typical triggers)
- Trauma anniversaries
- Medication changes or missed doses
- Sleep quality previous night
- Contact (or lack of contact) with family

**Environmental:**
- Staff changes (unfamiliar adults)
- Peer conflicts
- Routine disruptions
- Sensory environment (noise, crowding, temperature)

**Behavioral precursors:**
- Changes from baseline (withdrawal, agitation, sleep disruption)
- Increased frequency of minor incidents before major ones
- Verbal escalation patterns

### 3.2 Mathematical Models for Crisis Prediction

Published research uses:

**Machine learning approaches (AUC 0.68-0.88):**
- Electronic health record data predicting mental health crises in adolescents
- Hybrid models combining coded clinical data + narrative behavioral indicators
- Best results: sensitivity ~0.80, specificity ~0.87

**Time-to-event models:**
- Illinois foster care system: predicting risk of running away or residential placement entry
- Survival analysis with demographic, behavioral, and placement history features

**Setting-level intervention (CARE model):**
- Incident reports as markers for broader interaction patterns
- Staff sensitivity to emotional distress as a predictive moderator
- Springer Nature published study: "Intervening at the Setting Level to Prevent Behavioral Incidents in Residential Child Care: Efficacy of the CARE Program Model"

**Clanker relevance**: Crisis prediction maps to VADUG trajectory analysis. A time series of VADUG vectors can be analyzed for:
- Trend (is V declining over time? is A climbing?)
- Volatility (are swings getting wider?)
- Baseline drift (has the "resting" VADUG shifted?)
- Pattern matching (does this trajectory resemble previous pre-crisis patterns?)

---

## 4. Window of Tolerance, Allostatic Load, and Emotional Capacity

### 4.1 Window of Tolerance (Dan Siegel, 1999)

The **window of tolerance** is the optimal zone of arousal within which an individual can function effectively -- emotionally regulated, cognitively flexible, and socially engaged.

#### The Three Zones

```
    ┌─────────────────────────────────────┐
    │         HYPERAROUSAL ZONE           │
    │  panic, irritability, impulsivity   │
    │  hypervigilance, anxiety, rage      │
    │  fight-or-flight activation         │
    │  emotional flooding                 │
    ├─────────────────────────────────────┤  ← Upper threshold
    │                                     │
    │      WINDOW OF TOLERANCE            │
    │   (Optimal Arousal Zone)            │
    │                                     │
    │  emotionally regulated              │
    │  cognitively flexible               │
    │  socially engaged                   │
    │  can think, feel, and act           │
    │                                     │
    ├─────────────────────────────────────┤  ← Lower threshold
    │         HYPOAROUSAL ZONE            │
    │  emotional numbing, dissociation    │
    │  withdrawal, shutdown, flat affect  │
    │  disconnection, freeze response     │
    │  depressive collapse                │
    └─────────────────────────────────────┘
```

#### Key Properties

1. **Dynamic, not static**: The window width fluctuates with circumstances (stress, fatigue, safety, support).
2. **Expandable**: Consistent therapeutic practice, co-regulation, and safety can widen the window over time.
3. **Narrowable**: Trauma, chronic stress, sleep deprivation, and isolation narrow it.
4. **Individual**: Each person's window has different width and threshold positions.
5. **Bidirectional exits**: A person can leave the window upward (hyperarousal) or downward (hypoarousal).

#### Trauma's Effect on the Window

- PTSD: rapid oscillation between hyperarousal and hypoarousal, narrow window
- C-PTSD: sustained dysregulation, chronically narrow window
- BPD: narrow and unstable window, rapid shifting between emotional extremes
- Developmental trauma: window may never have been properly established

**VADUG mapping -- this is the critical connection:**

The window of tolerance maps directly to a **range parameter** on each VADUG dimension:

```
Window of Tolerance  →  VADUG "dark matter" range
─────────────────────────────────────────────────
Upper threshold      →  max_A before hyperarousal flag
Lower threshold      →  min_A before hypoarousal flag
Window width         →  tolerance_range = max_A - min_A
Window center        →  optimal_A (person's ideal arousal)
Window narrowing     →  tolerance_range decreases over accumulated stress
Window expansion     →  tolerance_range increases with safety/co-regulation
```

This applies to ALL five dimensions, not just Arousal:

| Dimension | Hyperarousal Exit | Hypoarousal Exit |
|-----------|-------------------|-------------------|
| V (Valence) | Manic positivity (V > 240) | Depressive collapse (V < 30) |
| A (Arousal) | Panic, rage (A > 220) | Shutdown, dissociation (A < 30) |
| D (Dominance) | Controlling, aggressive (D > 230) | Helpless, frozen (D < 20) |
| U (Urgency) | Everything is emergency (U > 200) | Nothing matters (U = 0 with low V) |
| G (Gravity) | Untethered mania (G > 240) | Crushing despair (G < 20) |

### 4.2 Allostatic Load (McEwen & Stellar, 1993)

**Allostatic load** is "the wear and tear on the body" that accumulates from repeated or chronic stress. It represents the cumulative physiological cost of maintaining stability (allostasis) under stress.

#### The Four Mechanisms of Accumulation

| Mechanism | Description | VADUG Analog |
|-----------|-------------|--------------|
| **Frequent activation** | Too many stress responses too often | High crisis frequency → dark matter drift |
| **Failed shutdown** | Stress response does not terminate after stressor ends | Recovery phase does not return to original baseline |
| **Inadequate response** | System fails to mount appropriate response | Blunted VADUG movement (flat affect despite stressors) |
| **Anticipatory load** | Chronic worry / hypervigilance even without stressor | Elevated resting A and U values |

#### Biomarkers and Scoring

The most established scoring method (MacArthur Studies of Successful Aging):

**Count-based method:**
1. Measure 10-14 biomarkers across systems (neuroendocrine, cardiovascular, metabolic, immune)
2. For each biomarker, determine if the individual falls in the **highest-risk quartile** (top 25%) -- or bottom 25% for protective markers like HDL and DHEA-S
3. Assign 1 point for each high-risk biomarker, 0 otherwise
4. **Allostatic Load Score = sum of high-risk counts** (range: 0-14)

**Primary mediators (stress hormones):**
- Cortisol (HPA axis)
- Epinephrine, Norepinephrine (sympathetic nervous system)
- DHEA-S (protective counter-regulator)

**Secondary outcomes (organ damage):**
- Systolic/diastolic blood pressure
- Waist-hip ratio
- HDL cholesterol, total cholesterol
- Glycated hemoglobin (HbA1c)
- C-reactive protein (inflammation)

**Energetic Model of Allostatic Load (EMAL):**
- The "load" is the additional energetic burden required to support allostasis
- Stressors increase Additional Stress Energy Expenditure (ASEE)
- When ASEE exceeds the organism's reserve capacity, it impinges on growth, maintenance, and repair
- Formula concept: `health_impact = max(0, ASEE - reserve_capacity)`

**VADUG mapping -- dark matter drift:**

```
Allostatic Load        →   Dark Matter Drift Accumulation
─────────────────────────────────────────────────────────
Biomarker count score  →   drift_score = count of VADUG dimensions
                           outside healthy range
Each crisis episode    →   drift += crisis_severity * recovery_deficit
Failed recovery        →   baseline shifts: new_baseline = old_baseline
                           + (crisis_peak - old_baseline) * leak_factor
Anticipatory load      →   resting_U and resting_A creep upward
Reserve capacity       →   window_of_tolerance width
```

### 4.3 Resilience Measurement Scales

Validated psychological instruments for quantifying emotional resilience:

| Scale | Items | Scoring | Range | What It Measures |
|-------|-------|---------|-------|-----------------|
| **Resilience Scale (RS-25)** | 25 | 7-point Likert | 25-175 | Personal competence, acceptance of self/life |
| **Connor-Davidson (CD-RISC-25)** | 25 | 0-4 scale | 0-100 | Hardiness, persistence, trust, tolerance, control |
| **Brief Resilience Scale (BRS)** | 6 | 5-point Likert | 1-5 (mean) | Ability to bounce back from stress |
| **Resilience Evaluation Scale (RES)** | 9 | Likert | varies | Self-confidence, self-efficacy |
| **Resilience Scale for Adults (RSA)** | 33 | 5 dimensions | varies | Intra- and inter-personal protective factors |

**Clanker relevance**: Resilience maps to **how quickly and completely VADUG returns to baseline** after perturbation. A high-resilience personality vector would have:
- Faster recovery rate (pendulum damping coefficient)
- Stronger baseline pull (spring constant toward home position)
- Wider window of tolerance (larger acceptable range before flagging)
- Lower drift accumulation (better "failed shutdown" prevention)

### 4.4 Polyvagal Theory (Stephen Porges, 1994)

A complementary framework that maps autonomic nervous system states to emotional/behavioral zones. Three hierarchical states:

| State | ANS Branch | Behavior | VADUG Zone |
|-------|-----------|----------|------------|
| **Ventral vagal** (social engagement) | Myelinated vagus | Calm, connected, flexible, social | Within window of tolerance |
| **Sympathetic** (mobilization) | Sympathetic NS | Fight-or-flight, anxiety, anger, panic | Hyperarousal zone |
| **Dorsal vagal** (immobilization) | Unmyelinated vagus | Freeze, shutdown, dissociation, collapse | Hypoarousal zone |

**Measurable biomarker**: Respiratory Sinus Arrhythmia (RSA) -- heart rate variability synchronized with breathing -- serves as a real-time index of vagal tone (ventral vagal engagement). Higher RSA = wider window of tolerance.

---

## 5. Synthesis: Mapping to Clanker-Lang

### 5.1 TCI Escalation Stages as VADUG Trajectory Patterns

```
Phase          V    A    D    U    G    Pendulum State
─────────────────────────────────────────────────────────
Baseline       128  100  128  0    128  At rest, slight oscillation
Trigger        110  140  100  40   105  Sharp perturbation
Escalation     80   180  70   120  70   Accelerating swing, momentum building
Outburst       30   250  15   255  10   Maximum displacement, crisis lock
Recovery-early 60   180  40   100  50   Swing reversing, still volatile
Recovery-late  100  120  90   20   110  Damping toward new baseline
Post-crisis    115  95   120  5    125  Near baseline (may be shifted)
```

### 5.2 Window of Tolerance as Dark Matter Range Parameter

```python
@dataclass
class DarkMatter:
    """Background emotional state -- the 'weather' beneath the 'waves'."""

    # Baseline (home position for each dimension)
    baseline_v: int = 128
    baseline_a: int = 100
    baseline_d: int = 128
    baseline_u: int = 0
    baseline_g: int = 128

    # Window of tolerance (per-dimension range)
    tolerance_v: int = 80   # V can swing 80 points before flagging
    tolerance_a: int = 100  # A has wider natural range
    tolerance_d: int = 70
    tolerance_u: int = 60
    tolerance_g: int = 80

    # Allostatic drift (accumulated stress burden)
    drift_v: float = 0.0    # cumulative baseline shift
    drift_a: float = 0.0
    drift_d: float = 0.0
    drift_u: float = 0.0
    drift_g: float = 0.0

    # Resilience parameters
    recovery_rate: float = 0.15   # how fast pendulum returns to baseline
    leak_factor: float = 0.02    # how much of each crisis bleeds into baseline
    crisis_count: int = 0        # running count for allostatic scoring

    def is_within_window(self, vadug: 'VADUG') -> dict:
        """Check which dimensions are inside window of tolerance."""
        return {
            'v': abs(vadug.v - self.baseline_v) <= self.tolerance_v / 2,
            'a': abs(vadug.a - self.baseline_a) <= self.tolerance_a / 2,
            'd': abs(vadug.d - self.baseline_d) <= self.tolerance_d / 2,
            'u': abs(vadug.u - self.baseline_u) <= self.tolerance_u / 2,
            'g': abs(vadug.g - self.baseline_g) <= self.tolerance_g / 2,
        }

    def allostatic_load_score(self) -> int:
        """Count-based allostatic load: how many dimensions have drifted
        beyond healthy range (analogous to biomarker quartile method)."""
        count = 0
        if abs(self.drift_v) > 15: count += 1
        if abs(self.drift_a) > 15: count += 1
        if abs(self.drift_d) > 15: count += 1
        if abs(self.drift_u) > 15: count += 1
        if abs(self.drift_g) > 15: count += 1
        return count  # 0-5 scale
```

### 5.3 Allostatic Load as Dark Matter Drift Accumulation

Each crisis episode that does not fully recover shifts the baseline:

```
new_baseline_v = old_baseline_v + (crisis_low_v - old_baseline_v) * leak_factor
drift_v += (crisis_low_v - old_baseline_v) * leak_factor
```

Over many episodes:
- `drift_v` accumulates negative (baseline V sinks)
- `drift_a` accumulates positive (resting arousal creeps up)
- `drift_u` accumulates positive (chronic urgency)
- `drift_g` accumulates negative (gravity pulls down)
- `tolerance_*` narrows (window shrinks with each incomplete recovery)

This produces the clinical picture: a traumatized individual with lower resting valence, higher resting arousal, chronic urgency, narrower tolerance for perturbation, and faster escalation through crisis phases.

### 5.4 De-Escalation Techniques as Response VADUG Targeting

TCI co-regulation strategies translate to specific VADUG targets for the **response** vector:

| Technique | Response VADUG Target | Mechanism |
|-----------|----------------------|-----------|
| Calm presence | A: 80-100, D: 128 | Models low arousal, conveys confidence |
| Simple language | U: low, A: low | Reduces cognitive load |
| Active listening | V: 140-160, D: 100-120 | Warmth without dominance |
| Space/time | A: decreasing over time | Allows natural damping |
| Validation | V: 130-150, G: 128 | Affirms without inflating |
| Environmental modification | Setting conditions → tolerance_range | Widens the window |

### 5.5 TCI Four Questions as Dual-VADUG Assessment

| Question | VADUG Operation |
|----------|----------------|
| What am I feeling? | Read own VADUG state |
| What does the child feel/need/want? | Estimate target's VADUG state |
| How is environment affecting this? | Evaluate setting conditions → dark matter |
| How do I best respond? | Calculate optimal response VADUG vector |

---

## 6. Key Formulas and Quantifiable Frameworks

### 6.1 Allostatic Load Score (Count-Based, MacArthur Method)

```
AL_score = sum(1 for biomarker in biomarkers if biomarker in high_risk_quartile)
Range: 0 to N (where N = number of biomarkers, typically 10-14)
High risk = top 25th percentile (or bottom 25th for protective markers)
```

### 6.2 Energetic Model of Allostatic Load (EMAL)

```
health_impact = max(0, ASEE - reserve_capacity)
where:
  ASEE = Additional Stress Energy Expenditure (cumulative)
  reserve_capacity = total_energy_budget - baseline_maintenance
```

### 6.3 Window of Tolerance Width (Conceptual)

```
window_width(dimension) = base_tolerance - (crisis_count * narrowing_rate) + (safety_exposure * expansion_rate)
effective_threshold_upper = baseline + window_width / 2
effective_threshold_lower = baseline - window_width / 2
```

### 6.4 Crisis Trajectory Prediction (from ML literature)

```
P(crisis in next T hours) = f(
    current_vadug_trajectory,    # trend and acceleration
    baseline_drift,              # allostatic load
    time_since_last_crisis,      # refractory period
    setting_conditions,          # environmental risk factors
    window_of_tolerance_width,   # current capacity
    trigger_history              # known vulnerabilities
)
Best published AUC: 0.68-0.88 depending on model and population
```

### 6.5 Resilience as Recovery Dynamics

```
resilience_score = f(
    recovery_rate,          # damping coefficient (how fast return to baseline)
    recovery_completeness,  # 1.0 = full return, <1.0 = residual drift
    rebound_stability,      # oscillation after perturbation
    window_width            # tolerance before dysregulation
)

Connor-Davidson CD-RISC: 0-100 scale (25 items, 0-4 each)
Brief Resilience Scale: 1.0-5.0 mean score
RS-25: 25-175 (25 items, 1-7 each)
```

### 6.6 Polyvagal State Detection (Biomarker)

```
RSA (Respiratory Sinus Arrhythmia) = heart_rate_variability synchronized with respiration
Higher RSA → ventral vagal engagement → within window of tolerance
Lower RSA → sympathetic or dorsal vagal dominance → outside window
```

---

## Sources

- [Therapeutic Crisis Intervention (TCI) - Cornell RCCP](https://rccp.cornell.edu/TCI_LevelOne.html)
- [TCI System Edition 7 Bulletin](https://rccp.cornell.edu/downloads/TCI_7_SYSTEM%20BULLETIN.pdf)
- [Stress Model of Crisis and Behavior Support Techniques - Green Chimneys](https://www.greenchimneys.org/wp-content/uploads/2020/11/Stress-Model-of-Crisis-Behavior-Support-Tech-11.24.20.pptx.pdf)
- [TCI Pocket Guide for Parents - Green Chimneys](https://www.greenchimneys.org/wp-content/uploads/2020/05/TCI-Pocket-Guide.pdf)
- [Therapeutic Crisis Intervention - Roberta M. Roy](https://robertamroy.wordpress.com/2013/08/31/therapeutic-crisis-intervention-cornell-university-tci/)
- [TCI Strategies - KNILT Albany](https://knilt.arcc.albany.edu/Therapeutic_Crisis_Intervention_Strategies)
- [All About TCI - Simply Special Ed](https://www.simplyspecialed.com/all-about-therapeutic-crisis-intervention/)
- [TCI Cheat Sheet - Neurolaunch](https://neurolaunch.com/cheat-sheet-therapeutic-crisis-intervention/)
- [CARE: Creating Conditions for Change - Cornell RCCP](https://rccp.cornell.edu/CARE_LevelOne.html)
- [CARE Program - CEBC](https://www.cebc4cw.org/program/children-and-residential-experiences-care/detailed)
- [CARE at Life Without Barriers](https://www.lwb.org.au/our-approach/child-youth-and-family/care-creating-conditions-for-change/)
- [CARE Program Model Efficacy - Springer Nature](https://link.springer.com/article/10.1007/s11121-016-0649-0)
- [Residential Child Care Project - Cornell Chronicle](https://news.cornell.edu/stories/2019/05/residential-child-care-project-aims-reduce-suffering-responding-it)
- [Window of Tolerance - Psychology Tools](https://www.psychologytools.com/resource/window-of-tolerance)
- [Window of Tolerance - Psychology Today](https://www.psychologytoday.com/us/blog/making-the-whole-beautiful/202205/what-is-the-window-of-tolerance-and-why-is-it-so-important)
- [Window of Tolerance - IPTrauma](https://iptrauma.org/docs/body-of-knowledge-of-psychotraumatology/understanding-the-window-of-tolerance-in-trauma-theory/)
- [Expanding the Window of Tolerance - Positive Psychology](https://positivepsychology.com/window-of-tolerance/)
- [Allostatic Load - Wikipedia](https://en.wikipedia.org/wiki/Allostatic_load)
- [Allostatic Load Impact on Health - PubMed](https://pubmed.ncbi.nlm.nih.gov/32799204/)
- [Energetic Cost of Allostasis - PubMed](https://pubmed.ncbi.nlm.nih.gov/36302295/)
- [Allostatic Load Scoring Variation (NHANES) - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5195908/)
- [Allostatic Load Importance and Markers - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6430278/)
- [Allostatic Load as Cumulative Biological Risk - MacArthur/PNAS](https://www.pnas.org/doi/10.1073/pnas.081072698)
- [Polyvagal Theory - Wikipedia](https://en.wikipedia.org/wiki/Polyvagal_theory)
- [Polyvagal Theory Current Status - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12302812/)
- [Resilience Measurement Scales - Positive Psychology](https://positivepsychology.com/3-resilience-scales/)
- [Connor-Davidson CD-RISC - PubMed](https://pubmed.ncbi.nlm.nih.gov/12964174/)
- [Resilience Scale - The Resilience Center](https://www.resiliencecenter.com/products/resilience-scales-and-tools-for-research/the-resilience-scale-rs/)
- [ML Models for Predicting Mental Health Crises in Adolescents - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12426581/)
- [Predictive Analytics in US Foster Care - CASCW](https://cascw.umn.edu/cw360deg-spring-2025/brief-overview-predictive-analytics-us-foster-care)
- [Life Space Interview I ESCAPE Steps](https://justiceandopportunity.org/wp-content/uploads/2014/09/Life-Space-Interviews-I-ESCAPE.pdf)
- [LSCI - Life Space Crisis Intervention](https://lsci.org/pdf/learn-more/LSCI-Article.pdf)
