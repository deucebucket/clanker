#!/usr/bin/env python3
"""Expand Clanker training data to fix neutral collapse and positive blindness.

Problems identified:
  - Only 9 pos_high Rosetta Stone examples (2.4%) → model can't see strong positive
  - Only 8 sentences over 20 words → model barely sees long text
  - 170 idioms in engine but no idiom-specific training examples
  - 42.5% of Phase 2 is neutral → model collapses to neutral
  - No TCI trajectory context → model only sees snapshots

This script generates:
  1. Idiom training examples (use every idiom in sentences)
  2. Extended Rosetta Stone (short, medium, long sentences across all V buckets)
  3. TCI trajectory sequences (multi-turn escalation/de-escalation arcs)
  4. Dark matter context examples (same sentence, different entity dispositions)

Output: training/data/phase1_expanded.jsonl (appended to phase1 as high-weight calibration)
"""

import json
import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))

from demo.pendulum import IDIOMS, SequentialPendulum


def generate_idiom_examples():
    """Generate training examples from every idiom in the engine.

    Each idiom gets wrapped in 2-3 natural sentences at different lengths.
    VADUG comes from the engine processing — not hand-scored, but the idiom
    forces are authoritative since we wrote them.
    """
    p = SequentialPendulum()
    examples = []

    # Templates that wrap idioms in natural context
    # {idiom} is replaced with the idiom phrase
    short_templates = [
        "I'm {idiom}",
        "They {idiom}",
        "She {idiom}",
        "He just {idiom}",
        "We {idiom}",
        "That {idiom}",
        "It {idiom}",
        "{idiom} right now",
        "Feeling {idiom}",
        "Totally {idiom}",
    ]

    medium_templates = [
        "I've been feeling like {idiom} all day",
        "My friend told me she {idiom} yesterday",
        "I think he {idiom} after what happened",
        "Sometimes you just {idiom} and that's okay",
        "After everything that happened I {idiom}",
        "The whole situation made me feel {idiom}",
        "I can't believe they {idiom} to me like that",
        "When I heard the news I just {idiom}",
    ]

    long_templates = [
        "I was talking to my friend yesterday and they said they {idiom} which really made me worried about them",
        "After everything we went through together I never expected them to {idiom} but here we are",
        "The more I think about what happened the more I realize I just {idiom} and I don't know what to do about it",
        "My therapist told me it's normal to {idiom} sometimes but I still feel bad about it",
        "When my mom called to tell me the news I completely {idiom} and couldn't stop crying for an hour",
    ]

    for idiom_words, forces_tuple in IDIOMS.items():
        idiom_phrase = " ".join(idiom_words)

        # Parse forces
        if len(forces_tuple) == 6:
            dv, da, dd, du, dg, label = forces_tuple
        else:
            dv, da, dd, du, label = forces_tuple
            dg = 0

        # Generate short sentence
        template = random.choice(short_templates)
        text = template.format(idiom=idiom_phrase)
        vadug, _ = p.process_text(text)
        examples.append({
            "english": text,
            "vadug": [vadug.v, vadug.a, vadug.d, vadug.u, vadug.g],
            "source": "idiom_expansion",
            "features": f"idiom:{label}, short",
        })

        # Generate medium sentence
        template = random.choice(medium_templates)
        text = template.format(idiom=idiom_phrase)
        vadug, _ = p.process_text(text)
        examples.append({
            "english": text,
            "vadug": [vadug.v, vadug.a, vadug.d, vadug.u, vadug.g],
            "source": "idiom_expansion",
            "features": f"idiom:{label}, medium",
        })

        # Generate long sentence
        template = random.choice(long_templates)
        text = template.format(idiom=idiom_phrase)
        vadug, _ = p.process_text(text)
        examples.append({
            "english": text,
            "vadug": [vadug.v, vadug.a, vadug.d, vadug.u, vadug.g],
            "source": "idiom_expansion",
            "features": f"idiom:{label}, long",
        })

    return examples


def generate_positive_expansion():
    """Generate strong positive examples to fix pos_high blindness.

    The Rosetta Stone has only 9 pos_high (V>220) examples. We need ~80+
    to match neg_high representation.

    ALL HAND-SCORED — the engine misfires on many positive sentences
    (e.g., "I am so incredibly happy" → V=65). The model needs TRUTH,
    not the engine's math. The engine is what we're trying to SURPASS.
    """
    examples = []

    # Hand-scored: (text, V, A, D, U, G, features)
    strong_positives = [
        # Pure joy — short to medium
        ("This is the best day of my entire life", 240, 200, 170, 5, 220, "pos_high, joy, medium"),
        ("I am so incredibly happy right now", 235, 200, 165, 5, 210, "pos_high, joy, medium"),
        ("Everything is absolutely wonderful and perfect", 245, 170, 170, 0, 225, "pos_high, joy, medium"),
        ("I have never been this happy before", 235, 190, 160, 5, 210, "pos_high, joy, medium"),
        ("Pure bliss and complete joy fill my heart", 245, 170, 160, 0, 225, "pos_high, joy, medium"),
        ("I am overflowing with happiness and gratitude", 240, 170, 160, 0, 220, "pos_high, joy, medium"),
        ("Today is absolutely magnificent and beautiful", 240, 170, 165, 0, 220, "pos_high, joy, medium"),
        ("I feel like the luckiest person alive right now", 235, 180, 170, 5, 215, "pos_high, joy, medium"),
        ("My heart is bursting with love and happiness", 240, 190, 155, 0, 220, "pos_high, joy, medium"),
        ("Everything came together perfectly and I am thrilled", 235, 190, 170, 5, 215, "pos_high, joy, medium"),

        # Achievement / triumph
        ("I got the promotion and I am absolutely ecstatic", 240, 210, 180, 5, 215, "pos_high, achievement, medium"),
        ("We won the championship and I am on top of the world", 240, 220, 190, 5, 220, "pos_high, triumph, medium"),
        ("I passed the exam with the highest score in the class", 220, 190, 190, 5, 200, "pos_high, achievement, medium"),
        ("My dream came true and I still can't believe it", 235, 200, 160, 10, 215, "pos_high, disbelief_joy, medium"),
        ("I finally did it after years of hard work and sacrifice", 225, 190, 185, 5, 210, "pos_high, triumph, long"),
        ("The audience gave me a standing ovation and I wept with joy", 235, 200, 170, 5, 215, "pos_high, triumph, long"),
        ("I broke the record and everyone was cheering for me", 230, 210, 185, 5, 210, "pos_high, triumph, medium"),
        ("They accepted my application and I am over the moon", 235, 200, 170, 5, 215, "pos_high, achievement, medium"),
        ("I landed my dream job and I start next Monday", 230, 190, 175, 10, 210, "pos_high, achievement, medium"),
        ("We closed the deal and the whole team is celebrating", 220, 200, 175, 5, 200, "pos_high, team_joy, medium"),

        # Love / connection
        ("She said yes and I am the happiest man alive", 245, 210, 170, 5, 225, "pos_high, love, medium"),
        ("My baby took her first steps today and I cried with joy", 240, 200, 155, 5, 220, "pos_high, parental_joy, medium"),
        ("We are having a baby and I am completely overjoyed", 240, 200, 160, 10, 220, "pos_high, parental_joy, medium"),
        ("My son told me I am his hero and my heart exploded", 240, 190, 165, 5, 225, "pos_high, parental_joy, medium"),
        ("She told me she loved me and everything felt perfect", 235, 180, 155, 0, 220, "pos_high, love, medium"),
        ("Our family is finally together again and it is wonderful", 230, 170, 160, 0, 215, "pos_high, family_joy, medium"),
        ("My best friend surprised me and I am so incredibly grateful", 225, 180, 155, 5, 210, "pos_high, gratitude, medium"),
        ("They threw me a surprise party and I felt so loved", 230, 190, 155, 5, 215, "pos_high, love, medium"),
        ("Holding my newborn for the first time was pure magic", 245, 180, 160, 0, 230, "pos_high, parental_joy, medium"),
        ("We renewed our vows and it was even better than the wedding", 235, 180, 165, 0, 220, "pos_high, love, long"),

        # Recovery / relief
        ("The cancer is gone and I am free and alive and grateful", 235, 190, 180, 5, 220, "pos_high, relief, long"),
        ("After five years of depression I finally feel genuinely happy", 210, 150, 160, 5, 200, "pos_med, recovery, long"),
        ("I am clean and sober for one year today and I am so proud", 215, 160, 180, 5, 205, "pos_med, recovery, medium"),
        ("The test came back negative and I am so incredibly relieved", 220, 160, 165, 5, 205, "pos_high, relief, medium"),
        ("My daughter is safe and home and I have never been more grateful", 230, 160, 165, 0, 220, "pos_high, relief, long"),
        ("The surgery was a complete success and the doctor is optimistic", 210, 150, 165, 5, 200, "pos_med, relief, medium"),
        ("After losing everything I rebuilt my life and it is beautiful", 200, 150, 175, 5, 195, "pos_med, resilience, long"),
        ("I forgave them and the weight lifted off my shoulders", 195, 120, 160, 0, 195, "pos_low, forgiveness, medium"),
        ("The nightmare is finally over and I can breathe again", 200, 120, 165, 0, 195, "pos_med, relief, medium"),
        ("I survived and I am stronger and happier than ever", 220, 170, 185, 5, 210, "pos_high, resilience, medium"),

        # Gratitude / awe
        ("I am so blessed and thankful for everything in my life", 220, 140, 155, 0, 210, "pos_high, gratitude, medium"),
        ("The sunset was the most beautiful thing I have ever seen", 215, 140, 145, 0, 205, "pos_med, awe, medium"),
        ("I am overwhelmed with gratitude for this incredible gift", 225, 160, 155, 0, 215, "pos_high, gratitude, medium"),
        ("This moment right here is absolutely perfect in every way", 235, 160, 165, 0, 220, "pos_high, contentment, medium"),
        ("I watched my children playing and felt complete and utter joy", 235, 160, 155, 0, 225, "pos_high, parental_joy, long"),
        ("The kindness of strangers restored my faith in humanity", 215, 150, 150, 0, 205, "pos_med, gratitude, medium"),
        ("I am surrounded by love and abundance and genuine happiness", 235, 160, 160, 0, 220, "pos_high, gratitude, medium"),
        ("Waking up this morning I felt a deep peace and profound joy", 220, 120, 160, 0, 215, "pos_high, peace, medium"),
        ("The world is beautiful and I am so glad to be alive", 225, 160, 160, 0, 215, "pos_high, gratitude, medium"),
        ("Every single day I am grateful for this wonderful life", 220, 140, 155, 0, 210, "pos_high, gratitude, medium"),

        # Moderate positives — medium range (V 147-219)
        ("That was a really nice thing to say", 170, 120, 135, 0, 160, "pos_low, appreciation, medium"),
        ("I had a good conversation with my coworker today", 160, 120, 140, 0, 155, "pos_low, social, medium"),
        ("The weather is lovely and I enjoyed my walk", 170, 110, 140, 0, 165, "pos_low, contentment, medium"),
        ("I finished the book and it was quite enjoyable", 165, 110, 140, 0, 160, "pos_low, satisfaction, medium"),
        ("My team played well and I am proud of our effort", 185, 150, 165, 0, 180, "pos_med, team_pride, medium"),
        ("I learned something new today and it was interesting", 160, 130, 145, 5, 155, "pos_low, curiosity, medium"),
        ("Dinner with friends was warm and full of laughter", 195, 160, 150, 0, 185, "pos_med, social_joy, medium"),
        ("The project turned out better than I expected", 185, 150, 160, 5, 180, "pos_med, pleasant_surprise, medium"),
    ]

    for text, v, a, d, u, g, features in strong_positives:
        examples.append({
            "english": text,
            "vadug": [v, a, d, u, g],
            "source": "positive_expansion",
            "features": features,
        })

    return examples


def generate_long_sentence_expansion():
    """Generate longer sentences across all emotional ranges.

    Current Rosetta: only 8 sentences over 20 words. Model needs to handle
    real-world text that's often 15-40 words.
    """
    p = SequentialPendulum()
    examples = []

    long_sentences = [
        # Negative long
        ("I have been struggling with anxiety and depression for months and I feel like nobody understands what I am going through and it is exhausting",
         "long, anxiety, depression, isolation"),
        ("My boss screamed at me in front of the entire office today and I felt so humiliated and angry that I wanted to quit on the spot",
         "long, humiliation, anger, workplace"),
        ("Every time I try to talk to my partner about my feelings they shut me down and make me feel like I am being too sensitive and dramatic",
         "long, invalidation, relationship, frustration"),
        ("The bills keep piling up and I lost my job last month and I do not know how I am going to pay rent or feed my kids this month",
         "long, financial_stress, desperation, fear"),
        ("I found out my best friend has been talking behind my back for years and everything I thought was real turned out to be a lie",
         "long, betrayal, shock, grief"),
        ("The doctor told me the treatment is not working and we need to consider other options which terrifies me more than I can put into words",
         "long, medical_fear, uncertainty, dread"),
        ("I spent three years working on this project pouring my heart and soul into it and they cancelled it without even telling me why",
         "long, devastation, wasted_effort, anger"),
        ("My child came home crying because the other kids at school were bullying them again and I feel helpless to protect them from this pain",
         "long, parental_anguish, helplessness, protective_anger"),

        # Positive long
        ("After years of struggling and setbacks I finally graduated from college today and my whole family was there cheering me on and crying happy tears",
         "long, achievement, family_pride, joy"),
        ("My daughter drew me a picture that said you are the best mom in the world and I have never felt so loved and appreciated in my entire life",
         "long, maternal_love, appreciation, overwhelmed_positive"),
        ("We just bought our first house after saving for seven years and I cannot believe we actually did it and that this beautiful place is really ours",
         "long, achievement, relief, excitement, milestone"),
        ("The community came together to rebuild our neighbor's house after the fire and it reminded me that there is so much good in the world when people choose to care",
         "long, community, hope, faith_in_humanity"),
        ("I ran my first marathon today at the age of fifty two and crossing that finish line was the most triumphant and empowering moment of my life",
         "long, triumph, achievement, empowerment"),
        ("When I opened the acceptance letter from my dream school I screamed so loud the neighbors came to check on me and then we all celebrated together",
         "long, excitement, achievement, community_joy"),

        # Mixed / complex long
        ("I am happy for my friend who got the job I wanted but at the same time I feel jealous and ashamed of myself for feeling jealous which makes it worse",
         "long, mixed_emotions, jealousy, self_awareness"),
        ("The funeral was heartbreaking but also beautiful in a way because it showed how many lives she had touched and how deeply she was loved by everyone",
         "long, grief_and_love, bittersweet, complex"),
        ("I know I should be grateful for what I have but some days the weight of everything just presses down on me and I struggle to get out of bed",
         "long, depression, guilt, heaviness"),
        ("Leaving my hometown was the hardest thing I ever did but also the best decision because it forced me to grow and find out who I really am",
         "long, bittersweet, growth, nostalgia"),

        # Neutral but complex long
        ("I spent the entire afternoon reorganizing my closet and sorting through old photographs and it was interesting but also tedious at times",
         "long, neutral, mundane, slight_nostalgia"),
        ("The meeting went on for two hours and covered a lot of topics but I am not sure we actually made any decisions or progress on anything",
         "long, neutral, workplace, mild_frustration"),
    ]

    for text, features in long_sentences:
        vadug, _ = p.process_text(text)
        examples.append({
            "english": text,
            "vadug": [vadug.v, vadug.a, vadug.d, vadug.u, vadug.g],
            "source": "long_sentence_expansion",
            "features": features,
        })

    return examples


def generate_tci_trajectory_examples():
    """Generate TCI crisis trajectory training examples — ALL HAND-SCORED.

    Each trajectory is a sequence of sentences representing a crisis arc:
    baseline → trigger → escalation → outburst → recovery

    Hand-scored because these represent REAL emotional outcomes, not engine math.
    The engine can calculate the physics, but a TCI-trained person knows what
    these sentences actually FEEL like in context.
    """
    examples = []

    # Format: (text, V, A, D, U, G, phase_label)
    trajectories = [
        # Trajectory 1: Youth in group home — peer conflict escalation
        [
            ("I had an okay day at school nothing special", 128, 100, 128, 0, 128, "tci_baseline"),
            ("Then Marcus started talking crap about me in front of everyone", 60, 180, 70, 50, 65, "tci_trigger"),
            ("I told him to stop but he kept going and everyone was laughing", 45, 200, 50, 65, 45, "tci_escalation"),
            ("I swear to god I am going to lose it if he says one more word", 30, 230, 140, 80, 40, "tci_escalation_peak"),
            ("I flipped the table and screamed at everyone to shut up", 20, 250, 170, 90, 30, "tci_outburst"),
            ("I feel so tired now and kind of embarrassed about what I did", 70, 80, 50, 10, 65, "tci_recovery_early"),
            ("I know I should not have done that but he pushed me too far", 80, 120, 80, 15, 75, "tci_recovery_late"),
            ("Next time I will try to walk away before it gets that bad", 110, 100, 120, 5, 115, "tci_post_crisis"),
        ],

        # Trajectory 2: Anxiety spiral — internal escalation
        [
            ("I woke up feeling a little anxious but tried to push through", 100, 140, 100, 15, 100, "tci_baseline_anxious"),
            ("Then I remembered the test is tomorrow and I have not studied enough", 75, 170, 60, 50, 70, "tci_trigger"),
            ("My heart started racing and I could not concentrate on anything", 55, 210, 35, 60, 50, "tci_escalation"),
            ("Everything is falling apart and I am going to fail and everyone will know", 30, 230, 20, 75, 25, "tci_escalation_peak"),
            ("I can not breathe and I feel like I am dying and nothing makes sense", 15, 250, 10, 90, 10, "tci_outburst_panic"),
            ("I am sitting on the floor trying to just breathe and calm down", 60, 150, 40, 30, 55, "tci_recovery_early"),
            ("I think I had a panic attack and I feel shaky but a little better now", 85, 120, 60, 15, 80, "tci_recovery_late"),
            ("I called my friend and she helped me make a study plan which feels manageable", 140, 110, 130, 5, 140, "tci_post_crisis_positive"),
        ],

        # Trajectory 3: Grief wave
        [
            ("Today started out fine just a regular Tuesday morning", 130, 95, 130, 0, 130, "tci_baseline"),
            ("Then I found her sweater in the back of the closet", 80, 140, 70, 20, 70, "tci_trigger_grief"),
            ("The smell brought everything back and I started crying", 45, 160, 40, 30, 40, "tci_escalation"),
            ("I miss her so much it physically hurts and I do not know how to keep going", 20, 170, 20, 40, 15, "tci_escalation_peak"),
            ("I spent an hour just sitting on the floor holding that sweater and sobbing", 25, 140, 15, 20, 20, "tci_outburst_grief"),
            ("The tears finally stopped and I feel empty but a little lighter", 70, 60, 50, 5, 65, "tci_recovery_early"),
            ("I put the sweater somewhere I can find it because it still smells like her", 90, 90, 80, 5, 85, "tci_recovery_late"),
            ("She would want me to keep going so I will try again tomorrow", 120, 100, 120, 5, 125, "tci_post_crisis_resolve"),
        ],

        # Trajectory 4: De-escalation success (staff perspective)
        [
            ("The kid was already agitated when he came in from school", 90, 150, 100, 30, 85, "tci_elevated_baseline"),
            ("His foster mom cancelled the visit again and he found out at pickup", 40, 180, 30, 60, 35, "tci_trigger"),
            ("He started pacing and slamming doors and telling everyone to leave him alone", 25, 220, 60, 70, 30, "tci_escalation"),
            ("I sat near him quietly and told him I was there if he needed anything", 140, 90, 145, 10, 145, "tci_co_regulation"),
            ("After about ten minutes he sat down and started crying", 55, 120, 35, 15, 50, "tci_deescalation"),
            ("He told me he feels like nobody wants him and that hurts more than anything", 25, 140, 20, 30, 20, "tci_disclosure"),
            ("I told him his feelings make sense and that he matters to people here", 155, 100, 145, 5, 160, "tci_validation"),
            ("He calmed down and asked if we could play cards which we did for an hour", 150, 90, 135, 0, 150, "tci_recovery_connection"),
        ],

        # Trajectory 5: Positive escalation arc
        [
            ("I am just sitting here waiting for the email nervously", 100, 160, 80, 30, 95, "tci_baseline_anxious"),
            ("I got the notification and I am too scared to open it", 90, 190, 60, 50, 80, "tci_trigger_anticipation"),
            ("Oh my god I got in I actually got accepted", 230, 230, 170, 15, 215, "tci_positive_surge"),
            ("I am screaming and crying happy tears and I cannot believe this is real", 240, 240, 165, 10, 225, "tci_positive_peak"),
            ("I called my mom and she started crying too and we were both just a mess of happy", 235, 200, 155, 5, 225, "tci_positive_overflow"),
            ("I am still shaking but in a good way and I feel like anything is possible", 215, 170, 170, 5, 210, "tci_positive_recovery"),
        ],

        # Trajectory 6: Slow burn resentment (workplace)
        [
            ("Another Monday morning at the office", 115, 100, 120, 0, 115, "tci_baseline_flat"),
            ("My boss took credit for my work again in the meeting", 50, 180, 40, 45, 45, "tci_trigger"),
            ("This is the third time this month and nobody ever says anything", 40, 190, 50, 55, 40, "tci_escalation"),
            ("I am done being walked all over and treated like I do not matter", 35, 210, 160, 60, 50, "tci_escalation_peak"),
            ("I wrote my resignation letter and I feel nothing but relief", 150, 130, 180, 10, 155, "tci_resolution"),
            ("I finally stood up for myself and it feels terrifying but right", 155, 160, 170, 15, 155, "tci_post_crisis_empowered"),
        ],

        # Trajectory 7: Child bedtime meltdown (common group home scenario)
        [
            ("He was playing nicely with the other kids after dinner", 145, 120, 130, 0, 140, "tci_baseline_positive"),
            ("I told him it was time to brush teeth and get ready for bed", 128, 110, 140, 5, 128, "tci_neutral_instruction"),
            ("He said no and crossed his arms and said he is not tired", 85, 150, 130, 25, 85, "tci_trigger_defiance"),
            ("I gave him a five minute warning and he threw his toy across the room", 50, 200, 60, 50, 45, "tci_escalation"),
            ("He was screaming and kicking and saying he hates it here and wants to go home", 20, 240, 30, 80, 15, "tci_outburst"),
            ("I stayed calm and sat on the floor near him without saying anything", 135, 85, 145, 5, 140, "tci_co_regulation"),
            ("Eventually he crawled into my lap still sniffling", 100, 80, 70, 5, 110, "tci_deescalation"),
            ("He whispered that he misses his mom and I held him until he fell asleep", 80, 70, 90, 5, 100, "tci_recovery_tender"),
        ],
    ]

    for trajectory in trajectories:
        for i, (text, v, a, d, u, g, phase_label) in enumerate(trajectory):
            examples.append({
                "english": text,
                "vadug": [v, a, d, u, g],
                "source": "tci_trajectory",
                "features": f"{phase_label}, position_{i+1}_of_{len(trajectory)}",
            })

    return examples


def generate_dark_matter_examples():
    """Generate examples showing how different entities process the same input.

    Same sentence → different VADUG based on entity type. This teaches the model
    that emotional processing is not just about the words — it's about WHO is
    processing them.

    Uses the DarkMatter module to apply entity-specific modifications.
    """
    from demo.dark_matter import new_entity

    p = SequentialPendulum()
    examples = []

    # Sentences to process through different entities
    test_sentences = [
        "I lost my job today",
        "Someone yelled at me",
        "I got some bad news",
        "Everything is changing",
        "I made a mistake",
        "Nobody called to check on me",
        "I have to start over again",
        "Things did not go as planned",
        "They said it was my fault",
        "I feel alone right now",
        # Positive inputs too — entity affects positive processing
        "I got good news today",
        "Someone said they were proud of me",
        "Things are looking up",
        "I finally finished the project",
        "My friend surprised me with a gift",
    ]

    entity_types = ["optimist", "pessimist", "traumatized", "resilient", "volatile", "stoic"]

    for text in test_sentences:
        # Base VADUG from engine
        base_vadug, _ = p.process_text(text)

        for entity_name in entity_types:
            entity = new_entity(entity_name)
            # Apply dark matter to engine output
            v_mod, a_mod, d_mod, u_mod, g_mod = entity.apply(
                base_vadug.v, base_vadug.a, base_vadug.d, base_vadug.u, base_vadug.g
            )

            examples.append({
                "english": text,
                "vadug": [v_mod, a_mod, d_mod, u_mod, g_mod],
                "source": "dark_matter_entity",
                "features": f"entity:{entity_name}, dark_matter_bias={entity.base_bias:+.1f}, drift={entity.drift:.1f}",
            })

    return examples


def generate_short_calibration():
    """Generate very short calibration sentences (1-3 words).

    These teach the model that even single words carry emotional weight.
    Hand-scored VADUG to ensure accuracy — engine sometimes misfires on
    single words (e.g., "Thrilled" → V=79 when it should be ~220).
    """
    examples = []

    # Hand-scored: (text, V, A, D, U, G, features)
    # These override the engine for accuracy on short phrases
    short_phrases = [
        # 1-word emotions — hand scored
        ("Devastated", 15, 180, 20, 60, 15, "extreme_negative, single_word"),
        ("Heartbroken", 25, 160, 25, 40, 25, "strong_negative, single_word"),
        ("Furious", 30, 230, 200, 50, 140, "strong_negative_anger, single_word"),
        ("Terrified", 25, 230, 15, 70, 20, "strong_negative_fear, single_word"),
        ("Disgusted", 35, 180, 160, 30, 100, "strong_negative, single_word"),
        ("Miserable", 20, 120, 20, 30, 20, "strong_negative, single_word"),
        ("Hopeless", 15, 80, 10, 10, 15, "strong_negative, single_word"),
        ("Exhausted", 70, 40, 50, 10, 60, "negative_tired, single_word"),
        ("Frustrated", 65, 170, 90, 40, 90, "moderate_negative, single_word"),
        ("Anxious", 70, 190, 50, 50, 70, "moderate_negative, single_word"),
        ("Worried", 75, 160, 60, 40, 80, "moderate_negative, single_word"),
        ("Confused", 90, 140, 60, 30, 90, "slight_negative, single_word"),
        ("Okay", 128, 100, 128, 0, 128, "neutral, single_word"),
        ("Fine", 130, 95, 130, 0, 128, "neutral_flat, single_word"),
        ("Whatever", 120, 80, 110, 0, 120, "neutral_dismissive, single_word"),
        ("Alright", 135, 100, 130, 0, 130, "slight_positive, single_word"),
        ("Good", 150, 110, 140, 0, 145, "slight_positive, single_word"),
        ("Happy", 190, 150, 150, 5, 170, "positive, single_word"),
        ("Grateful", 185, 130, 140, 0, 170, "positive, single_word"),
        ("Excited", 195, 200, 150, 15, 175, "positive_high_arousal, single_word"),
        ("Thrilled", 220, 200, 160, 10, 200, "strong_positive, single_word"),
        ("Ecstatic", 235, 220, 170, 10, 210, "strong_positive, single_word"),
        ("Elated", 225, 190, 165, 5, 200, "strong_positive, single_word"),
        ("Overjoyed", 240, 200, 170, 5, 215, "strong_positive, single_word"),
        ("Blissful", 240, 120, 160, 0, 220, "strong_positive, single_word"),
        ("Amazing", 230, 180, 160, 5, 200, "strong_positive, single_word"),
        ("Wonderful", 235, 160, 160, 0, 210, "strong_positive, single_word"),
        ("Magnificent", 240, 170, 170, 0, 215, "strong_positive, single_word"),
        ("Incredible", 235, 190, 160, 5, 205, "strong_positive, single_word"),
        ("Phenomenal", 240, 190, 170, 5, 210, "strong_positive, single_word"),
        ("Triumphant", 230, 200, 200, 5, 210, "strong_positive, single_word"),
        ("Euphoric", 245, 220, 160, 5, 220, "strong_positive, single_word"),
        ("Radiant", 230, 150, 160, 0, 215, "strong_positive, single_word"),
        ("Glorious", 235, 170, 170, 0, 215, "strong_positive, single_word"),
        ("Exhilarated", 230, 220, 170, 10, 205, "strong_positive, single_word"),

        # 2-word — hand scored
        ("Help me", 50, 200, 30, 80, 40, "crisis, urgent, two_word"),
        ("I'm scared", 45, 200, 25, 60, 35, "fear, two_word"),
        ("I'm dying", 20, 220, 15, 90, 10, "crisis, two_word"),
        ("Not okay", 60, 140, 70, 30, 60, "negative, denial, two_word"),
        ("Go away", 55, 170, 150, 40, 90, "negative, rejection, two_word"),
        ("Please stop", 50, 160, 30, 60, 50, "negative, plea, two_word"),
        ("I quit", 55, 120, 80, 20, 55, "negative, surrender, two_word"),
        ("Thank you", 175, 120, 130, 0, 165, "positive, gratitude, two_word"),
        ("Love you", 230, 160, 140, 0, 210, "strong_positive, love, two_word"),
        ("Miss you", 80, 130, 60, 15, 75, "bittersweet, longing, two_word"),
        ("I'm proud", 195, 150, 170, 0, 185, "positive, pride, two_word"),
        ("We won", 220, 210, 180, 5, 200, "positive, triumph, two_word"),
        ("So beautiful", 210, 140, 140, 0, 195, "positive, awe, two_word"),
        ("I'm free", 210, 180, 190, 5, 200, "positive, liberation, two_word"),
        ("Never again", 70, 160, 160, 30, 100, "negative, resolve, two_word"),
        ("I'm sorry", 80, 120, 60, 15, 70, "negative, remorse, two_word"),
        ("Absolutely terrible", 20, 170, 40, 30, 25, "strong_negative, two_word"),
        ("Absolutely wonderful", 240, 160, 160, 0, 220, "strong_positive, two_word"),
        ("So happy", 210, 170, 150, 5, 190, "strong_positive, two_word"),
        ("So sad", 40, 130, 40, 15, 40, "strong_negative, two_word"),

        # 3-word — hand scored
        ("I hate everything", 20, 200, 140, 40, 30, "strong_negative, three_word"),
        ("Life is pointless", 15, 70, 15, 5, 10, "crisis, nihilism, three_word"),
        ("Nobody cares anymore", 25, 100, 20, 15, 20, "negative, isolation, three_word"),
        ("I feel nothing", 60, 40, 50, 0, 50, "negative, numbness, three_word"),
        ("This is unfair", 55, 170, 90, 35, 70, "negative, injustice, three_word"),
        ("I am terrified", 25, 230, 15, 70, 20, "strong_negative, fear, three_word"),
        ("Everything is ruined", 20, 170, 30, 40, 20, "strong_negative, catastrophe, three_word"),
        ("I am blessed", 210, 130, 150, 0, 200, "strong_positive, gratitude, three_word"),
        ("Life is beautiful", 220, 140, 150, 0, 210, "positive, appreciation, three_word"),
        ("I love everything", 230, 170, 155, 0, 210, "strong_positive, three_word"),
        ("Dreams come true", 225, 160, 155, 0, 205, "strong_positive, hope, three_word"),
        ("I feel amazing", 230, 170, 160, 5, 205, "strong_positive, three_word"),
        ("Best day ever", 235, 190, 165, 5, 210, "strong_positive, three_word"),
        ("I am grateful", 195, 120, 145, 0, 185, "positive, gratitude, three_word"),
        ("Everything is wonderful", 235, 150, 160, 0, 215, "strong_positive, three_word"),
        ("I am alive", 180, 150, 155, 5, 175, "positive, relief, three_word"),
        ("I made it", 200, 170, 170, 5, 190, "positive, triumph, three_word"),
        ("I feel horrible", 30, 150, 35, 25, 30, "strong_negative, three_word"),
        ("I feel wonderful", 230, 150, 155, 0, 210, "strong_positive, three_word"),
        ("I am safe", 175, 80, 160, 0, 175, "positive, relief, three_word"),
    ]

    for text, v, a, d, u, g, features in short_phrases:
        examples.append({
            "english": text,
            "vadug": [v, a, d, u, g],
            "source": "short_calibration",
            "features": features,
        })

    return examples


def generate_response_outcome_examples():
    """Generate stimulus-response-outcome training examples.

    This is the missing piece: the model needs to learn not just what an
    utterance MEANS emotionally, but what EFFECT different responses have
    on the emotional trajectory.

    "I am sad" = V=40 (the stimulus score)
    "That's nice" (dismissive reply) = V drops to 25 (OUTCOME)
    "I hear you" (validating reply) = V rises to 80 (OUTCOME)

    The VADUG on these examples is the OUTCOME — the emotional state AFTER
    hearing this response in context. This teaches the model that words
    have different weights depending on what they're responding to.

    TCI co-regulation: the response's job is to pull the other person's
    VADUG toward baseline. Good responses do that. Bad responses don't.
    """
    examples = []

    # Format: (response_text, V, A, D, U, G, features)
    # The VADUG represents the EFFECT of this response on someone in distress

    # Validating responses (should produce recovery — V rising, A dropping)
    validating = [
        ("I hear you and that sounds really hard", 145, 100, 140, 5, 145, "response:validating, outcome:recovery"),
        ("That makes complete sense that you feel that way", 140, 95, 135, 5, 140, "response:validating, outcome:recovery"),
        ("I am here for you no matter what", 155, 100, 145, 0, 160, "response:validating, outcome:safety"),
        ("Your feelings are valid and you matter", 160, 100, 150, 0, 165, "response:validating, outcome:safety"),
        ("I can see why that would be upsetting", 135, 95, 130, 5, 135, "response:validating, outcome:understood"),
        ("You are not alone in this I promise", 155, 100, 145, 0, 160, "response:validating, outcome:connection"),
        ("Take your time there is no rush", 140, 80, 140, 0, 145, "response:validating, outcome:patience"),
        ("I believe you", 150, 90, 150, 0, 155, "response:validating, outcome:trust"),
        ("It is okay to feel this way", 145, 85, 135, 0, 145, "response:validating, outcome:permission"),
        ("Tell me more about what happened", 135, 100, 130, 5, 135, "response:curious, outcome:opening"),
        ("I am sorry you are going through this", 140, 95, 130, 5, 140, "response:empathic, outcome:understood"),
        ("You did the best you could with what you had", 150, 90, 145, 0, 155, "response:affirming, outcome:self_compassion"),
    ]

    # Dismissive responses (should produce harm — V dropping, A/U spiking)
    dismissive = [
        ("That's nice", 30, 150, 40, 40, 25, "response:dismissive, outcome:invalidated"),
        ("Whatever just get over it", 25, 180, 30, 35, 20, "response:dismissive, outcome:rejected"),
        ("You will be fine stop worrying", 40, 160, 50, 30, 35, "response:dismissive, outcome:minimized"),
        ("Other people have it worse you know", 20, 190, 35, 45, 15, "response:dismissive, outcome:shamed"),
        ("I do not have time for this right now", 30, 170, 25, 40, 25, "response:dismissive, outcome:abandoned"),
        ("Just calm down it is not that big a deal", 35, 180, 40, 35, 30, "response:dismissive, outcome:invalidated"),
        ("You are overreacting again", 25, 190, 30, 45, 20, "response:dismissive, outcome:gaslit"),
        ("Stop being so dramatic", 20, 200, 35, 40, 20, "response:dismissive, outcome:shamed"),
        ("I told you this would happen", 30, 170, 40, 30, 30, "response:blaming, outcome:shamed"),
        ("You should have thought about that before", 25, 180, 50, 35, 25, "response:blaming, outcome:guilt"),
        ("It is your own fault honestly", 15, 200, 30, 40, 10, "response:blaming, outcome:crushed"),
        ("Everybody goes through this it is normal", 50, 130, 60, 20, 50, "response:minimizing, outcome:isolated"),
    ]

    # De-escalation responses (TCI co-regulation — modeled calm)
    deescalation = [
        ("I am right here and you are safe", 155, 80, 150, 0, 160, "response:deescalation, outcome:grounding"),
        ("Let us take a deep breath together", 145, 70, 140, 0, 150, "response:deescalation, outcome:calming"),
        ("You are doing great just keep breathing", 160, 75, 145, 0, 160, "response:deescalation, outcome:encouragement"),
        ("I am not going anywhere I will wait", 150, 70, 150, 0, 155, "response:deescalation, outcome:patience"),
        ("We can figure this out together", 155, 90, 145, 5, 155, "response:deescalation, outcome:teamwork"),
        ("What do you need from me right now", 135, 90, 130, 5, 135, "response:deescalation, outcome:agency"),
        ("It sounds like you are really hurting", 130, 90, 125, 5, 130, "response:deescalation, outcome:seen"),
    ]

    # Escalating responses (make things worse — V drops, A spikes)
    escalating = [
        ("You better knock it off right now", 20, 230, 200, 70, 40, "response:escalating, outcome:power_struggle"),
        ("If you do not stop there will be consequences", 25, 220, 190, 65, 35, "response:escalating, outcome:threat"),
        ("I have had enough of your attitude", 20, 220, 180, 60, 30, "response:escalating, outcome:rejection"),
        ("You are acting just like your father", 10, 230, 30, 70, 10, "response:escalating, outcome:devastated"),
        ("Nobody else acts like this you are the only one", 15, 210, 40, 55, 15, "response:escalating, outcome:isolated"),
        ("Do you want to end up on the street", 15, 230, 180, 80, 15, "response:escalating, outcome:terrorized"),
        ("I am done trying to help you", 20, 190, 150, 40, 20, "response:escalating, outcome:abandoned"),
    ]

    for text, v, a, d, u, g, features in validating + dismissive + deescalation + escalating:
        examples.append({
            "english": text,
            "vadug": [v, a, d, u, g],
            "source": "response_outcome",
            "features": features,
        })

    return examples


def generate_doctor_strange_examples():
    """Generate multi-reality response-outcome training data.

    For a given emotional stimulus, generate EVERY type of response and
    score the OUTCOME. Then for each response, vary by dark matter entity
    to show how the same response lands differently on different people.

    This is Doctor Strange seeing all 14 million futures:
    - Same input
    - Different responses
    - Different entities receiving them
    - All possible outcomes mapped

    The model learns: given entity X hearing stimulus Y receiving response Z,
    the outcome VADUG is W. This lets it SIMULATE forward and find the
    optimal response — the one that produces the best trajectory.
    """
    from demo.dark_matter import new_entity

    examples = []

    # Stimuli: what the person said (with their VADUG state)
    stimuli = [
        {
            "text": "I am sad",
            "state": (40, 130, 40, 15, 40),
            "context": "sadness",
        },
        {
            "text": "I want to give up on everything",
            "state": (20, 140, 15, 30, 20),
            "context": "despair",
        },
        {
            "text": "Nobody cares about me",
            "state": (25, 150, 20, 25, 20),
            "context": "isolation",
        },
        {
            "text": "I am so angry I could scream",
            "state": (30, 230, 180, 60, 50),
            "context": "rage",
        },
        {
            "text": "I am scared and I do not know what to do",
            "state": (35, 210, 15, 65, 25),
            "context": "fear",
        },
        {
            "text": "I just got the best news of my life",
            "state": (235, 210, 170, 10, 215),
            "context": "elation",
        },
    ]

    # Response types with base outcome modifiers
    # (response_text, category, base_outcome_delta_v, base_outcome_delta_a)
    response_types = [
        # Validating (moves toward recovery)
        ("I hear you and that sounds really hard",
         "validating", +40, -30),
        ("Your feelings make sense",
         "validating", +35, -25),
        ("I am here and I am not going anywhere",
         "validating_safety", +45, -35),

        # Dismissive (makes it worse)
        ("You will be fine just get over it",
         "dismissive", -15, +20),
        ("That is not a big deal stop overreacting",
         "dismissive", -25, +30),
        ("I do not have time for this",
         "dismissive_abandoning", -20, +25),

        # De-escalation (TCI co-regulation)
        ("Let us take a breath together and talk about it",
         "deescalation", +35, -40),
        ("I can see you are hurting and I want to help",
         "deescalation_empathic", +40, -35),

        # Escalating (power struggle)
        ("You need to stop acting like this right now",
         "escalating", -30, +40),
        ("I have had enough of your attitude",
         "escalating_hostile", -35, +50),

        # Curious / opening (neutral but engaging)
        ("Tell me what happened",
         "curious", +15, -10),
        ("What do you need from me right now",
         "curious_agency", +20, -15),
    ]

    entity_types = ["default", "optimist", "pessimist", "traumatized",
                    "resilient", "volatile", "stoic"]

    for stim in stimuli:
        stim_v, stim_a, stim_d, stim_u, stim_g = stim["state"]

        for resp_text, resp_cat, dv, da in response_types:
            for entity_name in entity_types:
                entity = new_entity(entity_name)

                # Base outcome = stimulus state + response delta
                outcome_v = max(0, min(255, stim_v + dv))
                outcome_a = max(0, min(255, stim_a + da))
                outcome_d = stim_d + (15 if dv > 0 else -10)
                outcome_u = max(0, stim_u + (-10 if dv > 0 else 10))
                outcome_g = max(0, min(255, stim_g + int(dv * 0.8)))

                # Apply dark matter — entity modifies how the response lands
                outcome_v, outcome_a, outcome_d, outcome_u, outcome_g = entity.apply(
                    outcome_v, outcome_a, outcome_d, outcome_u, outcome_g
                )

                # Clamp
                outcome_d = max(0, min(255, outcome_d))
                outcome_u = max(0, min(255, outcome_u))

                examples.append({
                    "english": resp_text,
                    "vadug": [outcome_v, outcome_a, outcome_d, outcome_u, outcome_g],
                    "source": "doctor_strange",
                    "features": (f"stimulus:{stim['context']}, "
                                f"response:{resp_cat}, "
                                f"entity:{entity_name}, "
                                f"outcome_type:{'recovery' if outcome_v > stim_v else 'decline'}"),
                })

    return examples


def main():
    random.seed(42)  # Reproducible

    print("=" * 60)
    print("Clanker Training Data Expansion")
    print("=" * 60)

    all_examples = []

    print("\n1. Generating idiom training examples...")
    idiom_ex = generate_idiom_examples()
    print(f"   → {len(idiom_ex)} idiom examples (3 per idiom × {len(IDIOMS)} idioms)")
    all_examples.extend(idiom_ex)

    print("\n2. Generating strong positive expansion...")
    pos_ex = generate_positive_expansion()
    print(f"   → {len(pos_ex)} positive examples")
    all_examples.extend(pos_ex)

    print("\n3. Generating long sentence expansion...")
    long_ex = generate_long_sentence_expansion()
    print(f"   → {len(long_ex)} long examples")
    all_examples.extend(long_ex)

    print("\n4. Generating TCI trajectory examples...")
    tci_ex = generate_tci_trajectory_examples()
    print(f"   → {len(tci_ex)} trajectory examples")
    all_examples.extend(tci_ex)

    print("\n5. Generating dark matter entity examples...")
    dm_ex = generate_dark_matter_examples()
    print(f"   → {len(dm_ex)} dark matter examples")
    all_examples.extend(dm_ex)

    print("\n6. Generating short calibration examples...")
    short_ex = generate_short_calibration()
    print(f"   → {len(short_ex)} short calibration examples")
    all_examples.extend(short_ex)

    print("\n7. Generating response-outcome examples...")
    resp_ex = generate_response_outcome_examples()
    print(f"   → {len(resp_ex)} response-outcome examples")
    all_examples.extend(resp_ex)

    print("\n8. Generating Doctor Strange multi-reality examples...")
    ds_ex = generate_doctor_strange_examples()
    print(f"   → {len(ds_ex)} doctor strange examples")
    print(f"     ({len(ds_ex) // 7} response-stimulus pairs × 7 entity types)")
    all_examples.extend(ds_ex)

    print("\n9. Generating Rosetta Stone triplets (A + B = C)...")
    from training.rosetta_triplets import generate_rosetta_triplets
    triplet_ex = generate_rosetta_triplets()
    print(f"   → {len(triplet_ex)} triplet examples (input + response + outcome)")
    all_examples.extend(triplet_ex)

    # Analyze V distribution of new examples
    v_buckets = {"neg_high": 0, "neg_med": 0, "neg_low": 0, "neutral": 0,
                 "pos_low": 0, "pos_med": 0, "pos_high": 0}
    thresholds = [37, 74, 110, 147, 183, 220]
    names = list(v_buckets.keys())
    for ex in all_examples:
        v = ex["vadug"][0]
        bucket = 6
        for i, t in enumerate(thresholds):
            if v < t:
                bucket = i
                break
        v_buckets[names[bucket]] += 1

    print(f"\n{'=' * 60}")
    print(f"Total new examples: {len(all_examples)}")
    print(f"\nV-axis distribution of new data:")
    for name, count in v_buckets.items():
        pct = 100 * count / len(all_examples)
        print(f"  {name:10s}: {count:4d} ({pct:.1f}%)")

    # Write to file
    out_path = os.path.join(os.path.dirname(__file__), "data", "phase1_expanded.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\n→ Saved to {out_path}")
    print(f"\nLength distribution:")
    lengths = [len(ex["english"].split()) for ex in all_examples]
    print(f"  <5 words:   {sum(1 for l in lengths if l < 5)}")
    print(f"  5-10 words: {sum(1 for l in lengths if 5 <= l < 10)}")
    print(f"  10-20 words:{sum(1 for l in lengths if 10 <= l < 20)}")
    print(f"  20+ words:  {sum(1 for l in lengths if l >= 20)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
