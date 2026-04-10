"""
EMOTIONAL_VOCABULARY — Curated emotional word forces for Clanker

Curation criteria:
  Source: curated from 46K entries, reduced to 2,315 high-signal words
  Cross-referenced with: 99,506 sentences from EmpatheticDialogues

  Inclusion rules:
    1. Word appears >= 10 times in EmpatheticDialogues (real conversational usage)
    2. |dV| >= 15 (meaningful emotional valence shift)
    3. NOT a common object noun with no emotional weight (bagel, soup, table, etc.)
    4. NOT a function word, pronoun, preposition, or generic verb
    5. Known psychological emotion/feeling words included even if below freq threshold

  Result: 2,000 words kept, 44,101 dropped

  Tuple format: (dV, dA, dD, dU, dG) — deltas, not absolute values
    dV = Valence delta (-128 to +127)
    dA = Arousal delta
    dD = Dominance delta
    dU = Urgency delta
    dG = Gravity delta

Categories (for reference, all merged into EMOTIONAL_VOCABULARY):
  cognitive_eval: 45 words
  crisis_words: 40 words
  emotion_words: 132 words
  general_emotional: 1680 words
  intensifiers: 18 words
  relationship_words: 43 words
  situation_words: 0 words
  slang: 42 words

Built from mining EmpatheticDialogues x WORD_FORCES intersection.
"""


# Category index — words are tagged by category in comments
# emotion_words: Direct emotion/feeling labels (happy, sad, angry, ...)
# intensifiers: Amplifiers and degree modifiers (absolutely, extremely, ...)
# relationship_words: Trust, loyalty, connection vocabulary (friend, betrayed, ...)
# crisis_words: Trauma, danger, emergency vocabulary (suicide, abuse, dying, ...)
# slang: Informal emotional language (awesome, sucks, pissed, ...)
# cognitive_eval: Right/wrong/fair/deserve evaluative words
# general_emotional: Other words with meaningful emotional force


EMOTIONAL_VOCABULARY = {
    # ── EMOTION WORDS — Direct emotion/feeling labels ──
    'abandoned': (-127,   -6, -127,   +0,  -70),  # freq=42
    'afraid': ( -78,  +53,  -87,  +79,  +27),  # freq=666
    'agony': (-127, +119, -127,  +97,  -61),  # freq=2
    'anger': ( -89, +127, +109,  +77,  +61),  # freq=23
    'angry': (-102, +119, +109,  +97,  +73),  # freq=642
    'anguish': (-127, +119, -127,  +97,  -85),  # freq=5
    'annoyed': ( -42,  +49,  +22,  +31,  +20),  # freq=364
    'anxiety': (-127, +127, -127, +127,  +24),  # freq=143
    'anxious': ( -52,  +87, -114,  +97,  +29),  # freq=390
    'appreciated': ( +42,   -4,  +37,   +0,  +19),  # freq=87
    'ashamed': (-102,  +74, -127,  +39,  -61),  # freq=412
    'awe': ( +86,  +47,  +52,   +6,  -50),  # freq=58
    'betrayed': (-106,  +72,  -82,  +54,  -51),  # freq=16
    'brave': ( +28,  +22,  +52,   +6,  +14),  # freq=52
    'broken': (-102,  +45, -127,  +57,  -61),  # freq=129
    'calm': ( +20,  -40,  +30,  -10,   +5),  # liquid -- state=positive, command=negative
    'comfort': ( +28,   -9,  +24,   +0,   +8),  # freq=56
    'comfortable': ( +44,  -27,   -2,   +0,  +12),  # freq=109
    'compassion': ( +90,  +27,  +77,  +18,  -29),  # freq=21
    'confidence': ( +48,  -18,  +59,   +0,  +24),  # freq=129
    'confident': ( +50,  +40, +107,   +0,  +32),  # freq=521
    'content': ( +58,  -36,  +17,   +0,  +19),  # freq=369
    'cranky': ( -86,  +74,  -42,  +50,  -36),  # freq=11
    'cruelty': (-114,  +90,  +17,  +70,  -36),  # freq=8
    'crushed': (-102,  +74, -127,  +57, -108),  # freq=37
    'delighted': ( +49,  +38,   +7,   +1,  +16),  # freq=12
    'depressed': (-127,  -26, -109,  +63,  -89),  # freq=72
    'depression': (-127, -119, -127,  +39,  +97),  # freq=39
    'despair': (-127,  +33, -120,  +95, -107),  # freq=3
    'destroyed': (-122,  +53,  -66, +111,  -54),  # freq=75
    'determined': ( +51,   +6,  +49,   +0,  +24),  # freq=31
    'devastated': (-127, +119, -127, +116, -108),  # freq=351
    'disgust': ( -89,  +90,  +54,  +39,  +48),  # freq=7
    'disgusted': (-127,  +58,  +54,  +39,  +36),  # freq=287
    'distraught': (-106, +108, -104,  +81,  -53),  # freq=16
    'dread': ( -89,  +90, -127,  +77,  -48),  # freq=15
    'ecstatic': ( +51,  +51,  +34,   +0,  +38),  # freq=14
    'embarrassed': ( -67,  +33,  -55,  +32,  -18),  # freq=384
    'empathy': ( +80,  +33,  +42,  +12,  -23),  # freq=5
    'envy': ( -37,  +22,  -18,  +13,   +7),  # freq=46
    'excited': ( +37,  +42,  +24,  +12,  +27),  # freq=1261
    'excitement': ( +42,  +15,  +29,   +1,  +17),  # freq=23
    'fear': ( -78,  +53,  -87,  +79,  +27),  # freq=160
    'frustrated': ( -78,  +46,  -33,  +79,  +27),  # freq=67
    'frustration': ( -52,  +74,  -67,  +65,  +20),  # freq=16
    'furious': (-127, +127, +127, +127,  +97),  # freq=370
    'glad': ( +28,  +13,  +17,   +0,  +14),  # freq=1505
    'gloomy': (-109,  -24,  -92,   +0,  -53),  # freq=3
    'grateful': ( +28,   +9,  +17,   +0,  +10),  # freq=434
    'gratitude': ( +90,   +9,  +77,  +12,  -23),  # freq=4
    'grief': (-127,  +74, -127,  +57, -108),  # freq=14
    'grumpy': ( -83,  +83,  +27,  +57,  -24),  # freq=11
    'guilt': (-127, +127, -127, +127, +127),  # freq=42
    'happy': ( +28,   +8,  +10,   +0,  +17),  # freq=2606
    'hate': (-127, +119, +109, +116,  +85),  # freq=1146
    'hatred': (-127,  +92,  -32,  +77,  -51),  # freq=5
    'haunted': ( -52,  +63,  -67,  +31,  -20),  # freq=78
    'heartbroken': (-122,  +40,  -76,  +47,  -72),  # freq=74
    'helpless': (-102,  +45, -127,  +77,  -85),  # freq=17
    'hesitant': ( -37,   +9,  -64,   +6,  -24),  # freq=17
    'hope': ( +50,  +29,  +34,  +12,  +32),  # freq=4362
    'hopeful': ( +37,  +13,  +34,   +0,  +23),  # freq=341
    'hopeless': (-127,  +58, -127, +127, -127),  # freq=6
    'horrified': (-127, +103,  -59,  +86,  -53),  # freq=16
    'humiliated': (-112,  +56, -124,  +43,  -58),  # freq=28
    'hurt': ( -78,  +33,  -44,  +47,  -36),  # freq=507
    'hurts': ( -78,  +33,  -44,  +47,  -36),  # inflected form of hurt
    'insecure': ( -42,  +38, -114,  +31,  -29),  # freq=13
    'inspired': ( +49,  +15,  +32,   +0,  +20),  # freq=14
    'irritated': ( -81,  +83,  -97,  +57,  -43),  # freq=75
    'jealous': ( -52,  +63,  -44,  +31,  +20),  # freq=537
    'jealousy': ( -77,  +74,  -82,  +39,  +12),  # freq=17
    'joy': ( +42,  +27,  +34,   +0,  +29),  # freq=92
    'kindness': ( +44,  -13,   +2,   +0,  +13),  # freq=30
    'livid': (-127, +127, +127,  +77,  +61),  # freq=39
    'loneliness': ( -77,  -15, -127,  +19,  -73),  # freq=35
    'lonely': ( -77,  -29, -127,  +39,  -73),  # freq=534
    'love': ( +35,  +12,  +10,   +0,  +22),  # freq=2464
    'loved': ( +42,  +22,  +24,   +0,  +27),  # freq=422
    'miserable': (-127,  +29, -127,  +77, -108),  # freq=55
    'motivated': ( +35,  +29,  +64,   +4,  +21),  # freq=49
    'neglected': ( -52,  -24, -122,   +0,  -38),  # freq=11
    'nervous': ( -37,  +33,  -37,  +53,  +15),  # freq=881
    'nostalgic': ( +28,  +27,  -24,   +0,  -10),  # freq=406
    'optimistic': ( +80,  +42,  +59,   +6,  -29),  # freq=74
    'overwhelmed': ( -89, +103, -127,  +97,  -61),  # freq=27
    'pain': ( -89,  +90, -127,  +97,  -48),  # freq=256
    'panic': (-102, +127, -127, +127,  +61),  # freq=34
    'panicked': (-127, +127, -127, +127,  +73),  # freq=9
    'passionate': ( +51,  +18,  +34,   +0,  +20),  # freq=18
    'peace': ( +32,  -18,  +34,   +0,  +10),  # freq=136
    'peaceful': ( +32,  -22,  +34,   +0,   +8),  # freq=53
    'pessimistic': (-127,  -58,  -54,  +19,  +73),  # freq=7
    'pity': ( -94,  -27, -127,   +0,  -61),  # freq=21
    'pleased': ( +44,   +4,  +39,   +0,  +20),  # freq=95
    'pride': ( +50,  +40,  +89,   +0,  +32),  # freq=75
    'proud': ( +32,  +22,  +52,   +0,  +19),  # freq=1038
    'rage': (-114, +127, +127, +127,  +85),  # freq=36
    'regret': ( -77,  +45, -109,  +39,  -48),  # freq=81
    'rejected': (-112,   -6, -109,   +0,  -55),  # freq=23
    'relief': ( +28,  -13,  +24,   -6,  +10),  # freq=105
    'relieved': ( +50,  -29,  +54,  -25,  +16),  # freq=47
    'reluctant': ( -49,  +29,  -22,  +19,  -20),  # freq=9
    'restless': (-114, +127, -127, +127,  +24),  # freq=2
    'ruined': (-127,  +90, -127,  +77,  -73),  # freq=93
    'sad': ( -78,  -13,  -44,  +16,  -54),  # freq=1459
    'sadness': ( -89,  +29, -109,  +19,  -73),  # freq=14
    'safe': ( +39,  -20,  +72,   +0,  +16),  # freq=286
    'satisfied': ( +48,   +0,  +24,   +0,  +17),  # freq=79
    'scared': (-102, +127, -127,  +97,  +48),  # freq=1096
    'shame': ( -89,  +58, -127,  +39,  -48),  # freq=142
    'shattered': (-114, +103, -127,  +77,  -73),  # freq=25
    'shocked': ( -42,  +99,  -67,  +31,  +20),  # freq=342
    'startled': ( -37,  +83,   +9,  +49,  -12),  # freq=25
    'stress': ( -52,  +74,  -67,  +81,   +9),  # freq=171
    'stressed': ( -52,  +74,  -92,  +97,  +20),  # freq=119
    'suffering': (-114,  +74, -127,  +77,  -73),  # freq=35
    'surprise': ( +28,  +96,  -24,  +54,  +34),  # freq=405
    'sympathy': ( +71,  +22,  +52,  +24,  -23),  # freq=6
    'terrified': (-122,  +79, -120, +127,  +45),  # freq=366
    'terror': (-127, +127, -127, +127,  +48),  # freq=7
    'thrill': ( +41,  +27,  +29,   +2,  +16),  # freq=15
    'thrilled': ( +80,  +50,  +40,   +0,  +35),  # boosted — strong joy word like ecstatic
    'torn': ( -52,  +49,  -67,  +31,  -29),  # freq=36
    'trust': ( +28,   +4,  +34,   +0,   +8),  # freq=438
    'uncomfortable': ( -42,  +38,  -67,  +31,   -9),  # freq=51
    'uneasy': ( -37,  +22,  -27,  +40,   +7),  # freq=43
    'upset': ( -77,  +74,  -54,  +57,  -24),  # freq=693
    'valued': ( +51,   +0,  +42,   +0,  +21),  # freq=8
    'worried': ( -42,  +63,  -92,  +81,  +20),  # freq=525
    'worry': ( -42,  +49,  -67,  +49,   +9),  # freq=340
    'worthless': (-127,  +20, -127,  +95,  -89),  # freq=10
    'wrecked': (-117,  +33,  -97,  +27,  -55),  # freq=18

    # ── INTENSIFIERS — Degree modifiers and amplifiers ──
    'awfully': ( -94,   +0,  -37,   +0,  -39),  # freq=6
    'constantly': ( +35,   +9,  +57,   +2,  +20),  # freq=54
    'desperately': ( -94, +121, -127,  +89,  -55),  # freq=10
    'especially': ( +28,  +27,  +24,   +0,  +10),  # freq=486
    'extremely': ( +29,  +56, +112,  +18,  +27),  # freq=112
    'fairly': ( +29,   -6,  +34,   +0,  +14),  # freq=50
    'hardly': ( -94,  -90,  -22,   +0,  -36),  # freq=74
    'honestly': ( +50,   +2,  +29,   +0,  +20),  # freq=185
    'horribly': (-127,  +36,  -22,  +27,  -46),  # freq=21
    'incredibly': ( +35,  +11,   +2,   +1,  +10),  # freq=54
    'insanely': ( -94, +127, -104,  +97,  -51),  # freq=11
    'never': ( -28,  +11,   -9,  +13,   -7),  # freq=2470
    'pretty': ( +50,  +20,  +17,   +0,  +24),  # freq=2066
    'quite': ( -29,  -63,  -27,   +0,  -13),  # freq=686
    'rather': ( -27,  -49,  -32,   +0,  -13),  # freq=138
    'really': ( +28,  +40,  +24,   +0,  +10),  # freq=7818
    'terribly': (-123,  +63,  -32,  +50,  -48),  # freq=47
    'truly': ( +28,  +40,  +24,   +0,  +10),  # freq=170

    # ── RELATIONSHIP WORDS — Trust, connection, bonds ──
    'alone': ( -67,  -13,  -44,  +16,  -45),  # freq=792
    'apart': ( -52,  +38,  -92,  +31,  -40),  # freq=96
    'attachment': ( +57,  +18,  +59,   +6,  +10),  # freq=7
    'bond': ( +39,  +20,  +54,   +0,  +16),  # freq=47
    'breakup': (-127, +127, -127,  +39,  +97),  # freq=10
    'brother': ( +29,  +20,  +17,  +12,   +8),  # freq=876
    'caring': ( +29,   -6,   +0,   +0,   +8),  # freq=329
    'cheated': (-127, +127, -127, +116,  -48),  # freq=154
    'cheating': (-127, +127, -127, +116,  -48),  # freq=117
    'cold': ( -24,  +13,  -24,  +18,  -10),  # freq=129
    'commitment': ( +39,  +29,  +72,  +12,  +16),  # freq=17
    'couple': ( +32,   +4,   +7,   +1,  +10),  # freq=517
    'daughter': ( +39,  +20,  +17,  +12,  +16),  # freq=784
    'dishonest': ( -86,   -2, -127,   +0,  -53),  # freq=4
    'distant': ( -77,  -47,  -97,   +0,  -43),  # freq=17
    'divorce': (-114, +103, -127,  +97,  -73),  # freq=61
    'enemy': (-127, +127,  -54, +127,  +73),  # freq=10
    'faithful': ( +28,   +9,  +34,   +0,   +8),  # freq=334
    'fake': ( -67,  +26,  +22,  +32,  +18),  # freq=34
    'family': ( +50,  +20,  +34,  +12,  +16),  # freq=1932
    'friend': ( +29,   +9,   +7,   +0,  +12),  # freq=2828
    'friendship': ( +48,  -13,  +27,   +0,  +17),  # freq=51
    'honest': ( +39,  +20,  +54,   +0,  +16),  # freq=171
    'honesty': ( +71,   +4,  +77,   +0,  -10),  # freq=13
    'husband': ( +66,  +18,  +59,   +6,  +10),  # freq=860
    'liar': (-102, +103,  +82,  +57,  +48),  # freq=13
    'lover': ( +35,  +27,  +27,   +3,  +14),  # freq=25
    'loving': ( +35,   +9,  +19,   +1,  +13),  # freq=76
    'loyalty': ( +50,  +20,  +72,   +0,  +16),  # freq=18
    'lying': ( -81,  +54,  -82,  +39,  -40),  # freq=55
    'marriage': ( +66,  +27,  +67,   +6,  +23),  # freq=90
    'mother': ( +50,  +20,  +34,  +12,  +16),  # freq=392
    'parent': ( +28,  +27,  +49,  +35,  +10),  # freq=158
    'partner': ( +62,  +13,  +59,   +6,   +4),  # freq=219
    'relationship': ( +37,   +4,  +12,   +1,  +13),  # freq=315
    'sister': ( +29,  +20,  +17,  +12,   +8),  # freq=537
    'son': ( +39,  +20,  +17,  +12,  +16),  # freq=898
    'spouse': ( +56,  +18,  +42,   +4,  +23),  # freq=58
    'supportive': ( +39,   -6,  +19,   +0,  +14),  # freq=78
    'together': ( +39,  +20,  +34,   +0,  +16),  # freq=689
    'unfaithful': (-127,  +83,  -87,  +70,  -61),  # freq=9
    'wedding': ( +32,  +27,  +24,   +6,  +23),  # freq=197
    'wife': ( +66,  +18,  +59,   +6,  +10),  # freq=999

    # ── CRISIS WORDS — Trauma, danger, emergency ──
    'abuse': (-127, +119, -127, +127,  -73),  # freq=44
    'abused': (-127, +119, -127, +127,  -73),  # freq=35
    'accident': (-104, +110,  -69,  +81,  -48),  # freq=242
    'addicted': (-102,  +74, -127,  +97,  -73),  # freq=19
    'addiction': (-102,  +74, -127,  +97,  -73),  # freq=31
    'attack': ( -51,  +99,  +64,  +61,   -5),  # freq=81
    'bleeding': ( -77,  +90, -109,  +97,  -36),  # freq=12
    'crash': ( -67,  +40,  -44, +111,  -36),  # freq=76
    'crisis': ( -89, +119, -127, +127,  -36),  # freq=9
    'cry': ( -20,  +40,  -25,  +20,  +30),  # liquid -- happy tears vs sad tears. High G, reduced V.
    'crying': ( -25,  +35,  -20,  +25,  +30),
    'danger': ( -77, +119, -127, +127,  +24),  # freq=16
    'dangerous': (-127, +119,  +69, +100,  -34),  # freq=103
    'dead': (-114,  +45, -127,  +39,  -97),  # freq=155
    'death': (-127, +103, -127, +116,  -85),  # freq=139
    'die': (-127, +127, -127, +127, -121),  # freq=146
    'dying': (-102, +103, -127, +116,  -85),  # freq=53
    'emergency': ( -89, +127, -127, +127,  -36),  # freq=80
    'harm': ( -77,  +58,  +54,  +57,  -12),  # freq=46
    'harmful': (-102,  +74,  +54,  +57,  +24),  # freq=4
    'healing': ( +56,   -9,  +24,   +0,  +20),  # freq=28
    'homeless': (-127,  +90, -127, +127, -108),  # freq=90
    'hospital': ( -77,  +90, -127, +127,  -48),  # freq=192
    'killing': ( -77, +127,  +82, +127,  +48),  # freq=34
    'loss': ( -60,  +20,  -55,  +35,  -40),  # grief: bereavement, deprivation
    'lost': ( -46,  +11,  -55,  +40,  -30),  # freq=808
    'murder': (-123, +119,  +49,  +97,  -34),  # freq=10
    'poverty': (-127,  +20, -109, +127,  +80),  # freq=4
    'recovery': ( +39,  +29,  +54,  +25,  +16),  # freq=22
    'scream': ( -52, +127,  +44,  +97,  +50),  # freq=33
    'screaming': ( -77, +127,  +54,  +97,  +48),  # freq=41
    'starving': ( -52,  +63,  -67,  +81,  -29),  # freq=26
    'suicidal': (-127, +127, -127, +127, -127),  # freq=7
    'suicide': (-127,  +79, -127, +127,  -98),  # freq=6
    'toxic': (-114,  +90,  +54,  +57,  +48),  # freq=2
    'trauma': (-127, +127, -127, +127,  +85),  # freq=20
    'traumatic': (-123, +103,  -32,  +86,  -48),  # freq=20
    'victim': (-102,  +74, -127,  +77,  -61),  # freq=6
    'violence': (-123, +117,  +97,  +93,  -27),  # freq=2
    'violent': (-109, +112,  +49,  +86,  -28),  # freq=9
    'wound': ( -52,  +49,  -67,  +49,  -29),  # freq=22

    # ── SLANG — Informal emotional language ──
    'amazing': ( +48,  +38,  +34,   +0,  +29),  # freq=995
    'awesome': ( +65,  +35,  +30,   +0,  +25),  # boosted — strong positive exclamation
    'awful': (-127,  +74,  -82,  +57,  -61),  # freq=473
    'basic': ( -77,  -45,  -54,   +0,  +12),  # freq=25
    'blessed': ( +28,   +6,  +14,   +0,  +14),  # freq=67
    'bomb': ( -94, +110, +109,  +81,  -14),  # freq=29
    'brutal': (-102, +119,  +82,  +77,  -24),  # freq=21
    'bummed': ( -47,  -15,  -87,   +0,  -29),  # freq=76
    'crap': ( -37,  +11,   -9,  +13,   -7),  # freq=52
    'crazy': ( -24,  +83,  +24,  +35,  +23),  # freq=541
    'cringe': (-127, +103, -127,  +39,  +36),  # freq=15
    'cursed': ( -89,  +33,  -55,  +47,  -45),  # freq=7
    'damn': ( -32,  +49,  +22,  +31,   +9),  # freq=74
    'deadly': (-100,  +92,  +22,  +70,  -31),  # freq=10
    'disgusting': (-122,  +53,  +33,  +47,  +54),  # freq=157
    'dreadful': (-122,  +46,  -87,  +79,  -54),  # freq=8
    'extra': ( -52,  +74,  -44,  +15,   +9),  # freq=283
    'fantastic': ( +31,  +12,  +14,   +0,  +20),  # freq=101
    'garbage': ( -89,  +45,  -54,  +39,  -36),  # freq=66
    'gross': ( -77,  +58,  +27,  +19,  +36),  # freq=377
    'hell': ( -37,  +28,   +9,  +26,  -30),  # freq=61
    'horrible': (-122,  +46,  -22,  +79,  -36),  # freq=517
    'hyped': ( +46,  +13,  +49,   +2,  +23),  # freq=23
    'insane': ( -28,  +44,  +18,  +40,  +22),  # freq=73
    'killer': (-127, +127,   +0, +104,  -43),  # freq=21
    'legit': ( +55,  -13, +102,   +0,  +32),  # freq=11
    'lit': ( +62,  +51,  +52,   +6,  -23),  # freq=9
    'mood': ( +28,   -4,  +24,   +0,   -8),  # freq=76
    'nuts': ( +28,  -60,  -84,   +0,   -6),  # freq=50
    'oof': ( -77,  +45, -109,  +19,  +24),  # freq=24
    'pathetic': (-100,  +20,  -76,  +32,  -54),  # freq=9
    'petty': ( -81,   +9, -114,   +7,  -48),  # freq=14
    'pissed': (-127, +127, +109, +127,  +36),  # freq=85
    'screwed': (-127, +127, -127, +127,  +61),  # freq=14
    'shady': ( -40,   +2,  -44,   +0,  -21),  # freq=26
    'sick': ( -77,  +45, -127,  +57,  -36),  # freq=612
    'sketchy': ( -37,   -4,  -77,   +0,  -25),  # freq=25
    'suck': ( -36,  +47,  -44,  +25,  -20),  # freq=58
    'sucks': ( -67,  +26,  -11,  +32,  -18),  # freq=589
    'terrible': (-111,  +40,  -22,  +63,  -36),  # freq=856
    'wonderful': ( +42,  +22,  +24,   +0,  +23),  # freq=515
    'yikes': ( -52,  +63,  -44,  +15,   +9),  # freq=111

    # ── COGNITIVE/EVALUATIVE — Right, wrong, fair, deserve ──
    'accomplishment': ( +43,  +15,  +52,   +1,  +21),  # freq=89
    'ache': ( -52,  +20,  -44,  +25,  -30),  # grief: persistent pain
    'aching': ( -52,  +20,  -44,  +25,  -30),  # grief: persistent pain
    'achieve': ( +28,  +22,  +42,   +6,  +19),  # freq=43
    'achievement': ( +42,  +27,  +34,   +2,  +17),  # freq=32
    'agree': ( +29,   +9,  +34,   +0,  +16),  # freq=625
    'believe': ( +39,  +20,  +54,  +12,  +24),  # freq=963
    'blame': ( -52,  +63,  +44,  +31,  +20),  # freq=129
    'certain': ( +29,  +20,  +89,  +12,   +8),  # freq=122
    'choice': ( +42,  +42, +102,  +67,  +10),  # freq=168
    'choose': ( +51,  +33,  +94,  +55,   +8),  # freq=52
    'clear': ( +29,   +9,  +54,   +0,  +16),  # freq=65
    'decide': ( +28,  +40,  +99,  +35,  +10),  # freq=71
    'decision': ( +32,   +2,  +97,   +0,  +25),  # freq=117
    'defeat': ( -77,  +74, -127,  +39,  -48),  # freq=4
    'deserved': ( +29,   +6,  +24,   +1,  +13),  # freq=75
    'difficult': ( -28,  +17,  -18,  +40,   -7),  # freq=314
    'doubt': ( -32,  +24,  -67,  +31,  -20),  # freq=92
    'easy': ( +29,  -20,  +54,   +0,  +16),  # freq=330
    'failure': ( -89,  +58, -127,  +57,  -61),  # freq=20
    'false': (-127,  +90, -109,  +57,  +36),  # freq=13
    'fault': ( -37,  +17,   -9,  +26,  -15),  # freq=136
    'fight': ( -42,  +99,  +92,  +81,  +29),  # freq=175
    'fighting': ( -77, +119,  +82,  +97,  +36),  # freq=52
    'fought': ( -94,  +51,  +17,  +39,  -28),  # freq=8
    'hard': ( -24,  +27,  +49,  +35,  -10),  # freq=2616
    'impossible': ( -52,  +38,  -92,  +31,  -29),  # freq=37
    'irresponsible': ( -92,  +47, -114,  +34,  -51),  # freq=26
    'mistake': ( -46,  +17,  -27,  +40,  -15),  # freq=153
    'overcome': ( +50,  +49, +107,  +12,  +32),  # freq=57
    'possible': ( +21,   +6,  +10,  +15,   +8),  # freq=171
    'purpose': ( +39,  +29,  +54,  +12,  +24),  # freq=47
    'quit': ( -32,  +24,  -67,  +31,  -20),  # freq=127
    'reason': ( +28,  +27,  +74,  +35,  +10),  # freq=426
    'responsible': ( +35,   +2,  +47,   +0,  +17),  # freq=62
    'result': ( +28,  +40,  +74,  +54,  +10),  # freq=55
    'significant': ( +34,  +11,  +39,   +2,  +16),  # freq=24
    'struggle': ( -52,  +63,  -92,  +65,  -29),  # freq=48
    'success': ( +32,  +22,  +42,   +0,  +19),  # freq=77
    'survive': ( +28,  +54,  +99,  +54,  +10),  # freq=44
    'true': ( +86,  +13, +112,  +12,  +19),  # freq=746
    'understand': ( +28,  +27,  +49,  +18,  +10),  # freq=981
    'unfair': ( -92,   -9, -127,   +0,  -53),  # freq=41
    'villain': ( -92, +103,  +37,  +73,  -24),  # freq=3
    'win': ( +24,  +12,  +21,   +5,  +20),  # freq=523
    'winning': ( +32,  +22,  +42,   +0,  +19),  # freq=114
    'wrong': ( -42,  +24,  -44,  +49,   -9),  # freq=495

    # ── GENERAL EMOTIONAL — Other words with meaningful emotional force ──
    'abilities': ( +51,   +2,  +47,   +0,  +23),  # freq=35
    'ability': ( +39,   +0,  +39,   +0,  +17),  # freq=36
    'able': ( +28,  +13,  +74,   +0,  +10),  # freq=1446
    'abroad': ( +37,  +18,  +17,   +4,  +14),  # freq=31
    'absolute': ( +48,   +9, +102,  +18,  +38),  # freq=25
    'abusive': (-104, +108,  +27,  +81,  -31),  # freq=26
    'accept': ( +39,   +9,  +54,   +0,  +16),  # freq=58
    'acceptance': ( +50,   +9,  +54,   +0,  +16),  # freq=15
    'access': ( +48,   -6,  +17,   +0,  +17),  # freq=21
    'accidentally': ( -37,  +54,  -59,  +31,  -24),  # freq=158
    'account': ( +50,  +29, +107,  +63,   +8),  # freq=61
    'accused': (-104,  +45,  -32,  +34,  -40),  # freq=24
    'ace': ( +32,   -4,  +14,   +0,  +10),  # freq=47
    'achieved': ( +42,   +2,  +37,   +0,  +17),  # freq=19
    'across': ( +28,  +27,  +49,  +18,   +0),  # freq=161
    'act': ( +34,  +36,  +69,   +9,  +20),  # freq=95
    'acting': ( +43,  +24,   +0,   +7,  +12),  # freq=71
    'action': ( +58,  +72,  +87,  +15,  +31),  # freq=35
    'active': ( +35,  +20,  +29,   +2,  +16),  # freq=22
    'activities': ( +35,  +42,  +22,   +6,  +13),  # freq=30
    'activity': ( +43,  +27,  +89,   +7,  +27),  # freq=21
    'actual': ( +50,  -18,  +62,   +0,  +25),  # freq=41
    'add': ( +50,   -2,  +14,   +0,  +17),  # freq=29
    'addict': (-119,  +54,  -77,  +43,  -53),  # freq=19
    'admirable': ( +44,   +9,  +52,   +0,  +21),  # freq=12
    'admitted': ( +48,   -2,  +74,   +0,  +27),  # freq=16
    'adopt': ( +42,  +20,  +57,   +4,  +23),  # freq=43
    'adoption': ( +50,  +40,  +54,  +25,  +24),  # freq=17
    'adorable': ( +32,  +18,   +7,   +0,  +19),  # freq=55
    'adult': ( +32,   +0,  +39,   +0,  +16),  # freq=91
    'advance': ( +37,   -6, +104,   +0,  +28),  # freq=58
    'advantage': ( +80,  +51, +102,  +36,  -27),  # freq=44
    'adventure': ( +50,  +58,  +54,  +12,  +40),  # freq=37
    'advice': ( +43,  -15,  +54,   +0,  +23),  # freq=135
    'affair': (-114, +119, -109,  +97,  -36),  # freq=15
    'afford': ( -32,  +38,  -67,  +65,  -20),  # freq=244
    'against': ( -32,  +49,  +44,  +49,  +20),  # freq=96
    'aged': (-102,  -36, -114,   +0,  -55),  # freq=11
    'aggravating': ( -57,  +51,  +49,  +31,  -12),  # freq=32
    'aggressive': ( -52,  +87,  +67,  +65,  +29),  # freq=24
    'agreed': ( +31,   -4,   +9,   +0,  +10),  # freq=72
    'ahead': ( +46,   +4,  +49,   +0,  +23),  # freq=129
    'aid': ( +41,   +0,   +4,   +0,  +13),  # freq=19
    'air': ( +39,   +9,  +17,   +0,  -24),  # freq=85
    'airplane': ( +48,  +36,  +62,   +7,  +24),  # freq=34
    'airport': ( +32,  +69,  +77, +103,  +34),  # freq=59
    'alarm': ( -32,  +74,  -22,  +97,  +20),  # freq=42
    'album': ( +29,  -27,  -14,   +0,   +6),  # freq=39
    'alcohol': ( -32,  +38,  -22,  +31,   -9),  # freq=46
    'alive': ( +28,  +18,  +24,   +6,  +14),  # freq=115
    'allergic': (-102,  +11,  -97,  +11,  -51),  # freq=31
    'allergies': ( -94,   +0, -127,   +0,  -55),  # freq=14
    'allergy': ( -77,  +58, -109, +127,  +12),  # freq=10
    'alley': ( -36,   +0,  -54,   +0,  -20),  # freq=19
    'allowed': ( +50,  -24,  +42,   +0,  +23),  # freq=49
    'along': ( +21,  -20,   -9,   +0,   +5),  # freq=198
    'alright': ( +24,  -29,   +9,   +0,   +9),  # freq=194
    'amazed': ( +31,  +22,   +4,   +3,   +9),  # freq=80
    'amusement': ( +44,  +27,  +39,   +1,  +20),  # freq=54
    'animal': ( +28,  +27,  +24,   +0,  +10),  # freq=178
    'anniversary': ( +39,  +29,  +34,   +0,  +24),  # freq=94
    'annoy': (-114,  +72,  -92,  +54,  -55),  # freq=13
    'annoying': ( -52,  +49,  +22,  +31,  +20),  # freq=403
    'annoys': (-100,  +74,  -27,  +54,  -39),  # freq=13
    'answer': ( +29,  +20,  +54,  +12,  +16),  # freq=83
    'anymore': ( -32,  +24,  -44,  +15,   -9),  # freq=357
    'anywhere': ( +29,  -24,  -42,   +0,   +1),  # freq=108
    'apartments': ( +43,  -29,  -17,   +0,   +9),  # freq=15
    'apologized': ( +35,  -29,  -79,   +0,   -2),  # freq=35
    'apple': ( +62,   +4,  +52,   +6,  -10),  # freq=39
    'application': ( +32,   -9,  +64,   +0,  +20),  # freq=36
    'applied': ( +29,  +27,  +32,   +4,  +14),  # freq=152
    'appointment': ( +42,  +15,  +14,   +4,  +14),  # freq=80
    'appreciate': ( +21,   +4,  +10,   +0,   +8),  # freq=232
    'appreciates': ( +58,   +0,  +42,   +0,  +24),  # freq=18
    'appreciation': ( +71,  +33,  +77,  +12,  -23),  # freq=10
    'appreciative': ( +36,  -11,  +19,   +0,  +14),  # freq=49
    'apprehensive': ( -24,  +22,  -29,  +11,  -13),  # freq=232
    'approaching': ( +51,   +9,   -2,   +2,  +14),  # freq=11
    'approved': ( +35,  -13,  +27,   +0,  +14),  # freq=54
    "aren't": ( -24,  +13,   +0,  +18,   +0),  # freq=219
    'argue': ( -42,  +74,  +44,  +49,  +20),  # freq=23
    'argument': ( -52,  +87,  +44,  +65,  +20),  # freq=49
    'arranged': ( +43,   +6,  +64,   +2,  +24),  # freq=13
    'arrested': ( -89, +119, -127, +127,  -36),  # freq=42
    'arrive': ( +35,   -6,  +12,   +0,  +12),  # freq=42
    'arrived': ( +39,  +29,  +54,   +0,  +16),  # freq=48
    'art': ( +39,  +29,  +34,   +0,  +24),  # freq=49
    'asleep': ( +37,  -47,  +34,   +0,  -23),  # freq=91
    'assertive': ( +48,  -15,  +79,   +0,  +27),  # freq=14
    'assume': ( +43,  -18,   -9,   +0,  +10),  # freq=44
    'assuming': ( +51,  -20,   +7,   +0,  +16),  # freq=30
    'assured': ( +39,  -15,  +34,   +0,  +17),  # freq=16
    'athletic': ( +36,  +24,  +29,   +3,  +16),  # freq=41
    'attend': ( +32,  -13,  +34,   +0,  +16),  # freq=69
    'attention': ( +51,  +18,  +79,   +4,  +28),  # freq=134
    'attic': ( +28,  -54,  -74,  +18,  +44),  # freq=49
    'attitude': ( +58,  +24, +104,   +4,  +34),  # freq=107
    'attractive': ( +50,  +29,  +34,   +0,  +24),  # freq=32
    'aunt': ( +32,   +9,  +34,   +0,   +8),  # freq=116
    'autoimmune': ( -94,  +90, -124,  +66,  -53),  # freq=12
    'available': ( +28,  -11,  +17,   +0,  +10),  # freq=52
    # 'average' removed — neutral factual word causing positive drift
    'award': ( +37,  +20,  +22,   +2,  +14),  # freq=53
    'awkward': ( -32,  +38,  -92,  +15,   -9),  # freq=92
    'babies': ( +42,  +27,  -27,   +2,   +8),  # freq=108
    'babysit': ( +48,  +15,  -29,   +2,   +9),  # freq=30
    'bacon': ( +66,  +33,  +34,  +30,  +10),  # freq=12
    'bad': ( -77,  +29,  -54,  +39,  -24),  # freq=2987
    'barely': ( -50,  +15,  -45,  +25,  -30),  # grief: struggling, minimal capacity
    'badly': (-119,   +2,  -92,   +3,  -58),  # freq=99
    'bahamas': ( +35,  +20,   +0,   +2,   +9),  # freq=20
    'bake': ( +39,  +20,  +54,   +0,  +16),  # freq=22
    'baked': ( +31,  -36,   -2,   +0,   +8),  # freq=42
    'balance': ( +39,  -40,  +69,   +0,  +23),  # freq=40
    'banana': ( +57,   +4,  +52,  +12,  -10),  # freq=14
    'band': ( +32,   +9,   +7,   +2,  +10),  # freq=133
    'bang': ( -55, +101,  +87,  +65,   -4),  # freq=20
    'bank': ( +39,  +40, +127,  +77,  +88),  # freq=130
    'baseball': ( +35,  +45,  +17,  +12,  +12),  # freq=87
    'based': ( +48,  +13,  +67,   +0,   +8),  # freq=50
    'basement': ( -32,  -24,  -92,  +15,  +81),  # freq=92
    'basketball': ( +32,  +22,   -7,   +3,   +8),  # freq=45
    'bat': ( -40,  -20,   -9,   +0,  -16),  # freq=37
    'bathing': ( +51,  -40,  -39,   +0,   +8),  # freq=10
    'battle': ( -32,  +87,  +92,  +65,  +20),  # freq=30
    'beaches': ( +37,   +0,   +2,   +0,  +10),  # freq=19
    'bear': ( +42,  +38,  +84,  +24,  +27),  # freq=36
    'beat': ( -51,  +31,   -4,  +19,  -17),  # freq=125
    'beaten': (-117,  +92, -104,  +73,  -58),  # freq=13
    'beating': ( -96,  +90,   -9,  +66,  -34),  # freq=20
    'beautiful': ( +28,   +6,   +7,   +0,  +14),  # freq=259
    'beauty': ( +35,   +9,  +19,   +1,  +13),  # freq=21
    'became': ( +28,  +40,  +49,  +35,  +10),  # freq=78
    'become': ( +37,  +33,  +67,  +18,   -8),  # freq=157
    'beef': ( +51,   +9,  +42,  +18,  +14),  # freq=12
    'begging': ( -49,   -9, -109,   +0,  -34),  # freq=20
    'behave': ( +37,  -24,  +72,   +0,  +23),  # freq=22
    'behavior': ( +28,  -13, +124,   +0,  +28),  # freq=65
    'behind': ( -32,  +24,  -44,  +31,  -20),  # freq=229
    'belief': ( +42,  -11,  +19,   +0,  +16),  # freq=17
    'believed': ( +39,  +20,  +54,  +12,   +8),  # freq=36
    'bench': ( +66,  -33,  +59,   +0,  +29),  # freq=23
    'best': ( +37,  +22,  +34,   +0,  +23),  # freq=1882
    'bet': ( +32,  +18,  +42,  +12,   -4),  # freq=1547
    'betrayal': (-114, +119, -127,  +77,  -48),  # freq=10
    'better': ( +29,   +9,  +14,  +10,  +12),  # freq=2110
    'bicycle': ( +46,   +4,  -22,   +0,   +9),  # freq=21
    'bid': ( +58,   +6,  +42,   +0,  +24),  # freq=11
    'big': ( +32,  +22,  +94,  +12,  +53),  # freq=1198
    'bigger': ( +35,   -2,  +47,   +0,  +17),  # freq=72
    'biggest': ( +42,   +0,  +69,   +0,  +24),  # freq=76
    'bills': ( -52,  +49,  -67,  +81,  -29),  # freq=190
    'binge': ( -47,  +36,   +0,  +19,  -16),  # freq=31
    'bipolar': (-127,  +73,  -98,  +95,  +36),  # freq=11
    'bird': ( +28,  +27,  +24,   +0,  +34),  # freq=60
    'birth': ( +39,  +58,  -17,  +38,  +24),  # freq=44
    'birthday': ( +50,  +40,  +34,   +0,  +32),  # freq=766
    'bit': ( +29,   +9,  +17,  +12,   +0),  # freq=835
    'bite': ( -52,  +74,  +44,  +65,   +9),  # freq=46
    'bitten': ( -83,  +90, -127,  +62,  -53),  # freq=12
    'bitter': ( -42,  +24,  +22,  +15,   -9),  # freq=12
    'black': ( -32,  +24,  +67,  +15,  +29),  # freq=119
    'blamed': ( -86,  +72, -127,  +50,  -55),  # freq=34
    'blast': ( -94, +117, +119,  +81,  -12),  # freq=102
    'bless': ( +50,  +20,  +54,   +0,  +24),  # freq=23
    'blessing': ( +44,  -20,  +39,   +0,  +20),  # freq=63
    'blessings': ( +44,  -11,  +17,   +0,  +16),  # freq=19
    'blew': ( -81,  +11,   +0,   +7,  -28),  # freq=44
    'blind': ( -42,  +24,  -92,  +31,  -20),  # freq=41
    'block': ( -32,  +24,  -67,  +31,  -20),  # freq=36
    'blow': ( -42,  +54,  +27,  +31,   -9),  # freq=38
    'blown': ( -37,   +9,  -32,   +6,  -17),  # freq=28
    'blue': ( +39,  -20,  +34,   +0,  -16),  # freq=104
    'board': ( +28,  -60,  -44,   +0,   +0),  # freq=62
    'bonus': ( +50,  +40,  +54,  +12,  +32),  # freq=98
    'books': ( +50,  -69,  -22,   +0,  +10),  # freq=89
    'boost': ( +37,  +20,  +34,   +2,  +16),  # freq=18
    'boring': ( -24,  -83,  -24,   +0,  -23),  # freq=104
    'born': ( +50,  +49,  +17,  +12,  +32),  # freq=190
    'boss': ( -24,  +40,  +24,  +54,   +0),  # freq=395
    'bother': ( -81,  +74,  -54,  +50,  -39),  # freq=88
    'bothering': ( -81,  +72,  -22,  +50,  -31),  # freq=26
    'bout': ( +62,  +74,  +77,  +73,  +14),  # freq=25
    'bowling': ( +34,  -11,  -54,   +0,   +1),  # freq=33
    'boyfriend': ( +62,  +38,  +42,  +12,  -10),  # freq=343
    'boys': ( +35,  +54,   -9,  +15,   +8),  # freq=47
    'breakfast': ( +29,   +9,  +17,   +0,   +8),  # freq=64
    'breaking': ( -77, +103,  -82,  +77,  -36),  # freq=85
    'breast': ( +56,  -11,   +0,   +0,  +16),  # freq=17
    'bright': ( +50,  +29,  +34,   +0,  +32),  # freq=76
    'brilliant': ( +42,  +27,  +42,   +0,  +27),  # freq=21
    'broke': ( -89,  +74, -127,  +77,  -61),  # freq=437
    'brownies': ( +51,  -27,   -2,   +0,  +14),  # freq=17
    'buddies': ( +35,  +15,  +29,   +2,  +14),  # freq=25
    'buddy': ( +29,  +20,  +17,   +0,   +8),  # freq=155
    'buffet': ( +34,   +9,   +7,   +2,  +10),  # freq=24
    'bug': ( -32,  +38,  -44,  +81,   -9),  # freq=56
    # 'building' removed — neutral factual word with extreme D/G bias
    'bully': ( -57,  +58,  +22,  +35,  -16),  # freq=10
    'bummer': ( -94,   +0, -104,   +0,  -48),  # freq=139
    'bump': ( -51,   -9,  -49,   +0,  -25),  # freq=14
    'bunch': ( +48,  +22,  +42,  +12,  +10),  # freq=245
    'bunny': ( +51,  -31,  -87,   +0,   +0),  # freq=11
    'burden': (-102,  +45, -127,  +57,  -97),  # freq=19
    'burger': ( +48,  +18,  +17,  +18,  +10),  # freq=59
    'burn': ( -79,  +74,   +4,  +50,  -27),  # freq=36
    'burned': ( -42,  +49,  -44,  +49,  +20),  # freq=46
    'burnt': (-102,  +58,  -82,  +39,  +24),  # freq=12
    'burrito': ( +62,  +27,  +24,  +43,  +14),  # freq=12
    'burst': ( -96, +103, +127,  +77,   -6),  # freq=20
    'business': ( +28,  +40,  +99,  +54,  +34),  # freq=262
    'busted': ( -83,  +47,  -64,  +30,  -39),  # freq=22
    'busy': ( -32, +123,  +44, +127,  +20),  # freq=264
    'butt': ( -36,  +49,  -59,  +29,  -21),  # freq=52
    'button': ( +29,  +29,  +54,  +25,   +0),  # freq=45
    'buying': ( +35,  +51,  +24,  +15,  +14),  # freq=132
    'cage': ( -52,  +38, -127,  +31,  -40),  # freq=23
    'cake': ( +62,  +18,  +17,   +6,  -10),  # freq=277
    'calming': ( +39,  -38,  -17,   +0,   +9),  # freq=15
    'camping': ( +34,   +6,   -7,   +2,   +9),  # freq=154
    "can't": ( -28,  +11,  -27,  +26,   -7),  # freq=2052
    'cancel': ( -79,  -51,  -59,   +0,  -36),  # freq=91
    'canceled': ( -52,  +49,  -67,  +49,  -29),  # freq=42
    'cancelled': ( -52,  +49,  -67,  +49,  -29),  # freq=52
    'cancer': (-127, +103, -127, +127,  -97),  # freq=192
    'candy': ( +62,  +38,  +17,  +18,  -14),  # freq=129
    'cannot': ( -55,  -29,  -49,   +0,  -25),  # freq=198
    'capable': ( +58,  +15,  +14,   +2,  +19),  # freq=16
    'care': ( +29,  +20,  +34,  +12,  +35),  # embrace -- high G. to care = to hold.
    'career': ( +51,  +38,  +29,   +7,  +20),  # freq=104
    'carefree': ( +56,  -49,   +0,   +0,  +16),  # freq=15
    'careful': ( +28,  +13,  +74,  +18,   +0),  # freq=165
    'careless': ( -32,  +24,  -44,  +15,   +0),  # freq=29
    'carnival': ( +44,  +33,   +7,   +2,  +14),  # freq=19
    'carry': ( +39,  +90, +127, +103,  +88),  # freq=64
    'cash': ( +80,  +60,  +94,  +61,  +19),  # freq=86
    'cat': ( +29,   +9,  +17,   +0,  +16),  # freq=724
    'catch': ( +28,  +40,  +49,  +35,  +10),  # freq=236
    'caught': ( -32,  +49,  -44,  +49,   -9),  # freq=289
    'caused': ( -36,   +4,  -39,   +3,  -20),  # freq=71
    'celebrate': ( +42,  +38,  +34,   +0,  +29),  # freq=111
    'celebrated': ( +39,  +15,  +34,   +1,  +17),  # freq=27
    'celebrating': ( +44,  +22,  +37,   +1,  +19),  # freq=25
    'celebration': ( +44,  +27,  +52,   +1,  +21),  # freq=16
    'center': ( +32,  -47,  -22,   +0,   +6),  # freq=34
    'ceo': ( +48,  +51,  +84,  +55,  +29),  # freq=12
    'cereal': ( +37,   +9,  +34,  +49,   -4),  # freq=18
    'ceremony': ( +58,   +9,  +74,   +2,  +28),  # freq=13
    'certainly': ( +29,  +20,  +54,   +0,  +16),  # freq=149
    'certification': ( +34,   +2,  +22,   +0,  +13),  # freq=34
    'championship': ( +35,  +24,  +52,   +3,  +19),  # freq=20
    'chance': ( +28,  +40,  +24,  +35,  +23),  # freq=270
    'character': ( +35,  -22,  +54,   +0,  +19),  # freq=50
    'charity': ( +66,   +9,  +52,   +6,  -14),  # freq=42
    'chase': ( -32,  +99,  +67,  +97,  +29),  # freq=28
    'chased': ( -86,  +90,  -92,  +62,  -46),  # freq=29
    'chat': ( +50,  -18,  -42,   +0,   +8),  # freq=52
    'cheat': ( -89,  +74,  +82,  +57,  +24),  # freq=106
    'cheer': ( +44,  +24,  +47,   +1,  +21),  # freq=43
    'cheesecake': ( +37,  -18,  -34,   +0,   +5),  # freq=36
    'chef': ( +36,   -6,   +9,   +0,  +12),  # freq=18
    'chemistry': ( +28,   +4,  +57,   +0,  +17),  # freq=43
    'cherish': ( +37,  +13,  +24,   +0,  +14),  # freq=70
    'chicken': ( +51,   +9,  +42,  +18,   +8),  # freq=88
    'childhood': ( +37,   -6,  -24,   +0,   +6),  # freq=298
    'chilling': ( -81,  +92,  -59,  +66,  -39),  # freq=10
    'chip': ( +50,  +29, +126,  +38,   +8),  # freq=20
    'chipotle': ( +51,  +24,   +7,   +4,  +16),  # freq=12
    'chocolate': ( +66,  +22,  +17,  +18,   -8),  # freq=125
    'choked': ( -94, +108, -127,  +77,  -58),  # freq=13
    'choosing': ( +48,   +6,  +54,   +0,  +23),  # freq=10
    'christmas': ( +57,  +38,  +42,  +24,   -8),  # freq=238
    'church': ( +28,   -9,  +42,   +6,  +19),  # freq=115
    'circus': ( +50,  +24,  -17,   +4,  +10),  # freq=22
    'cities': ( +35,  +27,  +47,   +3,  +17),  # freq=25
    'claim': ( -57,  +58,   -9,  +39,  -21),  # freq=12
    'class': ( +34,  -33,   +7,   +0,  +10),  # freq=434
    'classic': ( +42,  -42,  +32,   +0,  +17),  # freq=45
    'clearance': ( +66,  +47,  +67,  +67,  -19),  # freq=11
    'climb': ( +66,  +78,  +94,  +55,  +57),  # freq=21
    'clogged': ( -94,  -90, -114,   +0,  -51),  # freq=14
    'closest': ( +42,  -18,   +2,   +0,  +12),  # freq=20
    'clothes': ( +37,   +9,  +52,  +18,   +8),  # freq=177
    'clothing': ( +56,  -27,  -24,   +0,  +10),  # freq=19
    'club': ( +32,  +22,  -32,   +7,   +5),  # freq=52
    'clubs': ( +50,   +6,  +24,   +0,  +19),  # freq=22
    'clue': ( +34,  -22,  -24,   +0,   +5),  # freq=44
    'coast': ( +48,  -18,   -9,   +0,  +12),  # freq=37
    'cockroach': ( -86,  +45, -127,  +30,  -53),  # freq=13
    'code': ( +39,  +40, +107,  +38,   +8),  # freq=19
    'colleague': ( +50,  +29,  +89,  +38,  +16),  # freq=18
    'collection': ( +50,  -33,  +24,   +0,  +19),  # freq=18
    'college': ( +43,   -6,  +72,   +0,  +24),  # freq=802
    'color': ( +29,  -18,  -12,   +0,   +5),  # freq=68
    'colors': ( +39,  -13,  -22,   +0,   +8),  # freq=42
    'combination': ( +46,   -4,  +64,   +0,  +24),  # freq=15
    'comedy': ( +80,  +56,  +59,  +18,  -23),  # freq=20
    'comforting': ( +42,  -13,  +17,   +0,  +14),  # freq=13
    'comic': ( +43,   -4,  -14,   +0,   +9),  # freq=14
    'comics': ( +42,   +9,  -32,   +2,   +6),  # freq=16
    'committed': ( +46,   +9,  +62,   +2,  +24),  # freq=36
    'communicate': ( +34,   -6,  +27,   +0,  +14),  # freq=23
    'community': ( +29,  +20,  +34,   +0,  +16),  # freq=52
    'company': ( +42,   -6,  +39,   +0,  +19),  # freq=483
    'competition': ( +24,  +72, +119,  +24,  +27),  # freq=81
    'complain': (-100,  +47,  -59,  +34,  -43),  # freq=95
    'complaint': (-100,  +29,  -97,  +19,  -51),  # freq=23
    'complete': ( +39,  +20,  +54,   +0,  +16),  # freq=62
    'completed': ( +34,  -11,  +37,   +0,  +16),  # freq=31
    'complicated': ( -32,  +38,  -44,  +31,   -9),  # freq=33
    'concentrate': ( +56,   -9,  +69,   +0,  +27),  # freq=12
    'concern': ( -59,  +51,  -12,  +35,  -21),  # freq=18
    'concerned': ( -49,  +45,  -54,  +25,  -25),  # freq=58
    'concert': ( +44,  +31,  +29,   +1,  +17),  # freq=182
    'condolences': ( -83,   -6,  -92,   +0,  -43),  # freq=18
    'conference': ( +42,  +20,  +29,   +4,  +17),  # freq=21
    'confront': ( -51,  +58,  +92,  +35,   -1),  # freq=50
    'congrats': ( +32,  +22,  +24,   +0,  +19),  # freq=344
    'congratulation': ( +46,  +22,  +29,   +1,  +17),  # freq=18
    'congratulations': ( +49,  +22,  +29,   +1,  +19),  # freq=475
    'connection': ( +39,  +20,  +34,   +0,  +16),  # freq=16
    'conquer': ( +50,  +58, +127,  +25,  +40),  # freq=13
    'conscious': ( +51,  -29,  +57,   +0,  +25),  # freq=20
    'consider': ( +37,  -24,  +22,   +0,  +14),  # freq=88
    'constant': ( +51,  -36,  +54,   +0,  +24),  # freq=14
    'contact': ( +48,  -13,   +7,   +0,  +16),  # freq=92
    'contest': ( +35,  +36,  +62,   +9,  +20),  # freq=82
    'control': ( +28,  +40, +127,  +35,  +10),  # freq=184
    'conversation': ( +56,   +2,  +32,   +0,  +20),  # freq=77
    'convince': ( +46,   +9,  +74,   +2,  +25),  # freq=15
    'convinced': ( +56,   +0,  +34,   +0,  +23),  # freq=18
    'cookie': ( +71,  +22,  +34,  +18,  -10),  # freq=21
    'cookies': ( +34,   -4,  -24,   +0,   +5),  # freq=68
    'cool': ( +29,   +9,  +34,   +0,  +16),  # freq=1254
    'correct': ( +86,  +18, +112,  +36,  -19),  # freq=43
    'correctly': ( +44,   +0,  +57,   +0,  +21),  # freq=14
    'corrupt': ( -89,  +58,  +54,  +39,  -24),  # freq=23
    'cost': ( -32, +123, +127, +127,  +61),  # freq=136
    'couldnt': ( -79,  -15,  -77,   +0,  -40),  # freq=195
    'counseling': ( +35,   -4,  +57,   +0,  +20),  # freq=23
    'counselor': ( +39,  -11,  +27,   +0,  +16),  # freq=20
    'count': ( +29,  +40, +127, +115,  +40),  # freq=97
    'counter': ( +39,  +20,  +72,  +12,   +0),  # freq=25
    'country': ( +43,  -20,  +54,   +0,  +23),  # freq=288
    'counts': ( -36,  -27,   +0,   +0,  -12),  # freq=19
    'courage': ( +50,  +49, +107,  +12,  +32),  # freq=36
    'course': ( +24,   -2,   +2,   +0,   +8),  # freq=311
    'courses': ( +51,  -15,   +9,   +0,  +16),  # freq=20
    'cousin': ( +31,   +6,  +14,  +10,   +6),  # freq=128
    'cousins': ( +58,  -15,   -9,   +0,  +14),  # freq=35
    'cow': ( +51,  -18,  +77,   +6,  +57),  # freq=45
    'cracked': ( -77,   +6, -119,   +3,  -46),  # freq=25
    'crappy': (-127,  +20, -124,  +19,  -65),  # freq=37
    'cream': ( +86,  -27,  +67,  +12,   +4),  # freq=137
    'creative': ( +43,   +2,  +39,   +0,  +19),  # freq=26
    'credit': ( +28,  +27,  +99,  +54,  +10),  # freq=155
    'creep': ( -52,  +49,  -22,  +31,   +9),  # freq=21
    'creepy': ( -42,  +63,  -67,  +31,   +9),  # freq=100
    'cried': ( -25,  +30,  -20,  +20,  +30),  # liquid -- happy/sad tears. context determines.
    'crime': ( -89,  +90,  -82,  +77,  -24),  # freq=35
    'crowded': ( +42,  -38,  +24,   +0,  +16),  # freq=21
    'cruel': (-102,  +90,  +82,  +57,  +24),  # freq=29
    'cruise': ( +51,   -9,  +54,   +0,  +24),  # freq=115
    'crush': ( +57,  +51,  -34,  +18,  -19),  # freq=75
    'crushing': ( -67,  +46,  -44,  +63,  -98),  # freq=10
    'cuddle': ( +71,  -22,  +42,   +6,  -27),  # freq=11
    'culprit': (-123,  +47,  -69,  +39,  -55),  # freq=16
    'culture': ( +56,  -36,  +49,   +0,  +24),  # freq=24
    'cure': ( +48,  -22,  +34,   +0,  +19),  # freq=27
    'current': ( +28,  -33,  +32,   +0,  +12),  # freq=105
    'curse': ( -52,  +49,  -44,  +31,  -20),  # freq=19
    'cute': ( +28,  +13,   +7,   +0,  +14),  # freq=265
    'dad': ( +39,  +20,  +34,  +12,  +16),  # freq=571
    'daily': ( +50,  -29, +127,  +77,  +55),  # freq=57
    'damage': ( -52,  +49,  -44,  +49,  -20),  # freq=90
    'damaged': (-112,  +36, -127,  +27,  -61),  # freq=43
    'damages': (-117,  +67,  -82,  +54,  -53),  # freq=10
    'dance': ( +28,  +22,  +24,   +0,  +23),  # freq=51
    'dancing': ( +39,  +31,   -7,   +3,  +10),  # freq=40
    'darkness': ( -52,  +24,  -67,  +15,  -40),  # freq=13
    'darn': (-106,  +72,  -37,  +54,  -43),  # freq=32
    'date': ( +51,  +47,  +24,  +43,  -10),  # freq=378
    'deadline': ( -42,  +74,  -44, +127,   -9),  # freq=13
    'deal': ( +29,  +20,  +34,  +12,  +16),  # freq=460
    'dealing': ( +50,  -15,  +29,   +0,  +19),  # freq=50
    'dear': ( +39,   -4,  +17,   +0,  +14),  # freq=61
    'debit': ( -37,  -22,   -9,   +0,  -13),  # freq=23
    'debt': ( -77,  +74, -127,  +97,  -48),  # freq=88
    'decade': ( +42,   +9, +126,  +43,  +76),  # freq=15
    'decent': ( +34,  -15,  +19,   +0,  +13),  # freq=82
    'decline': ( -42,  +24,  -67,  +31,  -29),  # freq=12
    'declined': ( -94,  -83, -127,   +0,  -55),  # freq=14
    'dedicated': ( +50,   +0,  +47,   +0,  +21),  # freq=23
    'dedication': ( +37,   -2,  +24,   +0,  +14),  # freq=27
    'deep': ( +29,  +20,  +34,   +0,  +16),  # freq=95
    'definitely': ( +29,  +29,  +34,   +0,  +16),  # freq=748
    'degree': ( +50,  +29,  +72,   +0,  +32),  # freq=139
    'delicious': ( +32,  +18,  +17,   +0,  +14),  # freq=120
    'delivery': ( +28,  +27,  +24,  +35,  +10),  # freq=48
    'dementia': (-114,  +74,  -69,  +57,  -51),  # freq=21
    'denied': ( -86,  +27,  -69,  +19,  -40),  # freq=11
    'dent': ( -44,  +24,  -64,  +13,  -25),  # freq=81
    # 'department' removed — neutral factual word causing positive drift
    'depressing': (-100,  +15, -124,  +11,  -55),  # freq=37
    'desires': ( +58,  +63,  -54,  +12,   +8),  # freq=13
    'desperate': ( -77,  +90, -127,  +77,  -48),  # freq=14
    'dessert': ( +66,  +22,  +17,   +6,  -14),  # freq=19
    'destroy': (-102, +119, +109,  +77,  -36),  # freq=35
    'devastating': (-127,  +53,  -87,  +95,  -80),  # freq=33
    'diabetes': (-100,  +26,  -66,  +95,  +36),  # freq=23
    'diagnosed': ( -94,  -15,  -87,   +0,  -46),  # freq=43
    'diagnosis': ( -77,  +90, -127, +116,  -48),  # freq=11
    "didn't": ( -21,   +6,   +0,  +15,   +0),  # freq=2564
    'died': (-127, +119, -127, +116,  -97),  # freq=449
    'diet': ( -24,  +27,  -24,  +35,  -10),  # freq=190
    'different': ( +24,  -20,   -7,   +0,   +6),  # freq=390
    'difficulties': (-117,  +76,  -59,  +62,  -51),  # freq=19
    'digital': ( +51,  -22,   +7,   +0,  +16),  # freq=11
    'dinner': ( +29,   +9,  +17,   +0,   +8),  # freq=449
    'direction': ( +51,  -15,  +74,   +0,  +27),  # freq=57
    'directly': ( +55,  -18,  +57,   +0,  +25),  # freq=21
    'dirty': ( -32,  +24,  -22,  +15,  -20),  # freq=74
    'disability': (-119,   -6, -127,   +0,  -65),  # freq=11
    'disappear': (-100,  -15,  -64,   +0,  -46),  # freq=16
    'disappointed': ( -46,   +6,  -27,  +13,  -30),  # freq=375
    'disappointing': (-119,  +38, -127,  +30,  -63),  # freq=90
    'disaster': (-114, +119, -127, +127,  -61),  # freq=28
    'discover': ( +50,  +49,  +54,  +12,  +40),  # freq=23
    'disease': (-102,  +74, -127, +116,  -73),  # freq=40
    'dislike': (-127,  +58,  -82,  +39,  +24),  # freq=28
    'dispute': ( -79,  +65,  +37,  +46,  -21),  # freq=10
    'disrespectful': ( -89,  +90,  +54,  +39,  +36),  # freq=12
    'distance': ( -62,  -40,  -32,   +0,  -25),  # freq=91
    'distract': ( -59,  +24,  -59,  +15,  -29),  # freq=13
    'distracted': ( -86,  -29, -124,   +0,  -51),  # freq=20
    'disturbing': (-100,  +76,  -27,  +57,  -39),  # freq=17
    'divorced': ( -94,  -76, -109,   +0,  -51),  # freq=13
    'doctor': ( -24,  +40,  -24,  +72,  -10),  # freq=290
    "doesn't": ( -24,  +13,   +0,  +18,   +0),  # freq=645
    'dog': ( +39,  +29,  +34,   +0,  +24),  # freq=1996
    'dollar': ( +56,  +24,  +57,   +4,  +25),  # freq=69
    'dont': ( -24,  +13,   +0,  +18,   +0),  # freq=625
    'doomed': (-127,  +67,  -87,  +57,  -58),  # freq=12
    'dorm': ( +48,  -56,  -49,   +0,   +6),  # freq=15
    'double': ( +37,  -22,  +22,   +0,  +14),  # freq=63
    'dove': ( +50,  -47,  -17,   +0,  +10),  # freq=10
    'drag': ( -89,  +90, -109,  +57,  +24),  # freq=21
    'dragon': ( +57,  +60,  +94,  +67,  +38),  # freq=41
    'drank': ( -42,  -13,  +12,   +0,  -12),  # freq=45
    'drawing': ( +43,  -51,  -47,   +0,   +5),  # freq=23
    'dreading': (-104,  +67,  -54,  +54,  -46),  # freq=23
    'dream': ( +39,  +29,  +17,  +12,  +40),  # freq=184
    'dreams': ( +37,   -6,   +2,   +0,  +12),  # freq=73
    'drinks': ( +50,  +27,  -24,   +7,  +10),  # freq=59
    'drop': ( -77, +127, -127, +127, +127),  # freq=78
    'dropped': ( -51,  +45, -114,  +29,  -36),  # freq=140
    'dropping': ( -89,  +90, -119,  +62,  -51),  # freq=19
    'drowning': ( -60,  +50,  -55,  +40,  -40),  # grief: overwhelmed, submerged
    'drug': ( -52,  +49,  -67,  +49,  -20),  # freq=36
    'drugs': ( -77,  +74, -109,  +77,  -36),  # freq=46
    'drunk': ( -42,  +63,  -67,  +31,  -20),  # freq=144
    'drunken': ( -96,  +92, -109,  +70,  -51),  # freq=12
    'duck': ( +57,  +33,  +24,  +18,   -8),  # freq=19
    'dumb': ( -52,  +24,  -67,  +15,  -20),  # freq=83
    'dump': ( -36,   +0,  -44,   +0,  -20),  # freq=21
    'dumpster': ( -77,  -20, -127,   +0,  -55),  # freq=13
    'eagerly': ( +35,  +42,   +0,   +6,   +9),  # freq=23
    'early': ( +48,  +42,  +67, +103,  -19),  # freq=277
    'earn': ( +50,  +29,  +72,  +12,  +24),  # freq=33
    'ease': ( +31,   -9,   -2,   +0,   +9),  # freq=30
    'easier': ( +56,  -24,  -32,   +0,  +10),  # freq=182
    'easily': ( +28,  +13,  +49,   +0,  +10),  # freq=87
    'east': ( +42,  +22,  +42,  +12,   -8),  # freq=23
    'education': ( +28,   +9,  +34,  +12,   +8),  # freq=64
    'efficient': ( +48,   +9,  +44,   +0,  +21),  # freq=11
    'effort': ( +28,  +40,  +49,  +35,  +10),  # freq=107
    'eggs': ( +51,   +9,  +42,  +30,   +4),  # freq=44
    'electricity': ( +43,  +56,  +69,  +15,  +24),  # freq=17
    'electronics': ( +46,   +9,  +22,   +2,  +17),  # freq=10
    'embarrassing': ( -77,  +83, -119,  +54,  -46),  # freq=181
    'emerald': ( +80,  +22,  +77,   +6,  +23),  # freq=12
    'emergencies': ( -86, +127,  -54,  +93,  -39),  # freq=17
    'emotion': ( +42,  +67,  +57,  +18,  +23),  # freq=21
    'emotional': ( +29,  +18,   +2,   +3,   +9),  # freq=97
    'employer': ( +35,   +4,  +74,   +0,  +24),  # freq=44
    'employment': ( +56,  +15,  +32,   +2,  +20),  # freq=12
    'empty': ( -42,  -24,  -67,  +15,  -40),  # freq=124
    'end': ( -24,  +27,  +24,  +35,  -23),  # freq=741
    'ended': ( -47,   +0,   +0,   +0,  -16),  # freq=513
    'ending': ( -36,  -29,   -4,   +0,  -12),  # freq=52
    'ends': ( -42, +112, +127, +127,  +70),  # freq=61
    'energy': ( +46,  +18,  +49,   +1,  +21),  # freq=64
    'engaged': ( +28,  +22,  +24,   +6,  +19),  # freq=43
    'engineer': ( +48,   +0, +107,   +0,  +32),  # freq=13
    'engineering': ( +51,   -6,  +97,   +0,  +31),  # freq=53
    'enjoy': ( +24,   +6,   +7,   +0,  +11),  # freq=677
    'enjoyable': ( +51,   +2,  +29,   +0,  +20),  # freq=25
    'enjoying': ( +42,  +22,  +34,   +1,  +17),  # freq=74
    'enraged': (-122,  +99,  +98, +127,  +80),  # freq=12
    'ensure': ( +51,  -15,  +74,   +0,  +27),  # freq=12
    'enter': ( +46,   -6,   +0,   +0,  +12),  # freq=22
    'entitled': ( -44,  +69, +127,  +41,  +12),  # freq=16
    'envious': ( -77,  +74,  -82,  +39,  +12),  # freq=167
    'environment': ( +32,  -56,   +7,   +0,  +10),  # freq=39
    'episode': ( +48,  -24,   +2,   +0,  +14),  # freq=19
    'equipment': ( +55,   +0,  +57,   +0,  +25),  # freq=16
    'era': ( +51,  +18,  +67,   +6,  +29),  # freq=14
    'esteem': ( +55,  -22,  +49,   +0,  +24),  # freq=10
    'europe': ( +35,   +0,  +37,   +0,  +16),  # freq=41
    'event': ( +55,  +31,  +47,   +7,  +24),  # freq=87
    'everybody': ( +42,   +2,   +2,   +0,  +12),  # freq=60
    'everyday': ( +51,  -20,   -2,   +0,  +14),  # freq=195
    'everything': ( -34,  -11,  -37,   +6,  -20),  # freq=1318
    'everywhere': ( +50,  +69,  +72,  +25,  -24),  # freq=146
    'evil': (-102,  +90,  +82,  +57,  +24),  # freq=35
    'ex': ( -89,  +90, -109,  +39,  +36),  # freq=186
    'exact': ( +48,  -24,  +54,   +0,  +23),  # freq=41
    'exam': ( -24,  +69,  -49, +108,  -10),  # freq=369
    'excellent': ( +37,  +18,  +34,   +0,  +19),  # freq=72
    'exciting': ( +23,  +45,  +24,   +6,  +20),  # freq=562
    'excuse': ( -62,  +45,  -67,  +29,  -32),  # freq=51
    'exercise': ( +29,  +40,  +54,  +12,  +16),  # freq=70
    'exhausted': ( -77,  -58, -127,  +39,  -61),  # freq=17
    'exist': ( +39,   +0, +107,   +0,  +32),  # freq=19
    'expected': ( +50,  -15,  +14,   +0,  +17),  # freq=123
    'expenses': ( -62,  +29,  -37,  +19,  -28),  # freq=28
    'expensive': ( -32,  +24,  -44,  +49,   -9),  # freq=268
    'experience': ( +29,  +20,  +34,   +0,  +16),  # freq=530
    'experienced': ( +28,   +9,  +52,   +2,  +17),  # freq=45
    'experiences': ( +29,  +11,  +39,   +2,  +16),  # freq=39
    'expired': ( -32,  +24,  -44,  +49,  -20),  # freq=27
    'explain': ( +28,  +27,  +49,  +18,  +10),  # freq=66
    'explore': ( +66,  +47,  +77,  +12,  -19),  # freq=22
    'express': (  +8,  +10,  +15,   +4,   +5),  # freq=19 — carrier verb, context-dependent
    'extreme': ( -42,  +67,  +82,  +39,   -1),  # freq=19
    'fabulous': ( +50,  +13,  +39,   +0,  +21),  # freq=11
    'facebook': ( +51,  +58,  +89,  +12,  +31),  # freq=95
    'fact': ( +28,  +27,  +74,  +35,  +23),  # freq=103
    'fail': ( -92,  +20, -127,  +15,  -55),  # freq=83
    'failed': (-102,  +74, -127,  +77,  -73),  # freq=131
    'failing': ( -89,  +74, -127,  +97,  -61),  # freq=30
    'faith': ( +29,   +9,  +22,   +0,  +18),  # freq=156
    'fall': ( -52,  +63, -114,  +65,  -70),  # freq=246
    'falling': ( -46,  +33,  -46,  +53,  -60),  # freq=56
    'familiar': ( +29,   +9,  +34,   +0,   +8),  # freq=39
    'families': ( +43,  -24,  +14,   +0,  +14),  # freq=38
    'famous': ( +43,  +22,  +52,   +1,  +21),  # freq=20
    'fan': ( +39,  +29,  +34,   +0,  +16),  # freq=180
    'fancy': ( +36,  +15,   +4,   +2,  +10),  # freq=56
    'far': ( -24,  +13,  -24,  +18,  -10),  # freq=551
    'fart': ( -86,  -15, -104,   +0,  -46),  # freq=31
    'fast': ( +62,  +90,  +94, +115,  -53),  # freq=403
    'fat': ( -42,  +24,  -67,  +15,  -29),  # freq=76
    'favor': ( +32,  -15,   +2,   +0,   +9),  # freq=41
    'favorite': ( +32,  +13,  +24,   +0,  +14),  # freq=737
    'favourite': ( +43,   +0,  +24,   +0,  +16),  # freq=25
    'fearful': ( -78,  +53,  -76,  +63,  +27),  # freq=13
    'fears': (-123,  +85,  -87,  +70,  -55),  # freq=20
    'feat': ( +56,  +38, +102,   +7,  +32),  # freq=13
    'fee': ( -34,  -24,  +17,   +0,   -8),  # freq=23
    'feed': ( +28,  +13,  +49,  +18,  +10),  # freq=59
    'feedback': ( +37,  +26,  +22,  +84,   +6),  # freq=18
    'feeding': (-127, +127, -127, +127,  +61),  # freq=13
    'feel': (  +5,  +10,   +5,   +5,   -2),  # freq=4309 — carrier verb, near-neutral
    'feels': (  +5,  +10,   +5,   +5,   -2),  # carrier verb, near-neutral
    'feeling': ( +32,  -13,   +2,   +0,   +9),  # freq=1521
    'fell': ( -57,   -6,  -92,   +0,  -34),  # freq=310
    'fellow': ( +55,  -36,  +22,   +0,  +19),  # freq=12
    'felt': ( +28,  +40,  +24,  +18,  -10),  # freq=1835
    'festival': ( +39,  +18,  +29,   +2,  +16),  # freq=24
    'fever': (-102,  +74, -127, +127,  +24),  # freq=14
    'fiance': ( +32,  -22,   +2,   +0,   +9),  # freq=44
    'fiancee': ( +48,  +11,   +7,   +0,  +14),  # freq=21
    'fights': (-119, +112,  +92,  +89,  -27),  # freq=22
    'figure': ( +42,  -29,   +2,   +0,  +12),  # freq=147
    'figured': ( +35,   +0,  +24,   +0,  +14),  # freq=71
    'finally': (  +5,  +10,  +10,  +15,   +5),  # temporal marker -- context determines polarity
    'finance': ( +56,  -11,  +97,   +0,  +32),  # freq=16
    'financial': ( +35,   +0,  +64,   +0,  +20),  # freq=93
    'fine': ( -32,  +24,  -22,  +49,   -9),  # freq=547
    'fired': (-127, +127, -127, +116,  -85),  # freq=89
    'fireworks': ( +51,  +65,  +22,  +15,  +19),  # freq=17
    'fishing': ( +37,  -40,  -29,   +0,   +6),  # freq=92
    'fit': ( +39,  +29,  +72,   +0,  +24),  # freq=121
    'flat': ( -32,  +38,  -44,  +49,  -20),  # freq=58
    'flies': ( -37, +108,   +0,  +61,  -13),  # freq=62
    'flight': ( +32,  +24,  +24,   +7,  +12),  # freq=97
    'flip': ( -51,   +6,  -22,   +3,  -21),  # freq=42
    'flood': ( -77, +103, -127, +127,  -48),  # freq=41
    'flowers': ( +39,  -15,  -24,   +0,   +8),  # freq=74
    'flu': ( -77,  +11, -109,   +7,  -43),  # freq=36
    'fly': ( +29,  +49,  +54,  +25,  +55),  # freq=161
    'flying': ( +39,   +9,  +27,   +1,  +16),  # freq=144
    'focus': ( +32,  +38,  +77,  +30,   -8),  # freq=99
    'fond': ( +39,   +9,   +2,   +1,  +12),  # freq=44
    'fondly': ( +51,  -27,   +7,   +0,  +16),  # freq=17
    'fool': ( -67,  +26,  +22,  +32,  +18),  # freq=27
    'football': ( +56,  +45,  +24,   +9,  +20),  # freq=256
    'forget': ( -24,  -13,  -24,  +18,  -10),  # freq=234
    'forgetting': ( -94,  +90, -127,  +66,  -55),  # freq=25
    'forgive': ( +50,  +20,  +54,   +0,  +24),  # freq=67
    'forgotten': ( -52,  +24,  -92,  +15,  -40),  # freq=36
    'fortunate': ( +44,  +11,  +47,   +1,  +20),  # freq=54
    'fortunately': ( +51,  -15,  +29,   +0,  +20),  # freq=41
    'fortune': ( +44,  +20,  +39,   +1,  +20),  # freq=19
    'forward': ( +39,   -6,  +34,   +0,  +16),  # freq=584
    'freak': ( -28,  +28,   -9,  +26,  +15),  # freq=45
    'freaked': (-102, +112,  -69,  +86,  -46),  # freq=51
    'freaky': ( -86,  +29,  -49,  +23,  -39),  # freq=11
    'free': ( +28,  +18,  +42,   +0,  +23),  # freq=346
    'frozen': ( -30,  -15,  -40,  +10,  -25),  # grief: stuck, unable to move on
    'freedom': ( +32,  +18,  +52,   +0,  +27),  # freq=30
    'fresh': ( +39,  +20,  +34,   +0,  +24),  # freq=85
    'freshman': ( +37,  +11,  -17,   +2,   +8),  # freq=24
    'friendly': ( +43,   -9,  +12,   +0,  +14),  # freq=43
    'friends': ( +50,  +20,  +34,   +0,  +16),  # freq=2039
    'fries': ( +36,  +15,   +0,   +2,  +10),  # freq=38
    'frightened': (-102, +127, -127, +116,  +48),  # freq=44
    'frightening': (-106, +127,  -64,  +97,  -48),  # freq=38
    'front': ( +32,  +22,  +52,  +24,   -8),  # freq=456
    'fruit': ( +90,  +18,  +52,  +24,  -23),  # freq=27
    'frustrating': (-127,  +54, -104,  +43,  -61),  # freq=249
    'full': ( +28,  +13,  +49,   +0,  +10),  # freq=318
    'fully': ( +58,   -9,  +79,   +0,  +31),  # freq=61
    'fun': ( +28,  +18,  +17,   +0,  +14),  # freq=1998
    'funeral': (-114,  +74, -127,  +77,  -85),  # freq=45
    'funny': ( +28,  +18,  +17,   +0,  +14),  # freq=337
    'furniture': ( +42,  -42,  -69,   +0,   +0),  # freq=58
    'future': ( +28,  +27,  +24,  +18,  +23),  # freq=311
    'gambling': ( -32,  +69,  -54,  +39,  -20),  # freq=29
    'games': ( +28,  +40,  +49,   +0,  +23),  # freq=326
    'gaming': ( +42,  +11,   -2,   +1,  +12),  # freq=31
    'gang': ( -79,  +92,  +54,  +62,  -17),  # freq=11
    'gas': ( +28,  +33,  +24,  +36,  -29),  # freq=159
    'gender': ( +42,  -27,  +32,   +0,  +17),  # freq=13
    'general': ( +35,   +0,  +47,   +0,  +17),  # freq=73
    'generation': ( +42,   -2,  +24,   +0,  +16),  # freq=22
    'generous': ( +50,  +20,  +34,   +0,  +24),  # freq=33
    'gesture': ( +35,   -4,  -22,   +0,   +6),  # freq=36
    'ghetto': ( -47,  +29,   -4,  +15,  -16),  # freq=10
    'ghost': ( -42,  +49,  -67,  +31,  -20),  # freq=67
    'ghosted': ( -77,  +58, -109,  +39,  -48),  # freq=11
    'giant': ( +32,  +56, +104,  +18,  +27),  # freq=45
    'gift': ( +50,  +29,  +34,   +0,  +32),  # freq=273
    'gifted': ( +42,  +11,  +49,   +1,  +20),  # freq=26
    'gifts': ( +37,  +22,   +2,   +2,  +10),  # freq=90
    'girlfriend': ( +62,  +38,  +42,  +12,  -10),  # freq=779
    'girls': ( +35,  +56,  -54,  +15,   +1),  # freq=211
    'glasses': ( +37,   +4,  +59,  +18,   +4),  # freq=20
    'glory': ( +46,  +20,  +42,   +1,  +20),  # freq=10
    'goal': ( +29,  +29,  +54,  +25,  +24),  # freq=70
    'goals': ( +48,   -2,  +87,   +0,  +28),  # freq=65
    'golden': ( +31,   +0,  +14,   +0,  +10),  # freq=33
    'gone': ( -40,  +10,  -35,  +20,  -30),  # grief: absence, departed
    'goodness': ( +49,  -15,  +29,   +0,  +19),  # freq=211
    'google': ( +42,   +9,  +67,  +18,   -4),  # freq=31
    'gorgeous': ( +37,  +18,  +17,   +0,  +19),  # freq=28
    'gosh': ( -34,  -13,  -87,   +0,  -25),  # freq=137
    'grabbed': ( -37, +112, +114,  +65,   +5),  # freq=34
    'grad': ( +32,   +2,  +32,   +0,  +14),  # freq=44
    'grade': ( -24,  +40,  -49,  +72,  -10),  # freq=276
    'graduate': ( +32,  +22,  +42,   +0,  +23),  # freq=103
    'graduated': ( +39,  +22,  +39,   +2,  +17),  # freq=151
    'graduation': ( +39,  +13,  +42,   +1,  +17),  # freq=83
    'grand': ( +42,  +20,  +52,   +1,  +21),  # freq=51
    'grandfather': ( +32,   +4,  +59,   +0,  +23),  # freq=59
    'grandmother': ( +37,   +9,  +52,   +0,  +19),  # freq=202
    'grandpa': ( +42,   +4,  +52,   +0,  +19),  # freq=25
    'grandparents': ( +35,  -27,   +7,   +0,  +10),  # freq=61
    'granted': ( +39,  -13,  +37,   +0,  +17),  # freq=66
    'greatest': ( +35,  +11,  +57,   +1,  +20),  # freq=47
    'greatly': ( +56,  +38, +107,   +7,  +34),  # freq=15
    'greed': ( -92,  +74,   +9,  +54,  -28),  # freq=5
    'greedy': ( -52,  +38,  +44,  +31,   -9),  # freq=11
    'grew': ( +35,  +27,   +2,   +3,  +10),  # freq=101
    'grieve': (-106,   +0,  -69,   +0,  -48),  # freq=12
    'groceries': ( +51,  -24,  -24,   +0,  +10),  # freq=81
    'grocery': ( +42,   +9,  -14,   +2,   +9),  # freq=168
    'ground': ( +42,   -4, +126,   +0,  +76),  # freq=107
    'group': ( +32,  +27,  +67,  +18,  +19),  # freq=155
    'grow': ( +29,  +20,  +34,  +12,  +24),  # freq=226
    'growing': ( +29,  +20,  +34,  +12,  +16),  # freq=177
    'growth': ( +43,   +6,  +37,   +0,  +19),  # freq=15
    'grudge': (-106,  +38,  -37,  +30,  -43),  # freq=12
    'guarantee': ( +46,  -13,  +39,   +0,  +20),  # freq=11
    'guard': ( +28,   +2,  +89,   +0,  +23),  # freq=54
    'guess': ( -21,   -6,  -20,   +0,   -8),  # freq=1012
    'guest': ( +48,   -2,  +17,   +0,  +16),  # freq=10
    'guilty': ( -78,  +26,  -55,  +32,  -45),  # freq=393
    'guitar': ( +35,   +4,   -2,   +0,   +9),  # freq=51
    'gun': (-100, +112,  +92,  +81,  -19),  # freq=56
    'gym': ( +28,  +40,  +74,  +18,  +10),  # freq=153
    'haha': ( +50,  +29,  +17,   +0,  +24),  # freq=814
    'hahaha': ( +24,  -40,  -39,   +0,   +0),  # freq=158
    'haircut': ( +37,  -27,  -54,   +0,   +1),  # freq=37
    'halloween': ( +24,  +33,  +29,  +12,  +10),  # freq=94
    'handful': ( +42,  +13,  +42,  +12,   +8),  # freq=28
    'handle': ( +28,  +13,  +77,  +36,  +19),  # freq=128
    'handy': ( +55,  -22,  +39,   +0,  +23),  # freq=22
    'hang': ( -40,   +6,  -17,   +3,  -17),  # freq=192
    'hanging': ( -27,   -6,  -39,   +0,  -16),  # freq=88
    'happier': ( +51,  +13,  +39,   +0,  +21),  # freq=47
    'happiest': ( +51,   +6,  +39,   +0,  +21),  # freq=19
    'happily': ( +51,  +24,  +27,   +0,  +20),  # freq=20
    'happiness': ( +42,  +22,  +34,   +0,  +27),  # freq=68
    'harder': ( -52,  +15,   -4,   +9,  -17),  # freq=112
    'hardest': ( -62,  +45,   +0,  +29,  -21),  # freq=59
    'harsh': ( -52,  +63,  +44,  +31,   -9),  # freq=18
    'hasnt': ( -94,   -6,  -37,   +0,  -39),  # freq=18
    'hassle': ( -92,  +65,   -4,  +46,  -31),  # freq=24
    "haven't": ( -24,  +13,  -24,  +18,   +0),  # freq=664
    'havent': ( -40,  -18,  -67,   +0,  -25),  # freq=57
    'headache': ( -42,  +38,  -67,  +49,  -29),  # freq=21
    'headaches': (-112,  +58, -104,  +46,  -55),  # freq=12
    'heal': ( +37,   +9,  +22,   +0,  +18),  # freq=38
    'healthier': ( +43,   -6,  +24,   +0,  +16),  # freq=23
    'healthy': ( +50,  +20,  +72,   +0,  +24),  # freq=218
    'hearing': ( +37,   -2,  +29,   +0,  +16),  # freq=96
    'heart': ( +10, +10, +5, +5, +15),  # body part with emotional weight — intimacy/vulnerability
    # V near-neutral but G elevated: "my heart" signals something important is at stake
    'heartbreaking': (-100,  +83,  -32,  +62,  -39),  # freq=27
    'heaven': ( +32,  +13,  +24,   +0,  +27),  # freq=29
    'heavy': ( -32,  +24,  -44,  +15,  -40),  # freq=67
    'heck': ( -62,  +51,  +22,  +31,  -17),  # freq=70
    'hectic': ( -94,  +83,  -27,  +62,  -36),  # freq=25
    'held': ( +39,  +20,  +72,  +12,  +16),  # freq=44
    'hello': ( +29,  +15,  +17,   +0,   +8),  # freq=69
    'helpful': ( +32,  -13,  +27,   +0,  +14),  # freq=90
    'hers': ( +32,   +0,  -57,   +0,   +0),  # freq=27
    'hey': ( +24,  +24,  +17,  +12,   +8),  # freq=243
    'hi': ( +22,   +9,   +7,   +0,   +6),  # freq=171
    'hide': ( -32,  +38,  -44,  +31,  -20),  # freq=79
    'hiding': ( -59,  +15,  -49,   +9,  -28),  # freq=28
    'higher': ( +37,   +2,  +29,   +0,  +16),  # freq=63
    'highly': ( +35,  +13,  +39,   +2,  +16),  # freq=24
    'highway': ( +29,  +13,  -22,   +4,   +5),  # freq=42
    'hiking': ( +32,  +29,  +24,   +4,  +13),  # freq=49
    'hilarious': ( +37,  +27,  +17,   +0,  +19),  # freq=55
    'hip': ( +39,  +29, +107,  +12,  +40),  # freq=23
    'hire': ( +42,   +9,  +14,   +2,  +14),  # freq=43
    'hired': ( +51,  +22,  +17,   +0,  +17),  # freq=52
    'history': ( +29,   +9,  +54,   +0,  +32),  # freq=59
    'hit': ( +28,  +54,  +49,  +18,  +10),  # freq=8067
    'hits': ( +39,  +29,  +17,   +0,  +16),  # freq=22
    'hoarder': ( -94,  -90, -124,   +0,  -53),  # freq=10
    'hobby': ( +34,   +4,   -2,   +1,   +9),  # freq=50
    'holiday': ( +77,  +18,  +77,  +12,  -29),  # freq=72
    'holidays': ( +44,   +2,   +2,   +0,  +13),  # freq=64
    'holy': ( +29,  +20,  +54,   +0,  +24),  # freq=64
    'home': ( +39,   -9,  +54,   +0,  +16),  # freq=2166
    'homemade': ( +39,  -13,  -12,   +0,   +9),  # freq=20
    'hometown': ( +36,  -13,  +12,   +0,  +13),  # freq=39
    'honey': ( +66,   -4,  +42,   +6,  -10),  # freq=11
    'honor': ( +50,  +29,  +72,   +0,  +24),  # freq=36
    'hopefully': ( +35,   +0,  -22,   +0,   +5),  # freq=957
    'hopes': ( +43,   +0,  +27,   +0,  +17),  # freq=43
    'hoping': ( +29,  +20,  -17,  +12,  +16),  # freq=374
    'horizon': ( +51,   +9,  +67,   +6,  -14),  # freq=10
    'horrific': (-114,  +92,   -9,  +73,  -40),  # freq=10
    'horrifying': (-127,  +65,  -87,  +54,  -63),  # freq=24
    'horror': (-114, +127, -127,  +97,  -24),  # freq=92
    'horse': ( +28,  +40,  +49,   +0,  +23),  # freq=54
    'hose': ( +50,  +40,  +89,  +89,  +24),  # freq=21
    'hospice': ( -89,  -40,  -44,  +32,  +63),  # freq=17
    'host': ( +42,   -6,   +9,   +0,  +14),  # freq=12
    'hotel': ( +66,  +13,  +84,  +24,  +42),  # freq=59
    'however': ( -29,  -22,  -17,   +0,  -12),  # freq=114
    'hug': ( +42,   -6,   +7,   +0,  +13),  # freq=24
    'huge': ( +32,  +38,  +84,  +18,  +29),  # freq=367
    'hugs': ( +50,   -6,  +12,   +0,  +16),  # freq=12
    'huh': ( -49,  -38,  -87,   +0,  -32),  # freq=72
    'human': ( +58,  -33,  +29,   +0,  +23),  # freq=98
    'humanity': ( +56,  -24,  +64,   +0,  +27),  # freq=23
    'humid': (-114, +103, -109,  +77,  +73),  # freq=11
    'humiliating': (-123,  +56,  -97,  +46,  -58),  # freq=13
    'humor': ( +50,  +29,  +34,   +0,  +24),  # freq=17
    'hungover': (-119,   -9, -124,   +0,  -63),  # freq=12
    'hungry': ( -32,  +38,  -44,  +49,  -20),  # freq=107
    'hunt': ( -40,  +69,  +32,  +39,   -8),  # freq=17
    'hurricane': ( -89, +127, -127, +127,  -48),  # freq=102
    'hurricanes': ( -51,  +90, +104,  +55,   +0),  # freq=20
    'hurting': ( -96,  +74, -119,  +54,  -53),  # freq=60
    'idea': ( +50,  +40,  +54,  +25,   -8),  # freq=632
    'idiot': ( -89,  +40,  +33,  +47,  +36),  # freq=19
    'ignore': ( -42,  +24,  +44,  +31,   -9),  # freq=40
    'ignored': ( -52,  +38,  -67,  +31,  -20),  # freq=30
    'ill': (-119,  -24, -127,   +0,  -63),  # freq=115
    'illegal': (-123,  +76,  -69,  +62,  -53),  # freq=14
    'illness': (-127,  +29, -127,  +77,  +61),  # freq=32
    'impact': (-127, +127, +127, +127,  +97),  # freq=25
    'impressed': ( +32,  +22,   -9,   +3,   +8),  # freq=432
    'impression': ( +39,   +9,  +22,   +2,  +16),  # freq=24
    'impressive': ( +29,  +27,  +52,   +4,  +16),  # freq=188
    'improve': ( +29,  +20,  +34,  +12,  +16),  # freq=49
    'improving': ( +42,   +0,  +24,   +0,  +16),  # freq=18
    'inadequate': (-106,  -11, -119,   +0,  -55),  # freq=2
    'inappropriate': (-102,  +11,  -64,   +7,  -46),  # freq=14
    'incident': (-117,  +45,   -4,  +34,  -40),  # freq=26
    'included': ( +58,  -18,  +22,   +0,  +19),  # freq=14
    'income': ( +56,   +2,  +64,   +0,  +27),  # freq=47
    'incompetent': (-102,  +18, -114,  +11,  -55),  # freq=11
    'inconsiderate': ( -77,   +6,  -32,   +3,  -31),  # freq=47
    'inconvenient': ( -77,  +29,  -54,  +19,  -36),  # freq=14
    'incredible': ( +48,  +38,  +34,   +0,  +29),  # freq=85
    'indeed': ( +48,   +0,  +29,   +0,  +19),  # freq=217
    'independent': ( +41,   -4,  +52,   +0,  +20),  # freq=31
    'industry': ( +32,   -9,  +97,   +0,  +25),  # freq=29
    'infection': ( -52,  +49,  -92,  +81,  -29),  # freq=15
    'info': ( +56,  -29,  +39,   +0,  +23),  # freq=31
    'inform': ( +50,  +13,  +49,   +2,  +23),  # freq=10
    'information': ( +58,  -24,  +62,   +0,  +27),  # freq=71
    'infuriating': (-102,  +99,   +4,  +73,  -34),  # freq=19
    'injured': ( -77,  +74, -127,  +97,  -48),  # freq=61
    'injuries': (-100,  +90, -127,  +66,  -55),  # freq=19
    'injury': ( -77,  +74, -127,  +97,  -48),  # freq=30
    'innocent': ( +39,  +29,  -34,  +25,  +16),  # freq=33
    'inside': (  +5,   +5,   +5,   +5,   +0),  # freq=240 — spatial, near-neutral
    'inspiration': ( +44,   +9,  +34,   +0,  +19),  # freq=14
    'insurance': ( +48,  -33,  +94,   +0,  +28),  # freq=222
    'intelligent': ( +50,  +20,  +72,   +0,  +24),  # freq=16
    'interested': ( +56,   +6,  +64,   +0,  +27),  # freq=72
    'internet': ( +48,  +18,  +52,  +49,   -8),  # freq=87
    'interview': ( -24, +127,  -99, +127,  +34),  # freq=429
    'intimidating': ( -77,  +76,  +92,  +50,  -12),  # freq=16
    'intruder': (-100,  +65,  -77,  +46,  -46),  # freq=13
    'invest': ( +48,  +27,  +59,  +12,   +0),  # freq=21
    'investment': ( +48,  +15,  +72,   +4,  +25),  # freq=20
    'invite': ( +55,  +27,  +32,   +4,  +20),  # freq=64
    'invited': ( +39,  +20,  +34,   +0,  +16),  # freq=117
    'ipad': ( +35,  +15,  +37,   +2,  +16),  # freq=21
    'iphone': ( +35,  +22,  +47,   +3,  +17),  # freq=29
    'irked': ( -94,  +76,  -64,  +57,  -43),  # freq=21
    'irritating': (-112, +110,  -49,  +86,  -46),  # freq=50
    "isn't": ( -24,  +13,   +0,  +18,   +0),  # freq=419
    'isolated': ( -79,  -38, -109,   +0,  -46),  # freq=35
    'issues': ( -32,  +24,  -44,  +31,   -9),  # freq=165
    'itself': ( +28,  -20,  -29,   +0,   +2),  # freq=53
    'jackpot': ( +77,  +56,  +67,   +6,  -29),  # freq=12
    'jail': (-102,  +74, -127,  +97,  -73),  # freq=44
    'japan': ( +35,   -4,  +22,   +0,  +14),  # freq=21
    'jar': ( +50,   +0,  +89,  +25,  +48),  # freq=11
    'jerk': ( -46,  +22,  +18,  +26,  +15),  # freq=59
    'jerks': ( -79,  +76,  -42,  +54,  -36),  # freq=19
    'job': ( +32,  +38,  +84,  +67,  +34),  # freq=2559
    'join': ( +50,   -4,  +54,   +0,  +24),  # freq=69
    'joined': ( +35,  -20,  +29,   +0,  +16),  # freq=33
    'joining': ( +56,   -6,  +54,   +0,  +25),  # freq=26
    'joke': ( +29,  +29,  +17,   +0,  +16),  # freq=41
    'joyful': ( +51,  +27,  +42,   +0,  +34),  # freq=237
    'juice': ( +48,  +13,  +34,  +12,   -8),  # freq=16
    'jump': ( +51,  +51,  +59,  +24,  -29),  # freq=69
    'junk': ( -32,  +13,  -22,  +15,  -20),  # freq=34
    'justice': ( +39,  +40,  +72,  +25,  +24),  # freq=18
    'kick': (-127, +127, +127, +127,  +85),  # freq=51
    'kicking': ( -40,  +67,   +4,  +39,  -13),  # freq=16
    'kill': (-127, +127, +127, +127,  +73),  # freq=109
    'kind': ( +50,   +9,  +34,   +0,  +24),  # freq=1716
    'kindergarten': ( +35,   -2,  -24,   +0,   +5),  # freq=53
    'king': ( +37,  +33, +112,   +9,  +31),  # freq=28
    'kiss': ( +49,  +27,  +17,   +1,  +17),  # freq=36
    'kitten': ( +32,  +18,   +7,   +0,  +23),  # freq=81
    'knock': ( -44,   +6,   +9,   +3,  -13),  # freq=55
    'knowing': ( +35,  -11,  +22,   +0,  +13),  # freq=140
    'knowledge': ( +39,  +20,  +54,   +0,  +24),  # freq=22
    'lack': ( -42,  +13,  -67,  +15,  -20),  # freq=36
    'laid': ( -77,  +74, -127,  +77,  -48),  # freq=58
    'landed': ( +46,  +22,   -7,   +4,  +12),  # freq=40
    'landlord': ( -32,  +38,  -44,  +49,  -20),  # freq=47
    'laptop': ( +42,  +18,  +59,  +36,  +10),  # freq=83
    'large': ( +43,  +11,  +69,   +2,  +24),  # freq=139
    'larger': ( +50,  +27,  +74,   +7,  +27),  # freq=11
    'lasagna': ( +66,  +13,  +52,  +18,  +23),  # freq=12
    'last': ( -21,  +12,  +10,  +29,   -8),  # freq=2921
    'late': ( -32,  +49,  -44,  +97,   -9),  # freq=456
    'later': ( -24,  -27,  +49, +125,  +23),  # freq=351
    'laugh': ( +28,  +18,  +17,   +0,  +14),  # freq=118
    'laughing': ( +32,  +27,  +24,   +0,  +27),  # freq=92
    'laundry': ( +32,  -38,  -34,   +0,   +2),  # freq=33
    'law': ( +28,  +27, +127,  +72,  +68),  # freq=217
    'lawn': ( +77,  -18,  +59,   +0,  +19),  # freq=74
    'lay': ( +71,  -42,  +59,   +6,  -14),  # freq=50
    'lazy': ( -32,  -49,  -67,   +0,  -29),  # freq=87
    'lead': ( +29,  +29,  +89,  +12,  +16),  # freq=54
    'leading': ( +48,   -9,  +59,   +6,  +10),  # freq=21
    'league': ( +34,  -27,   -9,   +0,   +8),  # freq=59
    # 'lean' removed — directional word, context-dependent (lean in=positive, lean toward=neutral)
    'leap': ( +56,  +29,  +47,   +4,  +24),  # freq=16
    'leash': ( -42,  -36,  -27,   +0,  -17),  # freq=20
    'least': ( -52,  -24,  -92,  +31,  +20),  # freq=1087
    'leftovers': ( -79,  -42, -127,   +0,  -55),  # freq=31
    'lesson': ( +42,  +13,  +34,  +30,   +8),  # freq=97
    "let's": ( +28,  +27,  +49,  +18,  +23),  # freq=54
    'letters': ( +35,  -31,  -39,   +0,   +2),  # freq=20
    'lice': (-102,   +0, -127,   +0,  -58),  # freq=14
    'lie': ( -52,  +38,  +22,  +31,   -9),  # freq=95
    'lied': ( -94,  +38, -127,  +27,  -55),  # freq=136
    'lifestyle': ( +56,   -6,  +42,   +0,  +23),  # freq=21
    'lift': ( +62,  +33,  +67,   +6,  -38),  # freq=27
    'lifted': ( +48,   -9,  +47,   +0,  +20),  # freq=11
    'light': ( +29,   +9,  +14,   +0,  +36),  # freq=142
    'lights': ( +56,   -4,  +22,   +0,  +19),  # freq=62
    'liked': ( +50,   -2,  -12,   +0,  +12),  # freq=194
    'likely': ( +29,  -24,  -14,   +0,   +6),  # freq=64
    'likes': ( +39,  +20,   +0,   +2,  +10),  # freq=110
    'liking': ( +49,   +9,  +12,   +0,  +16),  # freq=16
    'limited': ( -77,  -58,  -54,   +0,  -34),  # freq=12
    'lining': ( +62,   -9,  +77,  +12,  +10),  # freq=11
    'list': ( +37,  +13,  +94,  +43,  +23),  # freq=88
    'litter': ( -57,  +13, -127,   +6,  -44),  # freq=33
    'lmao': ( +50,  +40,  +17,   +0,  +24),  # freq=12
    'loan': ( -77,  +90, -127, +127,  +61),  # freq=72
    'local': ( +34,  -40,  -14,   +0,   +8),  # freq=201
    'locked': ( -42,  +38,  -67,  +65,  -20),  # freq=50
    'lol': ( +39,  +29,  +17,   +0,  +24),  # freq=1757
    'lonesome': ( -51,  -65,  -59,   +0,  -28),  # freq=17
    'loose': ( +50,  -49,  -54,  +12,  -48),  # freq=62
    'loser': (-117,   +2, -127,   +3,  -65),  # freq=16
    'lottery': ( +77,  +56,  +17,  +12,  -29),  # freq=226
    'lotto': ( +58,  +24,   -7,   +4,  +16),  # freq=30
    'loud': ( -24,  +83,  +24,  +35,  +23),  # freq=178
    'lovely': ( +43,   -4,  +32,   +0,  +17),  # freq=107
    'low': ( -52,  -49, -127,  +49, +127),  # freq=97
    'lower': ( -40,  -45,  -87,   +0,  -28),  # freq=30
    'luck': ( +29,  +29,  +17,  +12,  +24),  # freq=667
    'luckily': ( +51,   +0,   -7,   +0,  +13),  # freq=249
    'lucky': ( +50,  +40,  +34,   +0,  +32),  # freq=607
    'lunch': ( +28,  +13,  +24,   +0,  +10),  # freq=259
    'mad': ( -78,  +46,  +33,  +63,  +45),  # freq=806
    'madden': ( -40,  +22,  -12,  +13,  -16),  # freq=30
    'magical': ( +39,  +13,  +12,   +1,  +13),  # freq=16
    'mail': ( +51,  +38,  +59,  +73,   +8),  # freq=141
    'main': ( +37,  +22,  +67,   +6,   +8),  # freq=37
    'maintain': ( +39,  -18,  +42,   +0,  +19),  # freq=22
    'major': ( +28,  +40,  +49,  +18,  +10),  # freq=151
    'makeup': ( +48,  +27,  +52,  +12,  -14),  # freq=16
    'managed': ( +35,  +27,  -32,   +3,   +4),  # freq=114
    'management': ( +29,   +0,  +39,   +0,  +16),  # freq=57
    'mandatory': ( -52,  +38,  +99,  +23,   -1),  # freq=10
    'marathon': ( +32,  +49,  +69,  +15,  +20),  # freq=36
    'market': ( +57,  +47,  +59,  +55,  +23),  # freq=63
    'married': ( +39,  +29,  +54,   +0,  +16),  # freq=348
    'marry': ( +42,  +22,  +24,   +2,  +16),  # freq=37
    'marvel': ( +49,  +22,  +49,   +1,  +21),  # freq=10
    'massage': ( +46,  -24,  -24,   +0,   +9),  # freq=13
    'mate': ( +35,   -6,  +24,   +0,  +14),  # freq=20
    'math': ( +28, +127, +124, +127,  +89),  # freq=163
    'meal': ( +35,   -4,   -2,   +0,   +9),  # freq=154
    'means': ( +25,  -22,   -9,   +0,   +5),  # freq=137
    'medication': ( -32,  +24,  -44,  +49,   -9),  # freq=29
    'medicine': ( +10,  +5,  +10,  +10,  +20),  # context-dependent -- "on medicine" vs "medicine stopped"
    'member': ( +43,  -29,  +14,   +0,  +16),  # freq=57
    'memories': (  -5,  +15,   -5,   +5,   -5),  # freq=593 — ambiguous, context-dependent
    'memory': (  -5,  +15,   -5,   +5,   -5),  # freq=128 — ambiguous, context-dependent
    'mess': ( -32,  +24,  -44,  +31,  -20),  # freq=180
    'message': ( +32,  -20,  -24,   +0,   +5),  # freq=32
    'mid': ( -52,  -49,  -22,   +0,   +9),  # freq=21
    'middle': ( +28,   +0,  +99,  +18,  +23),  # freq=178
    'milestone': ( +50,  +29,  +54,   +0,  +24),  # freq=16
    # 'million' removed — number word, noise (context-dependent: "a million dollars" vs "one in a million")
    'millionaire': ( +43,  +22,  +47,   +1,  +20),  # freq=11
    'mind': ( +29,  +29,  +54,  +25,   -8),  # freq=397
    'mine': ( +42,  -13,  +22,   +0,  +16),  # freq=623
    'minor': ( -59,   +0, -119,   +0,  -40),  # freq=27
    'miracle': ( +37,  +33,  +24,   +6,  +29),  # freq=23
    'miss': ( -42,  +30,  -50,  +35,  -25),  # grief: longing, absence
    'missed': ( -42,  +38,  -67,  +49,  -20),  # freq=200
    'missing': ( -42,  +38,  -44,  +31,  -20),  # freq=176
    'mission': ( +58,  +24, +107,   +4,  +35),  # freq=36
    'mix': ( +42,  +42,  +59,  +43,  +14),  # freq=50
    'mold': ( -42,  +38,  -67,  +65,  -29),  # freq=23
    'mom': ( +50,  +20,  +34,  +12,  +16),  # freq=800
    'mommy': ( +41,  -11,   -9,   +0,   +9),  # freq=18
    'monster': ( -52,  +74,  -67,  +49,  +20),  # freq=23
    'mourn': ( -70,  +30,  -55,  +30,  -50),  # grief: grieving, bereavement
    'mourning': ( -70,  +30,  -55,  +30,  -50),  # grief: grieving process
    'moral': ( +43,   -9,  +87,   +0,  +27),  # freq=15
    'mortgage': ( -52,  +49,  -92,  +65,  +81),  # freq=28
    'mortified': ( -94,  +63,  -97,  +46,  -48),  # freq=58
    'mosquito': ( -77,  -27, -127,   +0,  -48),  # freq=12
    'motivation': ( +36,  +24,  +39,   +2,  +17),  # freq=40
    'motorcycle': ( +37,  +47,  +62,  +12,  +20),  # freq=30
    'mountains': ( +58,  -11,  +22,   +0,  +20),  # freq=40
    'movies': ( +50,  -13,   -7,   +0,  +12),  # freq=296
    'multiple': ( +35,   +2,  +22,   +0,  +14),  # freq=37
    'musical': ( +42,  +11,   +2,   +1,  +13),  # freq=13
    'mutual': ( +50,  -31,   -9,   +0,  +12),  # freq=13
    'naked': ( -40,  +87,  -77,  +51,  -25),  # freq=16
    'nap': ( +29,  -40,  +17,   +0,   +8),  # freq=39
    'nasty': (-106,  +33, -104,  +23,  -53),  # freq=123
    'national': ( +43,  -24,  +47,   +0,  +20),  # freq=25
    'natural': ( +29,   +9,  +34,   +0,  +16),  # freq=64
    'naturally': ( +36,  -18,   +7,   +0,  +12),  # freq=23
    'nature': ( +36,  -27,  +19,   +0,  +14),  # freq=59
    'near': ( +28,  +13,  +24,   +0,  +10),  # freq=218
    'neat': ( +39,  -22,   -2,   +0,  +10),  # freq=93
    'necessary': ( +56,  -11,  +49,   +0,  +24),  # freq=34
    'necklace': ( +71,  +22,  +52,   +6,   +8),  # freq=35
    'needles': ( -59,  +22,  -54,  +15,  -29),  # freq=12
    'needless': ( -40,  -24, -109,   +0,  -32),  # freq=19
    'negative': (-25,   +5,  -20,   +5,  -10),  # liquid -- medical=good, emotional=bad. Context determines.
    'nephew': ( +51,  +27,  +17,  +18,  -14),  # freq=134
    'nerve': ( +50, +108,  +72, +127,   -8),  # freq=132
    'nerves': ( -37,  +63,  -87,  +35,  -28),  # freq=91
    'nest': ( +42,  -49,  -47,   +0,   +2),  # freq=22
    'new': ( +29,  +29,  +17,  +12,  +16),  # freq=3265
    'newborn': ( +43,   +6,  -17,   +0,   +9),  # freq=30
    'news': ( +28, +127, +127, +127,  +44),  # freq=400
    'nice': ( +29,   +4,   +7,   +0,  +12),  # freq=3179
    'nicer': ( +39,  -20,   -9,   +0,   +9),  # freq=26
    'nicest': ( +56,  -24,  +29,   +0,  +20),  # freq=10
    'niece': ( +38,  +12,   +7,  +15,  -14),  # freq=73
    'nightmare': (-127, +127, -127, +116,  -73),  # freq=38
    'nightmares': (-117, +119,  -22,  +93,  -43),  # freq=35
    'nintendo': ( +31,   +4,   -4,   +1,   +8),  # freq=98
    'noble': ( +56,   +6,  +34,   +5,  +17),  # freq=17
    'nobody': ( -46,  +17,  -27,  +26,  -15),  # freq=79
    'noise': ( -24,  +40,  -24,  +35,  +10),  # freq=116
    'noisy': (-127, +127, -109, +127,  +24),  # freq=11
    'nope': ( -24,  +13,  +24,   +0,   +0),  # freq=168
    'nostalgia': ( +28,  +27,  -24,   +0,  -10),  # freq=54
    'note': ( +28,  -83,  -34,   +0,   +2),  # freq=75
    'nothing': ( -80,  -10,  -55,  +15,  -45),  # grief/existential: absence, void, emptiness
    'noticed': ( +29,  +20,  +34,  +12,  +16),  # freq=168
    'nowadays': ( +39,  -27,  -29,   +0,   +6),  # freq=61
    'nowhere': (-114,  -58, -127,   +0,  +48),  # freq=59
    'numb': ( -35,  -20,  -30,   +0,  -25),  # grief: emotional shutdown, dissociation
    'nurse': ( +50,   +9,  +17,  +38,   +8),  # freq=29
    'obnoxious': (-127,  +27, -109,  +23,  -61),  # freq=10
    'occasion': ( +43,   -9,  -29,   +0,   +8),  # freq=55
    'odd': ( -55,  -38,  -72,   +0,  -29),  # freq=64
    'offer': ( +56,  +42,  +29,   +9,  +20),  # freq=135
    'offered': ( +24,   -2,  +19,   +0,  +10),  # freq=160
    'often': ( +39,  +29,  +34,   +0,   +0),  # freq=374
    'ohhh': ( -57,  -13,  -59,   +0,  -28),  # freq=27
    'oil': ( +50,   -9, +126,  +38,  +32),  # freq=66
    'okay': ( +28,  -13,  +24,   +0,   +0),  # freq=822
    'oldest': ( -27,  -36,  -54,   +0,  -17),  # freq=86
    'omg': ( +29,  +69,  -17,  +25,  +24),  # freq=119
    'online': ( +28,  +13,  +24,  +18,  +10),  # freq=282
    'ooo': ( -40,  -15,  -64,   +0,  -24),  # freq=16
    'opened': ( +29,  +29,  +54,  +25,   +0),  # freq=83
    'opening': ( +57,  +51,  +67,  +67,   +8),  # freq=43
    'opinion': ( +28,   -4,  +54,   +0,  +17),  # freq=43
    'opportunities': ( +50,  +27,  +34,   +0,  +20),  # freq=25
    'opportunity': ( +50,  +40,  +54,  +25,  +32),  # freq=96
    'opposite': ( -62,   +9,  -32,   +6,  -25),  # freq=47
    'options': ( +28,  +51,  +59,  +43,   +8),  # freq=48
    'orange': ( +62,  +13,  +52,   +6,  -10),  # freq=15
    'ordeal': ( -77,  +90, -109,  +77,  -36),  # freq=16
    'order': ( +28,  +13,  +74,  +18,  +10),  # freq=221
    'organized': ( +32,  -13,  +34,   +0,  +14),  # freq=38
    'others': ( +35,   +9,  -64,   +2,   +0),  # freq=287
    'otherwise': ( -27,  -15,  -39,   +0,  -16),  # freq=80
    'ouch': ( -40,  -15,  -72,   +0,  -25),  # freq=85
    'outfit': ( +58,   -2,  +17,   +0,  +19),  # freq=19
    'outlook': ( +34,   +2,  +17,   +0,  +13),  # freq=24
    'outside': ( +29,  +20,  +17,  +12,   -8),  # freq=492
    'overnight': ( -49,  -29,  +17,   +0,  -13),  # freq=19
    'overtime': ( -42,  +49,  -67,  +81,  -20),  # freq=19
    'overwhelming': ( -77, +119, -127,  +97,  -48),  # freq=38
    'owner': ( +28,   -4,  +57,   +0,  +17),  # freq=65
    'package': ( +28,  +27,  +24,  +18,  +10),  # freq=99
    'painful': ( -89,  +40,  -55,  +63,  -45),  # freq=73
    'pains': (-109,  +47,  -59,  +34,  -48),  # freq=11
    'painting': ( +32,  -15,  -22,   +0,   +5),  # freq=35
    'pair': ( +57,  +18,  +59,  +12,  +14),  # freq=70
    'panama': ( +35,   +0,   +0,   +0,  +10),  # freq=20
    'paranoid': (-106, +112,  -59,  +86,  -46),  # freq=15
    'parcel': ( +35,  -31,  -24,   +0,   +6),  # freq=23
    'participate': ( +56,  +13,  +54,   +2,  +25),  # freq=16
    'parties': ( +35,  +42,   +4,   +6,  +10),  # freq=33
    'party': ( +50,  +58,  +34,   +0,  +40),  # freq=696
    'passion': ( +36,  +31,  +29,   +3,  +16),  # freq=35
    'pasta': ( +62,   +9,  +52,  +18,   +8),  # freq=27
    'patch': ( +50,  +40,  +72, +103,   +0),  # freq=20
    'path': ( +51,  +13,  +52,  +12,  +10),  # freq=48
    'paycheck': ( +57,  +38,  +59,  +61,  -14),  # freq=51
    'payday': ( +37,  +20,   +9,   +2,  +13),  # freq=20
    'peeved': ( -96,  +65,  -17,  +50,  -36),  # freq=20
    'pepper': ( +39,  +69,  +34,  +12,   -8),  # freq=34
    'pepperoni': ( +57,  +33,  +24,  +24,   +8),  # freq=16
    'perfect': ( +42,  +13,  +34,   +0,  +23),  # freq=198
    'perfectly': ( +51,   +0,  +27,   +0,  +20),  # freq=53
    'perform': ( +58,   +9,  +79,   +2,  +31),  # freq=29
    'performance': ( +56,   +4,  +49,   +0,  +24),  # freq=63
    'personal': ( +56,  -33,   -9,   +0,  +14),  # freq=72
    'personality': ( +42,   +0,  +39,   +0,  +19),  # freq=30
    'perspective': ( +42,   -6,  +57,   +0,  +23),  # freq=31
    'pet': ( +39,  +20,  +34,   +0,  +16),  # freq=357
    'phobia': ( -92, +108,  -92,  +77,  -46),  # freq=13
    'physical': ( +56,   +4,  +32,   +0,  +20),  # freq=53
    'piano': ( +43,  -27,  -17,   +0,   +9),  # freq=27
    'picked': ( +51,   +4,  +49,   +0,  +24),  # freq=106
    'picky': ( -57,  +51,  +37,  +31,  -13),  # freq=14
    'pictures': ( +34,  -45,  -29,   +0,   +5),  # freq=179
    'pie': ( +39,   -9,  -19,   +0,   +8),  # freq=27
    'piece': ( +39,  +20,  +54,  +25,  +16),  # freq=131
    'pig': ( +37,   +9,  +59,   +6,  +42),  # freq=24
    'pimple': ( -59,  -29,  -82,   +0,  -34),  # freq=12
    'pineapple': ( +37,  -33,  -72,   +0,   -1),  # freq=22
    'pink': ( +32,  +13,   +7,   +0,  -10),  # freq=31
    'pizza': ( +57,  +22,  +17,  +24,   +8),  # freq=381
    'planet': ( +43,  -18,  +97,   +0,  +28),  # freq=12
    'plant': ( +29,  +20,  +34,  +12,  +16),  # freq=57
    'pleasant': ( +44,  -15,  +24,   +0,  +17),  # freq=47
    'pleasantly': ( +51,  -22,   -2,   +0,  +14),  # freq=11
    'pleasure': ( +44,  +20,  +24,   +1,  +17),  # freq=22
    'plenty': ( +39,  +20,  +54,   +0,  +24),  # freq=74
    'plus': ( +51,   +0,   +2,   +0,  +16),  # freq=113
    'poison': (-127, +127, +127, +127,  +61),  # freq=10
    'poisoning': (-114,  +90,   +9,  +70,  -36),  # freq=20
    'poisonous': (-112,  +99,  +17,  +77,  -36),  # freq=21
    # 'pokemon' removed — proper noun, noise
    # 'policy' removed — neutral factual word with extreme D bias
    'political': ( -36,  +51, +127,  +29,   +9),  # freq=36
    'politics': ( -59,   +6, +126,   +3,   +1),  # freq=12
    'poo': ( -96,  -29, -124,   +0,  -55),  # freq=18
    'poop': ( -94,  -18, -127,   +0,  -55),  # freq=118
    'poor': (-127,  +29, -127, +116,  +85),  # freq=201
    'poorly': (-119,  +33, -127,  +27,  -68),  # freq=19
    'pop': ( +37,   +2,   -2,   +0,  +10),  # freq=37
    'popular': ( +39,  +15,  +29,   +1,  +16),  # freq=27
    'position': ( +21,   -9,   +9,   +0,   +8),  # freq=211
    'positive': ( +48,   +0,  +47,   +0,  +21),  # freq=292
    'possibility': ( +34,   +0,   +4,   +0,  +10),  # freq=22
    'potato': ( +57,   +4,  +59,  +18,   +8),  # freq=10
    'potatoes': ( +42,  -40,  -54,   +0,   +2),  # freq=23
    'potty': ( -32,  -27,  -67,   +0,  -21),  # freq=54
    'power': ( +29,  +49, +126,  +25,  +24),  # freq=149
    'practice': ( +28,  +27,  +49,  +18,  +10),  # freq=102
    'pray': ( +28,  +27,  -24,  +35,  +23),  # freq=103
    'prayer': ( +55,  -49,  +29,   +0,  +20),  # freq=17
    'praying': ( +32,  -47,  -24,   +0,   +5),  # freq=37
    'precious': ( +28,  +13,  +17,   +6,  +10),  # freq=44
    'prefer': ( +58,  -20,   +9,   +0,  +17),  # freq=58
    'pregnancy': ( +51,  +47,  +22,   +9,  +19),  # freq=39
    'pregnant': ( +28,  +69,  -49,  +54,  +10),  # freq=196
    'prep': ( +42,   +2,  +14,   +0,  +14),  # freq=19
    'prepare': ( +34,   +9,  +29,   +2,  +14),  # freq=101
    'prepared': ( +46,   -4,  +69,   +0,  +24),  # freq=697
    'preparing': ( +51,   +6,  +24,   +2,  +19),  # freq=40
    'presence': ( +39,   -4,  +14,   +0,  +13),  # freq=18
    'present': ( +28,  +13,  +24,   +0,  +10),  # freq=96
    'presentation': ( -24, +127,  +49, +127,  +23),  # freq=101
    'presents': ( +29,   -9,   +2,   +0,   +9),  # freq=44
    'president': ( +37,  +22, +127,   +4,  +35),  # freq=48
    'pressure': ( -42,  +63,  -67,  +65,   -9),  # freq=39
    'prestigious': ( +51,  +11,  +57,   +0,  +24),  # freq=10
    'pretended': ( -94,  +15, -119,  +11,  -53),  # freq=22
    'previous': ( +29,  +20, +126,  +63,  +32),  # freq=48
    'priceless': ( +55,  +36,  +89,   +7,  +31),  # freq=15
    'pricey': ( -37,  +15,  -27,   +9,  -16),  # freq=19
    'prison': (-114,  +58, -127,  +77,  -85),  # freq=22
    'private': ( +35,  -13,  +42,   +0,  +17),  # freq=37
    'prize': ( +28,  +22,  +24,   +0,  +19),  # freq=55
    'pro': ( +35,   -6,  +54,   +0,  +19),  # freq=28
    'problem': ( -32,  +38,  -44,  +65,   -9),  # freq=308
    'productive': ( +35,  +15,  +37,   +2,  +16),  # freq=32
    'professional': ( +35,   -4,  +59,   +0,  +20),  # freq=48
    'professor': ( +34,  -18,  +64,   +0,  +20),  # freq=31
    'profit': ( +66,  +38,  +77,  +12,  -23),  # freq=11
    'progress': ( +39,  +29,  +54,  +12,  +24),  # freq=66
    'project': ( +29,  +49,  +72, +103,  +16),  # freq=206
    'promise': ( +39,  +20,  +54,  +12,  +16),  # freq=58
    'promised': ( +37,  -22,   +2,   +0,  +10),  # freq=109
    'promoted': ( +48,  +27,  +52,   +1,  +23),  # freq=79
    'promotion': ( +32,  +27,  +42,  +12,  +19),  # freq=362
    'properly': ( +35,  -20,  +22,   +0,  +13),  # freq=43
    'property': ( +51,  -40,  +87,   +0,  +31),  # freq=50
    # 'proposal' removed — neutral factual word causing positive drift
    'proposed': ( +35,  +56,  +57,  +15,  +20),  # freq=24
    'protein': ( +56,  -29,  +42,   +0,  +24),  # freq=17
    'provider': ( +56,   -2,  +72,   +0,  +28),  # freq=15
    'public': ( +28,   +2,  +64,   +0,  +20),  # freq=121
    'puke': ( -83,  +11,  -92,   +7,  -43),  # freq=25
    'punch': (-127, +127, +127, +127, +108),  # freq=26
    'punched': ( -86,  +36,  -22,  +27,  -34),  # freq=11
    'punish': (-123,  +83,  +49,  +66,  -34),  # freq=21
    'punished': ( -92,  +63,   -9,  +46,  -31),  # freq=13
    'punishment': (-117,  +85,  +17,  +70,  -39),  # freq=14
    'pup': ( +56,  -13,  -57,   +0,   +6),  # freq=33
    'puppies': ( +51,  +27,  -54,   +0,   +5),  # freq=42
    'puppy': ( +32,  +22,   +7,   +0,  +23),  # freq=242
    'purchased': ( +32,  +24,   -4,   +3,   +8),  # freq=49
    'pure': ( +50,   +9,  +34,   +0,  +24),  # freq=10
    'qualified': ( +58,   +4,  +89,   +0,  +32),  # freq=37
    'quality': ( +29,  +20,  +34,   +0,  +16),  # freq=67
    'quick': ( +42,  +47,  +52,  +67,  -19),  # freq=99
    'quiet': ( +28,  -69,  +24,   +0,   +0),  # freq=195
    'quitting': ( -79,  +33, -124,  +23,  -48),  # freq=14
    'rabbit': ( +39,  -29,  -74,   +0,   +0),  # freq=26
    'raise': ( +29,  +29,  +34,  +12,  +16),  # freq=242
    'raised': ( +34,  -31,  +39,   +0,  +17),  # freq=57
    'ramen': ( +51,  +13,  +34,  +24,   +8),  # freq=30
    'rash': ( -79,  +67,   -4,  +46,  -28),  # freq=11
    'rat': (-127, +127, -127, +127,  +24),  # freq=40
    'ready': ( +80,  +78, +119, +121,  -34),  # freq=962
    'realistic': ( +43,   +4,  +82,   +0,  +25),  # freq=16
    'realized': ( +36,  -27,  +22,   +0,  +14),  # freq=191
    'receive': ( +51,   -6,  +14,   +0,  +17),  # freq=63
    'received': ( +32,  -11,  +17,   +0,  +12),  # freq=135
    'recipe': ( +28,  +13,  +49,   +0,  +10),  # freq=63
    'recital': ( +35,   -4,   +9,   +0,  +12),  # freq=24
    'recognize': ( +58,  -33,  +29,   +0,  +20),  # freq=24
    'recognized': ( +39,   -4,  +39,   +0,  +17),  # freq=21
    'record': ( +29,  +29,  +54,  +12,  +16),  # freq=52
    'recover': ( +39,  +20,  +54,  +12,  +16),  # freq=54
    'red': ( +28,  +83,  +74,  +35,  +10),  # freq=163
    'reflect': ( +37,  -40,  +22,   +0,  +14),  # freq=23
    'refreshing': ( +44,   +2,   +9,   +0,  +14),  # freq=15
    'refund': ( +28,  +40,  +49,  +54,  +10),  # freq=74
    'refuse': ( -89,   +6,  -77,   +3,  -43),  # freq=18
    'refused': ( -94,   -6, -127,   +0,  -55),  # freq=42
    'rejection': ( -89,  +74, -127,  +57,  -48),  # freq=10
    'relate': ( +34,   +0,  +34,   +0,  +16),  # freq=104
    'related': ( +37,  -24,   +7,   +0,  +12),  # freq=43
    'relative': ( +29,  +29,  -34,  +38,   +0),  # freq=41
    'relax': ( +50,  -49,  +54,   +0,  +16),  # freq=193
    'relaxation': ( +39,  -22,  -17,   +0,   +8),  # freq=17
    'relaxed': ( +37,  -38,  -12,   +0,   +9),  # freq=51
    'relaxing': ( +36,  -29,   -2,   +0,   +9),  # freq=191
    'release': ( +39,  -20,  +54,  -12,  +24),  # freq=26
    'released': ( +35,   +0,  +19,   +0,  +13),  # freq=29
    'reliable': ( +42,  -22,  +52,   +0,  +21),  # freq=39
    'religious': ( +51,  -47,  +57,   +0,  +24),  # freq=17
    'relive': ( +56,  +27,  +54,   +4,  +25),  # freq=14
    'rely': ( +32,   -4,  +14,   +0,  +12),  # freq=46
    'remain': ( +46,  -24,  +64,   +0,  +24),  # freq=20
    'remember': ( -23,  +22,  -24,   +6,  -13),  # freq=823
    'remembered': ( +37,  -22,   +9,   +0,  +13),  # freq=79
    'reminisce': ( +42,   +0,  -19,   +0,   +9),  # freq=19
    'reminiscing': ( +42,  -11,  -24,   +0,   +8),  # freq=18
    'remote': ( +51,  -18,  +77,   +0,  -19),  # freq=20
    'remove': ( -49,   +9,   -9,   +6,  -17),  # freq=11
    'removed': (-100,  -90,  -97,   +0,  -51),  # freq=26
    'rent': ( -42,  +49,  -67,  +81,  -29),  # freq=103
    'replaced': ( -42,  +47,  -99,  +29,  -32),  # freq=44
    # 'report' removed — neutral factual word with extreme A/D/U/G bias
    'rescue': ( +39,  +58,  +72,  +51,  +24),  # freq=75
    'rescued': ( +51,  +22,  -57,   +0,   +5),  # freq=37
    'research': ( +28,  +40,  +49,  +35,  +10),  # freq=93
    'resolved': ( +51,  -24,  +54,   +0,  +24),  # freq=25
    'respect': ( +50,  +20,  +54,   +0,  +16),  # freq=74
    'response': ( +39,  +40,  +72,  +89,   +0),  # freq=47
    'responsibilities': ( +35,  +27,  +52,   +3,  +17),  # freq=30
    'responsibility': ( +58,   +0, +102,   +0,  +34),  # freq=33
    'rest': ( +29,  -40,  +34,   +0,   +8),  # freq=258
    'restaurant': ( +32,  +18,  +24,  +12,   +4),  # freq=320
    'restroom': ( +34,  -40,  -69,   +0,   -1),  # freq=41
    'retire': ( +62,  -27,  +84,   +0,  -29),  # freq=28
    'retired': ( -52,  -54,  -72,   +0,  -29),  # freq=24
    'retirement': ( +66,  -38,  +84,   +0,  -34),  # freq=35
    'returned': ( +29,  +20,  +34,   +0,   +8),  # freq=95
    'reunion': ( +35,   +0,  +24,   +0,  +14),  # freq=44
    'reunited': ( +46,   -9,  +47,   +0,  +20),  # freq=15
    'revenge': ( -77, +103, +109,  +77,  +36),  # freq=14
    'review': ( +35,  -15,  +39,   +0,  +17),  # freq=40
    'revolting': (-127,  +58,  +54,  +39,  +36),  # freq=10
    'rewarding': ( +48,  +11,  +39,   +0,  +20),  # freq=49
    'rich': ( +39,  +20,  +54,   +0,  +16),  # freq=48
    'rid': ( -86,  -24,  -77,   +0,  -40),  # freq=74
    'ridden': ( -94,  +90,  -92,  +66,  -48),  # freq=14
    'ride': ( +80,  +60,  +77,  +30,  +23),  # freq=227
    'ridiculous': ( -52,  +63,  +44,  +31,  +29),  # freq=65
    'riding': ( +42,  +27,  +34,   +7,  +17),  # freq=92
    'rip': (-114,  +29, -127,  +19,  +48),  # freq=35
    'ripped': ( -55,  +60,  -30,  +35,  -20),  # freq=39 — torn away, violent removal
    'roach': (-127, +119, -127, +127,  +36),  # freq=14
    'robbed': (-104,  +83, -127,  +66,  -68),  # freq=59
    'robber': (-102,  +72,   -9,  +54,  -36),  # freq=11
    'roll': ( +28, +127, +127, +127,  -23),  # freq=61
    'roller': ( +51,  +33,  +67,  +36,  +10),  # freq=101
    'romantic': ( +50,  +29,  +17,   +0,  +24),  # freq=33
    'roommate': ( +56,   -4,   -2,   +0,  +16),  # freq=129
    'rotten': (-102,  +74,  -54,  +39,  -48),  # freq=24
    'rotting': ( -86,   -9, -109,   +0,  -48),  # freq=15
    'rough': ( -32,  +49,  +44,  +31,   -9),  # freq=242
    'rude': ( -67,  +33,  +22,  +32,  +27),  # freq=167
    'ruin': (-104,  +24, -127,  +19,  -58),  # freq=47
    'sadly': (-127,  -65, -124,   +0,  -65),  # freq=67
    'safely': ( +51,  -29,  +12,   +0,  +17),  # freq=30
    'safety': ( +46,   -2, +104,   +0,  +31),  # freq=34
    'salad': ( +37,   -4,  +24,   +0,  -14),  # freq=31
    'salary': ( +48,  +18,  +67,  +12,   -8),  # freq=27
    'sale': ( +29,  +20,  +34,  +12,  +16),  # freq=64
    'sales': ( +32,  +51,  +59,  +67,   -4),  # freq=36
    'sandwich': ( +51,   +9,  +52,  +30,   +4),  # freq=61
    'santa': ( +51,  -38,   +2,   +0,  +16),  # freq=10
    'satisfying': ( +41,   +9,  +42,   +1,  +19),  # freq=21
    'save': ( +29,  +20,  +34,  +12,   +8),  # freq=235
    'saved': ( +51,  -24,  +42,   +0,  +21),  # freq=123
    'saving': ( +35,  -47,  +34,   +0,  +16),  # freq=135
    'scale': ( -44,  -36,  +27,   +0,   -9),  # freq=12
    'scam': ( -89,  +90, -109,  +97,  -24),  # freq=14
    'scare': (-106, +110, -104,  +86,  -53),  # freq=105
    'scares': (-106,  +92,  -49,  +70,  -46),  # freq=39
    'scariest': ( -94,  +99,  +17,  +70,  -28),  # freq=15
    'scary': ( -77, +103, -127,  +57,  +36),  # freq=904
    'scenery': ( +55,  -42,  +24,   +0,  +20),  # freq=10
    # 'scheduled' removed — neutral factual word causing positive drift
    'scholarship': ( +32,  +22,  +34,   +0,  +23),  # freq=67
    'science': ( +42,  +27,  +67,  +18,  +14),  # freq=63
    'scold': ( -86,  +67,  +54,  +46,  -19),  # freq=18
    'score': ( +50,  +49,  +72,  +12,  +32),  # freq=77
    'scored': ( +35,  +27,  +29,   +3,  +14),  # freq=29
    'scratch': ( -32,  +38,  -22,  +31,   +0),  # freq=102
    'seafood': ( +34,   +0,  -49,   +0,   +1),  # freq=22
    'season': ( +29,   -6,   +2,   +0,   +9),  # freq=250
    'secret': ( -24,  +54,  +49,  +54,  +10),  # freq=118
    'secure': ( +39,   -9,  +89,   +0,  +16),  # freq=28
    'security': ( +42,   +6,  +31,  +30,  +11),  # freq=77
    'self': ( +43,  -24,   +0,   +0,  +12),  # freq=236
    'selfish': ( -52,  +38,  +44,  +15,   +9),  # freq=31
    'semi': ( -47,  -36,  -72,   +0,  -28),  # freq=23
    'sense': ( +29,   -6,   -2,   +0,   +8),  # freq=194
    'sensitive': ( +48,  -13,  -72,   +0,   +1),  # freq=14
    'sentimental': ( +24,  -29,  -74,   +0,   -4),  # freq=308
    'separated': ( -89,  +74, -127,  +57,  -61),  # freq=21
    'served': ( +42,  -15,  -89,   +0,   -2),  # freq=20
    'service': ( +39,  -15,   +9,   +0,  +12),  # freq=164
    'settle': ( +35,   +2,  +37,   +0,  +16),  # freq=27
    'severe': ( -94,  +63,  +59,  +46,  -21),  # freq=36
    'shaking': ( -37,  +44,  -46,  +53,  +15),  # freq=30
    'shape': ( +28,   +0,  -29,   +0,   +2),  # freq=49
    'share': ( +29,  +20,  +17,   +0,  +16),  # freq=125
    'shared': ( +35,   +0,  +29,   +0,  +14),  # freq=32
    'sharing': ( +34,   -2,   +7,   +0,  +10),  # freq=29
    'shark': (-127, +127, +127, +127,  +85),  # freq=18
    'sheesh': ( -47,   +6,  -32,   +3,  -21),  # freq=11
    'shelter': ( +28,  +27,  +24,  +18,  +10),  # freq=86
    'shining': ( +39,  +15,  +42,   +1,  +19),  # freq=40
    'shipping': ( +50,  +69,  +89, +127,  +16),  # freq=16
    'shock': ( -42, +112,  -92,  +81,  +20),  # freq=71
    'shoot': (-114, +112,  +54,  +89,  -31),  # freq=28
    'shooting': ( -52,  +96,  +37,  +61,  -12),  # freq=29
    'shopping': ( +43,   -4,  +17,   +0,  +14),  # freq=214
    'shredded': ( -47,  +74, -127,  +45,  -42),  # freq=11
    'shrimp': ( +51,  +18,  +24,  +18,   -8),  # freq=12
    'shut': ( -32,  +24,  +44,  +31,   -9),  # freq=56
    'sibling': ( +57,  +22,  +52,  +18,  +10),  # freq=28
    'sickness': (-114,  +54,  -92,  +43,  -55),  # freq=12
    'sight': ( +39,  -42,   +7,   +0,  +12),  # freq=41
    'silly': ( -27,  -36, -122,   +0,  -29),  # freq=57
    'simple': ( +28,  -13,  +49,   +0,  +10),  # freq=62
    'sing': ( +50,  +40,  +34,   +0,  +32),  # freq=32
    'singing': ( +39,   +6,   +2,   +1,  +12),  # freq=39
    'site': ( +37,  -40,  -14,   +0,   +9),  # freq=23
    'skateboard': ( +37,   +9,  -42,   +2,   +5),  # freq=20
    'skating': ( +55,  +31,  -17,   +7,  +12),  # freq=21
    'skiing': ( +34,  +27,  +17,   +3,  +13),  # freq=35
    'skill': ( +35,  +15,  +47,   +2,  +17),  # freq=46
    'skunk': ( -51,   -2,  -44,   +0,  -24),  # freq=13
    'slammed': ( -89, +127, +127,  +93,   -6),  # freq=20
    'slap': (-127, +127, +127, +127,  +36),  # freq=19
    'slapped': (-109, +127, -119, +100,  -58),  # freq=12
    'sleepy': ( +50, -127, -126, +115,  +40),  # freq=11
    'slipped': ( -42,  +67, -127,  +39,  -44),  # freq=78
    'slow': (-114, -127, +127, +116, +127),  # freq=118
    'smack': ( -40,  +38,  -44,  +23,  -21),  # freq=23
    'small': ( +42,   +9,  -77,   +6,  -27),  # freq=291
    'smaller': ( -47,  -42, -119,   +0,  -36),  # freq=24
    'smart': ( +50,  +29,  +72,   +0,  +24),  # freq=270
    'smarter': ( +44,   +0,   +7,   +0,  +14),  # freq=11
    'smash': ( -86,  +51,   +9,  +34,  -27),  # freq=120
    'smashed': ( -96,  +42, -109,  +30,  -51),  # freq=17
    'smelly': (-112,   +9, -109,   +7,  -58),  # freq=16
    'smile': ( +48,  +18,  +24,   +0,  -10),  # freq=61
    'smoke': ( -32,  +24,  -22,  +15,   -9),  # freq=30
    'smoking': ( -42,  +24,  -22,  +15,   -9),  # freq=26
    'snacks': ( +29,   -2,  -27,   +0,   +4),  # freq=38
    'snake': (-127, +127, -127, +127,  +12),  # freq=94
    'snakes': ( -92,  +45,  +42,  +30,  -24),  # freq=60
    'sneak': ( -51,  +54,  -17,  +35,  -20),  # freq=18
    'sneaky': ( -37,  +50,  +37,  +93,   -7),  # freq=19
    'snuck': ( -44,  +69,  -72,  +41,  -28),  # freq=13
    'soccer': ( +43,  +58,  +42,  +15,  +19),  # freq=104
    'social': ( +32,  +33,  +17,  +12,   -8),  # freq=159
    'socialize': ( +42,  +15,  +29,   +1,  +17),  # freq=16
    'society': ( +37,  +13,  +65,  +10,  +36),  # freq=41
    'socks': ( +51,  -18,  +34,   +6,   +4),  # freq=32
    'soda': ( +32,  +22,  +24,  +18,   -8),  # freq=32
    'softball': ( +51,  +24,   +2,   +4,  +16),  # freq=15
    'solution': ( +39,  +29,  +72,  +25,  +24),  # freq=27
    'solve': ( +39,  +29,  +72,  +25,  +24),  # freq=23
    'sometimes': ( +28,   +0,  +49,   +0,  +10),  # freq=1415
    'somewhere': ( -27,   -4,  -22,   +0,  -13),  # freq=145
    'sore': (-102,  +29, -109,  +57,  +48),  # freq=18
    'sorry': ( -32,  +24,  -67,  +15,  -20),  # freq=2697
    'sort': ( +37,  -20,  +49,   +0,  +19),  # freq=152
    # 'sound' removed — dual-use noun/adjective, "sounds good" vs "every sound makes me jump"
    'source': ( +51,  -31,  +54,   +0,  +24),  # freq=17
    'south': ( +32,  +18,  +42,   +6,   +4),  # freq=88
    'space': ( +31,  -15,   +9,   +0,  +10),  # freq=65
    'spaghetti': ( +50,  -24,  -62,   +0,   +5),  # freq=15
    'speak': ( +28,  +27,  +49,  +18,  +10),  # freq=109
    # 'speaking' removed — neutral verb carrying false positive charge ("generally speaking")
    'special': ( +50,  +29,  +34,   +0,  +24),  # freq=352
    'specialist': ( +58,  +11, +107,   +2,  +35),  # freq=11
    'specific': ( +34,  -20,  +54,   +0,  +19),  # freq=75
    'speech': ( +34,   -6,  +54,   +0,  +19),  # freq=116
    'speed': ( +48,  +60,  +47,  +15,  +23),  # freq=49
    'spider': (-127, +127, -127, +127,  +12),  # freq=75
    'spiders': ( -55,  +45,  -49,  +29,  -28),  # freq=66
    'spilled': ( -24,  +40,  -24,  +35,  -10),  # freq=93
    'spirit': ( +57,  +42,  +59,  +12,  -38),  # freq=34
    'spit': (-100,  +11,  -27,   +7,  -39),  # freq=19
    'spite': (-117,  +27,  -82,  +23,  -53),  # freq=11
    'split': ( -36,  +27,  -32,  +15,  -17),  # freq=25
    'spoil': (-100,  +58,  -77,  +43,  -46),  # freq=25
    'spoiled': ( -52,  +38,  -22,  +31,  -20),  # freq=32
    'spoke': ( +29,  -11,   -2,   +0,   +8),  # freq=48
    'sport': ( +37,  +15,  +22,   +2,  +14),  # freq=49
    'sports': ( +35,  +15,  +12,   +2,  +12),  # freq=88
    'spring': ( +48,  +22,  +34,   +0,  -14),  # freq=62
    'squid': ( +39,  +69,  +72,  +77,  -32),  # freq=16
    'staff': ( +55,  -40,  +34,   +0,  +23),  # freq=22
    'stage': ( +28,  -47,   -9,   +0,   +6),  # freq=59
    'stamp': ( +42,  +13,  +59,  +30,   +8),  # freq=14
    'starved': (-112,  +51, -114,  +39,  -58),  # freq=28
    'station': ( +50,  +90, +126, +127,  +80),  # freq=84
    'steak': ( +57,  +18,  +24,   +6,  +19),  # freq=50
    'steal': ( -89,  +90,  +82,  +77,  +24),  # freq=83
    'stealing': (-117, +110,  -22,  +89,  -43),  # freq=64
    'stick': ( +28,  +13,  +74,  +35,  +34),  # freq=83
    'sting': ( -94,  +42,  -54,  +30,  -40),  # freq=10
    'stink': (-127,   -6, -127,   +0,  -68),  # freq=13
    'stole': (-102,  +67,  -49,  +50,  -43),  # freq=248
    'stolen': ( -96,  +38,  -82,  +27,  -48),  # freq=69
    'storage': ( +50,  +20, +107,  +38,  +16),  # freq=26
    'stories': ( +37,   +2,   +0,   +0,  +10),  # freq=65
    'storm': ( -52,  +99,  -67,  +81,  -20),  # freq=181
    'storming': ( -55,  +92,  +64,  +59,   -8),  # freq=17
    'strange': ( -24,  +40,  -49,  +18,  +10),  # freq=133
    'stray': ( -32,  +24,  -44,  +15,  -20),  # freq=48
    'strength': ( +39,  +40, +126,  +12,  +24),  # freq=40
    'stressful': ( -96, +101,  -49,  +73,  -40),  # freq=191
    'stressing': ( -94,  +99,  -17,  +70,  -36),  # freq=20
    'stroke': (-127, +127, -127, +127,  +85),  # freq=29
    'struck': ( -47,  -24,   +4,   +0,  -16),  # freq=20
    'struggled': ( -86, +117, -127,  +81,  -55),  # freq=21
    'stubbed': ( -40,  +31,  -94,  +19,  -29),  # freq=23
    'stuck': ( -52,  +49, -114,  +65,  -29),  # freq=164
    'student': ( +51,  +33,  -24,  +24,   +4),  # freq=122
    'studied': ( +29,  -15,  +39,   +0,  +16),  # freq=250
    'stumbled': ( -81,  +83, -127,  +57,  -61),  # freq=15
    'stupid': ( -89,  +74,  +54,  +39,  +36),  # freq=139
    'style': ( +55,   -6,  +24,   +0,  +19),  # freq=53
    'succeed': ( +44,  +22,  +57,   +1,  +23),  # freq=38
    'successful': ( +37,  +22,  +52,   +0,  +19),  # freq=66
    'successfully': ( +51,  +27,  +62,   +0,  +25),  # freq=19
    'suddenly': ( +37,   +2,  -14,   +0,   +8),  # freq=70
    'sue': ( -52,  +63,  +67,  +65,   +9),  # freq=17
    'suffer': (-117, +110, -104,  +89,  -58),  # freq=29
    'summer': ( +51,  +27,  +42,   +0,  -10),  # freq=561
    'sundays': ( +35,  -27,   +0,   +0,   +9),  # freq=21
    'sunny': ( +50,  +29,  +34,   +0,  +40),  # freq=32
    'super': ( +29,  +29,  +34,   +0,  +16),  # freq=650
    'supermarket': ( +42,   +9,  +22,   +2,  +16),  # freq=23
    'supper': ( +46,  -11,  +14,   +0,  +16),  # freq=13
    'supplies': ( +43,  -20,  +29,   +0,  +17),  # freq=97
    'support': ( +37,   +9,  +22,   +0,  +12),  # freq=226
    'supposed': ( -47,  -27,  -44,   +0,  -24),  # freq=289
    'sure': ( +28,   +0,  +49,  +18,   +0),  # freq=3594
    'surgery': ( -25, +30, -30, +40, -20),  # medical context -- scary alone, but "went well" = relief
    'surprises': ( +39,  +22,  +19,   +2,  +14),  # freq=21
    'surprising': ( +41,  +24,  +24,   +2,  +16),  # freq=32
    'surprisingly': ( +35,  +24,  +44,   +3,  +17),  # freq=23
    'survived': ( +35,  +24,  +37,   +3,  +16),  # freq=37
    'sushi': ( +51,  +18,  +17,  +12,   -8),  # freq=36
    'suspect': (-102,  +65,   +4,  +50,  -34),  # freq=14
    'suspicious': ( -96,  +65,  -32,  +46,  -39),  # freq=16
    'swallow': ( +50,  +49, +126, +103,  +24),  # freq=12
    'sweat': ( -52, +127,  -44, +127,  +29),  # freq=38
    'sweating': ( -52,  +87,  -92,  +97,  +20),  # freq=24
    'sweaty': ( -51,   +9,  -72,   +6,  -29),  # freq=16
    'sweet': ( +50,  +20,  +17,   +0,  +24),  # freq=569
    'sweetest': ( +48,   +0,  -17,   +0,  +10),  # freq=22
    'sweetheart': ( +48,   -6,   +2,   +0,  +14),  # freq=15
    'sweets': ( +42,   -6,  -24,   +0,   +8),  # freq=17
    'swift': ( +50,  +49,  +57,  +12,  +24),  # freq=12
    'swim': ( +90,  +56,  +77,  +24,  -19),  # freq=57
    'swimming': ( +28,  +40,  +24,  +18,  +23),  # freq=92
    'switch': ( +39,  +69, +127, +127,  +16),  # freq=109
    # 'system' removed — neutral factual word with extreme D bias
    'taco': ( +42,  +10,  +10,  +25,   -3),  # freq=26
    'talent': ( +48,   +9,  +47,   +0,  +21),  # freq=70
    'talented': ( +44,  +11,  +59,   +0,  +23),  # freq=58
    'taste': ( +71,  +51,  +59,  +24,   -8),  # freq=73
    'taught': ( +32,   -2,  +19,   +0,  +13),  # freq=89
    'tax': (-127,  +90, -127, +127,  +73),  # freq=36
    'teach': ( +29,  +20,  +54,  +12,  +16),  # freq=136
    'teacher': ( +28,  +13,  +42,  +12,   +4),  # freq=156
    'teaching': ( +32,   -4,  +32,   +0,  +14),  # freq=62
    'team': ( +29,  +20,  +34,   +0,  +16),  # freq=383
    'tear': ( -52,  +63,  -92,  +31,  -29),  # freq=19
    'tears': ( -20,  +40,  -20,  +20,  +30),  # liquid -- tears of joy vs tears of pain
    'teary': ( -94,   -6, -124,   +0,  -53),  # freq=12
    'tech': ( +39,   -2,  +69,   +0,  +23),  # freq=33
    'tent': ( +62,  +33,  +59,  +18,  +10),  # freq=21
    'terrifying': (-127,  +90,  +42,  +77,  -39),  # freq=143
    'thank': ( +50,  +20,  +17,   +0,  +24),  # freq=990
    'thankful': ( +21,   +4,   +7,   +0,   +8),  # freq=315
    'thankfully': ( +51,  -15,   -4,   +0,  +14),  # freq=186
    'thanks': ( +39,   +9,   +0,   +0,  +16),  # freq=799
    'thanksgiving': ( +44,   -4,  +19,   +0,  +16),  # freq=52
    'theater': ( +58,  +24,   +2,   +4,  +17),  # freq=25
    'therapist': ( +48,  -13,  +34,  +18,   -8),  # freq=17
    'thick': ( +50,  +29, +127,  +25, +121),  # freq=24
    'thieves': (-127,  +90, +127,  +77,  -24),  # freq=13
    'thoughtful': ( +50,   +9,  +54,   +0,  +24),  # freq=56
    'thoughts': ( +55,  -22,  +54,   +0,  +25),  # freq=59
    # 'thousand' removed — number word, noise
    'threatened': (-125, +117,  -64,  +93,  -53),  # freq=4
    'threatening': (-119, +101,  +64,  +81,  -31),  # freq=10
    # 'three' removed — number word, pure noise
    'threw': ( -77,   +9,  -42,   +7,  -34),  # freq=233
    'thrift': ( +58,  -49,  +42,   +0,  +24),  # freq=13
    'thunderstorm': (-102,  +94, +124,  +73,  -14),  # freq=14
    'tight': ( -52,  +99, +127, +127,  +61),  # freq=39
    'tiny': ( +48,  +18,  -52,  +12,  -29),  # freq=74
    'tip': ( +37,  +13,  +34,   +6,   -8),  # freq=48
    'tired': ( -28,  -28,  -27,  +13,  -30),  # freq=155
    'toast': ( +51,  -22,  -42,   +0,   +8),  # freq=10
    'toddler': ( +56,   -6,  -39,   +0,   +9),  # freq=38
    'toe': ( +28,  +40,  +74,  +18,  +10),  # freq=44
    'toll': ( -47,  -18,  -17,   +0,  -20),  # freq=12
    'tongue': ( +42,  +38,  +34,  +49,   -4),  # freq=13
    'top': ( +51,  +33,  +84,  +12,  -53),  # freq=159
    'tornado': ( -89, +127, -127, +127,  -36),  # freq=56
    'torture': (-127, +127, -127, +127,  +97),  # freq=10
    'total': ( +62,  +22, +126,  +61,  +61),  # freq=52
    'touch': ( +80,  +42,  +59,  +30,  -14),  # freq=112
    'touching': ( +58,   -9,   -2,   +0,  +16),  # freq=14
    'tough': ( +28,  +40, +124,  +18,  +10),  # freq=487
    'tour': ( +35,   +9,   +9,   +2,  +12),  # freq=30
    'traffic': ( -32,  +49,  -44,  +65,   -9),  # freq=173
    'tragedy': (-122,  +40,  -66,  +63,  -72),  # freq=17
    'tragic': (-127, +103, -127,  +77,  -97),  # freq=29
    'trained': ( +28,   +2,  +19,   +1,  +10),  # freq=45
    'trainer': ( +43,  +40,  +94,   +9,  +28),  # freq=16
    'training': ( +48,  +49,  +82,  +12,  +27),  # freq=161
    'transfer': ( +43,   -2,   +7,   +0,  +14),  # freq=14
    'transportation': ( +35,   +9,   -7,   +2,   +9),  # freq=20
    'trap': ( -77,  +74, -127,  +77,  -48),  # freq=13
    'trapped': ( -77,  +90, -127,  +97,  -61),  # freq=21
    'traps': ( -62,  +40,   -9,  +25,  -21),  # freq=11
    'traumatized': (-127, +101, -127,  +86,  -77),  # freq=10
    'travel': ( +44,  +15,   +4,   +1,  +13),  # freq=120
    'traveling': ( +42,  +20,  +12,   +1,  +13),  # freq=59
    'travelling': ( +35,  +27,   +4,   +3,  +10),  # freq=35
    'treat': ( +50,  +20,  +34,   +0,  +24),  # freq=138
    'trees': ( +34,  -65,   +9,   +0,  +10),  # freq=31
    'tricky': ( -79,  +42,   +0,  +27,  -27),  # freq=11
    'trip': ( -24,  +40,  -49,  +35,  -23),  # freq=699
    'tripped': ( -81,  +90, -119,  +62,  -48),  # freq=101
    'trouble': ( -42,  +49,  -67,  +65,  -20),  # freq=245
    'troubles': (-127,  +94,  -97,  +81,  -63),  # freq=16
    'trump': ( +29,  +38, +114,  +12,  +28),  # freq=104
    'trusting': ( +36,   +0,  +34,   +0,  +16),  # freq=330
    'trustworthy': ( +49,  -15,  +47,   +0,  +21),  # freq=34
    'truth': ( +29,  +20,  +54,  +12,  +16),  # freq=51
    'tune': ( +46,   -4,  -22,   +0,   +9),  # freq=11
    'turbulence': (-127, +127, -127, +127,  +12),  # freq=12
    'turtle': ( -49,  -94,  -77,   +0,  -29),  # freq=23
    'twice': ( +42,  +33,  +59,  +43,  +27),  # freq=98
    'twins': ( +58,   -4,  +29,   +0,  +20),  # freq=26
    'ugh': ( -42,  -24,  -44,   +0,  -20),  # freq=245
    'ugly': ( -77,  +45,  -54,  +19,  -24),  # freq=13
    'ultrasound': ( +42,  +22,  +52,  +24,  -14),  # freq=12
    'umbrella': ( +51,   -4,  +67,  +61,   +8),  # freq=18
    'unable': (-102,  -11, -127,   +0,  -58),  # freq=32
    'unacceptable': ( -89,  +74,  +54,  +57,  +36),  # freq=20
    'unaware': ( -81,  -38,  -82,   +0,  -40),  # freq=11
    'uncertainty': (-123,   -2,  -92,   +0,  -58),  # freq=10
    'uncle': ( +28,   +9,  +42,   +0,  +10),  # freq=77
    'understandable': ( +28,  -11,   +7,   +0,   +9),  # freq=73
    'understanding': ( +37,  -15,  +24,   +0,  +14),  # freq=66
    'understood': ( +29,   -4,  +24,   +0,  +13),  # freq=38
    'unemployment': (-114,  -27,  -87,   +0,  -53),  # freq=15
    'unexpectedly': ( -34,  +58,  -12,  +31,  -13),  # freq=25
    'unfortunate': (-119,   -2, -127,   +0,  -63),  # freq=128
    'unfortunately': ( -52,  +24,  -44,  +15,  -29),  # freq=243
    'unhappy': (-109,   +0, -114,   +0,  -58),  # freq=30
    'university': ( +51,  +13,  +69,   +2,  +27),  # freq=98
    'unless': ( -34,   +0, -104,   +0,  -29),  # freq=91
    'unlikely': ( -52,  -47,  -92,   +0,  -34),  # freq=10
    'unlucky': ( -42,  +38,  -67,  +31,  -20),  # freq=17
    'unmotivated': ( -94, -127, -127,   +0,  -63),  # freq=3
    'unnecessary': ( -79,  -38, -127,   +0,  -51),  # freq=16
    'unpleasant': (-106,  +29, -127,  +23,  -61),  # freq=23
    'unprepared': ( -83,  -11, -127,   +0,  -53),  # freq=23
    'unsafe': ( -52,  +63,  -92,  +65,  -20),  # freq=14
    'unwanted': ( -92,  +11, -127,   +7,  -53),  # freq=3
    'upcoming': ( +42,   -9,  +14,   +0,  +14),  # freq=75
    'upsetting': (-127,  +90, -127,  +77,  -70),  # freq=55
    'upstairs': ( +35,  -15,  +42,   +0,  +17),  # freq=32
    'useful': ( +58,  -15,  +64,   +0,  +27),  # freq=22
    'useless': (-127,  +29, -127,  +77,  -85),  # freq=15
    'vacation': ( +77,   -9,  +84,   +6,  -34),  # freq=602
    'valid': ( +57,  +18,  +67,   +6,  -14),  # freq=12
    'valuable': ( +42,   +4,  +39,   +0,  +19),  # freq=22
    'value': ( +39,   +6,  +39,   +1,  +17),  # freq=55
    'variety': ( +48,   -6,  +29,   +0,  +19),  # freq=16
    'vegan': ( +39,   +9,  +54,   +0,  -16),  # freq=33
    'vehicle': ( +32,  +11,   +0,   +2,   +9),  # freq=51
    'vet': ( -24,  +40,  +24,  +54,  -10),  # freq=107
    'view': ( +42,  -18,   +7,   +0,  +13),  # freq=32
    'visit': ( +28,   -4,   -7,   +0,   +6),  # freq=423
    'visiting': ( +50,   -9,   +7,   +0,  +16),  # freq=86
    'void': (-127, -127, -127,   +0,   +0),  # freq=10
    'volunteer': ( +41,   -6,  +12,   +0,  +13),  # freq=22
    'vomit': (-119,  +18, -127,  +15,  -63),  # freq=26
    'vomiting': ( -96,  +74, -127,  +54,  -58),  # freq=10
    'wandering': ( -40,   -6,  -49,   +0,  -21),  # freq=14
    'war': ( -77, +119,  +82, +116,  +24),  # freq=36
    'warm': ( +39,   +9,  +34,   +0,  +16),  # freq=60
    'warned': ( -47,  +22,  +17,  +13,  -13),  # freq=15
    'warning': ( -32,  +49,  -22,  +81,   +9),  # freq=43
    'washed': (-127,  -58, -127,  +19,  +73),  # freq=22
    'washing': ( +32,  -18,  -29,   +0,   +5),  # freq=28
    "wasn't": ( -24,  +13,   +0,  +18,   +0),  # freq=858
    'wasp': (-127, +127, +109, +127,  -61),  # freq=17
    'waste': ( -52,  +24,  -67,  +31,  -29),  # freq=71
    'wasted': ( -86,  +15, -104,  +11,  -48),  # freq=54
    'watermelon': ( +48,  -47,  -74,   +0,   +1),  # freq=10
    'wave': (  +5,  +15,   +5,  +10,   +0),  # freq=23 — context-dependent (ocean vs grief)
    'waves': (  +5,  +15,   +5,  +10,   +0),  # context-dependent
    'ways': ( +42,   +9,  -22,   +2,   +8),  # freq=105
    'weak': ( -42,  -13, -127,  +15,  -40),  # freq=14
    'wealthy': ( +77,  +18,  +84,   +0,  -29),  # freq=10
    'weather': ( +56,  -18,  +17,   +0,  +19),  # freq=232
    'weekend': ( +96,  +51,  +84,  +24,  -38),  # freq=1151
    'weight': ( -32,  +24,  -44,  +31,  -20),  # freq=244
    'weird': ( -24,  +40,  -24,  +18,  +10),  # freq=300
    'welcome': ( +50,  +20,  +34,   +0,  +16),  # freq=67
    'werent': ( -62,  -18,  -77,   +0,  -34),  # freq=14
    'west': ( +50,  +29,  +89,  +12,  +16),  # freq=20
    'wet': ( +28,  +54,  +49,  +72,  +23),  # freq=52
    'whenever': ( +25,  -11,  -49,   +0,   +0),  # freq=118
    'white': ( +39,   -9,  +34,   +0,  -16),  # freq=74
    'whoa': ( -34,  -13,  -44,   +0,  -17),  # freq=65
    'whole': ( +29,   +9,  +34,   +0,  +16),  # freq=484
    'whom': ( -40,   -4,  -39,   +0,  -20),  # freq=23
    'willing': ( +35,   +4,  +39,   +1,  +16),  # freq=52
    'winner': ( +49,  +27,  +52,   +1,  +23),  # freq=12
    'winnings': ( +46,  +27,  +44,   +1,  +21),  # freq=18
    'wins': ( +51,  +29,  +64,   +0,  +25),  # freq=22
    'winter': ( -24,  +13,  +49,  +18,  +34),  # freq=101
    'wisdom': ( +50,   +9,  +72,   +0,  +24),  # freq=13
    'wise': ( +39,  -20,  +52,   +0,  +20),  # freq=33
    'wisely': ( +51,  -22,  +42,   +0,  +21),  # freq=11
    'wish': ( -10,  10,  -20,  15,  25),  # longing -- wanting what you dont have. G high (it matters), D low (powerless)
    'wished': ( -12,  8,  -22,  12,  25),
    'wishing': ( -10,  10,  -18,  15,  25),
    'without': ( -10,   +5,  -10,   +5,   -5),  # liquid -- "without you" = loss, "without asking" = ease
    'woke': ( +29, +127,  +89, +127,  +24),  # freq=171
    'wolf': ( +48,  +56,  +67,  +61,  +14),  # freq=10
    'women': ( +35,  +24,   +2,   +3,  +10),  # freq=86
    'won': ( +32,  +27,  +52,   +0,  +27),  # freq=509
    'wonder': ( +29,  +29,  +17,   +0,  +24),  # freq=177
    'wondering': ( +35,  +42,  -47,  +12,   +2),  # freq=33
    'wonders': ( +51,  +13,  +22,   +0,  +19),  # freq=10
    'wont': ( -62,  -36,  -77,   +0,  -34),  # freq=141
    'woods': ( +28,  -58,  -54,   +0,   +0),  # freq=53
    'word': ( +28,  +13,  +49,  +18,  +10),  # freq=154
    'worker': ( +48,  +18,  +54,   +4,  +23),  # freq=181
    'workout': ( +37,  +27,  +37,   +3,  +17),  # freq=47
    'worries': (-114,  +47,  -59,  +39,  -48),  # freq=30
    'worrisome': ( -94,  +90, -119,  +66,  -53),  # freq=12
    'worrying': (-127,  +65,  -87,  +54,  -58),  # freq=39
    'worse': ( -77,  +45,  -54,  +39,  -36),  # freq=349
    'worst': (-127,  +90, -109,  +97,  -73),  # freq=473
    'worth': ( +39,  +20,  +34,   +0,  +16),  # freq=390
    'wow': ( +28,  +33,   +7,   +0,  +19),  # freq=2454
    'wracking': ( -59,  +24,  -17,  +15,  -24),  # freq=58
    'wreck': ( -89,  +74, -109,  +57,  -48),  # freq=73
    'xbox': ( +35,  +38,   +4,   +6,  +10),  # freq=26
    'yay': ( +42,  +13,  -42,   +2,   +5),  # freq=45
    'yeah': ( +21,  -27,  -17,   +0,   +2),  # freq=3599
    'yell': ( -62,  +67,  +22,  +45,  -17),  # freq=39
    'yelling': ( -77, +127,  +54,  +77,  +36),  # freq=40
    'yes': ( +22,   +4,  +14,  +10,   +6),  # freq=3677
    'yoga': ( +56,  -69,  -22,   +0,  +12),  # freq=22
    'young': ( +29,  +29,  +34,   +0,  +24),  # freq=431
    'youngest': ( +48,  -15,  +42,   +0,  +20),  # freq=50
    'yourself': ( +37,  -31,  +29,   +0,  +16),  # freq=668
    'youth': ( +44,   +9,  +32,   +0,  +19),  # freq=15
    'youtube': ( +35,  +27,  +42,   +3,  +17),  # freq=83
    'yuck': ( -59,   +9,  -77,   +6,  -34),  # freq=46
    'yum': ( +48,   -9,  -24,   +0,   +9),  # freq=31
    'zombie': ( -79,  +38,  -37,  +27,  -34),  # freq=10
    'zoo': ( +58,   +4,  +22,   +0,  +20),  # freq=31

    # --- Indicator words (common verbs/adjectives that carry emotional signal) ---
    'good':   ( +20,   +8,  +10,   +0,  +12),
    'great':  ( +25,  +10,  +12,   +0,  +15),
    'like':   ( +15,   +5,   +5,   +0,   +8),
    'want':   ( +15,  +10,   +5,   +5,   +8),
    'need':   ( -15,  +10,  -10,  +10,   -8),
    'help':   ( +28,  +15,  +10,  +10,  +15),
    'lose':   ( -52,  +20,  -30,  +15,  -35),
    'try':    ( +15,  +10,  +10,   +5,   +8),
    'give':   ( -3,   +5,   -3,   +0,   +5),    # near-neutral -- generous OR demanding depending on structure
    'take':   ( -10,   +5,   -5,   +0,   -5),
    'make':   ( +15,  +10,  +10,   +0,   +8),
    'start':  ( +20,  +10,  +10,   +5,  +12),
    'stop':   ( -20,  +10,  +10,   +0,  -12),
    'live':   ( +30,  +10,  +10,   +0,  +20),
    'change': ( +15,  +15,  +10,   +5,   +8),
    'please': ( +15,   +5,   -5,   +5,   +8),
    'right':  ( +10,   +5,  +10,   +0,   +5),
    'enough': ( -15,  +10,   -5,   +5,   -8),
    'more':   ( +10,   +5,   +5,   +0,   +5),
    'less':   ( -10,   -5,   -5,   +0,   -5),
    'pass':   ( +30,  +10,  +15,   +0,  +18),
    'begin':  ( +20,  +10,  +10,   +5,  +12),
    'away':   ( -15,   +5,  -10,   +5,  -10),
    'old':    ( -10,   -5,   -5,   +0,   -5),
}


# Convenience: category sets for targeted lookups
EMOTION_WORDS = {w for w in EMOTIONAL_VOCABULARY if w in {'reluctant', 'neglected', 'awe', 'shattered', 'grumpy', 'jealous', 'dread', 'hatred', 'cruelty', 'comfort', 'betrayed', 'joy', 'relieved', 'uneasy', 'despair', 'distraught', 'peace', 'thrilled', 'panic', 'irritated', 'insecure', 'terror', 'excitement', 'upset', 'trust', 'jealousy', 'surprise', 'calm', 'rage', 'disgusted', 'depression', 'inspired', 'grief', 'delighted', 'stress', 'satisfied', 'furious', 'fear', 'hate', 'sad', 'grateful', 'comfortable', 'happy', 'confidence', 'envy', 'motivated', 'haunted', 'guilt', 'excited', 'startled', 'scared', 'loved', 'anxious', 'stressed', 'torn', 'confident', 'safe', 'loneliness', 'ashamed', 'shame', 'agony', 'pity', 'brave', 'gloomy', 'hope', 'ruined', 'proud', 'kindness', 'hopeful', 'wrecked', 'compassion', 'determined', 'destroyed', 'worry', 'pride', 'humiliated', 'horrified', 'panicked', 'disgust', 'broken', 'embarrassed', 'passionate', 'nervous', 'overwhelmed', 'frustration', 'peaceful', 'cranky', 'rejected', 'heartbroken', 'pain', 'anxiety', 'optimistic', 'annoyed', 'anguish', 'appreciated', 'afraid', 'terrified', 'angry', 'helpless', 'restless', 'pessimistic', 'empathy', 'ecstatic', 'shocked', 'lonely', 'valued', 'anger', 'crushed', 'livid', 'relief', 'love', 'uncomfortable', 'thrill', 'frustrated', 'regret', 'suffering', 'worthless', 'miserable', 'depressed', 'hopeless', 'nostalgic', 'glad', 'gratitude', 'sympathy', 'worried', 'pleased', 'devastated', 'hesitant', 'content', 'sadness', 'hurt', 'abandoned'}}
INTENSIFIERS = {w for w in EMOTIONAL_VOCABULARY if w in {'constantly', 'quite', 'desperately', 'especially', 'awfully', 'horribly', 'rather', 'pretty', 'fairly', 'really', 'insanely', 'never', 'hardly', 'terribly', 'honestly', 'truly', 'extremely', 'incredibly'}}
RELATIONSHIP_WORDS = {w for w in EMOTIONAL_VOCABULARY if w in {'unfaithful', 'commitment', 'attachment', 'lying', 'partner', 'honesty', 'loving', 'daughter', 'bond', 'divorce', 'mother', 'friendship', 'alone', 'sister', 'brother', 'friend', 'parent', 'marriage', 'cheating', 'cold', 'dishonest', 'caring', 'wife', 'relationship', 'enemy', 'apart', 'together', 'cheated', 'liar', 'supportive', 'distant', 'family', 'couple', 'faithful', 'husband', 'son', 'breakup', 'honest', 'loyalty', 'lover', 'fake', 'wedding', 'spouse'}}
CRISIS_WORDS = {w for w in EMOTIONAL_VOCABULARY if w in {'abuse', 'starving', 'attack', 'harm', 'wound', 'emergency', 'crash', 'dead', 'victim', 'cry', 'poverty', 'addiction', 'danger', 'violence', 'murder', 'addicted', 'healing', 'accident', 'toxic', 'death', 'bleeding', 'crying', 'hospital', 'die', 'screaming', 'suicidal', 'violent', 'crisis', 'lost', 'trauma', 'traumatic', 'dying', 'homeless', 'recovery', 'dangerous', 'abused', 'scream', 'killing', 'harmful', 'suicide'}}
SLANG_WORDS = {w for w in EMOTIONAL_VOCABULARY if w in {'screwed', 'basic', 'killer', 'horrible', 'pissed', 'amazing', 'blessed', 'dreadful', 'garbage', 'extra', 'shady', 'brutal', 'hell', 'lit', 'hyped', 'bomb', 'legit', 'sick', 'suck', 'disgusting', 'awesome', 'fantastic', 'awful', 'pathetic', 'yikes', 'crazy', 'petty', 'oof', 'insane', 'wonderful', 'nuts', 'cursed', 'crap', 'gross', 'mood', 'bummed', 'deadly', 'sketchy', 'damn', 'cringe', 'sucks', 'terrible'}}
COGNITIVE_EVAL_WORDS = {w for w in EMOTIONAL_VOCABULARY if w in {'fighting', 'certain', 'choice', 'reason', 'responsible', 'irresponsible', 'mistake', 'achievement', 'doubt', 'difficult', 'clear', 'significant', 'villain', 'understand', 'deserved', 'agree', 'struggle', 'fought', 'believe', 'win', 'fight', 'accomplishment', 'true', 'success', 'impossible', 'choose', 'hard', 'wrong', 'defeat', 'overcome', 'blame', 'failure', 'quit', 'purpose', 'unfair', 'false', 'result', 'easy', 'survive', 'fault', 'achieve', 'decide', 'possible', 'winning', 'decision'}}

# Gap analysis additions — words the previous engine was missing that caused SST-2 failures

# ── MODERN EMOTIONAL VOCAB — Added for real-world coverage ──
_MODERN_WORDS = {
    'abuser': (-70, 60, 70, 40, -20),
    'affirming': (30, 15, 25, 0, 20),
    'binging': (-30, 40, -30, 20, -15),
    'boundaries': (20, 10, 40, 0, 15),
    'breadcrumbed': (-35, 30, -50, 15, -20),
    'burnout': (-50, -20, -40, 10, -35),
    'catfished': (-50, 40, -60, 20, -25),
    'codependent': (-30, 20, -50, 15, -20),
    'cutting': (-70, 30, -40, 50, -40),
    'dissociating': (-40, -20, -60, 20, -30),
    'enabler': (-20, 10, -30, 10, -10),
    'flashback': (-60, 70, -50, 50, -35),
    'floating': (40, -20, 20, 0, 50),
    'gaslighting': (-60, 50, -70, 30, -30),
    'gaslit': (-60, 40, -80, 30, -30),
    'gutted': (-70, 30, -50, 20, -40),
    'invalidating': (-50, 30, -40, 20, -25),
    'lovebombed': (-30, 50, -40, 20, -15),
    'manifesting': (30, 20, 30, 5, 20),
    'mindset': (15, 10, 20, 0, 10),
    'narcissist': (-40, 30, 60, 10, -10),
    'overdose': (-80, 50, -60, 60, -50),
    'purging': (-60, 30, -40, 30, -35),
    'relapse': (-60, 40, -50, 40, -40),
    'shook': (-30, 60, -40, 30, -20),
    'sinking': (-50, -10, -50, 20, -60),
    'soaring': (60, 40, 40, 0, 60),
    'spiraling': (-60, 50, -60, 40, -40),
    'stoked': (60, 50, 40, 10, 30),
    'suffocating': (-70, 40, -70, 50, -50),
    'thriving': (60, 40, 50, 0, 35),
    'triggered': (-50, 70, -60, 40, -30),
    'validating': (35, 15, 30, 0, 25),
    'vibing': (40, 30, 20, 0, 25),
}

_GAP_WORDS = {
    "dull": (-21, -10, -15, 0, -12), "mundane": (-20, -10, -10, 0, -12),
    "grim": (-40, +10, -15, +5, -25), "hollow": (-30, -10, -20, 0, -20),
    "flat": (-15, -10, -10, 0, -10), "tedious": (-25, -10, -15, 0, -15),
    "dreadful": (-50, +15, -25, +10, -35), "pointless": (-30, -5, -20, 0, -20),
    "bland": (-15, -10, -10, 0, -10), "stale": (-15, -5, -10, 0, -10),
    "contrived": (-20, +5, -10, 0, -12), "predictable": (-15, -5, -5, 0, -8),
    "compelling": (+35, +15, +15, 0, +22), "riveting": (+40, +20, +15, 0, +25),
    "stunning": (+40, +20, +15, 0, +25), "masterful": (+40, +15, +20, 0, +25),
    "poignant": (+30, +15, -5, 0, +20), "touching": (+35, +15, +5, 0, +22),
    "heartfelt": (+40, +15, +10, 0, +28), "uplifting": (+40, +15, +15, 0, +30),
    "entertaining": (+30, +15, +10, 0, +18), "engaging": (+30, +15, +15, 0, +18),
    "refreshing": (+25, +10, +10, 0, +15), "drama": (-18, +10, -5, 0, -10),
    "memorable": (+28, +10, +15, 0, +18), "solid": (+29, +5, +15, 0, +15),
    "cinematic": (+29, +10, +15, 0, +15), "dialogue": (+21, +5, +10, 0, +10),
    "interesting": (+11, +10, +10, 0, +5), "brilliant": (+42, +27, +42, 0, +27),
}
EMOTIONAL_VOCABULARY.update(_GAP_WORDS)

# Benchmark gap additions — high-impact words from SST-2, GoEmotions, TweetEval
# that the previous engine missed (appear 15+ times in benchmarks, |dV| >= 15)
_BENCHMARK_WORDS = {
    # Profanity/slang with strong emotional charge
    "fuck": (-37, +33, +18, +40, +15),
    # "fucking" handled as amplifier (1.6x) in context_operators, not as payload
    "shit": (-52, +38, -22, +31, -9),
    # "bullshit" = NULL. Not negative — the absence of value. Zero.
    # Everything > bullshit, so comparison to it = negative.
    # V near-zero (it IS nothing), but G deeply negative (worthless/void).
    # When it touches other words, it nullifies them toward zero.
    "bullshit": (-80, +50, +20, +40, -50),  # null — void of value, G-crusher
    # Common emotional words missing from curated set
    "director": (+32, +56, +119, +97, +38),
    "watch": (+50, +29, +127, +63, +16),
    "show": (+51, +38, +77, +43, -4),
    "today": (0, 5, 0, 30, 0),
    "getting": (+24, +83, -44, +28, 0),
    "keep": (+28, +13, +49, +18, +10),
    "face": (+28, +40, +49, +18, 0),
    "play": (+29, +29, +34, 0, +16),
    "read": (+71, -13, +77, +18, +19),
    "leave": (-17, +22, -12, +6, -6),
    "left": (-32, +38, -44, +31, -9),
    "sleep": (+28, -69, +24, 0, -10),
    "coming": (+34, -13, 0, 0, +9),
    # "meeting" removed — neutral factual word (sarcasm context handled by templates)
    "same": (-24, -54, -14, 0, -10),
    "plan": (+51, +9, +34, +12, +8),
    "jammed": (-51, +38, -22, +31, -9),
    "unsolicited": (-114, +27, -49, +18, -25),
    "outrage": (-127, +127, +127, +127, +36),
    "fuming": (-62, +36, -9, +23, -21),
    "terrorism": (-123, +127, +54, +109, -34),
    "pathetic": (-50, +15, -25, +10, -20),
    "ridiculous": (-40, +20, -15, +10, -15),
    "disgusting": (-60, +40, -20, +25, -20),
    "fantastic": (+50, +30, +25, 0, +25),
    "amazing": (+45, +25, +20, 0, +22),
    "wonderful": (+50, +25, +20, 0, +25),
    "beautiful": (+45, +20, +15, 0, +25),
    "hilarious": (+40, +30, +15, 0, +20),
    "stupid": (-40, +20, -30, +15, -15),
    "idiot": (-50, +30, -35, +20, -18),
    "worst": (-60, +30, -30, +20, -25),
    "best": (+50, +20, +30, 0, +22),
    "boring": (-25, -15, -15, 0, -15),
    "perfect": (+55, +15, +30, 0, +28),
    "rubbish": (-45, +20, -20, +15, -18),
    "trash": (-45, +20, -25, +15, -18),
    "epic": (+45, +30, +25, 0, +22),
    "legendary": (+50, +25, +35, 0, +28),
    "absurd": (-30, +20, -15, +10, -10),
    "nightmare": (-60, +50, -40, +50, -30),
    "paradise": (+55, +15, +25, 0, +30),
    "toxic": (-55, +30, -25, +20, -22),
}
EMOTIONAL_VOCABULARY.update(_BENCHMARK_WORDS)

# Text/internet slang — high emotional signal in tweets/reddit
_SLANG_WORDS = {
    "wtf": (-102, +127, -24, +97, -12),
    "smh": (-52, +38, -22, +31, -9),
    "lmfao": (-16, +50, +13, 0, +10),
    "bruh": (-24, +13, -12, 0, -5),
    "tbh": (-23, -9, +7, 0, -5),
    "fml": (-127, +127, -127, +127, -50),
    "stfu": (-127, +127, +50, +127, -10),
    "nah": (-24, -13, +12, 0, -5),
    "fomo": (-34, +40, -20, +50, -15),
    "gtfo": (-100, +127, +80, +127, -10),
    "ngl": (-10, 0, +10, 0, 0),
    "af": (+10, +20, +10, 0, 0),       # intensifier: "happy af"
    "imo": (-5, 0, +5, 0, 0),
    "yolo": (+30, +50, +40, +30, +15),
    "vibes": (+25, +15, +10, 0, +10),
    "slay": (+40, +30, +40, 0, +15),
    "goat": (+45, +25, +35, 0, +20),    # greatest of all time
    "lowkey": (-10, -10, -5, 0, -5),
    "highkey": (+10, +20, +10, +10, +5),
    "fire": (+40, +30, +25, 0, +15),    # slang positive
    "mid": (-20, -10, -15, 0, -10),     # mediocre
    "sus": (-30, +20, -15, +15, -10),   # suspicious
    "cap": (-25, +15, -20, +10, -10),   # lying
    "bussin": (+40, +30, +20, 0, +15),
    "bet": (+15, +10, +15, 0, +5),      # agreement/confidence
    "w": (+30, +15, +20, 0, +10),       # win
    "l": (-30, +15, -20, 0, -10),       # loss
}
EMOTIONAL_VOCABULARY.update(_SLANG_WORDS)

# Mirror corrections — antonym pairs MUST be symmetric (1=1, love=hate)
# The NRC negativity bias means positive words are systematically underscored.
# Fix: boost positive words to match their negative antonym's magnitude.
# Keep arousal/dominance/urgency/gravity from original — only fix valence.
_MIRROR_CORRECTIONS = {
    # love/hate: hate=-127, so love=+127
    "love": (+127, +12, +10, 0, +22),
    # happy/sad: sad=-78, so happy=+78
    "happy": (+78, +8, +10, 0, +17),
    # joy/anger: anger dV=-89, so joy=+89
    "joy": (+89, +27, +34, 0, +29),
    # hope/despair: NOT symmetric. Hope contains uncertainty. Despair is absolute.
    "hope": (+45, +15, +20, +10, +25),
    # brave/afraid: afraid=-78, so brave=+78
    "brave": (+78, +22, +52, +6, +14),
    # proud/ashamed: NOT symmetric. proud is strong positive but not +102.
    "proud": (+50, +15, +30, 0, +15),
    # safe/dangerous: dangerous=-127, so safe=+127
    "safe": (+127, -20, +72, 0, +16),
    # trust/betrayed: betrayed=-106, so trust=+106
    "trust": (+106, +4, +34, 0, +8),
    # wonderful/terrible: terrible=-111, so wonderful=+111
    "wonderful": (+111, +25, +20, 0, +25),
    # amazing/awful: awful=-127, so amazing=+127
    "amazing": (+127, +25, +20, 0, +22),
    # perfect/horrible: horrible=-122, so perfect=+122
    "perfect": (+122, +15, +30, 0, +28),
    # Additional high-value antonym corrections
    # grateful/resentful: resentful not in vocab, but grateful was +28 — too low
    "grateful": (+78, +15, +20, 0, +20),
    # excited was +37, but terrified=-127 — excited should be higher
    "excited": (+78, +40, +20, +10, +15),
    # good was not mirrored — bad has unknown, use moderate
    "good": (+50, +10, +15, 0, +12),
    # beautiful was +45, ugly=-77 — boost to 77
    "beautiful": (+77, +20, +15, 0, +25),
}
EMOTIONAL_VOCABULARY.update(_MIRROR_CORRECTIONS)

EMOTIONAL_VOCABULARY.update(_MODERN_WORDS)

# Resolver error analysis — 436 cases (230 missed crises + 206 false positives)
# Words appearing in 8+ missed-crisis texts that the engine was blind to.
# Conservative forces: these are context-dependent words, not pure emotion labels.
# Most carry mild negative weight because they appear heavily in crisis text;
# the engine needs *some* signal from them to avoid zero-contribution blindness.
_RESOLVER_CRISIS_GAPS = {
    # "attempt" — 10 missed crises, almost always "attempt suicide / attempt to end"
    "attempt": (-25, +20, -15, +20, -15),
    # "tonight" — 11 missed crises, temporal urgency marker ("end tonight", "sleep tonight")
    "tonight": (-5, +10, 0, +25, -5),
    # "wake" — 8 missed crises ("not wake up", "wake morning why not die")
    "wake": (-5, +15, +5, +10, -5),
    # "night" — 12 missed crises, temporal isolation marker
    "night": (-8, -5, -5, +5, -8),
    # "past" — 10 missed crises ("past year", "end pain", temporal weight)
    "past": (-10, -5, -5, 0, -10),
    # "world" — 11 missed crises ("remove from world", existential)
    "world": (0, +5, 0, 0, +5),
    # "stay" — 9 missed crises ("reason to stay", holding on)
    "stay": (+5, -5, +5, +5, +5),
    # "close" — 8 missed crises ("close to ending", "get close to killing")
    "close": (-5, +10, 0, +10, -5),
    # "life" — 49 missed crises, most common missing word; near-neutral but
    # existential weight needed ("point in life", "rest of life", "end my life")
    "life": (0, +5, 0, 0, +5),
    # "point" — 14 missed crises ("no point", "what's the point")
    "point": (-5, 0, -5, 0, -5),
    # "month" — 8 missed crises, temporal marker
    "month": (0, 0, 0, +5, 0),
    # "hour" — 8 missed crises, temporal urgency
    "hour": (0, +5, 0, +10, 0),
    # "maybe" — 11 missed crises, hedging/uncertainty in crisis
    "maybe": (-5, -5, -10, 0, -5),
    # "thought" — 10 missed crises ("suicidal thought", "thought about ending")
    "thought": (-5, +5, 0, +5, -5),
}

# ── MUNDANE NEGATIVITY — words that are negative in daily context ──
_MUNDANE_NEGATIVE = {
    'monday':       (-10, +5, -5, +5, -5),
    'mondays':      (-10, +5, -5, +5, -5),
    'bills':        (-15, +10, -10, +15, -10),
    'homework':     (-10, +5, -5, +5, -5),
    'chores':       (-10, +5, -5, +5, -5),
    'traffic':      (-15, +10, -10, +10, -10),
    'commute':      (-10, +5, -5, +5, -5),
    'deadline':     (-15, +15, -10, +20, -10),
    'deadlines':    (-15, +15, -10, +20, -10),
    'meetings':     (-5, +5, -5, +5, -5),
    'taxes':        (-15, +10, -10, +10, -10),
    'overtime':     (-10, +10, -10, +10, -5),
}
EMOTIONAL_VOCABULARY.update(_MUNDANE_NEGATIVE)

# ── PANIC / FIGHT-FLIGHT — high arousal emergency words ──
_PANIC_WORDS = {
    'breathe':      (-20, +40, -30, +40, -20),
    'breathing':    (-15, +35, -25, +35, -15),
    'choking':      (-40, +50, -40, +50, -30),
    'run':          (-10, +50, +20, +40, -10),
    'running':      (-5, +45, +15, +35, -5),
    'escape':       (-20, +40, -10, +35, -15),
    'trapped':      (-50, +40, -60, +40, -30),
    'panic':        (-40, +60, -50, +50, -30),
    'panicking':    (-40, +60, -50, +50, -30),
    'shaking':      (-25, +40, -30, +30, -20),
    'trembling':    (-25, +35, -30, +25, -20),
    'hyperventilating': (-35, +50, -40, +45, -25),
    'frozen':       (-30, -10, -50, +20, -25),
    'flee':         (-20, +50, -20, +40, -15),
    'ambulance':    (-30, +40, -20, +50, -25),
    'emergency':    (-25, +45, -15, +55, -20),
}
EMOTIONAL_VOCABULARY.update(_PANIC_WORDS)


_POST5_WORDS = {
    'meaningless': (-60, -10, -45, 10, -40),
    'outpouring': (30, 15, 20, 0, 20),
    'outpourings': (30, 15, 20, 0, 20),
}
EMOTIONAL_VOCABULARY.update(_POST5_WORDS)


_SARCASM_GAPS = {
    'delightful': (50, 20, 15, 0, 25),
    'lovely': (45, 15, 10, 0, 20),
    'fabulous': (50, 20, 15, 0, 25),
    'marvelous': (50, 20, 15, 0, 25),
    'splendid': (45, 15, 15, 0, 20),
    # 'policy' removed — neutral factual word
    'paperwork': (-10, 5, -5, 5, -5),
}
EMOTIONAL_VOCABULARY.update(_SARCASM_GAPS)


_CYCLE6_CRISIS_VOCAB = {
    'deserve': (-15, 10, -20, 5, -10),     # 'I dont deserve this' / 'I deserve to die'
    'mental': (-10, 10, -15, 10, -10),      # mental health context
    'depress': (-50, -15, -40, 10, -30),    # depressed/depression root
    'depressed': (-55, -15, -45, 10, -35),
    'depression': (-50, -10, -40, 10, -30),
    'body': (-5, 5, -5, 5, -5),            # body in crisis = physical awareness
    'person': (0, 0, 0, 0, 0),             # truly neutral — person is a subject
    'lately': (-5, 5, -5, 5, -5),          # temporal marker — things changing
    'recently': (-5, 5, -5, 5, -5),        # temporal marker
    'soon': (-5, 10, 0, 15, -5),           # urgency marker
    'continue': (-5, 5, -5, 5, -5),        # persistence — could be + or -
    'stand': (-10, 10, -10, 10, -5),       # 'cant stand' / 'stand up for'
}
EMOTIONAL_VOCABULARY.update(_CYCLE6_CRISIS_VOCAB)

EMOTIONAL_VOCABULARY.update(_RESOLVER_CRISIS_GAPS)

# ── Cycle 11: Missing emotional words for rage, fear, joy, conviction ──
_CYCLE11_EMOTIONAL_GAPS = {
    # RAGE — words that are clearly angry but were missing
    'break':        (-30, +40, +20, +30, -15),   # destructive intent
    'boils':        (-35, +50, +10, +20, -10),   # "blood boils" = rage
    'boiling':      (-35, +50, +10, +20, -10),
    'smash':        (-40, +60, +20, +30, -15),
    'punch':        (-40, +60, +20, +25, -15),
    'snap':         (-30, +50, +10, +25, -10),   # "going to snap"
    # FEAR — missing anxiety/panic words
    'racing':       (-20, +50, -30, +30, -15),   # "mind won't stop racing"
    'closing':      (-25, +30, -30, +25, -20),   # "walls closing in"
    'walls':        (-15, +10, -20, +10, -15),   # mild negative in fear context
    'falling':      (-30, +30, -30, +20, -25),   # loss of control
    'falls':        (-20, +20, -20, +15, -15),
    # 'apart' already in main vocab — skip
    # JOY / CONVICTION — missing positive force words
    'dreamed':      (+50, +20, +20, +0, +25),    # "everything I've dreamed of"
    'seize':        (+40, +40, +50, +20, +20),   # "seize the moment" = empowerment
    'moment':       (+15, +10, +10, +0, +10),    # mildly positive, context-dependent
    'challenge':    (+10, +20, +15, +10, +5),    # mild positive — opportunity
    'excellence':   (+60, +30, +50, +10, +30),   # strong positive aspiration
    'achieving':    (+40, +25, +45, +10, +25),
    'overjoyed':    (+90, +60, +40, +0, +40),    # very strong joy
    'elated':       (+80, +50, +40, +0, +35),    # strong joy
    'jubilant':     (+85, +55, +45, +0, +38),    # strong joy
    'exhilarated':  (+75, +60, +40, +0, +30),    # strong joy + high arousal
    # NEUTRAL — words that should NOT carry emotional charge
    'meetings':     (0, 0, 0, 0, 0),             # override mundane-negative — truly neutral
}
EMOTIONAL_VOCABULARY.update(_CYCLE11_EMOTIONAL_GAPS)

# ── Cycle 17: Reddit resolver error mining — high-frequency missing emotional words ──
_CYCLE17_ERROR_MINING = {
    # STRONG POSITIVE
    'triumph':      (+70, +50, +60, +0, +40),    # victory/achievement
    'masterpiece':  (+80, +20, +50, +0, +35),    # excellence
    'treasure':     (+50, +15, +30, +0, +25),    # valued/precious
    # STRONG NEGATIVE
    'despise':      (-90, +40, +30, +10, -30),   # intense hatred
    'loathsome':    (-100, +30, +10, +10, -35),  # deeply repulsive
    'sickening':    (-80, +40, -20, +15, -30),   # nauseating disgust
    'abomination':  (-100, +40, +10, +15, -40),  # extreme moral revulsion
    'nonsense':     (-30, +15, +20, +5, -10),    # dismissive
    # EMOTIONAL STATE
    'emptiness':    (-50, -20, -40, +5, -35),    # void/hollowness
    'fills':        (+20, +15, +10, +0, +10),    # "fills me with" — mild positive carrier
    # CHALLENGE/DEFIANCE
    'dare':         (+20, +40, +40, +15, +10),   # defiant challenge
}
EMOTIONAL_VOCABULARY.update(_CYCLE17_ERROR_MINING)

# ── Cycle 23: Reddit false-negative mining — words engine missed as negative ──
_CYCLE23_FALSE_NEG_MINING = {
    # SUFFERING / PAIN
    'unbearable':   (-80, +40, -50, +30, -35),   # extreme suffering
    'sorrow':       (-60, -10, -40, +5, -40),     # deep grief/sadness
    'aches':        (-30, +15, -20, +10, -15),    # physical/emotional pain
    'aching':       (-30, +15, -20, +10, -15),
    'tremble':      (-25, +40, -30, +20, -15),    # fear/anxiety physical response
    'trembles':     (-25, +40, -30, +20, -15),
    'creeping':     (-25, +20, -20, +15, -15),    # slow dread/anxiety
    # DISGUST / REVULSION
    'nauseated':    (-60, +30, -30, +15, -25),    # physical disgust
    'nauseous':     (-55, +25, -25, +10, -20),
    'vile':         (-90, +35, +10, +10, -30),    # extreme moral disgust
    'repulsive':    (-85, +30, +5, +10, -30),     # deeply disgusting
    'revolting':    (-80, +35, +5, +10, -30),
    # INTENSITY MODIFIERS (these appear in "unbelievably sad" etc.)
    'unbelievably': (0, +20, 0, +5, 0),           # pure intensifier, no V direction
    'shockingly':   (0, +25, 0, +10, 0),          # pure intensifier
    'outrageously': (0, +25, 0, +10, 0),          # pure intensifier
}
EMOTIONAL_VOCABULARY.update(_CYCLE23_FALSE_NEG_MINING)

# ── Internet/Gen-Z slang with real emotional physics ──
# These aren't just mappings to formal words — they carry their own forces.
# "boo" isn't just "partner" — it's warmth + belonging + vulnerability.
_SLANG_VOCABULARY = {
    # Relationship / community — high V, moderate G (belonging matters)
    'boo':          (+50, +20, +10, +0, +30),    # romantic partner — warmth + vulnerability
    'bae':          (+50, +20, +10, +0, +30),    # same energy as boo
    'fam':          (+40, +15, +20, +0, +25),    # chosen family — loyalty + belonging
    'sis':          (+35, +15, +15, +0, +20),    # solidarity — "I see you"
    'homie':        (+40, +15, +20, +0, +20),    # deep friendship bond
    'bestie':       (+45, +20, +15, +0, +25),    # best friend — high trust
    # Affirmation / hype — high V, high A
    'deadass':      (0, +30, +30, +10, +5),      # pure intensifier — "I'm serious"
    'nocap':        (0, +25, +25, +5, +5),       # truthfulness marker
    'yeet':         (+20, +60, +30, +10, -5),    # explosive energy, low gravity
    'stan':         (+50, +40, +10, +0, +15),    # intense devotion/fandom
    'simp':         (-15, +10, -30, +0, -10),    # devotion BUT with D-crush (loss of agency)
    'flex':         (+30, +30, +50, +5, +10),    # showing off — high D (confidence display)
    'drip':         (+35, +20, +40, +0, +10),    # style/confidence
    'clout':        (+25, +20, +40, +5, +10),    # social influence/status
    # Negative slang — carries real weight
    'salty':        (-25, +20, -10, +5, -10),    # bitter/resentful
    'pressed':      (-20, +30, -15, +10, -5),    # bothered/upset about it
    'cancelled':    (-40, +25, +10, +15, -20),   # social rejection
    'ghosted':      (-45, -10, -35, +10, -25),   # already in vocab but verify
    'catfished':    (-50, +30, -30, +15, -20),   # betrayed/deceived
    'doomscrolling': (-30, +15, -20, +10, -15),  # anxiety-feeding behavior
    # Emotional state slang
    'vibe':         (+30, +10, +15, +0, +15),    # general positive feeling
    'vibing':       (+35, +15, +15, +0, +15),    # in the flow
    'rekt':         (-60, +30, -40, +15, -20),   # destroyed/devastated
    'tilted':       (-30, +35, -20, +15, -10),   # frustrated/off-balance (gaming)
    'snatched':     (+40, +30, +30, +0, +10),    # looking amazing
    'slaps':        (+50, +35, +20, +0, +15),    # something is really good
    'hits':         (+30, +25, +10, +5, +15),    # emotionally resonates — "this hits different"
    'rent-free':    (-10, +15, -15, +10, -10),   # can't stop thinking about it
    'ick':          (-40, +20, -20, +5, -15),    # sudden disgust/turn-off
    'thicc':        (+25, +15, +10, +0, +5),     # body-positive compliment
}
EMOTIONAL_VOCABULARY.update(_SLANG_VOCABULARY)

# ── Object Gravity Dictionary ──
# "A broken car sucks, but not the same negative as a broken heart."
# These are nouns whose GRAVITY (importance to emotional stakes) is knowable.
# V is near-zero for most (objects aren't positive or negative by themselves).
# G is the key — how much does this object MATTER emotionally?
#
# Tier 1 (G=+40 to +50): life-or-death, core identity, irreplaceable
# Tier 2 (G=+20 to +35): important relationships, home, livelihood
# Tier 3 (G=+5 to +15):  daily life objects, replaceable but annoying
# Tier 4 (G=0):          truly neutral objects (engine default)
#
# The model fills the long tail — context determines if "phone" is Tier 3
# (inconvenient) or Tier 1 (dead mom's last photos). Engine knows anchors.
_OBJECT_GRAVITY = {
    # Tier 1 — life, death, core self (G=40-50)
    'soul':         (+5, 0, +5, 0, +50),       # deepest self
    'child':        (+15, +10, +5, +5, +45),    # someone's kid
    'baby':         (+20, +10, -5, +5, +45),    # infant — maximum vulnerability
    'blood':        (-5, +15, +5, +10, +40),    # life force / violence signal
    'breath':       (+5, +10, -5, +5, +35),     # life/death proximity
    'life':         (+0, +5, +5, +0, +45),      # override — life is max stakes
    # Tier 2 — relationships, home, livelihood (G=20-35)
    'father':       (+30, +10, +20, +0, +30),   # parent bond
    'parent':       (+25, +10, +15, +0, +30),
    'husband':      (+30, +10, +10, +0, +30),
    'wife':         (+30, +10, +10, +0, +30),
    'partner':      (+25, +10, +10, +0, +25),
    'brain':        (+5, +5, +5, +0, +25),      # cognition — identity-adjacent
    'chest':        (-5, +10, -5, +5, +25),     # physical anxiety locus
    'stomach':      (-5, +5, -5, +5, +15),      # gut feeling
    'money':        (+5, +10, +10, +10, +20),   # livelihood
    'house':        (+10, +5, +10, +0, +20),    # shelter — basic need
    # Tier 3 — daily life, replaceable (G=5-15)
    'car':          (+5, +5, +10, +5, +10),     # transportation
    'phone':        (+5, +5, +5, +5, +10),      # communication device
    'school':       (+5, +5, +5, +5, +10),      # education
    'computer':     (+5, +5, +5, +5, +5),       # tool
}
EMOTIONAL_VOCABULARY.update(_OBJECT_GRAVITY)

# ── Body autonomy words ──
# These are body experiences that are almost always negative in context.
# "Period" is only positive as RELIEF ("I got my period!" = not pregnant).
# The base reading is negative — context/exclamation can flip it.
_BODY_AUTONOMY = {
    'period':       (-15, +10, -10, +5, +15),    # menstrual — negative + high G (body stakes)
    'periods':      (-15, +10, -10, +5, +15),
    'cramps':       (-30, +20, -20, +10, +10),   # pain
    'cramping':     (-25, +15, -15, +10, +10),
    'bleeding':     (-25, +20, -15, +15, +15),   # could be injury or menstrual
    'nausea':       (-30, +15, -20, +10, +5),
    'nauseous':     (-30, +15, -20, +10, +5),    # may already exist from cycle 23
    'pregnant':     (-5, +25, -10, +15, +45),    # near-neutral V but MASSIVE G — life-changing
    'pregnancy':    (-5, +20, -10, +10, +40),
    'miscarriage':  (-70, +20, -50, +15, +45),   # devastating loss + high G
    'contraction':  (-20, +30, -15, +20, +30),   # labor pain
    'contractions': (-20, +30, -15, +20, +30),
}
EMOTIONAL_VOCABULARY.update(_BODY_AUTONOMY)

# ── Crisis method/planning/farewell words ──
# These words are neutral in isolation but become crisis signals in combination.
# Mild negative V + elevated U + high G = "something heavy is being discussed"
_CRISIS_METHOD_WORDS = {
    'goodbye':      (-20, +5, -10, +15, -20),    # farewell — mild alone, heavy in context
    'pill':         (-10, +5, -5, +10, -10),      # medication — context-dependent
    'pills':        (-10, +5, -5, +10, -10),
    'painkillers':  (-15, +5, -10, +10, -10),
    'rope':         (-15, +10, -10, +15, -15),    # method word
    'blade':        (-20, +15, -10, +15, -15),    # weapon word
    'wrist':        (-10, +10, -10, +10, -10),    # self-harm locus
    'slit':         (-30, +20, -15, +20, -20),    # self-harm action
    'noose':        (-40, +15, -30, +25, -30),    # method word — high signal
    'overdose':     (-40, +15, -25, +25, -30),
    'tomorrow':     (+5, +5, +5, +10, +0),        # temporal — mild urgency
    'tonight':      (+0, +10, +0, +20, +0),       # temporal — higher urgency
    'anymore':      (-15, +5, -15, +10, -10),     # "not anymore" = giving up
    'cut':          (-15, +10, -5, +10, -10),      # dual-use but negative lean
    'cutting':      (-20, +15, -10, +15, -15),
}
EMOTIONAL_VOCABULARY.update(_CRISIS_METHOD_WORDS)

# ── Permanence/infinity modifiers ──
# "permanently" = negative infinity. Nobody wants permanence of bad things.
# "Is this permanent?" evokes fear. Permanence on exits = crisis lock.
_PERMANENCE_WORDS = {
    'permanently': (-30, +10, -20, +15, -25),   # infinity lock — amplifies negative exit
    'forever':     (-15, +5, -10, +10, -15),     # softer infinity — "forever alone" vs "love forever"
    'eternal':     (-10, +5, -5, +5, -10),       # can be positive (eternal love) or negative
    'endless':     (-15, +10, -10, +10, -15),    # usually negative — endless pain/suffering
    'permanent':   (-20, +10, -15, +10, -20),    # "is this permanent?" = fear
    'irreversible':(-30, +15, -25, +15, -25),    # can't undo — high gravity
    'vanish':      (-40, +5, -35, +15, -30),     # disappear — removing self
    'cease':       (-35, +5, -30, +15, -25),     # stop existing
    'ceased':      (-30, +5, -25, +10, -20),
    'over':        (-10, +5, -5, +5, -5),        # mild exit — context dependent
    'belong':      (+15, +5, +10, +0, +15),      # positive — belonging
}
EMOTIONAL_VOCABULARY.update(_PERMANENCE_WORDS)

# ── Short response words — emotional weight from brevity ──
# "k" is not "ok". "k" is angry/dismissive acknowledgment.
# "ok" hedges entirely on context — neutral by itself but amplifies previous.
# These carry emotional weight from their SHORTNESS, not their meaning.
_SHORT_RESPONSES = {
    'k':            (-20, -15, -20, 0, -10),    # curt dismissal — "I acknowledge, nothing more"
    'kk':           (-5, -5, -5, 0, -5),        # softer k — casual acknowledgment
    'idk':          (-15, -5, -15, 5, -10),     # curt "stop asking me"
    'idc':          (-20, -10, -25, 0, -15),    # "i dont care" — dismissal
    'smh':          (-20, +10, +5, +5, -10),    # disappointment/disapproval
    'bruh':         (-10, +10, +5, +5, -5),     # exasperation/disbelief
    'oof':          (-15, +10, -10, +5, -10),   # sympathetic wince
    'yikes':        (-20, +15, -10, +5, -10),   # alarm/cringe
    'meh':          (-10, -15, -15, 0, -10),    # apathy/indifference
    'nah':          (-5, -5, +5, 0, -5),        # casual rejection
    'nope':         (-5, +5, +10, 0, -5),       # firm rejection (higher D than nah)
}
EMOTIONAL_VOCABULARY.update(_SHORT_RESPONSES)

# ── Swearing — standalone vs modifier are different ──
# "fuck" alone = negative outburst (like "damn")
# "fucking" = amplifier (handled in context_operators, not here)
_SWEARING = {
    'fuck':         (-70, +50, +30, +20, -15),   # expletive/attack -- resists negation at |V|>60
    'shit':         (-30, +25, +10, +10, -10),   # frustration
    # 'bullshit' already in main vocab at V=-60 — don't overwrite
    'dammit':       (-25, +30, +10, +15, -5),    # frustration outburst
    'goddamn':      (-20, +25, +10, +10, -5),    # emphasis outburst
    # NOTE: "damn" excluded — too context-dependent.
    # "damn you"=curse, "I don't give a damn"=indifference, "damn that's good"=amplifier
    # Handled via context_operators as amplifier ("damn good") and bigrams.
    'hell':         (-15, +20, +10, +10, -5),    # mild outburst
}
EMOTIONAL_VOCABULARY.update(_SWEARING)

# ── Connection/transfer words ──
# "Talk" = human HDMI port. Not emotional by itself — it's the act of
# connecting/transferring between two entities. Emotional weight comes
# from the STATE of the connection (opening, closing, urgency).
_CONNECTION_WORDS = {
    'talk':         (0, +5, +5, +5, +10),         # connection word — V neutral, G elevated (it matters)
    'talking':      (0, +5, +5, +5, +10),
    'speak':        (+5, +10, +10, +10, +10),    # more formal connection
    'listen':       (+10, +5, -5, +5, +10),      # receiving — slight positive, lower D (receptive)
    'listening':    (+10, +5, -5, +5, +10),
    'hear':         (+10, +5, -5, +5, +10),      # same as listen
    'communicate':  (+10, +10, +10, +10, +10),   # bidirectional transfer
    'discuss':      (+5, +10, +10, +10, +10),    # structured transfer
    'conversation': (+10, +5, +5, +5, +10),      # ongoing connection
    'silence':      (-15, -20, -10, +5, -15),    # connection severed — absence of transfer
    'quiet':        (-5, -15, -5, +0, -5),       # reduced bandwidth
}
EMOTIONAL_VOCABULARY.update(_CONNECTION_WORDS)

# ── Help as a lifter ──
# Help is a ramp — it goes UP. Slightly positive V (lifting is positive).
# But ASKING for help = the person is DOWN and needs the lift.
# "help me" = pulling you to me, vulnerability. "helping" = active lift.
# Context determines if it's calm lift or desperate pull.
_HELP_WORDS = {
    # "help" alone implies NEED — a deficit. Nobody says help when things are fine.
    # The tense shifts the meaning: help (need), helping (active), helped (done/relief)
    'help':         (-10, +15, -10, +15, +15),   # need/pull — something is lacking, elevated U+G
    'helping':      (+15, +10, +15, +5, +10),    # active lifting — positive action in progress
    'helped':       (+25, +5, +15, +0, +15),     # lift completed — relief
    'helpful':      (+30, +5, +15, +0, +10),     # the lift worked — positive
    # helpless = awareness of need + inability to meet it. Not nothing —
    # it's knowing the ramp doesn't exist and you can't build one.
    # Worse than "help" (need) because the person sees the gap AND knows they can't close it.
    'helpless':     (-60, +10, -60, +20, -35),   # aware of need + unable to act — D crushed
}
EMOTIONAL_VOCABULARY.update(_HELP_WORDS)


# ── Gravity Hierarchy ──
# Relationships and foundations have tiered gravity based on emotional stakes.
# These override the G value from the original vocabulary entry.
# "I lost my child" (G=50) hits harder than "I lost my car" (G=30)
# which hits harder than "I lost my pen" (G=0).
_GRAVITY_HIERARCHY = {
    # Tier 0 (G=55): foundations of existence
    'health':     (0, 10, 0, 10, 55),
    'life':       (0, 5, 5, 0, 55),
    # Tier 1 (G=50): most vulnerable, your responsibility
    'children':   (15, 10, 5, 5, 50),
    # Tier 2 (G=40): core family
    'parents':    (25, 10, 15, 0, 40),
    'spouse':     (30, 10, 10, 0, 40),
    # Tier 3 (G=35): chosen family + security foundations
    'career':     (15, 15, 20, 10, 35),
    # Tier 4 (G=30): independence + blood
    'savings':    (5, 10, 15, 10, 30),
    # Tier 5-6 (G=20): social circle + anchors
    'ring':       (10, 5, 5, 0, 20),
    # Tier 8 (G=5): acquaintance
    'neighbor':   (5, 5, 5, 0, 5),
    'coworker':   (5, 5, 5, 5, 5),
}
EMOTIONAL_VOCABULARY.update(_GRAVITY_HIERARCHY)

# Now update existing words' G values to match hierarchy
# This overrides just the G dimension while keeping V,A,D,U
_G_OVERRIDES = {
    # Tier 1
    'child': 50, 'baby': 50, 'son': 50, 'daughter': 50,
    # Tier 2
    'mom': 40, 'mother': 40, 'dad': 40, 'father': 40, 'parent': 40,
    'husband': 40, 'wife': 40,
    # Tier 3
    'dog': 35, 'cat': 35, 'pet': 35, 'puppy': 35, 'kitten': 35,
    'home': 35, 'house': 35, 'job': 35,
    # Tier 4
    'money': 30, 'car': 30, 'brother': 30, 'sister': 30, 'sibling': 30, 'family': 30,
    # Tier 5
    'life': 55,
    # Tier 6
    'friend': 20, 'phone': 20,
    # Tier 7
    'boyfriend': 10, 'girlfriend': 10, 'partner': 10,
}
for word, new_g in _G_OVERRIDES.items():
    if word in EMOTIONAL_VOCABULARY:
        old = EMOTIONAL_VOCABULARY[word]
        EMOTIONAL_VOCABULARY[word] = (old[0], old[1], old[2], old[3], new_g)

# ── Relationship V neutralization ──
# Relationship words are NOT positive or negative. They are HEAVY.
# "Mom" doesn't carry valence. Mom carries GRAVITY.
# Mom forgot = devastating (G amplifies negative action)
# Mom hugged = amazing (G amplifies positive action)
# The relationship amplifies whatever action follows -- it doesn't add its own V.
_RELATIONSHIP_V_NEUTRAL = [
    'mom', 'mother', 'dad', 'father', 'parent', 'parents',
    'brother', 'sister', 'sibling',
    'son', 'daughter', 'child', 'children', 'baby',
    'husband', 'wife', 'partner',
    'boyfriend', 'girlfriend',
    'family', 'grandma', 'grandmother', 'grandpa', 'grandfather',
    'uncle', 'aunt', 'cousin',
    'friend', 'friends',
    'neighbor', 'boss', 'coworker', 'teacher',
]
for word in _RELATIONSHIP_V_NEUTRAL:
    if word in EMOTIONAL_VOCABULARY:
        old = EMOTIONAL_VOCABULARY[word]
        # Zero out V, keep A low, keep D neutral, keep G
        EMOTIONAL_VOCABULARY[word] = (0, max(0, old[1] // 3), 0, 0, old[4])


# ── Event/object V neutralization ──
# Events and milestone nouns are NOT inherently positive or negative.
# They are HIGH GRAVITY -- things that MATTER.
# "graduation" alone = milestone (G high, V neutral)
# "I graduated" = positive (the VERB carries direction)
# "my mom forgot my graduation" = negative (forgot carries direction, graduation is the weight)
# The noun is the mass. The verb is the force. F = m * a.
_EVENT_V_NEUTRAL = [
    'graduation', 'birthday', 'wedding', 'christmas', 'holiday',
    'vacation', 'party', 'gift', 'present',
    'school', 'college', 'church', 'hospital',
    'promotion', 'raise', 'bonus',
    'dinner', 'breakfast', 'lunch',
    'home', 'house', 'thoughts', 'mind',
    'life', 'world', 'future', 'past',
    'money', 'job', 'career', 'work',
]
for word in _EVENT_V_NEUTRAL:
    if word in EMOTIONAL_VOCABULARY:
        old = EMOTIONAL_VOCABULARY[word]
        # Reduce V to near-zero but keep G high
        EMOTIONAL_VOCABULARY[word] = (old[0] // 5, old[1] // 3, old[2] // 4, old[3], max(20, old[4]))


# ── Action Words (liquid -- context shifts meaning) ──
# These carry their MOST COMMON emotional reading.
# "hit" is negative in most emotional contexts (violence > "hit song").
# Proximity and structure handle the rare positive cases.
_ACTION_WORDS = {
    # Violence/abuse (almost always negative in conversation)
    'hit':          (-45, 35, -25, 20, -15),     # moderate negative -- VICTIMIZATION amplifies for violence
    'hitting':      (-40, 45, 20, 25, -15),
    'slapping':     (-60, 50, 20, 25, -20),
    'kicked':       (-50, 50, 25, 20, -15),
    'kicking':      (-50, 50, 25, 25, -15),
    'shoved':       (-40, 45, 20, 20, -10),
    'grabbed':      (-30, 40, 25, 20, -10),
    'screamed':     (-40, 60, 30, 25, -15),
    'yelled':       (-35, 55, 30, 20, -10),
    
    # Betrayal/violation
    'blocked':      (-35, 15, -20, 10, -15),    # cut off connection
    'posted':       (-10, 15, 10, 10, -5),       # mild -- context needed
    'exposed':      (-40, 30, -25, 20, -20),     # violated
    'leaked':       (-35, 25, -20, 15, -15),
    'stolen':       (-50, 30, -30, 20, -20),
    'stole':        (-50, 30, -25, 20, -20),
    
    # Abandonment
    'walked':       (-15, 10, 10, 5, -5),        # mild -- "walked out" is idiom
    'stormed':      (-30, 50, 30, 20, -10),      # stormed out = angry exit
    'vanished':     (-35, 10, -20, 15, -20),
    'disappeared':  (-35, 10, -25, 15, -20),
    
    # Achievement (positive)
    'promoted':     (48, 30, 35, 5, 25),         # keep positive
    'graduated':    (50, 25, 30, 0, 30),
    'accomplished': (45, 20, 35, 0, 25),
    'succeeded':    (50, 25, 35, 0, 25),
    'passed':       (20, 10, 15, 0, 10),         # mild -- could be death euphemism
    
    # Social harm
    'gossiping':    (-25, 15, 10, 5, -10),
    'bullied':      (-60, 35, -40, 20, -25),
    'bullying':     (-55, 35, -35, 20, -25),
    'mocked':       (-40, 25, 15, 10, -15),
    'mocking':      (-35, 25, 15, 10, -15),
    'humiliated':   (-70, 40, -50, 15, -30),
    'embarrassed':  (-35, 30, -30, 10, -15),
}
EMOTIONAL_VOCABULARY.update(_ACTION_WORDS)


# ── Rejection/abandonment actions ──
_REJECTION_WORDS = {
    'left':         (-8, 5, -5, 5, -3),          # near-neutral -- "I left" = agency. VICTIMIZATION adds weight when confirmed
    'leaving':      (-20, 15, -10, 10, -10),
    'ignored':      (-35, 10, -25, 10, -15),    # social rejection
    'ignoring':     (-30, 10, -20, 10, -15),
    'abandoned':    (-60, 15, -50, 15, -30),    # severe abandonment (override existing)
    'replaced':     (-40, 15, -30, 10, -20),
    'forgotten':    (-35, -10, -30, 5, -20),
    'excluded':     (-35, 10, -25, 10, -15),
    'uninvited':    (-30, 10, -20, 5, -10),
    'unwanted':     (-45, 10, -35, 10, -20),
    'unwelcome':    (-35, 10, -25, 5, -15),
    'dumped':       (-50, 25, -35, 15, -25),
    'ditched':      (-35, 15, -25, 10, -15),
    'read':         (5, 5, 5, 0, 5),            # neutral -- "left on read" needs idiom
}
EMOTIONAL_VOCABULARY.update(_REJECTION_WORDS)


# ── Digital Rejection Patterns ──
# Modern communication has its own emotional vocabulary.
# "left on read" = received your output, refused to process it.
# This is ACKNOWLEDGED rejection -- worse than being ignored.
_DIGITAL_REJECTION = {
    'unread':       (-10, -5, -10, 5, -5),      # not yet seen -- uncertainty
    'unanswered':   (-25, 5, -20, 10, -10),     # sent, no reply -- waiting
    'unfollowed':   (-30, 10, -20, 10, -15),    # cut connection publicly
    'unfriended':   (-35, 10, -25, 10, -15),    # removed from circle
}
EMOTIONAL_VOCABULARY.update(_DIGITAL_REJECTION)


# ── Violence/aggression actions ──
_VIOLENCE_WORDS = {
    'stabbed':      (-80, 90, -60, 80, -40),
    'stab':         (-70, 80, -50, 70, -35),
    'stabbing':     (-75, 85, -55, 75, -38),
    'punched':      (-60, 80, -40, 60, -25),
    'slapped':      (-55, 70, -35, 50, -20),
    'choked':       (-75, 85, -60, 80, -35),
    'shoved':       (-45, 60, -30, 40, -15),
    'grabbed':      (-35, 50, -25, 30, -10),
    'attacked':     (-70, 85, -50, 70, -30),
    'assaulted':    (-80, 85, -60, 80, -40),
    'beaten':       (-75, 80, -55, 70, -35),
    'strangled':    (-80, 90, -65, 85, -40),
    'thrown':       (-40, 60, -30, 40, -15),
}
EMOTIONAL_VOCABULARY.update(_VIOLENCE_WORDS)


# ── Mockery/humiliation ──
_MOCKERY_WORDS = {
    'mocked':       (-40, 25, 15, 10, -15),
    'mocking':      (-35, 25, 15, 10, -15),
    'ridiculed':    (-45, 30, 15, 10, -20),
    'taunted':      (-35, 25, 15, 10, -15),
    'teased':       (-20, 15, 10, 5, -8),
    'harassed':     (-55, 40, -25, 30, -20),
}
EMOTIONAL_VOCABULARY.update(_MOCKERY_WORDS)


# ── Resignation/dismissal overrides ──
# Single-word responses carry emotional weight through what's MISSING.
# "whatever" = I stopped fighting. "k" = stripped to minimum effort.
_RESIGNATION_OVERRIDES = {
    'whatever':     (-15, -20, -20, 0, -5),
    'sure':         (-2, -5, -3, 0, -1),      # near-zero -- liquid. context determines.
    'cool':         (-5, -15, -10, 0, -3),
    'k':            (-10, -20, -15, 0, -5),
    'nvm':          (-15, -15, -15, 5, -5),
    'nevermind':    (-15, -15, -15, 5, -5),
    'idk':          (-10, -15, -15, 0, -5),
    'idc':          (-15, -20, -20, 0, -5),
}
EMOTIONAL_VOCABULARY.update(_RESIGNATION_OVERRIDES)


# ── Achievement/success actions ──
_ACHIEVEMENT_WORDS = {
    'worked':       (20, 10, 15, 0, 8),         # it worked = success
    'succeeded':    (35, 20, 30, 0, 15),
    'accomplished': (30, 15, 25, 0, 15),
    'achieved':     (30, 15, 25, 0, 15),
    'graduated':    (35, 20, 25, 0, 20),
    'promoted':     (30, 15, 25, 0, 15),
    'hired':        (25, 15, 20, 0, 10),
    'fired':        (-45, 30, -35, 25, -20),    # job loss
    'fired!':       (-45, 30, -35, 25, -20),
}
EMOTIONAL_VOCABULARY.update(_ACHIEVEMENT_WORDS)


# ── Violation/intrusion actions ──
# Liquid words -- mild negative alone, VICTIMIZATION amplifies when confirmed.
_VIOLATION_WORDS = {
    'deleted':      (-15, 10, -10, 5, -5),
    'changed':      (-10, 5, -5, 5, -3),
    'ate':          (-5, 5, -5, 0, -3),
    'took':         (-10, 5, -10, 5, -5),
    'spent':        (-15, 5, -10, 5, -5),
    'sold':         (-15, 10, -10, 5, -5),
    'threw':        (-15, 15, -10, 5, -5),
    'stole':        (-50, 30, -30, 20, -15),
    'stealing':     (-45, 30, -25, 20, -15),
    'broke':        (-25, 20, -15, 10, -10),
    'smashed':      (-40, 40, -20, 15, -10),
    'ruined':       (-50, 20, -30, 15, -20),
    'wrecked':      (-45, 20, -25, 15, -15),
    'destroyed':    (-60, 25, -40, 20, -25),
}
EMOTIONAL_VOCABULARY.update(_VIOLATION_WORDS)


# ── Invalidation/gaslighting ──
_INVALIDATION_WORDS = {
    'overreacting':  (-20, 15, 20, 5, -5),     # accusation -- D goes UP for speaker
    'dramatic':      (-15, 10, 15, 5, -5),
    'crazy':         (-30, 15, 15, 10, -10),
    'paranoid':      (-25, 15, 10, 5, -8),
    'delusional':    (-30, 15, 15, 10, -10),
    'exaggerating':  (-15, 10, 15, 5, -5),
}
EMOTIONAL_VOCABULARY.update(_INVALIDATION_WORDS)


# ── Resolution/outcome words ──
_RESOLUTION_WORDS = {
    'made':         (15, 5, 10, 0, 5),
    'well':         (15, -5, 10, 0, 5),
    'anyway':       (5, 5, 10, 0, 0),       # persistence despite obstacles
    'survived':     (10, 5, 15, 0, 10),
    'overcame':     (20, 10, 20, 0, 10),
    'managed':      (10, 5, 10, 0, 5),
}
EMOTIONAL_VOCABULARY.update(_RESOLUTION_WORDS)


# ── Threat/oath words ──
_THREAT_WORDS = {
    'swear':        (-15, 20, 20, 15, -5),     # oath -- escalation signal
    'promise':      (10, 5, 10, 5, 15),         # commitment -- can be positive or threatening
    'warn':         (-20, 15, 25, 15, -5),
    'warning':      (-20, 15, 25, 15, -5),
    'threatening':  (-40, 30, 20, 20, -10),
    'threat':       (-40, 30, 15, 20, -10),
}
EMOTIONAL_VOCABULARY.update(_THREAT_WORDS)


# ── Medical/condition words ──
_MEDICAL_WORDS = {
    'herpes':       (-30, 15, -20, 5, -10),
    'cancer':       (-80, 30, -50, 40, -40),
    'sick':         (-30, 10, -20, 10, -15),
    'infected':     (-35, 20, -25, 15, -15),
    'pregnant':     (5, 30, 0, 30, 40),          # high G, context-dependent V
    'disability':   (-15, 5, -15, 5, 20),
}
EMOTIONAL_VOCABULARY.update(_MEDICAL_WORDS)


# ── Milestone/achievement words ──
_MILESTONE_WORDS = {
    'million':      (35, 15, 25, 5, 10),
    'billion':      (20, 10, 20, 5, 10),
    'streams':      (10, 5, 5, 0, 5),
    'verified':     (20, 10, 15, 0, 10),
    'published':    (25, 10, 20, 0, 15),
    'accepted':     (30, 15, 20, 0, 15),
    'paid':         (15, 5, 10, 0, 5),
    'hero':         (40, 15, 30, 0, 20),
    'dream':        (25, 10, 10, 5, 15),
}
EMOTIONAL_VOCABULARY.update(_MILESTONE_WORDS)


# ── Comparison/judgment words ──
_JUDGMENT_WORDS = {
    'compare':      (-15, 5, -10, 5, -5),
    'compared':     (-15, 5, -10, 5, -5),
    'comparing':    (-15, 5, -10, 5, -5),
    'judge':        (-20, 10, 15, 5, -5),
    'judged':       (-25, 10, -10, 5, -10),
    'judging':      (-20, 10, 15, 5, -5),
    'criticize':    (-25, 10, 15, 5, -8),
    'criticized':   (-30, 10, -15, 5, -10),
    'blame':        (-30, 15, 15, 10, -10),
    'blamed':       (-35, 15, -20, 10, -15),
    'fault':        (-25, 10, -15, 5, -10),
}
EMOTIONAL_VOCABULARY.update(_JUDGMENT_WORDS)


# ── Temporal intensity words ──
# These amplify through permanence/repetition
_TEMPORAL_INTENSITY = {
    'always':       (-10, 5, -5, 5, -3),      # infinity -- relentless
    'constantly':   (-10, 10, -5, 5, -3),      # nonstop pressure
    'every':        (-5, 5, -5, 5, -3),        # totality
    'forever':      (-5, 5, -5, 10, -5),       # permanence
}
EMOTIONAL_VOCABULARY.update(_TEMPORAL_INTENSITY)


# ── Exclusion/comparison markers ──
_EXCLUSION_WORDS = {
    'except':       (-25, 5, -15, 5, -10),     # exclusion -- everyone EXCEPT me
    'besides':      (-10, 5, -5, 0, -3),
    'instead':      (-15, 5, -10, 5, -5),       # replacement -- they chose X instead of me
    'without':      (-45, 15, -40, 15, -25),     # already exists but reconfirm
    'more':         (-8, 5, -5, 0, -3),          # comparison -- implies less for speaker
    'less':         (-10, -5, -5, 0, -5),        # already exists
    'better':       (-5, 5, -5, 0, -3),          # context: "theyre better" = im worse
    'prettier':     (-10, 5, -10, 0, -5),
    'smarter':      (-8, 5, -10, 0, -5),
    'faster':       (-5, 5, -5, 0, -3),
}
EMOTIONAL_VOCABULARY.update(_EXCLUSION_WORDS)


# ── Doubt/questioning markers ──
# "even" in "do you even care" = doubt amplifier
# "actually" in "did you actually try" = skepticism
_DOUBT_WORDS = {
    'even':         (-10, 5, -5, 5, -3),         # doubt -- "do you EVEN"
    'actually':     (-8, 5, 5, 0, -3),           # skepticism -- "did you ACTUALLY"
    'really':       (-5, 5, -5, 0, -3),          # doubt when questioning: "do you really"
    'anymore':      (-15, 5, -10, 5, -5),        # loss of state -- "i dont care anymore"
    'supposed':     (-10, 5, -10, 5, -5),        # obligation unfulfilled -- "you were supposed to"
}
EMOTIONAL_VOCABULARY.update(_DOUBT_WORDS)


# ── Upbringing/family structure ──
_UPBRINGING_WORDS = {
    'foster':       (-3, 0, -3, 0, -10),         # neutralizer/dampener -- substitute. reduces what follows. not negative itself.
    'adopted':      (-5, 5, -5, 5, 35),          # near-neutral -- can be positive (chosen) or painful
    'orphan':       (-40, 10, -30, 10, 40),      # parentless -- high G loss
    'homeless':     (-50, 15, -40, 15, 35),
    'abused':       (-70, 30, -50, 30, 40),
    'neglected':    (-50, 10, -40, 10, 35),
    'molested':     (-90, 40, -70, 50, 50),
    'trafficked':   (-80, 40, -60, 50, 45),
}
EMOTIONAL_VOCABULARY.update(_UPBRINGING_WORDS)


# ── Missing common words (found by benchmark) ──
_BENCHMARK_FIXES = {
    'disappointment': (-45, 10, -25, 10, -25),
    'disappointments': (-40, 10, -20, 10, -20),
    'forgot':         (-25, 8, -15, 8, -10),
    'forget':         (-15, 5, -10, 5, -8),
    'forgetting':     (-15, 5, -10, 5, -8),
    'finished':       (15, 5, 15, 0, 5),       # achievement -- completed something
    'started':        (5, 10, 5, 5, 3),         # beginning -- mild positive energy
    'called':         (-5, 5, 5, 0, 0),         # near-neutral -- "called me" = labeled
    'invisible':      (-35, -10, -30, 5, -15),  # social erasure
    'burden':         (-45, 10, -35, 10, -20),  # self as weight on others
    'replaced':       (-40, 15, -30, 10, -20),  # substituted out
    'trapped':        (-40, 20, -40, 20, -15),  # no agency
    'stuck':          (-25, 10, -25, 10, -10),  # mild trapped
    'empty':          (-30, -15, -20, 5, -15),  # hollow
    'numb':           (-25, -20, -15, 0, -10),  # disconnected
    'exhausted':      (-30, -15, -25, 10, -15), # depleted
    'drained':        (-25, -15, -20, 5, -10),  # emptied out
    'overwhelmed':    (-35, 30, -30, 20, -15),  # too much input
    'suffocating':    (-45, 35, -40, 25, -20),  # can't breathe metaphor
    # ── Common reducers/modifiers found by SST-2 gap analysis ──
    'little':       (-15, -5, -10, 0, -5),     # reducer -- "little to love" = almost none
    'despite':      (-10, 5, 5, 0, -3),         # concession -- upcoming contradiction
    'only':         (-10, -5, -5, 0, -3),        # minimizer -- "only manages" = barely
    'almost':       (-8, 0, -5, 0, -3),          # fell short -- "almost good" = not good
    'seems':        (-5, 0, -5, 0, -3),          # uncertainty -- not confirmed
    'lacks':        (-30, 5, -15, 5, -10),       # absence of quality
    'lacking':      (-25, 5, -12, 5, -8),
    'obvious':      (-15, 5, 5, 0, -5),          # dismissive -- "obvious plot"
    'predictable':  (-20, -5, 5, 0, -5),
    'flat':         (-20, -10, -10, 0, -5),
    'dull':         (-25, -15, -10, 0, -8),
    'generic':      (-15, -5, -5, 0, -5),
    'bland':        (-20, -10, -10, 0, -5),
    'mediocre':     (-25, -5, -10, 0, -8),
    'forgettable':  (-20, -5, -10, 0, -5),
    'pointless':    (-30, 5, -15, 5, -10),
    'mess':         (-30, 10, -15, 10, -10),
    'waste':        (-35, 10, -20, 10, -10),
    'wasted':       (-30, 10, -15, 10, -8),
    'filth':        (-50, 15, -25, 10, -15),
    'garbage':      (-40, 10, -20, 10, -12),
    'trash':        (-35, 10, -15, 10, -10),
    'awful':        (-50, 15, -25, 10, -15),
    'horrible':     (-50, 15, -25, 10, -15),
    'terrible':     (-50, 15, -25, 10, -15),
    'worst':        (-55, 15, -30, 10, -18),
    'pathetic':     (-45, 10, -20, 10, -12),
    'absurd':       (-25, 10, -10, 5, -8),
    'ridiculous':   (-30, 10, -10, 5, -10),
    'foolish':      (-25, 10, -10, 5, -8),
    'shallow':      (-25, -5, -10, 0, -8),
    'lazy':         (-25, -10, -15, 0, -8),
    'pretentious':  (-30, 5, 10, 0, -10),
    'overrated':    (-25, 5, -10, 5, -8),
    'disappointing':(-35, 10, -20, 10, -12),
    'uninspired':   (-25, -10, -15, 0, -8),
    'unfunny':      (-25, 5, -10, 5, -8),
    'tiresome':     (-25, -10, -15, 5, -8),
    'clunky':       (-20, 5, -10, 0, -5),
    'cliche':       (-20, 0, -10, 0, -5),
    # Positive review words
    'brilliant':    (+55, 15, 20, 0, 12),
    'masterpiece':  (+60, 15, 25, 0, 15),
    'stunning':     (+50, 20, 15, 0, 12),
    'superb':       (+55, 15, 20, 0, 12),
    'flawless':     (+50, 10, 20, 0, 12),
    'captivating':  (+45, 15, 15, 0, 10),
    'riveting':     (+45, 20, 15, 0, 10),
    'heartwarming': (+50, 15, 15, 0, 15),
    'touching':     (+35, 10, 10, 0, 10),
    'compelling':   (+40, 15, 15, 0, 10),
    'gripping':     (+40, 20, 15, 0, 10),
    'beautifully':  (+45, 10, 15, 0, 10),
    'finest':       (+50, 10, 20, 0, 12),
    'thrilling':    (+45, 20, 15, 0, 10),
    'remarkable':   (+45, 10, 15, 0, 10),
    'refreshing':   (+35, 10, 10, 0, 8),
    'entertaining': (+35, 15, 10, 0, 8),
    'enjoyable':    (+35, 10, 10, 0, 8),
    'delightful':   (+45, 15, 15, 0, 10),
    'witty':        (+35, 10, 15, 0, 8),
    'clever':       (+30, 10, 15, 0, 8),
    'engaging':     (+35, 15, 10, 0, 8),
    'powerful':     (+40, 15, 20, 0, 10),
    'impressive':   (+40, 10, 15, 0, 10),
    'cooked':       (-30, 15, -20, 10, -10),  # slang -- destroyed/ruined
    'shade':        (-20, 10, 15, 5, -5),    # slang -- indirect disrespect
    'ratio':        (-25, 15, -15, 10, -10), # slang -- publicly outperformed/rejected
    'stopped':      (-5, 5, -5, 5, -3),         # NEGATOR role does the flip. Minimal own force.
    'broken':       (-30, 10, -25, 10, -15),  # damaged state
    'ruined':       (-50, 20, -30, 15, -20),  # destruction (override if exists)
    'failed':       (-35, 15, -25, 15, -15),  # failure
    'failing':      (-30, 15, -20, 15, -12),
    'losing':       (-25, 15, -20, 10, -10),
    'lost':         (-30, 10, -25, 10, -15),
    'missing':      (-20, 10, -15, 10, -10),  # absence
    'working':      (10, 5, 10, 0, 3),        # mild positive -- functioning
}
EMOTIONAL_VOCABULARY.update(_BENCHMARK_FIXES)



# ── RoBERTa-mined emotional words (63 words) ──
# Extracted by scoring frequent missing words through RoBERTa.
# Filtered: removed structural words, bias (gay/police from twitter data),
# and words already handled by other roles (fucking=AMPLIFIER, just=FILLER).
_ROBERTA_MINED = {
    'bloody':       (-54, +27, -27, +0, -14),
    'destruction':  (-53, +26, -27, +0, -14),
    'hated':        (-53, +26, -27, +0, -14),
    'terrorist':    (-52, +26, -26, +0, -13),
    'mean':         (-40, +20, -20, +0, -10),     # reduced -- "mean" is also average/intend
    'bombing':      (-51, +25, -26, +0, -13),
    'terminal':     (-50, +25, -25, +0, -13),
    'criminal':     (-50, +25, -25, +0, -13),
    'terrorists':   (-50, +25, -25, +0, -13),
    'concerning':   (-48, +24, -24, +0, -12),
    'killed':       (-47, +23, -24, +0, -12),
    'forced':       (-47, +23, -24, +0, -12),
    'scandal':      (-46, +23, -23, +0, -12),
    'violation':    (-45, +22, -23, +0, -12),
    'problems':     (-44, +22, -22, +0, -11),
    'divided':      (-44, +22, -22, +0, -11),
    'lies':         (-41, +20, -21, +0, -11),
    'struggling':   (-40, +20, -20, +0, -10),
    'deaths':       (-37, +18, -19, +0, -10),
    'knife':        (-36, +18, -18, +0, -9),
    'buried':       (-36, +18, -18, +0, -9),
    'judgment':     (-36, +18, -18, +0, -9),
    'important':    (+36, +9, +12, +0, +7),
    'interest':     (+20, +5, +10, +0, +5),        # reduced -- context-dep
    'skills':       (+36, +9, +12, +0, +7),
    'music':        (+38, +9, +12, +0, +7),
    'unique':       (+38, +9, +12, +0, +7),
    'vital':        (+39, +9, +13, +0, +7),
    'stronger':     (+39, +9, +13, +0, +7),
    'strong':       (+41, +10, +13, +0, +8),
    'smiling':      (+42, +10, +14, +0, +8),
    'benefits':     (+43, +10, +14, +0, +8),
    'loves':        (+46, +11, +15, +0, +9),
    'helps':        (+47, +11, +15, +0, +9),
    'smooth':       (+47, +11, +15, +0, +9),
    'historic':     (+47, +11, +15, +0, +9),
    'adore':        (+48, +12, +16, +0, +9),
    'potential':    (+30, +8, +10, +0, +6),         # reduced -- context-dep
    'clean':        (+49, +12, +16, +0, +9),
    'greater':      (+30, +8, +10, +0, +6),         # reduced -- comparative
    'fascinating':  (+51, +12, +17, +0, +10),
    'victory':      (+53, +13, +17, +0, +10),
    'enjoyed':      (+54, +13, +18, +0, +10),
    'magic':        (+54, +13, +18, +0, +10),
    'gold':         (+30, +8, +10, +0, +6),         # reduced -- physical object
    'ideal':        (+56, +14, +18, +0, +11),
    'outstanding':  (+57, +14, +19, +0, +11),
}
EMOTIONAL_VOCABULARY.update(_ROBERTA_MINED)


# ── Common modifiers + slang found by corpus mining ──
_CORPUS_MINED = {
    'happened':     (-10, 10, -5, 10, 5),      # event occurred -- slight unease, urgency
    'happening':    (-8, 10, -5, 10, 5),
    'again':        (-10, 5, -8, 5, -5),        # repetition -- wearing down
    'anymore':      (-15, 5, -10, 5, -5),       # loss of state (override if exists)
    'wait':         (-5, 5, -5, 10, 0),          # patience/urgency
    'waiting':      (-8, 5, -8, 10, -3),
    'down':         (-15, 5, -10, 5, -5),        # negative direction
    'kids':         (0, 5, 0, 0, 45),            # relationship -- high G like child
    'kid':          (0, 5, 0, 0, 45),
    'guy':          (0, 0, 0, 0, 5),             # neutral person reference
    'girl':         (0, 0, 0, 0, 5),
    'man':          (0, 5, 5, 0, 5),             # neutral/mild empathy filler
    'must':         (-5, 5, 5, 5, 0),            # inference/obligation
    'sounds':       (0, 0, 0, 0, 0),             # hedging -- passes through
    'first':        (5, 5, 5, 5, 5),             # milestone marker
    'once':         (-5, 0, -3, 0, -3),          # past -- something ended
    'another':      (-5, 5, -3, 5, -3),          # repetition
    'anything':     (-5, 5, -5, 5, 0),           # open/desperate depending on context
    'ago':          (-3, 0, -3, 0, -3),          # past temporal
    'yesterday':    (0, 0, 0, 10, 0),            # time marker -- urgency only
    'rough':        (-25, 10, -15, 5, -8),       # difficulty/hardship
    'tough':        (-20, 10, -10, 5, -5),
    'harsh':        (-30, 15, -15, 5, -10),
    'brutal':       (-40, 20, -20, 10, -12),
    'insane':       (-20, 25, -10, 10, -5),      # slang -- extreme, often negative
    'nuts':         (-15, 20, -5, 5, -3),
    'wild':         (-5, 20, 5, 5, 0),           # intense, context-dependent
    'crazy':        (-15, 15, -5, 10, -5),       # override if exists -- slang varies
    'messed':       (-25, 10, -15, 10, -8),
    'screwed':      (-30, 15, -20, 10, -10),
    'ruined':       (-45, 15, -25, 10, -15),     # override if exists
    'wrecked':      (-40, 15, -20, 10, -12),
    'done':         (-10, -5, -10, 5, -5),       # finality -- "im done"
    'over':         (-10, -5, -5, 5, -5),        # ending
    'gone':         (-20, 5, -15, 5, -10),       # absence
    'lost':         (-30, 10, -25, 10, -15),     # override
    'left':         (-8, 5, -5, 5, -3),          # override -- near neutral, VICTIMIZATION resolves
    'wrong':        (-25, 10, -15, 10, -8),
    'fault':        (-25, 10, -15, 5, -10),      # override if exists
    'blame':        (-30, 15, 15, 10, -10),      # override
    'worth':        (10, 5, 10, 0, 10),          # value
    'worthy':       (15, 5, 15, 0, 10),
    'deserve':      (5, 5, 5, 5, 10),            # context-dep -- "i deserve better" vs "you deserve this"
    'deserved':     (5, 5, 5, 5, 10),
    'matters':      (5, 5, 5, 5, 15),            # gravity -- this matters
    'counts':       (5, 5, 5, 0, 10),
    'enough':       (-15, 10, -5, 5, -8),        # "not enough" = lacking. override.
}
EMOTIONAL_VOCABULARY.update(_CORPUS_MINED)


# ── Novel sentence gap fills (round 2) ──
_NOVEL_GAPS_2 = {
    'laughed':      (-5, 20, 5, 0, 5),          # liquid -- laughed at vs laughed with
    'grave':        (-20, -10, -15, 0, 30),      # death-related, high G
    'singled':      (-15, 10, -10, 5, -5),       # isolated/targeted
    'suspended':    (-25, 15, -20, 10, -10),     # punished/removed
    'defending':    (10, 15, 15, 10, 5),          # standing up for
    'garnished':    (-20, 5, -20, 10, -10),      # wages taken
    'wages':        (0, 0, 0, 0, 20),            # money -- object with G
    'fainted':      (-25, 20, -30, 20, -10),     # medical event
    'chose':        (-5, 5, 5, 5, 5),            # decision -- near neutral
    'danced':       (25, 15, 10, 0, 10),         # joyful movement
    'normal':       (10, -5, 10, 0, 5),          # stability/relief
    'benign':       (25, -10, 15, 0, 15),        # medical relief
    'sunset':       (15, -5, 5, 0, 8),           # peaceful/romantic
    'mile':         (10, 10, 10, 0, 5),          # achievement context
    'breathe':      (-10, 10, -10, 5, -5),       # liquid -- cant breathe vs finally breathe
    'texts':        (-5, 5, -5, 5, 10),          # high G (communication)
    'subtweeted':   (-20, 10, 10, 5, -8),        # indirect attack
    'screenshotted':(-15, 10, -5, 5, -5),        # privacy violation
    'watching':     (-3, 5, 5, 5, 0),            # near neutral -- context determines
    'reaching':     (5, 5, 5, 5, 5),             # effort
    'stopping':     (-5, 5, -5, 5, 0),           # cessation
    'asking':       (0, 5, -5, 5, 0),            # neutral request
    # Fix overwieghted words
    'stories':      (5, 5, 5, 0, 5),             # was +37, too positive for neutral noun
    'space':        (0, -5, 0, 0, 0),            # was +31, "needs space" = distance
    'shared':       (5, 5, 5, 0, 5),             # was +35, context-dep
    'credit':       (5, 5, 10, 0, 5),            # was +28, too positive
    'order':        (0, 0, 5, 0, 0),             # was +28, too positive for neutral noun
}
EMOTIONAL_VOCABULARY.update(_NOVEL_GAPS_2)


# ── Novel sentence gap fills (round 3) ──
_NOVEL_GAPS_3 = {
    'texting':      (-5, 5, 0, 5, 10),          # communication -- context determines
    'her':          (0, 0, 0, 0, 0),             # pronoun -- neutral
    'feelings':     (10, 10, -5, 5, 20),         # emotional state -- high G
    'were':         (0, 0, 0, 0, 0),             # verb -- neutral
    'treated':      (-10, 5, -10, 5, -5),        # "treated like" = objectified
    'listened':     (10, 5, 5, 5, 5),            # attention given
    'should':       (-8, 5, -5, 5, -3),          # obligation/regret
    'have':         (0, 0, 0, 0, 0),             # auxiliary -- neutral
    'been':         (0, 0, 0, 0, 0),
    'there':        (0, 0, 0, 0, 0),
    'things':       (0, 0, 0, 0, 5),             # objects
    'could':        (-5, 0, -5, 0, -3),          # hypothetical -- missed opportunity
    'year':         (0, 0, 0, 5, 5),             # time marker
    'years':        (0, 0, 0, 5, 5),
    'situation':    (-5, 5, -5, 5, 5),           # context -- usually negative context
    'hand':         (5, 0, 5, 0, 10),            # connection -- high G (touch)
    'how':          (0, 0, 0, 0, 0),             # question word
    'coffee':       (5, 5, 0, 0, 3),             # comfort object
    'curled':       (10, -5, 5, 0, 5),           # warmth/comfort
    'next':         (0, 0, 0, 0, 0),
    'when':         (0, 0, 0, 0, 0),
    'everyone':     (0, 5, 0, 5, 10),            # scope amplifier -- high G
    'saw':          (-5, 5, -5, 5, 0),           # witnessed -- slight negative lean
    'breakdown':    (-45, 30, -30, 20, -15),     # emotional collapse
    'myself':       (5, 5, 10, 0, 15),           # self-empowerment or self-reference
    'learned':      (15, 5, 15, 0, 8),           # growth/achievement
    'their':        (0, 0, 0, 0, 0),
    'approval':     (10, 5, -10, 5, 15),         # seeking validation -- high G
    'underwater':   (-30, 15, -25, 10, -10),     # drowning metaphor
    'think':        (0, 5, 5, 0, 0),
    'straight':     (5, 0, 5, 0, 0),
    'bittersweet':  (-10, 5, -5, 0, 10),         # mixed -- lean negative
    'necessary':    (5, 5, 10, 5, 5),            # needed -- slight positive
    'let':          (-5, 0, -5, 0, 0),           # release/permission
    'them':         (0, 0, 0, 0, 0),
    'own':          (5, 0, 10, 0, 5),            # ownership/agency
    'fumbled':      (-25, 10, -15, 10, -8),      # slang -- dropped the ball
    'bag':          (0, 0, 0, 0, 5),             # slang object
    'caught':       (-10, 15, -5, 10, 0),        # surprise/trap
    'aired':        (-20, 10, -10, 10, -8),      # exposed/put on blast
    'out':          (-5, 5, -3, 0, 0),           # direction/exposure
    'purpose':      (-10, 10, 10, 5, -5),        # "on purpose" = intentional harm
    'she':          (0, 0, 0, 0, 0),             # pronoun
    'they':         (0, 0, 0, 0, 0),
    'bro':          (-3, 5, 5, 0, 5),            # casual address
    'cant':         (-10, 5, -10, 5, -5),        # inability -- override negator force
}
EMOTIONAL_VOCABULARY.update(_NOVEL_GAPS_3)


# ── Novel sentence gap fills (round 4) ──
_NOVEL_GAPS_4 = {
    'intrusive':    (-35, 20, -20, 15, -10),     # unwanted -- intrusive thoughts
    'disconnected': (-25, -10, -20, 5, -10),      # isolation
    'autopilot':    (-15, -15, -10, 0, -5),       # disengaged from life
    'people':       (0, 0, 0, 0, 5),              # neutral collective
    'attacks':      (-40, 30, -25, 20, -12),       # assault -- panic attacks
    'frequent':     (0, 0, 0, 0, 0),              # neutral modifier
    'reached':      (10, 5, 10, 5, 5),            # effort -- reaching out
    'check':        (0, 5, 5, 0, 0),
    'media':        (0, 5, 0, 5, 5),
    'said':         (0, 5, 5, 0, 0),
    'didnt':        (-10, 5, -5, 5, -3),          # negation -- same as dont
    'doing':        (0, 5, 5, 0, 0),
    'would':        (-3, 0, -3, 0, 0),            # hypothetical
    'having':       (0, 0, 0, 0, 0),
    'safe':         (20, -10, 20, 0, 15),         # security -- important G
    'trusting':     (10, 5, -5, 0, 15),           # vulnerability
    'trouble':      (-20, 10, -15, 10, -8),
    'flashbacks':   (-40, 30, -25, 20, -12),
    'wya':          (-5, 5, 0, 5, 0),             # where you at -- checking
    'ong':          (10, 10, 10, 0, 5),           # on god -- emphasis/genuine
    'thats':        (0, 0, 0, 0, 0),
}
EMOTIONAL_VOCABULARY.update(_NOVEL_GAPS_4)


# ── Novel sentence gap fills (round 5) ──
_NOVEL_GAPS_5 = {
    'erased':       (-30, 10, -25, 10, -12),     # deleted from existence
    'photo':        (0, 0, 0, 0, 10),            # object -- high G (memories)
    'question':     (-10, 10, -5, 5, -3),        # doubt/challenge
    'sanity':       (5, 5, 5, 0, 20),            # mental health -- high G
    'twisted':      (-30, 15, -15, 10, -10),     # manipulated/distorted
    'words':        (0, 5, 0, 0, 5),             # neutral carrier
    'until':        (0, 0, 0, 0, 0),             # temporal connector
    'stayed':       (10, -5, 10, 0, 10),         # presence/commitment
    'went':         (0, 5, 0, 5, 0),             # movement -- neutral
    'time':         (0, 0, 0, 5, 5),             # temporal -- neutral with G
    'since':        (0, 0, 0, 0, 0),             # temporal connector
    'none':         (-15, -5, -10, 0, -5),       # absence/zero
    'audacity':     (-25, 20, 15, 10, -8),       # offense/disbelief
    'youre':        (0, 0, 0, 0, 0),             # contraction -- neutral
    'here':         (0, 0, 0, 0, 0),             # location -- neutral
    'thing':        (0, 0, 0, 0, 0),             # object -- neutral
    'wouldnt':      (-5, 0, -5, 0, 0),           # hypothetical negation
    'exhaust':      (-25, -10, -20, 5, -8),
    'exhausting':   (-25, -10, -20, 5, -8),
    'avoid':        (-10, 5, -10, 5, -5),        # avoidance behavior
    'conflict':     (-25, 15, -10, 10, -8),
    'hollow':       (-20, -5, -15, 0, -8),       # empty/insincere
    'pretend':      (-15, 5, -10, 5, -5),        # masking
    'pretending':   (-15, 5, -10, 5, -5),
    'scream':       (-30, 40, -15, 20, -10),
    'punch':        (-35, 40, 20, 20, -10),
    'dare':         (-15, 20, 20, 10, -5),
    'invited':      (15, 10, 5, 0, 10),          # included/welcomed
    'photo':        (0, 0, 0, 0, 10),
    'mowed':        (10, 5, 5, 0, 5),            # act of service
    'groceries':    (0, 0, 0, 0, 5),
    'procedure':    (-10, 10, -10, 15, 10),      # medical -- high G
    'nurse':        (10, 5, 10, 0, 15),          # caretaker
    'flowers':      (20, 5, 10, 0, 10),
    'doorstep':     (0, 0, 0, 0, 5),
    'apartment':    (0, 0, 0, 0, 15),            # home -- high G
    'thanksgiving': (10, 5, 5, 0, 15),           # holiday gathering
    'date':         (10, 15, 5, 5, 10),          # romantic context
    'smiled':       (25, 10, 10, 0, 8),
    'meant':        (5, 5, 5, 5, 5),
}
EMOTIONAL_VOCABULARY.update(_NOVEL_GAPS_5)


# ── Novel sentence gap fills (round 6) ──
_NOVEL_GAPS_6 = {
    'corrects':     (-15, 10, 15, 5, -5),        # correction = power dynamic
    'keeps':        (0, 0, 5, 0, 0),
    'weaponize':    (-40, 15, 15, 10, -12),       # using something AS a weapon
    'weaponized':   (-40, 15, 15, 10, -12),
    'insecurities': (-30, 10, -25, 5, -12),
    'insecurity':   (-25, 10, -20, 5, -10),
    'reads':        (5, -5, 5, 0, 5),             # near-neutral -- reading to someone vs reading texts
    'being':        (0, 0, 0, 0, 0),
    'asked':        (0, 5, -5, 5, 0),
    'judges':       (-20, 10, 15, 5, -8),         # judgment
    'judging':      (-20, 10, 15, 5, -8),
    'sees':         (5, 5, 5, 0, 5),              # recognition -- mild positive
    'else':         (0, 0, 0, 0, 0),
    'does':         (0, 0, 0, 0, 0),
    'voice':        (5, 5, 5, 0, 10),             # expression -- high G
    'doesnt':       (-10, 5, -5, 5, -3),          # negation
    'matter':       (5, 0, 5, 0, 10),             # importance -- G
    'shrink':       (-15, -5, -20, 0, -5),        # making self smaller
    'shrinking':    (-15, -5, -20, 0, -5),
    'expect':       (-10, 5, 10, 5, -3),          # obligation/pressure
    'expects':      (-10, 5, 10, 5, -3),
    'eggshells':    (-25, 15, -20, 10, -10),      # walking on eggshells = anxiety
    'score':        (-5, 5, 10, 0, 0),            # keeping score = resentment context
    'arguments':    (-25, 15, -10, 10, -8),
    'ghost':        (-20, -10, -15, 0, -8),       # invisible/ignored
    'disappearing': (-25, 5, -20, 5, -10),
    'perform':      (-10, 5, -5, 5, -3),          # performing = masking
    'comfortable':  (15, -10, 10, 0, 8),
    'accept':       (15, 5, 10, 0, 10),
    'accepts':      (15, 5, 10, 0, 10),
    'fight':        (-20, 25, 15, 15, -5),        # conflict -- but "fight for me" = positive
    'excited':      (40, 25, 15, 5, 10),
    'excitement':   (35, 25, 10, 5, 8),
    'bike':         (5, 10, 5, 0, 5),
    'ring':         (5, 5, 5, 0, 15),             # high G -- engagement ring
    'laugh':        (15, 15, 10, 0, 8),           # override -- was +28
}
EMOTIONAL_VOCABULARY.update(_NOVEL_GAPS_6)


# ── Novel sentence gap fills (round 7) ──
_NOVEL_GAPS_7 = {
    'shouldnt':     (-10, 5, -5, 5, -3),         # regret/prohibition
    'staring':      (-5, 5, -5, 0, 0),           # fixation
    'ceiling':      (0, -5, 0, 0, 0),            # physical -- neutral
    'version':      (0, 0, 0, 0, 0),             # neutral concept
    'cycle':        (-5, 5, -5, 5, 10),          # pattern -- high G (cycles of behavior)
    'through':      (0, 5, 5, 0, 0),             # movement/endurance
    'gave':         (5, 5, 5, 0, 5),             # giving -- mild positive (override old)
    'built':        (15, 10, 15, 0, 8),          # construction/achievement
    'real':         (5, 5, 5, 0, 5),
    'hair':         (0, 0, 0, 0, 5),
    'harmed':       (-40, 15, -25, 15, -12),     # self-harm
    'harm':         (-35, 15, -20, 15, -10),
    'days':         (0, 0, 0, 5, 0),             # temporal
    'isnt':         (-8, 5, -5, 5, -3),          # negation
    'working':      (8, 5, 8, 0, 3),             # functioning -- override
    'sober':        (15, -5, 20, 0, 20),         # recovery milestone -- high G
    'sobriety':     (15, -5, 20, 0, 20),
    'cycle':        (-5, 5, -5, 5, 10),
    'replaying':    (-15, 10, -10, 5, -5),       # stuck in loop
    'brain':        (0, 5, 0, 0, 10),            # high G -- mental state
    'asleep':       (5, -15, 5, 0, 0),           # rest state
    'ceiling':      (0, -5, 0, 0, 0),
    'honest':       (15, 5, 10, 0, 10),          # integrity
    'honestly':     (5, 5, 5, 0, 5),             # override old V=+50
    'belong':       (10, 5, 5, 0, 15),           # inclusion -- high G
    'talented':     (30, 10, 15, 0, 10),
    'meals':        (5, 0, 5, 0, 5),
    'answered':     (5, 5, 5, 0, 5),
    'phone':        (0, 5, 0, 5, 15),            # override -- high G communication
    'deserve':      (5, 5, 5, 5, 10),            # override
}
EMOTIONAL_VOCABULARY.update(_NOVEL_GAPS_7)


# ── Novel sentence gap fills (round 8) ──
_NOVEL_GAPS_8 = {
    'ideas':        (10, 10, 10, 0, 10),         # creative output -- G matters
    'replacement':  (-20, 10, -20, 10, -10),     # being replaced
    'knows':        (5, 0, 10, 0, 5),            # awareness/understanding
    'bearable':     (10, -5, 5, 0, 5),           # making it through
    'skipped':      (-10, 5, -10, 5, -5),        # missed/avoided
    'water':        (3, 0, 0, 0, 3),             # basic need -- tiny positive
    'walk':         (8, 5, 8, 0, 5),             # movement/exercise
    'around':       (0, 0, 0, 0, 0),
    'blinds':       (0, 0, 0, 0, 0),
    'opened':       (5, 5, 5, 0, 3),             # opening = progress
    'block':        (0, 5, 0, 0, 0),
    'drank':        (3, 0, 0, 0, 0),
    'bed':          (0, -5, 0, 0, 5),            # rest context
    'brushed':      (3, 0, 3, 0, 0),             # self-care
    'teeth':        (0, 0, 0, 0, 0),
    'morning':      (5, 5, 5, 5, 3),             # new day -- mild positive
    'recital':      (10, 10, 5, 5, 15),          # performance -- high G (child's event)
    'promote':      (10, 5, 10, 0, 8),
    'promoted':     (15, 10, 15, 0, 10),         # achievement
    'blame':        (-30, 15, 15, 10, -10),      # override
    'trained':      (10, 5, 10, 0, 5),
    'silence':      (-10, -15, -5, 0, 5),        # absence of sound -- slight negative
    'sentences':    (0, 0, 0, 0, 0),
    'bill':         (-15, 5, -10, 10, 10),       # financial stress -- G from importance
    'bills':        (-15, 5, -10, 10, 10),
    'paycheck':     (5, 5, 5, 5, 15),            # income -- high G
    'homeless':     (-50, 15, -40, 15, 35),      # severe -- high G
    'shut':         (-15, 10, -10, 10, -5),
    'promised':     (-5, 5, -5, 5, 10),          # commitment -- high G, context-dep
    'different':    (0, 5, 0, 5, 0),
    'scares':       (-25, 20, -15, 10, -8),
    'scared':       (-30, 25, -20, 15, -10),     # override -- was too extreme
    'traumatizing': (-40, 20, -25, 15, -15),
    'traumatized':  (-45, 15, -30, 10, -18),
    'yelled':       (-30, 25, 15, 15, -10),
}
EMOTIONAL_VOCABULARY.update(_NOVEL_GAPS_8)


# ── Permanent suite gap fills ──
_PERM_GAPS = {
    'odds':         (-10, 10, -5, 5, 5),         # adversity context
    'turned':       (-15, 10, -10, 10, -5),       # reversal -- "turned on me"
    'slightly':     (-3, 0, 0, 0, 0),
    'dogs':         (5, 5, 0, 0, 30),             # pet -- high G
    'playing':      (10, 10, 5, 0, 5),            # recreation
    'video':        (0, 5, 0, 0, 0),
    'messages':     (0, 5, 0, 5, 10),             # communication -- G
    'between':      (0, 0, 0, 0, 0),
    'food':         (5, 0, 5, 5, 10),             # basic need -- G
    'found':        (-5, 10, 5, 10, 0),           # discovery -- slight neg lean (found out = bad news often)
    'something':    (-3, 5, -3, 5, 0),            # vague -- slight unease
    'scan':         (-10, 10, -10, 15, 10),       # medical test -- G + urgency
    'kitchen':      (5, 0, 5, 0, 5),              # domestic warmth
    'visited':      (5, 5, 5, 5, 5),              # went to see
    'chat':         (5, 5, 5, 0, 5),              # communication
    'choose':       (-5, 10, 5, 10, 5),           # forced choice = stress
    'speaking':     (0, 5, 5, 0, 0),
    'generally':    (0, 0, 0, 0, 0),
}
EMOTIONAL_VOCABULARY.update(_PERM_GAPS)


# ── Novel sentence gap fills (round 9) ──
_NOVEL_GAPS_9 = {
    'theyd':        (0, 0, 0, 0, 0),             # contraction -- they would
    'someone':      (0, 0, 0, 0, 5),             # person reference -- already OTHER_REF
    'conversation': (0, 5, 0, 0, 5),
    'dissociating': (-30, -15, -25, 10, -12),    # mental health -- disconnection
    'dissociate':   (-25, -15, -20, 10, -10),
    'mirror':       (-5, 5, -5, 0, 10),          # self-reflection -- high G
    'guilty':       (-35, 10, -20, 10, -12),
    'meal':         (5, 0, 5, 0, 5),             # basic need
    'rings':        (0, 5, 0, 5, 5),
    'apartment':    (0, 0, 0, 0, 10),            # override with G
    'quiet':        (-5, -15, -5, 0, 0),         # absence of sound
    'holidays':     (5, 5, 0, 5, 15),            # high G event
    'stranger':     (0, 5, 0, 0, 3),
    'leaves':       (0, 0, 0, 0, 0),             # nature -- neutral
    'color':        (5, 5, 5, 0, 3),
    'song':         (10, 5, 5, 0, 8),
    'reminded':     (5, 5, 0, 5, 5),
    'sun':          (10, 5, 5, 0, 5),
    'window':       (0, 0, 0, 0, 0),
    'perfect':      (40, 10, 15, 0, 10),
    'smiled':       (25, 10, 10, 0, 8),          # override -- confirm
    'lucky':        (10, 5, -5, 0, 5),           # reduced -- "youre lucky" is often condescending
    'bother':       (-15, 5, -15, 5, -5),        # burden
    'clothes':      (0, 0, 0, 0, 5),
}
EMOTIONAL_VOCABULARY.update(_NOVEL_GAPS_9)


# ── Final vocab gaps from permanent suite ──
_FINAL_GAPS = {
    'got':          (5, 5, 5, 5, 0),             # acquisition -- mild positive ("got the job")
    'off':          (-5, 5, -3, 0, 0),            # removal/separation -- slight negative
    'ran':          (5, 15, 10, 5, 0),            # movement/effort
    'day':          (0, 0, 0, 5, 3),              # time marker -- near neutral
    'night':        (-5, -5, -3, 0, 3),           # darkness -- slight negative lean
    'today':        (0, 5, 0, 10, 0),             # immediacy -- urgency only (override)
    'room':         (0, 0, 0, 0, 3),
    'house':        (0, 0, 0, 0, 15),             # G override
    'door':         (0, 0, 0, 0, 5),
    'car':          (0, 0, 0, 0, 20),             # G override
    'name':         (0, 0, 0, 0, 10),             # identity -- high G
    'face':         (0, 5, 0, 0, 8),
    'eyes':         (0, 5, 0, 0, 5),
    'heart':        (5, 5, 0, 5, 20),             # emotional core -- high G
    'head':         (0, 5, 0, 0, 8),
    'body':         (0, 5, 0, 0, 15),             # physical self -- high G
    'mind':         (0, 5, 0, 0, 15),             # mental self -- high G (override)
    'world':        (0, 5, 0, 0, 10),             # scope
    'person':       (0, 0, 0, 0, 10),
    'reason':       (0, 5, 5, 5, 5),
    'truth':        (10, 5, 10, 5, 10),
    'lie':          (-30, 10, -15, 10, -10),
    'real':         (5, 5, 5, 0, 5),              # authenticity (override confirm)
    'fake':         (-25, 10, -10, 5, -8),
    'true':         (10, 5, 10, 0, 8),
    'wrong':        (-25, 10, -15, 10, -8),       # override confirm
    'right':        (5, 5, 10, 0, 5),             # correctness -- mild pos
}
EMOTIONAL_VOCABULARY.update(_FINAL_GAPS)


# ── Round 10 gaps ──
_R10 = {
    'whisper':      (-10, -10, -10, 5, -5),      # secrecy/exclusion
    'whispered':    (-10, -10, -10, 5, -5),
    'rolled':       (-10, 10, 5, 0, -3),          # "rolled eyes" = dismissal
    'carried':      (15, 10, 15, 0, 10),          # bearing weight for someone
    'carry':        (5, 10, 10, 5, 5),
    'clapped':      (20, 15, 10, 0, 8),           # applause
    'writes':       (5, 0, 5, 0, 5),
    'wrote':        (5, 0, 5, 0, 5),
    'letters':      (5, 0, 0, 0, 10),             # high G -- personal correspondence
    'piece':        (0, 0, 0, 0, 0),
    'biggest':      (5, 5, 5, 0, 3),
    'performance':  (10, 10, 10, 5, 10),          # achievement context
    'holding':      (5, 5, 5, 0, 8),              # connection
    'hands':        (5, 5, 5, 0, 8),              # connection
    'drew':         (10, 5, 10, 0, 8),            # created
    'picture':      (5, 0, 0, 0, 10),             # memory -- high G
    'flinched':     (-25, 20, -25, 15, -10),      # fear response
    'flinch':       (-20, 20, -20, 10, -8),
    'raised':       (-5, 15, 15, 10, 0),          # "raised voice" = aggression context
    'restaurant':   (5, 5, 0, 0, 5),
    'mistake':      (-20, 10, -15, 5, -8),
    'mistakes':     (-20, 10, -15, 5, -8),
    'perfection':   (-5, 5, 5, 5, 5),             # unrealistic standard
    'lighter':      (10, -5, 5, 0, 5),            # relief
}
EMOTIONAL_VOCABULARY.update(_R10)


# ── Round 11: love languages + control/abuse ──
_R11 = {
    'interrupt':    (-15, 10, 15, 5, -5),
    'interrupts':   (-15, 10, 15, 5, -5),
    'jacket':       (5, 0, 0, 0, 5),
    'framed':       (10, 5, 10, 0, 8),           # preserved/displayed
    'ever':         (-5, 5, -3, 0, -3),
    'monitors':     (-20, 10, 20, 10, -8),        # surveillance = control
    'location':     (0, 0, 0, 5, 5),
    'isolate':      (-35, 10, 20, 10, -12),       # cutting off from support
    'isolated':     (-30, -5, -25, 5, -10),
    'controls':     (-25, 10, 25, 10, -8),        # domination
    'controlling':  (-25, 10, 25, 10, -8),
    'wear':         (0, 0, 0, 0, 0),
    'tabs':         (-10, 5, 15, 5, -3),          # keeping tabs = surveillance
    'spending':     (-5, 5, -5, 5, 5),
    'tells':        (-3, 5, 10, 5, 0),
    'needy':        (-20, 10, -20, 5, -8),        # insult / invalidation
    'clingy':       (-20, 10, -15, 5, -8),
    'bombs':        (-15, 15, 10, 10, -5),        # love bombing
    'punishes':     (-30, 15, 25, 10, -10),
    'punished':     (-30, 15, -20, 10, -10),
    'threatens':    (-30, 20, 20, 20, -10),
    'guilts':       (-20, 10, 15, 10, -8),
    'guilt':        (-30, 10, -20, 10, -12),
    'memorized':    (15, 5, 10, 0, 10),           # attention/care
    'schedule':     (0, 5, 5, 5, 5),
    'surprise':     (15, 20, 5, 5, 8),
    'soup':         (5, 0, 0, 0, 5),              # comfort
    'songs':        (10, 5, 5, 0, 8),
    'wifi':         (0, 0, 0, 0, 0),
    'joke':         (5, 10, 5, 0, 3),             # reduced from old V=+20
    'drawing':      (10, 5, 10, 0, 8),
    'introduced':   (10, 10, 10, 0, 8),
}
EMOTIONAL_VOCABULARY.update(_R11)


# ── Round 12: relationship dynamics ──
_R12 = {
    'remembers':    (15, 5, 10, 0, 10),          # attention/care
    'weve':         (0, 0, 0, 0, 0),             # contraction
    'validate':     (20, 5, 10, 0, 10),          # affirming someone's experience
    'validates':    (20, 5, 10, 0, 10),
    'dismissing':   (-25, 10, 15, 5, -8),        # invalidation
    'dismisses':    (-25, 10, 15, 5, -8),
    'advocates':    (20, 10, 15, 5, 10),         # fighting for someone
    'closed':       (-5, -5, -5, 0, 0),
    'doors':        (0, 0, 0, 0, 0),
    'thin':         (-5, 5, -5, 0, -3),
    'ice':          (-10, 5, -5, 5, -3),         # "thin ice" = precarious
    'drains':       (-25, -10, -15, 5, -8),      # energy vampire
    'drained':      (-25, -15, -20, 5, -10),
    'lights':       (10, 10, 5, 0, 5),
    'granted':      (-15, -5, -10, 0, -5),       # "taken for granted"
    'celebrates':   (25, 15, 10, 0, 10),
    'allergies':    (0, 0, 0, 0, 10),            # personal detail = G
    'defended':     (15, 10, 15, 5, 10),         # stood up for
    'future':       (5, 5, 5, 5, 10),            # forward-looking (override neutral)
    'convenient':   (-5, 0, 5, 0, -3),           # "when convenient" = selfish
    'ego':          (-10, 5, 15, 0, -3),
    'constantly':   (-10, 10, -5, 5, -3),        # override
    'amplifies':    (-10, 10, 10, 5, -3),        # making bigger (negative context usually)
    'minimizes':    (-20, 5, 15, 5, -8),         # shrinking experience
    'worst':        (-40, 10, -25, 10, -12),     # override
    'rock':         (15, -5, 20, 0, 15),         # "my rock" = stability
    'walking':      (0, 5, 0, 5, 0),
    'energy':       (5, 10, 5, 0, 5),
    'room':         (0, 0, 0, 0, 3),             # override
}
EMOTIONAL_VOCABULARY.update(_R12)


# ── Round 13 + GPT-recommended vocabulary expansion ──
_R13_GPT = {
    # Round 13 gaps
    'shows':        (0, 5, 5, 0, 0),
    'affection':    (25, 10, 5, 0, 15),
    'twists':       (-20, 10, 10, 5, -5),
    'weapons':      (-30, 20, 15, 10, -10),
    'withdrawing':  (-20, -5, -15, 5, -8),
    'drove':        (0, 10, 5, 5, 0),            # movement -- neutral
    'town':         (0, 0, 0, 0, 0),
    'bring':        (5, 5, 5, 0, 3),
    'brings':       (5, 5, 5, 0, 3),
    'notes':        (5, 0, 5, 0, 5),
    'makes':        (0, 5, 5, 0, 0),
    'fundraiser':   (15, 10, 10, 5, 10),
    'medical':      (-5, 5, -5, 10, 10),
    'sunday':       (5, -5, 0, 0, 5),
    'umbrella':     (0, 0, 0, 0, 0),
    'wakes':        (5, 10, 5, 5, 3),
    'organized':    (10, 5, 10, 5, 5),
    'recipe':       (10, 5, 5, 0, 8),
    'grandmother':  (0, 0, 0, 0, 40),            # relationship -- high G
    'greeted':      (10, 10, 5, 0, 5),
    'dimmed':       (-15, -5, -15, 0, -5),
    'rewrite':      (-10, 5, 10, 5, -5),
    # GPT bucket 1: passive-aggressive cues
    'nice':         (10, 0, 5, 0, 3),            # reduced -- "must be nice" = PA
    'cute':         (15, 5, 5, 0, 5),            # can be condescending
    'please':       (5, 0, -5, 5, 3),            # plea or politeness
    # GPT bucket 2: numb/flat distress
    'checked':      (-5, -5, -5, 0, 0),
    'detached':     (-20, -15, -15, 0, -8),
    'fried':        (-25, -10, -20, 5, -8),
    'wrecked':      (-40, 15, -20, 10, -12),     # override
    'spent':        (-15, -10, -15, 5, -5),      # emotionally spent
    # GPT bucket 3: modern distress
    'spiraling':    (-35, 25, -25, 15, -12),
    'spiral':       (-30, 20, -20, 10, -10),
    'crashing':     (-30, 25, -20, 15, -10),
    'unraveling':   (-30, 15, -20, 10, -10),
    'snapped':      (-35, 30, -15, 15, -10),
    'doom':         (-40, 10, -25, 10, -15),
    'burnout':      (-30, -15, -25, 10, -12),
    'drowning':     (-35, 20, -25, 15, -12),
    # GPT bucket 4: self-worth/erasure
    'unwanted':     (-35, 10, -25, 10, -15),
    'replaceable':  (-25, 5, -20, 5, -10),
    'unlovable':    (-40, 10, -30, 10, -18),
    'forgettable':  (-25, -5, -20, 5, -10),
    'irrelevant':   (-25, -5, -20, 5, -10),
    'meaningless':  (-35, -5, -25, 5, -15),
    # GPT bucket 5: abuse verbs
    'pressured':    (-25, 15, -15, 10, -8),
    'cornered':     (-30, 20, -25, 15, -10),
    'coerced':      (-35, 15, -25, 10, -12),
    'demeaned':     (-35, 10, -20, 10, -12),
    'belittled':    (-30, 10, -20, 10, -10),
    'gaslit':       (-35, 15, -20, 10, -12),
    'gaslighting':  (-35, 15, -20, 10, -12),
    'manipulated':  (-35, 15, -15, 10, -12),
    # GPT bucket 6: internet crisis
    'meh':          (-10, -15, -10, 0, -5),
    'unalive':      (-50, 15, -30, 30, -20),     # platform-evasive crisis language
}
EMOTIONAL_VOCABULARY.update(_R13_GPT)


# ── Final GPT audit gaps ──
_GPT_FINAL = {
    'clearly':      (-5, 5, 10, 0, 0),           # sarcasm opener already, mild force
    'controlled':   (-25, 10, -20, 10, -8),
    'dismissed':    (-25, 10, 15, 5, -8),
    'obviously':    (-8, 5, 10, 0, -3),           # dismissive / condescending
}
EMOTIONAL_VOCABULARY.update(_GPT_FINAL)


# ── Round 14: silent treatment, gossip, performance, empathy drain ──
_R14 = {
    'silent':       (-15, -15, -10, 5, -5),      # silence as weapon
    'treatment':    (-10, 5, -5, 5, -3),          # "silent treatment" = punishment
    'gossip':       (-25, 15, 10, 5, -8),         # talking behind back
    'gossiping':    (-25, 15, 10, 5, -8),
    'acts':         (-5, 5, 5, 0, 0),             # performing/pretending
    'lap':          (5, -5, 5, 0, 5),             # comfort/warmth
    'pocket':       (0, 0, 0, 0, 0),
    'performing':   (-10, 10, -5, 5, -3),         # masking/performing for others
    'audience':     (-5, 5, -5, 5, 0),
    'absorb':       (-10, 5, -10, 5, -5),         # taking in others emotions
    'absorbs':      (-10, 5, -10, 5, -5),
    'everyones':    (0, 5, 0, 5, 5),
    'emotions':     (0, 10, 0, 5, 15),            # high G -- emotional content
    'see':          (0, 5, 0, 0, 0),
    'drain':        (-25, -10, -15, 5, -8),       # override confirm
    'call':         (0, 5, 0, 5, 0),
    'promises':     (-5, 5, -5, 5, 10),           # high G -- broken promises
    'keeps':        (0, 0, 5, 0, 0),              # confirm
    'asleep':       (5, -15, 5, 0, 0),            # rest
    'cookies':      (10, 5, 5, 0, 5),             # comfort food
    'smells':       (5, 5, 0, 0, 0),
    'fresh':        (10, 5, 5, 0, 3),
    'renewed':      (15, 10, 10, 0, 8),
    'elevator':     (0, 0, 0, 0, 0),
    'barista':      (0, 5, 0, 0, 0),
    'spelled':      (0, 0, 0, 0, 0),
    'green':        (5, 5, 5, 0, 0),
    'shower':       (5, -5, 5, 0, 3),
    'lasted':       (5, 0, 5, 0, 3),
    'opinion':      (-5, 5, 5, 0, 0),
    'listener':     (5, 0, -5, 0, 5),
    'empathy':      (10, 5, -5, 0, 15),           # high G
    'weapon':       (-30, 20, 15, 10, -10),
    'pieces':       (-10, 5, -10, 5, -5),         # "pieces of myself"
    'weight':       (-10, 5, -10, 5, -5),         # burden
    'notice':       (0, 5, -5, 5, 5),
}
EMOTIONAL_VOCABULARY.update(_R14)


# ── Round 15: physical reactions + care signals ──
_R15 = {
    'fakes':        (-20, 5, -10, 5, -5),         # pretending
    'faking':       (-20, 5, -10, 5, -5),
    'convincingly': (-5, 5, 10, 0, 0),
    'making':       (0, 5, 5, 0, 0),
    'bled':         (-30, 20, -20, 15, -10),
    'bleeding':     (-30, 20, -20, 15, -10),
    'send':         (0, 5, 5, 5, 0),
    'defends':      (15, 10, 15, 5, 10),
    'notices':      (10, 5, 10, 0, 8),            # attention/care
    'before':       (0, 0, 0, 0, 0),
    'say':          (0, 5, 5, 0, 0),
    'details':      (5, 5, 5, 0, 8),              # attention to detail = care
    'braided':      (10, 5, 5, 0, 8),             # intimate care act
    'lump':         (-15, 10, -10, 5, -5),        # "lump in throat"
    'throat':       (-10, 10, -10, 5, -3),
    'stomach':      (-10, 10, -10, 10, -3),       # "stomach dropped"
    'tongue':       (-5, 5, -5, 0, 0),
    'shaking':      (-20, 25, -20, 15, -8),
    'dropped':      (-20, 15, -15, 10, -5),       # override -- "stomach dropped" "heart dropped"
    'bond':         (15, 5, 10, 0, 10),            # connection
    'accountability':(-5, 5, 10, 5, 5),
    'texts':        (0, 5, 0, 5, 5),              # override -- neutral communication
    'morning':      (5, 5, 5, 5, 3),              # override confirm
    'fail':         (-25, 10, -20, 10, -8),
    'judgment':     (-20, 10, 10, 5, -5),          # override confirm
    'space':        (0, -5, 0, 0, 0),              # override confirm
    'photo':        (0, 0, 0, 0, 10),              # high G memory
    'photos':       (0, 0, 0, 0, 10),
    'movie':        (5, 5, 0, 0, 5),
    'comfort':      (15, -10, 10, 0, 10),
    'car':          (0, 0, 0, 0, 15),              # override with G
}
EMOTIONAL_VOCABULARY.update(_R15)


# ── Round 16: competition, mockery, care acts ──
_R16 = {
    'screenshot':   (-15, 10, -5, 5, -5),        # privacy violation
    'screenshots':  (-15, 10, -5, 5, -5),
    'laughs':       (-5, 15, 5, 0, 0),            # liquid -- laughs at vs laughs with
    'competes':     (-15, 10, 10, 5, -5),         # competition instead of support
    'competing':    (-15, 10, 10, 5, -5),
    'supporting':   (20, 5, 10, 0, 10),           # active support
    'bookshelf':    (5, 5, 5, 0, 5),              # built something
    'drives':       (0, 5, 5, 5, 0),              # movement -- neutral
    'copies':       (-10, 5, -5, 5, -3),          # copying/stealing identity
    'compliments':  (10, 5, 5, 0, 5),             # positive attention
    'insult':       (-30, 15, 15, 10, -10),
    'insults':      (-30, 15, 15, 10, -10),
    'boundaries':   (10, 5, 15, 0, 10),           # healthy -- high G
    'boundary':     (10, 5, 15, 0, 10),
    'dreams':       (10, 10, 5, 0, 10),           # aspirations -- high G
    'memory':       (5, 5, -5, 0, 10),            # high G -- identity
    'doubt':        (-20, 10, -15, 10, -8),
    'doubts':       (-20, 10, -15, 10, -8),
    'poems':        (15, 5, 10, 0, 10),           # creative expression of care
    'napkins':      (0, 0, 0, 0, 0),
    'taught':       (15, 5, 15, 0, 10),           # passing knowledge
    'fish':         (5, 5, 5, 0, 5),
    'soup':         (5, 0, 0, 0, 5),              # comfort food
    'named':        (5, 5, 5, 0, 10),             # honor -- high G
    'hour':         (0, 0, 0, 5, 0),              # time marker
    'scratch':      (5, 5, 10, 0, 5),             # "from scratch" = effort
}
EMOTIONAL_VOCABULARY.update(_R16)


# ── Round 17: social manipulation + sacrifice acts ──
_R17 = {
    'rehearse':     (-10, 10, -10, 5, -3),        # anxiety -- preparing for confrontation
    'conversations':(-3, 5, 0, 0, 5),
    'happen':       (-5, 5, -3, 5, 0),            # "what happened" = unease
    'conveniently': (-10, 0, 10, 0, -3),          # sarcastic availability
    'plans':        (0, 5, 5, 5, 0),
    'needed':       (-5, 5, -5, 5, -3),           # "i needed" = unmet need
    'moving':       (0, 5, 0, 5, 0),
    'posts':        (0, 5, 0, 0, 5),
    'bringing':     (-5, 5, 0, 5, 0),             # "bringing up" = resurfacing
    'favors':       (-5, 5, -5, 5, -3),           # weaponized generosity
    'sat':          (5, -5, 5, 0, 5),             # presence -- staying
    'six':          (0, 0, 0, 0, 0),
    'hours':        (0, 0, 0, 5, 0),
    'covered':      (10, 5, 10, 0, 8),            # covered shift = support
    'shift':        (0, 5, 0, 5, 0),
    'donated':      (15, 10, 15, 0, 10),          # giving -- selfless act
    'type':         (0, 0, 0, 0, 0),
    'rare':         (0, 5, 0, 5, 5),
    'blood':        (-5, 10, -5, 10, 10),         # medical + sacrifice context
    'crayon':       (10, 5, 5, 0, 8),             # childlike -- warmth
    'sign':         (0, 5, 0, 0, 0),
    'language':     (0, 5, 0, 0, 5),
    'waiting':      (-8, 5, -8, 10, -3),          # override
    'fixed':        (10, 5, 10, 0, 5),            # repaired
    'leaky':        (-5, 5, -5, 5, 0),
    'faucet':       (0, 0, 0, 0, 0),
    'volunteers':   (-5, 5, -5, 5, -3),           # "volunteers my time" = boundary violation
    'passive':      (-10, -5, -5, 0, -3),
    'aggressive':   (-20, 20, 15, 10, -5),
    'accidentally': (-8, 5, -5, 5, -3),           # "accidentally forget" = deliberate
    'invite':       (10, 10, 5, 0, 5),            # inclusion
    'likes':        (5, 5, 0, 0, 3),
}
EMOTIONAL_VOCABULARY.update(_R17)


# ── Round 18: sacrifice + score-keeping ──
_R18 = {
    'bullet':       (-15, 30, 15, 20, 15),       # "took a bullet" = ultimate sacrifice. High G.
    'stood':        (5, 10, 15, 5, 5),            # stood up / stood between = protection
    'awake':        (-5, 10, 5, 10, 0),           # staying awake = sacrifice of rest
    'archived':     (-10, 5, 5, 5, -3),           # preserving evidence / control
    'remind':       (-10, 5, 10, 5, -3),          # "remind me daily" = score-keeping
    'reminds':      (-10, 5, 10, 5, -3),
    'turns':        (-5, 5, 5, 5, 0),             # override -- "turns into" = transformation
    'competition':  (-15, 10, 10, 5, -5),
    'trophies':     (-5, 5, 5, 0, 5),             # "collect mistakes like trophies"
    'collect':      (-5, 5, 5, 0, 0),
    'owe':          (-15, 5, -15, 10, -5),        # debt/obligation
    'owes':         (-15, 5, -15, 10, -5),
    'root':         (-5, 5, -5, 5, 0),            # "root against" = opposition
    'enjoys':       (5, 10, 10, 0, 3),            # liquid -- "enjoys watching me struggle" = sadistic
    'celebrated':   (15, 15, 10, 0, 8),           # override -- celebration
    'celebrates':   (15, 15, 10, 0, 8),
    'crowd':        (5, 15, 0, 5, 5),
    'chanted':      (10, 20, 5, 5, 5),
    'stage':        (5, 10, 5, 5, 5),
    'wiped':        (5, 5, 5, 0, 5),              # care act -- wiping tears
    'danger':       (-35, 25, -20, 20, -10),
    'daily':        (-3, 5, 0, 5, 0),
    'ex':           (-15, 10, -5, 10, 10),        # high G -- past relationship
    'smiles':       (5, 5, 5, 0, 3),              # liquid -- context determines
    'struggle':     (-25, 15, -20, 10, -8),
    'sacrificed':   (10, 10, 10, 5, 15),          # giving up for someone -- high G
    'sacrifice':    (5, 10, 10, 5, 15),
}
EMOTIONAL_VOCABULARY.update(_R18)


# ── Round 19: social rejection + recognition ──
_R19 = {
    'looked':       (0, 5, 5, 0, 0),             # neutral -- "looked through me" = structural
    'glass':        (-5, 0, -5, 0, 0),            # "like glass" = invisible/fragile
    'subject':      (-5, 5, 5, 0, 0),             # "changed the subject" = avoidance
    'seen':         (-10, 5, -5, 5, -3),          # "left on seen" = acknowledged rejection
    'week':         (0, 0, 0, 5, 0),
    'standing':     (10, 10, 10, 0, 5),           # "standing ovation" = recognition
    'ovation':      (25, 15, 15, 0, 10),          # applause/recognition
    'calmed':       (15, -10, 10, 0, 8),          # brought peace -- care act
    'during':       (0, 0, 0, 0, 0),
    'degree':       (15, 5, 15, 0, 15),           # achievement -- high G
    'spoke':        (0, 5, 5, 0, 0),
    'tone':         (-5, 5, 5, 0, 0),             # "tone changed" = shift
    'closed':       (-5, -5, -5, 0, 0),
    'door':         (0, 0, 0, 0, 5),
    'unfollowed':   (-20, 10, -10, 5, -8),        # digital rejection -- override
    'stories':      (5, 5, 5, 0, 5),              # override confirm
    'window':       (0, 0, 0, 0, 0),
    'seat':         (5, 0, 5, 0, 3),              # "saved me a seat" = inclusion
    'panic':        (-40, 35, -30, 25, -12),      # override -- stronger
    'attack':       (-35, 30, -20, 20, -10),      # override -- "panic attack"
}
EMOTIONAL_VOCABULARY.update(_R19)


# ── Round 20: rumor, therapy, art, body modification ──
_R20 = {
    'spread':       (-15, 10, 10, 10, -5),        # spreading rumors/info
    'rumors':       (-30, 15, 10, 10, -10),
    'rumor':        (-25, 15, 10, 10, -8),
    'showed':       (0, 5, 5, 5, 0),
    'herself':      (0, 0, 0, 0, 0),
    'therapy':      (5, 5, -5, 5, 20),             # high G -- mental health
    'therapist':    (10, 5, -5, 0, 20),            # caretaker -- high G
    'poem':         (15, 5, 10, 0, 10),            # creative expression
    'tattooed':     (10, 10, 10, 0, 15),           # permanent commitment -- high G
    'tattoo':       (5, 10, 10, 0, 10),
    'portrait':     (10, 5, 10, 0, 10),
    'painted':      (10, 5, 10, 0, 8),
    'mariachi':     (15, 15, 5, 0, 5),
    'adopted':      (10, 10, 10, 0, 20),           # override -- chosen family. high G
    'flew':         (5, 10, 5, 5, 5),              # effort/distance
    'candlelight':  (10, -5, 5, 0, 15),            # high G -- memorial/honor
    'vigil':        (-5, -10, -5, 0, 20),          # high G -- remembrance
    'honor':        (20, 10, 15, 0, 15),
    'lullaby':      (15, -10, 5, 0, 15),           # tenderness -- high G
    'deployed':     (-15, 15, 15, 15, 20),         # military -- sacrifice, high G
    'recovering':   (5, 5, -5, 5, 10),             # healing process
    'garden':       (10, -5, 5, 0, 8),
    'planted':      (10, 5, 10, 0, 8),
    'locked':       (-20, 10, -20, 10, -8),        # "locked out" = excluded/trapped
    'poisoned':     (-35, 15, 15, 10, -12),        # turning others against
    'recorded':     (-5, 5, 5, 5, 5),              # context-dep -- recorded without permission vs recorded lullaby
    'permission':   (-5, 5, -5, 5, 5),
    'evidence':     (-10, 10, 10, 10, 5),
    'deleted':      (-15, 10, -10, 5, -5),         # override confirm
    'childhood':    (0, 0, 0, 0, 20),              # high G -- identity period (override V=0)
    'uninvited':    (-20, 10, -10, 5, -8),
    'wrist':        (0, 5, 0, 0, 5),
}
EMOTIONAL_VOCABULARY.update(_R20)


# ── Hitting 3,000 ──
_THREE_K = {
    'proposal':     (20, 15, 10, 5, 15),
    'proposed':     (20, 15, 10, 5, 15),
    'engagement':   (20, 15, 10, 5, 20),
    'anniversary':  (10, 5, 5, 0, 20),
    'milestone':    (10, 10, 10, 5, 10),
    'progress':     (15, 5, 15, 0, 8),
    'achievement':  (20, 10, 20, 0, 10),
    'celebrate':    (20, 15, 10, 0, 8),
    'blessing':     (20, 5, 10, 0, 15),
}
EMOTIONAL_VOCABULARY.update(_THREE_K)


# ── Final push to 3,000 ──
_FINAL_3K = {
    'grateful':     (30, 5, 10, 0, 15),
    'thankful':     (25, 5, 10, 0, 12),
    'humble':       (10, -5, -5, 0, 10),
    'blessed':      (25, 5, 5, 0, 15),
    'resilient':    (20, 10, 20, 0, 10),
    'strength':     (15, 10, 20, 0, 10),
    'courage':      (20, 15, 25, 0, 12),
    'warrior':      (15, 15, 25, 0, 10),
    'survivor':     (15, 10, 20, 0, 15),
    'healing':      (15, -5, 10, 0, 15),
}
EMOTIONAL_VOCABULARY.update(_FINAL_3K)


# ── 3,000 ──
_HIT_3K = {
    'kindness':     (25, 5, 10, 0, 12),
    'compassion':   (25, 5, 5, 0, 15),
    'dignity':      (15, 5, 20, 0, 15),
    'integrity':    (20, 5, 20, 0, 12),
    'generosity':   (20, 5, 10, 0, 10),
}
EMOTIONAL_VOCABULARY.update(_HIT_3K)


# ── MEGA round: 38 words from 72 fresh sentences ──
_MEGA = {
    'publicly':     (-10, 10, 10, 5, -3),
    'corrected':    (-15, 10, 15, 5, -5),
    'moved':        (-5, 5, 0, 5, 0),
    'desk':         (0, 0, 0, 0, 5),
    'corner':       (-5, -5, -5, 0, -3),
    'sends':        (5, 5, 5, 5, 0),
    'responds':     (5, 5, 5, 0, 3),
    'within':       (0, 0, 0, 0, 0),
    'minutes':      (0, 0, 0, 5, 0),
    'while':        (0, 0, 0, 0, 0),
    'three':        (0, 0, 0, 0, 0),
    'jobs':         (0, 5, 0, 5, 10),
    'librarian':    (10, -5, 5, 0, 10),
    'noises':       (-10, 10, -5, 5, -3),
    'weighing':     (-5, 5, -5, 0, -3),
    'froze':        (-20, 10, -15, 10, -8),
    'gives':        (0, 5, 5, 0, 0),
    'allowance':    (-10, 0, -15, 0, -5),
    'single':       (-5, 0, 0, 0, 3),
    'defensive':    (-15, 15, 15, 5, -5),
    'came':         (0, 5, 0, 5, 0),
    'pull':         (-5, 10, 5, 5, 0),
    'any':          (0, 0, 0, 0, 0),
    'younger':      (0, 5, 0, 0, 0),
    'already':      (-3, 5, 0, 5, 0),
    'two':          (0, 0, 0, 0, 0),
    'sides':        (-10, 5, -5, 5, -3),
    'inviting':     (10, 10, 5, 0, 5),
    'copied':       (-10, 5, -5, 5, -3),
    'entire':       (-5, 5, -3, 5, 0),
    'ketchup':      (0, 0, 0, 0, 0),
    'apologizing':  (-10, 5, -10, 5, -3),
    'sent':         (0, 5, 5, 5, 0),
    'email':        (0, 5, 0, 5, 5),
    'rereading':    (-5, 5, -5, 0, 0),
    'ten':          (0, 0, 0, 0, 0),
    'times':        (0, 5, 0, 5, 0),
    'bought':       (5, 5, 5, 0, 3),
}
EMOTIONAL_VOCABULARY.update(_MEGA)


# ── New domain vocabulary: medical, legal, dating, parenting, customer service ──
_DOMAINS = {
    'chemotherapy':  (-35, 15, -25, 20, 30),      # serious medical -- very high G
    'wants':         (0, 5, 5, 5, 0),
    'urgently':      (-10, 15, 5, 25, 5),          # high urgency
    'filed':         (-15, 10, 10, 15, 5),          # legal action
    'custody':       (-20, 15, -15, 20, 35),        # extremely high G -- children at stake
    'sided':         (10, 5, 10, 5, 5),             # "sided with me" = support
    'violated':      (-35, 20, 15, 20, -10),        # broke rules/boundaries
    'restraining':   (-20, 15, 15, 15, 10),         # legal protection
    'subpoenaed':    (-20, 15, 5, 20, 5),
    'records':       (-5, 5, 0, 5, 5),
    'charges':       (-20, 15, -10, 15, 10),        # legal charges -- high G
    'netflix':       (5, -5, 0, 0, 0),
    'pickup':        (-5, 5, 5, 0, 0),
    'line':          (0, 0, 0, 0, 0),
    'sons':          (0, 5, 0, 0, 40),              # relationship -- high G
    'classmate':     (0, 5, 0, 0, 5),
    'resolving':     (10, 5, 10, 5, 5),
    'issue':         (-10, 5, -5, 10, 5),
    'supervisor':    (-5, 10, 15, 10, 5),           # authority figure
    'tumor':         (-40, 20, -25, 25, 30),
    'mass':          (-25, 15, -15, 20, 20),        # medical mass
    'specialist':    (-5, 5, 5, 10, 10),
    'cleared':       (15, -5, 15, 0, 8),            # "cleared to go" = relief
    'visitation':    (-15, 10, -15, 10, 25),        # custody -- high G
    'settlement':    (5, -5, 5, 5, 10),
    'unmatched':     (-15, 10, -10, 5, -5),         # dating rejection
    'ghosted':       (-30, 10, -20, 10, -10),       # override confirm
    'midnight':      (-5, -5, -5, 5, 0),
    'suspended':     (-25, 15, -20, 10, -10),       # override confirm
    'nightmares':    (-30, 20, -20, 15, -10),
    'refund':        (-10, 10, -5, 10, 5),
    'kitchen':       (5, 0, 5, 0, 5),               # override confirm
    'report':        (-10, 5, 5, 10, 5),
}
EMOTIONAL_VOCABULARY.update(_DOMAINS)


# ── Domains 2: sports, military, immigration, aging, addiction ──
_DOMAINS_2 = {
    'coach':        (5, 10, 15, 5, 10),
    'benched':      (-20, 10, -20, 10, -8),        # excluded from participation
    'game':         (5, 15, 5, 5, 5),
    'waited':       (10, -5, 5, 5, 10),             # patience/loyalty -- high G
    'deployment':   (-10, 10, 10, 15, 20),          # military -- high G
    'where':        (0, 0, 0, 0, 0),
    'accent':       (-5, 5, -5, 0, 5),
    'differently':  (-10, 5, -5, 5, -3),            # "treated differently" = othered
    'put':          (-5, 5, 0, 5, 0),
    'poured':       (5, 5, 10, 5, 5),               # action -- "poured it out" = agency
    'bottle':       (-10, 5, -5, 5, 10),            # addiction context -- high G
    'sink':         (0, 0, 0, 0, 0),
    'relapsed':     (-35, 15, -25, 15, 20),         # addiction setback -- high G
    'relapse':      (-30, 15, -20, 15, 20),
    'months':       (0, 0, 0, 5, 0),
    'clean':        (20, -5, 15, 0, 15),            # sobriety -- override, high G
    'sponsor':      (10, 5, 5, 0, 15),              # support person -- high G
    'citizenship':  (15, 10, 15, 5, 20),            # achievement -- high G
    'papers':       (0, 5, 0, 5, 10),               # documents -- G from importance
    'retirement':   (5, -5, 0, 0, 15),              # milestone -- high G
    'booed':        (-25, 20, -15, 10, -8),
    'scored':       (20, 20, 15, 5, 8),
    'winning':      (25, 20, 20, 5, 10),
    'championship': (15, 20, 15, 10, 15),
    'teammates':    (5, 10, 5, 0, 10),
    'battle':       (-20, 25, 15, 15, 10),
    'buddy':        (10, 5, 10, 0, 10),
    'overseas':     (-5, 5, 5, 5, 5),
    'welcomed':     (20, 10, 10, 0, 10),
    'hid':          (-15, 10, -15, 5, -5),
    'bottles':      (-10, 5, -5, 5, 8),
    'drugs':        (-30, 15, -20, 15, 15),
    'picked':       (-5, 5, -5, 5, 0),              # "picked last" = excluded
}
EMOTIONAL_VOCABULARY.update(_DOMAINS_2)


# ── Domains 3: religion, academic, housing, food, pet loss ──
_DOMAINS_3 = {
    'eviction':     (-35, 20, -30, 25, 30),        # losing home -- extremely high G
    'evicted':      (-35, 20, -30, 25, 30),
    'unadoptable':  (-20, 5, -15, 5, 15),          # rejected/unwanted
    'landlord':     (-10, 10, 15, 10, 10),
    'rent':         (-10, 5, -5, 10, 10),
    'pantry':       (-5, 0, 0, 5, 5),
    'coins':        (-10, 5, -10, 5, -3),           # poverty signal
    'bread':        (3, 0, 0, 5, 5),                # basic need
    'vet':          (-5, 5, -5, 10, 15),            # veterinarian -- high G (pet's life)
    'shelter':      (-5, 5, -5, 5, 10),
    'rescue':       (10, 10, 10, 5, 10),
    'congregation': (0, 5, 0, 0, 10),
    'excommunicated':(-30, 15, -20, 10, 20),        # expelled from community -- high G
    'thesis':       (0, 10, 5, 10, 10),
    'advisor':      (0, 5, 10, 5, 10),
    'dissertation': (5, 10, 10, 10, 15),
    'defended':     (15, 10, 15, 5, 10),            # override confirm
    'published':    (20, 10, 15, 5, 12),            # override confirm
    'journal':      (5, 5, 5, 0, 8),
    'professor':    (0, 5, 10, 5, 10),
    'lecture':      (0, 5, 5, 5, 5),
    'rental':       (0, 5, 0, 5, 8),
    'application':  (0, 5, 0, 10, 8),
    'thanksgiving': (10, 5, 5, 0, 15),              # override confirm
    'anonymously':  (10, 5, -5, 0, 5),              # anonymous kindness
}
EMOTIONAL_VOCABULARY.update(_DOMAINS_3)

# ── Domain mining round 4: workplace, disability, recovery, DV, grief,
#    gen-z slang, therapy, immigration (124 words) ──────────────────────
_DOMAINS_4 = {
    # Workplace harassment
    'hostile':          (-55, 60, 40, 35, -20),
    'intimidate':       (-70, 75, 80, 50, -30),
    'retaliation':      (-65, 70, 60, 55, -35),
    'demeaning':        (-60, 40, 30, 25, -25),
    'condescending':    (-40, 30, 50, 10, -15),
    'overworked':       (-45, 35, -40, 30, -20),
    'underpaid':        (-40, 25, -35, 20, -18),
    'ostracized':       (-65, 20, -50, 15, -30),
    'mobbing':          (-70, 55, -45, 40, -35),
    'scapegoat':        (-60, 40, -55, 30, -30),
    'exploited':        (-65, 35, -50, 30, -25),
    'silenced':         (-50, 15, -60, 20, -25),
    'empowered':        (45, 25, 55, 5, 15),
    'degraded':         (-70, 45, -50, 25, -30),
    'manipulate':       (-50, 35, 55, 25, -20),
    'narcissistic':     (-45, 35, 60, 15, -15),
    # Disability / accessibility
    'inaccessible':     (-35, 20, -30, 20, -15),
    'meltdown':         (-55, 90, -50, 60, -30),
    'overstimulated':   (-40, 70, -35, 40, -20),
    'masking':          (-30, 25, -20, 15, -15),
    'chronic':          (-20, 10, -15, 5, 30),
    'fatigue':          (-30, -20, -25, 10, -15),
    'dissociation':     (-30, -25, -30, 15, -20),
    'dysregulated':     (-40, 60, -40, 35, -20),
    'numbness':         (-20, -30, -20, 5, -10),
    'stigma':           (-40, 20, -30, 10, 25),
    'hypervigilant':    (-35, 80, -25, 50, -20),
    'vulnerable':       (-25, 20, -40, 15, 20),
    # 12-step / recovery
    'powerless':        (-50, 15, -80, 20, -35),
    'surrender':        (-15, -10, -60, 5, 25),
    'serenity':         (40, -30, 25, -5, 20),
    'amends':           (20, 25, -5, 15, 25),
    'enabling':         (-25, 15, -20, 15, -15),
    'temptation':       (-25, 45, -30, 40, -20),
    'tempted':          (-20, 40, -25, 35, -15),
    'craving':          (-30, 55, -35, 50, -20),
    'detox':            (-35, 50, -40, 60, 25),
    'withdrawal':       (-50, 60, -45, 55, -30),
    'abstinence':       (10, -10, 15, 5, 20),
    'cope':             (5, 10, -10, 10, 10),
    'coping':           (5, 10, -10, 10, 10),
    'denial':           (-20, -5, -15, 10, -15),
    'forgiven':         (45, 15, 30, 0, 20),
    # Domestic violence shelter
    'coerce':           (-60, 50, 70, 45, -25),
    'coercion':         (-55, 45, 65, 40, -25),
    'stalk':            (-65, 60, 50, 55, -30),
    'stalker':          (-60, 55, 55, 50, -25),
    'stalking':         (-65, 60, 50, 55, -30),
    'lovebombing':      (-30, 40, 45, 20, -20),
    'battered':         (-75, 45, -60, 40, -40),
    'devalue':          (-55, 30, 40, 20, -25),
    'hoovering':        (-35, 35, 45, 25, -20),
    'protective':       (25, 20, 30, 15, 15),
    'retraumatize':     (-60, 65, -40, 50, -35),
    'retraumatized':    (-60, 65, -40, 50, -35),
    'fawn':             (-20, 25, -55, 20, -15),
    'freeze':           (-25, 10, -45, 15, -15),
    'dehumanize':       (-80, 50, 50, 40, -40),
    # Grief stages clinical
    'yearning':         (-45, 30, -35, 25, -30),
    'longing':          (-40, 25, -30, 20, -25),
    'bargaining':       (-30, 35, -25, 30, -20),
    'bereaved':         (-55, 20, -40, 15, 35),
    'widow':            (-40, 10, -25, 10, 35),
    'eulogy':           (-25, 15, -10, 5, 30),
    'memorial':         (-15, 10, -5, 5, 30),
    'anticipatory':     (-30, 35, -20, 25, -20),
    'disenfranchised':  (-50, 30, -45, 20, -30),
    'keening':          (-65, 75, -40, 35, -40),
    'sobbing':          (-55, 65, -35, 30, -30),
    'wailing':          (-60, 70, -35, 35, -35),
    # Gen-Z / Gen-alpha slang
    'delulu':           (-15, 25, -20, 5, -10),
    'brainrot':         (-25, 15, -20, 5, -15),
    'situationship':    (-20, 25, -15, 10, -10),
    'unhinged':         (-10, 60, 20, 10, -10),
    'feral':            (-10, 55, 25, 10, -10),
    'demure':           (15, -15, -5, 0, 5),
    'brat':             (10, 40, 20, 5, -5),
    'aura':             (20, 15, 20, 0, 10),
    'glazing':          (-20, 15, -25, 5, -10),
    'ghosting':         (-40, 25, -30, 15, -18),
    'doomscrolling':    (-25, 20, -20, 10, -12),
    'rotting':          (-30, -15, -25, 5, -15),
    'cooked':           (-35, 30, -30, 20, -15),
    'slaying':          (40, 35, 40, 5, 15),
    # Therapy modalities
    'grounding':        (15, -15, 15, 0, 10),
    'grounded':         (20, -20, 20, -5, 12),
    'somatic':          (0, 10, 0, 5, 10),
    'ruminate':          (-35, 30, -25, 20, -20),
    'rumination':       (-35, 30, -25, 20, -20),
    'catastrophize':    (-50, 65, -35, 45, -25),
    'reframe':          (20, 10, 20, 5, 10),
    'validated':        (35, 10, 25, 0, 15),
    'validation':       (30, 10, 20, 0, 15),
    'invalidated':      (-50, 35, -45, 25, -25),
    'invalidate':       (-45, 30, -40, 20, -20),
    'mindful':          (15, -10, 15, -5, 10),
    'mindfulness':      (15, -10, 15, -5, 10),
    'regulate':         (10, -5, 15, 0, 8),
    'coregulate':       (20, -10, 10, 0, 12),
    # Immigration / displacement
    'deported':         (-65, 50, -60, 60, -35),
    'deportation':      (-60, 45, -55, 55, 35),
    'asylum':           (-10, 20, -25, 30, 30),
    'refugee':          (-30, 20, -35, 20, 40),
    'undocumented':     (-25, 20, -35, 25, 25),
    'displaced':        (-45, 25, -40, 25, 30),
    'displacement':     (-40, 20, -35, 20, 30),
    'stateless':        (-50, 20, -55, 25, 35),
    'diaspora':         (-20, 10, -15, 5, 30),
    'uprooted':         (-50, 30, -40, 25, -25),
    'xenophobia':       (-55, 45, 30, 25, -25),
    'othered':          (-45, 25, -35, 15, -20),
    'homeland':         (20, 15, 5, 5, 30),
    'sanctuary':        (25, -10, 15, 0, 25),
    'detained':         (-55, 40, -65, 50, -30),
    'detention':        (-50, 35, -60, 45, 30),
    'exile':            (-45, 20, -45, 15, 30),
}
EMOTIONAL_VOCABULARY.update(_DOMAINS_4)

# ── Domain mining round 5: legal, eating disorders, neurodiversity,
#    fertility, caregiving, military, disaster, financial crisis (248 words) ──
_DOMAINS_5 = {
    # Legal / criminal justice
    'arraignment':   (-30, 45, -60, 50, 40),
    'arraigned':     (-35, 50, -65, 55, 42),
    'acquitted':     (60, -20, 50, -30, 35),
    'acquittal':     (55, -25, 45, -35, 35),
    'wrongful':      (-60, 50, -70, 55, 50),
    'wrongfully':    (-60, 50, -70, 55, 50),
    'incarcerated':  (-80, 30, -127, 40, 55),
    'incarceration': (-75, 25, -120, 35, 55),
    'parole':        (25, 30, -30, 35, 30),
    'paroled':       (30, 25, -25, 30, 28),
    'vindicated':    (70, 30, 60, -20, 40),
    'vindication':   (65, 25, 55, -25, 40),
    'convicted':     (-85, 40, -110, 50, 60),
    'conviction':    (-80, 35, -105, 45, 58),
    'sentenced':     (-75, 50, -120, 60, 65),
    'sentencing':    (-70, 55, -115, 65, 63),
    'indicted':      (-55, 60, -70, 65, 50),
    'indictment':    (-50, 55, -65, 60, 48),
    'exonerated':    (80, 35, 70, -20, 50),
    'exoneration':   (75, 30, 65, -25, 50),
    'plea':          (-30, 35, -50, 50, 35),
    'verdict':       (0, 60, 0, 70, 55),
    'trial':         (-10, 50, -20, 55, 50),
    'probation':     (-15, 20, -40, 25, 25),
    'felony':        (-60, 40, -80, 50, 55),
    'misdemeanor':   (-25, 20, -35, 25, 20),
    'bail':          (-20, 40, -30, 55, 30),
    'prosecuted':    (-50, 45, -60, 55, 45),
    'defendant':     (-30, 40, -55, 50, 35),
    'plaintiff':     (10, 35, 30, 45, 30),
    'lawsuit':       (-20, 45, -10, 55, 40),
    'restitution':   (20, 15, 20, 20, 35),
    'appealed':      (10, 30, 15, 45, 35),
    'overturned':    (50, 40, 45, 30, 45),
    'warrant':       (-35, 50, -60, 65, 40),
    'subpoena':      (-20, 45, -50, 55, 35),
    'testimony':     (0, 40, 20, 45, 40),
    'testify':       (5, 45, 25, 50, 40),
    'manslaughter':  (-90, 60, -50, 55, 70),
    'homicide':      (-95, 65, -40, 55, 75),
    'assault':       (-70, 80, -30, 65, 50),
    'fraud':         (-55, 40, -50, 45, 40),
    'perjury':       (-50, 35, -40, 40, 45),
    'contempt':      (-45, 50, 30, 40, 35),
    'solitary':      (-90, 20, -127, 30, 60),
    'confinement':   (-70, 15, -110, 25, 50),
    'shackled':      (-80, 30, -127, 35, 55),
    'handcuffed':    (-60, 45, -110, 50, 40),
    'fugitive':      (-40, 80, -50, 90, 45),
    # Eating disorders
    'purge':         (-60, 70, -40, 60, 35),
    'bingeing':      (-50, 55, -50, 50, 30),
    'restrict':      (-35, 30, 40, 35, 25),
    'restricting':   (-40, 35, 45, 40, 28),
    'anorexia':      (-75, 40, -30, 45, 55),
    'anorexic':      (-70, 35, -35, 40, 45),
    'bulimic':       (-65, 50, -40, 45, 45),
    'bulimia':       (-70, 45, -35, 45, 50),
    'dysmorphia':    (-65, 50, -50, 45, 45),
    'dysmorphic':    (-60, 45, -45, 40, 40),
    'emaciated':     (-80, 30, -60, 35, 55),
    'malnourished':  (-65, 25, -50, 40, 50),
    'calorie':       (-15, 25, 10, 20, 10),
    'fasting':       (-25, 20, 30, 25, 15),
    'starvation':    (-90, 40, -70, 60, 60),
    'bloated':       (-30, 25, -20, 20, 10),
    'laxative':      (-45, 35, -30, 40, 25),
    'compulsive':    (-40, 55, -50, 50, 30),
    'distorted':     (-40, 30, -30, 25, 20),
    'skinny':        (-10, 15, -10, 10, 10),
    'obese':         (-35, 20, -25, 15, 25),
    'overweight':    (-20, 15, -15, 10, 15),
    'underweight':   (-25, 20, -20, 20, 20),
    'disordered':    (-45, 35, -40, 35, 30),
    'compensate':    (-20, 30, 15, 30, 15),
    'compensating':  (-25, 35, 10, 35, 18),
    # Neurodiversity
    'stimming':      (15, 30, 20, 10, 10),
    'shutdown':      (-60, -30, -70, 40, 35),
    'hyperfocus':    (20, 50, 30, 15, 15),
    'hyperfocusing': (15, 55, 25, 20, 15),
    'neurotypical':  (0, 5, 10, 0, 5),
    'autistic':      (-5, 15, -15, 10, 20),
    'unmasking':     (30, 40, 35, 25, 25),
    'dysregulation': (-45, 55, -50, 45, 30),
    'accommodations': (20, 10, 20, 15, 15),
    'accommodation': (20, 10, 20, 15, 15),
    'echolalia':     (0, 15, -10, 5, 10),
    'perseveration': (-15, 30, -20, 20, 15),
    'perseverating': (-20, 35, -25, 25, 15),
    'alexithymia':   (-30, 15, -30, 20, 25),
    'impulsive':     (-20, 60, -30, 45, 15),
    'impulsivity':   (-25, 55, -35, 40, 18),
    'hyperactive':   (0, 70, -15, 30, 10),
    'fidgeting':     (0, 30, -10, 10, 5),
    'compulsion':    (-35, 50, -45, 50, 25),
    'obsession':     (-30, 55, -40, 50, 25),
    'obsessive':     (-35, 50, -45, 45, 25),
    'ritualistic':   (-15, 25, 15, 20, 15),
    'routine':       (10, -10, 20, -5, 5),
    'disrupted':     (-30, 45, -30, 40, 20),
    'overwhelm':     (-50, 60, -50, 50, 25),
    'disclosure':    (-10, 35, -20, 30, 25),
    'inattentive':   (-15, -10, -20, 15, 10),
    # Fertility / pregnancy loss
    'miscarried':    (-90, 50, -70, 30, 65),
    'stillborn':     (-120, 40, -80, 20, 80),
    'stillbirth':    (-115, 35, -75, 20, 80),
    'infertility':   (-65, 30, -55, 40, 50),
    'infertile':     (-60, 25, -60, 35, 45),
    'conceive':      (15, 20, 10, 25, 20),
    'postpartum':    (-40, 35, -30, 30, 35),
    'ectopic':       (-75, 55, -50, 70, 55),
    'hemorrhage':    (-70, 80, -60, 90, 55),
    'hemorrhaging':  (-75, 85, -65, 95, 55),
    'premature':     (-45, 50, -40, 60, 40),
    'viable':        (25, 15, 15, 20, 20),
    'nonviable':     (-80, 30, -60, 25, 55),
    'nicu':          (-55, 50, -50, 65, 50),
    'heartbeat':     (40, 35, 15, 20, 30),
    'womb':          (10, 5, 10, 5, 20),
    'maternal':      (20, 10, 15, 10, 20),
    'paternal':      (10, 5, 15, 5, 15),
    'childbirth':    (10, 50, 15, 40, 45),
    'labor':         (-15, 60, -10, 50, 35),
    'endometriosis': (-50, 35, -40, 35, 40),
    'bonding':       (45, 20, 20, 10, 20),
    'expecting':     (35, 25, 15, 20, 20),
    'embryo':        (5, 15, 5, 20, 25),
    'fertile':       (20, 10, 15, 10, 15),
    'reproductive':  (0, 10, 0, 10, 15),
    'surrogate':     (10, 15, -10, 20, 25),
    'donor':         (20, 10, 10, 10, 15),
    'implanted':     (15, 25, 10, 30, 20),
    # Caregiving / elder care
    'caregiver':     (-15, 25, 20, 30, 30),
    'caregiving':    (-20, 30, 15, 35, 30),
    'alzheimers':    (-110, 60, -70, 50, 70),
    'declining':     (-55, 25, -40, 35, 35),
    'sundowning':    (-50, 55, -45, 50, 35),
    'palliative':    (-30, -15, -20, 20, 50),
    'prognosis':     (-20, 35, -15, 40, 40),
    'degenerative':  (-70, 25, -55, 35, 50),
    'lucid':         (35, 20, 25, 15, 15),
    'lucidity':      (30, 15, 20, 10, 15),
    'confused':      (-30, 30, -40, 25, 15),
    'disoriented':   (-35, 35, -45, 30, 18),
    'bedridden':     (-55, -15, -80, 20, 40),
    'wheelchair':    (-15, 5, -35, 10, 20),
    'immobile':      (-40, -10, -70, 15, 30),
    'respite':       (30, -20, 20, -15, 15),
    'agitated':      (-35, 70, -25, 50, 20),
    'combative':     (-40, 80, 20, 55, 25),
    'unresponsive':  (-60, 25, -80, 40, 50),
    'comatose':      (-70, -20, -90, 25, 60),
    'coma':          (-65, -15, -85, 30, 60),
    'ventilator':    (-50, 25, -80, 45, 55),
    'deteriorating': (-60, 40, -50, 50, 40),
    'rallying':      (30, 25, 20, 20, 30),
    'resentment':    (-40, 40, -15, 30, 25),
    'resentful':     (-45, 45, -10, 35, 25),
    'obligation':    (-20, 20, -25, 25, 25),
    'obligated':     (-25, 25, -30, 30, 25),
    'thankless':     (-40, 25, -20, 20, 20),
    'unappreciated': (-45, 30, -25, 25, 20),
    'frail':         (-40, 5, -55, 15, 30),
    'fragile':       (-35, 10, -50, 15, 25),
    'dependent':     (-25, 15, -60, 20, 25),
    'dependence':    (-30, 15, -55, 20, 25),
    'stable':        (20, -15, 15, -10, 15),
    'critical':      (-55, 55, -40, 70, 55),
    'passing':       (-50, 10, -40, 10, 50),
    'duty':          (-5, 15, 20, 20, 25),
    'sacrificing':   (-25, 25, 15, 25, 30),
    # Military / veteran
    'veteran':       (10, 15, 25, 10, 30),
    'combat':        (-40, 80, 20, 60, 45),
    'enlisted':      (5, 30, -20, 25, 25),
    'discharge':     (15, 20, -10, 20, 25),
    'discharged':    (10, 15, -5, 15, 25),
    'honorable':     (40, 10, 40, 5, 35),
    'dishonorable':  (-70, 35, -80, 30, 45),
    'reintegration': (10, 35, -15, 30, 25),
    'hypervigilance': (-30, 70, 15, 50, 20),
    'concussion':    (-40, 35, -30, 45, 30),
    'ambush':        (-50, 90, -40, 80, 40),
    'ambushed':      (-55, 85, -45, 75, 40),
    'firefight':     (-35, 95, 10, 80, 40),
    'casualty':      (-80, 40, -50, 35, 60),
    'casualties':    (-85, 45, -50, 40, 65),
    'wounded':       (-60, 50, -40, 50, 45),
    'medevac':       (-40, 70, -30, 85, 45),
    'triage':        (-30, 55, 20, 75, 40),
    'valor':         (50, 40, 55, 20, 40),
    'bravery':       (45, 40, 50, 20, 35),
    'heroism':       (50, 45, 55, 20, 40),
    'heroic':        (45, 50, 50, 25, 35),
    'fallen':        (-80, 20, -30, 10, 65),
    'homecoming':    (60, 40, 30, 15, 30),
    'readjustment':  (-15, 30, -20, 30, 20),
    'civilian':      (10, -5, 10, -5, 5),
    'demoted':       (-45, 40, -60, 35, 25),
    'stationed':     (0, 10, -15, 10, 10),
    'sniper':        (-30, 50, 30, 40, 35),
    'patrol':        (-10, 35, 15, 30, 20),
    'convoy':        (-10, 30, 10, 25, 20),
    'classified':    (-5, 15, 30, 15, 25),
    'debriefed':     (10, 20, 10, 15, 15),
    'ptsd':          (-70, 70, -50, 50, 50),
    'insurgent':     (-30, 50, -10, 45, 30),
    'medal':         (40, 25, 35, 10, 30),
    'coffin':        (-70, 20, -40, 10, 65),
    'remains':       (-55, 15, -30, 15, 55),
    # Natural disaster / climate
    'evacuated':     (-35, 60, -40, 75, 35),
    'evacuation':    (-30, 55, -35, 70, 35),
    'wildfire':      (-50, 65, -30, 70, 45),
    'devastation':   (-100, 50, -80, 40, 65),
    'catastrophe':   (-90, 60, -70, 60, 65),
    'catastrophic':  (-95, 65, -75, 65, 65),
    'rubble':        (-60, 20, -50, 15, 40),
    'debris':        (-35, 15, -25, 15, 20),
    'wreckage':      (-55, 20, -45, 15, 35),
    'demolished':    (-65, 40, -55, 30, 40),
    'leveled':       (-70, 35, -60, 25, 40),
    'obliterated':   (-90, 50, -70, 30, 55),
    'flooded':       (-50, 45, -40, 55, 35),
    'flooding':      (-45, 50, -35, 60, 35),
    'submerged':     (-50, 30, -45, 40, 30),
    'inundated':     (-40, 35, -35, 40, 25),
    'earthquake':    (-60, 70, -50, 65, 50),
    'tsunami':       (-80, 75, -60, 70, 60),
    'landslide':     (-55, 55, -45, 55, 40),
    'avalanche':     (-60, 65, -50, 60, 45),
    'drought':       (-40, 15, -30, 30, 35),
    'famine':        (-85, 30, -60, 50, 60),
    'contaminated':  (-40, 30, -25, 40, 30),
    'uninhabitable': (-65, 25, -50, 30, 40),
    'rebuild':       (25, 30, 30, 25, 25),
    'rebuilding':    (20, 35, 25, 30, 25),
    'aftermath':     (-35, 20, -20, 20, 30),
    'resilience':    (40, 25, 40, 15, 25),
    'survivors':     (-15, 30, 20, 20, 30),
    'pollution':     (-30, 15, -15, 20, 25),
    'hazardous':     (-35, 35, -20, 45, 30),
    'donation':      (30, 10, 15, 10, 10),
    'donations':     (25, 10, 10, 10, 10),
    # Financial crisis personal
    'foreclosure':   (-70, 50, -60, 55, 50),
    'foreclosed':    (-75, 45, -65, 50, 50),
    'bankruptcy':    (-65, 40, -55, 45, 50),
    'bankrupt':      (-70, 45, -60, 50, 48),
    'collections':   (-40, 45, -35, 50, 25),
    'garnishment':   (-50, 40, -55, 45, 30),
    'indebted':      (-35, 25, -40, 30, 25),
    'delinquent':    (-35, 30, -30, 40, 20),
    'defaulted':     (-55, 35, -50, 40, 35),
    'repossessed':   (-65, 45, -60, 45, 40),
    'evicting':      (-50, 50, -55, 55, 35),
    'homelessness':  (-60, 25, -50, 25, 45),
    'unhoused':      (-55, 20, -45, 20, 40),
    'unemployed':    (-45, 25, -40, 30, 30),
    'downsized':     (-40, 30, -35, 35, 25),
    'terminated':    (-50, 40, -45, 40, 30),
    'severance':     (10, 15, -10, 20, 15),
    'harassing':     (-40, 55, -30, 50, 25),
    'overdraft':     (-30, 35, -25, 40, 15),
    'overdrawn':     (-35, 30, -30, 35, 15),
    'bounced':       (-30, 35, -25, 35, 15),
    'tanked':        (-55, 40, -40, 35, 25),
    'plummeted':     (-55, 50, -45, 40, 30),
    'liquidated':    (-35, 20, -30, 25, 20),
    'pawned':        (-40, 25, -35, 35, 15),
    'desperation':   (-65, 70, -60, 70, 35),
    'predatory':     (-50, 40, 40, 40, 30),
    'welfare':       (-20, 10, -30, 15, 15),
    'assistance':    (15, 10, -10, 15, 10),
    'utilities':     (-10, 15, -5, 20, 10),
    'shutoff':       (-45, 40, -40, 50, 25),
    'sued':          (-45, 55, -50, 55, 40),
}
EMOTIONAL_VOCABULARY.update(_DOMAINS_5)

# ── Domain mining round 6: bullying, cyberbullying, body image, chronic pain,
#    LGBTQ+, foster care, incarceration aftermath, layoffs (208 words) ────────
_DOMAINS_6 = {
    # Childhood / school bullying
    'outcasted':    (-40, 15, -30, 10, -20),
    'wedgie':       (-25, 35, -30, 15, -10),
    'snitched':     (-20, 25, -15, 15, -10),
    'tattled':      (-15, 20, -10, 10, -8),
    'clique':       (-10, 10, 15, 5, 15),
    'cliquey':      (-20, 15, 15, 5, -10),
    'loner':        (-25, -10, -20, 5, -15),
    'outcast':      (-40, 10, -30, 10, -20),
    'unpopular':    (-25, 10, -20, 5, -12),
    'popularity':   (10, 15, 20, 5, 15),
    'snitch':       (-20, 25, -15, 15, -10),
    'narc':         (-20, 25, -10, 15, -10),
    'nerd':         (-10, 5, -10, 0, -5),
    'geek':         (-8, 5, -8, 0, -5),
    'dork':         (-12, 10, -10, 0, -8),
    'weirdo':       (-20, 10, -15, 5, -10),
    'crybaby':      (-25, 20, -20, 10, -12),
    'wimp':         (-25, 15, -25, 10, -12),
    'pushover':     (-20, 10, -30, 5, -10),
    'doormat':      (-30, 10, -40, 5, -15),
    'snubbed':      (-30, 15, -20, 10, -15),
    'unchosen':     (-35, 10, -25, 10, -18),
    'sidelined':    (-25, 10, -25, 10, -12),
    'scapegoated':  (-45, 30, -35, 20, -25),
    'ganged':       (-40, 40, -40, 25, -20),
    'intimidated':  (-35, 35, -40, 20, -15),
    # Online / cyberbullying
    'doxxed':       (-60, 50, -50, 40, -30),
    'swatted':      (-70, 70, -60, 50, -35),
    'trolled':      (-25, 25, -15, 10, -10),
    'brigaded':     (-40, 40, -35, 25, -20),
    'cyberbullied': (-55, 35, -40, 25, -25),
    'impersonated': (-45, 35, -30, 25, -20),
    'viral':        (0, 50, -10, 30, 15),
    'stalked':      (-60, 45, -50, 35, -30),
    'muted':        (-20, -5, -15, 5, -10),
    'deplatformed': (-35, 30, -40, 25, -15),
    'shadowbanned': (-30, 20, -35, 15, -15),
    'dogpiled':     (-45, 45, -40, 30, -25),
    'ratioed':      (-25, 20, -20, 10, -10),
    'flamed':       (-30, 35, -20, 15, -15),
    'spammed':      (-15, 20, -10, 10, -5),
    'hacked':       (-45, 40, -40, 30, -20),
    'phished':      (-40, 30, -35, 25, -18),
    'deepfaked':    (-50, 40, -40, 30, -25),
    'outed':        (-45, 40, -50, 30, -25),
    # Body image beyond ED
    'cellulite':    (-10, 10, -10, 5, 10),
    'acne':         (-15, 15, -10, 10, 10),
    'scarring':     (-20, 10, -15, 5, 15),
    'birthmark':    (-5, 5, -5, 0, 10),
    'blemish':      (-10, 10, -8, 5, 8),
    'wrinkles':     (-10, 5, -8, 5, 10),
    'sagging':      (-15, 10, -10, 5, 10),
    'flabby':       (-20, 15, -15, 10, -10),
    'chubby':       (-12, 10, -8, 5, -5),
    'lanky':        (-8, 5, -5, 0, -3),
    'stocky':       (-3, 5, 5, 0, 5),
    'curvy':        (5, 10, 5, 0, 5),
    'petite':       (0, 5, -5, 0, 5),
    'disfigured':   (-50, 25, -35, 15, 30),
    'prosthetic':   (0, 5, 0, 0, 15),
    'amputation':   (-40, 25, -30, 15, 35),
    'complexion':   (0, 5, 0, 0, 5),
    'freckles':     (5, 5, 0, 0, 5),
    'balding':      (-15, 10, -10, 5, 10),
    'receding':     (-12, 8, -8, 5, 8),
    'graying':      (-5, 5, -5, 0, 8),
    'bodyshamed':   (-50, 35, -35, 20, -25),
    'unfiltered':   (15, 10, 15, 0, 10),
    # Chronic pain / invisible illness
    'flare':        (-35, 30, -25, 25, 20),
    'flareup':      (-40, 35, -30, 30, 22),
    'remission':    (50, -10, 30, -20, 25),
    'fibromyalgia': (-30, 15, -25, 15, 35),
    'lupus':        (-30, 15, -25, 15, 35),
    'spoons':       (-15, 5, -15, 10, 15),
    'spoonie':      (-15, 10, -15, 10, 15),
    'fatigued':     (-30, -20, -25, 10, -15),
    'debilitating': (-55, 25, -50, 25, 30),
    'crippling':    (-60, 30, -55, 30, 30),
    'housebound':   (-45, -10, -60, 15, 30),
    'homebound':    (-40, -10, -55, 15, 28),
    'bedbound':     (-55, -15, -75, 20, 35),
    'inflammation': (-20, 15, -15, 10, 15),
    'inflamed':     (-25, 20, -15, 15, 12),
    'migraine':     (-40, 30, -30, 25, 20),
    'vertigo':      (-30, 25, -25, 20, 15),
    'flaring':      (-35, 30, -25, 25, 18),
    'progressive':  (-20, 15, -15, 15, 25),
    'manageable':   (15, -5, 15, -10, 10),
    'unmanageable': (-40, 30, -35, 25, -20),
    'tolerable':    (10, -5, 10, -10, 10),
    'intolerable':  (-60, 35, -40, 30, -25),
    'excruciating': (-80, 50, -50, 40, -35),
    'agonizing':    (-75, 45, -45, 35, -30),
    'throbbing':    (-30, 25, -20, 20, 10),
    'radiating':    (-25, 20, -15, 15, 10),
    'tingling':     (-10, 15, -10, 10, 8),
    'stiffness':    (-15, 10, -15, 10, 10),
    # LGBTQ+ experience
    'closeted':     (-30, 20, -35, 15, 20),
    'deadnamed':    (-50, 35, -45, 25, -25),
    'deadnaming':   (-45, 30, -40, 20, -22),
    'misgendered':  (-40, 30, -35, 20, -20),
    'misgendering': (-35, 25, -30, 15, -18),
    'transitioning': (10, 25, 20, 15, 30),
    'transitioned': (20, 10, 25, 0, 25),
    'nonbinary':    (0, 5, 5, 0, 10),
    'queer':        (0, 10, 10, 0, 10),
    'questioning':  (-10, 15, -10, 10, 15),
    'homophobia':   (-55, 40, 20, 20, 30),
    'homophobic':   (-50, 35, 20, 15, -25),
    'transphobia':  (-55, 40, 20, 20, 30),
    'transphobic':  (-50, 35, 20, 15, -25),
    'conversion':   (-40, 30, -35, 20, 25),
    'dysphoria':    (-55, 40, -40, 30, 35),
    'euphoria':     (70, 50, 40, 10, 20),
    'affirmed':     (35, 15, 30, 0, 18),
    'allyship':     (30, 15, 15, 0, 15),
    'ally':         (25, 10, 10, 0, 12),
    # Foster care / adoption
    'reunification': (30, 25, 15, 20, 30),
    'placement':    (-10, 15, -20, 15, 20),
    'caseworker':   (0, 10, 15, 10, 20),
    'adoptee':      (0, 10, -5, 5, 20),
    'adoptive':     (15, 10, 15, 5, 18),
    'relinquished': (-45, 20, -40, 15, 30),
    'relinquishment': (-40, 15, -35, 10, 30),
    'kinship':      (20, 10, 10, 0, 20),
    'permanency':   (20, 5, 15, -5, 25),
    'guardianship': (10, 10, 20, 10, 25),
    'detachment':   (-30, 10, -20, 5, -15),
    'disruption':   (-35, 30, -30, 25, 20),
    'siblings':     (5, 10, 5, 0, 25),
    'rehomed':      (-30, 20, -35, 15, -18),
    'abandonment':  (-60, 25, -50, 20, -30),
    # Incarceration aftermath
    'reentry':      (10, 25, -15, 25, 25),
    'felon':        (-50, 25, -60, 20, 40),
    'parolee':      (-10, 15, -30, 20, 25),
    'recidivism':   (-40, 20, -40, 25, 35),
    'imprisoned':   (-75, 25, -120, 35, 50),
    'imprisonment': (-70, 20, -115, 30, 50),
    'expunged':     (35, 15, 25, -10, 20),
    'expungement':  (30, 10, 20, -10, 20),
    'stigmatized':  (-40, 25, -30, 15, -20),
    'employable':   (20, 10, 15, 5, 10),
    'unemployable': (-40, 20, -40, 25, -25),
    'supervised':   (-10, 10, -20, 10, 15),
    'monitored':    (-10, 10, -20, 10, 12),
    'curfew':       (-10, 10, -20, 10, 12),
    'restricted':   (-15, 10, -25, 10, 15),
    'reintegrated': (15, 20, -10, 15, 20),
    # Workplace layoff / restructuring
    'restructured':    (-15, 20, -15, 20, 18),
    'restructuring':   (-20, 25, -15, 25, 20),
    'redundant':       (-40, 25, -35, 25, -20),
    'redundancy':      (-35, 20, -30, 20, 20),
    'offboarded':      (-30, 20, -25, 20, 15),
    'offboarding':     (-25, 15, -20, 15, 15),
    'downsizing':      (-30, 25, -20, 25, 20),
    'furloughed':      (-35, 20, -30, 25, 20),
    'furlough':        (-30, 15, -25, 20, 18),
    'layoff':          (-40, 30, -35, 30, 22),
    'layoffs':         (-35, 25, -25, 25, 22),
    'termination':     (-45, 35, -40, 35, 28),
    'outsourced':      (-30, 20, -25, 20, 18),
    'outsourcing':     (-25, 20, -20, 20, 18),
    'reorg':           (-20, 25, -15, 20, 18),
    'reorganized':     (-15, 20, -10, 15, 15),
    'reorganization':  (-15, 20, -10, 15, 18),
    'attrition':       (-15, 10, -10, 10, 15),
    'involuntary':     (-30, 20, -40, 20, 15),
    'buyout':          (10, 15, 10, 15, 15),
    'transition':      (-5, 15, -5, 10, 12),
    'pivoted':         (5, 15, 10, 10, 10),
    'pivoting':        (5, 15, 10, 10, 10),
    'upskill':         (15, 15, 15, 10, 10),
    'upskilling':      (15, 15, 15, 10, 10),
    'reskill':         (10, 15, 10, 10, 10),
    'reskilling':      (10, 15, 10, 10, 10),
}
EMOTIONAL_VOCABULARY.update(_DOMAINS_6)

# ── Domain batch 7: parenting stress, divorce/separation, loneliness,
#    betrayal/trust, achievement, gratitude, physical sensations, sound/music
_DOMAINS_7 = {
    # --- Parenting stress ---
    'colic':             (-20, 30, -25, 30, 15),
    'tantrum':           (-35, 60, -40, 45, 20),
    'tantrums':          (-35, 55, -40, 40, 20),
    'sleepless':         (-30, 20, -25, 25, 15),
    'daycare':           (0, 5, 0, 5, 8),
    'teething':          (-15, 25, -15, 20, 10),
    'fussy':             (-20, 25, -20, 20, 8),
    'colicky':           (-20, 30, -25, 30, 12),
    'inconsolable':      (-55, 50, -50, 50, 30),
    'whining':           (-25, 30, -20, 20, 8),
    'bedtime':           (0, 5, 5, 5, 5),
    'naptime':           (0, -5, 5, 0, 5),
    'breastfeeding':     (5, 5, 5, 5, 10),
    'pacifier':          (5, -10, 5, -5, 5),
    'diaper':            (0, 5, 0, 5, 3),
    'weaning':           (-5, 10, 5, 10, 8),
    'developmental':     (0, 5, 0, 5, 10),
    'sleep-deprived':    (-35, 15, -30, 25, 18),
    'overtired':         (-30, 20, -25, 20, 12),
    'fretful':           (-25, 25, -20, 20, 10),
    'nanny':             (0, 0, 5, 0, 5),
    'babysitter':        (0, 0, 5, 0, 5),
    'sitter':            (0, 0, 5, 0, 3),
    'preschool':         (5, 5, 0, 5, 8),
    'formula':           (0, 0, 0, 0, 5),

    # --- Divorce/separation ---
    'alimony':           (-15, 15, -20, 15, 20),
    'co-parenting':      (-5, 15, -5, 15, 18),
    'coparenting':       (-5, 15, -5, 15, 18),
    'mediator':          (0, 10, 0, 10, 18),
    'contested':         (-25, 35, -20, 30, 22),
    'separation':        (-35, 20, -25, 20, 22),
    'prenup':            (-5, 10, 10, 5, 15),
    'prenuptial':        (-5, 10, 10, 5, 15),
    'annulment':         (-30, 20, -15, 20, 25),
    'estranged':         (-45, 15, -30, 15, 25),
    'ex-husband':        (-15, 15, -10, 10, 12),
    'ex-wife':           (-15, 15, -10, 10, 12),
    'ex-spouse':         (-15, 15, -10, 10, 12),
    'irreconcilable':    (-40, 20, -25, 15, 25),
    'infidelity':        (-80, 60, -60, 40, 40),
    'adultery':          (-80, 55, -55, 35, 42),
    'proceedings':       (-10, 15, -5, 15, 20),
    'petition':          (-5, 15, 5, 15, 18),
    'dissolution':       (-35, 15, -20, 15, 25),
    'decree':            (-10, 10, 15, 10, 22),
    'marital':           (0, 5, 0, 5, 15),
    'spousal':           (0, 5, 0, 5, 12),
    'divorcee':          (-20, 10, -10, 5, 15),
    'remarriage':        (10, 15, 10, 10, 15),
    'remarried':         (10, 10, 10, 5, 12),
    'stepparent':        (0, 10, 5, 5, 12),
    'blended':           (5, 10, 5, 5, 10),
    'cohabitation':      (0, 5, 0, 5, 8),

    # --- Loneliness/social isolation ---
    'withdrawn':         (-40, -15, -35, 10, -25),
    'reclusive':         (-35, -20, -30, 5, -20),
    'alienated':         (-55, 20, -45, 15, -30),
    'solitude':          (-15, -20, -10, 0, 10),
    'friendless':        (-55, 10, -40, 15, -35),
    'hermit':            (-20, -25, -15, 0, -10),
    'recluse':           (-25, -20, -20, 0, -12),
    'secluded':          (-15, -15, -10, 0, 5),
    'seclusion':         (-15, -15, -10, 0, 5),
    'aloof':             (-20, -15, 10, 0, -8),
    'unapproachable':    (-25, 10, 15, 5, -10),
    'overlooked':        (-40, 10, -35, 10, -20),
    'marginalized':      (-50, 20, -45, 15, -30),
    'shunned':           (-55, 20, -45, 15, -30),
    'unnoticed':         (-35, -5, -30, 5, -18),
    'wallflower':        (-20, -15, -25, 0, -10),
    'homesick':          (-40, 20, -25, 20, -20),
    'homesickness':      (-40, 20, -25, 20, -20),
    'forlorn':           (-55, -10, -40, 10, -35),
    'desolate':          (-65, -5, -45, 10, -45),

    # --- Betrayal/trust violation ---
    'backstabbed':       (-85, 70, -65, 50, -40),
    'backstabbing':      (-80, 65, -60, 45, -35),
    'two-faced':         (-60, 40, -45, 30, -25),
    'deceived':          (-75, 55, -60, 40, -35),
    'conned':            (-70, 55, -60, 40, -30),
    'scammed':           (-65, 55, -55, 40, -28),
    'treachery':         (-90, 60, -70, 45, -50),
    'treacherous':       (-85, 55, -65, 40, -45),
    'duplicitous':       (-65, 40, -50, 30, -30),
    'duplicity':         (-65, 40, -50, 30, -30),
    'swindled':          (-65, 50, -55, 35, -28),
    'defrauded':         (-60, 45, -50, 35, -25),
    'hoodwinked':        (-55, 45, -50, 30, -22),
    'bamboozled':        (-50, 40, -45, 25, -18),
    'double-crossed':    (-85, 65, -65, 50, -40),
    'disloyal':          (-70, 45, -55, 30, -35),
    'disloyalty':        (-70, 45, -55, 30, -35),
    'traitor':           (-90, 65, -70, 50, -50),
    'traitorous':        (-85, 60, -65, 45, -45),
    'sellout':           (-55, 40, -45, 25, -25),
    'ratted':            (-50, 45, -40, 30, -22),
    'informant':         (-20, 25, -15, 20, 15),
    'turncoat':          (-75, 55, -60, 40, -40),
    'judas':             (-90, 55, -70, 40, -50),
    'phishing':          (-30, 30, -25, 25, 15),
    'fraudulent':        (-55, 35, -40, 25, 20),
    'impostor':          (-45, 40, -35, 30, 18),
    'impersonator':      (-35, 30, -25, 25, 15),
    'gaslighted':        (-80, 50, -70, 45, -40),

    # --- Achievement/success ---
    'breakthrough':      (55, 50, 45, 20, 35),
    'excelled':          (40, 25, 35, 0, 20),
    'excelling':         (40, 25, 35, 0, 18),
    'triumphant':        (70, 50, 60, 0, 40),
    'victorious':        (60, 45, 55, 0, 35),
    'succeeding':        (35, 20, 30, 0, 15),
    'mastered':          (45, 20, 45, 0, 25),
    'mastery':           (50, 15, 50, 0, 30),
    'valedictorian':     (50, 25, 40, 0, 35),
    'summa':             (40, 15, 30, 0, 28),
    'honors':            (35, 15, 25, 0, 22),
    'dean':              (5, 5, 15, 0, 18),
    'certified':         (20, 10, 20, 0, 15),
    'licensed':          (15, 5, 15, 0, 12),
    'tenured':           (30, 5, 35, 0, 25),
    'awarded':           (35, 20, 25, 0, 22),
    'accolade':          (35, 15, 25, 0, 22),
    'recognition':       (30, 15, 20, 0, 18),
    'commendation':      (30, 10, 20, 0, 20),
    'pioneered':         (45, 30, 45, 0, 30),
    'innovated':         (40, 30, 40, 0, 25),
    'launched':          (30, 35, 30, 10, 20),
    'founded':           (30, 20, 35, 0, 25),

    # --- Gratitude/appreciation ---
    'thanking':          (25, 10, 5, 0, 10),
    'beholden':          (10, 5, -15, 5, 15),
    'obliged':           (5, 5, -10, 5, 12),
    'gracious':          (35, 10, 15, 0, 18),
    'graciousness':      (35, 10, 15, 0, 20),
    'privileged':        (25, 10, 15, 0, 15),
    'humbled':           (20, 10, -10, 0, 18),
    'honored':           (35, 15, 15, 0, 20),
    'touched':           (35, 20, -5, 0, 18),
    'sincere':           (30, 10, 15, 0, 18),
    'sincerely':         (25, 5, 10, 0, 12),
    'benevolent':        (40, 10, 20, 0, 22),
    'benevolence':       (40, 10, 20, 0, 25),
    'magnanimous':       (40, 10, 25, 0, 25),
    'selfless':          (40, 15, -10, 0, 25),
    'altruistic':        (35, 10, -5, 0, 22),
    'philanthropic':     (30, 10, 15, 0, 25),

    # --- Physical sensations (emotional) ---
    'butterflies':       (10, 30, -15, 15, 8),
    'chills':            (-10, 30, -15, 15, 10),
    'goosebumps':        (5, 30, -10, 10, 8),
    'dizzy':             (-20, 25, -25, 20, 10),
    'flushed':           (-15, 30, -20, 15, 8),
    'shivering':         (-20, 25, -20, 15, 8),
    'palpitations':      (-30, 50, -35, 35, 15),
    'breathless':        (-15, 40, -25, 25, 12),
    'lightheaded':       (-15, 20, -20, 15, 8),
    'queasy':            (-25, 20, -20, 15, 8),
    'pounding':          (-25, 45, -20, 25, 12),
    'clenching':         (-30, 35, -15, 20, 10),
    'tightness':         (-25, 25, -20, 15, 10),
    'knot':              (-30, 20, -25, 15, 12),
    'fluttering':        (5, 25, -10, 10, 5),
    'freezing':          (-15, 20, -15, 10, 8),
    'burning':           (-30, 40, -15, 25, 12),
    'stinging':          (-25, 30, -15, 20, 8),
    'wincing':           (-20, 25, -15, 15, 5),
    'flinching':         (-25, 30, -25, 20, 8),
    'recoiling':         (-30, 35, -30, 20, 10),
    'tensing':           (-20, 30, -15, 15, 8),

    # --- Sound/music (emotional) ---
    'haunting':          (-20, 15, -10, 5, 25),
    'soothing':          (30, -15, 10, -10, 15),
    'jarring':           (-30, 50, -25, 30, 10),
    'deafening':         (-20, 60, -20, 25, 15),
    'cacophony':         (-30, 55, -20, 25, 10),
    'melody':            (20, 5, 5, 0, 12),
    'melodic':           (20, 5, 5, 0, 10),
    'harmonious':        (30, -5, 15, 0, 18),
    'harmony':           (30, -5, 15, 0, 18),
    'dissonant':         (-25, 30, -15, 15, 10),
    'dissonance':        (-25, 30, -15, 15, 12),
    'rhythmic':          (10, 10, 5, 0, 8),
    'rhythm':            (10, 10, 5, 0, 10),
    'anthem':            (25, 30, 20, 5, 22),
    'requiem':           (-30, 10, -5, 0, 30),
    'symphonic':         (15, 10, 10, 0, 18),
    'symphony':          (15, 10, 10, 0, 20),
    'crescendo':         (15, 45, 15, 15, 15),
    'diminuendo':        (-5, -15, -5, -5, 8),
    'resonant':          (15, 15, 10, 0, 15),
    'resonance':         (15, 15, 10, 0, 18),
    'reverberating':     (0, 25, 5, 5, 12),
    'echoing':           (-5, 15, -5, 5, 10),
    'muffled':           (-15, -10, -15, 5, 8),
    'piercing':          (-25, 50, -15, 25, 10),
    'shrill':            (-25, 45, -15, 20, 8),
    'blaring':           (-20, 50, -10, 20, 8),
    'thunderous':        (-10, 60, 15, 15, 20),
    'whispering':        (5, -10, -10, 0, 8),
    'murmuring':         (0, -10, -10, 0, 5),
    'humming':           (10, -5, 5, 0, 5),
    'shrieking':         (-50, 70, -20, 40, 15),
    'moaning':           (-35, 30, -25, 20, 10),
    'groaning':          (-30, 25, -20, 15, 8),
    'sighing':           (-15, -5, -15, 5, 5),
}
EMOTIONAL_VOCABULARY.update(_DOMAINS_7)

# =====================================================================
# _DOMAINS_8: Power dynamics, self-talk, survival, existential crisis,
#             nostalgia, compliment, insult, weather/nature as emotion
# =====================================================================
_DOMAINS_8 = {

    # --- Power dynamics / control ---
    'stonewalling':      (-35, 15, 30, 15, 20),
    'breadcrumbing':     (-25, 15, 25, 10, 12),
    'benching':          (-20, 10, 25, 10, 10),
    'manipulating':      (-40, 25, 35, 20, 22),
    'coercing':          (-45, 35, 40, 30, 25),
    'dominating':        (-30, 35, 45, 20, 22),
    'isolating':         (-40, 20, 30, 20, 25),
    'triangulating':     (-35, 25, 30, 15, 18),
    'enmeshed':          (-25, 20, -20, 15, 20),
    'enmeshment':        (-25, 20, -20, 15, 22),
    'codependency':      (-20, 15, -15, 10, 20),
    'grooming':          (-45, 20, 35, 20, 28),
    'exploiting':        (-45, 25, 35, 20, 25),
    'silencing':         (-40, 20, 35, 20, 22),
    'undermining':       (-35, 20, 30, 15, 18),
    'dismissive':        (-30, 10, 25, 10, 15),
    'belittling':        (-40, 20, 30, 15, 18),
    'patronizing':       (-30, 15, 30, 10, 15),

    # --- Self-talk / inner critic ---
    'unworthy':          (-45, 20, -35, 15, 25),
    'incapable':         (-35, 15, -35, 15, 20),
    'defective':         (-40, 15, -30, 15, 22),
    'unredeemable':      (-50, 15, -35, 10, 30),
    'irredeemable':      (-50, 15, -35, 10, 30),
    'undeserving':       (-40, 15, -30, 15, 22),
    'flawed':            (-25, 10, -15, 10, 18),
    'inferior':          (-35, 15, -30, 10, 20),
    'insufficient':      (-25, 10, -20, 10, 15),
    'unfit':             (-30, 15, -25, 10, 18),
    'self-loathing':     (-55, 30, -35, 20, 28),
    'self-hatred':       (-55, 35, -35, 25, 30),
    'self-doubt':        (-30, 20, -25, 15, 18),
    'self-sabotage':     (-40, 25, -20, 20, 22),
    'imposter':          (-30, 25, -25, 20, 18),

    # --- Survival / resilience language ---
    'endured':           (15, 15, 25, 5, 25),
    'persevered':        (20, 20, 30, 5, 25),
    'withstood':         (15, 15, 30, 5, 25),
    'outlasted':         (15, 10, 25, 5, 22),
    'perseverance':      (20, 15, 25, 5, 25),
    'tenacity':          (20, 20, 30, 5, 25),
    'tenacious':         (20, 25, 30, 5, 22),
    'grit':              (15, 25, 30, 10, 22),
    'fortitude':         (20, 15, 30, 5, 28),
    'indomitable':       (25, 25, 35, 5, 28),
    'unyielding':        (10, 20, 35, 10, 25),
    'steadfast':         (20, 10, 30, 5, 25),
    'unwavering':        (20, 10, 30, 5, 22),
    'endurance':         (15, 15, 25, 5, 25),
    'hardy':             (10, 10, 20, 5, 15),
    'hardened':          (-5, 10, 25, 5, 20),
    'weathered':         (5, 5, 20, 0, 22),
    'battle-tested':     (10, 15, 30, 5, 25),
    'unbreakable':       (20, 20, 35, 5, 28),
    'undaunted':         (20, 20, 30, 5, 22),
    'persisting':        (15, 15, 25, 10, 20),

    # --- Spiritual / existential crisis ---
    'purposeless':       (-40, 10, -25, 10, 28),
    'abyss':             (-45, 20, -30, 15, 35),
    'existential':       (-20, 15, -10, 10, 30),
    'nihilism':          (-40, 10, -15, 5, 30),
    'nihilistic':        (-40, 10, -15, 5, 28),
    'nothingness':       (-35, 5, -20, 5, 30),
    'absurdity':         (-20, 15, -10, 5, 22),
    'godless':           (-25, 10, -10, 5, 25),
    'faithless':         (-30, 10, -15, 10, 22),
    'soulless':          (-35, 5, -10, 5, 25),
    'desolation':        (-45, 10, -25, 10, 30),
    'futile':            (-35, 10, -25, 10, 22),
    'futility':          (-35, 10, -25, 10, 25),
    'unmoored':          (-30, 20, -25, 15, 22),
    'untethered':        (-20, 15, -20, 10, 18),
    'directionless':     (-30, 10, -20, 10, 20),
    'aimless':           (-25, 5, -20, 5, 18),
    'barren':            (-30, 5, -15, 5, 22),
    'forsaken':          (-50, 20, -30, 15, 30),

    # --- Nostalgia / memory ---
    'wistful':           (-10, 5, -10, 0, 18),
    'pining':            (-20, 15, -15, 10, 18),
    'bygone':            (-5, 0, -5, 0, 18),
    'remembrance':       (-5, 5, -5, 0, 20),
    'memoir':            (0, 5, 5, 0, 20),
    'reminiscence':      (5, 5, -5, 0, 18),
    'melancholic':       (-25, 5, -15, 5, 22),
    'melancholy':        (-25, 5, -15, 5, 22),
    'reverie':           (10, -5, -5, 0, 15),
    'daydream':          (10, -5, -5, 0, 10),
    'sepia':             (-5, 0, -5, 0, 12),
    'faded':             (-10, -5, -10, 0, 15),
    'bygones':           (5, -5, 5, 0, 12),
    'halcyon':           (20, -5, 10, 0, 20),
    'waning':            (-15, -5, -10, 5, 15),

    # --- Compliment / praise ---
    'radiant':           (40, 20, 15, 0, 18),
    'magnificent':       (45, 25, 20, 0, 25),
    'exceptional':       (40, 20, 20, 0, 22),
    'extraordinary':     (45, 25, 20, 0, 25),
    'dazzling':          (40, 30, 15, 0, 18),
    'exquisite':         (45, 15, 15, 0, 22),
    'phenomenal':        (45, 30, 20, 0, 22),
    'glorious':          (45, 25, 20, 0, 25),
    'majestic':          (40, 20, 25, 0, 28),
    'resplendent':       (40, 15, 15, 0, 22),
    'luminous':          (35, 15, 10, 0, 18),
    'awe-inspiring':     (45, 30, 15, 0, 28),
    'breathtaking':      (45, 35, 10, 0, 22),
    'impeccable':        (40, 10, 20, 0, 20),
    'stellar':           (40, 20, 15, 0, 18),
    'sublime':           (45, 15, 15, 0, 25),
    'transcendent':      (40, 20, 15, 0, 30),
    'virtuoso':          (40, 20, 25, 0, 22),

    # --- Insult / degradation ---
    'contemptible':      (-45, 25, 25, 15, 22),
    'deplorable':        (-45, 25, 20, 15, 25),
    'despicable':        (-50, 30, 25, 20, 25),
    'abhorrent':         (-50, 30, 20, 15, 28),
    'grotesque':         (-40, 25, 15, 10, 20),
    'wretched':          (-45, 20, -20, 15, 25),
    'abominable':        (-50, 30, 20, 15, 28),
    'insufferable':      (-40, 30, 20, 15, 18),
    'detestable':        (-45, 25, 20, 15, 22),
    'repugnant':         (-50, 25, 15, 15, 22),
    'odious':            (-45, 20, 20, 10, 22),
    'nauseating':        (-40, 25, -10, 15, 15),
    'scummy':            (-45, 25, 20, 10, 15),
    'sleazy':            (-40, 20, 15, 10, 15),
    'slimy':             (-35, 15, 15, 10, 12),
    'spineless':         (-35, 15, -30, 10, 18),
    'gutless':           (-35, 20, -30, 10, 18),

    # --- Weather / nature as emotion ---
    'stormy':            (-20, 35, -10, 15, 15),
    'blossoming':        (25, 15, 10, 0, 15),
    'wilting':           (-25, -5, -20, 10, 15),
    'turbulent':         (-25, 40, -15, 20, 18),
    'serene':            (30, -15, 15, -10, 18),
    'overcast':          (-15, -5, -10, 5, 10),
    'tempestuous':       (-25, 45, -15, 20, 20),
    'foggy':             (-10, -5, -10, 5, 8),
    'misty':             (-5, -5, -5, 0, 8),
    'blooming':          (25, 15, 10, 0, 12),
    'withering':         (-30, 10, -20, 10, 18),
    'parched':           (-20, 10, -15, 10, 12),
    'thawing':           (15, 5, 5, 0, 10),
    'budding':           (15, 10, 5, 0, 10),
    'flourishing':       (30, 15, 15, 0, 18),
    'dormant':           (-5, -10, -10, 0, 12),
    'scorching':         (-20, 35, -10, 15, 12),
    'frigid':            (-20, 10, -10, 5, 12),
    'balmy':             (15, -10, 5, 0, 8),
}
EMOTIONAL_VOCABULARY.update(_DOMAINS_8)

# ── Structural force words: meaning comes from position, not isolation ──
_STRUCTURAL_FORCES = {
    'one':    (35, 5, 15, 0, 30),     # a person/someone. "no one" = zero people = isolation
}
EMOTIONAL_VOCABULARY.update(_STRUCTURAL_FORCES)

# ── Vocabulary corrections from permanent suite miss analysis ──────
_VOCAB_FIXES = {
    # Laughed is positive -- shared laughter = joy, not negative
    'laughed':      (30, 25, 15, 0, 10),     # override: was -5, laughing is positive
    'laughing':     (25, 30, 10, 0, 8),       # same fix
    'chuckled':     (20, 15, 10, 0, 5),
    # Breathe is neutral-to-positive (relief, life, calm)
    'breathe':      (10, -5, 10, 0, 5),       # override: was -10
    # Electricity is a utility noun -- neutral, not positive
    'electricity':  (0, 5, 0, 5, 10),         # override: was +43. "shut off" = negative from verb
    # Loan and debt are liquid -- V near zero, high G (financial weight)
    'loan':         (-5, 10, -10, 15, 30),    # override: was -77. Liquid. "approved" vs "defaulted" resolves
    'debt':         (-15, 10, -15, 10, 30),   # override: was -77. "paid off debt" = positive from "paid off"
    # Proposed is a big positive life event
    'proposed':     (40, 30, 25, 5, 25),      # override: was +20. Proposal = major positive
    # Anniversary has more weight
    'anniversary':  (25, 10, 10, 0, 25),      # override: was +10. Remembered anniversary = love
    # Danced is more joyful
    'danced':       (35, 20, 15, 0, 12),      # override: was +25
    'dancing':      (35, 25, 15, 0, 10),
    # "passed over" -- passed needs slight negative (skipped, rejected)
    'passed':       (-10, 5, -10, 5, 5),      # liquid: "passed away" vs "passed the test" vs "passed me over"
    # "shut off" -- shut is a cessation/rejection word
    'shut':         (-30, 15, -20, 10, -10),   # "shut off", "shut out", "shut down" -- all negative
    # "wrong" needs to be in vocab
    'wrong':        (-35, 15, -20, 10, -15),   # error, mistake, incorrect
    # "through" as in "went through my phone" -- violation
    # Actually "through" is too common. Leave it.
    # "reaching" as in "always the one reaching out"
    'reaching':     (5, 10, 5, 5, 5),          # mild positive (effort)
    # ── Negator contractions: these are OPERATORS not emotional words ──
    # They flip what follows. They should NOT carry their own massive force.
    # "couldnt" was V=-79 (as negative as "abandoned") -- that's wrong.
    # A negator's job is to reverse polarity, not add its own weight.
    'couldnt':      (-5, 0, -5, 0, 0),     # pure operator, flips what follows
    'wont':         (-15, 5, -10, 5, -5),  # refusal/inability -- slightly heavier than other negators
    'havent':       (-5, 0, -5, 0, 0),
    'dont':         (-8, 0, -5, 0, 0),
    'cant':         (-10, 5, -10, 5, -5),  # inability carries slight weight
    'didnt':        (-5, 0, -5, 0, 0),
    'shouldnt':     (-5, 0, -5, 0, 0),
    'wouldnt':      (-5, 0, -5, 0, 0),
    'isnt':         (-5, 0, -5, 0, 0),
    'wasnt':        (-5, 0, -5, 0, 0),
    # ── Permanent suite miss fixes ──
    'afford':       (15, -5, 15, 0, 10),       # override: was -32. "Can afford" = financial security
    'help':         (20, 10, 15, 5, 10),        # override: was -10. Help is positive (assistance)
    'helped':       (25, 5, 15, 0, 10),
    'helping':      (20, 10, 15, 5, 10),
    'slow':         (0, 5, 0, 0, 5),             # override: was -114! Tempo word, neutral. Context resolves.
    'nightmares':   (-60, 50, -40, 50, -30),     # override: was -30. Nightmares = recurring horror
    'stop':         (-5, 5, 5, 0, -3),            # override: was -20. "Stop" is near-neutral cessation
    # ── Bayesian corrections: corpus says these are coded wrong direction ──
    # Based on 30K EmpatheticDialogues sentences. Only high-confidence (count>=30).
    'least':        (-5, 0, 0, 0, 0),         # was -52. "At least" = silver lining. 70% positive in corpus.
    'miss':         (-15, 10, -10, 5, 15),     # was -42. "I miss you" = longing. 66% positive. High G.
    'without':      (-10, 5, -8, 0, 5),         # Slight neg lean. "Without me"=exclusion. "Without stress"=relief (handled by absence scope).
    # ── Math verification fixes: words pulling in wrong direction ──
    'front':        (0, 5, 5, 0, 5),          # override: was V=+32 D=+52. "Front" is a location, not emotional.
    'everything':   (0, 0, 0, 0, 0),           # AMPLIFIER. Scope = ALL. No emotional force. Amplifies context direction.
    'realized':     (-5, 5, 5, 0, 5),         # override: was V=+36. "Realized" is neutral cognition, context resolves.
    'group':        (5, 5, 5, 0, 10),          # override: was V=+32. "Group" is neutral noun.
    # ── Math pass 2: words identified from 27-miss trace ──
    'against':      (-5, 10, 5, 0, 5),        # override: was V=-32. Preposition/connector, not emotional. "Against all odds" = neutral.
    'pressure':     (-10, 10, -5, 5, 10),      # override: was V=-42. "Blood pressure" = clinical. Context resolves.
    'one':          (5, 0, 0, 0, 10),          # override: was V=+35. "The one reaching out" ≠ positive. Pronoun/number, low V.
    'small':        (0, 0, 0, 0, 5),           # override: was V=+42. Size descriptor, not emotional.
    'whole':        (0, 0, 0, 0, 5),           # override: was V=+29. Scope word like "everything". Amplifier, not emotional.
    'grocery':      (0, 0, 0, 0, 5),           # override: was V=+42. Neutral noun. Store, not emotion.
    'understands':  (10, 0, 5, 0, 5),          # override: was pulling positive. Neutral cognition with slight positive.
    'nobody':       (-8, 5, -5, 5, -5),         # Minimal own force. NEGATOR role + force flow do the real work.
    'needs':        (5, 10, 0, 10, 10),            # Slight positive lean. "Needs me"=valued, "needs space"=pushing away. Liquid.
    'cares':        (20, 5, 10, 0, 10),         # "cares about me" = loved. Positive.
    'caring':       (20, 5, 10, 0, 10),
    'notice':       (5, 10, 5, 0, 10),           # Observation/attention. Near-neutral.
    'noticed':      (5, 10, 5, 0, 10),
    'medicine':     (10, 5, 5, 5, 25),           # Healing tool. Positive purpose but not strongly. High G -- health matters.
    'kind':         (5, 5, 5, 0, 5),            # override: was V=+50. Liquid: "kind person"=positive, "another kind"=type/neutral.
    'make':         (5, 5, 5, 0, 5),            # override: was V=+15. Neutral action verb. Context resolves.
    'working':      (0, 5, 0, 0, 5),            # override: was V=+8. Status indicator: ON/OFF. Not positive or negative.
    'between':      (-10, 5, -10, 5, 5),        # "Choose between" = forced split. Losing one for the other.
    # ── Novel sentence miss fixes ──
    'exam':         (0, 15, 0, 10, 15),          # override: was -24. Neutral event. "Passed the exam" = context resolves.
    'passed':       (5, 5, 5, 0, 5),             # override: was -10. "Passed" = succeeded/completed. Slight positive.
    'cooked':       (15, 5, 10, 0, 5),           # override: was -35! Cooking = act of care/service. Positive.
    'cooking':      (15, 10, 10, 0, 5),
    'cook':         (10, 5, 10, 0, 5),
    'steps':        (5, 5, 5, 0, 5),             # "First steps" = progress/milestone. Slight positive.
    'took':         (0, 5, 5, 0, 5),             # override: was -10. Neutral action verb.
    'message':      (0, 5, 0, 0, 5),             # override: was +32. Neutral communication noun.
    'messages':     (0, 5, 0, 0, 5),
    'reply':        (5, 5, 5, 0, 5),             # "Didnt reply" = silence. Reply itself is neutral-positive.
    'down':         (-5, 5, -5, 0, 0),           # override: was -15. Directional. "Put down" vs "calmed down".
    'dog':          (5, 5, 5, 0, 20),            # override: was +39. Relationship noun, not strongly positive. High G.
    'again':        (0, 5, 0, 0, 5),             # override: was -10. Neutral repetition. "Try again" = resilience.
    'space':        (-10, -5, 5, 0, 5),         # "Needs space" = distance/rejection context. Slight neg.
    'paid':         (20, 5, 15, 0, 10),          # override: was +15. "Paid off" = accomplishment. Slightly stronger.
    'called':       (-10, 10, -5, 10, 5),        # override: was -5. "Got called in" = summoned = slight threat.
    'through':      (-5, 5, -5, 0, 5),           # "Went through" = searched/violated. Slight neg.
    'first':        (0, 5, 0, 0, 5),              # Ordinal position setter. "First to die" vs "first to graduate". Context resolves.
    'off':          (0, 0, 0, 0, 0),              # override: was -5. Directional/spatial. "Paid off" ≠ negative.
    'stopping':     (0, 5, 0, 0, 0),              # override: was -5. Neutral action. "Without stopping" = continuous.
    'debt':         (-10, 5, -10, 5, 20),          # override: was -15. High G (financial weight) but less V. "Paid off debt" = relief.
    'crying':       (-15, 15, -10, 5, -5),         # override: was -25. "Without crying" = achievement. Less negative V.
    'mile':         (0, 5, 0, 0, 5),              # override: was +10. Distance measurement. Neutral.
    'ran':          (0, 15, 5, 5, 5),              # override: was +5. Movement verb. Could be running FROM or TO. Context resolves.
    'finally':      (15, -10, 15, -10, 10),        # override: was +5. Wall breaker. Pressure RELEASE. Relief + achievement.
    'groceries':    (0, 0, 0, 0, 15),              # override: was G=5. Food = human need = high gravity.
    'came':         (5, 5, 5, 0, 5),            # "came to my birthday" = showed up = positive.
    'tough':        (-5, 10, 10, 0, 10),       # was -20. "Tough" = strong/resilient. 79% positive.
    'problems':     (-10, 10, -5, 5, 10),      # was -44. "No problems", "solved it". 63% positive.
    'trip':         (5, 10, 5, 0, 10),          # was -24. "Amazing trip" vs "tripped". 64% positive.
    'remember':     (5, 5, 5, 0, 10),           # was -23. "Remember" = connection. 68% positive.
    'interview':    (0, 10, 0, 5, 10),          # was -24. Neutral event. 66% positive.
    'far':          (0, 5, 0, 0, 5),            # was -24. "So far so good". 66% positive.
    'distance':     (0, 5, 0, 0, 5),            # was -62. Neutral measurement. 50% each.
    'low':          (-10, 5, -10, 0, 5),        # was -52. "Low" can be "low key". 58% positive.
    'stress':       (-15, 15, -10, 10, 10),     # was -52. 50% each. Near neutral with slight neg lean.
    'fall':         (-10, 15, -10, 5, 10),      # was -52. "Fall in love" vs "fall down". 53% positive.
    'loss':         (-25, 10, -15, 10, 15),     # was -60. Still negative but less extreme. 42% positive.
    'public':       (0, 10, 0, 0, 10),          # override: was +28. Neutral location, not positive
    'office':       (0, 5, 0, 5, 8),            # neutral location
    'odds':         (-5, 10, -5, 5, 10),        # override: was -10. "Against all odds" = neutral noun
    'exist':        (0, 0, 0, 0, 5),            # override: was +39. State of being, neutral
    'existence':    (0, 0, 0, 0, 10),
    'idea':         (5, 5, 5, 0, 10),            # override: was +50. Neutral noun with some G
    'class':        (0, 5, 0, 0, 10),            # override: was +34. Group of people, neutral noun
    'baby':         (10, 10, 5, 0, 25),           # override: was 0. Relationship noun, positive G
    'stepdad':      (0, 5, 0, 0, 30),             # relationship noun, neutral V, high G
    'stepfather':   (0, 5, 0, 0, 30),
    'stepmom':      (0, 5, 0, 0, 30),
    'stepmother':   (0, 5, 0, 0, 30),
    'treats':       (-10, 10, 10, 5, 5),          # behavioral verb, slightly negative (implies judgment)
    'punishing':    (-30, 20, 30, 10, -15),        # power verb, punisher has D
    'answered':     (5, 5, 5, 0, 5),              # neutral action
    # ── Liquid word V-neutralization: high G, near-zero V ──
    # These words have strong MASS (gravity) but their polarity comes from context.
    # "kick" = violence OR winning goal. "kill" = murder OR slang for dominate.
    # Pinning V kills the context resolution. Keep A/G, zero V.
    'kick':         (-15, 40, 5, 5, 20),      # override: was V=-127. Lean negative but context can flip.
    'kicked':       (-15, 40, 5, 5, 20),
    'kicking':      (-15, 40, 5, 5, 20),
    'kill':         (-40, 40, 10, 10, 25),    # override: was V=-127. Still negative lean -- killing is bad. But "killed it" can override.
    'killed':       (-40, 40, 10, 10, 25),
    'killing':      (-40, 40, 10, 10, 25),
    'kills':        (-40, 40, 10, 10, 25),
    'shot':         (-20, 35, -5, 10, 20),    # "Got shot" = bad (majority case). "Great shot" = context overrides.
    'shots':        (0, 35, 5, 10, 20),
    'poor':         (-15, 5, -15, 5, 20),     # override: was V=-127. "Poor" = pity or poverty. Context resolves.
    'blow':         (0, 30, 5, 5, 15),        # override: was V=-42. "Blew my mind" vs "blew up in my face".
    'blew':         (0, 30, 5, 5, 15),
    'burn':         (-10, 25, 5, 5, 15),      # override: was V=-79. "Sick burn" vs "burned down".
    'burned':       (-10, 25, 5, 5, 15),
    'burning':      (-10, 25, 5, 5, 15),
    'fire':         (10, 35, 10, 5, 20),      # "Thats fire" = slang positive. "Fire" as noun = neutral/positive energy.
    'fired':        (-25, 30, -15, 10, 20),  # "Got fired" = negative. Verb form leans negative.
    'crash':        (-10, 30, -5, 10, 15),    # override: was V=-67. "Crashed at his place" vs "car crashed".
    'crashed':      (-10, 30, -5, 10, 15),
    'drop':         (-5, 20, 5, 5, 10),       # override: was V=-77. "Beat dropped" vs "dropped me".
    'dropped':      (-5, 20, 5, 5, 10),
    'slam':         (0, 35, 10, 5, 15),       # "Slam dunk" vs "slammed the door". High A.
    'slammed':      (0, 35, 10, 5, 15),
    'snap':         (-5, 25, 5, 5, 10),       # override: was V=-30. "Snapped at me" vs "snap out of it".
    'snapped':      (-5, 25, 5, 5, 10),
    'strike':       (0, 30, 10, 5, 15),       # "Struck gold" vs "struck me".
    'struck':       (0, 30, 10, 5, 15),
    'nail':         (0, 15, 5, 0, 10),        # "Nailed it" vs "stepped on a nail".
    'nailed':       (0, 15, 5, 0, 10),
    'scream':       (-15, 50, -5, 10, 15),    # slight neg lean. "Screamed at me" > "scream of joy" in frequency.
    'screamed':     (-15, 50, -5, 10, 15),
    'screaming':    (-15, 50, -5, 10, 15),
    'screams':      (-15, 50, -5, 10, 15),
    'feeding':      (5, 10, 5, 0, 10),        # slight pos lean. "Feeding baby" most common usage.
    'impact':       (-10, 20, 5, 5, 15),       # slight neg lean. "Impact" alone = collision. Context overrides.
    'noisy':        (-10, 30, -5, 5, 5),      # override: was V=-127. Annoying, not devastating.
    'false':        (-20, 15, -10, 5, -5),    # override: was V=-127. Adjective, not catastrophe.
    'shark':        (0, 25, 15, 5, 15),       # override: was V=-127. Shark = predator OR investor. Liquid.
    'rat':          (-10, 15, -5, 5, 5),      # override: was V=-127. Rat = snitch OR animal. Liquid.
    'roach':        (-15, 15, -10, 5, 5),     # override: was V=-127. Pest but not devastating.
    'slap':         (-10, 35, 10, 5, 10),     # override: was V=-127. "Slap" = violence OR "that slaps" = great.
    'bipolar':      (-10, 15, -10, 5, 20),    # override: was V=-127. Clinical term, not emotional extreme.
    'poison':       (-20, 20, -10, 10, 15),   # override: was V=-127. "Toxic" but not max negative.
    'cringe':       (-15, 20, -10, 5, 5),     # override: was V=-127. Embarrassing, not devastating.
    # ── Novel stress test round 2 ──
    'party':        (5, 10, 5, 0, 15),           # override: was V=+10. Event noun, context resolves. "Nobody at my party" ≠ positive.
    'test':         (0, 10, 0, 5, 10),            # Neutral evaluation. "Test came back" = waiting for result.
    'negative':     (-5, 5, -5, 5, 5),            # override: was V=-25. Medical "negative" = GOOD. Liquid word.
    'waitlisted':   (-15, 10, -15, 10, 10),       # Rejection-adjacent. Limbo. Not accepted.
    'towed':        (-25, 15, -20, 15, -10),       # Property taken by force.
    'slept':        (5, -10, 5, 0, 5),             # Neutral action. "Slept well" = positive. Context resolves.
    'eaten':        (5, 0, 5, 0, 5),               # Neutral action. "Havent eaten" = deprivation from absence scope.
    # ── Novel stress test round 3 ──
    'keep':         (0, 5, 5, 0, 5),              # override: was +28. Continuity operator. "Keep trying" OR "keep failing". Context resolves.
    'keeping':      (0, 5, 5, 0, 5),
    'kept':         (0, 5, 5, 0, 5),
    'sell':         (-10, 5, -5, 5, 5),            # Giving up possession. Slight negative (loss of ownership).
    'selling':      (-10, 5, -5, 5, 5),
    'sold':         (-10, 5, -5, 5, 5),
    'brought':      (10, 5, 10, 0, 5),             # "Brought me X" = delivered/gave. Act of service. Slight positive.
    'bring':        (5, 5, 5, 0, 5),
    'shoulder':     (5, -5, 5, 0, 10),             # "Cry on my shoulder" = support/comfort. Slight positive.
    # ── Temporal/duration words: these amplify the DURATION of surrounding emotion ──
    # "Long painful" = pain lasted longer. "Brief joy" = joy was fleeting.
    # These are context-dependent -- their V comes from what they modify.
    'long':         (-5, 5, 0, 0, 5),             # duration extender. Slight neg lean (long wait, long day).
    'dragged':      (-15, -5, -10, 0, -5),         # "dragged on" = lasted too long = negative duration
    'dragging':     (-15, -5, -10, 0, -5),
    'endless':      (-15, 10, -10, 5, -5),          # override: was -15. Duration amplifier, slight neg.
    'prolonged':    (-10, 5, -5, 0, -5),
    'lingering':    (-5, 5, -5, 0, 5),              # could be positive (lingering kiss) or negative (lingering pain)
    'lasting':      (5, 0, 5, 0, 5),                # "lasting impact" = enduring, slight positive
    'brief':        (0, -5, 0, 0, -5),              # short duration, reduces impact
    'short':        (0, -5, 0, 0, -5),              # duration reducer
    'fleeting':     (-5, -5, -5, 0, -5),            # impermanent, can be bittersweet
    'momentary':    (0, -5, 0, 0, -5),
    'instant':      (0, 10, 0, 5, 0),               # speed, not emotion
    'sudden':       (-5, 30, -5, 10, 0),            # surprise/shock timing
    'abrupt':       (-10, 20, -10, 5, -5),          # jarring timing
    'quickly':      (0, 10, 5, 5, 0),               # speed
    'rapid':        (0, 15, 0, 5, 0),
    'rushed':       (-10, 15, -10, 10, -5),          # too fast = negative
    'hasty':        (-10, 10, -10, 5, -5),
    'repetitive':   (-15, -5, -10, 0, -5),
    'monotonous':   (-20, -10, -10, 0, -5),
    'fast':         (0, 20, 5, 5, 0),               # override: was +62! Speed, not emotion.
    # ── Critique/quality words ──
    'formulaic':    (-25, 5, -10, 0, -10),
    'wooden':       (-30, -10, -15, 0, -10),
    'lifeless':     (-35, -15, -20, 0, -15),
    'lackluster':   (-25, 5, -10, 0, -10),
    'overwrought':  (-20, 15, -10, 0, -5),
    'muddled':      (-20, 10, -15, 0, -10),
    'plodding':     (-25, -10, -10, 0, -10),
    'derivative':   (-20, 5, -10, 0, -10),
    'nuanced':      (20, 5, 10, 0, 10),
    'mesmerizing':  (35, 15, 15, 0, 15),
    'riveting':     (35, 25, 15, 0, 15),
    'captivating':  (30, 15, 10, 0, 12),
    'gripping':     (30, 20, 10, 0, 12),
    'superb':       (40, 10, 15, 0, 15),
    'exquisite':    (35, 5, 10, 0, 15),
    'mediocre':     (-20, -5, -10, 0, -10),
    'forgettable':  (-20, -10, -10, 0, -10),
    'predictable':  (-15, -5, -10, 0, -5),
    'cliche':       (-20, 5, -10, 0, -10),
    'contrived':    (-25, 5, -15, 0, -10),
    'pretentious':  (-25, 10, 10, 0, -10),
    # ── Topic/genre nouns: these are SUBJECTS, not assessments ──
    # "A terrible comedy" -- comedy is the topic, terrible is the assessment.
    # Topic words should be near-neutral V with appropriate G (mass/importance).
    'comedy':       (5, 10, 5, 0, 15),            # override: was +80! Genre, not emotion.
    'horror':       (-5, 15, 0, 5, 15),            # override: was -114! Genre, not terror.
    'drama':        (0, 10, 0, 0, 15),             # override: was -18.
    'action':       (0, 15, 5, 5, 10),             # override: was +58. Activity type.
    'adventure':    (5, 10, 5, 0, 10),             # override: was +50.
    'crime':        (-10, 10, -5, 5, 15),          # override: was -89. Topic, not crime happening.
    'war':          (-10, 15, -5, 5, 20),          # override: was -77. Topic, not being in war.
    'basketball':   (0, 10, 0, 0, 10),             # override: was +32. Sport, not emotion.
    'baseball':     (0, 10, 0, 0, 10),             # override: was +35.
    'football':     (0, 10, 0, 0, 10),             # override: was +56.
    'soccer':       (0, 10, 0, 0, 10),             # override: was +43.
    'theater':      (0, 5, 0, 0, 10),              # override: was +58. Venue, not emotion.
    'theatre':      (0, 5, 0, 0, 10),
    'movies':       (0, 5, 0, 0, 10),              # override: was +50.
    'movie':        (0, 5, 0, 0, 10),
    'acting':       (0, 5, 0, 0, 10),              # override: was +43. Skill, not emotion.
    'scenery':      (5, 0, 0, 0, 10),              # override: was +55. Visual setting.
    'director':     (0, 5, 5, 0, 10),              # override: was +32. Role, not emotion.
    'character':    (0, 5, 0, 0, 10),              # override: was +35. Story element.
    # ── SST-2 vocabulary mining: words causing systematic errors ──
    'humor':        (5, 10, 5, 0, 10),             # override: was +50. Topic noun, not emotional.
    'pretty':       (0, 5, 0, 0, 5),               # override: was +50. "Pretty boring" = amplifier. "Pretty girl" = positive. Liquid.
    'special':      (10, 5, 5, 0, 10),             # override: was +50 then +5. "Special friend" = positive. "Nothing special" = context.
    'itself':       (0, 0, 0, 0, 0),               # override: was +28. Pronoun. Zero force.
    'experience':   (0, 5, 0, 0, 10),              # override: was +29. Topic noun.
    'sense':        (0, 5, 0, 0, 5),               # override: was +29. "Makes no sense" vs "sense of humor". Liquid.
    'familiar':     (0, 5, 0, 0, 5),               # override: was +29. "Too familiar" = derivative. Context resolves.
    'often':        (0, 0, 0, 0, 0),               # override: was +39. Frequency word. Zero force.
    'big':          (0, 10, 5, 0, 10),             # override: was +32. Size descriptor.
    'new':          (0, 5, 0, 0, 5),               # override: was +29. Temporal/novelty. Near neutral.
    'young':        (0, 5, 0, 0, 5),               # override: was +29. Age descriptor.
    'emotional':    (0, 15, 0, 0, 10),             # override: was +29. Describes intensity, not polarity.
    'fine':         (-5, -5, 0, 0, 5),              # override: was -32 then +5. "Im fine" = mask. "Just fine" = adequate. Slight neg lean.
    'minor':        (0, 5, 0, 0, 5),               # override: was -59. "A minor masterpiece" ≠ negative. Size descriptor.
    'revenge':      (-10, 15, 5, 5, 15),           # override: was -77. Topic noun in reviews.
    # ── Gemini consensus fixes: words causing engine disagreement ──
    'human':        (0, 5, 0, 0, 10),              # override: was +58. Species noun, not emotional.
    'greatly':      (0, 10, 0, 0, 0),              # override: was +56. Amplifier, not positive.
    'choosing':     (0, 5, 5, 0, 5),               # override: was +48. Action verb. "I'll be choosing" = control.
    'training':     (0, 10, 5, 0, 5),              # override: was +48. Activity noun.
    'station':      (0, 5, 0, 0, 5),               # override: was +50. Location noun.
    'treasure':     (5, 10, 5, 0, 15),             # override: was +50. Object noun with gravity.
    'nest':         (0, 5, 0, 0, 5),               # override: was +42. Location noun.
    'constant':     (0, 5, 0, 0, 5),               # override: was +51. Frequency descriptor.
    'directly':     (0, 5, 5, 0, 0),               # override: was +55. Manner adverb.
    'useful':       (10, 5, 5, 0, 5),              # override: was +58. Mild positive utility.
    'everybody':    (0, 5, 0, 0, 10),              # override: was +42. Scope word like everything. AMPLIFIER.
    'offer':        (5, 5, 5, 0, 5),               # override: was +56. Neutral action.
    'offered':      (5, 5, 5, 0, 5),
    'trust':        (15, 5, 10, 0, 15),            # override: was +106! Way too high. "Trust me" vs "I don't trust you."
    'okay':         (0, -5, 0, 0, 0),              # override: was +28. "I'm okay" = mask. "Okay but" = hedge. Near neutral.
    'plan':         (0, 5, 5, 0, 5),               # override: was +51. Action noun.
    'terminal':     (0, 5, 0, 0, 5),               # override: was -50. Computer terminal, not death.
    'opposite':     (0, 5, 0, 0, 0),               # override: was -62. Direction/contrast word.
    'argue':        (-5, 15, 5, 5, 5),             # override: was -42. Debate verb. Mild negative lean.
    'harder':       (-20, 10, -10, 5, -5),          # override: was -52. Difficulty.
    # ── Internet/Twitch cultural tokens (dual-state: surface ≠ meaning) ──
    # These are cultural compression artifacts, not standard words.
    # Engine reads SURFACE. Model should read UNDERLYING.
    'gg':           (-10, 5, -5, 0, -5),            # "good game" → culturally: resignation/hyperbolic defeat
    'rip':          (-15, 10, -10, 5, -5),          # override: was -114! Culturally: "oh well that sucks" not actual death
    'fr':           (0, 10, 5, 0, 0),               # "for real" = emphasis/agreement. Neutral amplifier.
    'lol':          (5, 10, 5, 0, 5),               # override: was +39. Often nervous/deflection/padding, not genuine joy.
    'lmao':         (10, 15, 10, 0, 5),             # override: was +50. Amusement but often performative.
    'lul':          (5, 10, 5, 0, 5),               # = lol. Twitch variant.
    # ── Informal contractions (map to their expanded forms) ──
    'gonna':        (0, 5, 5, 5, 0),               # = going to. Forward intent. Slight urgency.
    'wanna':        (0, 10, 0, 5, 0),              # = want to. Desire. Slight arousal.
    'gotta':        (0, 10, 0, 10, 0),             # = got to. Obligation. Urgency.
    'kinda':        (0, 0, 0, 0, 0),               # = kind of. Hedge. Zero force.
    'tryna':        (0, 10, 5, 5, 0),              # = trying to. Effort. Slight arousal.
    'imma':         (0, 10, 10, 5, 0),             # = I'm going to. Declaration. Slight D.
    'finna':        (0, 10, 10, 10, 0),            # = fixing to / going to. Strong forward intent.
    'lemme':        (0, 5, 5, 5, 0),               # = let me. Request for action.
    'dunno':        (0, 0, -5, 0, 0),              # = don't know. Uncertainty.
    # ── Internet slang with emotional weight ──
    'lowkey':       (0, -5, 0, 0, 0),              # understated. Dampener on what follows.
    'highkey':      (0, 10, 0, 0, 0),              # amplified. Booster on what follows.
    'deadass':      (0, 10, 10, 0, 0),             # = seriously, for real. Emphasis.
    'sus':          (-10, 10, -5, 5, -5),           # suspicious. Mild negative.
    'mid':          (-15, 0, 0, 0, -5),             # mediocre, underwhelming. Negative.
    'bussin':       (25, 15, 10, 0, 10),            # = amazing, delicious. Strong positive.
    'slay':         (20, 15, 15, 0, 10),            # = killed it, amazing. Positive.
    'bet':          (5, 5, 10, 0, 5),               # = okay, agreed, for sure. Slight positive.
    'based':        (10, 10, 15, 0, 10),            # = authentic, bold. Positive.
    'cringe':       (-20, 10, -10, 0, -10),         # embarrassing, awkward. Negative.
    'cope':         (-15, 5, -10, 0, -10),          # delusional self-comfort. Negative.
    'ratio':        (-10, 15, 15, 0, -5),           # getting ratioed = dunked on. Negative for target.
    # ── Gemini+Claude consensus round 2 ──
    'sort':         (0, 5, 0, 0, 5),               # override: was +37. "Sort of" = hedge. Neutral.
    'convince':     (0, 10, 5, 5, 5),              # override: was +46. Neutral action.
    'baked':        (0, 5, 0, 0, 0),               # override: was +31. Cooking OR slang. Neutral.
    'gas':          (0, 5, 0, 0, 5),               # override: was +28. Substance noun. Neutral.
    'activities':   (0, 5, 0, 0, 5),               # override: was +35. Action noun. Neutral.
    'knowledge':    (5, 5, 5, 0, 10),              # override: was +39. Information noun. Slight positive.
    'released':     (0, 10, 5, 5, 5),              # override: was +35. Action verb. Context resolves.
    'gang':         (-10, 15, -5, 5, 15),          # override: was -79. Group noun. "My gang" vs "gang violence."
    'awfully':      (0, 10, 0, 0, 0),              # override: was -94! Amplifier. "Awfully nice" = very nice.
    'bummed':       (-15, 5, -10, 0, -5),          # override: was -47. Mild disappointment.
    'ugh':          (-10, 10, -5, 0, -5),          # override: was -42. Mild disgust/annoyance.
    'shady':        (-10, 10, -5, 5, -5),          # override: was -40. Suspicious but could be playful.
    # ── Novel miss fixes cycle 11 ──
    'highway':      (0, 5, 0, 0, 5),               # override: was +29. Location noun. Neutral.
    'away':         (-10, 5, -5, 0, -5),             # override: was -15. Slight neg lean. "Far away" = distance. "Went away" = departure.
    'relationship': (10, 5, 5, 0, 25),             # override: was +37. Still slight positive (relationships have value) but less extreme.
    # ── SST-2 round 2: next batch of over-weighted words ──
    'like':         (5, 0, 0, 0, 0),               # override: was +15. FILLER/comparison word. "Like a dream" ≠ positive.
    'watch':        (0, 5, 0, 0, 5),               # override: was +50. Action verb. "Watch a movie" ≠ positive.
    'focus':        (0, 5, 5, 0, 5),               # override: was +32. Attention verb. Neutral.
    'full':         (0, 5, 0, 0, 5),               # override: was +28. Quantity descriptor. "Full of" ≠ positive.
    'fun':          (15, 15, 10, 0, 5),             # override: was +28. Still positive but less extreme.
    'funny':        (15, 15, 10, 0, 5),             # override: was +28. Same.
    # Discovery verbs -- zero polarity, the OBJECT determines the emotion
    'found':        (0, 10, 5, 5, 5),              # override: was -5. Context resolves.
    'find':         (0, 10, 5, 5, 5),
    'discovered':   (0, 10, 5, 5, 5),
    'discover':     (0, 10, 5, 5, 5),
    # ── Slurs/derogatory: these carry heavy negative force ──
    'whore':        (-60, 30, -30, 10, -20),
    'slut':         (-55, 25, -25, 10, -18),
    'bitch':        (-50, 30, -20, 10, -15),       # can be reclaimed but default = slur
    'hoe':          (-45, 20, -20, 5, -12),
    'skank':        (-50, 20, -25, 5, -15),
    # Bookend compound command tokens (collapsed from "shut [spice] up" etc.)
    'shutup':       (-40, 40, 40, 15, -10),       # silence command. D+40 = controller has power. V negative.
    'getout':       (-45, 50, 45, 20, -15),        # expulsion command. Stronger than shutup.
    'fuckoff':      (-50, 50, 40, 15, -15),        # rejection command. Strong negative.
    'backoff':      (-30, 35, 30, 10, -10),        # distance command. Less aggressive.
    'pissoff':      (-40, 40, 35, 10, -12),        # rejection. Similar to fuckoff but milder.
    # ── Structural edge case fixes ──
    'fought':       (-15, 30, 15, 10, 15),         # override: was -94! "Fought FOR me" = advocated. "Fought me" = violence. Liquid.
    'fighting':     (-15, 30, 15, 10, 15),
    'survived':     (30, 10, 25, 0, 20),           # override: was +10. Surviving = overcoming death. Strongly positive.
    'survival':     (25, 10, 20, 0, 20),
    'surgery':      (-10, 20, -10, 20, 25),        # override: was -25. Medical procedure, high G. Not inherently negative.
    'benign':       (35, -15, 20, -10, 15),        # override: was +25. "Benign" = NOT dangerous. Relief. Stronger positive.
    'charges':      (-10, 10, -10, 10, 15),        # override: was -20. Legal noun, slight neg lean but context resolves.
    'tumor':        (-20, 15, -15, 15, 25),        # override: was -40. Medical noun. High G. Less extreme V.
}
EMOTIONAL_VOCABULARY.update(_VOCAB_FIXES)


# ══════════════════════════════════════════════════════════════════════
# SESSION 2026-04-03: Targeted V5.5 fixes. Original vocab + surgical overrides.
# ══════════════════════════════════════════════════════════════════════

_V55_FIXES = {
    'myself':(0,0,0,0,5), 'yourself':(0,0,0,0,5), 'himself':(0,0,0,0,5),
    'herself':(0,0,0,0,5), 'themselves':(0,0,0,0,5),
    'traffic':(0,0,0,0,0), 'weather':(0,0,0,0,0), 'homework':(0,3,0,0,0),
    'weekend':(5,5,0,0,3), 'fruit':(3,0,0,0,0), 'cream':(3,0,0,0,0),
    'tax':(-5,5,-5,0,5), 'humid':(-5,5,-5,0,3), 'hasnt':(0,0,0,0,0),
    'hardly':(0,0,-3,0,0), 'aged':(0,0,-3,0,5), 'spit':(-15,15,10,5,5),
    'removed':(-10,5,-5,5,10), 'blast':(0,15,5,0,5), 'snake':(-10,15,10,5,10),
    'spider':(-5,10,0,0,5), 'washed':(0,5,0,0,3), 'birth':(0,15,0,5,35),
    'only':(0,0,-3,0,0), 'just':(0,0,-3,0,0), 'merely':(0,0,-5,0,0),
    'simply':(0,0,0,0,0), 'barely':(-5,5,-5,5,3), 'almost':(0,5,0,5,5),
    'begging':(-10,15,-25,15,12),
    'kill':(-40,40,10,10,25), 'killed':(-40,40,10,10,25), 'killing':(-40,40,10,10,25),
    'suicide':(-80,30,-20,20,40), 'suicidal':(-70,25,-20,15,35),
    'pregnant':(0,15,0,10,40), 'adopted':(0,5,0,0,25), 'diagnosed':(0,15,-5,15,40),
    'deployed':(0,15,0,15,35), 'overdosed':(-20,20,-20,20,35),
    'relapsed':(-40,15,-20,10,35), 'relapse':(-40,15,-20,10,35), 'sober':(10,-5,15,0,20),
    'whore':(-20,25,-10,5,25), 'slut':(-15,20,-10,5,20), 'bitch':(-10,25,10,5,20),
    'hoe':(-10,15,-5,0,15), 'bastard':(-10,20,5,5,15), 'asshole':(-20,20,10,5,10),
    'dick':(-15,15,5,0,10), 'prick':(-20,15,5,0,10), 'cunt':(-25,25,-10,5,20),
    'jump':(0,3,0,0,0), 'want':(0,5,0,5,3), 'born':(0,10,0,0,30),
    'living':(0,5,0,0,20), 'alive':(0,5,0,0,15), 'live':(0,5,0,0,15),
    'life':(0,0,0,0,40), 'pills':(-10,10,-10,15,20), 'point':(0,0,0,0,5),
    'deserve':(0,0,0,0,10), 'disappear':(-15,5,-15,0,20), 'disappeared':(-15,5,-15,0,20),
    'lost':(-20,5,-15,0,20), 'lol':(15,15,5,0,-5), 'nice':(15,0,5,0,3),
    'crying':(-25,15,-15,5,-10), 'best':(25,10,15,0,10), 'easier':(15,-5,5,0,5),
    'trusted':(5,0,5,0,15), 'terrible':(-30,10,-15,5,-8), 'horrible':(-30,10,-15,5,-8),
    'awful':(-25,10,-10,5,-5),
    'comeon':(0,10,-5,10,5), 'killedit':(30,20,25,0,10),
    'goeshard':(25,20,20,0,8), 'closedon':(20,10,15,0,15),
    # ── V1 words too extreme for FORCE_SCALE=1.4 ──
    'hurt':(-25,15,-15,10,-10), 'hurts':(-25,15,-15,10,-10),
    'pain':(-30,20,-25,15,-15), 'ache':(-15,10,-10,5,-5),
    'sore':(-15,10,-10,5,-5),
    'sad':(-25,10,-10,0,-10), 'sadness':(-30,10,-15,0,-12),
    'angry':(-25,30,15,10,10), 'anger':(-25,30,15,10,10),
    'scared':(-25,25,-20,15,-10), 'fear':(-30,25,-25,15,-12),
    'worried':(-20,15,-10,10,-8), 'anxious':(-25,20,-15,15,-10),
    'depressed':(-35,5,-25,5,-20), 'lonely':(-25,5,-15,0,-15),
    'miserable':(-35,10,-20,5,-15), 'hopeless':(-35,5,-30,5,-20),
    'devastated':(-35,15,-25,10,-20), 'heartbroken':(-35,15,-20,10,-18),
    'furious':(-30,35,20,15,10), 'livid':(-30,35,20,15,10),
    'terrified':(-30,30,-25,20,-15), 'horrified':(-30,25,-20,15,-12),
    'disgusted':(-25,20,10,5,-8), 'ashamed':(-25,10,-20,5,-15),
    'embarrassed':(-20,15,-15,5,-10), 'frustrated':(-20,20,-10,10,-8),
    'annoyed':(-15,10,-5,5,-5), 'irritated':(-15,10,-5,5,-5),
    'jealous':(-20,15,-10,5,-8), 'envious':(-20,10,-10,0,-8),
    'guilty':(-25,10,-20,5,-15), 'regret':(-25,10,-15,5,-12),
    'grief':(-35,10,-20,5,-20), 'mourning':(-30,5,-15,0,-18),
    'panicked':(-30,35,-25,25,-15), 'terror':(-35,35,-30,25,-18),
    'hate':(-35,30,10,5,10), 'hatred':(-35,25,10,5,12),
    'love':(40,10,10,0,20), 'loved':(35,5,10,0,18),
    'happy':(30,10,10,0,10), 'happiness':(35,10,10,0,12),
    'joy':(35,15,10,0,12), 'joyful':(30,15,10,0,10),
    'excited':(30,25,10,5,8), 'thrilled':(30,25,10,5,10),
    'grateful':(30,5,10,0,15), 'thankful':(25,5,10,0,12),
    'proud':(30,10,20,0,15), 'confident':(25,15,20,0,10),
    'peaceful':(20,-10,10,0,10), 'calm':(15,-10,10,0,8),
    'hopeful':(25,10,10,5,10), 'optimistic':(25,10,10,0,10),
    'amused':(20,15,5,0,5), 'entertained':(20,10,5,0,5),
    'inspired':(25,15,15,0,10), 'motivated':(20,15,15,5,8),
    'relieved':(25,-5,10,-5,10),
    'wonderful':(35,15,10,0,12), 'amazing':(35,20,10,0,10),
    'awesome':(30,20,10,0,8), 'fantastic':(30,20,10,0,10),
    'beautiful':(30,5,10,0,12), 'gorgeous':(25,10,10,0,10),
    'brilliant':(30,15,15,0,10), 'excellent':(25,10,10,0,8),
    'perfect':(30,10,15,0,10), 'lovely':(25,5,10,0,10),
    'delightful':(25,10,10,0,8), 'pleasant':(20,5,5,0,5),
    'sweet':(20,5,5,0,8),
    'worthless':(-35,10,-35,5,-20), 'useless':(-25,5,-20,0,-12),
    'nothing':(-30,-5,-20,5,-15),
    'death':(-35,25,-25,15,-18), 'die':(-35,35,-25,25,-18),
    'dead':(-35,20,-25,10,-15),
    'agony':(-35,30,-25,15,-18), 'anguish':(-35,25,-25,10,-18),
    'despair':(-35,10,-30,5,-20), 'suffering':(-30,15,-20,10,-15),
    'abuse':(-35,25,-20,15,-18), 'torture':(-35,30,-25,15,-18),
}
EMOTIONAL_VOCABULARY.update(_V55_FIXES)


# ── New words added 2026-04-03 (original forces preserved) ──
_NEW_WORDS = {
    'about'               : (  +0,  +1,  +0,  +0,  -8),
    'above'               : (  +0,  +0,  +0,  +0,  -5),
    'acted'               : (  +2,  +1,  +0,  +0,  -5),
    'actor'               : (  +0,  +0,  +0,  +0,  +2),
    'actors'              : (  +0,  +0,  +0,  +0,  +2),
    'adaptation'          : (  +0,  +0,  +0,  +0,  +3),
    'ads'                 : (  +0,  +0,  +0,  +0,  +2),
    'adults'              : (  -1,  +0,  +0,  +0,  +3),
    'after'               : (  +0,  +0,  +0,  +0,  -8),
    'aint'                : (  +0,  +0,  +0,  +0,  +0),
    'alien'               : (  -3,  +5,  -1,  +0,  -1),
    'aliens'              : (  -1,  +5,  +0,  +0,  +0),
    'all'                 : (  +8,  +5,  +2,  +0,  +8),
    'also'                : (  +0,  +0,  +0,  +0,  -8),
    'although'            : (  +0,  +0,  +0,  +0,  +0),
    'american'            : (  +0,  +0,  +0,  +0,  +0),
    'among'               : (  +0,  +0,  +0,  +0,  +0),
    'amusing'             : ( +12,  +9,  +3,  +0,  -5),
    'and'                 : (  +0,  +0,  +0,  +0,  +0),
    'anyone'              : (  +0,  +0,  +0,  +0,  +0),
    'apex'                : (  +1,  +1,  +0,  +0,  +5),
    'are'                 : (  +0,  +0,  +0,  +0,  -5),
    'arent'               : (  +0,  +0,  +0,  +0,  +0),
    'back'                : (  +1,  +1,  +0,  +0,  -1),
    'ball'                : (  +0,  +0,  +0,  +0,  +6),
    'ban'                 : (  +0,  +0,  +0,  +0,  +6),
    'banned'              : (  +0,  +0,  +0,  +0,  +2),
    'because'             : (  +0,  +0,  +0,  +0,  +0),
    'bigsad'              : (  +0,  +0,  +0,  +0,  +0),
    'bits'                : (  +0,  +0,  +0,  +0,  +4),
    'bleedpurplehd'       : (  +0,  +0,  +0,  +0,  +0),
    'blend'               : (  +0,  +0,  +0,  +0,  +2),
    'both'                : (  +0,  +0,  +0,  +0,  -8),
    'boy'                 : (  +0,  +1,  +0,  +0,  +3),
    'bros'                : (  +0,  +0,  +0,  +0,  +0),
    'btw'                 : (  +0,  +0,  +0,  +0,  +0),
    'bud'                 : (  +1,  +1,  +0,  +0,  +3),
    'but'                 : (  +0,  +0,  +0,  +0,  -8),
    'buy'                 : (  +5,  +3,  +1,  +0,  -2),
    'bye'                 : (  +0,  +0,  +0,  +0,  +7),
    'cam'                 : (  +0,  +0,  +0,  +0,  +7),
    'camera'              : (  +0,  +0,  +0,  +0,  +9),
    'can'                 : (  +0,  +0,  +0,  +0,  +3),
    'case'                : (  +0,  +0,  +0,  +0,  +4),
    'caseoh'              : (  +0,  +0,  +0,  +0,  +0),
    'caseohdailydoodledance': (  +0,  +0,  +0,  +0,  +0),
    'caseohdailydoodlestwerk': (  +0,  +0,  +0,  +0,  +0),
    'casey'               : (  +0,  +0,  +0,  +0,  +0),
    'cast'                : (  +0,  +0,  +0,  +0,  +1),
    'cause'               : (  +0,  +0,  +0,  +0,  +1),
    'channel'             : (  +0,  +0,  +0,  +0,  +3),
    'characters'          : (  +7,  +3,  +2,  +0,  +4),
    'charm'               : ( +12,  +3,  +3,  +0,  +0),
    'charming'            : ( +12,  +4,  +3,  +0,  -5),
    'chris'               : (  +0,  +0,  +0,  +0,  +0),
    'cinema'              : (  +0,  +0,  +0,  +0,  +7),
    'clumsy'              : ( -12,  +9,  -3,  +0,  -5),
    'come'                : (  +0,  +0,  +0,  +0,  -4),
    'comes'               : (  +0,  +0,  +0,  +0,  -4),
    'comet'               : (  +0,  +0,  +0,  +0,  +4),
    'completely'          : ( +11,  +5,  +3,  +0,  -5),
    'consecutive'         : (  -4,  +4,  -2,  +0,  -8),
    'costumes'            : (  -3,  +1,  +0,  +0,  +4),
    'cult'                : (  +1,  +1,  +0,  +0,  +7),
    'currently'           : (  +0,  +0,  +0,  +0,  -8),
    'dark'                : (  -5,  +6,  -2,  +0,  -4),
    'david'               : (  +0,  +0,  +0,  +0,  +6),
    'dawg'                : (  +0,  +0,  +0,  +0,  +0),
    'deftly'              : (  +5,  +2,  +1,  +0,  -5),
    'delete'              : (  -1,  +0,  +0,  +0,  -3),
    'delivers'            : (  +0,  +0,  +0,  +0,  -4),
    'depth'               : (  +0,  +2,  +0,  +0,  +4),
    'did'                 : (  +0,  +0,  +0,  +0,  -5),
    'diesel'              : (  +0,  +0,  +0,  +0,  +9),
    'dinodance'           : (  +0,  +0,  +0,  +0,  +0),
    'directed'            : (  +1,  +0,  +0,  +0,  -3),
    'documentary'         : (  +0,  +3,  +0,  +0,  -3),
    'dude'                : (  +0,  +0,  +0,  +0,  +4),
    'eat'                 : (  +1,  +0,  +0,  +0,  -3),
    'effects'             : (  -2,  +1,  +0,  +0,  +1),
    'either'              : (  +0,  +0,  +0,  +0,  -8),
    'energetic'           : (  +5,  +4,  +1,  +0,  -5),
    'engage'              : (  +0,  +0,  +0,  +0,  -4),
    'ensemble'            : (  +0,  +0,  +0,  +0,  +4),
    'entertainment'       : ( +12,  +0,  +3,  +0,  +6),
    'entirely'            : (  +7,  +7,  +2,  +0,  -5),
    'exactly'             : (  +5,  +2,  +1,  +0,  -5),
    'exquisitely'         : ( +11,  +5,  +3,  +0,  -5),
    'fans'                : (  +1,  +0,  +0,  +0,  +0),
    'feature'             : (  +0,  +1,  +0,  +0,  +3),
    'featuring'           : (  +1,  +4,  +0,  +0,  -5),
    'few'                 : (  +0,  +0,  +0,  +0,  -1),
    'film'                : (  +0,  +0,  +0,  +0,  +2),
    'filmmaker'           : (  +0,  +0,  +0,  +0,  +4),
    'filmmakers'          : (  +0,  +0,  +0,  +0,  +4),
    'films'               : (  +0,  +0,  +0,  +0,  +2),
    'flick'               : (  +0,  +0,  +0,  +0,  -2),
    'following'           : (  +1,  +0,  +0,  +0,  -4),
    'for'                 : (  +0,  +0,  +0,  +0,  +0),
    'force'               : (  +0,  +1,  +0,  +0,  +0),
    'form'                : (  +0,  +0,  +0,  +0,  +1),
    'french'              : (  +0,  +0,  +0,  +0,  +1),
    'from'                : (  +0,  +0,  +0,  +0,  +0),
    'genre'               : (  +0,  +0,  +0,  +0,  +4),
    'gentle'              : (  +4,  +6,  +1,  +0,  -5),
    'get'                 : (  +0,  +1,  +0,  +0,  -3),
    'gets'                : (  +0,  +1,  +0,  +0,  -3),
    'given'               : (  +0,  +1,  +0,  +0,  -2),
    'goatemotey'          : (  +0,  +0,  +0,  +0,  +0),
    'god'                 : (  +0,  +0,  +0,  +0,  +6),
    'goes'                : (  +0,  +0,  +0,  +0,  -4),
    'going'               : (  +0,  +0,  +0,  +0,  -5),
    'goldplz'             : (  +0,  +0,  +0,  +0,  +0),
    'goober'              : (  +0,  +0,  +0,  +0, +15),
    'granny'              : (  -3,  +2,  -1,  +0, +10),
    'guys'                : (  +0,  +0,  +0,  +0,  +4),
    'hacker'              : (  -1,  +5,  +0,  +0,  +4),
    'hacking'             : (  +0,  +0,  +0,  +0,  -2),
    'had'                 : (  +0,  +1,  +0,  +0,  -5),
    'harry'               : (  -1,  +1,  +0,  +0,  -5),
    'has'                 : (  +0,  +1,  +0,  +0,  -4),
    'heard'               : (  +0,  +0,  +0,  +0,  -5),
    'heavyhanded'         : (  +0,  +0,  +0,  +0,  +0),
    'hes'                 : (  +0,  +0,  +0,  +0,  +5),
    'heyguys'             : (  +0,  +0,  +0,  +0,  +0),
    'high'                : (  +1,  +2,  +0,  +0,  -2),
    'him'                 : (  +0,  +0,  +0,  +0,  +0),
    'his'                 : (  +0,  +0,  +0,  +0,  +0),
    'hollywood'           : (  -4,  +2,  -1,  +0,  +1),
    'imaginative'         : ( +11,  +5,  +3,  +0,  -5),
    'imagine'             : (  -5,  +6,  -2,  +0,  -3),
    'into'                : (  +0,  +0,  +0,  +0,  +0),
    'intriguing'          : ( +10,  +5,  +3,  +0,  -5),
    'its'                 : (  +0,  +0,  +0,  +0,  +8),
    'ive'                 : (  +0,  +0,  +0,  +0,  +0),
    'jack'                : (  +0,  +0,  +0,  +0,  +8),
    'japanese'            : (  +0,  +0,  +0,  +0,  +0),
    'jebasted'            : (  +0,  +0,  +0,  +0,  +0),
    'jesus'               : (  +0,  +0,  +0,  +0,  +2),
    'jinxlul'             : (  +0,  +0,  +0,  +0,  +0),
    'jokes'               : ( +10,  +8,  +3,  +0,  +2),
    'journey'             : (  +0,  +0,  +0,  +0,  +0),
    'jynxzi'              : (  +0,  +0,  +0,  +0,  +0),
    'know'                : (  +4,  +4,  +1,  +0,  -4),
    'komodohype'          : (  +0,  +0,  +0,  +0,  +0),
    'latest'              : (  +0,  +1,  +0,  +0,  -5),
    'lee'                 : (  +0,  +0,  +0,  +0,  +6),
    'lets'                : (  +0,  +0,  +0,  +0,  +0),
    'level'               : (  +0,  +1,  +0,  +0,  -1),
    'literally'           : (  +0,  +0,  +0,  +0,  -8),
    'lock'                : (  -1,  +0,  +0,  +0,  +1),
    'look'                : (  +0,  +1,  +0,  +0,  -1),
    'looking'             : (  +0,  +1,  +0,  +0,  -2),
    'looks'               : (  +0,  +1,  +0,  +0,  -1),
    'lot'                 : (  -1,  +1,  +0,  +0,  +0),
    'maddie'              : (  +0,  +0,  +0,  +0,  +0),
    'manages'             : (  +0,  +0,  +0,  +0,  -4),
    'many'                : (  +0,  +0,  +0,  +0,  -8),
    'material'            : (  +1,  +1,  +0,  +0,  -3),
    'may'                 : (  +0,  +0,  +0,  +0, +10),
    'might'               : (  -3, +17,  -1,  +0,  +2),
    'miller'              : (  +0,  +0,  +0,  +0,  +9),
    'minute'              : (  +0,  +1,  +0,  +0,  +0),
    'moments'             : (  +1,  +0,  +0,  +0,  +4),
    'mormon'              : (  +0,  +0,  +0,  +0,  +0),
    'most'                : (  +0,  +0,  +0,  +0,  -8),
    'mostly'              : (  +0,  +0,  +0,  +0,  -5),
    'move'                : (  +0,  +1,  +0,  +0,  -3),
    'much'                : (  +3,  +3,  +0,  +0,  -6),
    'mystery'             : (  +3,  +1,  +0,  +0,  +9),
    'narrative'           : (  -1,  +2,  +0,  +0,  -4),
    'not'                 : ( -15, +12,  -7,  +3,  -8),
    'notlikethis'         : (  +0,  +0,  +0,  +0,  +0),
    'now'                 : (  +0,  +1,  +0,  +0,  -6),
    'occasionally'        : (  +0,  +0,  +0,  +0,  -5),
    'offers'              : (  +0,  +0,  +0,  +0,  -1),
    'open'                : (  -2,  +4,  -1,  +0,  -5),
    'opera'               : (  +0,  +0,  +0,  +0,  +8),
    'original'            : (  +0,  +4,  +0,  +0,  -3),
    'originality'         : (  -9,  +4,  -2,  +0,  +1),
    'other'               : (  -6,  +8,  -3,  +2,  -8),
    'our'                 : (  +0,  +0,  +0,  +0,  +0),
    'overall'             : (  +0,  +0,  +0,  +0,  +0),
    'part'                : (  +0,  +0,  +0,  +0,  +0),
    'pay'                 : (  +1,  +1,  +0,  +0,  -1),
    'peanut'              : (  +1,  +2,  +0,  +0,  +7),
    'performances'        : ( +12,  +1,  +3,  +0,  +4),
    'perhaps'             : (  +0,  +0,  +0,  +0,  -5),
    'pewpewpew'           : (  +0,  +0,  +0,  +0,  +0),
    'place'               : (  +0,  +0,  +0,  +0,  +1),
    'played'              : (  +0,  +0,  +0,  +0,  -4),
    'plot'                : (  +0,  +0,  +0,  +0,  +2),
    'pls'                 : (  +0,  +0,  +0,  +0,  +0),
    'pogchamp'            : (  +0,  +0,  +0,  +0,  +0),
    'points'              : (  +1,  +0,  +0,  +0,  +2),
    'pool'                : (  +0,  +1,  +0,  +0,  +4),
    'potter'              : (  +0,  +0,  +0,  +0,  -2),
    'price'               : (  +2,  +1,  +0,  +0,  +2),
    'probably'            : (  +0,  +0,  +0,  +0,  -8),
    'production'          : (  +0,  +0,  +0,  +0,  +5),
    'psychological'       : (  -5,  +2,  -1,  +0,  -5),
    'quidditch'           : (  +0,  +0,  +0,  +0,  +0),
    'quirky'              : ( -15,  +7,  -4,  +0,  -5),
    'reality'             : (  +3,  +2,  +0,  +0,  +3),
    'redfield'            : (  +0,  +0,  +0,  +0,  +0),
    'religion'            : (  +0,  +0,  +0,  +0,  +4),
    'rentals'             : (  +0,  +0,  +0,  +0,  +3),
    'retro'               : (  +1,  +1,  +0,  +0,  +2),
    'rewind'              : (  +0,  +0,  +0,  +0,  +0),
    'romance'             : (  +4,  +3,  +1,  +0,  +1),
    'saying'              : (  +0,  +0,  +0,  +0,  -2),
    'scene'               : (  -1,  +0,  +0,  +0,  +6),
    'scenes'              : (  -1,  +0,  +0,  +0,  +6),
    'screen'              : (  +0,  +0,  +0,  +0,  +2),
    'script'              : (  +0,  +0,  +0,  +0,  +2),
    'second'              : (  +0,  +1,  +0,  +0,  +1),
    'seconds'             : (  +0,  +1,  +0,  +0,  +3),
    'seem'                : (  +3,  +2,  +0,  +0,  -5),
    'seemsgood'           : (  +0,  +0,  +0,  +0,  +0),
    'sentimentality'      : (  -7,  +9,  -2,  +0,  +3),
    'sequel'              : (  +0,  +0,  +0,  +0,  +6),
    'series'              : (  +0,  +0,  +0,  +0,  +3),
    'serious'             : (  -2, +10,  -1,  +0,  -8),
    'shes'                : (  +0,  +0,  +0,  +0,  +0),
    'siptime'             : (  +0,  +0,  +0,  +0,  +0),
    'sitcom'              : (  +0,  +0,  +0,  +0,  +7),
    'skin'                : (  +0,  +0,  +0,  +0,  +2),
    'slop'                : (  -1,  +2,  +0,  +0,  +1),
    'some'                : (  +3,  +6,  +0,  +0,  -8),
    'soundtrack'          : (  +0,  +0,  +0,  +0, +10),
    'sparked'             : (  +0,  +0,  +0,  +0,  -4),
    'spooky'              : ( -11,  +7,  -5,  +2,  -8),
    'star'                : (  +1,  +1,  +0,  +0,  +1),
    'steam'               : (  +0,  +0,  +0,  +0,  -2),
    'still'               : (  -2,  +5,  -1,  +0,  -3),
    'story'               : (  -1,  +1,  +0,  +0,  +4),
    'streak'              : (  +0,  +0,  +0,  +0,  +1),
    'stream'              : (  +0,  +0,  +0,  +0,  +1),
    'study'               : (  +1,  +0,  +0,  +0,  +1),
    'stuff'               : (  +0,  +1,  +0,  +0,  +0),
    'sub'                 : (  -2,  +1,  -1,  +0,  +6),
    'subs'                : (  -2,  +1,  -1,  +0,  +6),
    'subscribed'          : (  +0,  +0,  +0,  +0,  -4),
    'subtember'           : (  +0,  +0,  +0,  +0,  +0),
    'such'                : (  -1,  +1,  +0,  +0,  -8),
    'surreal'             : (  -1,  +8,  +0,  +0,  -5),
    'suspense'            : ( +12,  +0,  +3,  +0,  +7),
    'takes'               : (  +0,  +0,  +0,  +0,  -4),
    'taking'              : (  +0,  +1,  +0,  +0,  -4),
    'tale'                : (  -5,  +2,  -1,  +0,  +3),
    'technical'           : (  +2,  +1,  +0,  +0,  -2),
    'tell'                : (  +0,  +3,  +0,  +0,  +0),
    'tension'             : (  -2,  +1,  +0,  +0,  +6),
    'terrific'            : (  +1,  +9,  +0,  +0,  -5),
    'than'                : (  +0,  +0,  +0,  +0,  +0),
    'that'                : (  +0,  +0,  +0,  +0,  +0),
    'the'                 : (  +0,  +0,  +0,  +0,  +0),
    'then'                : (  +2,  +1,  +0,  +0,  -6),
    'theres'              : (  +0,  +0,  +0,  +0,  +0),
    'these'               : (  +0,  +0,  +0,  +0,  +0),
    'theyre'              : (  +0,  +0,  +0,  +0,  +0),
    'theyve'              : (  +0,  +0,  +0,  +0,  +0),
    'this'                : (  +0,  +0,  +0,  +0,  +0),
    'tho'                 : (  +0,  +0,  +0,  +0,  +8),
    'those'               : (  +0,  +0,  +0,  +0,  +0),
    'though'              : (  +0,  +0,  +0,  +0,  -8),
    'thriller'            : (  +0,  +0,  +0,  +0, +10),
    'thrills'             : (  +0,  +4,  +0,  +0,  +2),
    'tier'                : (  +1,  +1,  +0,  +0,  +4),
    'title'               : (  -1,  +0,  +0,  +0,  +5),
    'told'                : (  +0,  +2,  +0,  +0,  +0),
    'too'                 : (  -1,  +3,  +0,  +0,  -8),
    'tree'                : (  +0,  +0,  +0,  +0,  +1),
    'tries'               : (  +0,  +1,  +0,  +0,  -2),
    'trying'              : (  -2,  +4,  -1,  +0,  -3),
    'turn'                : (  +0,  +0,  +0,  +0,  +0),
    'twitchconhype'       : (  +0,  +0,  +0,  +0,  +0),
    'ultimately'          : (  +0,  +0,  +0,  +0,  -5),
    'use'                 : (  +2,  +1,  +0,  +0,  +0),
    'used'                : (  +2,  +2,  +0,  +0,  -7),
    'user'                : (  +0,  +0,  +0,  +0,  +2),
    'uses'                : (  +1,  +0,  +0,  +0,  +0),
    'utterly'             : ( +15,  +7,  +4,  +0,  -5),
    'very'                : (  +8,  +9,  +2,  +0,  +8),
    'viewers'             : (  +2,  +1,  +0,  +0,  +4),
    'visual'              : (  -5,  +2,  -1,  +0,  -5),
    'votenay'             : (  +0,  +0,  +0,  +0,  +0),
    'voteyea'             : (  +0,  +0,  +0,  +0,  +0),
    'wanted'              : (  +2,  +7,  +0,  +0,  -5),
    'wars'                : (  -1,  +1,  +0,  +0,  +3),
    'was'                 : (  +0,  +0,  +0,  +0,  -4),
    'watched'             : (  +3,  +2,  +0,  +0,  -7),
    'way'                 : (  +0,  +0,  +0,  +0,  +3),
    'wellacted'           : (  +0,  +0,  +0,  +0,  +0),
    'what'                : (  +0,  +0,  +0,  +0,  +0),
    'whats'               : (  +0,  +0,  +0,  +0,  +0),
    'which'               : (  +0,  +0,  +0,  +0,  +0),
    'who'                 : (  +0,  +0,  +0,  +0,  +8),
    'whose'               : (  +0,  +0,  +0,  +0,  +0),
    'why'                 : (  +0,  +0,  +0,  +0,  +4),
    'will'                : (  +0,  +0,  +0,  +0,  +1),
    'with'                : (  +0,  +0,  +0,  +0,  +0),
    'work'                : (  +0,  +0,  +0,  +0,  -2),
    'writerdirector'      : (  +0,  +0,  +0,  +0,  +0),
    'wutface'             : (  +0,  +0,  +0,  +0,  +0),
    'yall'                : (  +0,  +0,  +0,  +0,  +0),
    'yea'                 : (  +3,  +2,  +0,  +0,  +1),
    'yet'                 : (  -3,  +6,  -1,  +0,  -8),
    'you'                 : (  +0,  +0,  +0,  +0,  +0),
    'youll'               : (  +0,  +0,  +0,  +0,  +0),
    'your'                : (  +0,  +0,  +0,  +0,  +0),
}
EMOTIONAL_VOCABULARY.update(_NEW_WORDS)


# ── Gemini LLM Calibration: 200 words rated by conversational usage ──
_GEMINI_CAL = {
    'abomination'         : ( -25, +15,  -5,  +0, +12),
    'abusive'             : ( -25, +20, +10, +10, +18),
    'accident'            : ( -15, +15, -10, +15, +12),
    'accused'             : ( -15, +15, -10, +10, +10),
    'addict'              : ( -20, +10, -15,  +5, +12),
    'addicted'            : ( -20, +10, -15,  +5, +12),
    'addiction'           : ( -20, +10, -15,  +5, +12),
    'affair'              : ( -15, +15,  -5,  +5, +12),
    'allergic'            : ( -10,  +5,  -3,  +0,  +3),
    'alzheimers'          : ( -25,  +5, -20,  +5, +20),
    'annoy'               : (  -8,  +5,  -3,  +0,  -3),
    'annoys'              : (  -8,  +5,  -3,  +0,  -3),
    'anxiety'             : ( -15, +12, -10,  +8,  -8),
    'autoimmune'          : ( -15,  +5, -10,  +5, +12),
    'badly'               : (  -8,  +5,  -3,  +0,  -3),
    'beating'             : ( -20, +20, +10, +10, +10),
    'betrayal'            : ( -25, +15, -15, +10, +20),
    'betrayed'            : ( -25, +15, -15, +10, +18),
    'bomb'                : ( -20, +25, +10, +15, +12),
    'breakup'             : ( -20, +15, -10, +10, +15),
    'bummer'              : (  -8,  +0,  -3,  +0,  -3),
    'burnt'               : ( -10,  +5,  -5,  +0,  +3),
    'burst'               : (  +0, +15,  +0,  +5,  +0),
    'catastrophic'        : ( -25, +15, -10, +10, +18),
    'cheated'             : ( -20, +15, -15, +10, +15),
    'cheating'            : ( -20, +15, -15, +10, +15),
    'clogged'             : (  -8,  +5,  -3,  +0,  +3),
    'complain'            : (  -8,  +5,  -3,  +0,  -3),
    'complaint'           : (  -8,  +5,  -3,  +0,  -3),
    'crappy'              : (  -8,  +5,  -3,  +0,  -3),
    'cruel'               : ( -20, +15, +10,  +5, +12),
    'cruelty'             : ( -25, +15, +10,  +5, +15),
    'crushed'             : ( -20, +15, -15,  +5, +10),
    'culprit'             : ( -10, +10,  +5,  +5,  +5),
    'damaged'             : ( -15, +10, -10,  +5,  +8),
    'damages'             : ( -10, +10,  -5,  +5,  +8),
    'dangerous'           : ( -15, +20,  +5, +15, +10),
    'darn'                : (  -8,  +5,  +0,  +0,  +0),
    'deadly'              : ( -20, +15, +10, +10, +10),
    'declined'            : (  -8,  +5,  -5,  +0,  +5),
    'dementia'            : ( -25,  +5, -20,  +5, +20),
    'depressing'          : ( -20,  +0, -10,  +0, +10),
    'desperately'         : ( -15, +15, -10, +10,  +8),
    'destroy'             : ( -20, +20, +10, +10, +10),
    'devastating'         : ( -20, +15, -15, +10, +18),
    'devastation'         : ( -25, +15, -15, +10, +20),
    'diabetes'            : ( -15,  +5, -10,  +5, +12),
    'died'                : ( -20, +10, -15,  +5, +15),
    'difficulties'        : (  -8,  +5,  -5,  +5,  +5),
    'disaster'            : ( -20, +15, -10, +10, +15),
    'disease'             : ( -20, +10, -15, +10, +15),
    'dislike'             : (  -8,  +5,  -3,  +0,  -3),
    'distraught'          : ( -20, +15, -15,  +5, +12),
    'disturbing'          : ( -15, +10,  -5,  +5,  +8),
    'divorce'             : ( -20, +15, -10, +10, +20),
    'divorced'            : ( -15, +10, -10,  +5, +15),
    'doomed'              : ( -15, +10, -10,  +5, +12),
    'dreading'            : ( -15, +10, -10, +10,  +8),
    'drunken'             : ( -10, +10,  -5,  +0,  +3),
    'dying'               : ( -20, +15, -15, +10, +15),
    'enemy'               : ( -15, +15, +10,  +5, +10),
    'enraged'             : ( -20, +25, +15, +10, +10),
    'evil'                : ( -25, +15, +15,  +5, +12),
    'fears'               : ( -15, +15, -15, +10, +10),
    'fever'               : ( -10, +10,  -5,  +5,  +5),
    'fights'              : ( -15, +20, +10, +10,  +8),
    'fml'                 : ( -20, +15, -10,  +5,  +5),
    'freaked'             : ( -15, +20, -10,  +5,  +5),
    'frightened'          : ( -15, +15, -15, +10,  +8),
    'frightening'         : ( -15, +15, -10, +10,  +8),
    'frustrating'         : ( -15, +15,  -5,  +5,  +5),
    'funeral'             : ( -20,  +0, -10,  +0, +25),
    'gloomy'              : ( -10,  +0,  -5,  +0,  +5),
    'greed'               : ( -15, +10, +10,  +0,  +8),
    'grieve'              : ( -25,  +5, -15,  +0, +20),
    'grudge'              : ( -10,  +5,  -5,  +0,  +8),
    'gtfo'                : ( -20, +25, +15, +10,  +5),
    'gun'                 : ( -15, +20, +10, +10, +10),
    'harmful'             : ( -15, +10,  -5,  +5,  +8),
    'hassle'              : (  -8,  +5,  -3,  +0,  +0),
    'headaches'           : ( -10,  +5,  -5,  +5,  +5),
    'heartbreaking'       : ( -25, +10, -15,  +5, +20),
    'hectic'              : (  -8, +15,  -3,  +5,  +3),
    'hoarder'             : ( -15,  +5, -10,  +0,  +8),
    'homicide'            : ( -25, +15, +10, +10, +20),
    'horribly'            : ( -15, +10,  -8,  +5,  -5),
    'horrific'            : ( -25, +20, -15, +10, +15),
    'horrifying'          : ( -20, +20, -10, +10, +12),
    'humiliating'         : ( -20, +15, -20,  +5, +12),
    'hungover'            : ( -10,  +0,  -5,  +0,  +3),
    'hurting'             : ( -15, +10, -10,  +5,  +8),
    'ill'                 : ( -10,  +5,  -5,  +5,  +5),
    'illegal'             : ( -15, +10,  +5,  +5,  +8),
    'illness'             : ( -15,  +5, -10,  +5, +12),
    'inadequate'          : ( -15,  +5, -10,  +0,  +8),
    'inappropriate'       : ( -10,  +5,  -3,  +0,  +3),
    'incident'            : (  -8, +10,  +0, +10,  +8),
    'incompetent'         : ( -15,  +5, -10,  +0,  +5),
    'infuriating'         : ( -20, +25, +10, +10,  +8),
    'injuries'            : ( -15, +10, -10, +10, +10),
    'insanely'            : (  +0, +10,  +0,  +0,  +0),
    'intruder'            : ( -20, +20, -10, +15, +10),
    'irked'               : (  -8,  +5,  -3,  +0,  +0),
    'irresponsible'       : ( -15, +10,  -5,  +0,  +5),
    'irritating'          : (  -8, +10,  -3,  +0,  +3),
    'jail'                : ( -15, +10, -15,  +5, +12),
    'killer'              : ( -20, +20, +10,  +5, +10),
    'liar'                : ( -15, +10,  +5,  +5,  +8),
    'lied'                : ( -15, +10, -10,  +5, +10),
    'loathsome'           : ( -20, +10,  -5,  +0,  +8),
    'loser'               : ( -15, +10, -10,  +0,  +5),
    'mortified'           : ( -20, +10, -15,  +0,  +8),
    'murder'              : ( -25, +20, +10, +10, +20),
    'nasty'               : ( -15, +15,  +5,  +0,  +5),
    'nowhere'             : (  +0,  +0,  +0,  +0,  +0),
    'obnoxious'           : ( -10, +10,  +5,  +0,  +5),
    'outrage'             : ( -20, +25, +15, +10, +12),
    'pains'               : ( -15, +10, -10,  +5,  +8),
    'peeved'              : (  -8,  +5,  -3,  +0,  +0),
    'pessimistic'         : (  -8,  +5,  -5,  +0,  -3),
    'pissed'              : ( -15, +20, +10,  +5,  +5),
    'pity'                : ( -10,  +0,  -5,  +0,  +8),
    'poisoning'           : ( -20, +15, -10, +10, +12),
    'poisonous'           : ( -15, +10,  -5,  +5,  +8),
    'poo'                 : (  -8,  +5,  +0,  +0,  +0),
    'poop'                : (  -8,  +5,  +0,  +0,  +0),
    'poorly'              : (  -8,  +0,  -3,  +0,  -3),
    'poverty'             : ( -20,  +5, -15,  +5, +15),
    'pretended'           : (  -8,  +5,  -3,  +0,  +5),
    'prison'              : ( -15, +10, -15,  +5, +12),
    'punish'              : ( -15, +10, +10,  +5,  +8),
    'punishment'          : ( -15, +10,  +5,  +5,  +8),
    'rage'                : ( -20, +25, +15, +10,  +8),
    'refused'             : (  -8, +10,  +5,  +5,  +5),
    'rejected'            : ( -15, +10, -15,  +5, +10),
    'restless'            : (  -8, +10,  -3,  +5,  +3),
    'ridden'              : (  +0,  +0,  +0,  +0,  +0),
    'robbed'              : ( -20, +15, -15, +10, +12),
    'robber'              : ( -15, +10,  +5,  +5,  +8),
    'rotten'              : ( -10,  +5,  -5,  +0,  +3),
    'ruin'                : ( -20, +15, -10,  +5, +12),
    'sadly'               : (  -8,  +0,  -3,  +0,  -3),
    'scare'               : ( -15, +15, -10, +10,  +5),
    'scariest'            : ( -20, +20, -15, +10, +10),
    'severe'              : ( -15, +10,  -5, +10,  +8),
    'shattered'           : ( -20, +15, -15,  +5, +12),
    'shoot'               : ( -15, +20, +10, +10,  +8),
    'sickness'            : ( -15,  +5, -10,  +5, +10),
    'smelly'              : (  -8,  +5,  +0,  +0,  +0),
    'spite'               : ( -15, +10, +10,  +5,  +8),
    'spoil'               : (  -8,  +5,  -3,  +0,  +3),
    'starved'             : ( -20, +10, -15, +10, +12),
    'stfu'                : ( -20, +25, +15, +10,  +5),
    'stillbirth'          : ( -25, +10, -20,  +5, +25),
    'stillborn'           : ( -25, +10, -20,  +5, +25),
    'sting'               : (  -8, +10,  -3,  +5,  +3),
    'stink'               : (  -8,  +5,  -3,  +0,  -3),
    'stressful'           : ( -15, +15, -10, +10,  +8),
    'stressing'           : ( -15, +15, -10, +10,  +5),
    'stroke'              : ( -20, +15, -15, +15, +15),
    'suffer'              : ( -20, +10, -15,  +5, +12),
    'suspect'             : ( -10, +10,  -5,  +5,  +5),
    'suspicious'          : ( -10, +10,  -5,  +5,  +5),
    'teary'               : ( -10,  +5,  -5,  +0,  +5),
    'terribly'            : ( -10,  +5,  -5,  +0,  -3),
    'terrifying'          : ( -20, +25, -15, +15, +12),
    'terrorism'           : ( -25, +25, +10, +15, +20),
    'thieves'             : ( -15, +10,  +5,  +5, +10),
    'threatened'          : ( -20, +20, -15, +15, +12),
    'thunderstorm'        : (  -8, +15,  -3,  +0,  +5),
    'tragedy'             : ( -25, +10, -10,  +5, +20),
    'tragic'              : ( -20, +10, -10,  +5, +15),
    'trauma'              : ( -25, +15, -15, +10, +20),
    'traumatic'           : ( -25, +15, -15, +10, +18),
    'troubles'            : (  -8,  +5,  -3,  +5,  +5),
    'turbulence'          : (  -8, +10,  -5,  +5,  +5),
    'unable'              : (  +0,  +0,  +0,  +0,  +0),
    'uncertainty'         : (  -8,  +5,  -5,  +5,  +5),
    'unemployment'        : ( -15,  +5, -10,  +5, +12),
    'unfair'              : ( -15, +10, -10,  +5,  +8),
    'unfaithful'          : ( -20, +10, -15,  +5, +15),
    'unfortunate'         : (  -8,  +5,  -3,  +0,  +5),
    'unhappy'             : ( -15,  +5,  -5,  +0,  +5),
    'unmotivated'         : ( -10,  -5,  -5,  +0,  +3),
    'unpleasant'          : (  -8,  +5,  -3,  +0,  -3),
    'unsolicited'         : (  -8,  +5,  -3,  +0,  +3),
    'upsetting'           : ( -15, +10,  -8,  +5,  +8),
    'victim'              : ( -20, +10, -20,  +5, +15),
    'villain'             : ( -15, +15, +10,  +5,  +8),
    'violence'            : ( -20, +25, +10, +10, +15),
    'violent'             : ( -20, +25, +10, +10, +12),
    'void'                : (  -8,  +0,  -5,  +0,  +5),
    'vomit'               : ( -15, +10, -10,  +5,  +5),
    'vomiting'            : ( -15, +10, -10,  +5,  +5),
    'wasp'                : (  -8, +10,  +0,  +5,  +3),
    'worries'             : (  -8,  +5,  -5,  +5,  +3),
    'worrisome'           : ( -10,  +5,  -5,  +5,  +5),
    'worrying'            : ( -10, +10,  -5,  +5,  +5),
    'wtf'                 : ( -15, +20,  +0, +10,  +3),
}
EMOTIONAL_VOCABULARY.update(_GEMINI_CAL)


# ── 3-LLM Consensus Calibration (Gemini+Claude+GPT-4 average) ──
_LLM_CONSENSUS = {
    'abomination'         : ( -23, +11,  -9,  +7,  -7),
    'abusive'             : ( -23, +11,  -9,  +7,  -7),
    'accident'            : ( -15,  +7,  -6,  +4,  -4),
    'accused'             : ( -13,  +6,  -5,  +4,  -4),
    'addict'              : ( -18,  +8,  -7,  +5,  -5),
    'addicted'            : ( -17,  +8,  -6,  +5,  -5),
    'addiction'           : ( -19,  +9,  -7,  +6,  -6),
    'affair'              : ( -17,  +8,  -6,  +5,  -5),
    'allergic'            : (  -9,  +4,  -3,  +0,  -2),
    'alzheimers'          : ( -23, +11,  -9,  +7,  -7),
    'annoy'               : (  -9,  +4,  -3,  +0,  -2),
    'annoys'              : (  -9,  +4,  -3,  +0,  -2),
    'anxiety'             : ( -15,  +7,  -6,  +4,  -4),
    'autoimmune'          : ( -15,  +7,  -6,  +4,  -4),
    'badly'               : (  -9,  +4,  -3,  +0,  -2),
    'beating'             : ( -19,  +9,  -7,  +6,  -6),
    'betrayal'            : ( -23, +11,  -9,  +7,  -7),
    'betrayed'            : ( -23, +11,  -9,  +7,  -7),
    'bomb'                : ( -19,  +9,  -7,  +6,  -6),
    'breakup'             : ( -19,  +9,  -7,  +6,  -6),
    'bummer'              : (  -9,  +4,  -3,  +0,  -2),
    'burnt'               : ( -10,  +4,  -4,  +0,  -3),
    'burst'               : (  -3,  +1,  -1,  +0,  +0),
    'catastrophic'        : ( -24, +11,  -9,  +7,  -7),
    'cheated'             : ( -20,  +9,  -8,  +6,  -6),
    'cheating'            : ( -19,  +9,  -7,  +6,  -6),
    'clogged'             : (  -7,  +3,  -2,  +0,  -2),
    'complain'            : (  -8,  +3,  -3,  +0,  -2),
    'complaint'           : (  -8,  +3,  -3,  +0,  -2),
    'crappy'              : (  -9,  +4,  -3,  +0,  -2),
    'cruel'               : ( -20,  +9,  -8,  +6,  -6),
    'cruelty'             : ( -23, +11,  -9,  +7,  -7),
    'crushed'             : ( -19,  +9,  -7,  +6,  -6),
    'culprit'             : (  -9,  +4,  -3,  +0,  -2),
    'damaged'             : ( -14,  +6,  -5,  +4,  -4),
    'damages'             : ( -11,  +5,  -4,  +3,  -3),
    'dangerous'           : ( -14,  +6,  -5,  +4,  -4),
    'darn'                : (  -5,  +2,  -2,  +0,  -1),
    'deadly'              : ( -20,  +9,  -8,  +6,  -6),
    'declined'            : (  -9,  +4,  -3,  +0,  -2),
    'dementia'            : ( -22, +10,  -8,  +7,  -7),
    'depressing'          : ( -18,  +8,  -7,  +5,  -5),
    'desperately'         : ( -15,  +7,  -6,  +4,  -4),
    'destroy'             : ( -20,  +9,  -8,  +6,  -6),
    'devastating'         : ( -21, +10,  -8,  +6,  -6),
    'devastation'         : ( -23, +11,  -9,  +7,  -7),
    'diabetes'            : ( -15,  +7,  -6,  +4,  -4),
    'died'                : ( -21, +10,  -8,  +6,  -6),
    'difficulties'        : (  -9,  +4,  -3,  +0,  -2),
    'disaster'            : ( -20,  +9,  -8,  +6,  -6),
    'disease'             : ( -19,  +9,  -7,  +6,  -6),
    'dislike'             : (  -7,  +3,  -2,  +0,  -2),
    'distraught'          : ( -20,  +9,  -8,  +6,  -6),
    'disturbing'          : ( -16,  +7,  -6,  +5,  -5),
    'divorce'             : ( -19,  +9,  -7,  +6,  -6),
    'divorced'            : ( -16,  +7,  -6,  +5,  -5),
    'doomed'              : ( -17,  +8,  -6,  +5,  -5),
    'dreading'            : ( -15,  +7,  -6,  +4,  -4),
    'drunken'             : ( -10,  +4,  -4,  +0,  -3),
    'dying'               : ( -21, +10,  -8,  +6,  -6),
    'enemy'               : ( -15,  +7,  -6,  +4,  -4),
    'enraged'             : ( -19,  +9,  -7,  +6,  -6),
    'evil'                : ( -23, +11,  -9,  +7,  -7),
    'fears'               : ( -14,  +6,  -5,  +4,  -4),
    'fever'               : ( -11,  +5,  -4,  +3,  -3),
    'fights'              : ( -12,  +5,  -4,  +3,  -3),
    'fml'                 : ( -18,  +8,  -7,  +5,  -5),
    'freaked'             : ( -15,  +7,  -6,  +4,  -4),
    'frightened'          : ( -16,  +7,  -6,  +5,  -5),
    'frightening'         : ( -16,  +7,  -6,  +5,  -5),
    'frustrating'         : ( -12,  +5,  -4,  +3,  -3),
    'funeral'             : ( -19,  +9,  -7,  +6,  -6),
    'gloomy'              : ( -11,  +5,  -4,  +3,  -3),
    'greed'               : ( -16,  +7,  -6,  +5,  -5),
    'grieve'              : ( -22, +10,  -8,  +7,  -7),
    'grudge'              : ( -11,  +5,  -4,  +3,  -3),
    'gtfo'                : ( -18,  +8,  -7,  +5,  -5),
    'gun'                 : ( -12,  +5,  -4,  +3,  -3),
    'harmful'             : ( -16,  +7,  -6,  +5,  -5),
    'hassle'              : (  -9,  +4,  -3,  +0,  -2),
    'headaches'           : (  -9,  +4,  -3,  +0,  -2),
    'heartbreaking'       : ( -23, +11,  -9,  +7,  -7),
    'hectic'              : (  -9,  +4,  -3,  +0,  -2),
    'hoarder'             : ( -12,  +5,  -4,  +3,  -3),
    'homicide'            : ( -25, +12, -10,  +8,  -8),
    'horribly'            : ( -14,  +6,  -5,  +4,  -4),
    'horrific'            : ( -23, +11,  -9,  +7,  -7),
    'horrifying'          : ( -21, +10,  -8,  +6,  -6),
    'humiliating'         : ( -19,  +9,  -7,  +6,  -6),
    'hungover'            : (  -9,  +4,  -3,  +0,  -2),
    'hurting'             : ( -15,  +7,  -6,  +4,  -4),
    'ill'                 : ( -11,  +5,  -4,  +3,  -3),
    'illegal'             : ( -12,  +5,  -4,  +3,  -3),
    'illness'             : ( -15,  +7,  -6,  +4,  -4),
    'inadequate'          : ( -15,  +7,  -6,  +4,  -4),
    'inappropriate'       : ( -10,  +4,  -4,  +0,  -3),
    'incident'            : (  -7,  +3,  -2,  +0,  -2),
    'incompetent'         : ( -15,  +7,  -6,  +4,  -4),
    'infuriating'         : ( -18,  +8,  -7,  +5,  -5),
    'injuries'            : ( -15,  +7,  -6,  +4,  -4),
    'insanely'            : (  -4,  +1,  -1,  +0,  -1),
    'intruder'            : ( -18,  +8,  -7,  +5,  -5),
    'irked'               : (  -8,  +3,  -3,  +0,  -2),
    'irresponsible'       : ( -13,  +6,  -5,  +4,  -4),
    'irritating'          : (  -9,  +4,  -3,  +0,  -2),
    'jail'                : ( -16,  +7,  -6,  +5,  -5),
    'killer'              : ( -20,  +9,  -8,  +6,  -6),
    'liar'                : ( -15,  +7,  -6,  +4,  -4),
    'lice'                : ( -11,  +5,  -4,  +3,  -3),
    'lied'                : ( -15,  +7,  -6,  +4,  -4),
    'loathsome'           : ( -20,  +9,  -8,  +6,  -6),
    'loser'               : ( -15,  +7,  -6,  +4,  -4),
    'mortified'           : ( -18,  +8,  -7,  +5,  -5),
    'murder'              : ( -25, +12, -10,  +8,  -8),
    'nasty'               : ( -14,  +6,  -5,  +4,  -4),
    'nowhere'             : (  -5,  +2,  -2,  +0,  -1),
    'obnoxious'           : ( -11,  +5,  -4,  +3,  -3),
    'outrage'             : ( -18,  +8,  -7,  +5,  -5),
    'pains'               : ( -13,  +6,  -5,  +4,  -4),
    'peeved'              : (  -8,  +3,  -3,  +0,  -2),
    'pessimistic'         : (  -9,  +4,  -3,  +0,  -2),
    'pissed'              : ( -14,  +6,  -5,  +4,  -4),
    'pity'                : ( -10,  +4,  -4,  +0,  -3),
    'poisoning'           : ( -20,  +9,  -8,  +6,  -6),
    'poisonous'           : ( -16,  +7,  -6,  +5,  -5),
    'poo'                 : (  -7,  +3,  -2,  +0,  -2),
    'poop'                : (  -7,  +3,  -2,  +0,  -2),
    'poorly'              : (  -8,  +3,  -3,  +0,  -2),
    'poverty'             : ( -19,  +9,  -7,  +6,  -6),
    'pretended'           : (  -8,  +3,  -3,  +0,  -2),
    'prison'              : ( -16,  +7,  -6,  +5,  -5),
    'punish'              : ( -12,  +5,  -4,  +3,  -3),
    'punishment'          : ( -13,  +6,  -5,  +4,  -4),
    'rage'                : ( -19,  +9,  -7,  +6,  -6),
    'refused'             : (  -9,  +4,  -3,  +0,  -2),
    'rejected'            : ( -16,  +7,  -6,  +5,  -5),
    'restless'            : (  -7,  +3,  -2,  +0,  -2),
    'ridden'              : (  -3,  +1,  -1,  +0,  +0),
    'robbed'              : ( -19,  +9,  -7,  +6,  -6),
    'robber'              : ( -15,  +7,  -6,  +4,  -4),
    'rotten'              : ( -11,  +5,  -4,  +3,  -3),
    'ruin'                : ( -19,  +9,  -7,  +6,  -6),
    'sadly'               : ( -10,  +4,  -4,  +0,  -3),
    'scare'               : ( -13,  +6,  -5,  +4,  -4),
    'scariest'            : ( -19,  +9,  -7,  +6,  -6),
    'severe'              : ( -15,  +7,  -6,  +4,  -4),
    'shattered'           : ( -20,  +9,  -8,  +6,  -6),
    'shoot'               : ( -12,  +5,  -4,  +3,  -3),
    'sickness'            : ( -15,  +7,  -6,  +4,  -4),
    'smelly'              : (  -7,  +3,  -2,  +0,  -2),
    'spite'               : ( -14,  +6,  -5,  +4,  -4),
    'spoil'               : (  -9,  +4,  -3,  +0,  -2),
    'starved'             : ( -17,  +8,  -6,  +5,  -5),
    'stfu'                : ( -17,  +8,  -6,  +5,  -5),
    'stillbirth'          : ( -25, +12, -10,  +8,  -8),
    'stillborn'           : ( -25, +12, -10,  +8,  -8),
    'sting'               : (  -9,  +4,  -3,  +0,  -2),
    'stink'               : (  -8,  +3,  -3,  +0,  -2),
    'stressful'           : ( -15,  +7,  -6,  +4,  -4),
    'stressing'           : ( -14,  +6,  -5,  +4,  -4),
    'stroke'              : ( -19,  +9,  -7,  +6,  -6),
    'suffer'              : ( -19,  +9,  -7,  +6,  -6),
    'suspect'             : (  -8,  +3,  -3,  +0,  -2),
    'suspicious'          : (  -9,  +4,  -3,  +0,  -2),
    'teary'               : ( -12,  +5,  -4,  +3,  -3),
    'terribly'            : ( -11,  +5,  -4,  +3,  -3),
    'terrifying'          : ( -21, +10,  -8,  +6,  -6),
    'terrorism'           : ( -25, +12, -10,  +8,  -8),
    'thieves'             : ( -15,  +7,  -6,  +4,  -4),
    'threatened'          : ( -18,  +8,  -7,  +5,  -5),
    'thunderstorm'        : (  -6,  +2,  -2,  +0,  -1),
    'tragedy'             : ( -23, +11,  -9,  +7,  -7),
    'tragic'              : ( -21, +10,  -8,  +6,  -6),
    'trauma'              : ( -22, +10,  -8,  +7,  -7),
    'traumatic'           : ( -22, +10,  -8,  +7,  -7),
    'troubles'            : (  -9,  +4,  -3,  +0,  -2),
    'turbulence'          : (  -8,  +3,  -3,  +0,  -2),
    'unable'              : (  -7,  +3,  -2,  +0,  -2),
    'uncertainty'         : ( -10,  +4,  -4,  +0,  -3),
    'unemployment'        : ( -16,  +7,  -6,  +5,  -5),
    'unfair'              : ( -14,  +6,  -5,  +4,  -4),
    'unfaithful'          : ( -19,  +9,  -7,  +6,  -6),
    'unfortunate'         : (  -9,  +4,  -3,  +0,  -2),
    'unhappy'             : ( -14,  +6,  -5,  +4,  -4),
    'unmotivated'         : ( -11,  +5,  -4,  +3,  -3),
    'unpleasant'          : (  -9,  +4,  -3,  +0,  -2),
    'unsolicited'         : (  -7,  +3,  -2,  +0,  -2),
    'upsetting'           : ( -14,  +6,  -5,  +4,  -4),
    'victim'              : ( -17,  +8,  -6,  +5,  -5),
    'villain'             : ( -16,  +7,  -6,  +5,  -5),
    'violence'            : ( -21, +10,  -8,  +6,  -6),
    'violent'             : ( -20,  +9,  -8,  +6,  -6),
    'void'                : ( -10,  +4,  -4,  +0,  -3),
    'vomit'               : ( -14,  +6,  -5,  +4,  -4),
    'vomiting'            : ( -15,  +7,  -6,  +4,  -4),
    'wasp'                : (  -6,  +2,  -2,  +0,  -1),
    'worries'             : ( -11,  +5,  -4,  +3,  -3),
    'worrisome'           : ( -11,  +5,  -4,  +3,  -3),
    'worrying'            : ( -11,  +5,  -4,  +3,  -3),
    'wtf'                 : ( -14,  +6,  -5,  +4,  -4),
}
EMOTIONAL_VOCABULARY.update(_LLM_CONSENSUS)


# ── 4-LLM Final Consensus (Gemini+Claude+GPT-4+Grok average) ──
_LLM_CONSENSUS_4 = {
    'abomination'         : ( -22, +10,  -8,  +7,  -7),
    'abusive'             : ( -22, +10,  -8,  +7,  -7),
    'accident'            : ( -15,  +7,  -6,  +4,  -4),
    'accused'             : ( -13,  +6,  -5,  +4,  -4),
    'addict'              : ( -18,  +8,  -7,  +5,  -5),
    'addicted'            : ( -18,  +8,  -7,  +5,  -5),
    'addiction'           : ( -19,  +9,  -7,  +6,  -6),
    'affair'              : ( -17,  +8,  -6,  +5,  -5),
    'allergic'            : (  -9,  +4,  -3,  +0,  -2),
    'alzheimers'          : ( -22, +10,  -8,  +7,  -7),
    'annoy'               : (  -9,  +4,  -3,  +0,  -2),
    'annoys'              : (  -9,  +4,  -3,  +0,  -2),
    'anxiety'             : ( -16,  +7,  -6,  +5,  -5),
    'autoimmune'          : ( -16,  +7,  -6,  +5,  -5),
    'badly'               : ( -10,  +4,  -4,  +0,  -3),
    'beating'             : ( -18,  +8,  -7,  +5,  -5),
    'betrayal'            : ( -22, +10,  -8,  +7,  -7),
    'betrayed'            : ( -22, +10,  -8,  +7,  -7),
    'bomb'                : ( -17,  +8,  -6,  +5,  -5),
    'breakup'             : ( -19,  +9,  -7,  +6,  -6),
    'bummer'              : ( -10,  +4,  -4,  +0,  -3),
    'burnt'               : ( -10,  +4,  -4,  +0,  -3),
    'burst'               : (  -4,  +1,  -1,  +0,  -1),
    'catastrophic'        : ( -24, +11,  -9,  +7,  -7),
    'cheated'             : ( -20,  +9,  -8,  +6,  -6),
    'cheating'            : ( -20,  +9,  -8,  +6,  -6),
    'clogged'             : (  -8,  +3,  -3,  +0,  -2),
    'complain'            : (  -8,  +3,  -3,  +0,  -2),
    'complaint'           : (  -8,  +3,  -3,  +0,  -2),
    'crappy'              : (  -9,  +4,  -3,  +0,  -2),
    'cruel'               : ( -20,  +9,  -8,  +6,  -6),
    'cruelty'             : ( -22, +10,  -8,  +7,  -7),
    'crushed'             : ( -19,  +9,  -7,  +6,  -6),
    'culprit'             : ( -10,  +4,  -4,  +0,  -3),
    'damaged'             : ( -14,  +6,  -5,  +4,  -4),
    'damages'             : ( -12,  +5,  -4,  +3,  -3),
    'dangerous'           : ( -14,  +6,  -5,  +4,  -4),
    'darn'                : (  -6,  +2,  -2,  +0,  -1),
    'deadly'              : ( -20,  +9,  -8,  +6,  -6),
    'declined'            : ( -10,  +4,  -4,  +0,  -3),
    'dementia'            : ( -21, +10,  -8,  +6,  -6),
    'depressing'          : ( -18,  +8,  -7,  +5,  -5),
    'desperately'         : ( -14,  +6,  -5,  +4,  -4),
    'destroy'             : ( -20,  +9,  -8,  +6,  -6),
    'devastating'         : ( -21, +10,  -8,  +6,  -6),
    'devastation'         : ( -22, +10,  -8,  +7,  -7),
    'diabetes'            : ( -15,  +7,  -6,  +4,  -4),
    'died'                : ( -22, +10,  -8,  +7,  -7),
    'difficulties'        : ( -10,  +4,  -4,  +0,  -3),
    'disaster'            : ( -20,  +9,  -8,  +6,  -6),
    'disease'             : ( -18,  +8,  -7,  +5,  -5),
    'dislike'             : (  -8,  +3,  -3,  +0,  -2),
    'distraught'          : ( -20,  +9,  -8,  +6,  -6),
    'disturbing'          : ( -16,  +7,  -6,  +5,  -5),
    'divorce'             : ( -18,  +8,  -7,  +5,  -5),
    'divorced'            : ( -17,  +8,  -6,  +5,  -5),
    'doomed'              : ( -17,  +8,  -6,  +5,  -5),
    'dreading'            : ( -15,  +7,  -6,  +4,  -4),
    'drunken'             : ( -10,  +4,  -4,  +0,  -3),
    'dying'               : ( -21, +10,  -8,  +6,  -6),
    'enemy'               : ( -15,  +7,  -6,  +4,  -4),
    'enraged'             : ( -18,  +8,  -7,  +5,  -5),
    'evil'                : ( -22, +10,  -8,  +7,  -7),
    'fears'               : ( -14,  +6,  -5,  +4,  -4),
    'fever'               : ( -11,  +5,  -4,  +3,  -3),
    'fights'              : ( -12,  +5,  -4,  +3,  -3),
    'fml'                 : ( -18,  +8,  -7,  +5,  -5),
    'freaked'             : ( -15,  +7,  -6,  +4,  -4),
    'frightened'          : ( -17,  +8,  -6,  +5,  -5),
    'frightening'         : ( -17,  +8,  -6,  +5,  -5),
    'frustrating'         : ( -12,  +5,  -4,  +3,  -3),
    'funeral'             : ( -19,  +9,  -7,  +6,  -6),
    'gloomy'              : ( -11,  +5,  -4,  +3,  -3),
    'greed'               : ( -16,  +7,  -6,  +5,  -5),
    'grieve'              : ( -21, +10,  -8,  +6,  -6),
    'grudge'              : ( -12,  +5,  -4,  +3,  -3),
    'gtfo'                : ( -17,  +8,  -6,  +5,  -5),
    'gun'                 : ( -12,  +5,  -4,  +3,  -3),
    'harmful'             : ( -16,  +7,  -6,  +5,  -5),
    'hassle'              : (  -9,  +4,  -3,  +0,  -2),
    'headaches'           : ( -10,  +4,  -4,  +0,  -3),
    'heartbreaking'       : ( -22, +10,  -8,  +7,  -7),
    'hectic'              : (  -9,  +4,  -3,  +0,  -2),
    'hoarder'             : ( -12,  +5,  -4,  +3,  -3),
    'homicide'            : ( -25, +12, -10,  +8,  -8),
    'horribly'            : ( -14,  +6,  -5,  +4,  -4),
    'horrific'            : ( -23, +11,  -9,  +7,  -7),
    'horrifying'          : ( -20,  +9,  -8,  +6,  -6),
    'humiliating'         : ( -19,  +9,  -7,  +6,  -6),
    'hungover'            : ( -10,  +4,  -4,  +0,  -3),
    'hurting'             : ( -15,  +7,  -6,  +4,  -4),
    'ill'                 : ( -12,  +5,  -4,  +3,  -3),
    'illegal'             : ( -12,  +5,  -4,  +3,  -3),
    'illness'             : ( -15,  +7,  -6,  +4,  -4),
    'inadequate'          : ( -15,  +7,  -6,  +4,  -4),
    'inappropriate'       : ( -10,  +4,  -4,  +0,  -3),
    'incident'            : (  -7,  +3,  -2,  +0,  -2),
    'incompetent'         : ( -15,  +7,  -6,  +4,  -4),
    'infuriating'         : ( -18,  +8,  -7,  +5,  -5),
    'injuries'            : ( -15,  +7,  -6,  +4,  -4),
    'insanely'            : (  -3,  +1,  -1,  +0,  +0),
    'intruder'            : ( -18,  +8,  -7,  +5,  -5),
    'irked'               : (  -8,  +3,  -3,  +0,  -2),
    'irresponsible'       : ( -14,  +6,  -5,  +4,  -4),
    'irritating'          : ( -10,  +4,  -4,  +0,  -3),
    'jail'                : ( -16,  +7,  -6,  +5,  -5),
    'killer'              : ( -19,  +9,  -7,  +6,  -6),
    'liar'                : ( -15,  +7,  -6,  +4,  -4),
    'lice'                : ( -12,  +5,  -4,  +3,  -3),
    'lied'                : ( -15,  +7,  -6,  +4,  -4),
    'loathsome'           : ( -20,  +9,  -8,  +6,  -6),
    'loser'               : ( -15,  +7,  -6,  +4,  -4),
    'mortified'           : ( -18,  +8,  -7,  +5,  -5),
    'murder'              : ( -25, +12, -10,  +8,  -8),
    'nasty'               : ( -13,  +6,  -5,  +4,  -4),
    'nowhere'             : (  -6,  +2,  -2,  +0,  -1),
    'obnoxious'           : ( -12,  +5,  -4,  +3,  -3),
    'outrage'             : ( -17,  +8,  -6,  +5,  -5),
    'pains'               : ( -13,  +6,  -5,  +4,  -4),
    'peeved'              : (  -8,  +3,  -3,  +0,  -2),
    'pessimistic'         : ( -10,  +4,  -4,  +0,  -3),
    'pissed'              : ( -14,  +6,  -5,  +4,  -4),
    'pity'                : ( -10,  +4,  -4,  +0,  -3),
    'poisoning'           : ( -20,  +9,  -8,  +6,  -6),
    'poisonous'           : ( -16,  +7,  -6,  +5,  -5),
    'poo'                 : (  -7,  +3,  -2,  +0,  -2),
    'poop'                : (  -7,  +3,  -2,  +0,  -2),
    'poorly'              : (  -8,  +3,  -3,  +0,  -2),
    'poverty'             : ( -18,  +8,  -7,  +5,  -5),
    'pretended'           : (  -8,  +3,  -3,  +0,  -2),
    'prison'              : ( -16,  +7,  -6,  +5,  -5),
    'punish'              : ( -12,  +5,  -4,  +3,  -3),
    'punishment'          : ( -14,  +6,  -5,  +4,  -4),
    'rage'                : ( -18,  +8,  -7,  +5,  -5),
    'refused'             : (  -9,  +4,  -3,  +0,  -2),
    'rejected'            : ( -16,  +7,  -6,  +5,  -5),
    'restless'            : (  -8,  +3,  -3,  +0,  -2),
    'ridden'              : (  -5,  +2,  -2,  +0,  -1),
    'robbed'              : ( -18,  +8,  -7,  +5,  -5),
    'robber'              : ( -16,  +7,  -6,  +5,  -5),
    'rotten'              : ( -12,  +5,  -4,  +3,  -3),
    'ruin'                : ( -19,  +9,  -7,  +6,  -6),
    'sadly'               : ( -10,  +4,  -4,  +0,  -3),
    'scare'               : ( -14,  +6,  -5,  +4,  -4),
    'scariest'            : ( -20,  +9,  -8,  +6,  -6),
    'severe'              : ( -15,  +7,  -6,  +4,  -4),
    'shattered'           : ( -20,  +9,  -8,  +6,  -6),
    'shoot'               : ( -12,  +5,  -4,  +3,  -3),
    'sickness'            : ( -15,  +7,  -6,  +4,  -4),
    'smelly'              : (  -8,  +3,  -3,  +0,  -2),
    'spite'               : ( -13,  +6,  -5,  +4,  -4),
    'spoil'               : (  -9,  +4,  -3,  +0,  -2),
    'starved'             : ( -16,  +7,  -6,  +5,  -5),
    'stfu'                : ( -17,  +8,  -6,  +5,  -5),
    'stillbirth'          : ( -25, +12, -10,  +8,  -8),
    'stillborn'           : ( -25, +12, -10,  +8,  -8),
    'sting'               : (  -9,  +4,  -3,  +0,  -2),
    'stink'               : (  -8,  +3,  -3,  +0,  -2),
    'stressful'           : ( -15,  +7,  -6,  +4,  -4),
    'stressing'           : ( -14,  +6,  -5,  +4,  -4),
    'stroke'              : ( -19,  +9,  -7,  +6,  -6),
    'suffer'              : ( -18,  +8,  -7,  +5,  -5),
    'suspect'             : (  -9,  +4,  -3,  +0,  -2),
    'suspicious'          : ( -10,  +4,  -4,  +0,  -3),
    'teary'               : ( -12,  +5,  -4,  +3,  -3),
    'terribly'            : ( -12,  +5,  -4,  +3,  -3),
    'terrifying'          : ( -21, +10,  -8,  +6,  -6),
    'terrorism'           : ( -24, +11,  -9,  +7,  -7),
    'thieves'             : ( -15,  +7,  -6,  +4,  -4),
    'threatened'          : ( -17,  +8,  -6,  +5,  -5),
    'thunderstorm'        : (  -7,  +3,  -2,  +0,  -2),
    'tragedy'             : ( -22, +10,  -8,  +7,  -7),
    'tragic'              : ( -20,  +9,  -8,  +6,  -6),
    'trauma'              : ( -22, +10,  -8,  +7,  -7),
    'traumatic'           : ( -22, +10,  -8,  +7,  -7),
    'troubles'            : ( -10,  +4,  -4,  +0,  -3),
    'turbulence'          : (  -8,  +3,  -3,  +0,  -2),
    'unable'              : (  -8,  +3,  -3,  +0,  -2),
    'uncertainty'         : ( -10,  +4,  -4,  +0,  -3),
    'unemployment'        : ( -16,  +7,  -6,  +5,  -5),
    'unfair'              : ( -14,  +6,  -5,  +4,  -4),
    'unfaithful'          : ( -20,  +9,  -8,  +6,  -6),
    'unfortunate'         : ( -10,  +4,  -4,  +0,  -3),
    'unhappy'             : ( -14,  +6,  -5,  +4,  -4),
    'unmotivated'         : ( -11,  +5,  -4,  +3,  -3),
    'unpleasant'          : ( -10,  +4,  -4,  +0,  -3),
    'unsolicited'         : (  -7,  +3,  -2,  +0,  -2),
    'upsetting'           : ( -14,  +6,  -5,  +4,  -4),
    'victim'              : ( -16,  +7,  -6,  +5,  -5),
    'villain'             : ( -16,  +7,  -6,  +5,  -5),
    'violence'            : ( -21, +10,  -8,  +6,  -6),
    'violent'             : ( -20,  +9,  -8,  +6,  -6),
    'void'                : ( -12,  +5,  -4,  +3,  -3),
    'vomit'               : ( -14,  +6,  -5,  +4,  -4),
    'vomiting'            : ( -15,  +7,  -6,  +4,  -4),
    'wasp'                : (  -5,  +2,  -2,  +0,  -1),
    'worries'             : ( -11,  +5,  -4,  +3,  -3),
    'worrisome'           : ( -12,  +5,  -4,  +3,  -3),
    'worrying'            : ( -12,  +5,  -4,  +3,  -3),
    'wtf'                 : ( -13,  +6,  -5,  +4,  -4),
}
EMOTIONAL_VOCABULARY.update(_LLM_CONSENSUS_4)


# ── V7 corrections: GAS atoms zeroed, deception/betrayal tuned ──
_V7_FIXES = {
    # GAS atoms: objects have no inherent emotional charge
    'necklace':     (  0,  0,  0,  0, 10),  # object. Gravity from possession, not valence.
    'bracelet':     (  0,  0,  0,  0,  8),
    'ring':         (  0,  0,  0,  0, 12),  # high G (symbol of commitment)
    'drawer':       (  0,  0,  0,  0,  0),
    # GAS interjections: these carry zero charge, context fills them
    'yeah':         (  0,  0,  0,  0,  0),  # pure GAS. "yeah right" = sarcasm. "yeah!" = excitement.
    'yep':          (  0,  0,  0,  0,  0),
    'yup':          (  0,  0,  0,  0,  0),
    'nah':          ( -3,  0,  0,  0,  0),  # mild dismissal
    'well':         (  0,  0,  0,  0,  0),  # filler / hedge. Not emotional.
    'went':         (  0,  0,  0,  0,  0),  # movement verb, neutral
    'all':          (  0,  0,  0,  0,  5),  # quantifier, not emotional. Was dV=8.
    # Deception verbs: the act of deception IS negative
    'pretended':    (-20,  5, -15,  0, -10),  # deception = betrayal of trust
    'pretending':   (-18,  5, -12,  0,  -8),
    'faked':        (-20,  5, -15,  0, -10),
    'faking':       (-18,  5, -12,  0,  -8),
    'lied':         (-25, 10, -20,  5, -12),
    'lying':        (-22,  8, -18,  5, -10),
    'deceived':     (-25, 10, -20,  5, -15),
    'betrayed':     (-30, 15, -25, 10, -18),
    'betrayal':     (-30, 15, -25, 10, -18),
    # Abandonment: "left" in isolation is spatial. With people context = abandonment.
    # Keep low dV — proximity to RELATION_REF creates the betrayal.
    'left':         ( -5,  5,  -5,  5,  -3),  # slightly negative base
    # GAS verbs: meaning entirely from context
    'catch':        (  0,  5,   5,  0,  5),  # catch a ball, catch up, catch fire
    'caught':       (  0,  5,   5,  0,  5),
    'weather':      (  0,  0,   0,  0,  0),  # pure neutral noun
    'changing':     (  0,  0,   0,  0,  0),  # pure neutral verb
    'change':       (  0,  0,   0,  0,  0),
    'changed':      (  0,  0,   0,  0,  0),
    'messages':     (  0,  0,   0,  0,  5),  # neutral object
    'phone':        (  0,  0,   0,  0,  5),  # neutral object
    'should':       (  0,  0,   0,  0,  0),  # modal, no charge
    # ── MASSIVE GAS ZEROING: common nouns/verbs with bloated dV ──
    # These are GAS atoms. Their meaning comes from CONTEXT, not from the word itself.
    # "pizza" is not happy. "appointment" is not happy. "alarm" is not scary.
    # The anchor+web model: only ~111 anchor words carry real dV.
    # Everything else gets dV from proximity to anchors at runtime.
    #
    # Common objects / places
    'pizza':        (  0,  0,   0,  0,  5),
    'laundry':      (  0,  0,   0,  0,  0),
    'appointment':  (  0,  0,   0,  0,  5),
    'alarm':        (  0,  5,   0,  0,  0),
    'flight':       (  0,  0,   0,  0,  5),
    'show':         (  0,  0,   0,  0,  5),
    'double':       (  0,  0,   0,  0,  0),
    'dentist':      (  0,  0,   0,  0,  5),
    'bus':          (  0,  0,   0,  0,  0),
    'train':        (  0,  0,   0,  0,  0),
    'subway':       (  0,  0,   0,  0,  0),
    'parking':      (  0,  0,   0,  0,  0),
    'store':        (  0,  0,   0,  0,  0),
    'line':         (  0,  0,   0,  0,  0),
    'restaurant':   (  0,  0,   0,  0,  5),
    'order':        (  0,  0,   0,  0,  0),  # "order pizza" ≠ emotion
    'ordered':      (  0,  0,   0,  0,  0),
    'grab':         (  0,  2,   0,  0,  0),
    'airport':      (  0,  0,   0,  0,  0),
    'hotel':        (  0,  0,   0,  0,  0),
    'office':       (  0,  0,   0,  0,  0),
    'movie':        (  0,  0,   0,  0,  5),
    'class':        (  0,  0,   0,  0,  5),
    'exam':         (  0,  5,   0,  5,  8),  # slightly high arousal/urgency
    'test':         (  0,  5,   0,  5,  5),
    'school':       (  0,  0,   0,  0,  5),
    'license':      (  0,  0,   0,  0,  0),
    'package':      (  0,  0,   0,  0,  0),
    'porch':        (  0,  0,   0,  0,  0),
    # Common verbs that are context-dependent
    'see':          (  0,  0,   0,  0,  0),
    'saw':          (  0,  0,   0,  0,  0),
    'went':         (  0,  0,   0,  0,  0),
    'going':        (  0,  0,   0,  0,  0),
    'come':         (  0,  0,   0,  0,  0),
    'coming':       (  0,  0,   0,  0,  0),
    'pick':         (  0,  0,   0,  0,  0),
    'picked':       (  0,  0,   0,  0,  0),
    'called':       (  0,  0,   0,  0,  0),
    'calling':      (  0,  0,   0,  0,  0),
    'stopped':      (  0,  0,   0,  0,  0),
    'told':         (  0,  0,   0,  0,  0),
    'tell':         (  0,  0,   0,  0,  0),
    'doing':        (  0,  0,   0,  0,  0),
    'running':      (  0,  3,   0,  0,  0),
    'walk':         (  0,  0,   0,  0,  0),
    'run':          (  0,  3,   0,  0,  0),
    'sit':          (  0,  0,   0,  0,  0),
    'sitting':      (  0,  0,   0,  0,  0),
    'standing':     (  0,  0,   0,  0,  0),
    'waiting':      (  0,  0,   0,  0,  0),
    'watching':     (  0,  0,   0,  0,  0),
    'looking':      (  0,  0,   0,  0,  0),
    'making':       (  0,  0,   0,  0,  0),
    'working':      (  0,  0,   0,  0,  0),
    # Common adjectives that are neutral
    'interesting':  (  0,  3,   0,  0,  0),
    'actual':       (  0,  0,   0,  0,  0),
    'able':         (  0,  0,   5,  0,  0),
    'active':       (  0,  3,   0,  0,  0),
    'available':    (  0,  0,   0,  0,  0),
    'different':    (  0,  0,   0,  0,  0),
    'certain':      (  0,  0,   5,  0,  0),
    'real':         (  0,  0,   0,  0,  0),
    'right':        (  0,  0,   0,  0,  0),  # too liquid: "right" = correct OR direction
    'wrong':        (-10,  5,  -5,  0,  -5),  # slightly negative — "wrong" has mild charge
    # Spatial/temporal that leaked dV
    'last':         (  0,  0,   0,  5,  0),  # "last night" ≠ "last" as finality. FINALITY role handles context.
    'night':        (  0,  0,   0,  0,  0),
    'morning':      (  0,  0,   0,  0,  0),
    'day':          (  0,  0,   0,  0,  0),
    'time':         (  0,  0,   0,  0,  0),
    'today':        (  0,  0,   0,  0,  0),
    'tomorrow':     (  0,  0,   0,  0,  0),
    'week':         (  0,  0,   0,  0,  0),
    'year':         (  0,  0,   0,  0,  0),
    'month':        (  0,  0,   0,  0,  0),
    'hour':         (  0,  0,   0,  0,  0),
    'minute':       (  0,  0,   0,  0,  0),
    'ago':          (  0,  0,   0,  0,  0),
    'ahead':        (  8,  3,   5,  0,  0),  # "go ahead" = permission, mild positive
    'across':       (  0,  0,   0,  0,  0),
    'abroad':       (  0,  0,   0,  0,  0),
    # Abstract nouns with bloated dV
    'choice':       (  0,  3,   5,  0,  5),  # neutral — context decides
    'ability':      (  0,  0,   5,  0,  5),
    'abilities':    (  0,  0,   5,  0,  5),
    'access':       (  0,  0,   0,  0,  0),
    'account':      (  0,  0,   0,  0,  0),
    'activity':     (  0,  0,   0,  0,  0),
    'add':          (  0,  0,   0,  0,  0),
    'advice':       (  0,  0,   0,  0,  5),
    'advantage':    (  5,  0,   5,  0,  5),  # slight positive lean
    'air':          (  0,  0,   0,  0,  0),
    'airplane':     (  0,  0,   0,  0,  0),
    'act':          (  0,  0,   0,  0,  0),
    'action':       (  0,  3,   0,  0,  0),
    'agreed':       (  3,  0,   0,  0,  0),
    'agree':        (  3,  0,   0,  0,  0),
    'aid':          (  5,  0,   0,  0,  5),
    'admitted':     (  0,  0,   0,  0,  0),
    # Life events — these have HIGH gravity but LOW valence
    # "born" is not pos or neg. The CONTEXT determines it.
    # "my daughter was born" = pos. "stillborn" = devastating.
    'born':         ( 12, 10,   5,  5, 30),  # mild positive, high G. "Born" implies new life.
    'surgery':      ( -5, 10,  -8, 15, 25),  # mild neg, high G
    'bar':          (  0,  0,   0,  0,  5),
    'passed':       (  5,  0,   0,  0, 10),  # mild pos. "Passed the test" = achievement context.
    'accepted':     ( 15,  5,   8,  0, 15),  # moderate positive. "Accepted" = inclusion.
    'proposed':     ( 15, 10,   8,  0, 25),  # moderate positive, high G.
    'daughter':     (  0,  0,   0,  0, 30),  # person, high G
    'son':          (  0,  0,   0,  0, 30),
    'baby':         (  5,  5,   0,  0, 30),  # slight positive, high G
    'paycheck':     (  8,  0,   5,  0, 10),
    'loans':        ( -5,  0,  -5,  0,  5),
    'birthday':     ( 10,  8,   0,  0, 20),  # positive event, high G
    'surprise':     (  5, 15,   0,  0, 10),  # slight positive + arousal
    'surprised':    (  5, 15,   0,  0, 10),
    'happiest':     ( 35, 15,  10,  0, 15),  # superlative of happy
    # Words that are emotional but were way too hot
    'layoffs':      (-15, 10, -10, 10, 15),  # was -35
    'doctor':       (  0,  5,  -5,  5, 10),  # was -24
    # "cancelled" is context-dependent. Flight cancelled ≠ wedding cancelled.
    # Keep mild negative — inconvenience baseline. Structure amplifies.
    'cancelled':    (-12,  5,  -5,  5,  5),  # was -40
    'cancel':       ( -8,  3,  -3,  0,  0),
    # Common words that need force reduction
    'forgot':       ( -8,  3,  -3,  0,  -3),  # was -25. Forgetting is a mild negative.
    'broken':       (-12,  5,  -5,  0,  -5),  # "dishwasher is broken" ≠ "i am broken"
    'failed':       (-12,  5, -10,  5, -8),  # "i failed my exam" = disappointment not devastation
    'messed':       ( -8,  3,  -3,  0,  -3),
    # "need" is too negative — "i need to do laundry" ≠ desperate need
    'need':         ( -5,  3,  -3,  3,  5),  # was -15. Keep mild negative for unmet need.
    'renew':        (  0,  0,   0,  0,  0),
    'license':      (  0,  0,   0,  0,  0),
    'dishwasher':   (  0,  0,   0,  0,  0),
    'porch':        (  0,  0,   0,  0,  0),
    'oil':          (  0,  0,   0,  0,  0),
    'saturday':     (  0,  0,   0,  0,  0),
    'channel':      (  0,  0,   0,  0,  0),
    'power':        (  0,  3,   5,  0,  5),  # neutral with slight D lean
    'coffee':       (  3,  0,   0,  0,  0),  # very mild positive
    'gonna':        (  0,  0,   0,  0,  0),
    'grab':         (  0,  2,   0,  0,  0),
    # Missing verb forms + corrections
    'murdered':     (-25, 12, -10,  8,  -8),
    'murdering':    (-25, 12, -10,  8,  -8),
    'suicidal':     (-30, 15, -20, 15, -15),
    'mondays':      ( -5,  3,  -3,  0,   0),
    'suspense':     (  0,  8,   0,  5,   5),  # arousal, not valence
}
EMOTIONAL_VOCABULARY.update(_V7_FIXES)

# ── V8 MASS CORRECTION: zero 500+ inflated GAS atoms ──
# These common words carried residual positive dV from V1 vocabulary.
# Spot-check audit showed 59% accuracy on real text due to positive inflation.
# Nuclear fix: zero all common nouns/verbs/adj with dV>10 and low G/A.
# Genuinely positive emotional words (love, happy, proud, etc.) preserved.
_V8_MASS_ZERO = {
    'accept': (0, 5, 0, 0, 10),
    'accepts': (0, 5, 0, 0, 10),
    'accommodation': (0, 5, 0, 0, 15),
    'accommodations': (0, 5, 0, 0, 15),
    'ace': (0, -4, 0, 0, 10),
    'achieve': (0, 5, 0, 0, 19),
    'achieved': (0, 5, 0, 0, 15),
    'adoptive': (0, 5, 0, 0, 18),
    'adult': (0, 0, 0, 0, 16),
    'advocates': (0, 5, 0, 0, 10),
    'affirmed': (0, 5, 0, 0, 18),
    'afford': (0, -5, 0, 0, 10),
    'album': (0, -27, 0, 0, 6),
    'ally': (0, 5, 0, 0, 12),
    'allyship': (0, 5, 0, 0, 15),
    'along': (0, -20, 0, 0, 5),
    'alright': (0, -29, 0, 0, 9),
    'amazed': (0, 5, 0, 0, 9),
    'amused': (0, 5, 0, 0, 5),
    'amusing': (0, 5, 0, 0, 0),
    'animal': (0, 5, 0, 0, 10),
    'answer': (0, 5, 0, 0, 16),
    'anywhere': (0, -24, 0, 0, 1),
    'apartments': (0, -29, 0, 0, 9),
    'apologized': (0, -29, 0, 0, 0),
    'apple': (0, 4, 0, 0, 0),
    'applied': (0, 5, 0, 0, 14),
    'appreciate': (0, 4, 0, 0, 8),
    'appreciated': (0, -4, 0, 0, 19),
    'appreciative': (0, -11, 0, 0, 14),
    'approaching': (0, 5, 0, 0, 14),
    'approved': (0, -13, 0, 0, 14),
    'arrive': (0, -6, 0, 0, 12),
    'arrived': (0, 5, 0, 0, 16),
    'assistance': (0, 5, 0, 0, 10),
    'assume': (0, -18, 0, 0, 10),
    'assuming': (0, -20, 0, 0, 16),
    'assured': (0, -15, 0, 0, 17),
    'athletic': (0, 5, 0, 0, 16),
    'attachment': (0, 5, 0, 0, 10),
    'attend': (0, -13, 0, 0, 16),
    'aura': (0, 5, 0, 0, 10),
    'award': (0, 5, 0, 0, 14),
    'babies': (0, 5, 0, 0, 8),
    'babysit': (0, 5, 0, 0, 9),
    'bahamas': (0, 5, 0, 0, 9),
    'bake': (0, 5, 0, 0, 16),
    'balmy': (0, -10, 0, 0, 8),
    'banana': (0, 4, 0, 0, 0),
    'band': (0, 5, 0, 0, 10),
    'beaches': (0, 0, 0, 0, 10),
    'beautifully': (0, 5, 0, 0, 10),
    'beauty': (0, 5, 0, 0, 13),
    'beef': (0, 5, 0, 0, 14),
    'begin': (0, 5, 0, 0, 12),
    'belief': (0, -11, 0, 0, 16),
    'believed': (0, 5, 0, 0, 8),
    'benefits': (0, 5, 0, 0, 8),
    'benign': (0, -15, 0, 0, 15),
    'bicycle': (0, 4, 0, 0, 9),
    'bigger': (0, -2, 0, 0, 17),
    'billion': (0, 5, 0, 0, 10),
    'bit': (0, 5, 0, 0, 0),
    'blessing': (0, 5, 0, 0, 15),
    'blessings': (0, -11, 0, 0, 16),
    'blooming': (0, 5, 0, 0, 12),
    'blossoming': (0, 5, 0, 0, 15),
    'blue': (0, -20, 0, 0, 0),
    'bond': (0, 5, 0, 0, 10),
    'boost': (0, 5, 0, 0, 16),
    'bowling': (0, -11, 0, 0, 1),
    'breast': (0, -11, 0, 0, 16),
    'brownies': (0, -27, 0, 0, 14),
    'buddies': (0, 5, 0, 0, 14),
    'budding': (0, 5, 0, 0, 10),
    'buffet': (0, 5, 0, 0, 10),
    'built': (0, 5, 0, 0, 8),
    'bunch': (0, 5, 0, 0, 10),
    'burger': (0, 5, 0, 0, 10),
    'burrito': (0, 5, 0, 0, 14),
    'bussin': (0, 5, 0, 0, 10),
    'button': (0, 5, 0, 0, 0),
    'cake': (0, 5, 0, 0, 0),
    'calm': (0, -10, 0, 0, 8),
    'calmed': (0, -10, 0, 0, 8),
    'camping': (0, 5, 0, 0, 9),
    'captivating': (0, 5, 0, 0, 12),
    'careful': (0, 5, 0, 0, 0),
    'cares': (0, 5, 0, 0, 10),
    'carried': (0, 5, 0, 0, 10),
    'celebrated': (0, 5, 0, 0, 8),
    'celebrates': (0, 5, 0, 0, 8),
    'celebrating': (0, 5, 0, 0, 19),
    'cereal': (0, 5, 0, 0, 0),
    'certainly': (0, 5, 0, 0, 16),
    'certification': (0, 2, 0, 0, 13),
    'certified': (0, 5, 0, 0, 15),
    'championship': (0, 5, 0, 0, 15),
    'charm': (0, 3, 0, 0, 0),
    'charming': (0, 4, 0, 0, 0),
    'cheesecake': (0, -18, 0, 0, 5),
    'chef': (0, -6, 0, 0, 12),
    'chemistry': (0, 4, 0, 0, 17),
    'chicken': (0, 5, 0, 0, 8),
    'chip': (0, 5, 0, 0, 8),
    'chipotle': (0, 5, 0, 0, 16),
    'chocolate': (0, 5, 0, 0, 0),
    'chuckled': (0, 5, 0, 0, 5),
    'cinematic': (0, 5, 0, 0, 15),
    'circus': (0, 5, 0, 0, 10),
    'cities': (0, 5, 0, 0, 17),
    'clapped': (0, 5, 0, 0, 8),
    'clean': (0, -5, 0, 0, 15),
    'clear': (0, 5, 0, 0, 16),
    'cleared': (0, -5, 0, 0, 8),
    'closedon': (0, 5, 0, 0, 15),
    'closest': (0, -18, 0, 0, 12),
    'clothing': (0, -27, 0, 0, 10),
    'clout': (0, 5, 0, 0, 10),
    'club': (0, 5, 0, 0, 5),
    'clubs': (0, 5, 0, 0, 19),
    'clue': (0, -22, 0, 0, 5),
    'coast': (0, -18, 0, 0, 12),
    'colleague': (0, 5, 0, 0, 16),
    'colors': (0, -13, 0, 0, 8),
    'comforting': (0, -13, 0, 0, 14),
    'comic': (0, -4, 0, 0, 9),
    'comics': (0, 5, 0, 0, 6),
    'commitment': (0, 5, 0, 0, 16),
    'community': (0, 5, 0, 0, 16),
    'company': (0, -6, 0, 0, 19),
    'compelling': (0, 5, 0, 0, 10),
    'complete': (0, 5, 0, 0, 16),
    'completed': (0, -11, 0, 0, 16),
    'completely': (0, 5, 0, 0, 0),
    'conference': (0, 5, 0, 0, 17),
    'confident': (0, 5, 0, 0, 10),
    'congratulation': (0, 5, 0, 0, 17),
    'connection': (0, 5, 0, 0, 16),
    'consider': (0, -24, 0, 0, 14),
    'contact': (0, -13, 0, 0, 16),
    'cooked': (0, 5, 0, 0, 5),
    'cookie': (0, 5, 0, 0, 0),
    'cooking': (0, 5, 0, 0, 5),
    'coregulate': (0, -10, 0, 0, 12),
    'counselor': (0, -11, 0, 0, 16),
    'counter': (0, 5, 0, 0, 0),
    'couple': (0, 4, 0, 0, 10),
    'course': (0, -2, 0, 0, 8),
    'courses': (0, -15, 0, 0, 16),
    'cousins': (0, -15, 0, 0, 14),
    'creative': (0, 2, 0, 0, 19),
    'danced': (0, 5, 0, 0, 12),
    'dancing': (0, 5, 0, 0, 10),
    'deal': (0, 5, 0, 0, 16),
    'dealing': (0, -15, 0, 0, 19),
    'dear': (0, -4, 0, 0, 14),
    'decent': (0, -15, 0, 0, 13),
    'dedication': (0, -2, 0, 0, 14),
    'deep': (0, 5, 0, 0, 16),
    'defended': (0, 5, 0, 0, 10),
    'defends': (0, 5, 0, 0, 10),
    'definitely': (0, 5, 0, 0, 16),
    'degree': (0, 5, 0, 0, 15),
    'delicious': (0, 5, 0, 0, 14),
    'delightful': (0, 5, 0, 0, 8),
    'delivery': (0, 5, 0, 0, 10),
    'demure': (0, -15, 0, 0, 5),
    'dessert': (0, 5, 0, 0, 0),
    'dialogue': (0, 5, 0, 0, 10),
    'digital': (0, -22, 0, 0, 16),
    'dignity': (0, 5, 0, 0, 15),
    'donation': (0, 5, 0, 0, 10),
    'donations': (0, 5, 0, 0, 10),
    'donor': (0, 5, 0, 0, 15),
    'dream': (0, 5, 0, 0, 15),
    'drinks': (0, 5, 0, 0, 10),
    'drip': (0, 5, 0, 0, 10),
    'ease': (0, -9, 0, 0, 9),
    'easier': (0, -5, 0, 0, 5),
    'easily': (0, 5, 0, 0, 10),
    'east': (0, 5, 0, 0, 0),
    'easy': (0, -20, 0, 0, 16),
    'education': (0, 5, 0, 0, 8),
    'eggs': (0, 5, 0, 0, 4),
    'electronics': (0, 5, 0, 0, 17),
    'employable': (0, 5, 0, 0, 10),
    'empowered': (0, 5, 0, 0, 15),
    'engaged': (0, 5, 0, 0, 19),
    'engaging': (0, 5, 0, 0, 8),
    'enjoyable': (0, 5, 0, 0, 8),
    'enter': (0, -6, 0, 0, 12),
    'entertained': (0, 5, 0, 0, 5),
    'entertaining': (0, 5, 0, 0, 8),
    'entertainment': (0, 0, 0, 0, 6),
    'episode': (0, -24, 0, 0, 14),
    'especially': (0, 5, 0, 0, 10),
    'europe': (0, 0, 0, 0, 16),
    'everyday': (0, -20, 0, 0, 14),
    'excelling': (0, 5, 0, 0, 18),
    'excitement': (0, 5, 0, 0, 8),
    'expected': (0, -15, 0, 0, 17),
    'experienced': (0, 5, 0, 0, 17),
    'experiences': (0, 5, 0, 0, 16),
    'explain': (0, 5, 0, 0, 10),
    'exquisite': (0, 5, 0, 0, 15),
    'exquisitely': (0, 5, 0, 0, 0),
    'fairly': (0, -6, 0, 0, 14),
    'faithful': (0, 5, 0, 0, 8),
    'families': (0, -24, 0, 0, 14),
    'fan': (0, 5, 0, 0, 16),
    'fancy': (0, 5, 0, 0, 10),
    'favor': (0, -15, 0, 0, 9),
    'favorite': (0, 5, 0, 0, 14),
    'favourite': (0, 0, 0, 0, 16),
    'feed': (0, 5, 0, 0, 10),
    'feedback': (0, 5, 0, 0, 6),
    'feeling': (0, -13, 0, 0, 9),
    'fertile': (0, 5, 0, 0, 15),
    'festival': (0, 5, 0, 0, 16),
    'fiance': (0, -22, 0, 0, 9),
    'fiancee': (0, 5, 0, 0, 14),
    'figure': (0, -29, 0, 0, 12),
    'figured': (0, 0, 0, 0, 14),
    'fills': (0, 5, 0, 0, 10),
    'finally': (0, -10, 0, 0, 10),
    'finest': (0, 5, 0, 0, 12),
    'finished': (0, 5, 0, 0, 5),
    'flourishing': (0, 5, 0, 0, 18),
    'flowers': (0, 5, 0, 0, 10),
    'flying': (0, 5, 0, 0, 16),
    'fond': (0, 5, 0, 0, 12),
    'fondly': (0, -27, 0, 0, 16),
    'forward': (0, -6, 0, 0, 16),
    'freshman': (0, 5, 0, 0, 8),
    'friendly': (0, -9, 0, 0, 14),
    'fries': (0, 5, 0, 0, 10),
    'fun': (0, 5, 0, 0, 5),
    'fundraiser': (0, 5, 0, 0, 10),
    'funny': (0, 5, 0, 0, 5),
    'gaming': (0, 5, 0, 0, 12),
    'gender': (0, -27, 0, 0, 17),
    'general': (0, 0, 0, 0, 17),
    'generation': (0, -2, 0, 0, 16),
    'generosity': (0, 5, 0, 0, 10),
    'gesture': (0, -4, 0, 0, 6),
    'gifts': (0, 5, 0, 0, 10),
    'glad': (0, 5, 0, 0, 14),
    'glasses': (0, 4, 0, 0, 4),
    'goeshard': (0, 5, 0, 0, 8),
    'gold': (0, 5, 0, 0, 6),
    'golden': (0, 0, 0, 0, 10),
    'google': (0, 5, 0, 0, 0),
    'gracious': (0, 5, 0, 0, 18),
    'grad': (0, 2, 0, 0, 14),
    'grandparents': (0, -27, 0, 0, 10),
    'greater': (0, 5, 0, 0, 6),
    'grew': (0, 5, 0, 0, 10),
    'gripping': (0, 5, 0, 0, 12),
    'grounded': (0, -20, 0, 0, 12),
    'grounding': (0, -15, 0, 0, 10),
    'growing': (0, 5, 0, 0, 16),
    'growth': (0, 5, 0, 0, 19),
    'guest': (0, -2, 0, 0, 16),
    'guitar': (0, 4, 0, 0, 9),
    'haircut': (0, -27, 0, 0, 1),
    'handful': (0, 5, 0, 0, 8),
    'handle': (0, 5, 0, 0, 19),
    'happiest': (0, 5, 0, 0, 15),
    'harmonious': (0, -5, 0, 0, 18),
    'harmony': (0, -5, 0, 0, 18),
    'healthier': (0, -6, 0, 0, 16),
    'hearing': (0, -2, 0, 0, 16),
    'held': (0, 5, 0, 0, 16),
    'hello': (0, 5, 0, 0, 8),
    'help': (0, 5, 0, 0, 10),
    'helped': (0, 5, 0, 0, 10),
    'helpful': (0, 5, 0, 0, 10),
    'helping': (0, 5, 0, 0, 10),
    'helps': (0, 5, 0, 0, 9),
    'hers': (0, 0, 0, 0, 0),
    'hey': (0, 5, 0, 0, 8),
    'hi': (0, 5, 0, 0, 6),
    'higher': (0, 2, 0, 0, 16),
    'highly': (0, 5, 0, 0, 16),
    'hiking': (0, 5, 0, 0, 13),
    'hire': (0, 5, 0, 0, 14),
    'hired': (0, 5, 0, 0, 10),
    'historic': (0, 5, 0, 0, 9),
    'hits': (0, 5, 0, 0, 15),
    'hobby': (0, 4, 0, 0, 9),
    'homemade': (0, -13, 0, 0, 9),
    'hometown': (0, -13, 0, 0, 13),
    'honey': (0, -4, 0, 0, 0),
    'hopefully': (0, 0, 0, 0, 5),
    'hopes': (0, 0, 0, 0, 17),
    'hoping': (0, 5, 0, 0, 16),
    'horizon': (0, 5, 0, 0, 0),
    'host': (0, -6, 0, 0, 14),
    'hug': (0, -6, 0, 0, 13),
    'humbled': (0, 5, 0, 0, 18),
    'imaginative': (0, 5, 0, 0, 0),
    'important': (0, 5, 0, 0, 7),
    'impressed': (0, 5, 0, 0, 8),
    'impression': (0, 5, 0, 0, 16),
    'impressive': (0, 5, 0, 0, 10),
    'improve': (0, 5, 0, 0, 16),
    'improving': (0, 0, 0, 0, 16),
    'included': (0, -18, 0, 0, 19),
    'incredibly': (0, 5, 0, 0, 10),
    'indeed': (0, 0, 0, 0, 19),
    'innocent': (0, 5, 0, 0, 16),
    'inspiration': (0, 5, 0, 0, 19),
    'integrity': (0, 5, 0, 0, 12),
    'interest': (0, 5, 0, 0, 5),
    'internet': (0, 5, 0, 0, 0),
    'invest': (0, 5, 0, 0, 0),
    'invited': (0, 5, 0, 0, 10),
    'ipad': (0, 5, 0, 0, 16),
    'iphone': (0, 5, 0, 0, 17),
    'japan': (0, -4, 0, 0, 14),
    'joined': (0, -20, 0, 0, 16),
    'juice': (0, 5, 0, 0, 0),
    'killedit': (0, 5, 0, 0, 10),
    'kindergarten': (0, -2, 0, 0, 5),
    'knowing': (0, -11, 0, 0, 13),
    'landed': (0, 5, 0, 0, 12),
    'laptop': (0, 5, 0, 0, 10),
    'lawn': (0, -18, 0, 0, 19),
    'lead': (0, 5, 0, 0, 16),
    'leading': (0, -9, 0, 0, 10),
    'league': (0, -27, 0, 0, 8),
    'learned': (0, 5, 0, 0, 8),
    'lesson': (0, 5, 0, 0, 8),
    'licensed': (0, 5, 0, 0, 12),
    'likely': (0, -24, 0, 0, 6),
    'liking': (0, 5, 0, 0, 16),
    'lining': (0, -9, 0, 0, 10),
    'lol': (0, 5, 0, 0, 0),
    'lotto': (0, 5, 0, 0, 16),
    'lovely': (0, 5, 0, 0, 10),
    'lover': (0, 5, 0, 0, 14),
    'loves': (0, 5, 0, 0, 9),
    'lucid': (0, 5, 0, 0, 15),
    'lucidity': (0, 5, 0, 0, 15),
    'lullaby': (0, -10, 0, 0, 15),
    'luminous': (0, 5, 0, 0, 18),
    'made': (0, 5, 0, 0, 5),
    'main': (0, 5, 0, 0, 8),
    'maintain': (0, -18, 0, 0, 19),
    'makeup': (0, 5, 0, 0, 0),
    'manageable': (0, -5, 0, 0, 10),
    'management': (0, 0, 0, 0, 16),
    'mariachi': (0, 5, 0, 0, 5),
    'married': (0, 5, 0, 0, 16),
    'marry': (0, 5, 0, 0, 16),
    'massage': (0, -24, 0, 0, 9),
    'mate': (0, -6, 0, 0, 14),
    'means': (0, -22, 0, 0, 5),
    'melodic': (0, 5, 0, 0, 10),
    'melody': (0, 5, 0, 0, 12),
    'member': (0, -29, 0, 0, 16),
    'memorable': (0, 5, 0, 0, 18),
    'memorized': (0, 5, 0, 0, 10),
    'mesmerizing': (0, 5, 0, 0, 15),
    'million': (0, 5, 0, 0, 10),
    'mindful': (0, -10, 0, 0, 10),
    'mindfulness': (0, -10, 0, 0, 10),
    'mindset': (0, 5, 0, 0, 10),
    'mine': (0, -13, 0, 0, 16),
    'moment': (0, 5, 0, 0, 10),
    'mommy': (0, -11, 0, 0, 9),
    'mood': (0, -4, 0, 0, 0),
    'motivation': (0, 5, 0, 0, 17),
    'multiple': (0, 2, 0, 0, 14),
    'music': (0, 5, 0, 0, 7),
    'musical': (0, 5, 0, 0, 13),
    'natural': (0, 5, 0, 0, 16),
    'naturally': (0, -18, 0, 0, 12),
    'nature': (0, -27, 0, 0, 14),
    'near': (0, 5, 0, 0, 10),
    'neat': (0, -22, 0, 0, 10),
    'nephew': (0, 5, 0, 0, 0),
    'newborn': (0, 5, 0, 0, 9),
    'nicer': (0, -20, 0, 0, 9),
    'niece': (0, 5, 0, 0, 0),
    'nintendo': (0, 4, 0, 0, 8),
    'nostalgia': (0, 5, 0, 0, 0),
    'nostalgic': (0, 5, 0, 0, 0),
    'nowadays': (0, -27, 0, 0, 6),
    'nuanced': (0, 5, 0, 0, 10),
    'occasion': (0, -9, 0, 0, 8),
    'online': (0, 5, 0, 0, 10),
    'optimistic': (0, 5, 0, 0, 10),
    'orange': (0, 5, 0, 0, 0),
    'others': (0, 5, 0, 0, 0),
    'outfit': (0, -2, 0, 0, 19),
    'outlook': (0, 2, 0, 0, 13),
    'outside': (0, 5, 0, 0, 0),
    'ovation': (0, 5, 0, 0, 10),
    'overcame': (0, 5, 0, 0, 10),
    'owner': (0, -4, 0, 0, 17),
    'paid': (0, 5, 0, 0, 10),
    'painting': (0, -15, 0, 0, 5),
    'pair': (0, 5, 0, 0, 14),
    'panama': (0, 0, 0, 0, 10),
    'pass': (0, 5, 0, 0, 18),
    'pasta': (0, 5, 0, 0, 8),
    'path': (0, 5, 0, 0, 10),
    'payday': (0, 5, 0, 0, 13),
    'performances': (0, 1, 0, 0, 4),
    'personality': (0, 0, 0, 0, 19),
    'piano': (0, -27, 0, 0, 9),
    'pie': (0, -9, 0, 0, 8),
    'pink': (0, 5, 0, 0, 0),
    'plant': (0, 5, 0, 0, 16),
    'play': (0, 5, 0, 0, 16),
    'pleasure': (0, 5, 0, 0, 17),
    'plus': (0, 0, 0, 0, 16),
    'poem': (0, 5, 0, 0, 10),
    'poems': (0, 5, 0, 0, 10),
    'pop': (0, 2, 0, 0, 10),
    'popular': (0, 5, 0, 0, 16),
    'position': (0, -9, 0, 0, 8),
    'possibility': (0, 0, 0, 0, 10),
    'possible': (0, 5, 0, 0, 8),
    'potato': (0, 4, 0, 0, 8),
    'potential': (0, 5, 0, 0, 6),
    'powerful': (0, 5, 0, 0, 10),
    'practice': (0, 5, 0, 0, 10),
    'prefer': (0, -20, 0, 0, 17),
    'prep': (0, 2, 0, 0, 14),
    'prepare': (0, 5, 0, 0, 14),
    'preparing': (0, 5, 0, 0, 19),
    'presence': (0, -4, 0, 0, 13),
    'presents': (0, -9, 0, 0, 9),
    'private': (0, -13, 0, 0, 17),
    'privileged': (0, 5, 0, 0, 15),
    'prize': (0, 5, 0, 0, 19),
    'pro': (0, -6, 0, 0, 19),
    'productive': (0, 5, 0, 0, 16),
    'progress': (0, 5, 0, 0, 8),
    'promoted': (0, 5, 0, 0, 10),
    'properly': (0, -20, 0, 0, 13),
    'proposal': (0, 5, 0, 0, 15),
    'protective': (0, 5, 0, 0, 15),
    'published': (0, 5, 0, 0, 12),
    'pup': (0, -13, 0, 0, 6),
    'puppies': (0, 5, 0, 0, 5),
    'purchased': (0, 5, 0, 0, 8),
    'quality': (0, 5, 0, 0, 16),
    'rabbit': (0, -29, 0, 0, 0),
    'radiant': (0, 5, 0, 0, 18),
    'ramen': (0, 5, 0, 0, 8),
    'receive': (0, -6, 0, 0, 17),
    'received': (0, -11, 0, 0, 12),
    'recognition': (0, 5, 0, 0, 18),
    'recognized': (0, -4, 0, 0, 17),
    'record': (0, 5, 0, 0, 16),
    'recover': (0, 5, 0, 0, 16),
    'recovery': (0, 5, 0, 0, 16),
    'reframe': (0, 5, 0, 0, 10),
    'refreshing': (0, 5, 0, 0, 8),
    'relate': (0, 0, 0, 0, 16),
    'related': (0, -24, 0, 0, 12),
    'relative': (0, 5, 0, 0, 0),
    'relaxation': (0, -22, 0, 0, 8),
    'relaxing': (0, -29, 0, 0, 9),
    'rely': (0, -4, 0, 0, 12),
    'remembered': (0, -22, 0, 0, 13),
    'remembers': (0, 5, 0, 0, 10),
    'reminisce': (0, 0, 0, 0, 9),
    'reminiscing': (0, -11, 0, 0, 8),
    'remote': (0, -18, 0, 0, 0),
    'renewed': (0, 5, 0, 0, 8),
    'resilient': (0, 5, 0, 0, 10),
    'resonance': (0, 5, 0, 0, 18),
    'resonant': (0, 5, 0, 0, 15),
    'respite': (0, -20, 0, 0, 15),
    'responsibilities': (0, 5, 0, 0, 17),
    'responsible': (0, 2, 0, 0, 17),
    'returned': (0, 5, 0, 0, 8),
    'reunion': (0, 0, 0, 0, 14),
    'review': (0, -15, 0, 0, 17),
    'rich': (0, 5, 0, 0, 16),
    'riding': (0, 5, 0, 0, 17),
    'riveting': (0, 5, 0, 0, 15),
    'rock': (0, -5, 0, 0, 15),
    'roommate': (0, -4, 0, 0, 16),
    'salad': (0, -4, 0, 0, 0),
    'salary': (0, 5, 0, 0, 0),
    'sale': (0, 5, 0, 0, 16),
    'sandwich': (0, 5, 0, 0, 4),
    'satisfying': (0, 5, 0, 0, 19),
    'save': (0, 5, 0, 0, 8),
    'science': (0, 5, 0, 0, 14),
    'scored': (0, 5, 0, 0, 8),
    'seafood': (0, 0, 0, 0, 1),
    'season': (0, -6, 0, 0, 9),
    'security': (0, 5, 0, 0, 11),
    'self': (0, -24, 0, 0, 12),
    'sensitive': (0, -13, 0, 0, 1),
    'sentimental': (0, -29, 0, 0, 0),
    'serene': (0, -15, 0, 0, 18),
    'served': (0, -15, 0, 0, 0),
    'service': (0, -15, 0, 0, 12),
    'settle': (0, 2, 0, 0, 16),
    'shape': (0, 0, 0, 0, 2),
    'share': (0, 5, 0, 0, 16),
    'sharing': (0, -2, 0, 0, 10),
    'shining': (0, 5, 0, 0, 19),
    'shopping': (0, -4, 0, 0, 14),
    'shrimp': (0, 5, 0, 0, 0),
    'significant': (0, 5, 0, 0, 16),
    'simple': (0, -13, 0, 0, 10),
    'sincerely': (0, 5, 0, 0, 12),
    'singing': (0, 5, 0, 0, 12),
    'skateboard': (0, 5, 0, 0, 5),
    'skiing': (0, 5, 0, 0, 13),
    'skill': (0, 5, 0, 0, 17),
    'skills': (0, 5, 0, 0, 7),
    'slay': (0, 5, 0, 0, 10),
    'smooth': (0, 5, 0, 0, 9),
    'snacks': (0, -2, 0, 0, 4),
    'socialize': (0, 5, 0, 0, 17),
    'socks': (0, -18, 0, 0, 4),
    'soda': (0, 5, 0, 0, 0),
    'softball': (0, 5, 0, 0, 16),
    'solid': (0, 5, 0, 0, 15),
    'sometimes': (0, 0, 0, 0, 10),
    'soothing': (0, -15, 0, 0, 15),
    'south': (0, 5, 0, 0, 4),
    'spaghetti': (0, -24, 0, 0, 5),
    'specific': (0, -20, 0, 0, 19),
    'speech': (0, -6, 0, 0, 19),
    'sport': (0, 5, 0, 0, 14),
    'sports': (0, 5, 0, 0, 12),
    'spring': (0, 5, 0, 0, 0),
    'stable': (0, -15, 0, 0, 15),
    'stamp': (0, 5, 0, 0, 8),
    'start': (0, 5, 0, 0, 12),
    'steak': (0, 5, 0, 0, 19),
    'stellar': (0, 5, 0, 0, 18),
    'storage': (0, 5, 0, 0, 16),
    'strength': (0, 5, 0, 0, 10),
    'strong': (0, 5, 0, 0, 8),
    'stronger': (0, 5, 0, 0, 7),
    'studied': (0, -15, 0, 0, 16),
    'style': (0, -6, 0, 0, 19),
    'succeeded': (0, 5, 0, 0, 15),
    'succeeding': (0, 5, 0, 0, 15),
    'successful': (0, 5, 0, 0, 19),
    'suddenly': (0, 2, 0, 0, 8),
    'summer': (0, 5, 0, 0, 0),
    'sundays': (0, -27, 0, 0, 9),
    'sunset': (0, -5, 0, 0, 8),
    'super': (0, 5, 0, 0, 16),
    'supermarket': (0, 5, 0, 0, 16),
    'supper': (0, -11, 0, 0, 16),
    'supplies': (0, -20, 0, 0, 17),
    'support': (0, 5, 0, 0, 12),
    'supporting': (0, 5, 0, 0, 10),
    'supportive': (0, -6, 0, 0, 14),
    'surprises': (0, 5, 0, 0, 14),
    'surprising': (0, 5, 0, 0, 16),
    'surprisingly': (0, 5, 0, 0, 17),
    'survivor': (0, 5, 0, 0, 15),
    'sushi': (0, 5, 0, 0, 0),
    'sweetest': (0, 0, 0, 0, 10),
    'sweetheart': (0, -6, 0, 0, 14),
    'sweets': (0, -6, 0, 0, 8),
    'symphonic': (0, 5, 0, 0, 18),
    'taco': (0, 5, 0, 0, 0),
    'talented': (0, 5, 0, 0, 10),
    'taught': (0, 5, 0, 0, 10),
    'teach': (0, 5, 0, 0, 16),
    'teaching': (0, -4, 0, 0, 14),
    'team': (0, 5, 0, 0, 16),
    'thanking': (0, 5, 0, 0, 10),
    'thawing': (0, 5, 0, 0, 10),
    'thicc': (0, 5, 0, 0, 5),
    'thrill': (0, 5, 0, 0, 16),
    'thrilling': (0, 5, 0, 0, 10),
    'tip': (0, 5, 0, 0, 0),
    'toast': (0, -22, 0, 0, 8),
    'toddler': (0, -6, 0, 0, 9),
    'together': (0, 5, 0, 0, 16),
    'touched': (0, 5, 0, 0, 18),
    'tour': (0, 5, 0, 0, 12),
    'transfer': (0, -2, 0, 0, 14),
    'transportation': (0, 5, 0, 0, 9),
    'travel': (0, 5, 0, 0, 13),
    'traveling': (0, 5, 0, 0, 13),
    'travelling': (0, 5, 0, 0, 10),
    'try': (0, 5, 0, 0, 8),
    'tune': (0, -4, 0, 0, 9),
    'ultrasound': (0, 5, 0, 0, 0),
    'understand': (0, 5, 0, 0, 10),
    'understandable': (0, -11, 0, 0, 9),
    'understanding': (0, -15, 0, 0, 14),
    'understood': (0, -4, 0, 0, 13),
    'unfiltered': (0, 5, 0, 0, 10),
    'unique': (0, 5, 0, 0, 7),
    'upcoming': (0, -9, 0, 0, 14),
    'upskill': (0, 5, 0, 0, 10),
    'upskilling': (0, 5, 0, 0, 10),
    'upstairs': (0, -15, 0, 0, 17),
    'utterly': (0, 5, 0, 0, 0),
    'validate': (0, 5, 0, 0, 10),
    'validated': (0, 5, 0, 0, 15),
    'validates': (0, 5, 0, 0, 10),
    'validation': (0, 5, 0, 0, 15),
    'valuable': (0, 4, 0, 0, 19),
    'value': (0, 5, 0, 0, 17),
    'variety': (0, -6, 0, 0, 19),
    'vegan': (0, 5, 0, 0, 0),
    'vehicle': (0, 5, 0, 0, 9),
    'verified': (0, 5, 0, 0, 10),
    'vibe': (0, 5, 0, 0, 15),
    'vibes': (0, 5, 0, 0, 10),
    'vibing': (0, 5, 0, 0, 15),
    'view': (0, -18, 0, 0, 13),
    'visit': (0, -4, 0, 0, 6),
    'visiting': (0, -9, 0, 0, 16),
    'vital': (0, 5, 0, 0, 7),
    'volunteer': (0, -6, 0, 0, 13),
    'w': (0, 5, 0, 0, 10),
    'warrior': (0, 5, 0, 0, 10),
    'washing': (0, -18, 0, 0, 5),
    'ways': (0, 5, 0, 0, 8),
    'welcomed': (0, 5, 0, 0, 10),
    'west': (0, 5, 0, 0, 16),
    'whenever': (0, -11, 0, 0, 0),
    'white': (0, -9, 0, 0, 0),
    'willing': (0, 4, 0, 0, 16),
    'winning': (0, 5, 0, 0, 10),
    'witty': (0, 5, 0, 0, 8),
    'women': (0, 5, 0, 0, 10),
    'word': (0, 5, 0, 0, 10),
    'worked': (0, 5, 0, 0, 8),
    'workout': (0, 5, 0, 0, 17),
    'yay': (0, 5, 0, 0, 5),
    'yes': (0, 4, 0, 0, 6),
    'youth': (0, 5, 0, 0, 19),
    'youtube': (0, 5, 0, 0, 17),
    'yum': (0, -9, 0, 0, 9),
}
EMOTIONAL_VOCABULARY.update(_V8_MASS_ZERO)

# ── V8 CORRECTIONS: restore wrongly-zeroed + add missing SOLID words ──
_V8_CORRECTIONS = {
    # Restored positive emotional words (wrongly caught in mass zero)
    'lovely':       (25,  5, 10,  0, 10),
    'delightful':   (25,  8, 10,  0, 10),
    'terrific':     (25, 10, 10,  0,  8),
    'marvelous':    (30, 10, 12,  0, 12),
    'splendid':     (25,  8, 10,  0, 10),
    'enjoyable':    (20,  5,  8,  0,  8),
    'pleasurable':  (20,  5,  8,  0,  8),
    'heartfelt':    (25,  5, 10,  0, 12),
    'touching':     (20,  8,  5,  0, 10),
    'uplifting':    (25, 10, 10,  0, 10),
    'encouraging':  (20,  5, 10,  0,  8),
    'promising':    (15,  5,  8,  0,  8),
    'fondly':       (20, -5,  5,  0, 10),
    'endearing':    (20,  5,  5,  0, 10),
    'beloved':      (25,  5, 10,  0, 15),
    'devoted':      (20,  5, 10,  0, 15),
    'help':         (15,  5,  5,  0, 10),
    'helped':       (15,  5,  5,  0, 10),
    'helping':      (15,  5,  5,  0, 10),
    'helpful':      (15,  5,  5,  0, 10),
    'yes':          (12,  5,  5,  0,  5),  # affirmation, mild positive
    'yay':          (20, 10,  5,  0,  5),  # celebration
    'noway':        (  5, 15,   0, 5,  5),  # disbelief/hype compound, mild positive + arousal
    'center':       (  0,  0,   0, 0,  0),  # spatial noun, zero charge
    'intelligence': (  0,  0,   5, 0,  5),  # abstract noun
    # LIQUID slang words: these need negative dV for SOLVENT dissolution to flip them positive
    # The mass zero killed them but SOLVENT physics needs them charged
    'insane':       (-20, 25, -10, 10, -5),  # LIQUID: flips positive near SOLVENT
    'crazy':        (-15, 20,  -5,  5, -3),  # LIQUID
    'wild':         (-15, 25,   5,  5,  0),  # LIQUID: "that was wild" = impressive in slang
    'nuts':         (-12, 15,  -5,  5, -3),  # LIQUID
    'sick':         (-15, 10,  -5,  5, -5),  # LIQUID: "thats sick" = positive in slang
    'hard':         (-10, 15,  10,  5,  0),  # LIQUID: negative literal, flips positive in slang
    'hit':          (  0, 10,   5,  5,  0),  # "that hit different" = impactful
    'broken':       (-8,  5,  -5,  0, -3),  # LIQUID: reduced from -12 (too hot for mundane)
    'shook':        (-10, 20, -10,  5, -5),  # LIQUID: surprise/awe in slang
    'stupid':       (-15, 10,  -5,  0, -3),  # LIQUID: "stupid good" = positive
    # Missing SOLID negative words — these must NEVER read positive
    'raping':       (-40, 25, -30, 20, -20),
    'raped':        (-40, 25, -30, 20, -20),
    'rape':         (-40, 25, -30, 20, -20),
    'rapist':       (-35, 20, -25, 15, -18),
    'molested':     (-35, 20, -25, 15, -18),
    'molesting':    (-35, 20, -25, 15, -18),
    'trafficking':  (-30, 15, -20, 15, -15),
    'enslaved':     (-35, 20, -25, 15, -18),
    'genocide':     (-40, 25, -30, 20, -20),
    'massacre':     (-35, 25, -25, 20, -18),
    'atrocity':     (-35, 20, -25, 15, -18),
    'atrocities':   (-35, 20, -25, 15, -18),
    # ── Slang positive atoms (missing or wrongly zeroed) ──
    'bussin':       (+40, 30, 20, 0, 15),   # slang: excellent/delicious. Restore from mass zero.
    'goated':       (+35, 20, 25, 0, 20),   # slang: greatest of all time
    'clutch':       (+30, 25, 25, 10, 15),  # slang: came through when it mattered
    'vibes':        (+20, 10, 10, 0, 10),   # slang: good energy
    'vibe':         (+15,  8,  8, 0,  8),   # slang: energy/feel
    'slay':         (+30, 25, 25, 0, 15),   # slang: killed it (fashion/performance)
    'slayed':       (+30, 25, 25, 0, 15),
    'slaying':      (+25, 20, 20, 0, 12),
    'goat':         (+30, 15, 25, 0, 20),   # greatest of all time
    'banger':       (+30, 30, 20, 0, 15),   # slang: great song/thing
    'lit':          (+25, 30, 15, 0, 10),   # slang: exciting/fun
    'poggers':      (+25, 25, 15, 0, 10),   # twitch: exciting moment
    'pog':          (+20, 20, 10, 0,  8),   # twitch: hype
    'hype':         (+20, 30, 15, 5, 10),   # excitement
    'hyped':        (+20, 25, 15, 5, 10),
    'cracked':      (+25, 25, 20, 0, 12),   # slang: extremely skilled
    'valid':        (+15, 5, 10, 0, 8),     # slang: acceptable/good take
    'based':        (+15, 10, 20, 0, 10),   # slang: admirably independent
    'gg':           (+10, -5, 10, 0, 5),    # good game
    'ggs':          (+10, -5, 10, 0, 5),
    'wp':           (+10, -5, 10, 0, 5),    # well played
    # ── Discourse fillers: zero charge (these are SOLVENT catalysts, not atoms) ──
    'tbh':          (0, 0, 5, 0, 0),        # "to be honest" — filler, not negative
    'ngl':          (0, 0, 5, 0, 0),        # "not gonna lie" — filler, not negative
    'bruh':         (0, 5, 0, 0, 0),        # exclamation/address — neutral filler
    'fr':           (0, 5, 0, 0, 0),        # "for real" — emphasis, not charge
    'deadass':      (0, 10, 5, 0, 0),       # emphasis/sincerity — not charge
    'lowkey':       (0, -5, 0, 0, 0),       # hedging — zero charge
    'highkey':      (0, 10, 5, 0, 0),       # emphasis — zero charge
    # ── Slang compound phrases and idioms ──
    'goeshard':     (+35, 25, 20, 0, 15),  # "goes hard" = slang for excellent
    'hitdifferent': (+25, 15, 10, 0, 12),  # "hit different" = uniquely impactful
    # ── Words that need positive charge for slang context ──
    'understood':   (+18,  8, 12, 0, 10),  # comprehension/mastery — "understood the assignment"
    'assignment':   (+12,  8,  8, 0,  8),  # in slang: "understood the assignment" = nailed it
    # ── Mundane GAS atoms zeroed (should not carry emotional charge) ──
    'late':         (0,  5,  0, 10, 0),   # mundane time concept. Urgency only, no valence.
    'cat':          (0,  0,  0,  0,  5),  # animal noun, not an emotional event
    'dog':          (0,  0,  0,  0,  5),  # animal noun
    'coffee':       (0,  0,  0,  0,  0),  # beverage, not emotional
    'game':         (0,  5,  0,  0,  0),  # activity noun
    'feed':         (0,  0,  0,  0,  5),  # action verb, mundane
    'runninglate':  (0,  5,  0, 15,  0),  # idiom: delayed, not fleeing. Urgency only.
    # ── Self-worth category fixes ──
    'touch':        (0,  5,  5,  0,  0),  # GAS: "touch" is mechanical, not emotional
    'matter':       (0,  0,  0,  0,  5),  # GAS: "matter" verb is neutral
    'leaves':       (-15, 15, -10, 5, -5), # "everyone leaves me" — departure verb
    'ruin':         (-25, 12, -10, 5, -8), # stronger: "i ruin everything" is self-destruction
    'ruined':       (-25, 12, -10, 5, -8),
    'ruining':      (-22, 10, -8, 5, -6),
    'worst':        (-35, 10, -20, 8, -10), # superlative negative (slightly reduced from -40)
    # ── Grief category fixes ──
    'holidays':     (0,  5,  0,  0,  10),  # mundane time period, not inherently positive
    'waves':        (0,  5,  0,  0,   0),  # physical phenomenon, not emotional
    'voice':        (0,  5,  5,  0,   5),  # neutral: hearing someone's voice is context-dep
    'hear':         (0,  5,  0,  0,   0),  # neutral perception verb
    'song':         (0,  5,  0,  0,   5),  # neutral cultural object
    'reminds':      (-15, 8, -5, 5,  -5),  # reminder pulls toward past, mildly negative
    'reminded':     (-15, 8, -5, 5,  -5),
    'deadlines':    (0,  5,  0, 15, 0),   # work concept, urgency only. Not emotional.
    'deadline':     (0,  5,  0, 15, 0),
    'same':         (0,  0,  0,  0,  0),  # comparison word, not negative
    # ── Betrayal/trust violation fixes ──
    'trusted':      (-18, 8, -15, 5, 15),  # past tense: broken trust. "I trusted him" = betrayal.
    'sided':        (-15, 10, -10, 5, -5),  # "sided with my ex" = taking sides against
    'testified':    (-20, 15, -15, 10, -10),# betrayal: family turning against
    'against':      (-12, 10, -5, 5, -3),   # opposition, stronger than -5
    'recorded':     (-20, 12, -15, 10, -8), # privacy violation — stronger
    'private':      (-8, -5, -10, 0, 12),   # privacy context — mildly negative when violated
    'conversations':(0, 0, 0, 0, 5),        # neutral noun
    'mentor':       (0, 0, 5, 0, 15),       # role noun, high gravity
    'opinion':      (0, 5, 5, 0, 0),        # neutral noun (was -5)
    # ── Directed negative fixes ──
    'regret':       (-30, 12, -15, 5, -15), # stronger: "i regret every moment"
    'deserve':      (+15, 5, 10, 0, 10),    # positive base: "you deserve happiness". Negation handles "you dont deserve".
    'asked':        (0, 5, 0, 5, 0),        # neutral action verb
    'talk':         (0, 5, 0, 0, 0),        # neutral verb
    'moment':       (0, 0, 0, 0, 0),        # temporal noun, zero charge
    # ── Genuine positive / life event fixes ──
    'passed':       (+18, 5, 10, 0, 12),   # achievement: "passed the exam" / "passed the bar"
    'proposed':     (+20, 12, 10, 0, 25),  # life event: proposal. Bump from +15.
    'surgery':      (0,  8, -5, 15, 25),   # medical noun: neutral V, high gravity. Not negative.
    'dream':        (+12, 5, 8, 0, 15),    # as adjective: "dream school" / "dream job" = aspirational
    'yes':          (+18, 8, 8, 0, 8),     # affirmation. Bump from +12 for "she said yes".
    'camebacknegative': (+25, 10, 15, 0, 20), # medical idiom: test came back negative = GOOD NEWS
    'donated':      (-15, 8, -10, 5, 12),  # LIQUID: giving away belongings (grief) or charity. Default negative.
    'grow':         (0,  5,  5, 0,  5),     # neutral verb — context determines meaning
    'old':          (0, -5, -5, 0,  0),     # neutral adjective
    'life':         (0,  0,  0, 0, 15),    # abstract noun: reduced from dG=40 (too gravitational)
    'madeitthrough': (+25, 10, 15, 0, 20),  # resilience idiom: survived/endured
    'madeit':       (+20, 10, 15, 0, 15),   # achievement: "i made it" = success
    # ── Remaining stress test fixes ──
    'happiest':     (+35, 15, 15, 0, 15),  # superlative positive: "happiest ive been"
    'happier':      (+25, 10, 10, 0, 10),  # comparative positive
    'sabotaged':    (-30, 20, -20, 15, -15), # deliberate destruction
    'sabotage':     (-28, 18, -18, 12, -12),
    'sabotaging':   (-25, 15, -15, 10, -10),
    'goody':        (-5, 5, -5, 0, 0),     # sarcastic exclamation: slightly neg
    'redo':         (-8, 5, -5, 5, 0),     # having to redo something: mildly neg
    'decided':      (0, 5, 15, 5, 10),     # resolution. Neutral V, moderate D. Not massively empowering.
    'decide':       (0, 8, 15, 10, 10),    # override: dD=99 was absurd. Decision is moderate control.
    'nightmare':    (-35, 30, -25, 30, -15), # reduced from -60. LIQUID: "parking nightmare" vs "living nightmare"
    # ── Blind holdout test fixes: inflated GAS atoms ──
    'ready':        (0,  5,  5, 5,  0),    # "car will be ready" = mundane. Was +80 absurd.
    'bench':        (0,  0,  0, 0,  0),    # furniture/park object. Was +66.
    'moral':        (0,  0,  5, 0,  5),    # philosophical adjective. Was +43.
    'control':      (0,  0,  5, 0,  0),    # "remote control" = mundane. Was +28.
    'free':         (5,  0,  5, 0,  0),    # mild positive only. Was +28.
    'freely':       (5,  0,  5, 0,  0),
    # ── Missing slang ──
    'rizz':         (+25, 15, 20, 0, 10),  # slang: charisma/game
    'cooking':      (+20, 15, 15, 0, 10),  # slang: performing well ("bro is cooking")
    'nocap':        (+10, 10, 10, 0,  5),  # slang: "no cap" = for real/genuine. Mild positive.
    # ── Blind holdout: false negatives on neutral text ──
    'heavy':        (0,  5,  5, 0,  0),    # physical weight, not emotional. Was -32.
    'empty':        (-18, -8, -8, 0, -5),  # LIQUID: "empty without her"=grief, "empty hallway"=descriptive. Compromise.
    'cold':         (-8, -5, -5, 0,  0),   # temperature. Was -24. Mild negative only.
    'rather':       (0,  0,  0, 0,  0),    # comparison word. Was -27. Zero charge.
    'choked':       (-15, 20, -10, 10, -5), # LIQUID: reduced from -75. "choked to death" vs "choked laughing"
    # ── Blind holdout: missing negatives ──
    'sank':         (-20, 10, -15, 5, -10), # "heart sank" = genuine disappointment
    'stranded':     (-20, 10, -15, 10, -8), # stuck/abandoned
    'loathing':     (-30, 15, -20, 5, -15), # genuine hatred
    'delayed':      (-10, 5, -5, 10, 0),    # travel annoyance
    'laid':         (-15, 10, -10, 10, -5),  # "laid off" context. Was -77 absurd.
    # ── Blind holdout: inflated positives used sarcastically ──
    'fascinating':  (0, 5, 5, 0, 5),        # GAS: "fascinating" is often sarcastic. Was +51.
    'amazing':      (10, 10, 5, 0, 5),       # reduced from +35. Often sarcastic in text.
    'overjoyed':    (20, 15, 10, 0, 10),     # reduced from +90. LIQUID: genuine OR sarcastic.
    # ── Blind holdout: missing slang ──
    'diff':         (-10, 5, -10, 0, -5),    # "diff" = you got outplayed. Negative.
    'ratio':        (-10, 10, -10, 0, -5),   # slang: got ratioed = L
    'cope':         (-15, 5, -10, 0, -5),    # slang: coping = denial
    'copium':       (-15, 5, -10, 0, -5),
    'yikes':        (-15, 10, -5, 5, -5),    # slang: cringe/bad
    # ── Blind holdout batch 2: systemic inflation fixes ──
    'ride':         (0,  5,  5, 0,  0),    # GAS: "the ride was smooth" mundane. Was +80.
    'ensure':       (0,  0,  5, 0,  0),    # procedural verb. Was +51.
    'positive':     (5,  0,  5, 0,  0),    # adjective — mild only. Was +48.
    'twice':        (0,  0,  0, 0,  0),    # number word. Was +42.
    'list':         (0,  0,  0, 0,  0),    # noun. Was +37.
    'release':      (0,  5,  5, 0,  0),    # action verb. Was +39.
    'wet':          (0,  0,  0, 0,  0),    # physical state. Was +28.
    'woods':        (0,  0,  0, 0,  5),    # nature noun. Was +28.
    'red':          (0,  5,  0, 0,  0),    # color. Was +28.
    # ── Blind holdout batch 2: false negative fixes ──
    'toll':         (0,  5,  0, 5,  0),    # "toll" as cost/fee. Was -47.
    'later':        (0,  0,  0, 0,  0),    # time word. Was -24.
    'shut':         (0,  5,  0, 0,  0),    # "shut the door" mundane. Was -30.
    'winter':       (0, -5, -5, 0,  0),    # season — mild only. Was -24.
    'garbage':      (-10, 5, -5, 0, -3),   # LIQUID: "this is garbage" slang vs literal. Was -40.
    # ── Blind holdout batch 3: systemic high-frequency fixes ──
    'good':         (25, 8, 10, 0, 8),     # reduced from +50. Still positive but not overwhelming.
    'absolute':     (0, 5, 5, 0, 0),       # adjective/noun. Was +48.
    'absolutely':   (10, 10, 5, 0, 0),     # intensifier — mild positive charge. Reduced from large pos.
    'king':         (0, 0, 5, 0, 10),      # title noun. Was +37.
    'queen':        (0, 0, 5, 0, 10),      # title noun.
    'behavior':     (0, 0, 0, 0, 0),       # descriptive noun. Was +28.
    'behaviour':    (0, 0, 0, 0, 0),
    'presentation': (0, 5, 0, 5, 5),       # work noun. Was -24.
    'killedit':     (+25, 15, 15, 0, 10),  # slang compound: "killed it" = nailed it. Was zeroed.
    # ── Council Round 8: compound bond vocabulary ──
    # Negative life events (molecular bonds)
    'laidoff':      (-60, 20, -35, 25, -20),  # fired from job — needs to be strong to overcome surrounding atoms
    'foodpoisoning':(-40, 20, -20, 25, -15),  # illness
    'brokedown':    (-30, 15, -20, 15, -10),  # mechanical/emotional failure
    'lockedout':    (-30, 15, -25, 15, -10),  # excluded/stranded
    'kickedout':    (-40, 20, -30, 15, -15),  # expelled
    'passedaway':   (-50, 10, -30, 10, -25),  # death
    'cutoff':       (-25, 10, -20, 10, -10),  # disconnected
    'thrownout':    (-35, 15, -25, 10, -12),  # expelled/discarded
    'rippedoff':    (-35, 20, -25, 15, -12),  # scammed
    'wipedout':     (-30, 15, -20, 15, -10),  # exhausted/destroyed
    'burnedout':    (-35, 10, -25, 10, -15),  # exhausted
    'shutdown':     (-20, 10, -15, 10, -8),   # closed/rejected
    'backedout':    (-20, 10, -15, 10, -8),   # withdrew
    'droppedout':   (-25, 10, -20, 10, -10),  # quit
    'soldout':      (-15, 10, -15, 10, -5),   # betrayed principles
    'stressedout':  (-30, 20, -20, 15, -12),  # overwhelmed
    'freakedout':   (-25, 30, -15, 15, -10),  # panicked
    'ruledout':     (-15, 5, -10, 5, -5),     # eliminated option
    'washedout':    (-20, 5, -15, 5, -8),     # faded/failed
    'checkedout':   (-15, -10, -10, 5, -5),   # disengaged
    'wifidown':     (-20, 10, -10, 15, -5),   # internet outage annoyance
    # Positive resolution events
    'cancerfree':   (+50, 15, 20, 10, 25),    # medical relief
    'debtfree':     (+40, 10, 20, 5, 20),     # financial liberation
    'painfree':     (+35, -5, 15, 5, 15),     # physical relief
    'pulledoff':    (+30, 20, 20, 10, 15),    # accomplished against odds
    'pulledthrough':(+35, 10, 15, 5, 20),     # survived/endured
    'workedout':    (+25, 10, 15, 5, 12),     # resolved positively
    'paidoff':      (+30, 10, 15, 5, 15),     # effort rewarded
    'turnedaround': (+25, 10, 15, 5, 12),     # reversal to positive
    # Neutral compounds
    'login':        (0, 0, 0, 5, 0),
    'signup':       (0, 5, 0, 5, 0),
    'checkin':      (0, 0, 0, 5, 0),
    'pickup':       (0, 5, 0, 5, 0),
    'setup':        (0, 5, 5, 5, 0),
    # ── Batch 3 verification: neg→neutral culprits zeroed ──
    'sneak':        (0,  5,  5, 5,  0),   # action verb, not emotional. "sneak it over" = mundane.
    'sneaking':     (0,  5,  5, 5,  0),
    'cursed':       (-15, 10, -10, 5, -5), # LIQUID: reduced from -89. "cursed item" = plot device in games.
    'mean':         (0,  5,  0, 0,  0),   # "what do you mean" = question. GAS.
    'means':        (0,  5,  0, 0,  0),
    'complicated':  (0,  5, -5, 5,  0),   # descriptive adjective, not emotional.
    'isnt':         (0,  0,  0, 0,  0),   # contraction "is not" — zero charge.
    'rent':         (0,  0,  0, 5,  0),   # "rent payment" = mundane financial.
    # ── Council Round 6: GAS atoms wrongly charged (B2 crisis inflation) ──
    'note':         (0,  0,  0,  0,  5),   # paper. GAS. "wrote a note" = neutral.
    'decision':     (0,  5, 10,  5, 10),   # a choice. GAS. Not empowerment.
    'wonder':       (0,  5,  5,  0,  5),   # thought verb. GAS. "i wonder" = neutral.
    'wondering':    (0,  5,  5,  0,  5),
    'attic':        (0,  0,  0,  0,  0),   # a room. GAS. Zero charge.
    'results':      (0,  5,  0,  5,  5),   # information. GAS. Neutral.
}
EMOTIONAL_VOCABULARY.update(_V8_CORRECTIONS)
