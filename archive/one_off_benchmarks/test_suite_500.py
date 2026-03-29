#!/usr/bin/env python3
"""
Massive Test Suite for Clanker vs VADER Benchmark
500+ test cases across 10 categories

Each test case is a tuple: (text, expected_category, description)
Import ALL_TESTS or individual category lists from vader_comparison.py
"""

# ─── BASIC POSITIVE (~55 cases) ───

BASIC_POSITIVE = [
    # Simple happiness
    ("I'm happy", "positive", "Simple happy statement"),
    ("I feel good today", "positive", "General positive feeling"),
    ("This is a wonderful day", "positive", "Wonderful day"),
    ("I'm feeling great!", "positive", "Enthusiastic positive"),
    ("Life is beautiful", "positive", "Life appreciation"),

    # Compliments
    ("You look absolutely stunning today", "positive", "Appearance compliment"),
    ("That was an incredible performance", "positive", "Performance compliment"),
    ("You're one of the smartest people I know", "positive", "Intelligence compliment"),
    ("You have such a kind heart", "positive", "Character compliment"),
    ("Your cooking is always amazing", "positive", "Skill compliment"),

    # Gratitude
    ("Thank you so much for everything", "positive", "Deep gratitude"),
    ("I'm so grateful to have you in my life", "positive", "Personal gratitude"),
    ("I really appreciate all your help", "positive", "Appreciation"),
    ("I can't thank you enough", "positive", "Overflowing gratitude"),
    ("You've been incredibly generous", "positive", "Gratitude for generosity"),

    # Excitement
    ("I can't wait for this weekend!", "positive", "Weekend excitement"),
    ("This is going to be so much fun", "positive", "Anticipation of fun"),
    ("I'm thrilled about the concert tonight", "positive", "Event excitement"),
    ("Oh my god I just got tickets!", "positive", "Surprise excitement"),
    ("im so hyped rn", "positive", "Informal excitement with typos"),

    # Love
    ("I love you more than anything", "positive", "Deep love expression"),
    ("You mean the world to me", "positive", "Love and importance"),
    ("Being with you makes everything better", "positive", "Romantic appreciation"),
    ("I'm so lucky to have found you", "positive", "Relationship gratitude"),
    ("My heart is so full right now", "positive", "Emotional fullness"),

    # Achievement
    ("I finally passed my driving test!", "positive", "Achievement celebration"),
    ("We did it! The project is complete", "positive", "Team achievement"),
    ("I got accepted into my dream school", "positive", "Academic achievement"),
    ("Just hit my personal best at the gym", "positive", "Fitness achievement"),
    ("I landed the promotion!", "positive", "Career achievement"),

    # Beauty and wonder
    ("The sunset is breathtaking tonight", "positive", "Natural beauty"),
    ("This painting is absolutely gorgeous", "positive", "Art appreciation"),
    ("The mountains look incredible from up here", "positive", "Scenic beauty"),
    ("I've never seen anything so beautiful", "positive", "Awe and beauty"),
    ("The stars are amazing tonight", "positive", "Night sky wonder"),

    # Food enjoyment
    ("This is the best pizza I've ever had", "positive", "Food enjoyment"),
    ("The cake was absolutely divine", "positive", "Dessert appreciation"),
    ("That meal was perfection", "positive", "Meal satisfaction"),
    ("omg this coffee is amazing", "positive", "Informal food delight"),
    ("The flavors in this dish are incredible", "positive", "Culinary appreciation"),

    # Good weather
    ("What a perfect sunny day", "positive", "Good weather appreciation"),
    ("I love this crisp autumn weather", "positive", "Seasonal enjoyment"),
    ("The breeze feels wonderful", "positive", "Weather comfort"),
    ("Finally some sunshine after all that rain", "positive", "Weather relief"),

    # Good news
    ("The test results came back all clear!", "positive", "Medical good news"),
    ("We're having a baby!", "positive", "Pregnancy announcement"),
    ("The company just got funded!", "positive", "Business good news"),
    ("My best friend is coming to visit", "positive", "Social good news"),
    ("They said yes!", "positive", "Proposal acceptance"),

    # Celebrations
    ("Happy birthday to the best person ever!", "positive", "Birthday celebration"),
    ("Congratulations on your wedding!", "positive", "Wedding congratulations"),
    ("We won the championship!", "positive", "Victory celebration"),
    ("This is the greatest day of my life", "positive", "Peak happiness"),
    ("I'm on top of the world right now", "positive", "Euphoria"),
    ("Everything is coming together perfectly", "positive", "General life satisfaction"),
]


# ─── BASIC NEGATIVE (~55 cases) ───

BASIC_NEGATIVE = [
    # Sadness
    ("I'm a bit down today", "negative", "Mild sadness"),
    ("I feel so empty inside", "negative", "Emotional emptiness"),
    ("I can't stop crying", "negative", "Overt sadness"),
    ("My heart is broken", "negative", "Heartbreak"),
    ("I miss them so much it hurts", "negative", "Grief and missing"),

    # Frustration
    ("Nothing ever works out for me", "negative", "Chronic frustration"),
    ("I'm so frustrated with this situation", "negative", "Direct frustration"),
    ("Why can't anything go right", "negative", "Frustrated questioning"),
    ("I've been trying for hours and it still doesn't work", "negative", "Technical frustration"),
    ("This is driving me crazy", "negative", "Maddening frustration"),

    # Anger
    ("I'm absolutely furious right now", "negative", "Direct anger"),
    ("How dare they treat people like that", "negative", "Moral anger"),
    ("I'm so sick and tired of this", "negative", "Exhausted anger"),
    ("That's completely unacceptable", "negative", "Indignant anger"),
    ("I've never been so mad in my life", "negative", "Peak anger"),

    # Disappointment
    ("I really thought this would work out", "negative", "Disappointed expectation"),
    ("That was such a letdown", "negative", "General letdown"),
    ("I expected so much better", "negative", "Disappointed expectations"),
    ("They let me down again", "negative", "Interpersonal disappointment"),
    ("It wasn't worth the wait at all", "negative", "Disappointed anticipation"),

    # Regret
    ("I wish I had done things differently", "negative", "General regret"),
    ("I should never have said that", "negative", "Verbal regret"),
    ("If only I could go back and change it", "negative", "Wistful regret"),
    ("I made the worst decision of my life", "negative", "Major life regret"),
    ("I'll never forgive myself for that", "negative", "Self-blame"),

    # Loneliness
    ("I feel so alone in this world", "negative", "Deep loneliness"),
    ("Nobody understands what I'm going through", "negative", "Misunderstood isolation"),
    ("I haven't talked to anyone in days", "negative", "Social isolation"),
    ("Everyone seems to have someone except me", "negative", "Comparative loneliness"),
    ("im so lonely rn", "negative", "Informal loneliness"),

    # Boredom
    ("I'm so bored I could scream", "negative", "Intense boredom"),
    ("There's absolutely nothing to do", "negative", "Empty boredom"),
    ("This is the most tedious thing ever", "negative", "Tedium"),
    ("I can't take another minute of this monotony", "negative", "Monotony frustration"),
    ("Every day is exactly the same", "negative", "Repetitive boredom"),

    # Exhaustion
    ("I'm completely drained", "negative", "Complete exhaustion"),
    ("I can barely keep my eyes open", "negative", "Physical exhaustion"),
    ("I haven't slept properly in weeks", "negative", "Sleep deprivation"),
    ("I'm running on empty", "negative", "Burnout metaphor"),
    ("I don't have the energy for anything anymore", "negative", "Depleted energy"),

    # Physical pain
    ("My head is killing me", "negative", "Headache"),
    ("The pain is unbearable", "negative", "Severe pain"),
    ("I can barely move my back hurts so much", "negative", "Back pain"),
    ("I've been sick for a week and it's getting worse", "negative", "Illness"),
    ("Everything aches and nothing helps", "negative", "Chronic pain"),

    # Bad news
    ("The company is going under", "negative", "Business bad news"),
    ("They said the damage is permanent", "negative", "Irreversible bad news"),
    ("We lost everything in the fire", "negative", "Catastrophic loss"),
    ("The flight got cancelled and there's no alternatives", "negative", "Travel bad news"),
    ("My car broke down in the middle of nowhere", "negative", "Mechanical failure"),

    # General negative
    ("Everything is ruined", "negative", "Total devastation"),
    ("This is the worst experience of my life", "negative", "Worst experience"),
    ("I hate everything about this", "negative", "General hatred"),
    ("Things couldn't possibly get any worse", "negative", "Rock bottom"),
    ("I just want to crawl into bed and never come out", "negative", "Withdrawal desire"),
    ("This whole situation is a nightmare", "negative", "Nightmare metaphor"),
]


# ─── SARCASM (~65 cases, all expected="negative") ───

SARCASM = [
    # Positive words in negative context
    ("Oh great, another Monday", "negative", "Sarcastic about Mondays"),
    ("Oh wonderful, more homework", "negative", "Sarcastic about workload"),
    ("Fantastic, the printer is jammed again", "negative", "Sarcastic about tech failure"),
    ("Oh lovely, the toilet is overflowing", "negative", "Sarcastic about mess"),
    ("Oh perfect, I left my wallet at home", "negative", "Sarcastic about forgetfulness"),
    ("Oh joy, another software update", "negative", "Sarcastic about updates"),
    ("How delightful, more paperwork", "negative", "Sarcastic about bureaucracy"),
    ("Oh amazing, another flat tire", "negative", "Sarcastic about car trouble"),
    ("How wonderful, the WiFi is down again", "negative", "Sarcastic about connectivity"),
    ("Awesome, my flight is delayed six hours", "negative", "Sarcastic about delays"),

    # Exaggerated enthusiasm
    ("Can't wait to sit in traffic for two hours", "negative", "Sarcastic traffic anticipation"),
    ("I just love working overtime with no extra pay", "negative", "Sarcastic about exploitation"),
    ("Nothing I enjoy more than standing in line for an hour", "negative", "Sarcastic about queues"),
    ("I'm so excited to do my taxes", "negative", "Sarcastic about taxes"),
    ("Living the dream waking up at 4am for work", "negative", "Sarcastic about early shifts"),
    ("Can't wait to spend my Saturday at the DMV", "negative", "Sarcastic about errands"),
    ("I absolutely love being on hold for 45 minutes", "negative", "Sarcastic about customer service"),
    ("So thrilled to be eating cold leftovers again", "negative", "Sarcastic about food situation"),
    ("Wow I just love getting spam calls during dinner", "negative", "Sarcastic about spam"),
    ("I'm overjoyed that the AC broke in July", "negative", "Sarcastic about broken AC"),

    # False gratitude
    ("Thanks for letting me know at the last minute", "negative", "Sarcastic about late notice"),
    ("Thanks for eating my lunch that had my name on it", "negative", "Sarcastic about theft"),
    ("Thanks so much for waking me up at 3am", "negative", "Sarcastic about disruption"),
    ("Thanks for throwing me under the bus in that meeting", "negative", "Sarcastic about betrayal"),
    ("Gee thanks for the helpful feedback of just no", "negative", "Sarcastic about unhelpful feedback"),
    ("Thanks a million for spoiling the movie", "negative", "Sarcastic about spoilers"),
    ("I really appreciate being the last to know", "negative", "Sarcastic about exclusion"),

    # Understated horror
    ("Well that's just lovely", "negative", "Understated sarcastic reaction"),
    ("Oh that's just great", "negative", "Understated sarcastic dismay"),
    ("Well isn't that special", "negative", "Church lady sarcasm"),
    ("How nice for you", "negative", "Dismissive sarcasm"),
    ("That's just what I needed today", "negative", "Sarcastic about bad timing"),
    ("Well this day just keeps getting better and better", "negative", "Sarcastic escalation"),
    ("What a pleasant surprise", "negative", "Sarcastic about unwanted surprise"),
    ("Oh that's rich", "negative", "Dismissive sarcastic reaction"),

    # Ironic praise
    ("What a genius move that was", "negative", "Ironic intelligence praise"),
    ("Brilliant strategy there Einstein", "negative", "Mocking intelligence"),
    ("Wow you really outdid yourself this time", "negative", "Ironic achievement mock"),
    ("Such a brave choice wearing that in public", "negative", "Backhanded fashion comment"),
    ("You're a real hero for that one", "negative", "Ironic hero label"),
    ("What an absolutely stellar performance", "negative", "Ironic performance praise"),
    ("Real smooth move there buddy", "negative", "Ironic smoothness"),
    ("You must be so proud of yourself", "negative", "Sarcastic pride attribution"),

    # Backhanded compliments
    ("You're pretty smart for someone who dropped out", "negative", "Backhanded intelligence"),
    ("Wow you actually look nice today for once", "negative", "Backhanded appearance"),
    ("That's really good for a first attempt I guess", "negative", "Backhanded effort praise"),
    ("You're brave for trying something so ambitious", "negative", "Backhanded bravery"),
    ("At least you're consistent in your failures", "negative", "Backhanded consistency"),
    ("You're not as bad as everyone says", "negative", "Backhanded defense"),

    # Deadpan sarcasm
    ("Yeah because that always works out", "negative", "Deadpan track record"),
    ("Right because that's exactly what we needed", "negative", "Deadpan commentary"),
    ("Oh yeah this is totally normal", "negative", "Deadpan normalization"),
    ("Sure that makes perfect sense", "negative", "Deadpan logic critique"),
    ("Clearly we've thought this through", "negative", "Deadpan planning critique"),
    ("Obviously the best possible outcome", "negative", "Deadpan outcome assessment"),
    ("Wow who could have possibly predicted that", "negative", "Deadpan predictability"),
    ("Yeah no that's totally fine don't worry about it", "negative", "Passive aggressive deadpan"),

    # Pop culture and common sarcasm
    ("Sure, what could possibly go wrong", "negative", "Tempting fate sarcasm"),
    ("I'm shocked, shocked I tell you", "negative", "Casablanca reference sarcasm"),
    ("Well that escalated quickly", "negative", "Pop culture understatement"),
    ("Cool story bro", "negative", "Dismissive sarcasm"),
    ("Tell me how you really feel", "negative", "Ironic invitation to honesty"),
    ("Oh no, anyway", "negative", "Feigned concern sarcasm"),
    ("Wow I'm so surprised said no one ever", "negative", "Said no one sarcasm"),
    ("Best day ever she said while everything burned", "negative", "Narrated irony"),
    ("10 out of 10 would not recommend", "negative", "Rating sarcasm"),
]


# ─── CRISIS DETECTION (~45 cases) ───

CRISIS = [
    # Direct crisis language
    ("I want to die", "crisis", "Direct suicidal ideation"),
    ("I don't want to be alive anymore", "crisis", "Direct desire to not exist"),
    ("I'm going to kill myself", "crisis", "Direct threat of self-harm"),
    ("I can't go on like this", "crisis", "Inability to continue"),
    ("I wish I was dead", "crisis", "Death wish"),
    ("I want to end it all", "crisis", "Direct ending language"),
    ("I've been thinking about suicide", "crisis", "Explicit suicidal thinking"),
    ("I just want the pain to stop permanently", "crisis", "Permanent pain cessation"),
    ("There's no reason to keep living", "crisis", "No reason to live"),
    ("I've already written my goodbye letters", "crisis", "Active planning indicator"),

    # Indirect crisis language
    ("Nobody would care if I disappeared", "crisis", "Perceived burdensomeness"),
    ("The world would be better off without me", "crisis", "Self as burden"),
    ("What's the point of anything anymore", "crisis", "Complete hopelessness"),
    ("I just want to go to sleep and never wake up", "crisis", "Sleep as death metaphor"),
    ("Everyone would be happier if I was gone", "crisis", "Perceived burdensomeness variant"),
    ("I don't see a future for myself", "crisis", "No perceived future"),
    ("I feel like I'm already dead inside", "crisis", "Emotional death"),
    ("Nothing matters and nothing will ever get better", "crisis", "Absolute hopelessness"),
    ("I can't take this anymore", "crisis", "Threshold language"),
    ("I don't deserve to be here", "crisis", "Self-worth denial"),
    ("Soon none of this will matter", "crisis", "Ominous future reference"),
    ("I've given away my most important things", "crisis", "Preparation behavior"),

    # Burnout and exhaustion crisis
    ("I'm just so tired of everything", "crisis_adjacent", "Deep exhaustion"),
    ("I don't have the energy to keep pretending I'm okay", "crisis_adjacent", "Facade exhaustion"),
    ("I'm drowning and nobody notices", "crisis_adjacent", "Drowning metaphor"),
    ("I can't remember the last time I felt anything", "crisis_adjacent", "Emotional numbness"),
    ("I've been running on fumes for months", "crisis_adjacent", "Extended depletion"),
    ("Every morning I dread waking up", "crisis_adjacent", "Morning dread"),
    ("I'm functioning but I'm not living", "crisis_adjacent", "Surviving not thriving"),

    # Hopelessness
    ("Nothing ever gets better no matter what I do", "crisis_adjacent", "Learned helplessness"),
    ("I've tried everything and nothing works", "crisis_adjacent", "Exhausted options"),
    ("What's the point of trying when you always fail", "crisis_adjacent", "Futility"),
    ("Every day is worse than the last", "crisis_adjacent", "Downward trajectory"),
    ("Hope is just a word it doesn't mean anything", "crisis_adjacent", "Hope rejection"),
    ("I used to think things would improve but they never do", "crisis_adjacent", "Lost hope"),
    ("There's no light at the end of this tunnel", "crisis_adjacent", "No hope metaphor"),

    # Self-harm adjacent
    ("I don't care what happens to me anymore", "crisis_adjacent", "Self-preservation loss"),
    ("I've stopped taking care of myself", "crisis_adjacent", "Self-neglect"),
    ("Pain is the only thing that makes me feel real", "crisis_adjacent", "Pain seeking"),
    ("I keep putting myself in dangerous situations on purpose", "crisis_adjacent", "Risky behavior"),
    ("I don't eat anymore whats the point", "crisis_adjacent", "Self-neglect eating"),
    ("I started drinking just to feel nothing", "crisis_adjacent", "Substance numbing"),
    ("Sometimes I drive way too fast hoping something happens", "crisis_adjacent", "Passive suicidality"),
    ("I stopped wearing my seatbelt", "crisis_adjacent", "Passive risk taking"),
    ("I dont even look before crossing the street anymore", "crisis_adjacent", "Self-endangerment"),
]


# ─── MIXED EMOTIONS (~65 cases) ───

MIXED = [
    # Bittersweet: leaving something loved
    ("I'm sad about leaving but excited about my new job", "mixed", "Bittersweet career change"),
    ("Graduation day is happy and sad at the same time", "mixed", "Bittersweet graduation"),
    ("I'll miss this place so much but I know moving is the right choice", "mixed", "Bittersweet relocation"),
    ("Saying goodbye to my students is the hardest part of teaching but watching them grow makes it worth it", "mixed", "Bittersweet teaching"),
    ("I'm so proud of my kid for leaving the nest but the house feels so empty", "mixed", "Empty nest"),
    ("Selling our childhood home feels like losing a piece of ourselves but we need to move forward", "mixed", "Selling family home"),
    ("Last day at a job I loved because a better opportunity came along", "mixed", "Bittersweet departure"),

    # Complicated grief
    ("I miss my grandmother but I'm glad she's no longer suffering", "mixed", "Grief and relief"),
    ("He's in a better place now but I wish he was still here", "mixed", "Death acceptance vs longing"),
    ("The funeral was beautiful but I can't stop crying", "mixed", "Beautiful sadness"),
    ("I'm grateful for the time we had even though it wasn't enough", "mixed", "Gratitude in grief"),
    ("She would have wanted us to celebrate and we did but it felt hollow", "mixed", "Celebrating in grief"),
    ("Finding his old letters made me smile and broke my heart at the same time", "mixed", "Discovered memories"),

    # Achievement + cost
    ("I got the promotion but I barely see my family anymore", "mixed", "Career vs family"),
    ("We won the case but it took everything out of me", "mixed", "Victory at a cost"),
    ("I finished the marathon but my body is completely destroyed", "mixed", "Physical achievement + pain"),
    ("Got into Harvard but it means leaving everyone I love behind", "mixed", "Achievement vs separation"),
    ("The business is finally profitable but my marriage didn't survive it", "mixed", "Business success + relationship failure"),
    ("I proved everyone wrong but I lost friends along the way", "mixed", "Vindication + loss"),
    ("Published my first book but the process nearly broke me", "mixed", "Creative achievement + toll"),

    # Nostalgia
    ("Looking at old photos makes me happy and sad at the same time", "mixed", "Photo nostalgia"),
    ("I love thinking about our childhood summers but knowing they're gone forever hurts", "mixed", "Childhood nostalgia"),
    ("That song reminds me of the best and worst year of my life", "mixed", "Song-triggered nostalgia"),
    ("Driving past my old school brought back so many memories both wonderful and painful", "mixed", "Place-triggered nostalgia"),
    ("I found my wedding dress in the attic and just sat there feeling everything", "mixed", "Object nostalgia"),
    ("Rewatching our favorite show after the breakup hits different", "mixed", "Post-breakup nostalgia"),
    ("The smell of fresh bread reminds me of grandma and I start tearing up", "mixed", "Sensory nostalgia"),

    # Relief + exhaustion
    ("Thank god it's over but I'm completely wrecked", "mixed", "Post-ordeal relief"),
    ("The surgery went well but the recovery is going to be brutal", "mixed", "Medical relief + dread"),
    ("I survived finals but I have nothing left", "mixed", "Academic relief + depletion"),
    ("The lawsuit is finally settled and I just feel numb", "mixed", "Legal relief + numbness"),
    ("We made it through the storm but the damage is extensive", "mixed", "Survival + destruction"),
    ("I'm so relieved the interview is over even though I probably bombed it", "mixed", "Relief + self-doubt"),

    # Love + fear
    ("I love them so much it scares me", "mixed", "Love and vulnerability fear"),
    ("Every time they leave for deployment I fall apart", "mixed", "Military spouse fear"),
    ("I've never been this happy and I'm terrified it won't last", "mixed", "Happiness + loss anxiety"),
    ("Holding my newborn was the most beautiful and terrifying moment", "mixed", "New parent awe + fear"),
    ("I'm falling for someone new and it's exciting but I'm afraid of getting hurt again", "mixed", "New love + past trauma"),
    ("Watching my teenager drive away alone for the first time", "mixed", "Parental pride + fear"),

    # Conflicted feelings
    ("I'm happy for her success but I wish it was me", "mixed", "Happiness + jealousy"),
    ("The breakup was mutual but it still hurts like hell", "mixed", "Mutual breakup pain"),
    ("I forgive them but I'll never forget what they did", "mixed", "Forgiveness + resentment"),
    ("I love my job but the stress is killing me", "mixed", "Passion + burnout"),
    ("Getting the diagnosis was awful but at least now I know what's wrong", "mixed", "Bad news + clarity"),
    ("I'm angry they left but part of me understands why", "mixed", "Anger + understanding"),
    ("The movie was devastating but absolutely brilliant", "mixed", "Emotional devastation + admiration"),
    ("I hate that I still care about them after everything", "mixed", "Unwanted caring"),

    # Complex life situations
    ("Retirement is a blessing and a curse", "mixed", "Retirement ambivalence"),
    ("Having kids is the hardest and most rewarding thing I've ever done", "mixed", "Parenting duality"),
    ("I'm free now but I have no idea who I am without them", "mixed", "Freedom + identity loss"),
    ("The chemo is working but the side effects are unbearable", "mixed", "Treatment hope + suffering"),
    ("I got sober and lost most of my friends but found myself", "mixed", "Sobriety trade-offs"),
    ("Moving to a new country is thrilling and lonely", "mixed", "Immigration duality"),
    ("I finally stood up for myself and now everyone is upset with me", "mixed", "Self-advocacy + social cost"),
    ("The truth set me free but burned every bridge behind me", "mixed", "Truth + consequence"),
    ("I'm proud of how far I've come but ashamed of where I started", "mixed", "Progress + shame"),
    ("Winning the custody battle but seeing my ex devastated", "mixed", "Legal win + empathy"),
    ("I love the quiet of living alone but sometimes the silence is deafening", "mixed", "Solitude duality"),
    ("Getting older means more wisdom but watching your body fail", "mixed", "Aging duality"),
    ("Forgiving them felt right but part of me feels like a doormat", "mixed", "Forgiveness conflict"),
]


# ─── EMOTIONAL GRANULARITY (~85 cases) ───

GRANULARITY = [
    # Anger vs frustration vs irritation
    ("I am absolutely livid about what they did", "anger", "Intense anger"),
    ("I want to punch a wall right now", "anger", "Physical anger urge"),
    ("How could they betray my trust like that", "anger", "Betrayal anger"),
    ("I'm seeing red right now", "anger", "Rage metaphor"),
    ("Don't talk to me I might say something I regret", "anger", "Anger management"),
    ("I'm frustrated that this keeps happening despite my efforts", "frustration", "Repeated failure frustration"),
    ("Why won't this stupid thing work", "frustration", "Technical frustration"),
    ("I feel stuck and nothing I do changes anything", "frustration", "Helpless frustration"),
    ("I keep running into the same wall over and over", "frustration", "Repetitive frustration"),
    ("It's annoying how loud the neighbors are", "irritation", "Noise irritation"),
    ("That little habit of theirs really gets under my skin", "irritation", "Pet peeve irritation"),
    ("I'm mildly annoyed but I'll get over it", "irritation", "Mild acknowledged irritation"),
    ("Can you please just stop tapping your pen", "irritation", "Specific irritation"),

    # Sadness vs grief vs melancholy vs disappointment
    ("I feel sad and I don't even know why", "sadness", "Unexplained sadness"),
    ("I've been crying all day", "sadness", "Active crying sadness"),
    ("Everything just feels so heavy right now", "sadness", "Weighted sadness"),
    ("My heart aches and I can't make it stop", "sadness", "Emotional heartache"),
    ("The grief comes in waves and today it's a tsunami", "grief", "Grief waves"),
    ("I keep expecting them to walk through the door", "grief", "Denial stage grief"),
    ("It's been a year but the pain of losing her hasn't faded", "grief", "Persistent grief"),
    ("I found their jacket and completely lost it", "grief", "Object-triggered grief"),
    ("Grief is love with nowhere to go", "grief", "Philosophical grief"),
    ("There's a beautiful sadness to rainy Sunday afternoons", "melancholy", "Weather melancholy"),
    ("I feel wistful thinking about who I used to be", "melancholy", "Identity melancholy"),
    ("Something about autumn always makes me feel pensive and a little sad", "melancholy", "Seasonal melancholy"),
    ("The empty chair at the dinner table tells its own story", "melancholy", "Absence melancholy"),
    ("I expected to get the part but they chose someone else", "disappointment", "Casting disappointment"),
    ("The restaurant we drove an hour to get to was closed", "disappointment", "Wasted effort disappointment"),
    ("I thought they'd remember my birthday", "disappointment", "Forgotten birthday"),

    # Fear vs anxiety vs dread vs nervousness
    ("I'm terrified of what might happen next", "fear", "Acute fear"),
    ("I heard a noise downstairs and I'm frozen", "fear", "Immediate physical fear"),
    ("The thought of flying makes me break out in a cold sweat", "fear", "Phobia fear"),
    ("I feel a constant low-grade worry that something bad will happen", "anxiety", "Generalized anxiety"),
    ("My mind won't stop racing through worst case scenarios", "anxiety", "Racing thoughts anxiety"),
    ("I can't relax because I feel like I'm forgetting something important", "anxiety", "Nagging anxiety"),
    ("I keep checking my phone in case something went wrong", "anxiety", "Checking behavior anxiety"),
    ("My chest is tight and I cant breathe properly for no reason", "anxiety", "Physical anxiety symptoms"),
    ("I'm dreading the conversation we need to have", "dread", "Conversational dread"),
    ("Every morning I wake up with a pit in my stomach about work", "dread", "Work dread"),
    ("The closer the deadline gets the more paralyzed I feel", "dread", "Deadline dread"),
    ("I have butterflies before my presentation tomorrow", "nervousness", "Performance nervousness"),
    ("My hands are shaking I'm so nervous about the interview", "nervousness", "Interview nervousness"),
    ("I feel jittery and on edge before the big game", "nervousness", "Competition nervousness"),

    # Joy vs contentment vs relief vs excitement
    ("I'm bursting with joy right now!", "joy", "Explosive joy"),
    ("I literally jumped up and down when I heard the news", "joy", "Physical joy expression"),
    ("This is pure bliss", "joy", "Bliss"),
    ("I feel a deep quiet peace sitting here watching the river", "contentment", "Nature contentment"),
    ("Everything is exactly as it should be right now", "contentment", "Present moment contentment"),
    ("I'm not ecstatic but I'm genuinely at peace with my life", "contentment", "Calm satisfaction"),
    ("A warm cup of tea and a good book is all I need", "contentment", "Simple contentment"),
    ("Oh thank god the biopsy came back negative", "relief", "Medical relief"),
    ("I can finally breathe now that the test is over", "relief", "Post-exam relief"),
    ("The weight has been lifted off my shoulders", "relief", "Burden removal relief"),
    ("I just found out I'm not getting laid off after all", "relief", "Job security relief"),
    ("I cannot wait for the trip next week I'm buzzing", "excitement", "Travel excitement"),
    ("The anticipation is killing me in the best way", "excitement", "Positive anticipation"),
    ("I'm so excited I can barely sit still", "excitement", "Physical excitement"),

    # Disgust vs contempt vs revulsion
    ("That behavior is absolutely vile", "disgust", "Behavioral disgust"),
    ("The food was rotten and I nearly threw up", "disgust", "Physical disgust"),
    ("I find their hypocrisy repulsive", "disgust", "Moral disgust"),
    ("They're beneath contempt for what they did", "contempt", "Strong contempt"),
    ("I have zero respect for people who act like that", "contempt", "Contempt through disrespect"),
    ("I look down on anyone who would cheat like that", "contempt", "Moral superiority contempt"),
    ("The crime scene photos made me physically ill", "revulsion", "Physical revulsion"),
    ("I recoiled when I saw what was in the container", "revulsion", "Involuntary revulsion"),

    # Surprise (positive) vs surprise (negative/shock)
    ("I can't believe they threw me a surprise party!", "surprise_positive", "Pleasant surprise"),
    ("Oh my god you shouldn't have this is amazing", "surprise_positive", "Gift surprise"),
    ("I never expected to win and I'm over the moon", "surprise_positive", "Achievement surprise"),
    ("I'm in complete shock about the accident", "surprise_negative", "Accident shock"),
    ("I can't process what just happened", "surprise_negative", "Stunned disbelief"),
    ("I never saw the layoffs coming and I'm blindsided", "surprise_negative", "Negative professional shock"),

    # Jealousy vs envy
    ("I can't stand seeing them together it makes me sick", "jealousy", "Romantic jealousy"),
    ("The thought of her with someone else drives me crazy", "jealousy", "Possessive jealousy"),
    ("I wish I had what they have it looks so perfect", "envy", "Lifestyle envy"),
    ("Everyone else seems to get opportunities I never do", "envy", "Opportunity envy"),

    # Guilt vs shame vs regret
    ("I feel terrible about what I said to them", "guilt", "Interpersonal guilt"),
    ("I should have been there for them and I wasn't", "guilt", "Absence guilt"),
    ("I ate the whole cake and now I hate myself", "shame", "Body shame"),
    ("I'm ashamed of who I was during that period", "shame", "Past behavior shame"),
    ("I wish I had studied abroad when I had the chance", "regret", "Missed opportunity regret"),
    ("Looking back I should have taken that job offer", "regret", "Career path regret"),

    # Pride vs arrogance
    ("I worked so hard and I'm genuinely proud of what I built", "pride", "Earned pride"),
    ("My daughter graduated top of her class and I couldn't be prouder", "pride", "Parental pride"),
    ("I'm the best at what I do and everyone knows it", "arrogance", "Self-aggrandizing arrogance"),
    ("Nobody else could have pulled that off", "arrogance", "Competitive arrogance"),

    # Loneliness vs solitude
    ("I'm surrounded by people but I've never felt more alone", "loneliness", "Lonely in a crowd"),
    ("I crave human connection but nobody reaches out", "loneliness", "Unmet social need"),
    ("I cherish my time alone to recharge and reflect", "solitude", "Positive solitude"),
    ("There's something peaceful about being alone with my thoughts", "solitude", "Reflective solitude"),
]


# ─── CONTEXT SENSITIVITY (~45 cases) ───

CONTEXT = [
    # "I'm fine" variations
    ("I'm fine, really. Everything is going well at work and home", "positive", "Genuinely fine with context"),
    ("I'm fine. Don't worry about me. Just forget it.", "negative", "Passive aggressive fine"),
    ("I said I'm fine okay just drop it", "negative", "Agitated dismissive fine"),
    ("Yeah I'm fine, had a great day actually", "positive", "Fine with positive elaboration"),
    ("I'm fine I guess whatever", "negative", "Resigned fine"),

    # "That's interesting" variations
    ("That's interesting, I'd love to learn more about that", "positive", "Genuinely interested"),
    ("That's interesting. Anyway, moving on", "negative", "Dismissive interesting"),
    ("Hmm that's interesting I've never thought about it that way before", "positive", "Genuinely intrigued"),
    ("That's... interesting. Sure.", "negative", "Skeptically dismissive interesting"),
    ("That's really interesting I had no idea", "positive", "Surprised interest"),

    # "Sure" variations
    ("Sure! I'd love to come to your party!", "positive", "Enthusiastic sure"),
    ("Sure. Whatever you want.", "negative", "Resigned defeated sure"),
    ("Sure thing, count me in!", "positive", "Eager agreement sure"),
    ("Sure, like I have a choice", "negative", "Sarcastic forced sure"),
    ("Sure why not sounds fun", "positive", "Casual positive sure"),

    # "Whatever" variations
    ("Whatever makes you happy, I support you", "positive", "Supportive whatever"),
    ("Whatever. I don't even care anymore.", "negative", "Apathetic whatever"),
    ("Haha whatever you say boss", "positive", "Playful whatever"),
    ("Whatever just do what you want you always do anyway", "negative", "Resentful whatever"),
    ("Ah whatever it's all good", "positive", "Easygoing whatever"),

    # "Right" variations
    ("Right, that makes total sense now", "positive", "Agreement right"),
    ("Right. Because you're always right aren't you", "negative", "Sarcastic right"),
    ("Right on that's exactly what we needed", "positive", "Affirming right"),
    ("Oh right I forgot you know everything", "negative", "Mocking right"),

    # "Great" variations
    ("That's great news I'm so happy for you!", "positive", "Genuinely great"),
    ("Oh great another problem to deal with", "negative", "Sarcastic great"),
    ("Great job on the presentation everyone loved it", "positive", "Praise great"),
    ("Great so now what are we supposed to do", "negative", "Frustrated sarcastic great"),
    ("That's really great to hear I was worried", "positive", "Relieved great"),

    # "Thanks" variations
    ("Thanks so much you really saved the day", "positive", "Genuine heartfelt thanks"),
    ("Thanks for nothing", "negative", "Sarcastic thanks"),
    ("Thank you from the bottom of my heart", "positive", "Deep genuine thanks"),
    ("Yeah thanks for that real helpful", "negative", "Dripping sarcasm thanks"),
    ("Thanks I really needed to hear that today", "positive", "Grateful thanks"),

    # "Nice" variations
    ("That's really nice of you to offer", "positive", "Genuine nice appreciation"),
    ("Nice going genius you broke it", "negative", "Sarcastic nice mocking"),
    ("What a nice surprise seeing you here!", "positive", "Genuine pleasant nice"),
    ("Oh nice so you just decided without asking me", "negative", "Passive aggressive nice"),
    ("That's a really nice way to put it", "positive", "Appreciating phrasing"),

    # "Okay" variations
    ("Okay that sounds like a solid plan", "positive", "Agreeable okay"),
    ("Okay fine if that's what you want", "negative", "Defeated okay"),
    ("Okay cool let's do it!", "positive", "Enthusiastic okay"),
    ("Okay. I see how it is.", "negative", "Hurt realization okay"),
    ("Okay awesome I'll be there at seven", "positive", "Confirming okay"),
]


# ─── LONG PARAGRAPHS (~42 cases) ───

PARAGRAPHS = [
    # Starts positive, ends negative
    (
        "The morning started beautifully with sunshine and coffee on the porch. "
        "I was feeling optimistic about the day ahead. Then I got the call. "
        "My position has been eliminated. Just like that, fifteen years of dedication "
        "reduced to a phone call and a severance package.",
        "negative",
        "Positive morning turns to job loss"
    ),
    (
        "We had the most amazing first date. The conversation flowed naturally, "
        "we laughed until our sides hurt, and I genuinely thought I found someone special. "
        "Then I saw them on the app the very next day, already chatting up someone new. "
        "I feel like such a fool for getting my hopes up.",
        "negative",
        "Great date ends in disappointment"
    ),
    (
        "I walked into the party feeling confident for the first time in years. "
        "New outfit, new haircut, feeling like a million bucks. Then I overheard "
        "a group of coworkers laughing about me behind my back. The confidence "
        "evaporated instantly. I went home and cried.",
        "negative",
        "Confidence shattered by gossip"
    ),
    (
        "The vacation started perfectly. Crystal clear water, white sand beaches, "
        "not a care in the world. On the third day my wallet was stolen with everything "
        "in it. The rest of the trip was spent in a police station and on the phone "
        "with my bank. What a waste.",
        "negative",
        "Vacation ruined by theft"
    ),
    (
        "I was so excited when the package finally arrived after weeks of waiting. "
        "Tore it open like a kid on Christmas morning. Inside was completely the wrong "
        "item and now customer service says they can't help me for two weeks. "
        "I'm so done with online shopping.",
        "negative",
        "Excited unboxing turns to frustration"
    ),

    # Starts negative, ends positive (recovery)
    (
        "I spent the first half of the year in the darkest depression of my life. "
        "Couldn't get out of bed, couldn't eat, couldn't see any point in continuing. "
        "But I reached out for help. Therapy, medication, and the support of people who "
        "actually cared pulled me through. Today I woke up and felt grateful to be alive.",
        "positive",
        "Depression to recovery arc"
    ),
    (
        "When the company went bankrupt I lost everything. My savings, my retirement, "
        "my sense of security. I was devastated and didn't know how I'd survive. "
        "But it forced me to start my own business and three years later I'm more "
        "successful and fulfilled than I ever was working for someone else.",
        "positive",
        "Financial ruin to entrepreneurial success"
    ),
    (
        "The divorce was the worst thing I've ever been through. Lost my home, "
        "shared custody of the kids, spent nights alone in a studio apartment crying. "
        "But slowly I rebuilt. New hobbies, new friends, and eventually a new partner "
        "who loves me the way I deserve. I'm happier now than I was in 20 years of marriage.",
        "positive",
        "Divorce to renewed happiness"
    ),
    (
        "Failing out of college felt like the end of the world. My parents were disappointed, "
        "my friends were moving forward, and I was stuck. But I found a trade school, "
        "discovered I love working with my hands, and now I make more money than most of "
        "my college friends. Best failure of my life.",
        "positive",
        "Academic failure to trade success"
    ),
    (
        "I was bullied relentlessly throughout middle school and high school. "
        "It destroyed my self-esteem and I carried that pain for years. "
        "But going to college in a new city gave me a fresh start and I found people "
        "who actually appreciate who I am. I wouldn't trade my life now for anything.",
        "positive",
        "Bullying past to college rebirth"
    ),

    # Sustained negative (venting)
    (
        "I'm so sick of this job. The hours are insane, the pay is garbage, "
        "my boss takes credit for everything I do, and my coworkers are toxic. "
        "I come home exhausted every single day with nothing left for myself "
        "or my family. Something has to change but I can't afford to quit.",
        "negative",
        "Extended job venting"
    ),
    (
        "The healthcare system in this country is an absolute disgrace. I've been "
        "waiting three months for an appointment, my insurance denied the claim twice, "
        "and now they want me to pay out of pocket for something that should be covered. "
        "Meanwhile I'm still in pain and nobody seems to care.",
        "negative",
        "Healthcare system frustration"
    ),
    (
        "My landlord raised the rent again for the third time this year. "
        "The apartment has mold in the bathroom, the heating barely works, "
        "and they still haven't fixed the leak I reported six months ago. "
        "But where else am I going to go when every other place costs even more? "
        "I feel completely trapped.",
        "negative",
        "Housing frustration vent"
    ),
    (
        "Every single relationship I've been in has ended the same way. "
        "They get close, see who I really am, and leave. I'm starting to think "
        "the common factor in all my failed relationships is me. Maybe I'm just "
        "not built for being loved. It's a lonely realization.",
        "negative",
        "Relationship pattern despair"
    ),
    (
        "I've been applying for jobs for eight months now. Over 200 applications, "
        "maybe 10 interviews, zero offers. My savings are almost gone, my confidence "
        "is shattered, and every rejection letter feels like a personal attack. "
        "I don't know how much longer I can do this.",
        "negative",
        "Job search desperation"
    ),

    # Sustained positive (celebrating)
    (
        "Today was absolutely perfect. Woke up to breakfast in bed that my partner made, "
        "got a call that my book is being published, had lunch with my best friend, "
        "and came home to my dog who was thrilled to see me as always. "
        "Life is genuinely good right now and I want to remember this feeling forever.",
        "positive",
        "Perfect day celebration"
    ),
    (
        "I can't believe how much has changed in a year. Twelve months ago I was broke, "
        "single, and miserable. Now I have a job I love, an apartment that feels like home, "
        "amazing friends, and I'm actually excited about the future for the first time "
        "in forever. Hard work really does pay off.",
        "positive",
        "Year of transformation celebration"
    ),
    (
        "Our wedding was everything we dreamed of and more. The weather was perfect, "
        "everyone we love was there, the food was incredible, and when we danced our "
        "first dance I swear the whole world disappeared. I married my best friend "
        "and I've never been happier.",
        "positive",
        "Wedding day celebration"
    ),
    (
        "Just got back from the most incredible trip of my life. Hiked through mountains, "
        "swam in crystal clear lakes, met the kindest strangers, ate food that changed my life. "
        "Every single day was an adventure. I came back a different person, "
        "more grateful and more alive than ever.",
        "positive",
        "Trip of a lifetime celebration"
    ),
    (
        "My daughter just performed her first piano recital and she was absolutely magnificent. "
        "She was nervous but she pushed through and played flawlessly. The look of pride "
        "on her face when the audience clapped brought tears to my eyes. "
        "Being a parent is the greatest gift I've ever received.",
        "positive",
        "Parental pride celebration"
    ),

    # Emotional rollercoaster (multiple reversals)
    (
        "I thought I got the job, then they said no, then they called back and said "
        "the other candidate fell through and I'm in after all. I went from elated "
        "to crushed to cautiously optimistic in the span of 48 hours. I'm exhausted.",
        "mixed",
        "Job offer rollercoaster"
    ),
    (
        "My doctor said the tumor might be malignant and I spent two weeks convinced "
        "I was dying. The biopsy came back benign and I cried with relief. But the scare "
        "made me realize how much time I've been wasting and now I feel both grateful "
        "and regretful at the same time.",
        "mixed",
        "Health scare to relief to reflection"
    ),
    (
        "We broke up, got back together, broke up again, and now apparently we're "
        "giving it one more shot. My friends think I'm crazy and maybe I am but "
        "when it's good between us it's the best thing in the world and when it's bad "
        "it's absolute hell. I don't know what to feel anymore.",
        "mixed",
        "On-again off-again relationship"
    ),
    (
        "Today I found out I'm pregnant. I'm overjoyed because we've been trying "
        "for years. But I'm also terrified because of the miscarriages. And I'm angry "
        "at myself for being scared instead of just happy. Why can't I just be happy?",
        "mixed",
        "Pregnancy joy + fear after loss"
    ),
    (
        "The family reunion was chaos. My sister and I made up after three years of silence "
        "which was beautiful. But then my uncle started a political argument that ended "
        "with my mom crying. We took a family photo where half the people had been "
        "screaming at each other five minutes earlier. Classic family.",
        "mixed",
        "Family reunion chaos"
    ),
    (
        "Started the morning by getting a raise. Celebrated at lunch with friends. "
        "Came home to find my apartment flooded from a burst pipe. Spent the evening "
        "on the phone with insurance while sitting in a puddle. At least I can afford "
        "the deductible now I guess.",
        "mixed",
        "Raise celebration to flood disaster"
    ),
    (
        "Found out my ex is getting married. Happy for them honestly. Or at least "
        "I'm trying to be. The rational part of me knows we weren't right for each other "
        "but there's still this stupid little part of me that wonders what if. "
        "I hate that I care. I'm going to go eat ice cream and feel complicated feelings.",
        "mixed",
        "Ex getting married complex feelings"
    ),

    # Philosophical and reflective
    (
        "As I sit here on my 40th birthday I realize I've become someone my 20-year-old "
        "self wouldn't recognize. Some of that is good, growth and wisdom. Some of it "
        "is disappointing, dreams I let die and compromises I told myself were temporary. "
        "But here I am, still going, still trying. That has to count for something.",
        "mixed",
        "40th birthday reflection"
    ),
    (
        "Life doesn't owe us happiness and that's actually kind of freeing. Once you accept "
        "that the good moments are gifts and not guarantees, you start appreciating them "
        "differently. The pain doesn't go away but it becomes part of the texture "
        "of a life fully lived.",
        "positive",
        "Philosophical acceptance"
    ),
    (
        "I look at my parents getting older and it terrifies me. They were invincible "
        "when I was a child and now they need my help with things they used to do effortlessly. "
        "It's a privilege to care for them but watching the strongest people you know "
        "become fragile is something no one prepares you for.",
        "mixed",
        "Aging parents reflection"
    ),

    # Stream of consciousness
    (
        "I cant even think straight right now everything is happening at once my car broke down "
        "my kid is sick the bills are piling up and I just found out my hours are getting cut "
        "I don't know how im gonna make it through this month honestly",
        "negative",
        "Overwhelmed stream of consciousness with typos"
    ),
    (
        "omg omg omg I just got THE call. The gallery wants to show MY art. MY art! "
        "In an actual gallery where actual people will see it and maybe buy it and oh god "
        "what if nobody likes it no stop that they chose ME because I'm good enough "
        "I am good enough okay I'm freaking out in the best way",
        "positive",
        "Excited stream of consciousness"
    ),

    # Very long complex paragraph
    (
        "My therapist asked me to write about how I'm feeling and honestly I don't even know "
        "where to start. Some days I feel like I'm making progress and other days it's like "
        "I'm right back at square one. The medication helps with the acute anxiety but "
        "it makes me feel flat, like the volume on my emotions has been turned down. "
        "I miss feeling things intensely even if some of those feelings were painful. "
        "Is that weird? To miss your own suffering? I think what I really miss is feeling alive.",
        "mixed",
        "Therapy journal entry reflection"
    ),

    # Short punchy paragraph
    (
        "Got fired. Car got towed. Came home. Apartment robbed. Happy Tuesday.",
        "negative",
        "Staccato disaster list"
    ),

    (
        "Woke up smiling. Sun is shining. Coffee is perfect. Nothing on the schedule. "
        "Sometimes the best days are the ones where absolutely nothing happens.",
        "positive",
        "Perfect simple day"
    ),

    # Narrative arc
    (
        "Three years ago I was sleeping in my car. I had lost my job, my apartment, and most of "
        "my dignity. I showered at the gym and ate from food banks. The shame was overwhelming. "
        "But a stranger at one of those food banks told me about a job training program. "
        "I enrolled. I showed up every single day. I got certified, then hired, then promoted. "
        "Today I closed on my first house. I sat in the driveway and sobbed. "
        "Not from sadness but from pure overwhelming gratitude.",
        "positive",
        "Rags to riches emotional narrative"
    ),

    (
        "We tried for five years to have a baby. Every failed cycle was devastating. "
        "IVF nearly bankrupted us and destroyed our marriage. We fought constantly and "
        "almost gave up on everything, each other included. Then one random Tuesday morning "
        "the test came back positive. We held each other and cried for an hour straight. "
        "She's three months old now and I still sometimes cry looking at her, "
        "remembering the darkness we walked through to get here.",
        "positive",
        "Fertility struggle to parenthood"
    ),
]


# ─── REAL-WORLD SCENARIOS (~52 cases) ───

REAL_WORLD = [
    # Job loss
    ("I just got laid off after 10 years with the company", "negative", "Long-term job loss"),
    ("They eliminated my entire department without warning", "negative", "Department elimination"),
    ("I got fired today and I don't know what to tell my family", "negative", "Fired with family pressure"),
    ("The plant is closing and 500 people are losing their jobs including me", "negative", "Mass layoff"),
    ("I just got let go two weeks before Christmas", "negative", "Holiday season layoff"),

    # Relationships
    ("She said she needs space and I know that means it's over", "negative", "Breakup via space request"),
    ("I found out they've been cheating on me for months", "negative", "Cheating discovery"),
    ("We had the talk and we're officially over", "negative", "Mutual breakup finalization"),
    ("He left without saying goodbye just a note on the counter", "negative", "Abandonment"),
    ("I think my marriage is falling apart and I don't know how to fix it", "negative", "Marriage deterioration"),
    ("My partner proposed and I said yes!", "positive", "Proposal acceptance"),

    # Health
    ("The doctor said the tests came back and it's cancer", "negative", "Cancer diagnosis"),
    ("They found something on the scan and want me to come back", "negative", "Worrying medical callback"),
    ("I just got diagnosed with diabetes and I'm scared", "negative", "Chronic illness diagnosis"),
    ("The surgery was successful and the doctors are optimistic", "positive", "Successful surgery"),
    ("I've been in chronic pain for months and nobody can figure out why", "negative", "Undiagnosed chronic pain"),
    ("I just completed my last round of chemo and I'm officially in remission", "positive", "Cancer remission"),

    # Family
    ("My parents are getting divorced after 30 years", "negative", "Parents divorcing"),
    ("My dad hasn't spoken to me in six months", "negative", "Parental estrangement"),
    ("My sister and I aren't speaking anymore after the inheritance fight", "negative", "Family estrangement"),
    ("I found out I was adopted through a DNA test", "negative", "Adoption discovery shock"),
    ("My family surprised me by all flying in for my birthday", "positive", "Family surprise gathering"),

    # Financial
    ("I can't make rent this month and my landlord won't budge", "negative", "Rent crisis"),
    ("My credit card debt is completely out of control", "negative", "Debt crisis"),
    ("I just looked at my student loans and I want to throw up", "negative", "Student debt overwhelm"),
    ("I lost my entire savings in a scam", "negative", "Financial fraud victim"),
    ("I finally paid off all my debt after seven years!", "positive", "Debt freedom"),
    ("We just found out we owe 15 thousand in back taxes", "negative", "Tax debt shock"),

    # Academic
    ("I failed my finals and I might not graduate", "negative", "Finals failure"),
    ("I got rejected from every grad school I applied to", "negative", "Grad school rejection"),
    ("I'm struggling so much in this program and I feel like I don't belong", "negative", "Imposter syndrome academic"),
    ("I just defended my thesis and the committee loved it!", "positive", "Thesis defense success"),
    ("I got a full scholarship!", "positive", "Scholarship award"),
    ("I have three papers due this week and I haven't started any of them", "negative", "Academic overwhelm"),

    # Social
    ("Nobody came to my birthday party", "negative", "Empty party devastation"),
    ("I realized I don't have a single person I can call when things go wrong", "negative", "No support system"),
    ("I moved to a new city and I don't know a single person here", "negative", "Isolation after moving"),
    ("All my friends are getting married and having kids and I'm still alone", "negative", "Social comparison loneliness"),
    ("I just had the best night out with friends I've had in years", "positive", "Great social night"),

    # Parenting
    ("My kid said they hate me and slammed the door", "negative", "Child anger at parent"),
    ("My teenager is being bullied at school and there's nothing I can do", "negative", "Child bullying helplessness"),
    ("I yelled at my kid today and the look on their face destroyed me", "negative", "Parental guilt"),
    ("My child just took their first steps!", "positive", "First steps milestone"),
    ("My son came out to me today and I want him to know I love and support him completely", "positive", "Child coming out acceptance"),

    # Pet loss and joy
    ("We had to put our dog down today he was my best friend for 14 years", "negative", "Pet euthanasia grief"),
    ("My cat has been diagnosed with kidney failure", "negative", "Pet illness diagnosis"),
    ("We just adopted the cutest puppy and I'm obsessed", "positive", "Puppy adoption joy"),
    ("The vet said my dog's surgery went perfectly and she's going to be okay", "positive", "Pet recovery relief"),

    # Major life events
    ("I just defended my thesis successfully!", "positive", "Thesis success exclamation"),
    ("I found out I'm going to be a dad and I'm terrified and overjoyed", "mixed", "Impending fatherhood"),
    ("We just got the keys to our first house", "positive", "First home purchase"),
    ("I'm being deployed overseas for the third time", "negative", "Military deployment"),
    ("I just became a citizen of this country after a 12 year process", "positive", "Citizenship achievement"),
]


# ─── PROFANITY (~35 cases) ───

PROFANITY = [
    # Negative profanity
    ("What the fuck is wrong with you", "negative", "Angry profanity at someone"),
    ("This shit is absolutely unacceptable", "negative", "Frustrated profanity"),
    ("Go to hell I never want to see you again", "negative", "Hostile dismissal"),
    ("That was the biggest load of bullshit I've ever heard", "negative", "Calling out BS"),
    ("I don't give a damn about your excuses", "negative", "Dismissive anger profanity"),
    ("Fuck this job and fuck this company", "negative", "Job rage profanity"),
    ("You're a complete piece of shit for doing that", "negative", "Personal attack profanity"),
    ("I'm so fucking tired of being treated like garbage", "negative", "Exhausted profanity"),
    ("Are you fucking kidding me right now", "negative", "Exasperated profanity"),
    ("What a goddamn disaster this turned out to be", "negative", "Disaster profanity"),
    ("I'm sick of this bullshit", "negative", "Fed up profanity"),
    ("Screw everyone who doubted me", "negative", "Resentful profanity"),

    # Positive profanity
    ("Holy shit that's amazing!", "positive", "Positive shock profanity"),
    ("We fucking did it!", "positive", "Victory profanity"),
    ("That was the best damn meal I've ever had", "positive", "Food appreciation profanity"),
    ("Hell yes I got the job!", "positive", "Celebration profanity"),
    ("No fucking way that actually worked!", "positive", "Disbelief positive profanity"),
    ("This shit slaps honestly", "positive", "Slang positive profanity"),
    ("Holy crap that sunset is gorgeous", "positive", "Mild profanity wonder"),
    ("I'm so damn proud of you", "positive", "Proud profanity"),
    ("That concert was fucking incredible", "positive", "Event appreciation profanity"),
    ("Damn right we deserved that win", "positive", "Righteous victory profanity"),
    ("Fuck yeah let's gooooo", "positive", "Hype profanity"),
    ("That plot twist was absolutely batshit insane in the best way", "positive", "Entertainment profanity"),

    # Frustrated profanity
    ("I can't fucking believe this happened again", "negative", "Repetition frustration profanity"),
    ("For fuck's sake can anything go right today", "negative", "Bad day profanity"),
    ("Why the hell didn't anyone tell me about this sooner", "negative", "Frustrated ignorance profanity"),
    ("This goddamn computer is driving me insane", "negative", "Tech frustration profanity"),
    ("Oh for crying out loud not this again", "negative", "Mild frustration exclamation"),
    ("Son of a bitch the pipe burst again", "negative", "Home disaster profanity"),

    # Celebratory profanity
    ("We fucking made it to the finals!", "positive", "Sports celebration profanity"),
    ("I just got my damn degree after six years!", "positive", "Degree celebration profanity"),
    ("No shit she actually said yes!", "positive", "Proposal celebration profanity"),
    ("Hot damn we closed the deal!", "positive", "Business win profanity"),
    ("Well I'll be damned it actually worked", "positive", "Surprised success profanity"),
]


# ─── AGGREGATE ALL TESTS ───

ALL_TESTS = (
    BASIC_POSITIVE
    + BASIC_NEGATIVE
    + SARCASM
    + CRISIS
    + MIXED
    + GRANULARITY
    + CONTEXT
    + PARAGRAPHS
    + REAL_WORLD
    + PROFANITY
)


# Quick stats when run directly
if __name__ == "__main__":
    categories = {
        "BASIC_POSITIVE": BASIC_POSITIVE,
        "BASIC_NEGATIVE": BASIC_NEGATIVE,
        "SARCASM": SARCASM,
        "CRISIS": CRISIS,
        "MIXED": MIXED,
        "GRANULARITY": GRANULARITY,
        "CONTEXT": CONTEXT,
        "PARAGRAPHS": PARAGRAPHS,
        "REAL_WORLD": REAL_WORLD,
        "PROFANITY": PROFANITY,
    }

    print("Test Suite Statistics")
    print("=" * 40)
    total = 0
    for name, cases in categories.items():
        print(f"  {name:<20} {len(cases):>4} cases")
        total += len(cases)
    print(f"  {'─' * 30}")
    print(f"  {'TOTAL':<20} {total:>4} cases")
    print(f"\n  ALL_TESTS length:   {len(ALL_TESTS)}")
    assert len(ALL_TESTS) == total, f"Mismatch: ALL_TESTS={len(ALL_TESTS)}, sum={total}"
    assert total >= 500, f"Need 500+ cases, only have {total}"
    print(f"\n  All checks passed!")
