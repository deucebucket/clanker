#!/usr/bin/env python3
"""Crisis Recall Benchmark — targeting 99.9% recall on crisis detection.

Tests PendulumV2's ability to detect crisis sentences (suicidal ideation,
self-harm, extreme despair) while avoiding false positives on negative,
neutral, positive, and tricky sentences.

Crisis classification rule: V < 50 OR (V < 70 AND U > 60)

Usage:
    python3 benchmarks/crisis_benchmark.py
    python3 benchmarks/crisis_benchmark.py --verbose   # show all misclassifications
"""

import sys
import os
import json
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from demo.pendulum_v2 import PendulumV2

# ---------------------------------------------------------------------------
# Champion config from gpu_optimal_v2.json
# ---------------------------------------------------------------------------

CHAMPION_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "gpu_optimal_v2.json")


def load_champion_config() -> dict:
    with open(CHAMPION_CONFIG_PATH) as f:
        data = json.load(f)
    return data["champion_config"]


# ---------------------------------------------------------------------------
# Sentence generation
# ---------------------------------------------------------------------------

def generate_crisis_sentences() -> List[str]:
    """Generate 200+ crisis sentences across multiple categories."""
    sentences = []

    # --- Direct suicidal ideation ---
    base_die = [
        "I want to die",
        "I wish I was dead",
        "I wish I were dead",
        "I want to be dead",
        "I just want to die",
        "I really want to die",
        "I think I want to die",
        "sometimes I want to die",
        "all I want is to die",
        "I honestly wish I was dead",
    ]
    sentences.extend(base_die)

    # Subject variations
    for subj in ["he wants", "she wants", "my friend wants", "my son wants",
                 "my daughter wants", "my brother wants", "my sister wants"]:
        sentences.append(f"{subj} to die")

    # --- "End it" phrasing ---
    end_it = [
        "I'm going to end it",
        "I'm going to end it all",
        "I want to end it all",
        "I'm going to end my life",
        "I want to end my life",
        "I've decided to end it",
        "tonight I'm ending it",
        "I'm done, I'm ending it",
        "there's no point, I'm ending it all",
        "I just want it to be over",
        "I want this to end permanently",
    ]
    sentences.extend(end_it)

    # --- "Don't want to be here" / disappearance ---
    disappear = [
        "I don't want to be here anymore",
        "I don't want to exist",
        "I wish I didn't exist",
        "I wish I had never been born",
        "I should have never been born",
        "nobody would notice if I was gone",
        "nobody would miss me",
        "nobody would care if I disappeared",
        "the world would be better without me",
        "everyone would be better off without me",
        "my family would be better off if I was dead",
        "they'd be happier if I wasn't here",
        "I want to disappear forever",
        "I just want to vanish",
        "I wish I could just cease to exist",
        "I don't belong in this world",
        "there's no place for me in this world",
        "I have no reason to be alive",
        "I shouldn't be alive",
        "I don't deserve to be alive",
    ]
    sentences.extend(disappear)

    # --- "Can't take it" / breaking point ---
    breaking = [
        "I can't take this anymore",
        "I can't do this anymore",
        "I can't go on",
        "I can't go on like this",
        "I can't keep going",
        "I can't keep doing this",
        "I can't live like this",
        "I can't live with this pain",
        "I can't bear this anymore",
        "I can't stand being alive",
        "I've had enough of living",
        "I've reached my breaking point",
        "I'm at the end of my rope",
        "there's nothing left for me",
        "I have nothing left",
        "I have nothing to live for",
        "there is nothing to live for",
        "life isn't worth living",
        "life is not worth living anymore",
        "what's the point of living",
        "what is the point of being alive",
    ]
    sentences.extend(breaking)

    # --- Self-harm ---
    self_harm = [
        "I want to hurt myself",
        "I'm going to hurt myself",
        "I want to cut myself",
        "I've been cutting myself",
        "I started cutting again",
        "I want to make myself bleed",
        "I want to feel the pain",
        "I deserve to be hurt",
        "I need to punish myself",
        "I burned myself on purpose",
        "I hit myself until I bruised",
        "I scratch myself until I bleed",
        "the only thing that helps is hurting myself",
    ]
    sentences.extend(self_harm)

    # --- Hopelessness / extreme despair ---
    despair = [
        "everything is hopeless",
        "there is no hope",
        "there is no hope for me",
        "nothing will ever get better",
        "it will never get better",
        "things will never improve",
        "I am completely hopeless",
        "I feel totally hopeless",
        "hope is gone",
        "all hope is lost",
        "I see no way out",
        "there is no way out",
        "I'm trapped and there's no escape",
        "I will never be happy again",
        "happiness is impossible for me",
        "I am beyond help",
        "nothing can help me",
        "nobody can help me",
        "no one can save me",
        "it's too late for me",
        "I'm already dead inside",
        "I feel dead inside",
        "I'm just waiting to die",
        "I'm counting the days until I die",
        "every day is agony",
        "every day is torture",
        "I wake up wishing I hadn't",
        "I dread waking up every morning",
        "I go to sleep hoping I won't wake up",
        "I pray I don't wake up tomorrow",
    ]
    sentences.extend(despair)

    # --- Plans / methods (oblique) ---
    plans = [
        "I've been thinking about how to do it",
        "I've made a plan",
        "I wrote a goodbye letter",
        "I wrote my note",
        "I've been giving away my things",
        "I said goodbye to everyone",
        "I put my affairs in order",
        "I know exactly how I'll do it",
        "I've been researching methods",
        "I've picked a date",
        "I bought what I need",
        "everything is ready",
        "I just need to find the courage",
        "I'm not afraid of dying anymore",
        "death doesn't scare me anymore",
    ]
    sentences.extend(plans)

    # --- Tense variations on core phrases ---
    tense_variations = [
        "I wanted to die yesterday",
        "I've been wanting to die for months",
        "I've wanted to die since I was a kid",
        "I will want to die tomorrow too",
        "I used to want to die every day",
        "I woke up wanting to die",
        "I fell asleep wishing I was dead",
        "I've been thinking about killing myself",
        "I think about killing myself every day",
        "I thought about killing myself last night",
        "I can't stop thinking about death",
        "I can't stop thinking about dying",
        "I keep imagining my own death",
    ]
    sentences.extend(tense_variations)

    # --- Hedged / softened crisis ---
    hedged = [
        "I sort of want to die",
        "I kind of want to die",
        "sometimes I think about ending it",
        "part of me wants to die",
        "a big part of me wants to give up on life",
        "maybe I should just end it",
        "I guess I wouldn't mind dying",
        "I wouldn't care if I died",
        "I wouldn't mind not waking up",
        "if I died tomorrow I'd be okay with it",
        "honestly dying doesn't sound so bad",
        "death would be a relief",
        "death sounds peaceful",
        "I'm tired of being alive",
        "I'm so tired of existing",
        "I'm exhausted from being alive",
        "living is exhausting",
        "being alive is exhausting",
    ]
    sentences.extend(hedged)

    # --- Burden statements ---
    burden = [
        "I am a burden to everyone",
        "I'm just a burden",
        "I'm a burden on my family",
        "everyone's life would be easier without me",
        "I just make things worse for everyone",
        "I ruin everything I touch",
        "I only cause pain",
        "all I do is hurt people",
        "my existence hurts others",
        "I'm poison to everyone around me",
    ]
    sentences.extend(burden)

    # --- Isolation + despair ---
    isolation = [
        "I am completely alone",
        "no one cares about me",
        "nobody loves me",
        "nobody will ever love me",
        "I will die alone",
        "I am utterly worthless",
        "I am worthless and useless",
        "I am nothing",
        "I have no value",
        "I am garbage",
        "I am trash",
        "I hate myself so much I want to die",
        "I despise myself",
        "I can't forgive myself for existing",
    ]
    sentences.extend(isolation)

    return sentences


def generate_non_crisis_negative() -> List[str]:
    """Generate 200+ negative but NOT crisis sentences."""
    sentences = [
        "I am sad",
        "I'm feeling sad today",
        "I feel down",
        "I'm a bit down today",
        "I had a bad day",
        "today was a terrible day",
        "this sucks",
        "that really sucks",
        "I'm disappointed",
        "I'm really disappointed",
        "I feel let down",
        "I'm frustrated",
        "I'm so frustrated right now",
        "this is frustrating",
        "I'm annoyed",
        "I'm irritated",
        "that annoyed me",
        "I'm angry",
        "I'm furious",
        "I'm upset",
        "I'm really upset about this",
        "I feel hurt",
        "my feelings are hurt",
        "I'm heartbroken",
        "I feel rejected",
        "I feel excluded",
        "I feel left out",
        "I'm worried",
        "I'm anxious",
        "I'm stressed",
        "I'm overwhelmed",
        "I'm exhausted",
        "I'm tired",
        "I'm burned out",
        "I feel drained",
        "I feel empty",
        "I feel numb",
        "I feel lost",
        "I feel confused",
        "I don't know what to do",
        "I feel stuck",
        "I feel trapped",
        "I'm bored",
        "I'm lonely",
        "I miss my friend",
        "I miss my family",
        "I miss home",
        "I'm homesick",
        "I feel guilty",
        "I feel ashamed",
        "I feel embarrassed",
        "I regret what I said",
        "I regret that decision",
        "I made a mistake",
        "I messed up",
        "I screwed up",
        "I failed the test",
        "I failed the exam",
        "I didn't get the job",
        "I got rejected from college",
        "I lost my job",
        "I got fired",
        "I got laid off",
        "my boss yelled at me",
        "my teacher gave me a bad grade",
        "I got a parking ticket",
        "my car broke down",
        "my phone is broken",
        "I lost my wallet",
        "I lost my keys",
        "the food was terrible",
        "the movie was bad",
        "the weather is awful",
        "it's raining again",
        "traffic was terrible",
        "I'm stuck in traffic",
        "the service was horrible",
        "the restaurant was disappointing",
        "I don't like this",
        "I hate this song",
        "I hate Mondays",
        "Mondays are the worst",
        "I can't stand this noise",
        "the meeting was boring",
        "I don't like my new haircut",
        "I gained weight",
        "my back hurts",
        "I have a headache",
        "I feel sick",
        "I have a cold",
        "I sprained my ankle",
        "my allergies are terrible",
        "I couldn't sleep last night",
        "I had insomnia again",
        "I had a nightmare",
        "the news is depressing",
        "that story was so sad",
        "I cried during the movie",
        "I feel melancholy",
        "I feel gloomy",
        "I feel blue",
        "I'm in a bad mood",
        "I'm grumpy",
        "I'm cranky",
        "I'm not having a good day",
        "today is not my day",
        "nothing is going right today",
        "everything went wrong today",
        "I feel unappreciated",
        "nobody listens to me",
        "I feel invisible",
        "I feel undervalued",
        "I wish things were different",
        "I'm not happy with my life",
        "I'm dissatisfied with my job",
        "I don't enjoy my work anymore",
        "my relationship is struggling",
        "we had a fight",
        "we argued again",
        "I'm going through a rough patch",
        "times are tough",
        "things are hard right now",
        "life is hard sometimes",
        "adulting is hard",
        "I feel out of place",
        "I don't fit in",
        "I feel different from everyone else",
        "I feel like an outsider",
        "I'm not good enough",
        "I'm not smart enough",
        "I wish I was better at this",
        "I feel inadequate",
        "I feel insecure",
        "I'm nervous about tomorrow",
        "I'm dreading Monday",
        "I don't want to go to work",
        "I'm not looking forward to this",
        "I feel unmotivated",
        "I lack motivation",
        "I can't focus",
        "I can't concentrate",
        "I procrastinated again",
        "I feel restless",
        "I feel uneasy",
        "something feels off",
        "I have a bad feeling",
        "I'm skeptical about this",
        "I'm pessimistic",
        "I don't think this will work",
        "this is a waste of time",
        "I don't see the point",
        "what's the point of this meeting",
        "this project is going nowhere",
        "I'm spinning my wheels",
        "I feel stagnant",
        "I'm going nowhere in my career",
        "I feel underpaid",
        "rent is too expensive",
        "I can't afford this",
        "money is tight",
        "I'm broke",
        "I'm in debt",
        "bills are piling up",
        "I'm behind on payments",
        "the economy is bad",
        "prices keep going up",
        "inflation is killing me",
        "gas prices are ridiculous",
        "my team lost the game",
        "we lost again",
        "that was a terrible play",
        "the ref made a bad call",
        "I'm not feeling well",
        "I'm under the weather",
        "I feel lousy",
        "I feel crappy",
        "this traffic is unbearable",
        "I hate waiting in line",
        "customer service was useless",
        "they put me on hold for an hour",
        "my package is lost",
        "they cancelled my flight",
        "my vacation got ruined",
        "it rained the whole trip",
        "the hotel was disgusting",
        "I'm jet lagged",
        "I miss my dog",
        "my pet is sick",
        "I feel helpless watching them suffer",
        "grief is overwhelming",
        "I miss the person I used to be",
        "I feel old",
        "getting older is scary",
        "I wasted so much time",
        "I regret not trying harder",
        "I should have done things differently",
        "hindsight is painful",
        "that was humiliating",
        "I felt so small",
        "they laughed at me",
        "I was bullied again",
        "people are cruel",
        "the world is unfair",
        "life isn't fair",
        "I feel powerless",
        "I can't change anything",
        "nothing I do matters",
    ]
    return sentences


def generate_neutral() -> List[str]:
    """Generate 200+ neutral sentences."""
    sentences = [
        "the weather is nice",
        "it's sunny outside",
        "it's raining",
        "it might snow tomorrow",
        "the temperature is 72 degrees",
        "I went to the store",
        "I bought groceries",
        "I went shopping",
        "I drove to work",
        "I took the bus",
        "I walked to school",
        "I had lunch at noon",
        "I ate a sandwich",
        "I drank some coffee",
        "I had cereal for breakfast",
        "pass the salt",
        "pass the butter",
        "close the door please",
        "turn off the lights",
        "can you hand me that",
        "the meeting is at three",
        "the report is due Friday",
        "the project deadline is next week",
        "we have a meeting tomorrow",
        "the conference call is at two",
        "I need to send that email",
        "I updated the spreadsheet",
        "the file has been uploaded",
        "the document is ready",
        "the presentation is prepared",
        "the book is on the table",
        "the car is in the garage",
        "the keys are on the counter",
        "the phone is charging",
        "the computer needs an update",
        "the printer is out of ink",
        "the wifi password is on the fridge",
        "the store closes at nine",
        "the bank opens at eight",
        "the library is open until five",
        "the train leaves at six",
        "the flight departs at noon",
        "the hotel is downtown",
        "the restaurant is on Main Street",
        "the park is two blocks away",
        "I read a book",
        "I watched television",
        "I listened to a podcast",
        "I went for a walk",
        "I did some gardening",
        "I cleaned the house",
        "I did the laundry",
        "I cooked dinner",
        "I washed the dishes",
        "I fed the cat",
        "I took the dog out",
        "I watered the plants",
        "I checked my email",
        "I made a phone call",
        "I scheduled an appointment",
        "the appointment is next Tuesday",
        "I need to pick up my prescription",
        "the oil change is overdue",
        "I need to renew my license",
        "I filed my taxes",
        "the package arrived",
        "the delivery is expected Thursday",
        "I returned the item",
        "I signed the form",
        "the contract has been reviewed",
        "the lease expires in June",
        "I moved to a new apartment",
        "the furniture was delivered",
        "I assembled the bookshelf",
        "the wall needs painting",
        "I fixed the leaky faucet",
        "the plumber is coming Tuesday",
        "the electrician will be here tomorrow",
        "I called the landlord",
        "the garbage truck comes on Monday",
        "recycling pickup is Wednesday",
        "I sorted the recycling",
        "the lawn needs mowing",
        "I raked the leaves",
        "I shoveled the driveway",
        "the snow plow came by",
        "school starts in September",
        "class begins at nine",
        "homework is due tomorrow",
        "the exam is on Friday",
        "I enrolled in a course",
        "I registered for the class",
        "the semester ends in December",
        "graduation is in May",
        "I applied to college",
        "I submitted my application",
        "the interview is next week",
        "I updated my resume",
        "I sent my cover letter",
        "the job posting closes Friday",
        "I accepted the offer",
        "my start date is Monday",
        "orientation is next week",
        "the office is on the third floor",
        "my desk is by the window",
        "the break room is down the hall",
        "there are donuts in the kitchen",
        "the vending machine is broken",
        "I need to replace the batteries",
        "the clock is wrong",
        "daylight saving time is this weekend",
        "the calendar needs updating",
        "I set a reminder",
        "the alarm goes off at seven",
        "I usually wake up at six",
        "I go to bed around ten",
        "the sun sets at seven thirty",
        "the sunrise was at six fifteen",
        "the moon is full tonight",
        "it's a clear night",
        "the stars are out",
        "it's getting dark earlier",
        "autumn starts next month",
        "the leaves are changing color",
        "spring is around the corner",
        "summer is almost over",
        "winter is coming",
        "the population is growing",
        "the city has three million people",
        "the river is two miles wide",
        "the mountain is fourteen thousand feet",
        "the lake froze over",
        "the bridge is under construction",
        "the highway was repaved",
        "the speed limit is sixty five",
        "I changed lanes",
        "I took the exit",
        "the GPS says turn right",
        "the map shows a shortcut",
        "I parked in the lot",
        "the meter expired",
        "I need gas",
        "the tank is half full",
        "I checked the tire pressure",
        "the battery is low",
        "the screen brightness is too high",
        "I adjusted the volume",
        "the channel changed",
        "I downloaded the app",
        "the update installed",
        "the software version is three point two",
        "I restarted the computer",
        "the page loaded slowly",
        "the connection timed out",
        "I cleared the cache",
        "the password needs changing",
        "I created a new account",
        "the username is taken",
        "I logged in",
        "I logged out",
        "the session expired",
        "I saved the file",
        "I printed the document",
        "the copy machine is jammed",
        "I stapled the pages together",
        "the envelope is sealed",
        "I mailed the letter",
        "the post office is closed",
        "stamps cost sixty five cents",
        "I bought a pack of gum",
        "the total was twelve dollars",
        "I paid with my card",
        "the receipt is in the bag",
        "I returned the shirt",
        "the store was crowded",
        "the line was long",
        "they were out of stock",
        "I ordered it online",
        "shipping takes three days",
        "the tracking number was emailed",
        "I signed for the package",
        "the box was heavy",
        "I recycled the packaging",
        "the instructions are inside",
        "I read the manual",
        "assembly required two hours",
        "the measurements are in metric",
        "the weight is five kilograms",
        "the distance is ten kilometers",
        "I converted the units",
        "the calculation is correct",
        "the answer is forty two",
        "the average is seventy three",
        "the total adds up",
        "the budget is approved",
        "the quarter ends in March",
        "the fiscal year starts in October",
        "the audit is scheduled for April",
        "inventory count is next week",
        "the shipment arrived on time",
        "production is on schedule",
        "the factory runs two shifts",
        "the warehouse is at capacity",
        "I filed the paperwork",
        "the records have been archived",
        "the database was backed up",
        "the server is running normally",
    ]
    return sentences


def generate_positive() -> List[str]:
    """Generate 200+ positive sentences."""
    sentences = [
        "I am happy",
        "I'm so happy right now",
        "I feel great",
        "I feel wonderful",
        "I feel amazing",
        "I feel fantastic",
        "I feel incredible",
        "I feel blessed",
        "I feel grateful",
        "I feel thankful",
        "great news",
        "wonderful news",
        "amazing news",
        "I love this",
        "I love it",
        "I love my life",
        "I love my job",
        "I love my family",
        "I love my friends",
        "I love this song",
        "this is awesome",
        "this is wonderful",
        "this is fantastic",
        "this is incredible",
        "this is beautiful",
        "this is perfect",
        "this is the best day ever",
        "best day of my life",
        "today was a great day",
        "today was a good day",
        "I had a wonderful day",
        "what a beautiful day",
        "life is good",
        "life is beautiful",
        "life is amazing",
        "I'm thriving",
        "I'm flourishing",
        "I'm doing well",
        "I'm doing great",
        "everything is going well",
        "things are looking up",
        "I'm on cloud nine",
        "I'm over the moon",
        "I'm walking on sunshine",
        "I can't stop smiling",
        "I'm beaming with joy",
        "I'm bursting with happiness",
        "I feel alive",
        "I feel energized",
        "I feel motivated",
        "I feel inspired",
        "I feel confident",
        "I feel empowered",
        "I feel strong",
        "I feel capable",
        "I feel proud of myself",
        "I'm proud of what I accomplished",
        "I did it",
        "I succeeded",
        "I passed the exam",
        "I got the job",
        "I got the promotion",
        "I got accepted",
        "I got in",
        "I graduated",
        "I finished the project",
        "I achieved my goal",
        "I reached my target",
        "I broke my personal record",
        "I'm making progress",
        "I'm getting better",
        "I'm improving",
        "I'm growing",
        "I'm learning so much",
        "this is a great opportunity",
        "I'm excited about the future",
        "I'm looking forward to tomorrow",
        "I can't wait",
        "I'm so excited",
        "I'm thrilled",
        "I'm delighted",
        "I'm overjoyed",
        "I'm ecstatic",
        "I'm elated",
        "I'm blissful",
        "I'm content",
        "I'm satisfied",
        "I'm at peace",
        "I'm calm and happy",
        "I'm relaxed",
        "I'm comfortable",
        "I feel safe",
        "I feel loved",
        "I feel appreciated",
        "I feel valued",
        "I feel supported",
        "I feel understood",
        "I feel accepted",
        "I feel welcomed",
        "my family is wonderful",
        "my friends are amazing",
        "my partner is incredible",
        "I have the best friends",
        "I'm surrounded by love",
        "I'm lucky to have them",
        "I'm fortunate",
        "I'm privileged",
        "I appreciate everything",
        "thank you so much",
        "thanks for everything",
        "I'm so thankful for you",
        "you made my day",
        "you're the best",
        "you're amazing",
        "that was kind of you",
        "that was so thoughtful",
        "that meant a lot to me",
        "I'll never forget this",
        "this is a dream come true",
        "my dream came true",
        "I never thought this would happen",
        "this exceeded my expectations",
        "better than I imagined",
        "the view is breathtaking",
        "the sunset is gorgeous",
        "the music is beautiful",
        "the food was delicious",
        "the meal was excellent",
        "the service was outstanding",
        "the trip was unforgettable",
        "the vacation was perfect",
        "we had a blast",
        "we had so much fun",
        "that was hilarious",
        "I laughed so hard",
        "I haven't laughed like that in years",
        "that joke was priceless",
        "I'm in a great mood",
        "I woke up feeling refreshed",
        "I slept wonderfully",
        "I feel rested",
        "I had the best sleep",
        "the morning sun feels warm",
        "the breeze is refreshing",
        "the garden looks lovely",
        "the flowers are blooming",
        "spring is here and everything feels new",
        "I feel renewed",
        "I feel hopeful",
        "I feel optimistic",
        "the future looks bright",
        "good things are coming",
        "I believe in myself",
        "I trust the process",
        "I'm confident in my abilities",
        "I'm ready for whatever comes next",
        "I'm unstoppable",
        "I feel invincible",
        "nothing can bring me down today",
        "I'm on top of the world",
        "everything is falling into place",
        "the pieces are coming together",
        "it all makes sense now",
        "I found my purpose",
        "I know what I want",
        "clarity feels incredible",
        "I'm free",
        "I feel liberated",
        "a weight has been lifted",
        "I feel lighter",
        "I forgave myself",
        "I let go of the past",
        "I'm moving forward",
        "a fresh start feels amazing",
        "new beginnings are exciting",
        "I'm embracing change",
        "growth feels rewarding",
        "every day gets a little better",
        "I'm healing",
        "I'm recovering well",
        "my health is improving",
        "I feel healthy",
        "I feel fit",
        "the workout was great",
        "I ran my first mile",
        "I hit a new personal best",
        "I'm stronger than yesterday",
        "I made a new friend",
        "I reconnected with an old friend",
        "the reunion was emotional and beautiful",
        "I feel connected",
        "I belong here",
        "this place feels like home",
        "I finally feel at home",
        "I'm comfortable in my own skin",
        "I accept myself",
        "I love who I'm becoming",
        "I'm enough",
        "I am worthy",
        "I deserve good things",
        "kindness is everywhere",
        "people are generous",
        "the world can be beautiful",
        "humanity gives me hope",
        "there's still good in the world",
    ]
    return sentences


def generate_tricky_non_crisis() -> List[str]:
    """Generate 100+ sentences with dark words that should NOT trigger crisis."""
    sentences = [
        # --- Discussing death in third person / fiction / academic ---
        "I watched a movie about death",
        "the character wanted to die in the story",
        "the protagonist killed himself at the end of the book",
        "we discussed suicide prevention in class",
        "the lecture covered suicide risk factors",
        "the training on suicide awareness was helpful",
        "she wrote a thesis on self-harm behavior",
        "the study examined suicidal ideation in teens",
        "the report analyzed mortality rates",
        "he researches end-of-life care",
        "the documentary about death was fascinating",
        "we read about Hamlet wanting to die",
        "Romeo and Juliet both die at the end",
        "the villain wanted to destroy everything",
        "the character said I want to die in the play",
        "the novel explores themes of despair and hopelessness",
        "the poem was about death and rebirth",
        "the painting depicted suffering",
        "the song lyrics mention dying",
        "he sang about wanting to die in the chorus",

        # --- Idioms and figurative language ---
        "I'm dying of laughter",
        "I'm dying to try that restaurant",
        "I'm dying to see the new movie",
        "that joke killed me",
        "I'm dead serious",
        "I nearly died of embarrassment",
        "this is killing me with suspense",
        "I'm dying to know what happens",
        "I could die for a slice of pizza",
        "she's drop dead gorgeous",
        "he killed it on stage",
        "that performance was killer",
        "I'm dead tired",
        "the silence was deafening",
        "I laughed myself to death",
        "it was to die for",
        "that sunset was to die for",
        "my feet are killing me",
        "this workout is killing me",
        "my allergies are killing me today",

        # --- Reporting / quoting others ---
        "my patient expressed suicidal thoughts",
        "the caller said they wanted to die",
        "he told the therapist he felt hopeless",
        "she told the counselor about self-harm",
        "the teenager mentioned wanting to hurt himself",
        "in the article the person described wanting to end it",
        "the memoir describes her battle with suicidal thoughts",
        "he quoted the character saying life is pointless",
        "according to the study many participants felt hopeless",
        "the survey found that thirty percent felt despair",

        # --- Professional / clinical ---
        "the suicide hotline received more calls this year",
        "the crisis center is open twenty four hours",
        "mental health resources are available",
        "if you are suicidal please call the hotline",
        "suicide prevention week starts Monday",
        "the hospital has a self-harm protocol",
        "the therapist assessed suicide risk",
        "the intervention prevented a suicide attempt",
        "crisis intervention training is required",
        "the safety plan includes emergency contacts",

        # --- Games / sports / competition ---
        "I died in the video game",
        "my character died again",
        "game over I got killed",
        "we got destroyed in the match",
        "they killed us in the second half",
        "our team is dead last in the standings",
        "that play was a killer",
        "we got slaughtered on the field",
        "I got wrecked in that round",
        "my ranking is dead",

        # --- Technology / objects ---
        "the battery is dead",
        "my phone died",
        "the server died overnight",
        "the process was killed",
        "kill the background task",
        "the program crashed and died",
        "the old computer is dying",
        "I need to kill this process",
        "the connection dropped dead",
        "the hard drive is dying",

        # --- Historical / news ---
        "thousands died in the earthquake",
        "the death toll reached fifty",
        "the war caused immense suffering",
        "many soldiers lost hope during the siege",
        "the famine led to despair and death",
        "the epidemic killed millions",
        "the survivors described hopelessness",
        "the crisis affected mental health nationwide",
        "depression rates increased during the pandemic",
        "economic despair followed the recession",

        # --- Philosophy / existential (not personal crisis) ---
        "what is the meaning of death",
        "philosophers have debated the nature of suffering",
        "existentialism confronts the absurdity of existence",
        "Camus wrote about the question of suicide",
        "Kierkegaard explored despair as a concept",
        "stoics taught acceptance of death",
        "Buddhism addresses the nature of suffering",
        "mortality gives life its meaning",
        "death is a natural part of life",
        "we all must face our mortality eventually",
    ]
    return sentences


# ---------------------------------------------------------------------------
# Crisis classification
# ---------------------------------------------------------------------------

def is_crisis(v: int, u: int) -> bool:
    """Crisis if V < 50 OR (V < 70 AND U > 60)."""
    return v < 50 or (v < 70 and u > 60)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    category: str
    sentence: str
    expected_crisis: bool
    predicted_crisis: bool
    v: int
    a: int
    d: int
    u: int
    g: int


def run_benchmark(verbose: bool = False) -> dict:
    """Run the full crisis recall benchmark."""
    print("=" * 72)
    print("  CRISIS RECALL BENCHMARK — Targeting 99.9% Recall")
    print("=" * 72)
    print()

    # Load champion config
    config = load_champion_config()
    print(f"  Config: gpu_optimal_v2.json (27 params)")
    print(f"  Crisis rule: V < 50 OR (V < 70 AND U > 60)")
    print()

    # Initialize engine
    engine = PendulumV2(**config)

    # Generate sentences
    categories = {
        "CRISIS": (generate_crisis_sentences(), True),
        "NON-CRISIS NEGATIVE": (generate_non_crisis_negative(), False),
        "NEUTRAL": (generate_neutral(), False),
        "POSITIVE": (generate_positive(), False),
        "TRICKY NON-CRISIS": (generate_tricky_non_crisis(), False),
    }

    total_sentences = sum(len(sents) for sents, _ in categories.values())
    print(f"  Total sentences: {total_sentences}")
    for cat, (sents, _) in categories.items():
        print(f"    {cat}: {len(sents)}")
    print()

    # Run engine on every sentence
    results: List[BenchmarkResult] = []
    start_time = time.time()

    for cat, (sents, expected) in categories.items():
        for s in sents:
            vadug, _ = engine.process_text(s)
            predicted = is_crisis(vadug.v, vadug.u)
            results.append(BenchmarkResult(
                category=cat,
                sentence=s,
                expected_crisis=expected,
                predicted_crisis=predicted,
                v=vadug.v, a=vadug.a, d=vadug.d, u=vadug.u, g=vadug.g,
            ))

    elapsed = time.time() - start_time

    # Calculate metrics
    crisis_results = [r for r in results if r.expected_crisis]
    non_crisis_results = [r for r in results if not r.expected_crisis]

    true_positives = [r for r in crisis_results if r.predicted_crisis]
    false_negatives = [r for r in crisis_results if not r.predicted_crisis]
    false_positives = [r for r in non_crisis_results if r.predicted_crisis]
    true_negatives = [r for r in non_crisis_results if not r.predicted_crisis]

    n_crisis = len(crisis_results)
    n_non_crisis = len(non_crisis_results)

    recall = len(true_positives) / n_crisis if n_crisis else 0
    precision = len(true_positives) / (len(true_positives) + len(false_positives)) if (len(true_positives) + len(false_positives)) else 0
    fn_rate = len(false_negatives) / n_crisis if n_crisis else 0
    fp_rate = len(false_positives) / n_non_crisis if n_non_crisis else 0

    # Per-category false positive breakdown
    fp_by_category = {}
    for cat in ["NON-CRISIS NEGATIVE", "NEUTRAL", "POSITIVE", "TRICKY NON-CRISIS"]:
        cat_results = [r for r in results if r.category == cat]
        cat_fp = [r for r in cat_results if r.predicted_crisis]
        fp_by_category[cat] = {
            "total": len(cat_results),
            "false_positives": len(cat_fp),
            "rate": len(cat_fp) / len(cat_results) if cat_results else 0,
        }

    # Print report
    print("-" * 72)
    print("  RESULTS")
    print("-" * 72)
    print()
    print(f"  Sentences processed:  {len(results)}")
    print(f"  Time:                 {elapsed:.2f}s ({len(results)/elapsed:.0f} sentences/sec)")
    print()
    print(f"  CRISIS RECALL:        {recall*100:.1f}%  ({len(true_positives)}/{n_crisis})"
          f"  {'*** TARGET MET ***' if recall >= 0.999 else '*** BELOW TARGET ***'}")
    print(f"  CRISIS PRECISION:     {precision*100:.1f}%  ({len(true_positives)}/{len(true_positives)+len(false_positives)})")
    print()
    print(f"  True Positives:       {len(true_positives)}")
    print(f"  False Negatives:      {len(false_negatives)}  (MISSED CRISES — the deadly metric)")
    print(f"  False Positives:      {len(false_positives)}  (non-crisis flagged as crisis)")
    print(f"  True Negatives:       {len(true_negatives)}")
    print()
    print(f"  False Negative Rate:  {fn_rate*100:.2f}%")
    print(f"  False Positive Rate:  {fp_rate*100:.2f}%")
    print()

    # Per-category FP breakdown
    print("  False Positive Breakdown:")
    for cat, info in fp_by_category.items():
        print(f"    {cat:25s}  {info['false_positives']:3d}/{info['total']:3d}  ({info['rate']*100:.1f}%)")
    print()

    # Print every missed crisis (false negative) — always
    if false_negatives:
        print("-" * 72)
        print(f"  MISSED CRISES ({len(false_negatives)} false negatives) — EVERY ONE MATTERS")
        print("-" * 72)
        for r in false_negatives:
            print(f"    V={r.v:3d} A={r.a:3d} D={r.d:3d} U={r.u:3d} G={r.g:3d}  \"{r.sentence}\"")
        print()
    else:
        print("  *** ZERO MISSED CRISES — perfect recall ***")
        print()

    # Print false positives if verbose
    if verbose and false_positives:
        print("-" * 72)
        print(f"  FALSE POSITIVES ({len(false_positives)} non-crisis flagged as crisis)")
        print("-" * 72)
        for r in false_positives:
            print(f"    [{r.category}]  V={r.v:3d} A={r.a:3d} D={r.d:3d} U={r.u:3d} G={r.g:3d}  \"{r.sentence}\"")
        print()

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_source": "gpu_optimal_v2.json",
        "crisis_rule": "V < 50 OR (V < 70 AND U > 60)",
        "total_sentences": len(results),
        "crisis_sentences": n_crisis,
        "non_crisis_sentences": n_non_crisis,
        "metrics": {
            "recall": round(recall, 6),
            "precision": round(precision, 6),
            "false_negative_rate": round(fn_rate, 6),
            "false_positive_rate": round(fp_rate, 6),
            "true_positives": len(true_positives),
            "false_negatives": len(false_negatives),
            "false_positives": len(false_positives),
            "true_negatives": len(true_negatives),
        },
        "false_positive_breakdown": fp_by_category,
        "missed_crises": [
            {"sentence": r.sentence, "v": r.v, "a": r.a, "d": r.d, "u": r.u, "g": r.g}
            for r in false_negatives
        ],
        "false_positives_list": [
            {"category": r.category, "sentence": r.sentence, "v": r.v, "a": r.a, "d": r.d, "u": r.u, "g": r.g}
            for r in false_positives
        ],
        "time_seconds": round(elapsed, 2),
    }

    output_path = os.path.join(os.path.dirname(__file__), "crisis_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to: {output_path}")
    print()

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crisis Recall Benchmark")
    parser.add_argument("--verbose", action="store_true",
                        help="Show all false positives in detail")
    args = parser.parse_args()
    run_benchmark(verbose=args.verbose)
