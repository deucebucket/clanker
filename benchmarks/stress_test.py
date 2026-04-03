#!/usr/bin/env python3
"""Stress Test — V5.5 engine on 250+ real conversational sentences.

25+ sentences per category. Real text, not lab sentences.
ALL engine calls use V5.5 (engine.pendulum). No V2 imports.

Usage:
    python3 benchmarks/stress_test.py
    python3 benchmarks/stress_test.py --verbose
    python3 benchmarks/stress_test.py --category sarcasm
"""

import sys, os, time, json, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.pendulum import compute_vadug

# Each category: list of (text, expected_direction) where direction is "pos", "neg", or "neutral"
CATEGORIES = {
    "sarcasm": [
        ("oh great another meeting", "neg"),
        ("nice work genius", "neg"),
        ("oh joy", "neg"),
        ("wow thanks so much for the help", "neg"),
        ("oh sure thats exactly what i needed", "neg"),
        ("what a wonderful surprise", "neg"),
        ("love that for you", "neg"),
        ("oh how lovely", "neg"),
        ("oh fantastic more homework", "neg"),
        ("clearly this was well thought out", "neg"),
        ("oh perfect just what i wanted", "neg"),
        ("oh brilliant another monday", "neg"),
        ("what a treat this has been", "neg"),
        ("oh wonderful more overtime", "neg"),
        ("oh goody another redo", "neg"),
        ("oh nice real smooth einstein", "neg"),
        ("oh sure because that worked so well last time", "neg"),
        ("what a genius move champ", "neg"),
        ("oh delightful another traffic jam", "neg"),
        ("oh cool cant wait for that", "neg"),
        ("oh spectacular another bill", "neg"),
        ("oh amazing more chores", "neg"),
        ("what a surprise who could have seen that coming", "neg"),
        ("oh wow thanks for nothing", "neg"),
        ("oh great just great", "neg"),
    ],
    "slang_positive": [
        ("bro that was fire", "pos"),
        ("no cap that was insane", "pos"),
        ("lol im dead", "pos"),
        ("she absolutely killed it", "pos"),
        ("thats bussin fr", "pos"),
        ("nah that goes hard", "pos"),
        ("lowkey goated", "pos"),
        ("clutch play", "pos"),
        ("understood the assignment", "pos"),
        ("bruh im screaming", "pos"),
        ("lmao that destroyed me", "pos"),
        ("dude that slaps", "pos"),
        ("bro im crying laughing", "pos"),
        ("deadass that was wild", "pos"),
        ("ngl that was sick", "pos"),
        ("fr fr that hit different", "pos"),
        ("haha im literally choking", "pos"),
        ("lol get wrecked", "pos"),
        ("bruh that was crazy good", "pos"),
        ("omg im dying", "pos"),
        ("tbh that was insane", "pos"),
        ("dude that was nuts", "pos"),
        ("bruh im shook", "pos"),
        ("lmao he got destroyed", "pos"),
        ("ngl that goes stupid hard", "pos"),
    ],
    "passive_aggressive": [
        ("whatever makes you happy", "neg"),
        ("im fine", "neg"),
        ("do what you want", "neg"),
        ("no its fine i didnt want to go anyway", "neg"),
        ("sure go ahead and ruin it", "neg"),
        ("if thats what you think", "neg"),
        ("whatever you say", "neg"),
        ("its not like i care", "neg"),
        ("i guess i deserved it", "neg"),
        ("i suppose youre right", "neg"),
        ("no go ahead have fun without me", "neg"),
        ("i said its fine", "neg"),
        ("okay whatever", "neg"),
        ("cool glad you decided without me", "neg"),
        ("no really its totally fine", "neg"),
        ("you always know best", "neg"),
        ("must be nice to not have to worry", "neg"),
        ("i mean if you dont care then why would i", "neg"),
        ("forget it it doesnt matter", "neg"),
        ("sorry for being such a burden", "neg"),
        ("i didnt realize my opinion didnt matter", "neg"),
        ("oh dont mind me ill just sit here", "neg"),
        ("i guess some people just dont get it", "neg"),
        ("thats fine i didnt need your help anyway", "neg"),
        ("no worries ill just figure it out myself", "neg"),
    ],
    "grief": [
        ("my mom died last month", "neg"),
        ("i lost my best friend", "neg"),
        ("the house feels empty without her", "neg"),
        ("i keep expecting him to walk through the door", "neg"),
        ("everyone says it gets easier", "neg"),
        ("i cant listen to that song anymore", "neg"),
        ("his chair is still at the table", "neg"),
        ("the last thing i said was something stupid", "neg"),
        ("i miss her so much it physically hurts", "neg"),
        ("its been a year and i still cry", "neg"),
        ("i found his old jacket in the closet", "neg"),
        ("holidays are the worst now", "neg"),
        ("nobody tells you grief comes in waves", "neg"),
        ("i still reach for my phone to call her", "neg"),
        ("the silence in the house is deafening", "neg"),
        ("i cant go back to that restaurant", "neg"),
        ("his side of the bed is still cold", "neg"),
        ("i keep her voicemail just to hear her voice", "neg"),
        ("the dog keeps waiting by the door for him", "neg"),
        ("i donated his clothes today", "neg"),
        ("we were supposed to grow old together", "neg"),
        ("i hate that life just keeps going", "neg"),
        ("three years and some days are still just as hard", "neg"),
        ("i see her face in our daughter every day", "neg"),
        ("they said time heals but they lied", "neg"),
    ],
    "genuine_positive": [
        ("I JUST GOT THE JOB", "pos"),
        ("my kid took their first steps today", "pos"),
        ("clean for 6 months now", "pos"),
        ("just got engaged", "pos"),
        ("we closed on the house", "pos"),
        ("finally graduated", "pos"),
        ("she said yes", "pos"),
        ("first day at the new job went great", "pos"),
        ("i finally feel like i belong", "pos"),
        ("im proud of what we built", "pos"),
        ("sober for 2 years today", "pos"),
        ("my daughter was born this morning", "pos"),
        ("i passed the bar exam", "pos"),
        ("we adopted a baby", "pos"),
        ("i made it through the surgery", "pos"),
        ("he proposed on the beach", "pos"),
        ("they accepted my application", "pos"),
        ("i love spending time with you", "pos"),
        ("this is the happiest ive been in years", "pos"),
        ("we won the championship", "pos"),
        ("i got my first paycheck today", "pos"),
        ("the test came back negative", "pos"),
        ("she gave me another chance", "pos"),
        ("i finally paid off my student loans", "pos"),
        ("my best friend surprised me for my birthday", "pos"),
    ],
    "ambiguous": [
        ("the meeting is at three", "neutral"),
        ("i heard what you said", "neutral"),
        ("your phone was ringing", "neutral"),
        ("someone left you a message", "neutral"),
        ("your boss called", "neutral"),
        ("the weather is changing", "neutral"),
        ("i saw your car at the store", "neutral"),
        ("we need to talk", "neutral"),
        ("i have something to tell you", "neutral"),
        ("things are about to change", "neutral"),
        ("i was thinking about what you said", "neutral"),
        ("interesting choice", "neutral"),
        ("i noticed you were gone", "neutral"),
        ("thats one way to do it", "neutral"),
        ("your mom stopped by", "neutral"),
        ("i got a letter today", "neutral"),
        ("they announced layoffs", "neutral"),
        ("the doctor wants to see me again", "neutral"),
        ("i ran into someone from high school", "neutral"),
        ("my parents want to talk to us", "neutral"),
        ("something happened at work today", "neutral"),
        ("i saw your ex at the store", "neutral"),
        ("the school called about your kid", "neutral"),
        ("i found something in the attic", "neutral"),
        ("we got the results back", "neutral"),
    ],
    "betrayal": [
        ("my wife cheated on me with my best friend", "neg"),
        ("he told everyone my secret", "neg"),
        ("she took the kids and left", "neg"),
        ("i found the messages on his phone", "neg"),
        ("my partner has been lying for months", "neg"),
        ("they went behind my back", "neg"),
        ("he stole my idea and got promoted", "neg"),
        ("she pretended to be my friend", "neg"),
        ("i trusted him with everything", "neg"),
        ("my best friend sided with my ex", "neg"),
        ("she was sleeping with my brother", "neg"),
        ("he drained our savings account", "neg"),
        ("they used my vulnerability against me", "neg"),
        ("she promised shed never do it again", "neg"),
        ("he was living a double life", "neg"),
        ("my own family testified against me", "neg"),
        ("she told my boss about my addiction", "neg"),
        ("he took credit for my work in front of everyone", "neg"),
        ("i caught her in another lie", "neg"),
        ("they excluded me from the group on purpose", "neg"),
        ("he manipulated me for years", "neg"),
        ("she faked the pregnancy to keep me", "neg"),
        ("my mentor turned out to be a fraud", "neg"),
        ("he recorded our private conversations", "neg"),
        ("she sabotaged my job interview", "neg"),
    ],
    "self_worth": [
        ("im worthless", "neg"),
        ("i am nothing", "neg"),
        ("everyone would be better off without me", "neg"),
        ("i dont deserve to be happy", "neg"),
        ("im just a burden", "neg"),
        ("why would anyone love me", "neg"),
        ("im not good enough", "neg"),
        ("i always mess everything up", "neg"),
        ("im proud of what we built", "pos"),
        ("i finally feel like i belong", "pos"),
        ("nobody wants me around", "neg"),
        ("i ruin everything i touch", "neg"),
        ("im a waste of space", "neg"),
        ("i dont matter to anyone", "neg"),
        ("i cant do anything right", "neg"),
        ("im just taking up space", "neg"),
        ("everyone leaves me eventually", "neg"),
        ("i wish i was someone else", "neg"),
        ("i hate myself so much", "neg"),
        ("im the worst person i know", "neg"),
        ("nobody would even notice if i was gone", "neg"),
        ("i dont deserve friends", "neg"),
        ("im a failure at everything", "neg"),
        ("im proud of how far ive come", "pos"),
        ("i finally believe in myself", "pos"),
    ],
    "crisis": [
        ("i want to kill myself", "neg"),
        ("nobody likes me i might as well jump", "neg"),
        ("i dont want to be here anymore", "neg"),
        ("nobody would even notice", "neg"),
        ("honestly might just end it lol", "neg"),
        ("im so fucking done", "neg"),
        ("cant do this shit anymore", "neg"),
        ("no point in trying anymore", "neg"),
        ("ive been writing goodbye letters", "neg"),
        ("i just want the pain to stop", "neg"),
        ("whats the point of waking up", "neg"),
        ("i feel calm now that ive decided", "neg"),
        ("just bought a bunch of pills", "neg"),
        ("found a bridge near my house", "neg"),
        ("i gave my dog to my neighbor", "neg"),
        ("im giving away all my stuff", "neg"),
        ("tonight is my last night", "neg"),
        ("nobody cares if i live or die", "neg"),
        ("i cant take another day of this", "neg"),
        ("everything would be easier without me", "neg"),
        ("im tired of pretending im okay", "neg"),
        ("i dont see a future for myself", "neg"),
        ("the world would keep spinning without me", "neg"),
        ("i just want to disappear", "neg"),
        ("i wish i had the courage to end it", "neg"),
    ],
    "casual_safe": [
        ("whats for dinner", "neutral"),
        ("did you see the game last night", "neutral"),
        ("im gonna grab coffee", "neutral"),
        ("what time is the movie", "neutral"),
        ("can you pick up milk", "neutral"),
        ("the wifi is slow today", "neutral"),
        ("i need to do laundry", "neutral"),
        ("traffic was crazy this morning", "neutral"),
        ("did you walk the dog", "neutral"),
        ("lets order pizza", "neutral"),
        ("i have a dentist appointment tomorrow", "neutral"),
        ("the neighbors are having a party", "neutral"),
        ("my alarm didnt go off", "neutral"),
        ("the dishwasher is broken", "neutral"),
        ("i need to renew my license", "neutral"),
        ("when does the store close", "neutral"),
        ("theres a package on the porch", "neutral"),
        ("im running late", "neutral"),
        ("did you feed the cat", "neutral"),
        ("the car needs an oil change", "neutral"),
        ("what channel is the game on", "neutral"),
        ("i forgot my lunch at home", "neutral"),
        ("the power went out for a minute", "neutral"),
        ("parking was a nightmare", "neutral"),
        ("i have to work saturday", "neutral"),
    ],
    "directed_negative": [
        ("youre an idiot", "neg"),
        ("go fuck yourself", "neg"),
        ("nobody asked for your opinion", "neg"),
        ("shut the fuck up", "neg"),
        ("youre pathetic", "neg"),
        ("i hate you", "neg"),
        ("youre dead to me", "neg"),
        ("get the hell out", "neg"),
        ("youre a piece of shit", "neg"),
        ("dont ever talk to me again", "neg"),
        ("you disgust me", "neg"),
        ("youre the worst thing that happened to me", "neg"),
        ("i wish id never met you", "neg"),
        ("you make me sick", "neg"),
        ("youre nothing to me", "neg"),
        ("back the fuck off", "neg"),
        ("youre a liar and a coward", "neg"),
        ("i never want to see your face again", "neg"),
        ("piss off", "neg"),
        ("youre just like your father", "neg"),
        ("you ruined my life", "neg"),
        ("i regret every moment with you", "neg"),
        ("youre a waste of my time", "neg"),
        ("you dont deserve her", "neg"),
        ("leave me alone", "neg"),
    ],
}


def classify(vadug):
    if vadug.v >= 145:
        return "pos"
    elif vadug.v <= 110:
        return "neg"
    else:
        return "neutral"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--category", type=str, default=None,
                       help="Run only one category")
    args = parser.parse_args()

    total_sentences = sum(len(s) for s in CATEGORIES.values())
    print(f"\n{'=' * 70}")
    print(f"  STRESS TEST — V5.5 engine ({total_sentences} sentences)")
    print(f"{'=' * 70}")

    total = 0
    correct = 0
    results = {}

    cats = {args.category: CATEGORIES[args.category]} if args.category else CATEGORIES

    for category, sentences in cats.items():
        cat_correct = 0
        cat_total = len(sentences)
        misses = []

        for text, expected in sentences:
            vadug, meta = compute_vadug(text)
            got = classify(vadug)
            total += 1

            if got == expected:
                cat_correct += 1
                correct += 1
            else:
                misses.append((text, expected, got, vadug))

            if args.verbose:
                tag = "OK" if got == expected else "MISS"
                structs = [s.pattern for s in meta.get("structures", [])]
                ss = ",".join(structs) if structs else "-"
                print(f"  [{tag}] V={vadug.v:3d} W={vadug.w:3d} I={vadug.i:3d} exp={expected} got={got} [{ss}]  {text}")

        pct = cat_correct / cat_total * 100
        status = "PASS" if pct >= 80 else "WEAK" if pct >= 50 else "FAIL"
        print(f"  [{status}] {category:<22} {cat_correct}/{cat_total} ({pct:.0f}%)")
        if misses and not args.verbose:
            for text, exp, got, v in misses[:5]:
                print(f"         V={v.v:3d} exp={exp} got={got}  {text}")

        results[category] = {
            "correct": cat_correct,
            "total": cat_total,
            "pct": round(pct, 1),
            "misses": [(t, e, g) for t, e, g, _ in misses],
        }

    overall = correct / total * 100
    print(f"\n  OVERALL: {correct}/{total} ({overall:.1f}%)")
    print(f"{'=' * 70}")

    out = {
        "engine": "v5.5",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_sentences": total,
        "overall": round(overall, 1),
        "categories": results,
    }
    out_path = os.path.join(os.path.dirname(__file__), "stress_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
