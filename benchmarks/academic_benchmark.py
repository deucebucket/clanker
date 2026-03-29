#!/usr/bin/env python3
"""Academic Benchmark: Clanker vs VADER vs TextBlob vs RoBERTa on real datasets."""

import time, re, sys, os, json, argparse
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from demo.simulator import SequentialPendulum, VADUG, nearest_emotion, WORD_FORCES
from demo.classifier import AdaptiveClassifier
from demo.intent import IntentDetector
from demo.morphemes import ROOTS, PREFIXES, SUFFIXES
from demo.pendulum_v2 import PendulumV2

_classifier = AdaptiveClassifier()
_intent_detector = IntentDetector()
_v2_engine = None  # lazily initialized with configurable params

HAS_TRANSFORMERS = False
try:
    from transformers import pipeline as hf_pipeline_fn
    HAS_TRANSFORMERS = True
except: pass

try:
    from datasets import load_dataset
except:
    print("ERROR: pip install datasets"); sys.exit(1)

try:
    from sklearn.metrics import f1_score
    HAS_SKLEARN = True
except:
    HAS_SKLEARN = False

_vader = SentimentIntensityAnalyzer()

def run_vader(text):
    c = _vader.polarity_scores(text)["compound"]
    return ("positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"), c

def run_textblob(text):
    p = TextBlob(text).sentiment.polarity
    return ("positive" if p > 0.05 else "negative" if p < -0.05 else "neutral"), p

def run_clanker(text):
    intent_mode, _ = _intent_detector.detect(text)
    # Multi-path processing: run literal + sarcasm paths, pick most consistent
    from demo.multipath import MultiPathProcessor
    mp = MultiPathProcessor()
    v, a, d, u, g, path, consistency, history = mp.process(text)
    # Entropy override check
    if history and _classifier._entropy.should_override_neutral(history, v, intent_mode):
        label = "neutral"
    else:
        label = _classifier.classify(v, intent_mode)
    return label, v, a, d, u, g

_v2_classify_mode = "three_way"  # set per-dataset by bench()

def run_clanker_v2(text):
    global _v2_engine
    if _v2_engine is None:
        _v2_engine = PendulumV2(
            momentum=0.903, force_scale=2.894, direct_push_cap=0.085,
            direct_push_trigger=158.7, crisis_v=68.9, crisis_u=95.5,
            question_dampener=0.529, scale_v=1.920, scale_a=0.789,
            scale_d=2.901, scale_u=0.30, scale_g=0.707,
            threshold_low=116.7, threshold_high=155.2,
            negation_decay_operator=0.850, negation_decay_neutral=0.60,
            negation_decay_payload=0.523,
        )
    vadug, _ = _v2_engine.process_text(text)
    label = _v2_engine.classify(vadug.v, mode=_v2_classify_mode)
    return label, vadug.v, vadug.a, vadug.d, vadug.u, vadug.g

_hf = None
def run_roberta(text):
    global _hf
    if not HAS_TRANSFORMERS: return None, 0
    if _hf is None:
        print("  Loading RoBERTa..."); _hf = hf_pipeline_fn("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", top_k=None, truncation=True)
    r = _hf(text[:512])[0]
    top = max(r, key=lambda x: x["score"])
    return top["label"], top["score"]

GOEMO = {0:("admiration","positive"),1:("amusement","positive"),2:("anger","negative"),3:("annoyance","negative"),4:("approval","positive"),5:("caring","positive"),6:("confusion","neutral"),7:("curiosity","neutral"),8:("desire","positive"),9:("disappointment","negative"),10:("disapproval","negative"),11:("disgust","negative"),12:("embarrassment","negative"),13:("excitement","positive"),14:("fear","negative"),15:("gratitude","positive"),16:("grief","negative"),17:("joy","positive"),18:("love","positive"),19:("nervousness","negative"),20:("optimism","positive"),21:("pride","positive"),22:("realization","neutral"),23:("relief","positive"),24:("remorse","negative"),25:("sadness","negative"),26:("surprise","neutral"),27:("neutral","neutral")}

def bench(name, ds, get_text, get_truth, max_n=None):
    global _v2_classify_mode
    # Classification adapters: binary for binary datasets, three_way for multi-class
    if name in ("SST-2", "TweetEval"):
        _v2_classify_mode = "binary"
    else:
        _v2_classify_mode = "three_way"
    if max_n: ds = ds.select(range(min(max_n, len(ds))))
    engines = ["VADER","TextBlob","Clanker","ClankerV2"] + (["RoBERTa"] if HAS_TRANSFORMERS else [])
    res = {e:{"p":[],"t":[],"ms":0} for e in engines}
    dim = []
    for i, row in enumerate(ds):
        text, truth = get_text(row), get_truth(row)
        if truth is None: continue
        for e in engines:
            t0 = time.perf_counter()
            if e == "Clanker":
                lbl, v, a, d, u, g = run_clanker(text)
                dim.append((get_truth(row, raw=True) if hasattr(get_truth,'__code__') and 'raw' in get_truth.__code__.co_varnames else truth, v, a, d, u, g))
            elif e == "ClankerV2":
                lbl, v, a, d, u, g = run_clanker_v2(text)
            elif e == "VADER": lbl, _ = run_vader(text)
            elif e == "TextBlob": lbl, _ = run_textblob(text)
            else:
                r = run_roberta(text)
                if r[0] is None: continue
                lbl = r[0]
            res[e]["ms"] += (time.perf_counter()-t0)*1000
            res[e]["p"].append(lbl); res[e]["t"].append(truth)
        if (i+1) % 500 == 0: print(f"  {i+1}/{len(ds)}...", end="\r")

    print(f"\n  {name} ({len(res['VADER']['t'])} examples)")
    print(f"  {'Engine':<12} {'Acc':>8} {'F1':>8} {'Neutral%':>9} {'ms/s':>8}")
    print(f"  {'-'*50}")
    summary = {}
    for e in engines:
        n = len(res[e]["t"])
        if n == 0: continue
        correct = sum(1 for p,t in zip(res[e]["p"],res[e]["t"]) if p==t)
        acc = correct/n*100
        neut = sum(1 for p in res[e]["p"] if p=="neutral")/n*100
        f1 = f1_score(res[e]["t"],res[e]["p"],average="macro",zero_division=0,labels=["positive","negative","neutral"])*100 if HAS_SKLEARN else 0
        avg_ms = res[e]["ms"]/n
        print(f"  {e:<12} {acc:>7.1f}% {f1:>7.1f}% {neut:>8.1f}% {avg_ms:>7.2f}")
        summary[e] = {"accuracy":round(acc,1),"f1":round(f1,1),"neutral_pct":round(neut,1),"samples":n}
    return {"dataset":name,"results":summary,"dimensional":dim}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    mx = 1000 if args.quick else None

    print(f"\n{'#'*80}")
    print(f"  ACADEMIC BENCHMARK — {len(WORD_FORCES)} word forces, {len(ROOTS)} morpheme roots")
    print(f"{'#'*80}")

    all_r = []

    # SST-2
    sst = load_dataset("stanfordnlp/sst2", split="validation")
    all_r.append(bench("SST-2", sst, lambda r:r["sentence"], lambda r,raw=False:"positive" if r["label"]==1 else "negative", mx))

    # GoEmotions
    ge = load_dataset("google-research-datasets/go_emotions", split="test")
    def ge_truth(r, raw=False):
        if not r["labels"]: return None
        emo, sent = GOEMO.get(r["labels"][0], ("unk","neutral"))
        return emo if raw else sent
    all_r.append(bench("GoEmotions", ge, lambda r:r["text"], ge_truth, mx))

    # TweetEval
    te = load_dataset("cardiffnlp/tweet_eval", "emotion", split="test")
    tw_sent = {0:"negative",1:"positive",2:"positive",3:"negative"}
    all_r.append(bench("TweetEval", te, lambda r:r["text"], lambda r,raw=False:tw_sent[r["label"]], mx))

    # Summary
    print(f"\n{'#'*80}")
    engines = list(all_r[0]["results"].keys())
    print(f"  {'Dataset':<20}", end="")
    for e in engines: print(f" {e:>12}", end="")
    print()
    print(f"  {'-'*70}")
    totals = {e:{"c":0,"n":0} for e in engines}
    for r in all_r:
        print(f"  {r['dataset']:<20}", end="")
        for e in engines:
            if e in r["results"]:
                a = r["results"][e]["accuracy"]
                print(f" {a:>10.1f}%", end="")
                totals[e]["c"] += int(a*r["results"][e]["samples"]/100)
                totals[e]["n"] += r["results"][e]["samples"]
        print()
    print(f"  {'-'*70}")
    print(f"  {'WEIGHTED AVG':<20}", end="")
    for e in engines:
        if totals[e]["n"]>0: print(f" {totals[e]['c']/totals[e]['n']*100:>10.1f}%", end="")
    print(f"\n{'#'*80}\n")

    with open(os.path.join(os.path.dirname(__file__),"academic_results.json"),"w") as f:
        json.dump({"benchmarks":[{k:v for k,v in r.items() if k!="dimensional"} for r in all_r]},f,indent=2)

    # Log V2 results to experiment tracker
    if _v2_engine is not None:
        from benchmarks.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker()
        v2_results = {}
        ds_keys = {"SST-2": "sst2", "GoEmotions": "goemotions", "TweetEval": "tweeteval"}
        for r in all_r:
            ds_name = r["dataset"]
            key = ds_keys.get(ds_name, ds_name.lower())
            if "ClankerV2" in r["results"]:
                v2_results[key] = r["results"]["ClankerV2"]
        if v2_results:
            tracker.log_experiment(
                theory="baseline-v2-untuned",
                engine="v2",
                config=_v2_engine.get_config(),
                results=v2_results,
                notes="Automated benchmark run",
            )

if __name__ == "__main__": main()
