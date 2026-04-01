#!/usr/bin/env python3
"""
Massive trace scorer for training data generation.
Scores Twitch chat, FO76 dialogue, and Gutenberg sentences with full word-by-word traces.
"""

import sys
import json
import re
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/var/home/deucebucket/ai-drive/clanker-lang')
from engine.pendulum import compute_vadug
from engine.forces_curated import EMOTIONAL_VOCABULARY as V
from engine.word_classifier import classify_sentence

OUTPUT_PATH = Path('/var/home/deucebucket/ai-drive/clanker-lang/training/data/massive_traces.jsonl')
PARAGRAPHS_PATH = Path('/var/home/deucebucket/ai-drive/clanker-lang/training/data/massive_traces_paragraphs.jsonl')

TWITCH_31 = Path('/home/deucebucket/Desktop/31.txt')
TWITCH_32 = Path('/home/deucebucket/Desktop/32.txt')
FO76_PATH = Path('/var/home/deucebucket/ai-drive/fo76-data/data/orphaned_voices/transcriptions.json')
GUTENBERG_PATH = Path('/var/home/deucebucket/ai-drive/clanker-lang/benchmarks/gutenberg_sentences.jsonl')

# Regex for twitch line: [2026-03-31 00:00:26] #channel username: message
TWITCH_RE = re.compile(
    r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+#(\S+)\s+(\S+):\s+(.+)$'
)

BOT_NAMES = {'nightbot', 'streamelements', 'moobot', 'fossabot', 'wizebot', 'soundalerts',
             'botrix', 'coebot', 'deepbot', 'ohbot', 'phantombot', 'ankhbot', 'pretzelrocks'}

def is_valid_twitch(username, message):
    """Filter out bots, commands, links, subscriptions, and short messages."""
    if username.lower() in BOT_NAMES:
        return False
    if message.startswith('!'):
        return False
    if 'http://' in message or 'https://' in message:
        return False
    if len(message) < 10:
        return False
    # Skip subscription/raid/host notices (often contain special chars or are mostly numbers)
    if re.match(r'^(subscribed|gifted|raided|hosted|bits)', message.lower()):
        return False
    return True

def clean_twitch_message(message):
    """Remove @mentions and excessive emotes, normalize whitespace."""
    # Remove @username mentions
    msg = re.sub(r'@\S+', '', message)
    # Remove repeated emote-like tokens (uppercase runs that repeat)
    msg = re.sub(r'\b([A-Z]{3,})\s+\1(\s+\1)*\b', r'\1', msg)
    # Collapse whitespace
    msg = re.sub(r'\s+', ' ', msg).strip()
    return msg

def full_trace(text):
    """Score text with full word-by-word trace. Returns (user, assistant) pair or None on error."""
    try:
        vadug, meta = compute_vadug(text)
        words = text.lower().split()
        if not words:
            return None
        roles = classify_sentence(words)

        word_data = []
        for r in roles:
            force = V.get(r.word)
            word_data.append({
                'word': r.word,
                'role': r.role,
                'force': list(force) if force else None,
            })

        structs = [s.pattern for s in meta.get('structures', [])]

        trace_parts = []
        for wd in word_data:
            if wd['force']:
                trace_parts.append(f"{wd['word']}({wd['role']} V={wd['force'][0]:+d})")
            else:
                trace_parts.append(f"{wd['word']}({wd['role']})")

        trace_str = ' '.join(trace_parts)
        struct_str = ', '.join(structs) if structs else 'none'

        answer = (
            f"Words: {trace_str}. "
            f"Patterns: {struct_str}. "
            f"Result: V={vadug.v} A={vadug.a} D={vadug.d} "
            f"U={vadug.u} G={vadug.g} W={vadug.w} I={vadug.i}"
        )

        return {
            'user': f'Score: "{text}"',
            'assistant': answer,
        }
    except Exception as e:
        return None


def parse_twitch_file(path, max_messages=10000):
    """Parse a twitch chat log, return (timestamp_str, username, message) tuples."""
    results = []
    seen = set()
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            m = TWITCH_RE.match(line)
            if not m:
                continue
            ts_str, channel, username, message = m.groups()
            if not is_valid_twitch(username, message):
                continue
            cleaned = clean_twitch_message(message)
            if len(cleaned) < 10:
                continue
            key = cleaned.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            results.append((ts_str, username, cleaned))
            if len(results) >= max_messages:
                break
    return results


def group_into_paragraphs(chat_messages, window_seconds=60):
    """
    Group consecutive messages from the same user within window_seconds into paragraphs.
    chat_messages: list of (ts_str, username, message)
    Returns list of (username, paragraph_text)
    """
    from datetime import datetime

    paragraphs = []
    if not chat_messages:
        return paragraphs

    # Parse timestamps
    parsed = []
    for ts_str, username, msg in chat_messages:
        try:
            dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
        except Exception:
            continue
        parsed.append((dt, username, msg))

    if not parsed:
        return paragraphs

    # Group by user + time window
    groups = []
    current_group = [parsed[0]]
    for i in range(1, len(parsed)):
        dt, username, msg = parsed[i]
        prev_dt, prev_user, _ = current_group[-1]
        if username == prev_user and (dt - prev_dt).total_seconds() <= window_seconds:
            current_group.append(parsed[i])
        else:
            if len(current_group) >= 2:
                groups.append(current_group)
            current_group = [parsed[i]]
    if len(current_group) >= 2:
        groups.append(current_group)

    for group in groups:
        username = group[0][1]
        paragraph = ' '.join(msg for _, _, msg in group)
        if len(paragraph) >= 20:
            paragraphs.append((username, paragraph))

    return paragraphs


def score_twitch(path, max_messages=10000, label='twitch'):
    print(f"\n--- Twitch: {path.name} ---")
    messages = parse_twitch_file(path, max_messages)
    print(f"  Parsed {len(messages)} unique valid messages")

    records = []
    errors = 0
    for i, (ts_str, username, msg) in enumerate(messages):
        rec = full_trace(msg)
        if rec:
            rec['source'] = label
            records.append(rec)
        else:
            errors += 1
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(messages)} scored ({errors} errors)")

    print(f"  Done: {len(records)} scored, {errors} errors")

    # Also return raw parsed for paragraph grouping
    return records, messages


def score_fo76():
    print("\n--- FO76 Dialogue ---")
    with open(FO76_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    transcriptions = data['transcriptions'] if isinstance(data, dict) else data
    print(f"  Total entries: {len(transcriptions)}")

    records = []
    errors = 0
    for entry in transcriptions:
        text = entry.get('transcript', '').strip()
        if not text or len(text) < 5:
            continue
        rec = full_trace(text)
        if rec:
            rec['source'] = 'fo76_dialogue'
            records.append(rec)
        else:
            errors += 1

    print(f"  Done: {len(records)} scored, {errors} errors")
    return records


def score_gutenberg(max_sentences=20000):
    print("\n--- Gutenberg ---")
    records = []
    errors = 0
    seen = set()
    count = 0

    with open(GUTENBERG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= max_sentences:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get('text', '').strip()
            except Exception:
                continue

            if not text or len(text) < 10:
                continue

            # Skip lines that are mostly non-alphabetic (illustration captions, etc.)
            alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
            if alpha_ratio < 0.5:
                continue

            # Skip Gutenberg boilerplate lines
            if '***' in text or '[Illustration' in text or 'GUTENBERG EBOOK' in text:
                continue

            key = text.lower()[:100]
            if key in seen:
                continue
            seen.add(key)

            rec = full_trace(text)
            if rec:
                rec['source'] = obj.get('source', 'gutenberg')
                records.append(rec)
                count += 1
            else:
                errors += 1

            if count % 2000 == 0 and count > 0:
                print(f"  {count}/{max_sentences} scored ({errors} errors)")

    print(f"  Done: {len(records)} scored, {errors} errors")
    return records


def score_paragraphs(chat31_messages, chat32_messages):
    print("\n--- Twitch Paragraphs ---")
    all_messages = chat31_messages + chat32_messages
    paragraphs = group_into_paragraphs(all_messages)
    print(f"  Found {len(paragraphs)} multi-message paragraph groups")

    records = []
    errors = 0
    for username, paragraph in paragraphs:
        rec = full_trace(paragraph)
        if rec:
            rec['source'] = 'twitch_paragraph'
            rec['username'] = username
            records.append(rec)
        else:
            errors += 1

    print(f"  Done: {len(records)} paragraph records, {errors} errors")
    return records


def main():
    start = time.time()
    print("=== Massive Trace Scorer ===")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Paragraphs: {PARAGRAPHS_PATH}")

    # Score all sources
    twitch31_records, twitch31_messages = score_twitch(TWITCH_31, max_messages=10000, label='twitch_31')
    twitch32_records, twitch32_messages = score_twitch(TWITCH_32, max_messages=10000, label='twitch_32')
    fo76_records = score_fo76()
    gutenberg_records = score_gutenberg(max_sentences=20000)
    paragraph_records = score_paragraphs(twitch31_messages, twitch32_messages)

    # Combine sentence records
    all_records = twitch31_records + twitch32_records + fo76_records + gutenberg_records

    print(f"\n=== Summary ===")
    print(f"  Twitch 31:  {len(twitch31_records):>6}")
    print(f"  Twitch 32:  {len(twitch32_records):>6}")
    print(f"  FO76:       {len(fo76_records):>6}")
    print(f"  Gutenberg:  {len(gutenberg_records):>6}")
    print(f"  Paragraphs: {len(paragraph_records):>6}")
    print(f"  TOTAL:      {len(all_records):>6} sentences + {len(paragraph_records)} paragraphs")

    # Write sentence records
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    # Write paragraph records separately
    with open(PARAGRAPHS_PATH, 'w', encoding='utf-8') as f:
        for rec in paragraph_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Sentences -> {OUTPUT_PATH}")
    print(f"Paragraphs -> {PARAGRAPHS_PATH}")


if __name__ == '__main__':
    main()
