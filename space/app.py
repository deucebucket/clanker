#!/usr/bin/env python3
"""Clanker-Lang HuggingFace Space: side-by-side chat + glass-box debug view.

Two-panel Gradio app showing a clean chat interface (left) and full
emotional processing internals (right). Every number is auditable.

Resolves: #48, #54, #31
"""

import gradio as gr
import sys
import os

# Ensure demo package is importable
# On HuggingFace Spaces, app.py is at repo root alongside demo/
# Locally, app.py is in space/ so we need to go up one level
_here = os.path.dirname(os.path.abspath(__file__))
_root = _here if os.path.isdir(os.path.join(_here, "demo")) else os.path.dirname(_here)
sys.path.insert(0, _root)

from demo.shared import VADUG, PersonalityVector
from demo.pendulum_v2 import PendulumV2
from demo.intent import IntentDetector
from demo.tonal import TonalAnalyzer
from demo.response import compute_harmony
from demo.grader import SentenceGrader
from demo.classifier import AdaptiveClassifier

# ── Personality presets ──────────────────────────────────────────────

PRESETS = {
    "Default": PersonalityVector(),
    "Therapist": PersonalityVector(
        gullibility=80, agreeableness=200, suggestibility=60,
        truthfulness=220, safety=230, curiosity=180,
        assertiveness=80, playfulness=60,
    ),
    "Best Friend": PersonalityVector(
        gullibility=140, agreeableness=200, suggestibility=120,
        truthfulness=180, safety=160, curiosity=180,
        assertiveness=120, playfulness=200,
    ),
    "Drill Sergeant": PersonalityVector(
        gullibility=20, agreeableness=20, suggestibility=10,
        truthfulness=250, safety=140, curiosity=60,
        assertiveness=250, playfulness=10,
    ),
    "Stoic": PersonalityVector(
        gullibility=30, agreeableness=100, suggestibility=20,
        truthfulness=200, safety=180, curiosity=100,
        assertiveness=140, playfulness=20,
    ),
    "Empath": PersonalityVector(
        gullibility=160, agreeableness=240, suggestibility=180,
        truthfulness=200, safety=220, curiosity=200,
        assertiveness=60, playfulness=80,
    ),
    "Chaos Goblin": PersonalityVector(
        gullibility=180, agreeableness=60, suggestibility=160,
        truthfulness=100, safety=80, curiosity=240,
        assertiveness=160, playfulness=250,
    ),
}

# ── Pipeline components (stateless singletons) ──────────────────────

intent_detector = IntentDetector()
tonal_analyzer = TonalAnalyzer()
grader = SentenceGrader()
classifier = AdaptiveClassifier()

# ── VADUG bar helper ────────────────────────────────────────────────


def _vadug_bar(label: str, value: int, color_low: str, color_high: str) -> str:
    """Build an HTML progress bar for a single VADUG axis."""
    pct = round(value / 255 * 100)
    # Interpolate color: low color when value is 0, high color when 255
    return (
        f'<div style="margin:2px 0">'
        f'<span style="display:inline-block;width:80px;font-weight:bold">{label}: {value}</span>'
        f'<div style="display:inline-block;width:200px;height:16px;background:#2a2a2a;'
        f'border-radius:4px;vertical-align:middle">'
        f'<div style="width:{pct}%;height:100%;background:linear-gradient(90deg,{color_low},{color_high});'
        f'border-radius:4px"></div>'
        f'</div></div>'
    )


def _vadug_bars(vadug: VADUG) -> str:
    """Build visual bar chart for all 5 VADUG axes."""
    bars = [
        _vadug_bar("V (Valence)", vadug.v, "#d32f2f", "#4caf50"),
        _vadug_bar("A (Arousal)", vadug.a, "#42a5f5", "#ff9800"),
        _vadug_bar("D (Dominance)", vadug.d, "#9e9e9e", "#7b1fa2"),
        _vadug_bar("U (Urgency)", vadug.u, "#ffffff", "#d32f2f"),
        _vadug_bar("G (Gravity)", vadug.g, "#263238", "#e0e0e0"),
    ]
    return "\n".join(bars)


# ── Main processing function ────────────────────────────────────────


def process_message(text: str, preset_name: str, chat_history: list):
    """Run the full Clanker pipeline and return chat + debug output."""
    if not text or not text.strip():
        return chat_history or [], "*Type a message to see the pipeline in action...*", ""

    personality = PRESETS.get(preset_name, PRESETS["Default"])

    # Layer 0: Intent detection
    intent_mode, intent_conf = intent_detector.detect(text)
    processing_path = intent_detector.get_processing_path(intent_mode)

    # PendulumV2 processing (3-pass: pre-pass, word-pass, post-pass)
    engine = PendulumV2()
    input_vadug, trace = engine.process_text(text)

    # Tonal analysis (pass 2 — uses word trace from V2)
    tonal = tonal_analyzer.analyze(trace, intent_mode=intent_mode)

    # Adaptive classification
    sentiment = classifier.classify_with_entropy(input_vadug.v, intent_mode, trace)

    # Response VADUG via harmony computation
    response_vadug = compute_harmony(input_vadug, personality)

    # Grading
    chunk = [{"vadug": input_vadug}]
    grade, grade_rules = grader.compute_grade(chunk)

    # ── Build debug markdown ────────────────────────────────────

    debug = f"""## 1. Intent Detection
```
Mode: {intent_mode} ({intent_conf:.2f})
Processing path: {'full pipeline' if processing_path.get('pendulum') else processing_path.get('response', 'short-circuit')}
Pendulum: {'ON' if processing_path.get('pendulum') else 'OFF'}  |  Chunker: {'ON' if processing_path.get('chunker') else 'OFF'}  |  Grader: {'ON' if processing_path.get('grader') else 'OFF'}
```

## 2. Word-by-Word Pendulum Trace (V2 Engine)
| Word | Role | V | A | D | U | G | Note |
|------|------|---|---|---|---|---|------|
"""
    for step in trace:
        w = step.get("word", "?")
        role = step.get("role", "?")
        sv = step.get("v", 128)
        sa = step.get("a", 128)
        sd = step.get("d", 128)
        su = step.get("u", 0)
        sg = step.get("g", 128)
        note = step.get("note", "")
        # Truncate long note strings for table readability
        if len(note) > 50:
            note = note[:47] + "..."
        debug += f'| "{w}" | {role} | {sv} | {sa} | {sd} | {su} | {sg} | {note} |\n'

    debug += f"""
## 3. Tonal Analysis
```
Tone: {tonal['tone']} ({tonal['confidence']:.2f})
Signals: {'; '.join(tonal.get('signals', [])) or 'none'}
```

## 4. VADUG Result

{_vadug_bars(input_vadug)}

**Sentiment:** {sentiment} | **Emotional state:** {input_vadug.describe()}

## 5. Grade + Response VADUG
```
Grade: {grade} ({grade_rules.get('desc', '')})
Tone required: {grade_rules.get('tone', '?')}
Personality: {preset_name}
Response VADUG: V={response_vadug.v} A={response_vadug.a} D={response_vadug.d} U={response_vadug.u} G={response_vadug.g}
```
"""

    # ── Generate a placeholder response from the math ───────────
    # (No LLM — this is pure pipeline output driving a template)

    if grade.startswith("F"):
        response = "I'm here. You're not alone."
    elif sentiment == "positive":
        response = f"That's great to hear! [{grade} | V={response_vadug.v} G={response_vadug.g}]"
    elif sentiment == "negative":
        response = f"I hear you. That sounds tough. [{grade} | V={response_vadug.v} G={response_vadug.g}]"
    else:
        response = f"Got it. [{grade} | V={response_vadug.v} G={response_vadug.g}]"

    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": text})
    chat_history.append({"role": "assistant", "content": response})

    return chat_history, debug, ""


# ── Build the Gradio app ────────────────────────────────────────────

with gr.Blocks(
    title="Clanker-Lang Emotional Physics Engine",
) as app:
    gr.Markdown(
        "# Clanker-Lang Emotional Physics Engine\n"
        "*Glass-box emotional AI -- every number is auditable. "
        "No hidden layers. No black boxes.*"
    )

    with gr.Row():
        preset = gr.Dropdown(
            choices=list(PRESETS.keys()),
            value="Therapist",
            label="Personality Preset",
        )

    with gr.Row():
        # Left panel: Chat
        with gr.Column(scale=1):
            gr.Markdown("### Chat")
            chatbot = gr.Chatbot(height=500, label="Conversation")
            msg = gr.Textbox(
                placeholder="Type a message...",
                label="",
                show_label=False,
            )
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")

        # Right panel: Debug internals
        with gr.Column(scale=1):
            gr.Markdown("### Debug Internals (Glass Box)")
            debug_output = gr.Markdown(
                value="*Send a message to see the emotional processing pipeline...*",
            )

    def respond(message, preset_name, history):
        history, debug, cleared = process_message(message, preset_name, history)
        return history, debug, cleared

    send.click(respond, [msg, preset, chatbot], [chatbot, debug_output, msg])
    msg.submit(respond, [msg, preset, chatbot], [chatbot, debug_output, msg])
    clear.click(
        lambda: ([], "*Send a message to see the emotional processing pipeline...*"),
        outputs=[chatbot, debug_output],
    )

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7865))
    app.launch(server_name="0.0.0.0", server_port=port, ssr_mode=False)
