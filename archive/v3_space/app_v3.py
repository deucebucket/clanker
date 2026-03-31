#!/usr/bin/env python3
"""Clanker V3 HuggingFace Space -- emotional physics engine.

Tab 1: Score any text (VADUG + structures + word roles)
Tab 2: Two AI characters argue (SmolLM2 + Clanker scoring)

How would this land on the receiving end?
"""

import gradio as gr
import sys
import os
import json

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here if os.path.isdir(os.path.join(_here, "engine")) else os.path.dirname(_here)
sys.path.insert(0, _root)

from engine.pendulum import compute_vadug
from engine.word_classifier import classify_sentence


# ── Helpers ──

def emotion_label(v):
    if v > 200: return "elated"
    if v > 170: return "joyful"
    if v > 150: return "happy"
    if v > 140: return "content"
    if v > 132: return "okay"
    if v > 124: return "neutral"
    if v > 115: return "uneasy"
    if v > 105: return "hurt"
    if v > 90: return "distressed"
    if v > 70: return "anguished"
    if v > 50: return "devastated"
    return "shattered"


def dim_label(val, dim):
    n = (val - 128) / 128
    if dim == "v":
        if n > 0.15: return "positive"
        if n > -0.15: return "neutral"
        return "negative"
    if dim == "a":
        if n > 0.15: return "alert"
        if n > -0.15: return "calm"
        return "subdued"
    if dim == "d":
        if n > 0.15: return "confident"
        if n > -0.15: return "neutral"
        return "yielding"
    if dim == "g":
        if n > 0.15: return "important"
        if n > -0.15: return "moderate"
        return "minor"
    return ""


# ── Tab 1: Score Text ──

def score_text(text):
    if not text or not text.strip():
        return "", "", "", ""

    vadug, trace = compute_vadug(text.strip())
    structures = trace["structures"]

    # VADUG display
    emo = emotion_label(vadug.v)
    vadug_text = f"""## {emo.upper()}

| Dimension | Value | Reading |
|-----------|-------|---------|
| **V** Valence | {vadug.v} | {dim_label(vadug.v, 'v')} |
| **A** Arousal | {vadug.a} | {dim_label(vadug.a, 'a')} |
| **D** Dominance | {vadug.d} | {dim_label(vadug.d, 'd')} |
| **U** Urgency | {vadug.u} | {'urgent' if vadug.u > 30 else 'low'} |
| **G** Gravity | {vadug.g} | {dim_label(vadug.g, 'g')} |
"""

    # Structures
    if structures:
        struct_text = "\n".join(
            f"- **{s.pattern}** ({s.confidence:.0%}) -- {s.description}"
            for s in structures
        )
    else:
        struct_text = "No structural patterns detected."

    # Word roles trace
    roles_text = "| # | Word | Role | Coeff | V |\n|---|------|------|-------|---|\n"
    for entry in trace["trace"]:
        roles_text += f"| {entry.get('word', '')} | {entry['role']} | {entry['coeff']:+.3f} | {entry['v']} |\n"

    # Word chips (colored)
    words = text.strip().split()
    roles = classify_sentence(words)
    chips = " ".join(f"`{r.word}:{r.role}`" for r in roles)

    return vadug_text, struct_text, roles_text, chips


# ── Tab 2: Conversation (without SmolLM2 for Space -- uses templates) ──

PERSONALITIES = {
    "Hothead": "Short temper, explosive, says things they regret. Swears when angry.",
    "Peacekeeper": "Avoids conflict, apologizes too much. Quietly resentful underneath.",
    "Ice": "Emotionally guarded, short sentences, uses silence as a weapon.",
    "Empath": "Feels everything deeply, cries easily, takes things personally.",
    "Joker": "Uses humor to deflect, never serious, afraid of vulnerability.",
}

TOPICS = [
    "You found out your partner lied about where they were last night.",
    "Your roommate keeps eating your food without asking.",
    "One of you forgot the other's birthday.",
    "One of you said something hurtful and never apologized.",
    "Your best friend is dating your ex.",
]


def run_conversation(pers_a, pers_b, topic, num_turns):
    """Run a conversation and return the scored log."""
    if not topic:
        return "Pick a topic first."

    # Try to load SmolLM2 if available
    try:
        from llama_cpp import Llama
        model_path = os.path.expanduser("~/Downloads/SmolLM2-1.7B-Instruct.F16.gguf")
        if os.path.exists(model_path):
            llm = Llama(model_path=model_path, n_ctx=512, n_gpu_layers=-1, verbose=False)
            return _run_with_llm(llm, pers_a, pers_b, topic, int(num_turns))
    except Exception:
        pass

    return "SmolLM2 not available. Run locally with: python3 demo/conversation_loop.py"


def _run_with_llm(llm, pers_a, pers_b, topic, turns):
    desc_a = PERSONALITIES.get(pers_a, "Normal person.")
    desc_b = PERSONALITIES.get(pers_b, "Normal person.")

    state_a = {"v": 128, "a": 128, "d": 128}
    state_b = {"v": 128, "a": 128, "d": 128}

    log = f"## {pers_a} vs {pers_b}\n**Topic:** {topic}\n\n"
    last = topic

    for turn in range(turns):
        if turn % 2 == 0:
            name, desc, state, listener = pers_a, desc_a, state_a, state_b
        else:
            name, desc, state, listener = pers_b, desc_b, state_b, state_a

        emo = emotion_label(state["v"])
        sys_prompt = (
            f"You are a character called {name}. {desc} "
            f"You feel {emo} right now. "
            f"Topic: {topic}. "
            f"Respond in 1 short sentence. No quotes. No narration."
        )

        r = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": last},
            ],
            max_tokens=50, temperature=0.85, repeat_penalty=1.15,
        )
        reply = r["choices"][0]["message"]["content"].strip().split("\n")[0]

        vadug, trace = compute_vadug(reply)
        structs = [s.pattern for s in trace["structures"]]

        listener["v"] = int(listener["v"] * 0.65 + vadug.v * 0.35)
        listener["a"] = int(listener["a"] * 0.7 + vadug.a * 0.3)

        emo_tag = emotion_label(vadug.v)
        struct_tag = f" [{', '.join(structs)}]" if structs else ""
        log += f"**{name}:** {reply}\n"
        log += f"*V={vadug.v} {emo_tag}{struct_tag}*\n\n"
        last = reply

    log += f"---\n**Final:** {pers_a} V={state_a['v']} ({emotion_label(state_a['v'])}) | {pers_b} V={state_b['v']} ({emotion_label(state_b['v'])})"
    return log


# ── Build App ──

with gr.Blocks(title="Clanker", theme=gr.themes.Base(primary_hue="gray")) as demo:
    gr.Markdown("# Clanker\n*How would this land on the receiving end?*\n\n5D emotional physics engine. 300KB. 0.15ms/sentence. 92% accuracy on unambiguous text. Zero false positives.")

    with gr.Tab("Score Text"):
        gr.Markdown("Type anything. See how it reads emotionally.")
        text_input = gr.Textbox(label="Say something", placeholder="type here...", lines=2)
        score_btn = gr.Button("Score")
        with gr.Row():
            vadug_out = gr.Markdown(label="VADUG")
            struct_out = gr.Markdown(label="Structures")
        with gr.Row():
            trace_out = gr.Markdown(label="Word Trace")
            chips_out = gr.Markdown(label="Word Roles")
        score_btn.click(score_text, inputs=[text_input], outputs=[vadug_out, struct_out, trace_out, chips_out])
        text_input.submit(score_text, inputs=[text_input], outputs=[vadug_out, struct_out, trace_out, chips_out])

        gr.Examples(
            examples=[
                "I love my mom",
                "my wife cheated on me with my best friend",
                "haha yeah im totally okay",
                "oh joy",
                "why do you always do that",
                "do you even love me",
                "she left me on read for three days",
                "give me a break",
                "whatever",
            ],
            inputs=text_input,
        )

    with gr.Tab("AI Conversation"):
        gr.Markdown("Two AI characters argue. SmolLM2 speaks. Clanker scores the impact.")
        with gr.Row():
            char_a = gr.Dropdown(list(PERSONALITIES.keys()), value="Hothead", label="Character A")
            char_b = gr.Dropdown(list(PERSONALITIES.keys()), value="Ice", label="Character B")
        topic_input = gr.Dropdown(TOPICS, value=TOPICS[4], label="Topic")
        turns_slider = gr.Slider(2, 10, value=6, step=1, label="Turns")
        convo_btn = gr.Button("Start Conversation")
        convo_out = gr.Markdown(label="Conversation")
        convo_btn.click(run_conversation, inputs=[char_a, char_b, topic_input, turns_slider], outputs=[convo_out])

demo.launch()
