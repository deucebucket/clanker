"use strict";

const composer = document.querySelector("#composer");
const message = document.querySelector("#message");
const send = document.querySelector("#send");
const transcript = document.querySelector("#transcript");
const formStatus = document.querySelector("#form-status");
const messageCount = document.querySelector("#message-count");
const reset = document.querySelector("#reset");
const exportLink = document.querySelector("#export");
const encoder = new TextEncoder();
let requestInFlight = false;

function element(name, className, text) {
  const node = document.createElement(name);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function setStatus(text, isError = false) {
  formStatus.textContent = text;
  formStatus.classList.toggle("is-error", isError);
}

function updateMessageCount() {
  const bytes = encoder.encode(message.value).length;
  messageCount.textContent = String(bytes);
  return bytes;
}

function updateMessageState() {
  const bytes = updateMessageCount();
  if (bytes > 4096) setStatus("Message exceeds the 4 KiB limit.", true);
  else if (!send.disabled) setStatus("Enter to send · Shift + Enter for a new line");
}

function addTurn(speakerName, text, evidence) {
  const article = element("article", `turn turn--${speakerName === "You" ? "user" : "assistant"}`);
  const speaker = element("p", "speaker", speakerName);
  const bubble = element("div", "bubble");
  bubble.append(element("p", "", text));
  article.append(speaker, bubble);

  if (evidence) {
    const rail = element("dl", "evidence-rail");
    const fields = [
      { label: "Answer", value: evidence.answer_status },
      { label: "Truth", value: evidence.truth },
      { label: "Source", value: evidence.source },
      { label: "Certainty", value: `${evidence.certainty} / 255` },
      { label: "Memory", value: `r${evidence.memory_revision}` },
      {
        label: "VADUG",
        value: Object.entries(evidence.vadug).map(([key, value]) => `${key.toUpperCase()}${value}`).join(" · "),
        modifier: "evidence-field--vadug",
      },
    ];
    fields.forEach(({ label, value, modifier = "" }) => {
      const field = element("div", `evidence-field${modifier ? ` ${modifier}` : ""}`);
      field.append(element("dt", "", label), element("dd", "", String(value)));
      rail.append(field);
    });
    bubble.append(rail);
  }

  transcript.append(article);
  article.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "nearest" });
}

async function requestJson(path, options) {
  const response = await fetch(path, options);
  let data = null;
  try { data = await response.json(); } catch (_error) { /* The UI shows a generic boundary below. */ }
  if (!response.ok) {
    const messageText = data && data.error && data.error.message
      ? data.error.message
      : "The request could not be completed.";
    throw new Error(messageText);
  }
  return data;
}

async function submitMessage() {
  if (requestInFlight) return;
  const text = message.value;
  const bytes = encoder.encode(text).length;
  if (!text.trim()) {
    setStatus("Write a message first.", true);
    message.focus();
    return;
  }
  if (bytes > 4096) {
    setStatus("Message exceeds the 4 KiB limit.", true);
    message.focus();
    return;
  }

  addTurn("You", text);
  message.value = "";
  updateMessageState();
  requestInFlight = true;
  send.disabled = true;
  reset.disabled = true;
  setStatus("Reasoning…");
  try {
    const data = await requestJson("/api/chat", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    addTurn("Clanker", data.response, data.evidence);
    exportLink.classList.remove("is-disabled");
    exportLink.setAttribute("aria-disabled", "false");
    setStatus("Answer added with evidence.");
  } catch (error) {
    addTurn("Workbench", "I couldn’t complete that turn. Your message was not retried.");
    setStatus(error instanceof Error ? error.message : "The request failed.", true);
  } finally {
    requestInFlight = false;
    send.disabled = false;
    reset.disabled = false;
    message.focus();
    updateMessageCount();
  }
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  void submitMessage();
});

message.addEventListener("input", updateMessageState);
message.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    void submitMessage();
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    message.value = button.dataset.prompt || "";
    updateMessageState();
    message.focus();
  });
});

reset.addEventListener("click", async () => {
  if (requestInFlight) return;
  requestInFlight = true;
  reset.disabled = true;
  send.disabled = true;
  setStatus("Resetting…");
  try {
    await requestJson("/api/reset", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    transcript.querySelectorAll(".turn:not(.turn--system)").forEach((turn) => turn.remove());
    exportLink.classList.add("is-disabled");
    exportLink.setAttribute("aria-disabled", "true");
    setStatus("Session reset. Memory is clear.");
  } catch (error) {
    setStatus(error instanceof Error ? error.message : "Reset failed.", true);
  } finally {
    requestInFlight = false;
    reset.disabled = false;
    send.disabled = false;
    message.focus();
  }
});

exportLink.addEventListener("click", (event) => {
  if (exportLink.getAttribute("aria-disabled") === "true") event.preventDefault();
});

updateMessageState();
