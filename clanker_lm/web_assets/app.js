"use strict";

const composer = document.querySelector("#composer");
const message = document.querySelector("#message");
const send = document.querySelector("#send");
const transcript = document.querySelector("#transcript");
const formStatus = document.querySelector("#form-status");
const messageCount = document.querySelector("#message-count");
const reset = document.querySelector("#reset");
const exportLink = document.querySelector("#export");
const changelogOpen = document.querySelector("#changelog-open");
const changelogDialog = document.querySelector("#changelog-dialog");
const changelogClose = document.querySelector("#changelog-close");
const changelogRetry = document.querySelector("#changelog-retry");
const changelogStatus = document.querySelector("#changelog-status");
const releaseList = document.querySelector("#release-list");
const latestReleaseLabel = document.querySelector("#latest-release-label");
const deployedVersion = document.querySelector("#deployed-version");
const deployedBuildCommit = document.querySelector("#deployed-build-commit");
const deployedState = document.querySelector("#deployed-state");
const encoder = new TextEncoder();
let requestInFlight = false;
let releaseFeedPromise = null;

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

function setChangelogStatus(text, isError = false) {
  changelogStatus.textContent = text;
  changelogStatus.classList.toggle("is-error", isError);
}

function appendTextList(parent, items, className) {
  const list = element("ul", className);
  items.forEach((item) => list.append(element("li", "", item)));
  parent.append(list);
}

function repositoryEvidenceLink(record) {
  const stagingReceiptUrl = "https://github.com/deucebucket/clanker/issues/112#issuecomment-5539229707";
  const url = new URL(record.url);
  const allowedPath = /^\/deucebucket\/clanker\/(pull\/\d+|commit\/[0-9a-f]{40}|actions\/runs\/\d+)$/;
  const isStandardEvidence = url.protocol === "https:"
    && url.host === "github.com"
    && allowedPath.test(url.pathname)
    && !url.search
    && !url.hash;
  if (record.url !== stagingReceiptUrl && !isStandardEvidence) {
    throw new Error("Release evidence is outside the repository boundary.");
  }
  const link = element("a", "release-link", record.label);
  link.href = url.href;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}

function deploymentLink(deployment, linkLabel) {
  if (deployment.url !== "https://bazzite.tail85f65f.ts.net:8444/") {
    throw new Error("Release deployment is outside the pinned workbench.");
  }
  const link = element("a", "release-link", linkLabel);
  link.href = deployment.url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}

const lifecyclePresentation = Object.freeze({
  live: Object.freeze({
    cardClass: "release-card--live",
    badgeClass: "deployment-badge--live",
    badgeLabel: "Live · private Tailnet",
    capabilityHeading: "What is live",
    deploymentLinkLabel: "Open live workbench",
  }),
  pending: Object.freeze({
    cardClass: "release-card--pending",
    badgeClass: "deployment-badge--pending",
    badgeLabel: "Pending · live verification",
    capabilityHeading: "What passed review",
    deploymentLinkLabel: "Open current live baseline",
  }),
  retired: Object.freeze({
    cardClass: "release-card--history",
    badgeClass: "deployment-badge--history",
    badgeLabel: "Retired · release history",
    capabilityHeading: "What shipped",
    deploymentLinkLabel: "Open current live workbench",
  }),
  rolled_back: Object.freeze({
    cardClass: "release-card--history",
    badgeClass: "deployment-badge--history",
    badgeLabel: "Rolled back · release history",
    capabilityHeading: "What shipped before rollback",
    deploymentLinkLabel: "Open current live workbench",
  }),
});

function renderRelease(release) {
  const presentation = lifecyclePresentation[release.deployment.state];
  if (!presentation) throw new Error("Release lifecycle state is unsupported.");
  const item = element("li", "release-item");
  const article = element("article", `release-card ${presentation.cardClass}`);
  if (release.deployment.state === "live") article.setAttribute("aria-current", "true");
  const heading = element("header", "release-card__heading");
  const titleBlock = element("div", "");
  const marker = element("p", "release-marker", release.release_id);
  const title = element("h3", "", release.title);
  titleBlock.append(marker, title);

  const dateNode = element("time", "release-date", release.date);
  dateNode.dateTime = release.date;
  heading.append(titleBlock, dateNode);

  const identity = element("dl", "release-identity");
  const versionField = element("div", "");
  versionField.append(element("dt", "", "Milestone package"), element("dd", "", release.package_version));
  const commitField = element("div", "");
  const commitValue = element("dd", "");
  commitValue.append(element("code", "", release.milestone_commit));
  commitField.append(element("dt", "", "Milestone commit"), commitValue);
  identity.append(versionField, commitField);

  const capabilitySection = element("section", "release-section");
  capabilitySection.append(element("h4", "", presentation.capabilityHeading));
  appendTextList(capabilitySection, release.capabilities, "release-points");

  const evidenceSection = element("section", "release-section");
  evidenceSection.append(element("h4", "", "Evidence"));
  const evidenceList = element("ul", "release-links");
  release.evidence.forEach((record) => {
    const evidenceItem = element("li", "");
    evidenceItem.append(repositoryEvidenceLink(record));
    evidenceList.append(evidenceItem);
  });
  evidenceSection.append(evidenceList);

  const limitationSection = element("section", "release-section release-section--limits");
  limitationSection.append(element("h4", "", "Known limits"));
  appendTextList(limitationSection, release.limitations, "release-points");

  const deployment = element("footer", "release-deployment");
  deployment.append(
    element("span", `deployment-badge ${presentation.badgeClass}`, presentation.badgeLabel),
    element("p", "", release.deployment.detail),
    deploymentLink(release.deployment, presentation.deploymentLinkLabel),
  );

  article.append(heading, identity, capabilitySection, evidenceSection, limitationSection, deployment);
  item.append(article);
  return item;
}

function renderReleaseFeed(feed) {
  if (
    !feed
    || !Array.isArray(feed.releases)
    || feed.releases.length === 0
    || !feed.latest_shipped_release
    || typeof feed.running_package_version !== "string"
    || !/^[0-9a-f]{40}$/.test(feed.deployed_build_commit)
  ) {
    throw new Error("The release feed is empty or malformed.");
  }
  const current = feed.releases[0];
  const currentPresentation = lifecyclePresentation[current.deployment && current.deployment.state];
  const states = feed.releases.map((release) => release.deployment && release.deployment.state);
  const liveCount = states.filter((state) => state === "live").length;
  const pendingCount = states.filter((state) => state === "pending").length;
  const historyCount = states.filter((state) => state === "retired" || state === "rolled_back").length;
  if (
    current.release_id !== feed.latest_shipped_release.release_id
    || current.package_version !== feed.latest_shipped_release.package_version
    || current.milestone_commit !== feed.latest_shipped_release.milestone_commit
    || current.package_version !== feed.running_package_version
    || current.deployment.state !== "live"
    || !currentPresentation
    || liveCount !== 1
    || liveCount + pendingCount + historyCount !== feed.releases.length
  ) {
    throw new Error("The displayed release does not match the deployed identity.");
  }

  releaseList.replaceChildren(...feed.releases.map(renderRelease));
  const shortCommit = current.milestone_commit.slice(0, 7);
  latestReleaseLabel.textContent = `v${current.package_version} · ${shortCommit}`;
  deployedVersion.textContent = `v${feed.running_package_version}`;
  deployedBuildCommit.textContent = feed.deployed_build_commit;
  deployedState.textContent = currentPresentation.badgeLabel;
  releaseList.setAttribute("aria-busy", "false");
  changelogRetry.hidden = true;
  setChangelogStatus(`${feed.releases.length} reviewed release${feed.releases.length === 1 ? "" : "s"}: ${liveCount} current live, ${pendingCount} pending, ${historyCount} history.`);
}

async function loadReleaseFeed() {
  releaseList.setAttribute("aria-busy", "true");
  changelogRetry.hidden = true;
  setChangelogStatus("Loading reviewed releases…");
  try {
    const feed = await requestJson("/api/releases", {
      method: "GET",
      credentials: "same-origin",
    });
    renderReleaseFeed(feed);
  } catch (_error) {
    releaseFeedPromise = null;
    releaseList.setAttribute("aria-busy", "false");
    changelogRetry.hidden = false;
    setChangelogStatus("The reviewed release record could not be loaded.", true);
  }
}

function ensureReleaseFeed() {
  if (!releaseFeedPromise) releaseFeedPromise = loadReleaseFeed();
  return releaseFeedPromise;
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

changelogOpen.addEventListener("click", () => {
  if (!changelogDialog.open) changelogDialog.showModal();
  changelogClose.focus();
  void ensureReleaseFeed();
});

changelogClose.addEventListener("click", () => changelogDialog.close());
changelogRetry.addEventListener("click", () => { void ensureReleaseFeed(); });
changelogDialog.addEventListener("click", (event) => {
  if (event.target === changelogDialog) changelogDialog.close();
});
changelogDialog.addEventListener("close", () => changelogOpen.focus());

updateMessageState();
