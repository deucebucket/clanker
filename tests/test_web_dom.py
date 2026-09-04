from __future__ import annotations

import json
import re

import pytest
from starlette.testclient import TestClient

from clanker_lm import web as web_module
from clanker_lm.web import WebConfig, create_app


def test_changelog_wiring_executes_in_a_real_browser_dom() -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    build_commit = "b" * 40
    app = create_app(config=WebConfig(build_commit=build_commit))
    with TestClient(app, base_url="http://localhost:8765") as client:
        html = client.get("/").text
        script = client.get("/assets/app.js").text
        styles = client.get("/assets/app.css").text
        payload = client.get("/api/releases").json()

    marker = '<img src=x onerror="window.privateTranscript=1">'
    payload["releases"][0]["title"] = marker
    payload["releases"][0]["deployment"]["label"] = "Pending · forged metadata"
    payload["releases"][1]["deployment"]["label"] = "Live · forged metadata"
    html = re.sub(r"<link\b[^>]*app\.css[^>]*>", "", html)
    html = re.sub(r"<script\b[^>]*app\.js[^>]*></script>", "", html)

    with playwright.sync_playwright() as runtime:
        try:
            browser = runtime.chromium.launch()
        except playwright.Error as exc:
            pytest.skip(f"Chromium browser binary is unavailable: {exc}")
        try:
            page = browser.new_page(viewport={"width": 360, "height": 800})
            page.set_content(html)
            page.add_style_tag(content=styles)
            page.evaluate(
                "feed => { window.fetch = async () => ({ ok: true, json: async () => feed }); }",
                payload,
            )
            page.add_script_tag(content=script)
            page.get_by_role("button", name="Changelog").click()
            page.get_by_role("heading", name=marker).wait_for()

            assert page.locator("#deployed-build-commit").text_content() == build_commit
            assert page.locator(".release-identity code").first.text_content() == (
                "9ae77f072f8afda0b1d2b757ab492757cabff0f8"
            )
            assert page.locator("#release-list img").count() == 0
            assert page.locator("#release-list").evaluate("node => node.scrollWidth") <= 360
            assert page.locator("#changelog-status").text_content() == (
                "2 reviewed releases: 1 current live, 1 pending, 0 history."
            )
            assert page.locator(".release-card--live").count() == 1
            assert page.locator(".release-card--live").get_attribute("aria-current") == "true"
            assert page.locator(".release-card--live .deployment-badge").text_content() == (
                "Live · private Tailnet"
            )
            assert page.locator(".release-card--pending").count() == 1
            assert page.locator(".release-card--pending h4").first.text_content() == (
                "What passed review"
            )
            assert page.locator(".deployment-badge--pending").text_content() == (
                "Pending · live verification"
            )
            assert "forged metadata" not in page.locator("#release-list").text_content()
            assert page.locator("#deployed-state").text_content() == (
                "Live · private Tailnet"
            )
            assert page.locator(".release-card--pending").get_by_role(
                "link", name="Open current live baseline"
            ).count() == 1
            assert page.locator("#changelog-close").evaluate(
                "node => node === document.activeElement"
            )
            page.keyboard.press("Escape")
            assert page.locator("#changelog-open").evaluate(
                "node => node === document.activeElement"
            )
        finally:
            browser.close()
