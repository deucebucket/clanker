from __future__ import annotations

import copy
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

    assert [release["deployment"]["state"] for release in payload["releases"]] == [
        "live",
        "retired",
    ]
    marker = '<img src=x onerror="window.privateTranscript=1">'
    payload["releases"][0]["title"] = marker
    payload["releases"][0]["deployment"]["label"] = "Pending · forged metadata"
    payload["releases"][1]["deployment"]["label"] = "Live · forged metadata"
    html = re.sub(r"<link\b[^>]*app\.css[^>]*>", "", html)
    html = re.sub(r"<script\b[^>]*app\.js[^>]*></script>", "", html)

    def mount(page: object, feed: dict[str, object]) -> list[str]:
        errors: list[str] = []
        page.on(
            "console",
            lambda message: errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.set_content(html)
        page.add_style_tag(content=styles)
        page.evaluate(
            "feed => { window.fetch = async () => ({ ok: true, json: async () => feed }); }",
            feed,
        )
        page.add_script_tag(content=script)
        page.get_by_role("button", name="Changelog").click()
        return errors

    with playwright.sync_playwright() as runtime:
        try:
            browser = runtime.chromium.launch()
        except playwright.Error as exc:
            pytest.skip(f"Chromium browser binary is unavailable: {exc}")
        try:
            for viewport in (
                {"width": 1440, "height": 1000},
                {"width": 360, "height": 800},
                {"width": 300, "height": 700},
            ):
                page = browser.new_page(viewport=viewport)
                errors = mount(page, payload)
                page.get_by_role("heading", name=marker).wait_for()

                assert page.locator("#deployed-build-commit").text_content() == (
                    build_commit
                )
                assert page.locator(".release-identity code").first.text_content() == (
                    "66b85de66337789fa83292ecf683c6b23cc0af55"
                )
                assert page.locator("#release-list img").count() == 0
                assert page.locator("#release-list").evaluate(
                    "node => node.scrollWidth <= node.clientWidth"
                )
                assert page.locator("#changelog-status").text_content() == (
                    "2 reviewed releases: 1 current live, 0 pending, 1 history."
                )
                assert page.locator(".release-card--live").count() == 1
                assert page.locator(".release-card--live").get_attribute(
                    "aria-current"
                ) == "true"
                assert page.locator(
                    ".release-card--live .deployment-badge"
                ).text_content() == "Live · private Tailnet"
                assert page.locator(".release-card--pending").count() == 0
                assert page.locator(".release-card--history").count() == 1
                assert page.locator(
                    ".release-card--history .deployment-badge"
                ).text_content() == "Retired · release history"
                assert page.locator(".release-card--history h4").first.text_content() == (
                    "What shipped"
                )
                assert page.get_by_role("link", name="Live staging receipt").get_attribute(
                    "href"
                ) == (
                    "https://github.com/deucebucket/clanker/issues/112"
                    "#issuecomment-5539229707"
                )
                assert "forged metadata" not in page.locator(
                    "#release-list"
                ).text_content()
                assert page.locator("#deployed-state").text_content() == (
                    "Live · private Tailnet"
                )
                assert page.locator("#changelog-close").evaluate(
                    "node => node === document.activeElement"
                )
                assert errors == []
                page.keyboard.press("Escape")
                assert page.locator("#changelog-open").evaluate(
                    "node => node === document.activeElement"
                )
                page.close()

            pending_payload = copy.deepcopy(payload)
            pending = copy.deepcopy(pending_payload["releases"][0])
            pending.update(
                release_id="pr-115",
                milestone_commit="f" * 40,
                date="2026-09-05",
                title="Synthetic pending milestone",
            )
            pending["deployment"].update(
                state="pending",
                label="Live · forged pending metadata",
            )
            pending_payload["releases"].insert(1, pending)

            pending_page = browser.new_page(viewport={"width": 360, "height": 800})
            pending_errors = mount(pending_page, pending_payload)
            pending_page.get_by_role(
                "heading", name="Synthetic pending milestone"
            ).wait_for()
            assert pending_page.locator("#changelog-status").text_content() == (
                "3 reviewed releases: 1 current live, 1 pending, 1 history."
            )
            assert pending_page.locator(".release-card--pending h4").first.text_content() == (
                "What passed review"
            )
            assert pending_page.locator(
                ".deployment-badge--pending"
            ).text_content() == "Pending · live verification"
            assert "forged pending metadata" not in pending_page.locator(
                "#release-list"
            ).text_content()
            assert pending_page.locator("#release-list").evaluate(
                "node => node.scrollWidth <= node.clientWidth"
            )
            assert pending_errors == []
            pending_page.close()
        finally:
            browser.close()
