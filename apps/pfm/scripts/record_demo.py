"""Record the real browser + API demo. No mocked routes or prerecorded UI responses.

Requires a running synthetic demo, playwright chromium, ffmpeg, and macOS `say` for
narration. Live inference uses only the running app's Truffles integrations.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

OUT = Path(__file__).resolve().parents[1] / "artifacts"
RAW = OUT / "raw-video"
RAW.mkdir(parents=True, exist_ok=True)
BASE = os.getenv("PFM_DEMO_URL", "http://127.0.0.1:8765")
SCENES = [
    (
        "IB3 / Conversational personal finance",
        "This is Truffles, a personal finance assistant for the I B three banking "
        "use case. Everything shown here uses synthetic Indian banking data.",
    ),
    (
        "Chat grounded in ledger facts",
        "Ask for a monthly summary. Truffles interprets the question with OpenAI, "
        "while ordinary application code calculates every financial amount from "
        "the ledger.",
    ),
    (
        "Incoming payment → category → budget nudge",
        "A new Swiggy payment arrives. The service categorises it automatically, "
        "updates the ledger, and creates a budget nudge. Common merchants need no "
        "model request.",
    ),
    (
        "Unfamiliar merchant → reviewable AI category",
        "An unfamiliar restaurant goes through the Truffles classification agent. "
        "The suggestion is clearly marked for customer review. The model cannot "
        "change amounts or move money.",
    ),
    (
        "Customer-controlled savings goals",
        "Create a holiday goal with a target and an amount already saved. The "
        "assistant explains progress and an illustrative daily plan. Creating a "
        "goal does not transfer funds.",
    ),
    (
        "Hindi summary, same financial facts",
        "The same assistant can answer in Hindi. The numbers still come from the "
        "same scoped ledger and deterministic calculations.",
    ),
    (
        "Opt-in voice, through Truffles",
        "Voice is optional. After consent, this synthetic recording is "
        "transcribed through Truffles. The answer can then be played using an "
        "explicitly identified AI voice.",
    ),
    (
        "Auditable changes and visible fallback",
        "The activity trail records changes without raw conversations or "
        "credentials. If AI is disabled or unavailable, the interface clearly "
        "labels its local fallback.",
    ),
    (
        "Ready for bank integration review",
        "This working reference supports a bank-hosted compatible model. "
        "Production rollout still requires the bank identity gateway, managed "
        "storage, security assessment, and model qualification.",
    ),
]


def duration(path):
    return float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
        )
    )


clips = []
for index, (_, narration) in enumerate(SCENES):
    path = RAW / f"narration-{index}.aiff"
    if not path.exists():
        subprocess.run(
            ["say", "-v", "Samantha", "-r", "165", "-o", str(path), narration], check=True
        )
    clips.append((path, duration(path)))

chapters, audio_tracks, browser_errors, failures = [], [], [], []
with sync_playwright() as playwright:
    browser = playwright.chromium.launch()
    context = browser.new_context(
        viewport={"width": 1440, "height": 1000},
        record_video_dir=str(RAW),
        record_video_size={"width": 1440, "height": 1000},
    )
    page = context.new_page()
    # Capture the actual browser playback Blob for muxing; no responses are mocked.
    page.add_init_script(
        "window.__recordingBlobs = new Map(); "
        "const original = URL.createObjectURL.bind(URL); "
        "URL.createObjectURL = blob => { const url = original(blob); "
        "window.__recordingBlobs.set(url, blob); return url; };"
    )
    page.set_default_timeout(45000)
    expect.set_options(timeout=30000)
    page.on("pageerror", lambda error: browser_errors.append(str(error)))
    page.on(
        "response",
        lambda response: (
            failures.append({"path": response.url.split("?")[0], "status": response.status})
            if response.status >= 400
            else None
        ),
    )
    started = time.monotonic()

    def chapter(index):
        elapsed = time.monotonic() - started
        chapters.append({"start": round(elapsed, 3), "title": SCENES[index][0]})
        audio_tracks.append((str(clips[index][0]), elapsed))
        return time.monotonic()

    def finish(index, when):
        remaining = clips[index][1] + 2 - (time.monotonic() - when)
        if remaining > 0:
            page.wait_for_timeout(remaining * 1000)

    def ask(question):
        before = page.locator(".evidence").count()
        expect(page.get_by_role("button", name="Send question")).to_be_enabled()
        page.locator("#question").fill(question)
        expect(page.locator("#question")).to_have_value(question)
        page.locator("#question").press("Enter")
        expect(page.locator(".evidence")).to_have_count(before + 1)
        return page.locator(".evidence").last

    page.goto(BASE)
    expect(page.locator("#income")).to_contain_text("95,000")
    expect(page.locator("#model-state")).to_contain_text("gpt-4.1-mini")
    step = chapter(0)
    page.screenshot(path=str(OUT / "demo-overview.png"), full_page=True)
    finish(0, step)

    step = chapter(1)
    expect(ask("Summarise my spending this month")).to_contain_text("Truffles AI intent")
    finish(1, step)

    step = chapter(2)
    page.locator("#ledger").scroll_into_view_if_needed()
    page.locator("#merchant").fill("Swiggy")
    page.locator("#amount").fill("1450")
    page.locator("#simulate").click()
    expect(page.locator("#event-result")).to_contain_text("budget nudge created")
    page.locator(".nudge-card").scroll_into_view_if_needed()
    expect(page.locator("#nudges")).to_contain_text("80%")
    finish(2, step)

    step = chapter(3)
    page.locator("#ledger").scroll_into_view_if_needed()
    page.locator("#merchant").fill("Saffron Leaf Vegetarian Restaurant")
    page.locator("#amount").fill("350")
    page.locator("#simulate").click()
    expect(page.locator("#event-result")).to_contain_text("model suggestion")
    expect(page.locator("#transactions tr").first).to_contain_text("Dining")
    page.screenshot(path=str(OUT / "demo-categorisation.png"))
    finish(3, step)

    step = chapter(4)
    page.locator("#goal-btn").click()
    page.locator("#goal-form input[name=name]").fill("Goa holiday")
    page.locator("#goal-form input[name=target]").fill("60000")
    page.locator("#goal-form input[name=saved]").fill("15000")
    page.wait_for_timeout(1500)
    page.get_by_role("button", name="Create my goal").click()
    expect(page.locator("#goal-list")).to_contain_text("Goa holiday")
    expect(ask("How are my savings goals doing?")).to_contain_text("Truffles AI intent")
    expect(page.locator(".bubble.bot").last).to_contain_text("Goa holiday")
    page.screenshot(path=str(OUT / "demo-goals.png"))
    finish(4, step)

    step = chapter(5)
    expect(ask("इस महीने मेरे खर्च का सारांश बताओ")).to_contain_text("Truffles AI intent")
    expect(page.locator(".bubble.bot").last).to_contain_text("खर्च")
    finish(5, step)

    step = chapter(6)
    page.locator("#settings-btn").click()
    page.locator("input[name=voice_consent]").check()
    page.locator("input[name=ai_consent]").check()
    page.get_by_role("button", name="Save preferences").click()
    expect(page.locator("#settings-dialog")).not_to_be_visible()
    finish(6, step)
    voice_start = time.monotonic() - started
    audio_tracks.append((str(OUT / "voice-question.wav"), voice_start))
    before = page.locator(".evidence").count()
    page.locator("#audio-file").set_input_files(str(OUT / "voice-question.wav"))
    expect(page.locator(".evidence")).to_have_count(before + 1)
    expect(page.locator(".bubble.user").last).to_contain_text("Voice:")
    page.wait_for_timeout(
        max(0, duration(OUT / "voice-question.wav") - (time.monotonic() - started - voice_start))
        * 1000
        + 500
    )
    with page.expect_response(lambda r: "/api/speech/" in r.url and r.status == 200):
        page.locator("#listen").click()
    spoken = OUT / "demo-voice-answer.mp3"
    expect(page.locator("#voice-status")).to_contain_text("Playing an AI-generated voice")
    captured = bytes(
        page.locator("#playback").evaluate(
            "async el => Array.from(new Uint8Array("
            "await window.__recordingBlobs.get(el.src).arrayBuffer()))"
        )
    )
    assert len(captured) > 1000, "Browser playback must contain real audio"
    spoken.write_bytes(captured)
    audio_tracks.append((str(spoken), time.monotonic() - started))
    expect(page.locator("#playback")).to_be_visible()
    page.screenshot(path=str(OUT / "demo-voice.png"))
    page.wait_for_timeout((duration(spoken) + 1) * 1000)

    step = chapter(7)
    page.locator("#audit-btn").click()
    expect(page.locator("#audit-list")).to_contain_text("transaction ingested")
    page.wait_for_timeout(3500)
    page.locator("[data-close=audit-dialog]").click()
    page.locator("#settings-btn").click()
    page.locator("input[name=ai_consent]").uncheck()
    page.get_by_role("button", name="Save preferences").click()
    expect(page.locator("#settings-dialog")).not_to_be_visible()
    expect(ask("Summarise my spending")).to_contain_text("Local fallback")
    finish(7, step)

    step = chapter(8)
    # Leave the delivered demo enabled for the user; the fallback test above is explicit.
    page.locator("#settings-btn").click()
    page.locator("input[name=ai_consent]").check()
    page.get_by_role("button", name="Save preferences").click()
    expect(page.locator("#settings-dialog")).not_to_be_visible()
    page.locator('a[href="#overview"]').click()
    finish(8, step)
    assert not browser_errors, browser_errors
    assert not failures, failures
    total = time.monotonic() - started
    video_path = Path(page.video.path())
    context.close()
    browser.close()

(OUT / "demo-chapters.json").write_text(
    json.dumps(
        {
            "chapters": chapters,
            "duration_seconds": round(total, 2),
            "browser_errors": browser_errors,
            "failed_http_responses": failures,
            "synthetic_only": True,
            "audio_tracks": audio_tracks,
        },
        indent=2,
    )
)
shutil.copy(video_path, OUT / "demo-screen.webm")
# Mix narration, the synthetic spoken question, and the actual live Truffles speech response.
command = [
    "ffmpeg",
    "-hide_banner",
    "-loglevel",
    "error",
    "-y",
    "-i",
    str(OUT / "demo-screen.webm"),
]
for path, _ in audio_tracks:
    command += ["-i", path]
filters = []
for index, (_, seconds) in enumerate(audio_tracks, 1):
    filters.append(f"[{index}:a]adelay={round(seconds * 1000)}:all=1[a{index}]")
filters.append(
    "".join(f"[a{i}]" for i in range(1, len(audio_tracks) + 1))
    + f"amix=inputs={len(audio_tracks)}:normalize=0,apad[audio]"
)
command += [
    "-filter_complex",
    ";".join(filters),
    "-map",
    "0:v",
    "-map",
    "[audio]",
    "-c:v",
    "libx264",
    "-preset",
    "medium",
    "-crf",
    "20",
    "-pix_fmt",
    "yuv420p",
    "-c:a",
    "aac",
    "-b:a",
    "128k",
    "-t",
    str(total),
    "-movflags",
    "+faststart",
    str(OUT / "truffles-pfm-demo.mp4"),
]
subprocess.run(command, check=True)
print(
    json.dumps(
        {
            "video": str(OUT / "truffles-pfm-demo.mp4"),
            "seconds": round(total, 2),
            "browser_errors": browser_errors,
        }
    )
)
