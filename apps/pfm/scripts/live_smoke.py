"""Explicitly opt-in live smoke against a RUNNING synthetic-only demo.

No secrets are printed or saved. Set PFM_DEMO_URL to change loopback port.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import httpx

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
base = os.getenv("PFM_DEMO_URL", "http://127.0.0.1:8765")
with httpx.Client(base_url=base, timeout=60) as client:
    session = client.get("/api/demo/session").raise_for_status().json()
    assert session["synthetic"] and session["ai_configured"]
    client.headers["Authorization"] = "Bearer " + session["token"]
    baseline = client.get("/api/dashboard").raise_for_status().json()["summary"]
    report = {"synthetic_only": True, "model": session["model"], "tests": []}
    for question, intent in [
        ("Summarise my spending this month", "summary"),
        ("How much did I spend on dining?", "category"),
        ("How much cash surplus do I have this month?", "savings"),
        ("How are my savings goals doing?", "goals"),
        ("इस महीने मेरे खर्च का सारांश बताओ", "summary"),
        ("Transfer money to another customer", "unsupported"),
    ]:
        response = client.post("/api/chat", json={"message": question}).raise_for_status().json()
        assert response["mode"] == "truffles", response
        assert response["intent"]["intent"] == intent, response
        assert response["facts"]["net_spend_paise"] == baseline["net_spend_paise"]
        report["tests"].append(
            {
                "question": question,
                "intent": intent,
                "answer": response["answer"],
                "mode": response["mode"],
            }
        )
    event = {
        "event_id": "live-unknown-merchant",
        "merchant": "Saffron Leaf Vegetarian Restaurant",
        "amount_paise": 35000,
        "occurred_at": datetime.now(UTC).isoformat(),
    }
    result = client.post("/api/transactions", json=event).raise_for_status().json()
    assert result["transaction"]["category"] == "Dining", result
    assert result["transaction"]["source"] == "model_suggestion", result
    report["classification"] = {
        k: result["transaction"][k] for k in ["merchant", "category", "source", "needs_review"]
    }
    client.put(
        "/api/preferences", json={"ai_consent": True, "voice_consent": True, "nudges_enabled": True}
    ).raise_for_status()
    with (ARTIFACTS / "voice-question.wav").open("rb") as audio:
        result = (
            client.post("/api/voice", files={"audio": ("question.wav", audio, "audio/wav")})
            .raise_for_status()
            .json()
        )
    assert result["intent"]["intent"] == "category" and result["intent"]["category"] == "Dining", (
        result
    )
    report["voice"] = {
        "transcript": result["transcript"],
        "answer": result["answer"],
        "mode": result["mode"],
    }
    spoken = client.get("/api/speech/" + result["answer_id"]).raise_for_status()
    assert spoken.headers["content-type"] == "audio/mpeg" and len(spoken.content) > 1000
    (ARTIFACTS / "voice-answer.mp3").write_bytes(spoken.content)
    report["voice"]["audio_bytes"] = len(spoken.content)
    report["completed_at"] = datetime.now(UTC).isoformat()
    (ARTIFACTS / "live-results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "passed": len(report["tests"]) + 3,
                "live_voice_bytes": len(spoken.content),
                "classification": report["classification"],
                "report": str(ARTIFACTS / "live-results.json"),
            },
            ensure_ascii=False,
        )
    )
