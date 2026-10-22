from datetime import UTC, datetime, timedelta

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from truffles_pfm.api import create_app
from truffles_pfm.config import Settings
from truffles_pfm.security import RateLimit


class SpeechStub:
    def __init__(self):
        self.inputs = []

    def transcribe(self, data, **kwargs):
        self.inputs.append(data)
        return "Summarise my spending"

    def speak(self, text):
        self.inputs.append(text)
        return b"ID3-synthetic-audio"


@pytest.fixture
def client(repo, ai):
    app = create_app(Settings(), repository=repo, intelligence=ai, speech=SpeechStub())
    with TestClient(app) as client:
        token = client.get("/api/demo/session").json()["token"]
        client.headers["Authorization"] = "Bearer " + token
        yield client


def test_dashboard_and_unauthenticated_access(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    result = client.get("/api/dashboard")
    assert result.status_code == 200
    assert result.json()["summary"]["income_paise"] == 9500000
    assert result.headers["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in result.headers["Content-Security-Policy"]
    client.headers.pop("Authorization")
    assert client.get("/api/dashboard").status_code == 401


def test_bad_inputs_no_echo_and_size_limit(client):
    assert (
        client.post("/api/chat", json={"message": "hi", "customer_id": "kabir"}).status_code == 422
    )
    response = client.post("/api/chat", json={"message": "", "secret": "sensitive-credential"})
    assert "sensitive-credential" not in response.text
    assert client.post("/api/chat", content="x" * 17000).status_code == 413
    assert (
        client.post(
            "/api/chat", json={"message": "hi"}, headers={"Origin": "https://evil.example"}
        ).status_code
        == 403
    )
    assert client.get("/api/dashboard", headers={"Host": "evil.example"}).status_code == 400


def test_ingestion_retry_and_goal_flow(client):
    body = {
        "event_id": "api-test",
        "amount_paise": 145000,
        "merchant": "Swiggy",
        "occurred_at": datetime.now(UTC).isoformat(),
    }
    result = client.post("/api/transactions", json=body)
    assert result.status_code == 200
    assert result.json()["transaction"]["category"] == "Dining"
    assert result.json()["nudges"][0]["level"] == "near"
    assert client.post("/api/transactions", json=body).json()["duplicate"]
    assert client.post("/api/transactions", json={**body, "amount_paise": 1}).status_code == 409
    before = client.get("/api/dashboard").json()["summary"]["net_spend_paise"]
    due = (datetime.now(UTC) + timedelta(days=180)).date().isoformat()
    result = client.post(
        "/api/goals",
        json={"name": "Trip", "target_paise": 6000000, "saved_paise": 1500000, "due_date": due},
    )
    assert result.status_code == 201
    assert client.get("/api/dashboard").json()["summary"]["net_spend_paise"] == before


def test_voice_consent_and_scoped_playback(client):
    assert (
        client.post("/api/voice", files={"audio": ("hi.wav", b"fake", "audio/wav")}).status_code
        == 403
    )
    client.put("/api/preferences", json={"ai_consent": True, "voice_consent": True})
    assert (
        client.post("/api/voice", files={"audio": ("hi.txt", b"fake", "text/plain")}).status_code
        == 415
    )
    response = client.post("/api/voice", files={"audio": ("hi.wav", b"fake", "audio/wav")})
    assert response.status_code == 200
    result = response.json()
    assert result["transcript"] == "Summarise my spending"
    audio = client.get("/api/speech/" + result["answer_id"])
    assert audio.status_code == 200 and audio.content == b"ID3-synthetic-audio"
    assert audio.headers["X-Audio-Disclosure"] == "AI-generated voice"
    other = client.app.state.auth.tokens["kabir"]
    client.headers["Authorization"] = "Bearer " + other
    client.put("/api/preferences", json={"ai_consent": True, "voice_consent": True})
    assert client.get("/api/speech/" + result["answer_id"]).status_code == 404
    assert (
        client.post(
            "/api/chat", json={"message": "continue", "conversation_id": result["conversation_id"]}
        ).status_code
        == 404
    )


def test_voice_error_is_sanitized_and_consent_revocation(client, monkeypatch):
    client.put("/api/preferences", json={"ai_consent": True, "voice_consent": True})
    monkeypatch.setattr(
        SpeechStub, "transcribe", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("sk-secret"))
    )
    response = client.post("/api/voice", files={"audio": ("hi.wav", b"fake", "audio/wav")})
    assert response.status_code == 503 and "sk-secret" not in response.text
    answer = client.post("/api/chat", json={"message": "summary"}).json()["answer_id"]
    client.put("/api/preferences", json={"ai_consent": False, "voice_consent": False})
    assert client.get("/api/speech/" + answer).status_code == 403


def test_rate_limit(client):
    statuses = [
        client.post("/api/chat", json={"message": "summary"}).status_code for _ in range(41)
    ]
    assert statuses[-1] == 429


def test_config_fail_closed():
    with pytest.raises(ValueError):
        Settings(mode="bank")
    with pytest.raises(ValueError):
        Settings(base_url="http://untrusted.example/v1")
    with pytest.raises(ValueError):
        Settings(base_url="https://user:secret@host/v1")
    assert Settings(base_url="http://127.0.0.1:8000/v1").base_url


def test_bank_auth_rs256_expiry_scope_roles_and_no_demo(repo, ai):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = (
        private.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    settings = Settings(mode="bank", jwt_public_key=public, jwt_issuer="https://bank.example")
    app = create_app(settings, repository=repo, intelligence=ai)
    now = int(datetime.now(UTC).timestamp())
    claims = {
        "sub": "alice",
        "bank_id": "bank-one",
        "aud": "truffles-pfm",
        "iss": "https://bank.example",
        "iat": now,
        "exp": now + 300,
        "permissions": ["pfm:read"],
    }
    with TestClient(app) as client:
        assert client.get("/api/demo/session").status_code == 404
        assert client.get("/api/docs").status_code == 404
        token = jwt.encode(claims, private, algorithm="RS256")
        client.headers["Authorization"] = "Bearer " + token
        assert client.get("/api/dashboard").json()["transactions"] == []
        assert (
            client.put("/api/budget", json={"category": "Dining", "limit_paise": 100}).status_code
            == 403
        )
        for bad in (
            {**claims, "exp": now - 100},
            {**claims, "aud": "wrong"},
            {**claims, "iss": "wrong"},
            {**claims, "permissions": "pfm:read"},
        ):
            client.headers["Authorization"] = "Bearer " + jwt.encode(
                bad, private, algorithm="RS256"
            )
            assert client.get("/api/dashboard").status_code == 401
        client.headers["Authorization"] = "Bearer " + jwt.encode(
            claims, "x" * 32, algorithm="HS256"
        )
        assert client.get("/api/dashboard").status_code == 401


def test_limiter_recovers_after_window(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("truffles_pfm.security.time.monotonic", lambda: clock[0])
    limit = RateLimit(maximum=1, seconds=60)
    limit.check("a")
    with pytest.raises(Exception) as error:
        limit.check("a")
    assert error.value.status_code == 429
    clock[0] = 61
    limit.check("a")


def test_chunked_oversized_payload_is_rejected(client):
    def chunks():
        yield b"x" * 10000
        yield b"x" * 10000

    assert client.post("/api/chat", content=chunks()).status_code == 413


def test_oversized_audio_and_empty_audio(client):
    client.put("/api/preferences", json={"ai_consent": True, "voice_consent": True})
    for data in (b"", b"x" * 2_000_001):
        assert (
            client.post("/api/voice", files={"audio": ("a.wav", data, "audio/wav")}).status_code
            == 413
        )


def test_diagnostics_and_database_readiness(client):
    assert client.get("/readyz").json() == {"status": "ready"}
    diagnostics = client.get("/api/diagnostics").json()
    assert set(diagnostics) == {
        "ai_calls",
        "fallbacks",
        "last_failure_type",
        "last_model_latency_ms",
    }
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert "Download OpenAPI schema" in client.get("/api/docs").text
    assert client.get("/api/openapi.json").json()["info"]["title"] == "Truffles PFM · IB3"


def test_api_correction_budget_preferences_and_audit(client):
    assert (
        client.patch(
            "/api/transactions/seed-ananya-4/category", json={"category": "Groceries"}
        ).status_code
        == 200
    )
    assert (
        client.patch(
            "/api/transactions/missing/category", json={"category": "Groceries"}
        ).status_code
        == 404
    )
    assert (
        client.put("/api/budget", json={"category": "Dining", "limit_paise": 200000}).status_code
        == 200
    )
    assert client.get("/api/audit").json()["events"][0]["action"] == "budget_updated"
    assert client.post("/api/nudges/missing/dismiss").status_code == 404
