"""HTTP delivery adapter; credentials determine scope, never request-body customer IDs."""

import asyncio
import html
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from truffles import speech_model

from .config import Settings
from .domain import Budget, Correction, Goal, Preferences, Question
from .domain import Transaction as TransactionInput
from .intelligence import Intelligence
from .repository import Conflict, NotFound, Repository
from .security import Auth, RateLimit
from .seed import seed
from .service import PFM

STATIC = Path(__file__).parent / "static"


class BodyLimit:
    """Buffer bounded request bodies before any JSON/multipart parsing, including chunked."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = 2_200_000 if scope["path"] == "/api/voice" else 16_384
        parts, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > limit:
                return await JSONResponse({"detail": "Request body too large"}, 413)(
                    scope, receive, send
                )
            parts.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        delivered = False

        async def buffered():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"".join(parts), "more_body": False}
            return await receive()

        await self.app(scope, buffered, send)


def create_app(settings=None, *, repository=None, intelligence=None, speech=None, clock=None):
    settings = settings or Settings.from_env()
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    repo = repository or Repository(settings.database)
    ai = intelligence or Intelligence(settings)
    service = PFM(repo, ai, clock)
    auth = Auth(settings)
    limiter = RateLimit()
    voice_slots = asyncio.Semaphore(2)
    if settings.mode == "demo":
        seed(service)
    if speech is None and settings.audio_enabled:
        speech = speech_model(
            "openai",
            api_key=settings.audio_api_key,
            base_url=settings.audio_base_url,
            timeout=20,
            max_retries=0,
            transcription_model=settings.transcription_model,
            synthesis_model=settings.synthesis_model,
        )

    @asynccontextmanager
    async def lifespan(app):
        yield
        if repository is None:
            repo.close()
        if speech is not None and hasattr(speech, "close"):
            speech.close()

    app = FastAPI(
        title="Truffles PFM · IB3",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json" if settings.mode == "demo" else None,
    )
    app.state.service, app.state.auth = service, auth
    app.add_middleware(BodyLimit)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))

    @app.middleware("http")
    async def boundaries(request, call_next):
        request_id = uuid.uuid4().hex
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "Cross-origin requests are not accepted"}, 403)
        try:
            response = await call_next(request)
        except Exception:
            response = JSONResponse({"detail": "Request failed", "request_id": request_id}, 500)
        response.headers.update(
            {
                "X-Request-ID": request_id,
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "Referrer-Policy": "no-referrer",
                "Permissions-Policy": "microphone=(self), camera=(), geolocation=()",
                "Content-Security-Policy": (
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; "
                    "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
                ),
            }
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        return JSONResponse(
            {
                "detail": "Invalid input",
                "fields": [".".join(map(str, e["loc"])) for e in exc.errors()],
            },
            422,
        )

    @app.exception_handler(ValueError)
    async def invalid_value(request, exc):
        if isinstance(exc, Conflict):
            return JSONResponse({"detail": str(exc)}, 409)
        if isinstance(exc, NotFound):
            return JSONResponse({"detail": str(exc)}, 404)
        return JSONResponse({"detail": str(exc)}, 422)

    def permission(required):
        def resolve(authorization: str | None = Header(default=None)):
            principal = auth.authenticate(authorization)
            if required not in principal.permissions:
                raise HTTPException(403, "Insufficient permission")
            limiter.check((principal.scope.bank_id, principal.scope.customer_id, required))
            return principal.scope

        return resolve

    read, write, ingest = permission("pfm:read"), permission("pfm:write"), permission("pfm:ingest")

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.get("/api/docs", response_class=HTMLResponse)
    def api_docs():
        if settings.mode != "demo":
            raise HTTPException(404)
        schema = app.openapi()
        sections = []
        for path, methods in schema["paths"].items():
            for method, description in methods.items():
                title = html.escape(f"{method.upper()} {path}")
                detail = html.escape(json.dumps(description, indent=2))
                sections.append(f"<details><summary>{title}</summary><pre>{detail}</pre></details>")
        models = html.escape(json.dumps(schema.get("components", {}), indent=2))
        return (
            "<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>Truffles PFM API</title><link rel='stylesheet' href='/static/style.css'>"
            "<article class='card'><h1>Truffles PFM API</h1>"
            "<p>Synthetic demo API. Use a Bearer token for customer endpoints. "
            "See docs/api.md for permissions and integration semantics.</p>"
            "<p><a href='/api/openapi.json'>Download OpenAPI schema</a></p>"
            + "".join(sections)
            + "<details><summary>Input and output schemas</summary>"
            f"<pre>{models}</pre></details></article></html>"
        )

    @app.get("/api/demo/session")
    def demo_session():
        if settings.mode != "demo":
            raise HTTPException(404)
        return {
            "token": auth.tokens["ananya"],
            "name": "Ananya",
            "synthetic": True,
            "model": settings.model,
            "ai_configured": ai.enabled,
            "audio_enabled": speech is not None,
        }

    @app.get("/api/diagnostics")
    def diagnostics(scope=Depends(permission("pfm:ops"))):
        return {
            "ai_calls": getattr(ai, "calls", 0),
            "fallbacks": getattr(ai, "fallbacks", 0),
            "last_failure_type": getattr(ai, "last_failure", None),
            "last_model_latency_ms": getattr(ai, "last_latency_ms", 0),
        }

    @app.get("/readyz")
    def ready():
        with repo.lock:
            repo.db.execute("SELECT 1").fetchone()
        return {"status": "ready"}

    @app.get("/api/dashboard")
    def dashboard(scope=Depends(read)):
        with repo.atomic():
            return {
                "summary": service.summary(scope),
                "goals": service.goals(scope),
                "transactions": sorted(
                    repo.list(scope, "transaction"),
                    key=lambda t: datetime.fromisoformat(t["occurred_at"]),
                    reverse=True,
                ),
                "budgets": repo.list(scope, "budget"),
                "nudges": repo.list(scope, "nudge"),
                "preferences": service.preferences(scope),
            }

    @app.post("/api/transactions")
    def transaction(value: TransactionInput, scope=Depends(ingest)):
        return service.ingest(scope, value)

    @app.patch("/api/transactions/{event_id}/category")
    def correction(event_id: str, value: Correction, scope=Depends(write)):
        return service.correct(scope, event_id, value.category)

    @app.put("/api/budget")
    def budget(value: Budget, scope=Depends(write)):
        return service.budget(scope, value)

    @app.post("/api/goals", status_code=201)
    def goal(value: Goal, scope=Depends(write)):
        return service.goal(scope, value)

    @app.put("/api/preferences")
    def preferences(value: Preferences, scope=Depends(write)):
        return service.set_preferences(scope, value)

    @app.post("/api/nudges/{key}/dismiss")
    def dismiss(key: str, scope=Depends(write)):
        return service.dismiss_nudge(scope, key)

    @app.post("/api/chat")
    def chat(value: Question, scope=Depends(read)):
        return service.chat(scope, value)

    @app.get("/api/audit")
    def audit(scope=Depends(read)):
        return {"events": repo.audit_events(scope)}

    def voice_allowed(scope):
        prefs = service.preferences(scope)
        if not prefs["ai_consent"] or not prefs["voice_consent"]:
            raise HTTPException(403, "Enable AI and voice consent first")
        if speech is None:
            raise HTTPException(503, "Voice is not configured")

    @app.post("/api/voice")
    async def voice(audio: UploadFile = File(...), scope=Depends(read)):
        voice_allowed(scope)
        extensions = {
            "audio/webm": "webm",
            "audio/wav": "wav",
            "audio/x-wav": "wav",
            "audio/mpeg": "mp3",
            "audio/mp4": "m4a",
            "audio/ogg": "ogg",
        }
        extension = extensions.get(audio.content_type)
        if not extension:
            raise HTTPException(415, "Use WebM, WAV, MP3, M4A or OGG audio")
        data = await audio.read(2_000_001)
        await audio.close()
        if not data or len(data) > 2_000_000:
            raise HTTPException(413, "Audio must be between 1 byte and 2 MB")
        if voice_slots.locked():
            raise HTTPException(429, "Voice capacity reached")
        async with voice_slots:
            try:
                transcript = await run_in_threadpool(
                    speech.transcribe, data, filename=f"recording.{extension}"
                )
                question = Question(message=transcript)
            except Exception:
                raise HTTPException(
                    503, "Could not transcribe this recording; please type your question"
                ) from None
            result = await run_in_threadpool(service.chat, scope, question)
            return {"transcript": transcript, **result}

    @app.get("/api/speech/{answer_id}")
    async def speak(answer_id: str, scope=Depends(read)):
        voice_allowed(scope)
        answer = repo.get(scope, "answer", answer_id)
        if not answer or datetime.fromisoformat(answer["created_at"]) < service.clock() - timedelta(
            minutes=15
        ):
            raise HTTPException(404, "Answer expired or not found")
        if voice_slots.locked():
            raise HTTPException(429, "Voice capacity reached")
        async with voice_slots:
            try:
                data = await run_in_threadpool(speech.speak, answer["answer"])
            except Exception:
                raise HTTPException(503, "Speech is temporarily unavailable") from None
            return Response(
                data, media_type="audio/mpeg", headers={"X-Audio-Disclosure": "AI-generated voice"}
            )

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        if settings.mode != "demo":
            raise HTTPException(404)
        return FileResponse(STATIC / "index.html")

    return app
