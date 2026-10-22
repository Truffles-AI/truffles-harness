"""Explicit configuration. No credentials in source or browser bundles."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class Settings:
    mode: str = "demo"
    database: str = ".pfm/pfm.sqlite3"
    model: str = "openai:gpt-4.1-mini"
    model_timeout: float = 12.0
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    jwt_public_key: str = ""
    jwt_issuer: str = ""
    jwt_audience: str = "truffles-pfm"
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "testserver")
    audio_enabled: bool = False
    audio_base_url: str | None = None
    audio_api_key: str | None = field(default=None, repr=False)
    transcription_model: str = "whisper-1"
    synthesis_model: str = "tts-1"

    def __post_init__(self):
        if not 1 <= self.model_timeout <= 60:
            raise ValueError("Model timeout must be between 1 and 60 seconds")
        if self.mode not in {"demo", "bank"}:
            raise ValueError("PFM_MODE must be demo or bank")
        if self.mode == "bank" and (not self.jwt_public_key or not self.jwt_issuer):
            raise ValueError("Bank mode requires RS256 verification key and issuer")
        for url in (self.base_url, self.audio_base_url):
            if url:
                parsed = urlparse(url)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                    raise ValueError("Model endpoint must be an explicit HTTP(S) URL")
                if parsed.username or parsed.password or parsed.query or parsed.fragment:
                    raise ValueError("Endpoint must not contain credentials, query or fragment")
                if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1"}:
                    raise ValueError("Non-loopback model endpoints require HTTPS")

    @classmethod
    def from_env(cls):
        key_file = os.getenv("PFM_JWT_PUBLIC_KEY_FILE")
        return cls(
            mode=os.getenv("PFM_MODE", "demo"),
            database=os.getenv("PFM_DATABASE", ".pfm/pfm.sqlite3"),
            model=os.getenv("PFM_MODEL", "openai:gpt-4.1-mini"),
            model_timeout=float(os.getenv("PFM_MODEL_TIMEOUT", "12")),
            base_url=os.getenv("PFM_MODEL_BASE_URL"),
            api_key=os.getenv("PFM_MODEL_API_KEY") or os.getenv("OPENAI_API_KEY"),
            jwt_public_key=Path(key_file).read_text() if key_file else "",
            jwt_issuer=os.getenv("PFM_JWT_ISSUER", ""),
            jwt_audience=os.getenv("PFM_JWT_AUDIENCE", "truffles-pfm"),
            allowed_hosts=tuple(os.getenv("PFM_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")),
            audio_enabled=os.getenv("PFM_AUDIO_ENABLED") == "1",
            audio_base_url=os.getenv("PFM_AUDIO_BASE_URL"),
            audio_api_key=os.getenv("PFM_AUDIO_API_KEY") or os.getenv("OPENAI_API_KEY"),
            transcription_model=os.getenv("PFM_TRANSCRIPTION_MODEL", "whisper-1"),
            synthesis_model=os.getenv("PFM_SYNTHESIS_MODEL", "tts-1"),
        )
