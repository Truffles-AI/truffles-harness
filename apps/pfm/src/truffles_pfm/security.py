"""Authentication adapters and per-process request controls."""

import secrets
import time
from collections import deque
from dataclasses import dataclass
from threading import Lock

import jwt
from fastapi import HTTPException

from .domain import Scope


@dataclass(frozen=True)
class Principal:
    scope: Scope
    permissions: frozenset[str]


class Auth:
    def __init__(self, settings):
        self.settings = settings
        self.tokens = {}
        if settings.mode == "demo":
            for customer in ("ananya", "kabir"):
                self.tokens[customer] = secrets.token_urlsafe(32)

    def authenticate(self, header):
        if not header or not header.startswith("Bearer "):
            raise HTTPException(401, "Bearer token required")
        token = header[7:]
        if self.settings.mode == "demo":
            for customer, expected in self.tokens.items():
                if secrets.compare_digest(token, expected):
                    return Principal(
                        Scope(bank_id="synthetic-bank", customer_id=customer),
                        frozenset({"pfm:read", "pfm:write", "pfm:ingest", "pfm:ops"}),
                    )
            raise HTTPException(401, "Invalid session")
        try:
            claims = jwt.decode(
                token,
                self.settings.jwt_public_key,
                algorithms=["RS256"],
                audience=self.settings.jwt_audience,
                issuer=self.settings.jwt_issuer,
                options={"require": ["exp", "iat", "sub", "iss", "aud", "bank_id", "permissions"]},
            )
            permissions = claims["permissions"]
            if not isinstance(permissions, list) or not all(
                isinstance(x, str) for x in permissions
            ):
                raise ValueError("Invalid permissions")
            return Principal(
                Scope(bank_id=claims["bank_id"], customer_id=claims["sub"]), frozenset(permissions)
            )
        except (jwt.PyJWTError, ValueError, TypeError):
            raise HTTPException(401, "Invalid or expired session") from None


class RateLimit:
    def __init__(self, maximum=40, seconds=60):
        self.maximum, self.seconds = maximum, seconds
        self.windows = {}
        self.lock = Lock()

    def check(self, key):
        now = time.monotonic()
        with self.lock:
            self.windows = {
                k: v for k, v in self.windows.items() if v and v[-1] > now - self.seconds
            }
            times = self.windows.setdefault(key, deque())
            while times and times[0] <= now - self.seconds:
                times.popleft()
            if len(times) >= self.maximum:
                raise HTTPException(
                    429, "Request limit reached; try again shortly", headers={"Retry-After": "60"}
                )
            if len(self.windows) > 10000:
                raise HTTPException(503, "Capacity reached")
            times.append(now)
