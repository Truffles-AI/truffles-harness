"""SQLite reference adapter: scoped keys, transactions, WAL and a hash-linked audit log.

The transaction boundary deliberately lives here, not inside LLM execution. Replace
this adapter with bank-managed PostgreSQL + RLS before a horizontally scaled rollout.
"""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock

from .domain import Conflict as Conflict
from .domain import NotFound as NotFound
from .domain import Scope, canonical


class Repository:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=10)
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version not in {0, 1}:
            self.db.close()
            raise ValueError("Unsupported database schema version; run an approved migration")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=10000")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS records (
          bank TEXT NOT NULL, customer TEXT NOT NULL, kind TEXT NOT NULL,
          id TEXT NOT NULL, payload TEXT NOT NULL,
          PRIMARY KEY(bank, customer, kind, id)
        );
        CREATE TABLE IF NOT EXISTS audit (
          seq INTEGER PRIMARY KEY AUTOINCREMENT, bank TEXT NOT NULL, customer TEXT NOT NULL,
          action TEXT NOT NULL, ref TEXT NOT NULL, at TEXT NOT NULL,
          previous TEXT NOT NULL, digest TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS audit_scope ON audit(bank,customer,seq);
        PRAGMA user_version=1;
        """)

    @contextmanager
    def atomic(self):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def get(self, scope: Scope, kind, key):
        with self.lock:
            row = self.db.execute(
                "SELECT payload FROM records WHERE bank=? AND customer=? AND kind=? AND id=?",
                (scope.bank_id, scope.customer_id, kind, key),
            ).fetchone()
            return json.loads(row[0]) if row else None

    def list(self, scope: Scope, kind):
        with self.lock:
            rows = self.db.execute(
                "SELECT payload FROM records WHERE bank=? AND customer=? AND kind=? ORDER BY id",
                (scope.bank_id, scope.customer_id, kind),
            ).fetchall()
            return [json.loads(row[0]) for row in rows]

    def put(self, scope: Scope, kind, key, payload):
        # Call within atomic() when part of a business state transition.
        with self.lock:
            self.db.execute(
                "INSERT INTO records VALUES(?,?,?,?,?) ON CONFLICT(bank,customer,kind,id) "
                "DO UPDATE SET payload=excluded.payload",
                (scope.bank_id, scope.customer_id, kind, key, canonical(payload)),
            )

    def delete(self, scope: Scope, kind):
        with self.lock:
            self.db.execute(
                "DELETE FROM records WHERE bank=? AND customer=? AND kind=?",
                (scope.bank_id, scope.customer_id, kind),
            )

    def audit(self, scope: Scope, action, reference):
        # Deliberately exclude prompts, tokens, credentials and merchant narrations.
        row = self.db.execute(
            "SELECT digest FROM audit WHERE bank=? AND customer=? ORDER BY seq DESC LIMIT 1",
            (scope.bank_id, scope.customer_id),
        ).fetchone()
        previous = row[0] if row else "0" * 64
        at = datetime.now(UTC).isoformat()
        data = [scope.bank_id, scope.customer_id, action, reference, at, previous]
        digest = hashlib.sha256(canonical(data).encode()).hexdigest()
        self.db.execute(
            "INSERT INTO audit(bank,customer,action,ref,at,previous,digest) VALUES(?,?,?,?,?,?,?)",
            (*data, digest),
        )

    def audit_events(self, scope):
        with self.lock:
            return [
                dict(row)
                for row in self.db.execute(
                    "SELECT * FROM audit WHERE bank=? AND customer=? ORDER BY seq DESC LIMIT 100",
                    (scope.bank_id, scope.customer_id),
                )
            ]

    def close(self):
        self.db.close()
