"""Local synthetic benchmark; no AI requests, no network, not a bank SLA."""

import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from truffles_pfm.domain import Scope, Transaction
from truffles_pfm.repository import Repository
from truffles_pfm.service import PFM


class NoAI:
    def classify(self, merchant):
        raise AssertionError("Known MCC transactions must not call an LLM")


def main():
    repo = Repository(":memory:")
    scope = Scope(bank_id="benchmark", customer_id="synthetic")
    now = datetime.now(UTC)
    service = PFM(repo, NoAI(), clock=lambda: now)
    timings = []
    for number in range(1000):
        tx = Transaction(
            event_id=f"event-{number}",
            amount_paise=12345,
            merchant="Synthetic grocer",
            mcc="5411",
            occurred_at=now,
        )
        start = time.perf_counter()
        service.ingest(scope, tx)
        timings.append((time.perf_counter() - start) * 1000)
    start = time.perf_counter()
    facts = service.summary(scope)
    elapsed = (time.perf_counter() - start) * 1000
    assert facts["net_spend_paise"] == 12_345_000
    report = {
        "synthetic_events": 1000,
        "ai_calls": 0,
        "ingest_median_ms": round(statistics.median(timings), 3),
        "ingest_p95_ms": round(sorted(timings)[949], 3),
        "summary_1000_ms": round(elapsed, 3),
        "net_spend_paise": facts["net_spend_paise"],
        "scope": "single process, in-memory SQLite; no network; not an SLA",
    }
    path = Path(__file__).resolve().parents[1] / "artifacts/benchmark.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    repo.close()


if __name__ == "__main__":
    main()
