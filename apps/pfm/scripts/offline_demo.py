"""A complete synthetic PFM flow with inference explicitly disabled."""

import json
from datetime import UTC, datetime
from pathlib import Path

from truffles_pfm.config import Settings
from truffles_pfm.domain import Question, Scope, Transaction
from truffles_pfm.intelligence import Intelligence
from truffles_pfm.repository import Repository
from truffles_pfm.seed import seed
from truffles_pfm.service import PFM

repo = Repository(":memory:")
now = datetime(2026, 10, 21, 12, tzinfo=UTC)
service = PFM(repo, Intelligence(Settings(api_key=None)), clock=lambda: now)
seed(service)
customer = Scope(bank_id="synthetic-bank", customer_id="ananya")
event = Transaction(
    event_id="offline-dining", amount_paise=145000, merchant="Swiggy", occurred_at=now
)
result = service.ingest(customer, event)
assert result["transaction"]["category"] == "Dining"
assert result["nudges"][0]["level"] == "near"
assert service.ingest(customer, event)["duplicate"]
answer = service.chat(customer, Question(message="Summarise my spending"))
assert answer["facts"]["net_spend_paise"] == 3_498_900
assert answer["mode"] == "fallback"
report = {
    "category": result["transaction"]["category"],
    "nudge": result["nudges"][0]["message"],
    "answer": answer["answer"],
    "mode": answer["mode"],
    "model_calls": service.ai.calls,
}
Path(__file__).resolve().parents[1].joinpath("artifacts/offline-demo.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2)
)
print(json.dumps(report, ensure_ascii=False, indent=2))
repo.close()
