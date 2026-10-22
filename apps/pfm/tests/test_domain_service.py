import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from truffles_pfm.domain import (
    Budget,
    Category,
    Classification,
    Goal,
    Preferences,
    Question,
    Scope,
    Transaction,
    money,
    redact,
)
from truffles_pfm.repository import Conflict, NotFound, Repository, canonical
from truffles_pfm.service import PFM


def tx(key="one", amount=10000, **kwargs):
    return Transaction(
        event_id=key,
        amount_paise=amount,
        merchant="Swiggy",
        occurred_at=datetime(2026, 10, 10, tzinfo=UTC),
        **kwargs,
    )


@pytest.mark.parametrize("value", [0, -1, 1.1, True, "100", 100_000_000_001])
def test_money_inputs_reject_lossy_or_invalid_amounts(value):
    with pytest.raises(ValidationError):
        tx(amount=value)


def test_dates_references_and_extra_fields():
    with pytest.raises(ValidationError):
        Transaction(event_id="x", amount_paise=100, merchant="x", occurred_at="2026-10-10")
    with pytest.raises(ValidationError):
        tx(kind="refund")
    with pytest.raises(ValidationError):
        tx(customer_id="someone-else")
    with pytest.raises(ValidationError):
        tx(currency="USD")


def test_indian_money_and_redaction():
    assert money(123456789) == "₹12,34,567.89"
    assert money(-1) == "−₹0.01"
    result = redact("send to alice@bank 9999999999 ABCDE1234F 1234-5678-1234")
    assert "alice" not in result and "9999" not in result and "ABCDE" not in result


def test_rules_and_mcc_avoid_ai(service, scope, ai):
    service.set_preferences(scope, Preferences(ai_consent=True))
    assert service.ingest(scope, tx())["transaction"]["category"] == "Dining"
    assert service.ingest(scope, tx("two", mcc="5411"))["transaction"]["category"] == "Groceries"
    assert ai.calls == 0


def test_model_categories_are_reviewable_and_consent_gated(service, scope, ai):
    unknown = tx().model_copy(update={"merchant": "Unfamiliar cafe"})
    assert service.ingest(scope, unknown)["transaction"]["category"] == "Other"
    assert ai.calls == 0
    service.set_preferences(scope, Preferences(ai_consent=True))
    ai.classification = Classification(category="Dining", confidence=0.95)
    result = service.ingest(scope, unknown.model_copy(update={"event_id": "two"}))
    assert result["transaction"]["category"] == "Dining"
    assert result["transaction"]["needs_review"]
    assert ai.calls == 1


@pytest.mark.parametrize(
    "category,confidence", [("Income", 0.99), ("Dining", 0.4), ("Transfers", 1)]
)
def test_invalid_model_category_fails_closed(service, scope, ai, category, confidence):
    service.set_preferences(scope, Preferences(ai_consent=True))
    ai.classification = Classification(category=category, confidence=confidence)
    row = service.ingest(scope, tx().model_copy(update={"merchant": "Unknown"}))["transaction"]
    assert row["category"] == "Other" and row["needs_review"]


def test_idempotent_replay_conflict_and_concurrency(service, scope):
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: service.ingest(scope, tx()), range(20)))
    assert sum(not x["duplicate"] for x in results) == 1
    assert service.summary(scope)["net_spend_paise"] == 10000
    with pytest.raises(Conflict):
        service.ingest(scope, tx(amount=10001))
    assert len(service.repo.audit_events(scope)) == 1


def test_refunds_and_transfers_do_not_inflate_savings(service, scope):
    service.ingest(scope, tx("purchase", amount=10001))
    service.ingest(scope, tx("salary", amount=100000, kind="income"))
    service.ingest(scope, tx("own", amount=50000, kind="transfer"))
    service.ingest(scope, tx("refund", amount=3001, kind="refund", original_event_id="purchase"))
    totals = service.summary(scope)
    assert totals["net_spend_paise"] == 7000
    assert totals["surplus_paise"] == 93000
    assert totals["transfers_excluded_paise"] == 50000
    with pytest.raises(ValueError, match="exceeds"):
        service.ingest(
            scope, tx("too-much", amount=7001, kind="refund", original_event_id="purchase")
        )
    service.ingest(scope, tx("rest", amount=7000, kind="refund", original_event_id="purchase"))
    assert service.summary(scope)["net_spend_paise"] == 0


def test_refund_scope_and_original_validation(service, scope):
    other = Scope(bank_id="test-bank", customer_id="other")
    service.ingest(other, tx())
    with pytest.raises(ValueError, match="this customer's"):
        service.ingest(scope, tx("refund", kind="refund", original_event_id="one"))
    with pytest.raises(NotFound):
        service.correct(scope, "one", Category.DINING)


def test_correction_propagates_to_refunds_preserves_fingerprint(service, scope):
    original = service.ingest(scope, tx())["transaction"]
    service.ingest(scope, tx("refund", amount=4000, kind="refund", original_event_id="one"))
    row = service.correct(scope, "one", Category.GROCERIES)
    assert row["fingerprint"] == original["fingerprint"]
    assert service.summary(scope)["categories"] == {"Groceries": 6000}
    assert service.ingest(scope, tx())["duplicate"]
    with pytest.raises(ValueError):
        service.correct(scope, "refund", Category.DINING)


def test_nudge_threshold_dedup_consent_and_history(service, scope):
    service.budget(scope, Budget(category="Dining", limit_paise=10000))
    assert not service.ingest(scope, tx("a", 7900))["nudges"]
    near = service.ingest(scope, tx("b", 100))["nudges"]
    assert len(near) == 1 and near[0]["level"] == "near"
    assert not service.ingest(scope, tx("c", 100))["nudges"]
    assert service.ingest(scope, tx("d", 1900))["nudges"][0]["level"] == "exceeded"
    assert not service.ingest(scope, tx("d", 1900))["nudges"]
    service.dismiss_nudge(scope, near[0]["id"])
    service.set_preferences(scope, Preferences(nudges_enabled=False))
    assert not service.ingest(scope, tx("e", 100))["nudges"]
    with pytest.raises(NotFound):
        service.dismiss_nudge(scope, "missing")


def test_indian_month_boundary_and_future(service, scope):
    transaction = tx().model_copy(update={"occurred_at": datetime(2026, 9, 30, 19, tzinfo=UTC)})
    service.ingest(scope, transaction)
    assert service.summary(scope, "2026-10")["transaction_count"] == 1
    assert service.summary(scope, "2026-09")["transaction_count"] == 0
    with pytest.raises(ValueError, match="Future"):
        service.ingest(
            scope,
            tx("future").model_copy(update={"occurred_at": datetime(2026, 12, 1, tzinfo=UTC)}),
        )


def test_goals_round_up_and_do_not_change_ledger(service, scope):
    service.ingest(scope, tx())
    before = service.summary(scope)
    service.goal(scope, Goal(name="Trip", target_paise=100, saved_paise=0, due_date="2026-10-24"))
    assert service.goals(scope)[0]["daily_plan_paise"] == 34
    assert service.summary(scope) == before
    with pytest.raises(ValueError):
        service.goal(
            scope, Goal(name="Old", target_paise=100, saved_paise=0, due_date="2026-10-01")
        )


def test_conversation_and_answer_isolation(service, scope, ai):
    service.ingest(scope, tx())
    first = service.chat(scope, Question(message="summary"))
    assert "₹100.00" in first["answer"]
    other = Scope(bank_id="another-bank", customer_id="alice")
    assert service.summary(other)["transaction_count"] == 0
    with pytest.raises(NotFound):
        service.chat(other, Question(message="continue", conversation_id=first["conversation_id"]))
    assert service.repo.get(other, "answer", first["answer_id"]) is None
    assert ai.calls == 0  # no consent


def test_audit_chain_and_rollback(service, scope):
    service.ingest(scope, tx())
    service.correct(scope, "one", Category.GROCERIES)
    events = list(reversed(service.repo.audit_events(scope)))
    previous = "0" * 64
    for e in events:
        assert e["previous"] == previous
        value = [e["bank"], e["customer"], e["action"], e["ref"], e["at"], e["previous"]]
        assert e["digest"] == hashlib.sha256(canonical(value).encode()).hexdigest()
        previous = e["digest"]
    with pytest.raises(RuntimeError), service.repo.atomic():
        service.repo.put(scope, "goal", "fail", {})
        service.repo.audit(scope, "bad", "fail")
        raise RuntimeError("rollback")
    assert service.repo.get(scope, "goal", "fail") is None
    assert len(service.repo.audit_events(scope)) == 2


def test_persistence_restart(tmp_path, ai, scope):
    path = str(tmp_path / "pfm.db")
    repo = Repository(path)
    app = PFM(repo, ai)
    app.ingest(scope, tx())
    repo.close()
    second = Repository(path)
    assert second.get(scope, "transaction", "one")["amount_paise"] == 10000
    second.close()


@pytest.mark.parametrize("language", ["en", "hi"])
@pytest.mark.parametrize(
    "intent", ["summary", "category", "goals", "savings", "help", "unsupported"]
)
def test_controlled_answers_for_every_supported_intent(service, scope, ai, language, intent):
    from truffles_pfm.domain import Intent

    service.ingest(scope, tx())
    service.goal(
        scope, Goal(name="Trip", target_paise=60000, saved_paise=10000, due_date="2026-12-31")
    )
    ai.selected = Intent(
        intent=intent, category="Dining" if intent == "category" else None, language=language
    )
    answer = service.chat(scope, Question(message="test"))
    assert answer["answer"]
    assert answer["facts"]["net_spend_paise"] == 10000
    assert answer["intent"]["language"] == language
    if intent in {"summary", "category", "savings"}:
        assert "₹100.00" in answer["answer"] or "−₹100.00" in answer["answer"]
        assert answer["citations"] == ["one"]
    else:
        assert answer["citations"] == []


def test_no_data_and_no_goals_do_not_invent_results(service, scope, ai):
    from truffles_pfm.domain import Intent

    result = service.chat(scope, Question(message="summary"))
    assert "No recorded transactions" in result["answer"]
    ai.selected = Intent(intent="goals")
    assert "No goals yet" in service.chat(scope, Question(message="goals"))["answer"]


def test_overdue_goals_are_not_presented_as_future_plans(service, scope, ai):
    from truffles_pfm.domain import Intent

    service.goal(
        scope, Goal(name="Trip", target_paise=10000, saved_paise=100, due_date="2026-10-22")
    )
    service.clock = lambda: datetime(2026, 11, 1, tzinfo=UTC)
    ai.selected = Intent(intent="goals")
    assert "date has passed" in service.chat(scope, Question(message="goals"))["answer"]


def test_refund_before_original_and_history_nudges(service, scope):
    service.budget(scope, Budget(category="Dining", limit_paise=100))
    old = tx().model_copy(update={"occurred_at": datetime(2026, 9, 20, tzinfo=UTC)})
    assert service.ingest(scope, old)["nudges"] == []
    earlier = tx("refund", 10, kind="refund", original_event_id="one").model_copy(
        update={"occurred_at": datetime(2026, 9, 19, tzinfo=UTC)}
    )
    with pytest.raises(ValueError, match="precede"):
        service.ingest(scope, earlier)


def test_domain_category_and_goal_constraints():
    from truffles_pfm.domain import Correction

    with pytest.raises(ValidationError):
        Budget(category="Income", limit_paise=100)
    with pytest.raises(ValidationError):
        Correction(category="Transfers")
    with pytest.raises(ValidationError):
        Goal(name="Trip", target_paise=100, saved_paise=101, due_date="2027-01-01")
    with pytest.raises(ValidationError):
        Transaction(
            event_id="x", amount_paise=100, merchant="hello\nworld", occurred_at=datetime.now(UTC)
        )


def test_refund_supersedes_stale_budget_alert(service, scope):
    service.budget(scope, Budget(category="Dining", limit_paise=10000))
    service.ingest(scope, tx("purchase", 9000))
    assert service.repo.list(scope, "nudge")[0]["status"] == "unread"
    service.ingest(scope, tx("refund", 9000, kind="refund", original_event_id="purchase"))
    assert service.repo.list(scope, "nudge")[0]["status"] == "superseded"


def test_unknown_schema_is_not_silently_downgraded(tmp_path):
    import sqlite3

    path = str(tmp_path / "future.db")
    db = sqlite3.connect(path)
    db.execute("PRAGMA user_version=99")
    db.close()
    with pytest.raises(ValueError, match="schema version"):
        Repository(path)


def test_refund_only_month_does_not_invent_positive_spending(service, scope):
    purchase = tx().model_copy(update={"occurred_at": datetime(2026, 9, 20, tzinfo=UTC)})
    service.ingest(scope, purchase)
    service.ingest(scope, tx("refund", 10000, kind="refund", original_event_id="one"))
    answer = service.chat(scope, Question(message="summary"))
    assert answer["facts"]["net_spend_paise"] == -10000
    assert "Largest spending category" not in answer["answer"]
