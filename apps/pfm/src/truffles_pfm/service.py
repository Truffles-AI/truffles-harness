"""Application use cases. Arithmetic, policies and persisted state never depend on an LLM."""

import hashlib
import uuid
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from .categorization import categorize
from .domain import Conflict, NotFound, Preferences, canonical, money
from .ports import IntelligencePort, RepositoryPort

IST = ZoneInfo("Asia/Kolkata")


class PFM:
    def __init__(self, repository: RepositoryPort, intelligence: IntelligencePort, clock=None):
        self.repo = repository
        self.ai = intelligence
        self.clock = clock or (lambda: datetime.now(UTC))

    def today(self):
        return self.clock().astimezone(IST).date()

    def preferences(self, scope):
        return self.repo.get(scope, "preferences", "current") or Preferences().model_dump()

    def set_preferences(self, scope, value):
        with self.repo.atomic():
            self.repo.put(scope, "preferences", "current", value.model_dump())
            if not value.voice_consent or not value.ai_consent:
                self.repo.delete(scope, "answer")
            self.repo.audit(scope, "preferences_updated", "current")
        return value.model_dump()

    def ingest(self, scope, tx):
        if tx.occurred_at > self.clock() + timedelta(minutes=5):
            raise ValueError("Future transactions are not accepted")
        fingerprint = hashlib.sha256(canonical(tx.model_dump(mode="json")).encode()).hexdigest()
        existing = self.repo.get(scope, "transaction", tx.event_id)
        if existing:
            if existing["fingerprint"] != fingerprint:
                raise Conflict("event_id was already used for a different payload")
            return {"transaction": existing, "duplicate": True, "nudges": []}
        prefs = self.preferences(scope)
        if tx.kind != "refund":
            category, source, review = categorize(tx, self.ai, consent=prefs["ai_consent"])
        with self.repo.atomic():
            # Recheck inside the write boundary: simultaneous retries cannot double-book.
            existing = self.repo.get(scope, "transaction", tx.event_id)
            if existing:
                if existing["fingerprint"] != fingerprint:
                    raise Conflict("event_id was already used for a different payload")
                return {"transaction": existing, "duplicate": True, "nudges": []}
            if tx.kind == "refund":
                original = self.repo.get(scope, "transaction", tx.original_event_id)
                if not original or original["kind"] != "purchase":
                    raise ValueError("Refund must reference this customer's purchase")
                if tx.occurred_at < datetime.fromisoformat(original["occurred_at"]):
                    raise ValueError("Refund cannot precede the purchase")
                refunded = sum(
                    t["amount_paise"]
                    for t in self.repo.list(scope, "transaction")
                    if t["original_event_id"] == tx.original_event_id
                )
                if refunded + tx.amount_paise > original["amount_paise"]:
                    raise ValueError("Cumulative refund exceeds original purchase")
                category, source, review = (
                    original["category"],
                    "original_purchase",
                    original["needs_review"],
                )
            row = {
                **tx.model_dump(mode="json"),
                "category": str(category),
                "source": source,
                "needs_review": review,
                "fingerprint": fingerprint,
                "month": tx.occurred_at.astimezone(IST).strftime("%Y-%m"),
            }
            self.repo.put(scope, "transaction", tx.event_id, row)
            self.repo.audit(scope, "transaction_ingested", tx.event_id)
            if tx.kind == "refund":
                self._supersede_nudges(scope, {row["category"]}, row["month"])
            nudges = self._nudges(scope, row) if tx.kind == "purchase" else []
        return {"transaction": row, "duplicate": False, "nudges": nudges}

    def summary(self, scope, month=None):
        month = month or self.today().strftime("%Y-%m")
        rows = [t for t in self.repo.list(scope, "transaction") if t["month"] == month]
        categories = {}
        income, spending, refunds, transfers = 0, 0, 0, 0
        for tx in rows:
            amount = tx["amount_paise"]
            if tx["kind"] == "income":
                income += amount
            elif tx["kind"] == "transfer":
                transfers += amount
            else:
                signed = -amount if tx["kind"] == "refund" else amount
                categories[tx["category"]] = categories.get(tx["category"], 0) + signed
                if tx["kind"] == "refund":
                    refunds += amount
                else:
                    spending += amount
        net = spending - refunds
        return {
            "month": month,
            "income_paise": income,
            "gross_spend_paise": spending,
            "refunds_paise": refunds,
            "net_spend_paise": net,
            "surplus_paise": income - net,
            "transfers_excluded_paise": transfers,
            "categories": dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)),
            "transaction_count": len(rows),
            "needs_review": sum(t["needs_review"] for t in rows),
            "source_event_ids": [t["event_id"] for t in rows],
            "as_of": self.clock().isoformat(),
            "currency": "INR",
        }

    def _nudges(self, scope, row):
        # Historical ingestion doesn't send today's customer a stale budget alert.
        prefs = self.preferences(scope)
        month = self.today().strftime("%Y-%m")
        if not prefs["nudges_enabled"] or row["month"] != month:
            return []
        budget = self.repo.get(scope, "budget", row["category"])
        if not budget:
            return []
        spent = self.summary(scope, month)["categories"].get(row["category"], 0)
        limit = budget["limit_paise"]
        level = "exceeded" if spent >= limit else "near" if spent * 100 >= limit * 80 else None
        # One alert per category / month / threshold; retries cannot redeliver.
        key = f"{month}:{row['category']}:{level}"
        if not level or self.repo.get(scope, "nudge", key):
            return []
        threshold_text = (
            "You have reached your budget."
            if level == "exceeded"
            else "You have used at least 80%."
        )
        message = (
            f"{row['category']} spending is {money(spent)} against your {money(limit)} "
            f"monthly budget. {threshold_text} "
            "Review your recent transactions when convenient."
        )
        item = {
            "id": key,
            "message": message,
            "category": row["category"],
            "level": level,
            "created_at": self.clock().isoformat(),
            "status": "unread",
            "trigger_event_id": row["event_id"],
        }
        self.repo.put(scope, "nudge", key, item)
        self.repo.audit(scope, "nudge_created", key)
        return [item]

    def _supersede_nudges(self, scope, categories, month):
        for nudge in self.repo.list(scope, "nudge"):
            if (
                nudge["status"] == "unread"
                and nudge["category"] in categories
                and nudge["id"].startswith(month + ":")
            ):
                nudge["status"] = "superseded"
                self.repo.put(scope, "nudge", nudge["id"], nudge)

    def dismiss_nudge(self, scope, key):
        with self.repo.atomic():
            row = self.repo.get(scope, "nudge", key)
            if not row:
                raise NotFound("Nudge not found")
            row["status"] = "dismissed"
            self.repo.put(scope, "nudge", key, row)
            self.repo.audit(scope, "nudge_dismissed", key)
        return row

    def budget(self, scope, value):
        with self.repo.atomic():
            self.repo.put(scope, "budget", value.category, value.model_dump(mode="json"))
            self._supersede_nudges(scope, {value.category}, self.today().strftime("%Y-%m"))
            self.repo.audit(scope, "budget_updated", value.category)
        return value.model_dump(mode="json")

    def goal(self, scope, value, key=None):
        if value.due_date <= self.today():
            raise ValueError("Goal due date must be in the future")
        key = key or uuid.uuid4().hex
        with self.repo.atomic():
            if key and self.repo.get(scope, "goal", key):
                raise Conflict("Goal already exists")
            if len(self.repo.list(scope, "goal")) >= 50:
                raise ValueError("Maximum 50 goals per customer")
            row = {"id": key, **value.model_dump(mode="json")}
            self.repo.put(scope, "goal", key, row)
            self.repo.audit(scope, "goal_created", key)
        return row

    def goals(self, scope):
        rows = self.repo.list(scope, "goal")
        for row in rows:
            due = date.fromisoformat(row["due_date"])
            days = max(1, (due - self.today()).days)
            remaining = row["target_paise"] - row["saved_paise"]
            row["remaining_paise"] = remaining
            row["daily_plan_paise"] = (remaining + days - 1) // days
            row["progress_percent"] = row["saved_paise"] * 100 // row["target_paise"]
            row["overdue"] = due <= self.today() and remaining > 0
        return rows

    def correct(self, scope, event_id, category):
        with self.repo.atomic():
            row = self.repo.get(scope, "transaction", event_id)
            if not row:
                raise NotFound("Transaction not found")
            if row["kind"] != "purchase":
                raise ValueError(
                    "Only purchases can be recategorised; refunds follow their purchase"
                )
            previous_category = row["category"]
            row.update(category=str(category), source="customer", needs_review=False)
            self.repo.put(scope, "transaction", event_id, row)
            for refund in self.repo.list(scope, "transaction"):
                if refund["original_event_id"] == event_id:
                    refund.update(category=str(category), needs_review=False)
                    self.repo.put(scope, "transaction", refund["event_id"], refund)
            self._supersede_nudges(scope, {previous_category, str(category)}, row["month"])
            self.repo.audit(scope, "category_corrected", event_id)
        return row

    def chat(self, scope, question):
        conversation = question.conversation_id or uuid.uuid4().hex
        previous = self.repo.get(scope, "conversation", conversation)
        if question.conversation_id and not previous:
            raise NotFound("Conversation not found")
        month = self.today().strftime("%Y-%m")
        intent, mode = self.ai.intent(
            question.message, month, previous, consent=self.preferences(scope)["ai_consent"]
        )
        facts = self.summary(scope, intent.month or month)
        hindi = intent.language == "hi"
        if intent.intent == "summary":
            if hindi:
                answer = (
                    f"{facts['month']} में आय {money(facts['income_paise'])}, "
                    f"खर्च {money(facts['net_spend_paise'])} और नकदी अधिशेष "
                    f"{money(facts['surplus_paise'])} है। यह खाते का बैलेंस नहीं है।"
                )
            else:
                answer = (
                    f"In {facts['month']}, your income was {money(facts['income_paise'])}, "
                    f"net spending was {money(facts['net_spend_paise'])}, and cash surplus was "
                    f"{money(facts['surplus_paise'])}. This is cash flow, not your account balance."
                )
            top = next(iter(facts["categories"]), None)
            if top and facts["categories"][top] > 0:
                label = "सबसे बड़ा खर्च" if hindi else "Largest spending category"
                answer += f" {label}: {top} ({money(facts['categories'][top])})."
        elif intent.intent == "category" and intent.category:
            amount = facts["categories"].get(intent.category, 0)
            answer = f"{facts['month']}: {intent.category} — {money(amount)}. " + (
                "रिफंड घटाने के बाद।" if hindi else "Net of refunds in this month."
            )
        elif intent.intent == "savings":
            answer = (
                f"{facts['month']}: {'नकदी अधिशेष' if hindi else 'cash surplus'} "
                f"{money(facts['surplus_paise'])}. "
                + (
                    "यह आय घटा खर्च है, बैंक बैलेंस या निवेश सलाह नहीं।"
                    if hindi
                    else "This is income minus net spending, not a bank balance or a guaranteed "
                    "amount available to save. Transfers are excluded."
                )
            )
        elif intent.intent == "goals":
            goals = self.goals(scope)
            if not goals:
                answer = (
                    "अभी कोई लक्ष्य नहीं है। लक्ष्य फॉर्म का उपयोग करें।"
                    if hindi
                    else "No goals yet. Use the goal form to create one."
                )
            else:
                answer = "\n".join(
                    f"{g['name']}: {money(g['saved_paise'])} / {money(g['target_paise'])} "
                    f"({g['progress_percent']}%). {'बाकी' if hindi else 'Remaining'} "
                    f"{money(g['remaining_paise'])}; "
                    + (
                        (
                            "लक्ष्य की तारीख बीत गई है।"
                            if hindi
                            else "target date has passed; review your plan."
                        )
                        if g["overdue"]
                        else f"{'दैनिक योजना' if hindi else 'illustrative daily plan'} "
                        f"{money(g['daily_plan_paise'])} "
                        f"{'तक' if hindi else 'until'} {g['due_date']}."
                    )
                    for g in goals
                )
                answer += "\n" + (
                    "कोई पैसा ट्रांसफर नहीं हुआ।"
                    if hindi
                    else "These are tracking goals; no money has been moved."
                )
        elif intent.intent == "unsupported":
            answer = (
                "मैं आपके खर्च और लक्ष्यों का सारांश दे सकता हूँ। "
                "भुगतान या निवेश सलाह के लिए बैंक की अधिकृत सेवा का उपयोग करें।"
                if hindi
                else "I can explain your spending and goals. "
                "Payments, investments, credit decisions "
                "and other customers' records are outside this assistant. "
                "Use your bank's authorised service."
            )
        else:
            answer = (
                "खर्च, किसी श्रेणी का खर्च, बचत या लक्ष्य पूछें।"
                if hindi
                else "Ask for a monthly spending summary, spending by category, cash surplus, "
                "or your goals. Use the forms to set budgets and goals."
            )
        if facts["transaction_count"] == 0 and intent.intent in {"summary", "category", "savings"}:
            answer = (
                f"{facts['month']}: कोई लेन-देन उपलब्ध नहीं है।"
                if hindi
                else f"No recorded transactions for {facts['month']}; "
                "I cannot infer your finances for that period."
            )
        if facts["needs_review"] and intent.intent in {"summary", "category"}:
            answer += (
                f" {facts['needs_review']} श्रेणियों की समीक्षा बाकी है।"
                if hindi
                else f" {facts['needs_review']} transaction categories still need review."
            )
        answer_id = uuid.uuid4().hex
        result = {
            "answer": answer,
            "answer_id": answer_id,
            "conversation_id": conversation,
            "intent": intent.model_dump(mode="json"),
            "mode": mode,
            "facts": facts,
            "citations": facts["source_event_ids"]
            if intent.intent in {"summary", "category", "savings"}
            else [],
            "created_at": self.clock().isoformat(),
        }
        with self.repo.atomic():
            self.repo.put(scope, "conversation", conversation, intent.model_dump(mode="json"))
            # Retain only recent answer text for optional speech, never raw chat/audio.
            answers = self.repo.list(scope, "answer")
            if len(answers) >= 20:
                self.repo.delete(scope, "answer")
            self.repo.put(scope, "answer", answer_id, result)
            self.repo.audit(scope, "chat_answered", answer_id)
        return result
