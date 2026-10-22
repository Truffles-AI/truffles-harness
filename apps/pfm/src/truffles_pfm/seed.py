"""Fixed, clearly synthetic personas; no real account identifiers or personal data."""

from datetime import datetime, timedelta

from .domain import Budget, Category, Goal, Preferences, Scope, Transaction
from .service import IST


def seed(service):
    today = service.today()
    for customer, scale in (("ananya", 1), ("kabir", 2)):
        scope = Scope(bank_id="synthetic-bank", customer_id=customer)
        if service.repo.list(scope, "transaction"):
            continue
        service.set_preferences(scope, Preferences())
        service.budget(scope, Budget(category=Category.DINING, limit_paise=500_000))
        service.budget(scope, Budget(category=Category.GROCERIES, limit_paise=800_000))
        entries = [
            ("Salary", 9_500_000, "income", None),
            ("Monthly rent", 2_200_000, "purchase", None),
            ("BigBasket", 248_000, "purchase", "5411"),
            ("DMart", 182_000, "purchase", "5411"),
            ("Swiggy", 119_000, "purchase", "5812"),
            ("Zomato", 169_000, "purchase", "5812"),
            ("Uber", 72_000, "purchase", "4121"),
            ("BESCOM", 215_000, "purchase", "4900"),
            ("Apollo Pharmacy", 84_000, "purchase", "5912"),
            ("Netflix", 64_900, "purchase", "7832"),
            ("Own account transfer", 1_000_000, "transfer", None),
        ]
        for i, (merchant, amount, kind, mcc) in enumerate(entries):
            day = min(today.day, i + 1)
            service.ingest(
                scope,
                Transaction(
                    event_id=f"seed-{customer}-{i}",
                    amount_paise=amount * scale,
                    occurred_at=datetime(today.year, today.month, day, 0, 0, tzinfo=IST),
                    merchant=merchant,
                    mcc=mcc,
                    kind=kind,
                ),
            )
        service.correct(scope, f"seed-{customer}-1", Category.RENT)
        service.goal(
            scope,
            Goal(
                name="Emergency fund",
                target_paise=30_000_000,
                saved_paise=7_500_000 * scale,
                due_date=today + timedelta(days=365),
            ),
            key="seed-emergency",
        )
        service.set_preferences(scope, Preferences(ai_consent=True, voice_consent=False))
