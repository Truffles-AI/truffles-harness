"""Deterministic category policy; only unknown merchants reach the intelligence port."""

import re

from .domain import SPEND_CATEGORIES, Category

MCC = {
    "5411": Category.GROCERIES,
    "5812": Category.DINING,
    "5814": Category.DINING,
    "4121": Category.TRANSPORT,
    "4111": Category.TRANSPORT,
    "4900": Category.BILLS,
    "5912": Category.HEALTH,
    "8062": Category.HEALTH,
    "8211": Category.EDUCATION,
    "5311": Category.SHOPPING,
    "7832": Category.ENTERTAINMENT,
}
MERCHANTS = {
    "bigbasket": Category.GROCERIES,
    "dmart": Category.GROCERIES,
    "swiggy": Category.DINING,
    "zomato": Category.DINING,
    "uber": Category.TRANSPORT,
    "ola": Category.TRANSPORT,
    "bescom": Category.BILLS,
    "airtel": Category.BILLS,
    "apollo pharmacy": Category.HEALTH,
    "netflix": Category.ENTERTAINMENT,
}


def categorize(tx, intelligence, *, consent):
    if tx.kind == "income":
        return Category.INCOME, "ledger", False
    if tx.kind == "transfer":
        return Category.TRANSFERS, "ledger", False
    if tx.mcc in MCC:
        return MCC[tx.mcc], "mcc", False
    for merchant, category in MERCHANTS.items():
        if re.search(r"\b" + re.escape(merchant) + r"\b", tx.merchant.casefold()):
            return category, "merchant_rule", False
    proposed = intelligence.classify(tx.merchant) if consent else None
    if proposed and proposed.confidence >= 0.85 and proposed.category in SPEND_CATEGORIES:
        return proposed.category, "model_suggestion", True
    return Category.OTHER, "needs_review", True
