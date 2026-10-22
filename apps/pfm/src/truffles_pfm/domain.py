"""Pure validated inputs and money arithmetic; no model, web or database imports."""

import json
import re
from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Category(StrEnum):
    GROCERIES = "Groceries"
    DINING = "Dining"
    TRANSPORT = "Transport"
    SHOPPING = "Shopping"
    BILLS = "Bills"
    HEALTH = "Health"
    EDUCATION = "Education"
    ENTERTAINMENT = "Entertainment"
    RENT = "Rent"
    INCOME = "Income"
    TRANSFERS = "Transfers"
    OTHER = "Other"


SPEND_CATEGORIES = set(Category) - {Category.INCOME, Category.TRANSFERS}


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Scope(Input):
    bank_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    customer_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")


class Transaction(Input):
    event_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,100}$")
    amount_paise: int = Field(strict=True, gt=0, le=100_000_000_000)
    currency: Literal["INR"] = "INR"
    occurred_at: datetime
    merchant: str = Field(min_length=1, max_length=120)
    mcc: str | None = Field(default=None, pattern=r"^\d{4}$")
    kind: Literal["purchase", "income", "transfer", "refund"] = "purchase"
    original_event_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,100}$")

    @field_validator("occurred_at")
    @classmethod
    def aware(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone")
        return value

    @field_validator("merchant")
    @classmethod
    def no_controls(cls, value):
        if any(ord(c) < 32 for c in value):
            raise ValueError("control characters are not accepted")
        return value

    @model_validator(mode="after")
    def reference(self):
        if (self.kind == "refund") != (self.original_event_id is not None):
            raise ValueError("Only refunds require original_event_id")
        return self


class Goal(Input):
    name: str = Field(min_length=1, max_length=60)
    target_paise: int = Field(strict=True, gt=0, le=100_000_000_000)
    saved_paise: int = Field(strict=True, ge=0, le=100_000_000_000)
    due_date: date

    @model_validator(mode="after")
    def bounded(self):
        if self.saved_paise > self.target_paise:
            raise ValueError("saved_paise cannot exceed target_paise")
        return self


class Budget(Input):
    category: Category
    limit_paise: int = Field(strict=True, gt=0, le=100_000_000_000)

    @field_validator("category")
    @classmethod
    def spending(cls, value):
        if value not in SPEND_CATEGORIES:
            raise ValueError("Budget must be for a spending category")
        return value


class Correction(Input):
    category: Category

    @field_validator("category")
    @classmethod
    def spending(cls, value):
        return Budget.spending(value)


class Preferences(Input):
    ai_consent: bool = False
    voice_consent: bool = False
    nudges_enabled: bool = True


class Question(Input):
    message: str = Field(min_length=1, max_length=1000)
    conversation_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


class Intent(Input):
    intent: Literal["summary", "category", "goals", "savings", "help", "unsupported"]
    category: Category | None = None
    month: str | None = Field(default=None, pattern=r"^20\d{2}-(0[1-9]|1[0-2])$")
    language: Literal["en", "hi"] = "en"

    @field_validator("category", "month", mode="before")
    @classmethod
    def nullable_output(cls, value):
        # Small models sometimes encode JSON null as the string "null".
        # Normalize only this harmless representation; unknown values still fail closed.
        return None if value == "null" else value


class Classification(Input):
    category: Category
    confidence: float = Field(ge=0, le=1)


def money(paise: int) -> str:
    """Indian digit grouping without float conversion."""
    sign = "−" if paise < 0 else ""
    whole, fraction = divmod(abs(paise), 100)
    digits = str(whole)
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while head:
            groups.insert(0, head[-2:])
            head = head[:-2]
        digits = ",".join([*groups, tail])
    return f"{sign}₹{digits}.{fraction:02d}"


def redact(text: str) -> str:
    """Defence in depth, NOT a complete bank DLP system. Model sees no ledger data."""
    text = re.sub(r"[\w.+-]+@[\w.-]+", "[identifier]", text)
    text = re.sub(r"\b[A-Z]{5}\d{4}[A-Z]\b", "[identifier]", text, flags=re.I)
    text = re.sub(r"(?:\+?\d[\s-]?){7,}", "[number]", text)
    return text[:1000]


class Conflict(ValueError):
    """An idempotency key conflicts with existing state."""


class NotFound(ValueError):
    """A resource does not exist within the authenticated scope."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
