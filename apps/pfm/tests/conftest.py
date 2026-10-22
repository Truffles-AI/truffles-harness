from datetime import UTC, datetime

import pytest
from truffles_pfm.domain import Intent, Scope
from truffles_pfm.repository import Repository
from truffles_pfm.service import PFM


class StubAI:
    enabled = True

    def __init__(self):
        self.classification = None
        self.selected = Intent(intent="summary")
        self.calls = 0

    def classify(self, merchant):
        self.calls += 1
        return self.classification

    def intent(self, message, month, previous=None, *, consent=True):
        self.calls += int(consent)
        return self.selected, "truffles" if consent else "fallback"


@pytest.fixture
def repo():
    value = Repository(":memory:")
    yield value
    value.close()


@pytest.fixture
def ai():
    return StubAI()


@pytest.fixture
def scope():
    return Scope(bank_id="test-bank", customer_id="alice")


@pytest.fixture
def service(repo, ai):
    return PFM(repo, ai, clock=lambda: datetime(2026, 10, 21, 12, tzinfo=UTC))
