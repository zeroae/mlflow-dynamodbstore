"""Unit tests for list_webhooks_by_event (no vendored compatibility test exists)."""

import pytest
from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent
from mlflow.exceptions import MlflowException


@pytest.fixture(autouse=True)
def _set_encryption_key(monkeypatch):
    """Set a fixed encryption key for deterministic tests."""
    from cryptography.fernet import Fernet

    monkeypatch.setenv("MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY", Fernet.generate_key().decode())


def _make_event(entity=WebhookEntity.MODEL_VERSION, action=WebhookAction.CREATED):
    return WebhookEvent(entity, action)


class TestListWebhooksByEvent:
    def test_basic_filtering(self, registry_store):
        """Webhooks are returned only when they subscribe to the queried event."""
        event_mv_created = _make_event(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)
        event_rm_created = _make_event(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED)

        registry_store.create_webhook(
            name="hook1", url="https://example.com/1", events=[event_mv_created]
        )
        registry_store.create_webhook(
            name="hook2", url="https://example.com/2", events=[event_rm_created]
        )

        result = registry_store.list_webhooks_by_event(event_mv_created)
        assert len(result) == 1
        assert result[0].name == "hook1"

        result = registry_store.list_webhooks_by_event(event_rm_created)
        assert len(result) == 1
        assert result[0].name == "hook2"

    def test_no_results(self, registry_store):
        """Returns empty list when no webhooks subscribe to the event."""
        event = _make_event(WebhookEntity.MODEL_VERSION_TAG, WebhookAction.DELETED)
        result = registry_store.list_webhooks_by_event(event)
        assert len(result) == 0
        assert result.token is None

    def test_multiple_events_on_same_webhook(self, registry_store):
        """A webhook with multiple events appears in queries for each event."""
        event1 = _make_event(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)
        event2 = _make_event(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED)

        registry_store.create_webhook(
            name="multi", url="https://example.com/multi", events=[event1, event2]
        )

        result1 = registry_store.list_webhooks_by_event(event1)
        assert len(result1) == 1
        assert result1[0].name == "multi"
        assert len(result1[0].events) == 2

        result2 = registry_store.list_webhooks_by_event(event2)
        assert len(result2) == 1
        assert result2[0].name == "multi"

    def test_deleted_webhook_excluded(self, registry_store):
        """Deleted webhooks do not appear in list_webhooks_by_event."""
        event = _make_event()
        wh = registry_store.create_webhook(
            name="to_delete", url="https://example.com/del", events=[event]
        )
        registry_store.delete_webhook(wh.webhook_id)

        result = registry_store.list_webhooks_by_event(event)
        assert len(result) == 0

    def test_pagination(self, registry_store):
        """Pagination works correctly for list_webhooks_by_event."""
        event = _make_event()
        for i in range(5):
            registry_store.create_webhook(
                name=f"hook{i}", url=f"https://example.com/{i}", events=[event]
            )

        page1 = registry_store.list_webhooks_by_event(event, max_results=2)
        assert len(page1) == 2
        assert page1.token is not None

        page2 = registry_store.list_webhooks_by_event(event, max_results=2, page_token=page1.token)
        assert len(page2) == 2
        assert page2.token is not None

        # No duplicates
        page1_ids = {w.webhook_id for w in page1}
        page2_ids = {w.webhook_id for w in page2}
        assert page1_ids.isdisjoint(page2_ids)

    def test_invalid_max_results(self, registry_store):
        """max_results must be between 1 and 1000."""
        event = _make_event()
        with pytest.raises(MlflowException, match="max_results must be between 1 and 1000"):
            registry_store.list_webhooks_by_event(event, max_results=0)
        with pytest.raises(MlflowException, match="max_results must be between 1 and 1000"):
            registry_store.list_webhooks_by_event(event, max_results=1001)
