# Webhooks for DynamoDB Registry Store

## Summary

Implement webhook CRUD operations in `DynamoDBRegistryStore` to pass all 20
compatibility tests currently marked `xfail` plus the workspace-scoped webhook
test.  Adds a new `list_webhooks_by_event` unit test since no vendored
compatibility test covers that method.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Secret encryption | Fernet via `MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY` env var | Matches MLflow SQL store pattern (`mlflow.environment_variables`); no extra AWS dependency |
| Webhook ID format | ULID (`generate_ulid()`) | Consistent with rest of DynamoDB store (model versions, gateway entities) |
| Partition design | One partition per webhook (META + EVENT items) | Follows gateway entity pattern; events are first-class items |
| GSI strategy | Reuse existing gsi2/gsi3 via sparse indexing | No new GSI definitions needed |
| Soft delete | Remove GSI keys + delete EVENT items | Matches existing model version soft-delete pattern |
| `test_webhook` | Not implemented | REST-only concern; not implemented in SqlAlchemyStore either |

## DynamoDB Data Model

### Partition & Sort Key Layout

```
# Webhook META item
PK:  WH#{webhook_ulid}
SK:  WH#META
Attributes:
  name              (S)  — webhook name
  url               (S)  — endpoint URL
  description       (S)  — optional description
  status            (S)  — "ACTIVE" or "DISABLED"
  encrypted_secret  (S)  — Fernet-encrypted secret (optional)
  creation_timestamp      (N)  — epoch millis
  last_updated_timestamp  (N)  — epoch millis
  workspace         (S)  — workspace name
  deleted_timestamp (N)  — set on soft delete (sparse)
  ttl               (N)  — optional TTL for eventual hard-delete (sparse)
  gsi2pk            (S)  — WEBHOOKS#{workspace}  (removed on delete)
  gsi2sk            (S)  — {webhook_ulid}        (removed on delete)

# Webhook EVENT items (one per subscribed event)
PK:  WH#{webhook_ulid}
SK:  WH#EVT#{entity}#{action}
Attributes:
  entity    (S)  — e.g. "REGISTERED_MODEL"
  action    (S)  — e.g. "CREATED"
  workspace (S)  — workspace name
  gsi3pk    (S)  — WH_EVT#{workspace}#{entity}#{action}
  gsi3sk    (S)  — {webhook_ulid}
```

### GSI Usage (Sparse Indexes)

| GSI | Purpose | PK Value | SK Value | Items |
|-----|---------|----------|----------|-------|
| gsi2 | `list_webhooks` | `WEBHOOKS#{workspace}` | `{webhook_ulid}` | META only |
| gsi3 | `list_webhooks_by_event` | `WH_EVT#{workspace}#{entity}#{action}` | `{webhook_ulid}` | EVENT only |

## Access Patterns

| Access Pattern | Method | Table/Index | Key Condition | Filter |
|---------------|--------|-------------|---------------|--------|
| Create webhook | `create_webhook` | Base table | `batch_write`: PK=`WH#{ulid}`, SK=`WH#META` + N × SK=`WH#EVT#{entity}#{action}` | — |
| Get webhook by ID | `get_webhook` | Base table | Query PK=`WH#{ulid}`, SK begins_with `WH#` | `deleted_timestamp` is null |
| List all webhooks in workspace | `list_webhooks` | gsi2 | PK=`WEBHOOKS#{workspace}` | — (deleted items have GSI keys removed) |
| List webhooks by event | `list_webhooks_by_event` | gsi3 | PK=`WH_EVT#{workspace}#{entity}#{action}` | — (deleted event items are hard-deleted) |
| Update webhook fields | `update_webhook` | Base table | `update_item` PK=`WH#{ulid}`, SK=`WH#META` | — |
| Update webhook events | `update_webhook` (events changed) | Base table | `batch_delete` old EVTs + `batch_write` new EVTs | — |
| Soft-delete webhook | `delete_webhook` | Base table | `update_item` META: set `deleted_timestamp`, remove `gsi2pk`/`gsi2sk` + `batch_delete` all EVT items | — |

## Store Methods

### Internal Helpers

```python
def _get_webhook_items(self, webhook_ulid: str) -> list[dict]:
    """Query PK=WH#{webhook_ulid}, returns META + EVENT items."""

def _webhook_items_to_entity(self, items: list[dict]) -> Webhook:
    """Classify items by SK prefix (WH#META vs WH#EVT#), decrypt secret,
    assemble Webhook entity."""

def _get_webhook_by_id(self, webhook_id: str) -> Webhook:
    """Calls _get_webhook_items, raises RESOURCE_DOES_NOT_EXIST if not
    found or soft-deleted."""
```

### Public Methods

| # | Method | Signature | DynamoDB Operations |
|---|--------|-----------|-------------------|
| 1 | `create_webhook` | `(name, url, events, description=None, secret=None, status=None) -> Webhook` | Validate → `batch_write` (1 META + N EVT items) |
| 2 | `get_webhook` | `(webhook_id) -> Webhook` | `_get_webhook_by_id` → partition query |
| 3 | `list_webhooks` | `(max_results=None, page_token=None) -> PagedList[Webhook]` | gsi2 query `WEBHOOKS#{workspace}`, fetch full partitions |
| 4 | `list_webhooks_by_event` | `(event, max_results=None, page_token=None) -> PagedList[Webhook]` | gsi3 query `WH_EVT#{workspace}#{entity}#{action}`, fetch META for each |
| 5 | `update_webhook` | `(webhook_id, name=None, description=None, url=None, events=None, secret=None, status=None) -> Webhook` | `_get_webhook_by_id` → `update_item` META + if events changed: delete old EVTs, write new EVTs |
| 6 | `delete_webhook` | `(webhook_id) -> None` | `_get_webhook_by_id` → `update_item` META (set `deleted_timestamp`, remove GSI keys) + `batch_delete` EVT items |

### Not Implemented

| Method | Reason |
|--------|--------|
| `test_webhook` | REST-only; not implemented in SqlAlchemyStore |

## Validation

Reuse MLflow's existing validators directly:

- `_validate_webhook_name(name)` — regex, 63-char max, starts/ends with alphanumeric
- `_validate_webhook_url(url)` — non-empty, valid scheme (HTTP/HTTPS), valid hostname
- `_validate_webhook_events(events)` — non-empty list of `WebhookEvent` instances

## Encryption

- Encrypt secrets on write with `cryptography.fernet.Fernet` using key from
  `MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY` environment variable
  (defined in `mlflow.environment_variables`)
- If env var is not set, generate a key with `Fernet.generate_key()` (matches
  `EncryptedString` behavior in `mlflow.store.model_registry.dbmodels.models`)
- Decrypt on read; return `None` if no secret stored
- Store encrypted value in `encrypted_secret` attribute on META item

## Soft Delete Behavior

On `delete_webhook(webhook_id)`:

1. Verify webhook exists and is not already deleted (`_get_webhook_by_id`)
2. `update_item` on META:
   - SET `deleted_timestamp` = current time millis
   - SET `last_updated_timestamp` = current time millis
   - REMOVE `gsi2pk`, `gsi2sk` (disappears from `list_webhooks`)
   - Optionally SET `ttl` for eventual hard-delete
3. `batch_delete` all EVT items (disappears from `list_webhooks_by_event`)

After deletion:
- `get_webhook(id)` → `RESOURCE_DOES_NOT_EXIST`
- `list_webhooks` → webhook excluded (no GSI2 keys)
- `list_webhooks_by_event` → webhook excluded (EVT items deleted)

## Pagination

- Default `max_results=100`, valid range 1–1000
- Use DynamoDB-native cursor pagination via `_encode_page_token` /
  `_decode_page_token` (base64-encoded `LastEvaluatedKey`) — matches
  existing DynamoDB store pagination pattern
- `list_webhooks`: ordered by `creation_timestamp` DESC (ULID sort key in GSI2
  gives chronological order; scan_forward=false for DESC)

## Schema Constants

New entries in `src/mlflow_dynamodbstore/dynamodb/schema.py`:

```python
# Webhook partition
PK_WEBHOOK_PREFIX = "WH#"
SK_WEBHOOK_META = "WH#META"
SK_WEBHOOK_EVT_PREFIX = "WH#EVT#"

# GSI prefixes for webhooks
GSI2_WEBHOOKS_PREFIX = "WEBHOOKS#"
GSI3_WH_EVT_PREFIX = "WH_EVT#"
```

## Test Plan

| Test File | Changes |
|-----------|---------|
| `tests/compatibility/test_registry_compat.py` | Remove `_xfail_webhook` from 19 webhook tests; replace `test_webhook_secret_encryption` with DynamoDB-specific version (vendored test does raw SQL `SELECT secret FROM webhooks` which won't work against DynamoDB) |
| `tests/compatibility/test_registry_workspace_compat.py` | Remove `xfail` from `test_webhook_operations_are_workspace_scoped` |
| `tests/unit/test_webhook_by_event.py` (new) | Unit tests for `list_webhooks_by_event`: basic filtering, pagination, no results, deleted webhook excluded, multiple events on same webhook |

## Files Modified

| File | Change |
|------|--------|
| `src/mlflow_dynamodbstore/dynamodb/schema.py` | Add 5 webhook constants |
| `src/mlflow_dynamodbstore/registry_store.py` | Add 6 public methods + 3 internal helpers |
| `tests/compatibility/test_registry_compat.py` | Remove xfail from 19 tests; replace `test_webhook_secret_encryption` with DynamoDB-specific version |
| `tests/compatibility/test_registry_workspace_compat.py` | Remove xfail from 1 test |
| `tests/unit/test_webhook_by_event.py` | New — unit tests for `list_webhooks_by_event` |
