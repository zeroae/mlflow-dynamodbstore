# Phase 5a: Gateway Secrets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 5 gateway secret store methods (`create_gateway_secret`, `get_secret_info`, `update_gateway_secret`, `delete_gateway_secret`, `list_secret_infos`) on `DynamoDBTrackingStore`, unlocking 13 compatibility tests.

**Architecture:** Add schema constants, then implement each method directly on `DynamoDBTrackingStore` following the existing DynamoDB operation patterns. Uses MLflow's built-in `KEKManager` and `_encrypt_secret`/`_decrypt_secret` for envelope encryption. Store DynamoDB Binary for encrypted data, native Maps for `masked_value` and `auth_config`.

**Tech Stack:** boto3/DynamoDB (via `DynamoDBTable` wrapper), MLflow crypto utilities, moto for testing

**Spec:** `docs/superpowers/specs/2026-03-17-ai-gateway-design.md` (Secrets sections)

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/mlflow_dynamodbstore/dynamodb/schema.py` | Add gateway schema constants (PK/SK prefixes, GSI prefixes) — only the ones needed for Phase 5a |
| `src/mlflow_dynamodbstore/tracking_store.py` | Implement 5 secret methods on `DynamoDBTrackingStore` |
| `tests/compatibility/test_gateway_compat.py` | Remove xfail from 13 secret tests |

## Key Reference Files

| File | What to learn |
|------|---------------|
| `src/mlflow_dynamodbstore/tracking_store.py` | Existing patterns: `self._table.put_item()`, `self._table.get_item()`, `self._table.query()`, `self._table.update_item()`, `self._table.delete_item()`. Error patterns with `MlflowException`. `self._workspace` property. |
| `src/mlflow_dynamodbstore/dynamodb/table.py` | `DynamoDBTable` method signatures — `put_item(item, condition)`, `get_item(pk, sk)`, `query(pk, index_name, filter_expression)`, `update_item(pk, sk, updates, removes)`, `delete_item(pk, sk)` |
| `src/mlflow_dynamodbstore/ids.py` | `generate_ulid()` for ID generation |
| `.venv/.../mlflow/store/tracking/gateway/sqlalchemy_mixin.py` | Reference implementation for all 5 methods — encryption flow, validation, error handling |
| `.venv/.../mlflow/utils/crypto.py` | `KEKManager`, `_encrypt_secret(value, kek_manager, secret_id, secret_name)`, `_decrypt_secret(...)`, `_mask_secret_value(dict)` |
| `.venv/.../mlflow/entities/gateway_secrets.py` | `GatewaySecretInfo` dataclass — note field is `masked_values` (plural, dict), not `masked_value` (JSON string) |
| `docs/superpowers/specs/2026-03-17-ai-gateway-design.md` | DynamoDB item design, access patterns AP1-AP5, store method pseudocode |

## DynamoDB Item Design (from spec)

```
PK: GW_SECRET#<secret_id>     SK: GW#META

Attributes:
  secret_name, encrypted_value (Binary), wrapped_dek (Binary),
  kek_version, masked_value (Map), provider, auth_config (Map),
  created_at, last_updated_at, created_by, last_updated_by, workspace

GSI1: gsi1pk = GW_SECRET_NAME#<ws>#<secret_name>  gsi1sk = <secret_id>
GSI2: gsi2pk = GW_SECRETS#<ws>                     gsi2sk = <secret_id>
```

---

### Task 1: Add gateway schema constants

**Files:**
- Modify: `src/mlflow_dynamodbstore/dynamodb/schema.py`

- [ ] **Step 1: Add the Phase 5a constants**

Add at the end of `schema.py`, before any trailing newline:

```python
# --- Phase 5: Gateway ---

# Gateway partition prefixes
PK_GW_SECRET_PREFIX = "GW_SECRET#"

# Gateway sort keys
SK_GW_META = "GW#META"

# GSI prefixes for gateway
GSI1_GW_SECRET_NAME_PREFIX = "GW_SECRET_NAME#"
GSI2_GW_SECRETS_PREFIX = "GW_SECRETS#"
GSI4_GW_MODELDEF_SECRET_PREFIX = "GW_MODELDEF_SECRET#"
```

Add constants needed by Phase 5a (including `GSI4_GW_MODELDEF_SECRET_PREFIX` for `delete_gateway_secret`'s orphan cleanup). Phase 5b/5c will add their own constants.

- [ ] **Step 2: Verify import works**

Run: `uv run python -c "from mlflow_dynamodbstore.dynamodb.schema import PK_GW_SECRET_PREFIX, SK_GW_META, GSI1_GW_SECRET_NAME_PREFIX, GSI2_GW_SECRETS_PREFIX, GSI4_GW_MODELDEF_SECRET_PREFIX; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```
git add src/mlflow_dynamodbstore/dynamodb/schema.py
git commit -m "feat: add gateway secret schema constants"
```

---

### Task 2: Implement `create_gateway_secret`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`

- [ ] **Step 1: Add imports and implement `create_gateway_secret`**

Add imports near the top of `tracking_store.py` (with the other `mlflow` imports):

```python
import json as _json

from mlflow.entities import GatewaySecretInfo
from mlflow.utils.crypto import KEKManager, _encrypt_secret, _mask_secret_value
from mlflow.utils.time import get_current_time_millis
```

Add the schema imports (with the other schema imports):

```python
from mlflow_dynamodbstore.dynamodb.schema import (
    GSI1_GW_SECRET_NAME_PREFIX,
    GSI2_GW_SECRETS_PREFIX,
    GSI4_GW_MODELDEF_SECRET_PREFIX,
    PK_GW_SECRET_PREFIX,
    SK_GW_META,
)
```

Add the method on `DynamoDBTrackingStore` (at the end of the class, before any private helper section):

```python
def create_gateway_secret(
    self,
    secret_name,
    secret_value,
    provider=None,
    auth_config=None,
    created_by=None,
):
    # Check name uniqueness via GSI1
    existing = self._table.query(
        pk=f"{GSI1_GW_SECRET_NAME_PREFIX}{self._workspace}#{secret_name}",
        index_name="gsi1",
        limit=1,
    )
    if existing:
        raise MlflowException(
            f"Secret with name '{secret_name}' already exists",
            error_code=RESOURCE_ALREADY_EXISTS,
        )

    secret_id = f"s-{generate_ulid()}"
    now = get_current_time_millis()

    # Encrypt
    kek_manager = KEKManager()
    value_to_encrypt = _json.dumps(secret_value)
    encrypted = _encrypt_secret(value_to_encrypt, kek_manager, secret_id, secret_name)
    masked_value = _mask_secret_value(secret_value)

    item = {
        "PK": f"{PK_GW_SECRET_PREFIX}{secret_id}",
        "SK": SK_GW_META,
        "secret_name": secret_name,
        "encrypted_value": encrypted.encrypted_value,
        "wrapped_dek": encrypted.wrapped_dek,
        "kek_version": encrypted.kek_version,
        "masked_value": masked_value,
        "created_at": now,
        "last_updated_at": now,
        "workspace": self._workspace,
        # GSI projections
        "gsi1pk": f"{GSI1_GW_SECRET_NAME_PREFIX}{self._workspace}#{secret_name}",
        "gsi1sk": secret_id,
        "gsi2pk": f"{GSI2_GW_SECRETS_PREFIX}{self._workspace}",
        "gsi2sk": secret_id,
    }
    if provider is not None:
        item["provider"] = provider
    if auth_config is not None:
        item["auth_config"] = auth_config
    if created_by is not None:
        item["created_by"] = created_by
        item["last_updated_by"] = created_by

    self._table.put_item(item)
    self._invalidate_secret_cache()

    return GatewaySecretInfo(
        secret_id=secret_id,
        secret_name=secret_name,
        masked_values=masked_value,
        created_at=now,
        last_updated_at=now,
        provider=provider,
        auth_config=auth_config,
        workspace=self._workspace,
        created_by=created_by,
        last_updated_by=created_by,
    )
```

- [ ] **Step 2: Run the create secret compat tests**

Run: `uv run pytest tests/compatibility/test_gateway_compat.py -k "test_create_gateway_secret" -v`
Expected: The 3 create tests pass (they're still marked xfail so they'll show as `xpass`). If they xpass, that confirms the implementation works.

- [ ] **Step 3: Commit**

```
git add src/mlflow_dynamodbstore/tracking_store.py
git commit -m "feat: implement create_gateway_secret"
```

---

### Task 3: Implement `get_secret_info`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`

- [ ] **Step 1: Add `get_secret_info` and `_get_secret_item` helper**

Add a private helper for reuse by `update` and `delete`:

```python
def _get_secret_item(self, secret_id):
    """Fetch raw DynamoDB item for a secret by ID. Returns None if not found."""
    return self._table.get_item(
        pk=f"{PK_GW_SECRET_PREFIX}{secret_id}",
        sk=SK_GW_META,
    )

def _secret_item_to_entity(self, item):
    """Convert a raw DynamoDB secret item to a GatewaySecretInfo entity."""
    secret_id = item["PK"].removeprefix(PK_GW_SECRET_PREFIX)
    return GatewaySecretInfo(
        secret_id=secret_id,
        secret_name=item["secret_name"],
        masked_values=item["masked_value"],
        created_at=int(item["created_at"]),
        last_updated_at=int(item["last_updated_at"]),
        provider=item.get("provider"),
        auth_config=item.get("auth_config"),
        workspace=item.get("workspace"),
        created_by=item.get("created_by"),
        last_updated_by=item.get("last_updated_by"),
    )

def get_secret_info(self, secret_id=None, secret_name=None):
    # Validate exactly one of secret_id or secret_name
    if (secret_id is None) == (secret_name is None):
        raise MlflowException(
            "Exactly one of `secret_id` or `secret_name` must be specified",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if secret_id:
        item = self._get_secret_item(secret_id)
    else:
        # Look up by name via GSI1
        results = self._table.query(
            pk=f"{GSI1_GW_SECRET_NAME_PREFIX}{self._workspace}#{secret_name}",
            index_name="gsi1",
            limit=1,
        )
        if not results:
            raise MlflowException(
                f"Secret with name '{secret_name}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        found_secret_id = results[0]["gsi1sk"]
        item = self._get_secret_item(found_secret_id)

    if item is None:
        identifier = secret_id if secret_id else secret_name
        raise MlflowException(
            f"Secret '{identifier}' not found",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    return self._secret_item_to_entity(item)
```

Note: `INVALID_PARAMETER_VALUE` must be imported from `mlflow.protos.databricks_pb2`. Check if it's already imported in `tracking_store.py` — if not, add it.

- [ ] **Step 2: Run the get secret compat tests**

Run: `uv run pytest tests/compatibility/test_gateway_compat.py -k "test_get_gateway_secret_info" -v`
Expected: All 4 get tests pass (xpass).

- [ ] **Step 3: Commit**

```
git add src/mlflow_dynamodbstore/tracking_store.py
git commit -m "feat: implement get_secret_info"
```

---

### Task 4: Implement `update_gateway_secret`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`

- [ ] **Step 1: Add `update_gateway_secret`**

```python
def update_gateway_secret(
    self,
    secret_id,
    secret_value=None,
    auth_config=None,
    updated_by=None,
):
    # Fetch existing to verify it exists and get secret_name for AAD
    item = self._get_secret_item(secret_id)
    if item is None:
        raise MlflowException(
            f"Secret '{secret_id}' not found",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    now = get_current_time_millis()
    updates = {"last_updated_at": now}
    removes = []

    if secret_value is not None:
        kek_manager = KEKManager()
        value_to_encrypt = _json.dumps(secret_value)
        encrypted = _encrypt_secret(
            value_to_encrypt, kek_manager, secret_id, item["secret_name"]
        )
        updates["encrypted_value"] = encrypted.encrypted_value
        updates["wrapped_dek"] = encrypted.wrapped_dek
        updates["kek_version"] = encrypted.kek_version
        updates["masked_value"] = _mask_secret_value(secret_value)

    if auth_config is not None:
        if auth_config:
            updates["auth_config"] = auth_config
        else:
            # Empty dict means clear auth_config
            removes.append("auth_config")

    if updated_by is not None:
        updates["last_updated_by"] = updated_by

    self._table.update_item(
        pk=f"{PK_GW_SECRET_PREFIX}{secret_id}",
        sk=SK_GW_META,
        updates=updates,
        removes=removes if removes else None,
    )

    # Re-fetch to return the full updated entity
    updated_item = self._get_secret_item(secret_id)
    self._invalidate_secret_cache()
    return self._secret_item_to_entity(updated_item)
```

Note: `_invalidate_secret_cache` — check if this is already defined on the class. If not, add a no-op:

```python
def _invalidate_secret_cache(self):
    """Invalidate the gateway secret cache. No-op until cache is implemented."""
    pass
```

- [ ] **Step 2: Run the update secret compat tests**

Run: `uv run pytest tests/compatibility/test_gateway_compat.py -k "test_update_gateway_secret" -v`
Expected: All 3 update tests pass (xpass).

- [ ] **Step 3: Commit**

```
git add src/mlflow_dynamodbstore/tracking_store.py
git commit -m "feat: implement update_gateway_secret"
```

---

### Task 5: Implement `delete_gateway_secret` and `list_secret_infos`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`

- [ ] **Step 1: Add `delete_gateway_secret`**

```python
def delete_gateway_secret(self, secret_id):
    item = self._get_secret_item(secret_id)
    if item is None:
        raise MlflowException(
            f"Secret '{secret_id}' not found",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    # Orphan model definitions that reference this secret (SET NULL behavior)
    model_defs = self._table.query(
        pk=f"{GSI4_GW_MODELDEF_SECRET_PREFIX}{secret_id}",
        index_name="gsi4",
    )
    for md_item in model_defs:
        self._table.update_item(
            pk=md_item["PK"],
            sk=md_item["SK"],
            removes=["secret_id", "gsi4pk", "gsi4sk"],
        )

    self._table.delete_item(
        pk=f"{PK_GW_SECRET_PREFIX}{secret_id}",
        sk=SK_GW_META,
    )
    self._invalidate_secret_cache()
```

- [ ] **Step 2: Add `list_secret_infos`**

```python
def list_secret_infos(self, provider=None):
    from boto3.dynamodb.conditions import Attr

    filter_expr = Attr("provider").eq(provider) if provider else None

    items = self._table.query(
        pk=f"{GSI2_GW_SECRETS_PREFIX}{self._workspace}",
        index_name="gsi2",
        filter_expression=filter_expr,
    )

    # GSI2 has ALL projection — items include all base table attributes
    # (PK, SK, secret_name, masked_value, etc.), no re-fetch needed.
    return [self._secret_item_to_entity(item) for item in items]
```

- [ ] **Step 3: Run delete and list compat tests**

Run: `uv run pytest tests/compatibility/test_gateway_compat.py -k "test_delete_gateway_secret or test_list_gateway_secret" -v`
Expected: Delete and list tests pass (xpass).

- [ ] **Step 4: Commit**

```
git add src/mlflow_dynamodbstore/tracking_store.py src/mlflow_dynamodbstore/dynamodb/schema.py
git commit -m "feat: implement delete_gateway_secret and list_secret_infos"
```

---

### Task 6: Remove xfail from secret compat tests

**Files:**
- Modify: `tests/compatibility/test_gateway_compat.py`

- [ ] **Step 1: Remove the xfail markers from all 13 secret tests**

Remove these lines (the `_xfail_gateway(...)` wrapping for each secret test):

```python
test_create_gateway_secret = _xfail_gateway(test_create_gateway_secret)
test_create_gateway_secret_duplicate_name_raises = _xfail_gateway(...)
test_create_gateway_secret_with_auth_config = _xfail_gateway(...)
test_create_gateway_secret_with_dict_value = _xfail_gateway(...)
test_delete_gateway_secret = _xfail_gateway(test_delete_gateway_secret)
test_get_gateway_secret_info_by_id = _xfail_gateway(...)
test_get_gateway_secret_info_by_name = _xfail_gateway(...)
test_get_gateway_secret_info_not_found = _xfail_gateway(...)
test_get_gateway_secret_info_requires_one_of_id_or_name = _xfail_gateway(...)
test_list_gateway_secret_infos = _xfail_gateway(test_list_gateway_secret_infos)
test_update_gateway_secret = _xfail_gateway(test_update_gateway_secret)
test_update_gateway_secret_clear_auth_config = _xfail_gateway(...)
test_update_gateway_secret_with_auth_config = _xfail_gateway(...)
```

Keep the `_xfail_gateway` marker definition and all non-secret xfail lines.

- [ ] **Step 2: Run all 13 secret tests**

Run: `uv run pytest tests/compatibility/test_gateway_compat.py -k "secret" -v`
Expected: 26 passed (13 tests x 2 workspace params). Zero xfail, zero failed.

- [ ] **Step 3: Run full gateway compat suite to confirm nothing else broke**

Run: `uv run pytest tests/compatibility/test_gateway_compat.py -v --tb=short 2>&1 | tail -5`
Expected: 26 passed, 84 xfailed (42 non-secret tests x 2 workspace params). Zero failed.

- [ ] **Step 4: Commit**

```
git add tests/compatibility/test_gateway_compat.py
git commit -m "test: remove xfail from 13 gateway secret compat tests"
```
