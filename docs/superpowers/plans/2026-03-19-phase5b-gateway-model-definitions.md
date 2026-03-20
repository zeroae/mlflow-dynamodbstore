# Phase 5b: Gateway Model Definitions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 5 gateway model definition store methods (`create_gateway_model_definition`, `get_gateway_model_definition`, `list_gateway_model_definitions`, `update_gateway_model_definition`, `delete_gateway_model_definition`) on `DynamoDBTrackingStore`, unlocking 10 compatibility tests.

**Architecture:** Follow Phase 5a patterns — add schema constants, implement each method on `DynamoDBTrackingStore` using the `DynamoDBTable` wrapper. Model definitions reference secrets (resolve `secret_name` from secret META). Delete uses RESTRICT check via GSI5 (endpoints using the model def must be detached first).

**Tech Stack:** boto3/DynamoDB (via `DynamoDBTable` wrapper), moto for testing

**Spec:** `docs/superpowers/specs/2026-03-17-ai-gateway-design.md` (Model Definitions sections)

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/mlflow_dynamodbstore/dynamodb/schema.py` | Add Phase 5b schema constants |
| `src/mlflow_dynamodbstore/tracking_store.py` | Implement 5 model definition methods + helpers |
| `tests/compatibility/test_gateway_compat.py` | Remove xfail from 10 model definition tests |

## Key Reference Files

| File | What to learn |
|------|---------------|
| `src/mlflow_dynamodbstore/tracking_store.py:5234-5470` | Phase 5a gateway secret methods — follow same patterns for helpers, entity construction, error handling |
| `docs/superpowers/specs/2026-03-17-ai-gateway-design.md:60-64` | ModelDef item design (PK/SK/attributes) |
| `docs/superpowers/specs/2026-03-17-ai-gateway-design.md:128-137` | Access patterns AP6-AP11 |
| `docs/superpowers/specs/2026-03-17-ai-gateway-design.md:249-286` | Store method pseudocode |
| `.venv/.../mlflow/entities/gateway_model_definition.py` | `GatewayModelDefinition` dataclass fields |

## DynamoDB Item Design (from spec)

```
PK: GW_MODELDEF#<model_def_id>     SK: GW#META

Attributes:
  name, secret_id, provider, model_name,
  created_at, last_updated_at, created_by, last_updated_by, workspace

GSI2: gsi2pk = GW_MODELDEFS#<ws>                          gsi2sk = <model_def_id>
GSI3: gsi3pk = GW_MODELDEF_NAME#<ws>#<name>               gsi3sk = <model_def_id>
GSI4: gsi4pk = GW_MODELDEF_SECRET#<secret_id>             gsi4sk = <model_def_id>
```

## Key Design Decisions

- **`secret_name` resolution**: `GatewayModelDefinition` has both `secret_id` and `secret_name`. DynamoDB stores only `secret_id` on the model def item. `secret_name` is resolved by reading the secret META via `_get_secret_item(secret_id)`. If `secret_id` is None (orphaned), `secret_name` is None.
- **`delete` uses RESTRICT**: Query GSI5 to check if any endpoints use this model def. If any found, raise `INVALID_STATE`. The `delete_gateway_model_definition_in_use_raises` test depends on `create_gateway_endpoint` which isn't implemented yet (Phase 5c). This test will need a separate xfail.
- **`update` GSI projection management**: When `name` changes, `gsi3pk` must be updated. When `secret_id` changes, `gsi4pk`/`gsi4sk` must be updated (or removed if set to None).

---

### Task 1: Add Phase 5b schema constants

**Files:**
- Modify: `src/mlflow_dynamodbstore/dynamodb/schema.py`

- [ ] **Step 1: Add the Phase 5b constants**

Append to the `# --- Phase 5: Gateway ---` section in `schema.py`:

```python
PK_GW_MODELDEF_PREFIX = "GW_MODELDEF#"
GSI2_GW_MODELDEFS_PREFIX = "GW_MODELDEFS#"
GSI3_GW_MODELDEF_NAME_PREFIX = "GW_MODELDEF_NAME#"
GSI5_GW_ENDPOINT_MODELDEF_PREFIX = "GW_ENDPOINT_MODELDEF#"
```

Note: `GSI4_GW_MODELDEF_SECRET_PREFIX` and `SK_GW_META` already exist from Phase 5a.

- [ ] **Step 2: Verify import works**

Run: `uv run python -c "from mlflow_dynamodbstore.dynamodb.schema import PK_GW_MODELDEF_PREFIX, GSI2_GW_MODELDEFS_PREFIX, GSI3_GW_MODELDEF_NAME_PREFIX, GSI5_GW_ENDPOINT_MODELDEF_PREFIX; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```
git add src/mlflow_dynamodbstore/dynamodb/schema.py
git commit -m "feat: add gateway model definition schema constants"
```

---

### Task 2: Implement `create_gateway_model_definition`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`

- [ ] **Step 1: Add imports and implement**

Add to the existing schema imports block:

```python
from mlflow_dynamodbstore.dynamodb.schema import (
    GSI2_GW_MODELDEFS_PREFIX,
    GSI3_GW_MODELDEF_NAME_PREFIX,
    GSI5_GW_ENDPOINT_MODELDEF_PREFIX,
    PK_GW_MODELDEF_PREFIX,
)
```

Add to the existing mlflow entity imports:

```python
from mlflow.entities import GatewayModelDefinition
```

Add a new section after the Gateway Secrets section (after `list_secret_infos`):

```python
# -----------------------------------------------------------------------
# Gateway Model Definitions
# -----------------------------------------------------------------------

def _resolve_secret_name(self, secret_id):
    """Resolve secret_name from secret_id. Returns None if secret_id is None or not found."""
    if not secret_id:
        return None
    item = self._get_secret_item(secret_id)
    return item["secret_name"] if item else None

def create_gateway_model_definition(
    self,
    name,
    secret_id,
    provider,
    model_name,
    created_by=None,
):
    # Check name uniqueness via GSI3
    existing = self._table.query(
        pk=f"{GSI3_GW_MODELDEF_NAME_PREFIX}{self._workspace}#{name}",
        index_name="gsi3",
        limit=1,
    )
    if existing:
        raise MlflowException(
            f"Model definition with name '{name}' already exists",
            error_code=RESOURCE_ALREADY_EXISTS,
        )

    # Verify secret exists
    secret_item = self._get_secret_item(secret_id)
    if secret_item is None:
        raise MlflowException(
            f"Secret '{secret_id}' not found",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )
    secret_name = secret_item["secret_name"]

    model_definition_id = f"d-{generate_ulid()}"
    now = get_current_time_millis()

    item = {
        "PK": f"{PK_GW_MODELDEF_PREFIX}{model_definition_id}",
        "SK": SK_GW_META,
        "name": name,
        "secret_id": secret_id,
        "provider": provider,
        "model_name": model_name,
        "created_at": now,
        "last_updated_at": now,
        "workspace": self._workspace,
        # GSI projections
        "gsi2pk": f"{GSI2_GW_MODELDEFS_PREFIX}{self._workspace}",
        "gsi2sk": model_definition_id,
        "gsi3pk": f"{GSI3_GW_MODELDEF_NAME_PREFIX}{self._workspace}#{name}",
        "gsi3sk": model_definition_id,
        "gsi4pk": f"{GSI4_GW_MODELDEF_SECRET_PREFIX}{secret_id}",
        "gsi4sk": model_definition_id,
    }
    if created_by is not None:
        item["created_by"] = created_by
        item["last_updated_by"] = created_by

    self._table.put_item(item)

    return GatewayModelDefinition(
        model_definition_id=model_definition_id,
        name=name,
        secret_id=secret_id,
        secret_name=secret_name,
        provider=provider,
        model_name=model_name,
        created_at=now,
        last_updated_at=now,
        created_by=created_by,
        last_updated_by=created_by,
        workspace=self._workspace,
    )
```

- [ ] **Step 2: Run create tests**

Run: `uv run pytest tests/compatibility/test_gateway_compat.py -k "test_create_gateway_model_definition" -v`
Expected: `test_create_gateway_model_definition` and `test_create_gateway_model_definition_nonexistent_secret_raises` xpass. `test_create_gateway_model_definition_duplicate_name_raises` xpasses.

- [ ] **Step 3: Commit**

```
git add src/mlflow_dynamodbstore/tracking_store.py
git commit -m "feat: implement create_gateway_model_definition"
```

---

### Task 3: Implement `get_gateway_model_definition`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`

- [ ] **Step 1: Add helpers and implement**

```python
def _get_model_def_item(self, model_definition_id):
    """Fetch raw DynamoDB item for a model definition by ID. Returns None if not found."""
    return self._table.get_item(
        pk=f"{PK_GW_MODELDEF_PREFIX}{model_definition_id}",
        sk=SK_GW_META,
    )

def _model_def_item_to_entity(self, item):
    """Convert a raw DynamoDB model definition item to a GatewayModelDefinition entity."""
    model_definition_id = item["PK"].removeprefix(PK_GW_MODELDEF_PREFIX)
    secret_id = item.get("secret_id")
    return GatewayModelDefinition(
        model_definition_id=model_definition_id,
        name=item["name"],
        secret_id=secret_id,
        secret_name=self._resolve_secret_name(secret_id),
        provider=item["provider"],
        model_name=item["model_name"],
        created_at=int(item["created_at"]),
        last_updated_at=int(item["last_updated_at"]),
        created_by=item.get("created_by"),
        last_updated_by=item.get("last_updated_by"),
        workspace=item.get("workspace"),
    )

def get_gateway_model_definition(self, model_definition_id=None, name=None):
    if (model_definition_id is None) == (name is None):
        raise MlflowException(
            "Exactly one of `model_definition_id` or `name` must be specified",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if model_definition_id:
        item = self._get_model_def_item(model_definition_id)
    else:
        results = self._table.query(
            pk=f"{GSI3_GW_MODELDEF_NAME_PREFIX}{self._workspace}#{name}",
            index_name="gsi3",
            limit=1,
        )
        if not results:
            raise MlflowException(
                f"Model definition with name '{name}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        found_id = results[0]["gsi3sk"]
        item = self._get_model_def_item(found_id)

    if item is None:
        identifier = model_definition_id if model_definition_id else name
        raise MlflowException(
            f"Model definition '{identifier}' not found",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    return self._model_def_item_to_entity(item)
```

- [ ] **Step 2: Run get tests**

Run: `uv run pytest tests/compatibility/test_gateway_compat.py -k "test_get_gateway_model_definition" -v`
Expected: All 3 get tests xpass.

- [ ] **Step 3: Commit**

```
git add src/mlflow_dynamodbstore/tracking_store.py
git commit -m "feat: implement get_gateway_model_definition"
```

---

### Task 4: Implement `list_gateway_model_definitions`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`

- [ ] **Step 1: Implement**

```python
def list_gateway_model_definitions(self, provider=None, secret_id=None):
    if secret_id:
        # Direct lookup via GSI4
        items = self._table.query(
            pk=f"{GSI4_GW_MODELDEF_SECRET_PREFIX}{secret_id}",
            index_name="gsi4",
        )
    else:
        # List all via GSI2
        items = self._table.query(
            pk=f"{GSI2_GW_MODELDEFS_PREFIX}{self._workspace}",
            index_name="gsi2",
        )

    # GSI has ALL projection — items include all base table attributes
    model_defs = [self._model_def_item_to_entity(item) for item in items]

    # Apply provider filter in-memory if specified
    if provider is not None:
        model_defs = [md for md in model_defs if md.provider == provider]

    return model_defs
```

- [ ] **Step 2: Run list tests**

Run: `uv run pytest tests/compatibility/test_gateway_compat.py -k "test_list_gateway_model_definitions" -v`
Expected: List test xpasses.

- [ ] **Step 3: Commit**

```
git add src/mlflow_dynamodbstore/tracking_store.py
git commit -m "feat: implement list_gateway_model_definitions"
```

---

### Task 5: Implement `update_gateway_model_definition`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`

- [ ] **Step 1: Implement**

```python
def update_gateway_model_definition(
    self,
    model_definition_id,
    name=None,
    secret_id=None,
    model_name=None,
    updated_by=None,
    provider=None,
):
    item = self._get_model_def_item(model_definition_id)
    if item is None:
        raise MlflowException(
            f"Model definition '{model_definition_id}' not found",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    now = get_current_time_millis()
    updates = {"last_updated_at": now}
    removes = []

    if name is not None:
        # Check new name uniqueness
        existing = self._table.query(
            pk=f"{GSI3_GW_MODELDEF_NAME_PREFIX}{self._workspace}#{name}",
            index_name="gsi3",
            limit=1,
        )
        if existing and existing[0]["gsi3sk"] != model_definition_id:
            raise MlflowException(
                f"Model definition with name '{name}' already exists",
                error_code=RESOURCE_ALREADY_EXISTS,
            )
        updates["name"] = name
        updates["gsi3pk"] = f"{GSI3_GW_MODELDEF_NAME_PREFIX}{self._workspace}#{name}"

    if secret_id is not None:
        # Verify new secret exists
        secret_item = self._get_secret_item(secret_id)
        if secret_item is None:
            raise MlflowException(
                f"Secret '{secret_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        updates["secret_id"] = secret_id
        updates["gsi4pk"] = f"{GSI4_GW_MODELDEF_SECRET_PREFIX}{secret_id}"
        updates["gsi4sk"] = model_definition_id

    if model_name is not None:
        updates["model_name"] = model_name

    if provider is not None:
        updates["provider"] = provider

    if updated_by is not None:
        updates["last_updated_by"] = updated_by

    self._table.update_item(
        pk=f"{PK_GW_MODELDEF_PREFIX}{model_definition_id}",
        sk=SK_GW_META,
        updates=updates,
        removes=removes if removes else None,
    )

    updated_item = self._get_model_def_item(model_definition_id)
    self._invalidate_secret_cache()
    return self._model_def_item_to_entity(updated_item)
```

- [ ] **Step 2: Run update tests**

Run: `uv run pytest tests/compatibility/test_gateway_compat.py -k "test_update_gateway_model_definition" -v`
Expected: Update test xpasses.

- [ ] **Step 3: Commit**

```
git add src/mlflow_dynamodbstore/tracking_store.py
git commit -m "feat: implement update_gateway_model_definition"
```

---

### Task 6: Implement `delete_gateway_model_definition`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`

- [ ] **Step 1: Implement**

```python
def delete_gateway_model_definition(self, model_definition_id):
    item = self._get_model_def_item(model_definition_id)
    if item is None:
        raise MlflowException(
            f"Model definition '{model_definition_id}' not found",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    # RESTRICT: check if any endpoints use this model def via GSI5
    endpoints = self._table.query(
        pk=f"{GSI5_GW_ENDPOINT_MODELDEF_PREFIX}{model_definition_id}",
        index_name="gsi5",
    )
    if endpoints:
        raise MlflowException(
            "Cannot delete model definition that is currently in use by endpoints. "
            "Detach it from all endpoints first.",
            error_code=INVALID_STATE,
        )

    self._table.delete_item(
        pk=f"{PK_GW_MODELDEF_PREFIX}{model_definition_id}",
        sk=SK_GW_META,
    )
    self._invalidate_secret_cache()
```

`INVALID_STATE` is not currently imported in `tracking_store.py`. Add it to the existing `from mlflow.protos.databricks_pb2 import` block:

```python
from mlflow.protos.databricks_pb2 import (
    INVALID_STATE,
    # ... existing imports
)
```

- [ ] **Step 2: Run delete tests**

Run: `uv run pytest tests/compatibility/test_gateway_compat.py -k "test_delete_gateway_model_definition and not in_use" -v`
Expected: `test_delete_gateway_model_definition` xpasses.

Note: `test_delete_gateway_model_definition_in_use_raises` calls `create_gateway_endpoint` (Phase 5c) so it will fail with `NotImplementedError`, not pass. It stays xfailed.

- [ ] **Step 3: Commit**

```
git add src/mlflow_dynamodbstore/tracking_store.py
git commit -m "feat: implement delete_gateway_model_definition"
```

---

### Task 7: Remove xfail from model definition compat tests

**Files:**
- Modify: `tests/compatibility/test_gateway_compat.py`

- [ ] **Step 1: Remove xfail markers from 9 of 10 model definition tests**

Remove the `_xfail_gateway(...)` wrapping for these 9 tests:

```
test_create_gateway_model_definition
test_create_gateway_model_definition_duplicate_name_raises
test_create_gateway_model_definition_nonexistent_secret_raises
test_delete_gateway_model_definition
test_get_gateway_model_definition_by_id
test_get_gateway_model_definition_by_name
test_get_gateway_model_definition_requires_one_of_id_or_name
test_list_gateway_model_definitions
test_update_gateway_model_definition
```

**Keep xfail** on `test_delete_gateway_model_definition_in_use_raises` — it depends on `create_gateway_endpoint` (Phase 5c). Change its xfail reason:

```python
test_delete_gateway_model_definition_in_use_raises = pytest.mark.xfail(
    raises=NotImplementedError,
    reason="Depends on create_gateway_endpoint (Phase 5c)",
)(test_delete_gateway_model_definition_in_use_raises)
```

- [ ] **Step 2: Run model definition tests**

Run: `uv run pytest tests/compatibility/test_gateway_compat.py -k "model_definition" -v`
Expected: 18 passed (9 tests x 2 workspace params), 2 xfailed (`in_use_raises` x 2 workspace params).

- [ ] **Step 3: Run full gateway compat suite**

Run: `uv run pytest tests/compatibility/test_gateway_compat.py -q`
Expected: 44 passed (22 tests x 2), 66 xfailed (33 remaining x 2). Zero failed.

- [ ] **Step 4: Commit**

```
git add tests/compatibility/test_gateway_compat.py
git commit -m "test: remove xfail from 9 gateway model definition compat tests"
```
