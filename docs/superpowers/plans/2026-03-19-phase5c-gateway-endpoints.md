# Phase 5c: Gateway Endpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 12 gateway endpoint store methods across 4 categories — Endpoints (5), Model Attachment (2), Bindings (3), Tags (2) — on `DynamoDBTrackingStore`, unlocking 31 remaining xfailed tests plus 1 from Phase 5b (`test_delete_gateway_model_definition_in_use_raises`) = 32 total.

**Architecture:** All endpoint children (mappings, bindings, tags) live under a single DynamoDB partition `PK=GW_ENDPOINT#<endpoint_id>` with different SK prefixes. `get_gateway_endpoint` queries the full partition and classifies items by SK prefix. Model mappings carry GSI5 projections for RESTRICT delete checks. Bindings carry GSI2 projections for reverse lookups.

**Tech Stack:** boto3/DynamoDB (via `DynamoDBTable` wrapper), MLflow entity classes, moto for testing

**Spec:** `docs/superpowers/specs/2026-03-17-ai-gateway-design.md` (Endpoints, Model Attachment, Bindings, Tags sections — access patterns AP12-AP27)

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/mlflow_dynamodbstore/dynamodb/schema.py` | Add Phase 5c schema constants (endpoint PK prefix, SK prefixes for MAP/BIND/TAG, GSI prefixes) |
| `src/mlflow_dynamodbstore/tracking_store.py` | Implement 12 endpoint methods on `DynamoDBTrackingStore` |
| `tests/compatibility/test_gateway_compat.py` | Remove xfail from 32 tests (31 Phase 5c + 1 Phase 5b) |

## Key Reference Files

| File | What to learn |
|------|---------------|
| `src/mlflow_dynamodbstore/tracking_store.py` | Existing gateway patterns: `_get_secret_item`, `_secret_item_to_entity`, `_get_model_def_item`, `_model_def_item_to_entity`, `_UNSET` sentinel, `_invalidate_secret_cache`, `_resolve_secret_name` |
| `src/mlflow_dynamodbstore/dynamodb/table.py` | `DynamoDBTable` method signatures — `put_item(item, condition)`, `get_item(pk, sk)`, `query(pk, sk_prefix, index_name)`, `update_item(pk, sk, updates, removes)`, `delete_item(pk, sk)`, `batch_write(items)`, `batch_delete(keys)` |
| `src/mlflow_dynamodbstore/ids.py` | `generate_ulid()` for ID generation |
| `vendor/mlflow/mlflow/store/tracking/gateway/sqlalchemy_mixin.py` | Reference implementation for all 12 methods |
| `vendor/mlflow/mlflow/entities/gateway_endpoint.py` | Entity dataclass definitions — `GatewayEndpoint`, `GatewayEndpointModelMapping`, `GatewayEndpointBinding`, `GatewayEndpointTag`, `GatewayEndpointModelConfig`, `RoutingStrategy`, `FallbackConfig`, `FallbackStrategy`, `GatewayModelLinkageType`, `GatewayResourceType` |
| `docs/superpowers/specs/2026-03-17-ai-gateway-design.md` | DynamoDB item design, access patterns AP12-AP27, store method pseudocode, cache invalidation table |

## DynamoDB Item Design (from spec)

```
PK: GW_ENDPOINT#<endpoint_id>     SK: GW#META
Attributes:
  name, routing_strategy, fallback_config (Map), experiment_id,
  usage_tracking (bool), created_at, last_updated_at, created_by,
  last_updated_by, workspace, tags (denormalized dict)
GSI2: gsi2pk = GW_ENDPOINTS#<ws>              gsi2sk = <endpoint_id>
GSI3: gsi3pk = GW_ENDPOINT_NAME#<ws>#<name>   gsi3sk = <endpoint_id>

PK: GW_ENDPOINT#<endpoint_id>     SK: GW#MAP#<model_def_id>#<linkage_type>
Attributes:
  model_definition_id, weight (Decimal), linkage_type, fallback_order,
  created_at, created_by, mapping_id
GSI5: gsi5pk = GW_ENDPOINT_MODELDEF#<model_def_id>   gsi5sk = <endpoint_id>

PK: GW_ENDPOINT#<endpoint_id>     SK: GW#BIND#<resource_type>#<resource_id>
Attributes:
  endpoint_id, resource_type, resource_id, created_at, last_updated_at,
  created_by, last_updated_by, display_name
GSI2: gsi2pk = GW_BIND#<resource_type>#<resource_id>  gsi2sk = <endpoint_id>

PK: GW_ENDPOINT#<endpoint_id>     SK: GW#TAG#<key>
Attributes:
  key, value
```

---

### Task 1: Add Phase 5c schema constants

**Files:**
- Modify: `src/mlflow_dynamodbstore/dynamodb/schema.py`

- [ ] **Step 1: Add the Phase 5c constants**

Add at the end of `schema.py`:

```python
# Gateway endpoint prefixes
PK_GW_ENDPOINT_PREFIX = "GW_ENDPOINT#"
SK_GW_MAP_PREFIX = "GW#MAP#"
SK_GW_BIND_PREFIX = "GW#BIND#"
SK_GW_TAG_PREFIX = "GW#TAG#"

# GSI prefixes for gateway endpoints
GSI2_GW_ENDPOINTS_PREFIX = "GW_ENDPOINTS#"
GSI2_GW_BIND_PREFIX = "GW_BIND#"
GSI3_GW_ENDPOINT_NAME_PREFIX = "GW_ENDPOINT_NAME#"
```

Note: `GSI5_GW_ENDPOINT_MODELDEF_PREFIX` already exists (added in Phase 5b for RESTRICT delete checks).

- [ ] **Step 2: Verify import works**

Run: `uv run python -c "from mlflow_dynamodbstore.dynamodb.schema import PK_GW_ENDPOINT_PREFIX, SK_GW_MAP_PREFIX, SK_GW_BIND_PREFIX, SK_GW_TAG_PREFIX, GSI2_GW_ENDPOINTS_PREFIX, GSI2_GW_BIND_PREFIX, GSI3_GW_ENDPOINT_NAME_PREFIX; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```
git add src/mlflow_dynamodbstore/dynamodb/schema.py
git commit -m "feat: add gateway endpoint schema constants"
```

---

### Task 2: Endpoint helpers and `create_gateway_endpoint`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`

- [ ] **Step 1: Add imports and implement helpers + `create_gateway_endpoint`**

Add to the entity imports (in the `from mlflow.entities import (` block):

```python
    FallbackConfig,
    FallbackStrategy,
    GatewayEndpoint,
    GatewayEndpointBinding,
    GatewayEndpointModelConfig,
    GatewayEndpointModelMapping,
    GatewayEndpointTag,
    GatewayModelLinkageType,
    GatewayResourceType,
    RoutingStrategy,
```

Add to the schema imports (in the `from mlflow_dynamodbstore.dynamodb.schema import (` block):

```python
    GSI2_GW_BIND_PREFIX,
    GSI2_GW_ENDPOINTS_PREFIX,
    GSI3_GW_ENDPOINT_NAME_PREFIX,
    PK_GW_ENDPOINT_PREFIX,
    SK_GW_BIND_PREFIX,
    SK_GW_MAP_PREFIX,
    SK_GW_TAG_PREFIX,
```

Add to the mlflow_tags import line (near `from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION`):

```python
from mlflow.utils.mlflow_tags import (
    MLFLOW_ARTIFACT_LOCATION,
    MLFLOW_EXPERIMENT_IS_GATEWAY,
    MLFLOW_EXPERIMENT_SOURCE_ID,
    MLFLOW_EXPERIMENT_SOURCE_TYPE,
)
```

Add the following at the end of `DynamoDBTrackingStore`, after the `delete_gateway_model_definition` method, as a new section:

```python
    # -----------------------------------------------------------------------
    # Gateway Endpoints
    # -----------------------------------------------------------------------

    def _get_endpoint_items(self, endpoint_id: str) -> list[dict[str, Any]]:
        """Query all items in an endpoint partition (META + mappings + bindings + tags)."""
        return self._table.query(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
        )

    def _endpoint_items_to_entity(
        self, endpoint_id: str, items: list[dict[str, Any]]
    ) -> GatewayEndpoint:
        """Convert a list of raw DynamoDB items from an endpoint partition to a GatewayEndpoint."""
        meta: dict[str, Any] | None = None
        mapping_items: list[dict[str, Any]] = []
        binding_items: list[dict[str, Any]] = []
        tag_items: list[dict[str, Any]] = []

        for item in items:
            sk = item["SK"]
            if sk == SK_GW_META:
                meta = item
            elif sk.startswith(SK_GW_MAP_PREFIX):
                mapping_items.append(item)
            elif sk.startswith(SK_GW_BIND_PREFIX):
                binding_items.append(item)
            elif sk.startswith(SK_GW_TAG_PREFIX):
                tag_items.append(item)

        if meta is None:
            raise MlflowException(
                f"Endpoint '{endpoint_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        # Resolve model definitions for mappings (batch to avoid N+1)
        model_def_ids = {m["model_definition_id"] for m in mapping_items}
        model_def_cache: dict[str, GatewayModelDefinition] = {}
        for md_id in model_def_ids:
            md_item = self._get_model_def_item(md_id)
            if md_item is not None:
                model_def_cache[md_id] = self._model_def_item_to_entity(md_item)

        model_mappings = []
        for m_item in mapping_items:
            md_id = m_item["model_definition_id"]
            model_mappings.append(
                GatewayEndpointModelMapping(
                    mapping_id=m_item["mapping_id"],
                    endpoint_id=endpoint_id,
                    model_definition_id=md_id,
                    model_definition=model_def_cache.get(md_id),
                    weight=float(m_item["weight"]),
                    linkage_type=GatewayModelLinkageType(m_item["linkage_type"]),
                    fallback_order=int(m_item["fallback_order"])
                    if m_item.get("fallback_order") is not None
                    else None,
                    created_at=int(m_item["created_at"]),
                    created_by=m_item.get("created_by"),
                )
            )

        tags = [
            GatewayEndpointTag(
                key=t["key"],
                value=t.get("value"),
            )
            for t in tag_items
        ]

        # Reconstruct routing_strategy
        routing_strategy = None
        if meta.get("routing_strategy"):
            routing_strategy = RoutingStrategy(meta["routing_strategy"])

        # Reconstruct fallback_config
        fallback_config = None
        if meta.get("fallback_config"):
            fc = meta["fallback_config"]
            fallback_config = FallbackConfig(
                strategy=FallbackStrategy(fc["strategy"]) if fc.get("strategy") else None,
                max_attempts=int(fc["max_attempts"]) if fc.get("max_attempts") is not None else None,
            )

        return GatewayEndpoint(
            endpoint_id=endpoint_id,
            name=meta.get("name"),
            created_at=int(meta["created_at"]),
            last_updated_at=int(meta["last_updated_at"]),
            model_mappings=model_mappings,
            tags=tags,
            created_by=meta.get("created_by"),
            last_updated_by=meta.get("last_updated_by"),
            routing_strategy=routing_strategy,
            fallback_config=fallback_config,
            experiment_id=str(meta["experiment_id"]) if meta.get("experiment_id") is not None else None,
            usage_tracking=bool(meta.get("usage_tracking", False)),
            workspace=meta.get("workspace"),
        )

    def _build_mapping_item(
        self,
        endpoint_id: str,
        config: GatewayEndpointModelConfig,
        mapping_id: str,
        now: int,
        created_by: str | None,
    ) -> dict[str, Any]:
        """Build a DynamoDB mapping item from a GatewayEndpointModelConfig."""
        item: dict[str, Any] = {
            "PK": f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            "SK": f"{SK_GW_MAP_PREFIX}{config.model_definition_id}#{config.linkage_type.value}",
            "model_definition_id": config.model_definition_id,
            "weight": Decimal(str(config.weight)),
            "linkage_type": config.linkage_type.value,
            "created_at": now,
            "mapping_id": mapping_id,
            # GSI5 projection for RESTRICT delete checks
            "gsi5pk": f"{GSI5_GW_ENDPOINT_MODELDEF_PREFIX}{config.model_definition_id}",
            "gsi5sk": endpoint_id,
        }
        if config.fallback_order is not None:
            item["fallback_order"] = config.fallback_order
        if created_by is not None:
            item["created_by"] = created_by
        return item

    def create_gateway_endpoint(
        self,
        name: str,
        model_configs: list[GatewayEndpointModelConfig],
        created_by: str | None = None,
        routing_strategy: RoutingStrategy | None = None,
        fallback_config: FallbackConfig | None = None,
        experiment_id: str | None = None,
        usage_tracking: bool = False,
    ) -> GatewayEndpoint:
        if not model_configs:
            raise MlflowException(
                "Endpoint must have at least one model configuration",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # Check name uniqueness via GSI3
        existing = self._table.query(
            pk=f"{GSI3_GW_ENDPOINT_NAME_PREFIX}{self._workspace}#{name}",
            index_name="gsi3",
            limit=1,
        )
        if existing:
            raise MlflowException(
                f"Endpoint with name '{name}' already exists",
                error_code=RESOURCE_ALREADY_EXISTS,
            )

        # Validate all model definitions exist
        for config in model_configs:
            md_item = self._get_model_def_item(config.model_definition_id)
            if md_item is None:
                raise MlflowException(
                    f"Model definitions not found: {config.model_definition_id}",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )

        endpoint_id = f"e-{generate_ulid()}"
        now = get_current_time_millis()

        # Auto-create experiment if usage_tracking is enabled and no experiment_id provided
        if usage_tracking and experiment_id is None:
            exp_name = f"gateway/{name}"
            existing_exp = self.get_experiment_by_name(exp_name)
            experiment_id = existing_exp.experiment_id if existing_exp else self.create_experiment(exp_name)
            self.set_experiment_tag(
                experiment_id, ExperimentTag(MLFLOW_EXPERIMENT_SOURCE_TYPE, "GATEWAY")
            )
            self.set_experiment_tag(
                experiment_id, ExperimentTag(MLFLOW_EXPERIMENT_SOURCE_ID, endpoint_id)
            )
            self.set_experiment_tag(
                experiment_id, ExperimentTag(MLFLOW_EXPERIMENT_IS_GATEWAY, "true")
            )

        # Build META item
        meta_item: dict[str, Any] = {
            "PK": f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            "SK": SK_GW_META,
            "name": name,
            "created_at": now,
            "last_updated_at": now,
            "workspace": self._workspace,
            "usage_tracking": usage_tracking,
            "tags": {},
            # GSI projections
            "gsi2pk": f"{GSI2_GW_ENDPOINTS_PREFIX}{self._workspace}",
            "gsi2sk": endpoint_id,
            "gsi3pk": f"{GSI3_GW_ENDPOINT_NAME_PREFIX}{self._workspace}#{name}",
            "gsi3sk": endpoint_id,
        }
        if routing_strategy is not None:
            meta_item["routing_strategy"] = routing_strategy.value
        if fallback_config is not None:
            meta_item["fallback_config"] = {
                "strategy": fallback_config.strategy.value if fallback_config.strategy else None,
                "max_attempts": fallback_config.max_attempts,
            }
        if experiment_id is not None:
            meta_item["experiment_id"] = experiment_id
        if created_by is not None:
            meta_item["created_by"] = created_by
            meta_item["last_updated_by"] = created_by

        # Build mapping items
        mapping_items = []
        for config in model_configs:
            mapping_id = f"m-{generate_ulid()}"
            mapping_items.append(
                self._build_mapping_item(endpoint_id, config, mapping_id, now, created_by)
            )

        # Write META
        self._table.put_item(meta_item)

        # Batch write mapping items
        if mapping_items:
            self._table.batch_write(mapping_items)

        # Return the full entity
        all_items = [meta_item] + mapping_items
        return self._endpoint_items_to_entity(endpoint_id, all_items)
```

- [ ] **Step 2: Run the create endpoint compat tests**

Run: `uv run pytest tests/compatibility/test_gateway_compat.py -k "test_create_gateway_endpoint" -v`
Expected: The 4 create tests xpass (create, auto_creates_experiment, empty_models_raises, nonexistent_model_raises).

- [ ] **Step 3: Commit**

```
git add src/mlflow_dynamodbstore/tracking_store.py src/mlflow_dynamodbstore/dynamodb/schema.py
git commit -m "feat: implement create_gateway_endpoint with helpers"
```

---

### Task 3: `get_gateway_endpoint` and `list_gateway_endpoints`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`

- [ ] **Step 1: Implement `get_gateway_endpoint` and `list_gateway_endpoints`**

Add after `create_gateway_endpoint`:

```python
    def get_gateway_endpoint(
        self,
        endpoint_id: str | None = None,
        name: str | None = None,
    ) -> GatewayEndpoint:
        if (endpoint_id is None) == (name is None):
            raise MlflowException(
                "Exactly one of `endpoint_id` or `name` must be specified",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if name:
            # Resolve name to endpoint_id via GSI3
            results = self._table.query(
                pk=f"{GSI3_GW_ENDPOINT_NAME_PREFIX}{self._workspace}#{name}",
                index_name="gsi3",
                limit=1,
            )
            if not results:
                raise MlflowException(
                    f"GatewayEndpoint not found (name='{name}')",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            endpoint_id = results[0]["gsi3sk"]

        items = self._get_endpoint_items(endpoint_id)
        if not items:
            raise MlflowException(
                f"GatewayEndpoint not found (endpoint_id='{endpoint_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        return self._endpoint_items_to_entity(endpoint_id, items)

    def list_gateway_endpoints(
        self,
        provider: str | None = None,
        secret_id: str | None = None,
    ) -> list[GatewayEndpoint]:
        if secret_id:
            # Targeted path: find model_defs by secret -> find endpoints by model_def -> deduplicate
            md_items = self._table.query(
                pk=f"{GSI4_GW_MODELDEF_SECRET_PREFIX}{secret_id}",
                index_name="gsi4",
            )
            endpoint_ids: set[str] = set()
            for md_item in md_items:
                md_id = md_item["PK"].removeprefix(PK_GW_MODELDEF_PREFIX)
                ep_items = self._table.query(
                    pk=f"{GSI5_GW_ENDPOINT_MODELDEF_PREFIX}{md_id}",
                    index_name="gsi5",
                )
                for ep_item in ep_items:
                    endpoint_ids.add(ep_item["gsi5sk"])

            endpoints = []
            for ep_id in endpoint_ids:
                items = self._get_endpoint_items(ep_id)
                if items:
                    endpoints.append(self._endpoint_items_to_entity(ep_id, items))
            return endpoints

        # Default path: list all endpoints via GSI2
        gsi2_items = self._table.query(
            pk=f"{GSI2_GW_ENDPOINTS_PREFIX}{self._workspace}",
            index_name="gsi2",
        )

        endpoints = []
        for gsi2_item in gsi2_items:
            ep_id = gsi2_item["gsi2sk"]
            items = self._get_endpoint_items(ep_id)
            if items:
                endpoints.append(self._endpoint_items_to_entity(ep_id, items))

        if provider is not None:
            endpoints = [
                ep
                for ep in endpoints
                if any(
                    m.model_definition and m.model_definition.provider == provider
                    for m in ep.model_mappings
                )
            ]

        return endpoints
```

- [ ] **Step 2: Run the get/list endpoint compat tests**

Run: `uv run pytest tests/compatibility/test_gateway_compat.py -k "test_get_gateway_endpoint or test_list_gateway_endpoints" -v`
Expected: 5 tests xpass (by_id, by_name, requires_one_of_id_or_name, list).

- [ ] **Step 3: Commit**

```
git add src/mlflow_dynamodbstore/tracking_store.py
git commit -m "feat: implement get_gateway_endpoint and list_gateway_endpoints"
```

---

### Task 4: `update_gateway_endpoint` and `delete_gateway_endpoint`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`

- [ ] **Step 1: Implement `update_gateway_endpoint` and `delete_gateway_endpoint`**

Add after `list_gateway_endpoints`:

```python
    def update_gateway_endpoint(
        self,
        endpoint_id: str,
        name: str | None = None,
        updated_by: str | None = None,
        routing_strategy: RoutingStrategy | None = None,
        fallback_config: FallbackConfig | None = None,
        model_configs: list[GatewayEndpointModelConfig] | None = None,
        experiment_id: str | None = None,
        usage_tracking: bool | None = None,
    ) -> GatewayEndpoint:
        # Get existing endpoint items
        items = self._get_endpoint_items(endpoint_id)
        meta = next((i for i in items if i["SK"] == SK_GW_META), None)
        if meta is None:
            raise MlflowException(
                f"GatewayEndpoint not found (endpoint_id='{endpoint_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        now = get_current_time_millis()
        updates: dict[str, Any] = {"last_updated_at": now}
        removes: list[str] = []

        if name is not None:
            # Check new name uniqueness
            existing = self._table.query(
                pk=f"{GSI3_GW_ENDPOINT_NAME_PREFIX}{self._workspace}#{name}",
                index_name="gsi3",
                limit=1,
            )
            if existing and existing[0]["gsi3sk"] != endpoint_id:
                raise MlflowException(
                    f"Endpoint with name '{name}' already exists",
                    error_code=RESOURCE_ALREADY_EXISTS,
                )
            updates["name"] = name
            updates["gsi3pk"] = f"{GSI3_GW_ENDPOINT_NAME_PREFIX}{self._workspace}#{name}"

        # Handle usage_tracking update
        if usage_tracking is not None:
            updates["usage_tracking"] = usage_tracking

        # Auto-create experiment if usage_tracking is enabled and no experiment_id provided
        if usage_tracking and experiment_id is None and meta.get("experiment_id") is None:
            endpoint_name = name if name is not None else meta.get("name")
            exp_name = f"gateway/{endpoint_name}"
            existing_exp = self.get_experiment_by_name(exp_name)
            experiment_id = existing_exp.experiment_id if existing_exp else self.create_experiment(exp_name)
            self.set_experiment_tag(
                experiment_id, ExperimentTag(MLFLOW_EXPERIMENT_SOURCE_TYPE, "GATEWAY")
            )
            self.set_experiment_tag(
                experiment_id, ExperimentTag(MLFLOW_EXPERIMENT_SOURCE_ID, endpoint_id)
            )
            self.set_experiment_tag(
                experiment_id, ExperimentTag(MLFLOW_EXPERIMENT_IS_GATEWAY, "true")
            )

        if experiment_id is not None:
            updates["experiment_id"] = experiment_id

        if routing_strategy is not None:
            updates["routing_strategy"] = routing_strategy.value

        if updated_by is not None:
            updates["last_updated_by"] = updated_by

        # Replace model configs if provided (full replacement)
        if model_configs is not None:
            # Validate all model definitions exist
            for config in model_configs:
                md_item = self._get_model_def_item(config.model_definition_id)
                if md_item is None:
                    raise MlflowException(
                        f"Model definition '{config.model_definition_id}' not found",
                        error_code=RESOURCE_DOES_NOT_EXIST,
                    )

            # Delete existing mapping items
            existing_mappings = [i for i in items if i["SK"].startswith(SK_GW_MAP_PREFIX)]
            if existing_mappings:
                self._table.batch_delete(
                    [{"PK": m["PK"], "SK": m["SK"]} for m in existing_mappings]
                )

            # Write new mapping items
            new_mapping_items = []
            for config in model_configs:
                mapping_id = f"m-{generate_ulid()}"
                new_mapping_items.append(
                    self._build_mapping_item(endpoint_id, config, mapping_id, now, updated_by)
                )
            if new_mapping_items:
                self._table.batch_write(new_mapping_items)

            # Update fallback_config with new model config info
            fallback_model_def_ids = [
                config.model_definition_id
                for config in model_configs
                if config.linkage_type == GatewayModelLinkageType.FALLBACK
            ]
            if fallback_config or fallback_model_def_ids:
                updates["fallback_config"] = {
                    "strategy": fallback_config.strategy.value
                    if fallback_config and fallback_config.strategy
                    else None,
                    "max_attempts": fallback_config.max_attempts if fallback_config else None,
                }
            elif not fallback_model_def_ids and "fallback_config" not in updates:
                # No fallback models and no explicit fallback_config, clear it
                updates["fallback_config"] = {
                    "strategy": None,
                    "max_attempts": None,
                }

        elif fallback_config is not None:
            # Update fallback_config without replacing model configs
            updates["fallback_config"] = {
                "strategy": fallback_config.strategy.value
                if fallback_config.strategy
                else None,
                "max_attempts": fallback_config.max_attempts,
            }

        self._table.update_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=SK_GW_META,
            updates=updates,
            removes=removes if removes else None,
        )

        self._invalidate_secret_cache()

        # Re-fetch to return the full updated entity
        updated_items = self._get_endpoint_items(endpoint_id)
        return self._endpoint_items_to_entity(endpoint_id, updated_items)

    def delete_gateway_endpoint(self, endpoint_id: str) -> None:
        # Query all items in partition
        items = self._get_endpoint_items(endpoint_id)
        if not items:
            raise MlflowException(
                f"GatewayEndpoint not found (endpoint_id='{endpoint_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        # Batch delete all items (META, mappings, bindings, tags)
        self._table.batch_delete(
            [{"PK": item["PK"], "SK": item["SK"]} for item in items]
        )
        self._invalidate_secret_cache()
```

- [ ] **Step 2: Run the update/delete endpoint compat tests**

Run: `uv run pytest tests/compatibility/test_gateway_compat.py -k "test_update_gateway_endpoint or test_delete_gateway_endpoint" -v`
Expected: 2 tests xpass (update, delete).

- [ ] **Step 3: Commit**

```
git add src/mlflow_dynamodbstore/tracking_store.py
git commit -m "feat: implement update_gateway_endpoint and delete_gateway_endpoint"
```

---

### Task 5: `attach_model_to_endpoint` and `detach_model_from_endpoint`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`

- [ ] **Step 1: Implement `attach_model_to_endpoint` and `detach_model_from_endpoint`**

Add after `delete_gateway_endpoint`:

```python
    def attach_model_to_endpoint(
        self,
        endpoint_id: str,
        model_config: GatewayEndpointModelConfig,
        created_by: str | None = None,
    ) -> GatewayEndpointModelMapping:
        # Verify endpoint exists
        meta = self._table.get_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=SK_GW_META,
        )
        if meta is None:
            raise MlflowException(
                f"GatewayEndpoint not found (endpoint_id='{endpoint_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        # Verify model definition exists
        md_item = self._get_model_def_item(model_config.model_definition_id)
        if md_item is None:
            raise MlflowException(
                f"Model definition '{model_config.model_definition_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        mapping_id = f"m-{generate_ulid()}"
        now = get_current_time_millis()

        mapping_item = self._build_mapping_item(
            endpoint_id, model_config, mapping_id, now, created_by
        )

        # Conditional put for atomic uniqueness
        from botocore.exceptions import ClientError

        try:
            self._table.put_item(mapping_item, condition="attribute_not_exists(SK)")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise MlflowException(
                    f"Model definition '{model_config.model_definition_id}' is already attached to "
                    f"endpoint '{endpoint_id}'",
                    error_code=RESOURCE_ALREADY_EXISTS,
                ) from e
            raise

        # Update endpoint META last_updated_at
        update_fields: dict[str, Any] = {"last_updated_at": now}
        if created_by:
            update_fields["last_updated_by"] = created_by
        self._table.update_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=SK_GW_META,
            updates=update_fields,
        )

        self._invalidate_secret_cache()

        # Return the mapping entity
        model_def = self._model_def_item_to_entity(md_item)
        return GatewayEndpointModelMapping(
            mapping_id=mapping_id,
            endpoint_id=endpoint_id,
            model_definition_id=model_config.model_definition_id,
            model_definition=model_def,
            weight=float(model_config.weight),
            linkage_type=model_config.linkage_type,
            fallback_order=model_config.fallback_order,
            created_at=now,
            created_by=created_by,
        )

    def detach_model_from_endpoint(
        self,
        endpoint_id: str,
        model_definition_id: str,
        linkage_type: str | None = None,
    ) -> None:
        # Verify endpoint exists
        meta = self._table.get_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=SK_GW_META,
        )
        if meta is None:
            raise MlflowException(
                f"GatewayEndpoint not found (endpoint_id='{endpoint_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        if linkage_type:
            # Deterministic SK — direct delete
            sk = f"{SK_GW_MAP_PREFIX}{model_definition_id}#{linkage_type}"
            existing = self._table.get_item(
                pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
                sk=sk,
            )
            if existing is None:
                raise MlflowException(
                    f"Model definition '{model_definition_id}' is not attached to "
                    f"endpoint '{endpoint_id}' with linkage type '{linkage_type}'",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            self._table.delete_item(
                pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
                sk=sk,
            )
        else:
            # Query all linkage variants for this model_def
            mapping_items = self._table.query(
                pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
                sk_prefix=f"{SK_GW_MAP_PREFIX}{model_definition_id}#",
            )
            if not mapping_items:
                raise MlflowException(
                    f"Model definition '{model_definition_id}' is not attached to "
                    f"endpoint '{endpoint_id}'",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            self._table.batch_delete(
                [{"PK": m["PK"], "SK": m["SK"]} for m in mapping_items]
            )

        # Update endpoint META last_updated_at
        now = get_current_time_millis()
        self._table.update_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=SK_GW_META,
            updates={"last_updated_at": now},
        )
        self._invalidate_secret_cache()
```

- [ ] **Step 2: Run the attach/detach compat tests**

Run: `uv run pytest tests/compatibility/test_gateway_compat.py -k "test_attach or test_detach" -v`
Expected: 4 tests xpass (attach, attach_duplicate_raises, detach, detach_nonexistent_raises).

- [ ] **Step 3: Commit**

```
git add src/mlflow_dynamodbstore/tracking_store.py
git commit -m "feat: implement attach_model_to_endpoint and detach_model_from_endpoint"
```

---

### Task 6: `create_endpoint_binding`, `delete_endpoint_binding`, `list_endpoint_bindings`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`

- [ ] **Step 1: Implement binding methods**

Add after `detach_model_from_endpoint`:

```python
    def create_endpoint_binding(
        self,
        endpoint_id: str,
        resource_type: str,
        resource_id: str,
        created_by: str | None = None,
    ) -> GatewayEndpointBinding:
        # Verify endpoint exists
        meta = self._table.get_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=SK_GW_META,
        )
        if meta is None:
            raise MlflowException(
                f"GatewayEndpoint not found (endpoint_id='{endpoint_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        now = get_current_time_millis()

        item: dict[str, Any] = {
            "PK": f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            "SK": f"{SK_GW_BIND_PREFIX}{resource_type}#{resource_id}",
            "endpoint_id": endpoint_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "created_at": now,
            "last_updated_at": now,
            # GSI2 projection for reverse lookup (AP25)
            "gsi2pk": f"{GSI2_GW_BIND_PREFIX}{resource_type}#{resource_id}",
            "gsi2sk": endpoint_id,
        }
        if created_by is not None:
            item["created_by"] = created_by
            item["last_updated_by"] = created_by

        self._table.put_item(item)
        self._invalidate_secret_cache()

        return GatewayEndpointBinding(
            endpoint_id=endpoint_id,
            resource_type=GatewayResourceType(resource_type),
            resource_id=resource_id,
            created_at=now,
            last_updated_at=now,
            created_by=created_by,
            last_updated_by=created_by,
        )

    def delete_endpoint_binding(
        self,
        endpoint_id: str,
        resource_type: str,
        resource_id: str,
    ) -> None:
        # Verify binding exists
        sk = f"{SK_GW_BIND_PREFIX}{resource_type}#{resource_id}"
        existing = self._table.get_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=sk,
        )
        if existing is None:
            raise MlflowException(
                f"GatewayEndpointBinding not found (endpoint_id='{endpoint_id}', "
                f"resource_type='{resource_type}', resource_id='{resource_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        self._table.delete_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=sk,
        )
        self._invalidate_secret_cache()

    def list_endpoint_bindings(
        self,
        endpoint_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
    ) -> list[GatewayEndpointBinding]:
        if endpoint_id is not None:
            # Direct query within endpoint partition
            sk_prefix = SK_GW_BIND_PREFIX
            if resource_type is not None:
                sk_prefix = f"{SK_GW_BIND_PREFIX}{resource_type}#"
                if resource_id is not None:
                    sk_prefix = f"{SK_GW_BIND_PREFIX}{resource_type}#{resource_id}"

            items = self._table.query(
                pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
                sk_prefix=sk_prefix,
            )
            return [
                GatewayEndpointBinding(
                    endpoint_id=item["endpoint_id"],
                    resource_type=GatewayResourceType(item["resource_type"]),
                    resource_id=item["resource_id"],
                    created_at=int(item["created_at"]),
                    last_updated_at=int(item["last_updated_at"]),
                    created_by=item.get("created_by"),
                    last_updated_by=item.get("last_updated_by"),
                    display_name=item.get("display_name"),
                )
                for item in items
            ]

        if resource_type is not None and resource_id is not None:
            # Reverse lookup via GSI2
            gsi2_items = self._table.query(
                pk=f"{GSI2_GW_BIND_PREFIX}{resource_type}#{resource_id}",
                index_name="gsi2",
            )
            bindings = []
            for gsi2_item in gsi2_items:
                ep_id = gsi2_item["gsi2sk"]
                sk = f"{SK_GW_BIND_PREFIX}{resource_type}#{resource_id}"
                bind_item = self._table.get_item(
                    pk=f"{PK_GW_ENDPOINT_PREFIX}{ep_id}",
                    sk=sk,
                )
                if bind_item:
                    bindings.append(
                        GatewayEndpointBinding(
                            endpoint_id=bind_item["endpoint_id"],
                            resource_type=GatewayResourceType(bind_item["resource_type"]),
                            resource_id=bind_item["resource_id"],
                            created_at=int(bind_item["created_at"]),
                            last_updated_at=int(bind_item["last_updated_at"]),
                            created_by=bind_item.get("created_by"),
                            last_updated_by=bind_item.get("last_updated_by"),
                            display_name=bind_item.get("display_name"),
                        )
                    )
            return bindings

        # No endpoint_id and no resource filter: list all endpoints, query bindings per endpoint
        gsi2_items = self._table.query(
            pk=f"{GSI2_GW_ENDPOINTS_PREFIX}{self._workspace}",
            index_name="gsi2",
        )
        bindings = []
        for gsi2_item in gsi2_items:
            ep_id = gsi2_item["gsi2sk"]
            bind_items = self._table.query(
                pk=f"{PK_GW_ENDPOINT_PREFIX}{ep_id}",
                sk_prefix=SK_GW_BIND_PREFIX,
            )
            for item in bind_items:
                bindings.append(
                    GatewayEndpointBinding(
                        endpoint_id=item["endpoint_id"],
                        resource_type=GatewayResourceType(item["resource_type"]),
                        resource_id=item["resource_id"],
                        created_at=int(item["created_at"]),
                        last_updated_at=int(item["last_updated_at"]),
                        created_by=item.get("created_by"),
                        last_updated_by=item.get("last_updated_by"),
                        display_name=item.get("display_name"),
                    )
                )
        return bindings
```

- [ ] **Step 2: Run the binding compat tests**

Run: `uv run pytest tests/compatibility/test_gateway_compat.py -k "test_create_gateway_endpoint_binding or test_delete_gateway_endpoint_binding or test_list_gateway_endpoint_bindings" -v`
Expected: 3 tests xpass.

- [ ] **Step 3: Commit**

```
git add src/mlflow_dynamodbstore/tracking_store.py
git commit -m "feat: implement create/delete/list_endpoint_binding"
```

---

### Task 7: `set_gateway_endpoint_tag` and `delete_gateway_endpoint_tag`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`

- [ ] **Step 1: Implement tag methods**

Add after `list_endpoint_bindings`:

```python
    def set_gateway_endpoint_tag(
        self,
        endpoint_id: str,
        tag: GatewayEndpointTag,
    ) -> None:
        # Verify endpoint exists
        meta = self._table.get_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=SK_GW_META,
        )
        if meta is None:
            raise MlflowException(
                f"GatewayEndpoint not found (endpoint_id='{endpoint_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        now = get_current_time_millis()

        # Write the tag item (upsert)
        tag_item: dict[str, Any] = {
            "PK": f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            "SK": f"{SK_GW_TAG_PREFIX}{tag.key}",
            "key": tag.key,
            "value": tag.value,
        }
        self._table.put_item(tag_item)

        # Update denormalized tags on META
        tags_dict = dict(meta.get("tags", {}))
        tags_dict[tag.key] = tag.value
        self._table.update_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=SK_GW_META,
            updates={"tags": tags_dict, "last_updated_at": now},
        )

    def delete_gateway_endpoint_tag(
        self,
        endpoint_id: str,
        key: str,
    ) -> None:
        # Verify endpoint exists
        meta = self._table.get_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=SK_GW_META,
        )
        if meta is None:
            raise MlflowException(
                f"GatewayEndpoint not found (endpoint_id='{endpoint_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        now = get_current_time_millis()

        # Delete the tag item (no-op if it doesn't exist)
        self._table.delete_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=f"{SK_GW_TAG_PREFIX}{key}",
        )

        # Update denormalized tags on META
        tags_dict = dict(meta.get("tags", {}))
        tags_dict.pop(key, None)
        self._table.update_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=SK_GW_META,
            updates={"tags": tags_dict, "last_updated_at": now},
        )
```

- [ ] **Step 2: Run the tag compat tests**

Run: `uv run pytest tests/compatibility/test_gateway_compat.py -k "tag" -v`
Expected: 8 tests xpass (set, set_update_existing, set_multiple, set_nonexistent_endpoint_raises, delete, delete_nonexistent_endpoint_raises, delete_nonexistent_key_no_op, tags_deleted_with_endpoint).

- [ ] **Step 3: Commit**

```
git add src/mlflow_dynamodbstore/tracking_store.py
git commit -m "feat: implement set/delete_gateway_endpoint_tag"
```

---

### Task 8: Remove xfail from all remaining tests

**Files:**
- Modify: `tests/compatibility/test_gateway_compat.py`

- [ ] **Step 1: Remove xfail markers from all 32 tests**

Remove the following xfail lines from `test_gateway_compat.py` (lines 133-210):

```python
# Model Definitions — in_use_raises depends on create_gateway_endpoint (Phase 5c)
test_delete_gateway_model_definition_in_use_raises = pytest.mark.xfail(
    raises=NotImplementedError,
    reason="Depends on create_gateway_endpoint (Phase 5c)",
)(test_delete_gateway_model_definition_in_use_raises)

# Endpoints
test_create_gateway_endpoint = _xfail_gateway(test_create_gateway_endpoint)
test_create_gateway_endpoint_auto_creates_experiment = _xfail_gateway(
    test_create_gateway_endpoint_auto_creates_experiment
)
test_create_gateway_endpoint_empty_models_raises = _xfail_gateway(
    test_create_gateway_endpoint_empty_models_raises
)
test_create_gateway_endpoint_nonexistent_model_raises = _xfail_gateway(
    test_create_gateway_endpoint_nonexistent_model_raises
)
test_delete_gateway_endpoint = _xfail_gateway(test_delete_gateway_endpoint)
test_get_gateway_endpoint_by_id = _xfail_gateway(test_get_gateway_endpoint_by_id)
test_get_gateway_endpoint_by_name = _xfail_gateway(test_get_gateway_endpoint_by_name)
test_get_gateway_endpoint_requires_one_of_id_or_name = _xfail_gateway(
    test_get_gateway_endpoint_requires_one_of_id_or_name
)
test_list_gateway_endpoints = _xfail_gateway(test_list_gateway_endpoints)
test_update_gateway_endpoint = _xfail_gateway(test_update_gateway_endpoint)

# Model Attachment
test_attach_duplicate_model_raises = _xfail_gateway(test_attach_duplicate_model_raises)
test_attach_model_to_gateway_endpoint = _xfail_gateway(test_attach_model_to_gateway_endpoint)
test_detach_model_from_gateway_endpoint = _xfail_gateway(test_detach_model_from_gateway_endpoint)
test_detach_nonexistent_mapping_raises = _xfail_gateway(test_detach_nonexistent_mapping_raises)

# Bindings
test_create_gateway_endpoint_binding = _xfail_gateway(test_create_gateway_endpoint_binding)
test_delete_gateway_endpoint_binding = _xfail_gateway(test_delete_gateway_endpoint_binding)
test_list_gateway_endpoint_bindings = _xfail_gateway(test_list_gateway_endpoint_bindings)

# Tags
test_delete_gateway_endpoint_tag = _xfail_gateway(test_delete_gateway_endpoint_tag)
test_delete_gateway_endpoint_tag_nonexistent_endpoint_raises = _xfail_gateway(
    test_delete_gateway_endpoint_tag_nonexistent_endpoint_raises
)
test_delete_gateway_endpoint_tag_nonexistent_key_no_op = _xfail_gateway(
    test_delete_gateway_endpoint_tag_nonexistent_key_no_op
)
test_endpoint_tags_deleted_with_endpoint = _xfail_gateway(test_endpoint_tags_deleted_with_endpoint)
test_set_gateway_endpoint_tag = _xfail_gateway(test_set_gateway_endpoint_tag)
test_set_gateway_endpoint_tag_nonexistent_endpoint_raises = _xfail_gateway(
    test_set_gateway_endpoint_tag_nonexistent_endpoint_raises
)
test_set_gateway_endpoint_tag_update_existing = _xfail_gateway(
    test_set_gateway_endpoint_tag_update_existing
)
test_set_multiple_endpoint_tags = _xfail_gateway(test_set_multiple_endpoint_tags)

# Scorer Integration
test_get_scorer_resolves_endpoint_id_to_name = _xfail_gateway(
    test_get_scorer_resolves_endpoint_id_to_name
)
test_get_scorer_with_deleted_endpoint_sets_model_to_null = _xfail_gateway(
    test_get_scorer_with_deleted_endpoint_sets_model_to_null
)
test_list_scorers_batch_resolves_endpoint_ids = _xfail_gateway(
    test_list_scorers_batch_resolves_endpoint_ids
)
test_register_scorer_resolves_endpoint_name_to_id = _xfail_gateway(
    test_register_scorer_resolves_endpoint_name_to_id
)

# Fallback / Traffic Routing
test_create_gateway_endpoint_with_fallback_routing = _xfail_gateway(
    test_create_gateway_endpoint_with_fallback_routing
)
test_create_gateway_endpoint_with_traffic_split = _xfail_gateway(
    test_create_gateway_endpoint_with_traffic_split
)
```

Keep only the `_xfail_gateway` marker definition (for future use) and the `test_register_scorer_with_nonexistent_endpoint_raises` xfail (separate reason — DynamoDB store does not validate endpoint existence in `register_scorer`).

The resulting file should look like:

```python
"""Phase 5: MLflow gateway store tests run against DynamoDB.

Test functions are imported from the vendored MLflow test suite.
The module-level `store` fixture overrides conftest's default (registry_store)
to use the tracking_store instead.

Excluded tests (11 total):
- test_secret_id_and_name_are_immutable_at_database_level: uses sqlalchemy.text()
  and ManagedSessionMaker to test SQL column triggers
- 10 config_resolver tests (test_get_resource_gateway_endpoint_configs,
  test_get_resource_endpoint_configs_*, test_get_gateway_endpoint_config*):
  config_resolver.py does isinstance(store, SqlAlchemyStore) check and uses
  ORM queries with ManagedSessionMaker
"""

import uuid

import pytest
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.utils.workspace_context import WorkspaceContext


@pytest.fixture
def store(tracking_store):
    return tracking_store


@pytest.fixture(autouse=True)
def set_kek_passphrase(monkeypatch):
    monkeypatch.setenv("MLFLOW_CRYPTO_KEK_PASSPHRASE", "test-passphrase-for-gateway-tests")


@pytest.fixture(autouse=True, params=[False, True], ids=["workspace-disabled", "workspace-enabled"])
def workspaces_enabled(request, monkeypatch):
    enabled = request.param
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true" if enabled else "false")
    if enabled:
        workspace_name = f"gateway-test-{uuid.uuid4().hex}"
        with WorkspaceContext(workspace_name):
            yield enabled
    else:
        yield enabled


# =============================================================================
# Secret Operations (13 tests)
# =============================================================================
# =============================================================================
# Model Definition Operations (10 tests)
# =============================================================================
# =============================================================================
# Endpoint Operations (10 tests)
# =============================================================================
# =============================================================================
# Model Attachment (4 tests)
# =============================================================================
# =============================================================================
# Bindings (3 tests)
# =============================================================================
# =============================================================================
# Tags (8 tests)
# =============================================================================
# =============================================================================
# Scorer Integration (5 tests)
# =============================================================================
# =============================================================================
# Fallback / Traffic Routing (2 tests)
# =============================================================================
from tests.store.tracking.test_gateway_sql_store import (  # noqa: E402, F401
    test_attach_duplicate_model_raises,
    test_attach_model_to_gateway_endpoint,
    test_create_gateway_endpoint,
    test_create_gateway_endpoint_auto_creates_experiment,
    test_create_gateway_endpoint_binding,
    test_create_gateway_endpoint_empty_models_raises,
    test_create_gateway_endpoint_nonexistent_model_raises,
    test_create_gateway_endpoint_with_fallback_routing,
    test_create_gateway_endpoint_with_traffic_split,
    test_create_gateway_model_definition,
    test_create_gateway_model_definition_duplicate_name_raises,
    test_create_gateway_model_definition_nonexistent_secret_raises,
    test_create_gateway_secret,
    test_create_gateway_secret_duplicate_name_raises,
    test_create_gateway_secret_with_auth_config,
    test_create_gateway_secret_with_dict_value,
    test_delete_gateway_endpoint,
    test_delete_gateway_endpoint_binding,
    test_delete_gateway_endpoint_tag,
    test_delete_gateway_endpoint_tag_nonexistent_endpoint_raises,
    test_delete_gateway_endpoint_tag_nonexistent_key_no_op,
    test_delete_gateway_model_definition,
    test_delete_gateway_model_definition_in_use_raises,
    test_delete_gateway_secret,
    test_detach_model_from_gateway_endpoint,
    test_detach_nonexistent_mapping_raises,
    test_endpoint_tags_deleted_with_endpoint,
    test_get_gateway_endpoint_by_id,
    test_get_gateway_endpoint_by_name,
    test_get_gateway_endpoint_requires_one_of_id_or_name,
    test_get_gateway_model_definition_by_id,
    test_get_gateway_model_definition_by_name,
    test_get_gateway_model_definition_requires_one_of_id_or_name,
    test_get_gateway_secret_info_by_id,
    test_get_gateway_secret_info_by_name,
    test_get_gateway_secret_info_not_found,
    test_get_gateway_secret_info_requires_one_of_id_or_name,
    test_get_scorer_resolves_endpoint_id_to_name,
    test_get_scorer_with_deleted_endpoint_sets_model_to_null,
    test_list_gateway_endpoint_bindings,
    test_list_gateway_endpoints,
    test_list_gateway_model_definitions,
    test_list_gateway_secret_infos,
    test_list_scorers_batch_resolves_endpoint_ids,
    test_register_scorer_resolves_endpoint_name_to_id,
    test_register_scorer_with_nonexistent_endpoint_raises,
    test_set_gateway_endpoint_tag,
    test_set_gateway_endpoint_tag_nonexistent_endpoint_raises,
    test_set_gateway_endpoint_tag_update_existing,
    test_set_multiple_endpoint_tags,
    test_update_gateway_endpoint,
    test_update_gateway_model_definition,
    test_update_gateway_secret,
    test_update_gateway_secret_clear_auth_config,
    test_update_gateway_secret_with_auth_config,
)

# --- Gateway store methods not yet implemented ---
_xfail_gateway = pytest.mark.xfail(
    raises=NotImplementedError,
    reason="Gateway store methods not yet implemented (Phase 5)",
)

# Scorer Integration — DynamoDB register_scorer does not validate gateway endpoint existence
test_register_scorer_with_nonexistent_endpoint_raises = pytest.mark.xfail(
    reason="DynamoDB store register_scorer does not validate gateway endpoint existence"
)(test_register_scorer_with_nonexistent_endpoint_raises)
```

- [ ] **Step 2: Run the full gateway compat suite**

Run: `uv run pytest tests/compatibility/test_gateway_compat.py -v --tb=short 2>&1 | tail -10`
Expected: 108 passed (54 tests x 2 workspace params), 2 xfailed (`test_register_scorer_with_nonexistent_endpoint_raises` x 2 workspace params). Zero failed.

- [ ] **Step 3: Run the full test suite to confirm nothing else broke**

Run: `uv run pytest tests/ -v --tb=short -q 2>&1 | tail -5`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```
git add tests/compatibility/test_gateway_compat.py
git commit -m "test: remove xfail from 32 gateway endpoint/binding/tag/scorer compat tests"
```
