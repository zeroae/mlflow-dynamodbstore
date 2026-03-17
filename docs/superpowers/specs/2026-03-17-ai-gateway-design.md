# Phase 5: AI Gateway — DynamoDB Implementation

## Context

Phases 1–4 achieved full `AbstractStore` tracking parity. Phase 5 implements the `GatewayStoreMixin` — 22 methods across secrets, model definitions, endpoints, bindings, and tags — completing full parity with the `SqlAlchemyStore`.

`AbstractStore` inherits from `GatewayStoreMixin`, so `DynamoDBTrackingStore` already inherits all 22 methods as `NotImplementedError` stubs. The gateway enables MLflow to route LLM requests through configured endpoints (e.g., OpenAI, Anthropic, Bedrock), manage API credentials, and bind endpoints to scorers for online evaluation.

## Scope

22 store methods across 6 categories:

| Method | Category |
|--------|----------|
| `create_gateway_secret` | Secrets |
| `get_secret_info` | Secrets |
| `update_gateway_secret` | Secrets |
| `delete_gateway_secret` | Secrets |
| `list_secret_infos` | Secrets |
| `create_gateway_model_definition` | Model Definitions |
| `get_gateway_model_definition` | Model Definitions |
| `list_gateway_model_definitions` | Model Definitions |
| `update_gateway_model_definition` | Model Definitions |
| `delete_gateway_model_definition` | Model Definitions |
| `create_gateway_endpoint` | Endpoints |
| `get_gateway_endpoint` | Endpoints |
| `list_gateway_endpoints` | Endpoints |
| `update_gateway_endpoint` | Endpoints |
| `delete_gateway_endpoint` | Endpoints |
| `attach_model_to_endpoint` | Model Attachment |
| `detach_model_from_endpoint` | Model Attachment |
| `create_endpoint_binding` | Bindings |
| `delete_endpoint_binding` | Bindings |
| `list_endpoint_bindings` | Bindings |
| `set_gateway_endpoint_tag` | Tags |
| `delete_gateway_endpoint_tag` | Tags |

## DynamoDB Item Design

Gateway entities get their own partition families — they are independent of experiments and runs. This follows the same cross-experiment pattern used for evaluation datasets (`DS#<dataset_id>`).

**Workspace scoping**: All gateway entities store a `workspace` attribute on their META items. GSI PK values that include `<workspace>` use `self._workspace` (set during store initialization, default `"default"`) — the same pattern used by experiments, registered models, and Phase 3 sessions. All store methods that construct GSI PK values with `<ws>` substitute `self._workspace`.

### Partition Families

| Prefix | Purpose |
|--------|---------|
| `GW_SECRET#<secret_id>` | Secret items |
| `GW_MODELDEF#<model_def_id>` | Model definition items |
| `GW_ENDPOINT#<endpoint_id>` | Endpoint + children (mappings, bindings, tags) |

### Item Types

#### Secrets

| Item | PK | SK | Attributes |
|------|----|----|------------|
| Secret META | `GW_SECRET#<secret_id>` | `GW#META` | secret_name, encrypted_value (Binary), wrapped_dek (Binary), kek_version, masked_value (JSON), provider, auth_config (JSON), created_at, last_updated_at, created_by, last_updated_by, workspace |

#### Model Definitions

| Item | PK | SK | Attributes |
|------|----|----|------------|
| ModelDef META | `GW_MODELDEF#<model_def_id>` | `GW#META` | name, secret_id, provider, model_name, created_at, last_updated_at, created_by, last_updated_by, workspace |

#### Endpoints

| Item | PK | SK | Attributes |
|------|----|----|------------|
| Endpoint META | `GW_ENDPOINT#<endpoint_id>` | `GW#META` | name, routing_strategy, fallback_config (JSON), experiment_id, usage_tracking, created_at, last_updated_at, created_by, last_updated_by, workspace, tags (denormalized dict) |
| Model Mapping | `GW_ENDPOINT#<endpoint_id>` | `GW#MAP#<model_def_id>#<linkage_type>` | model_definition_id, weight (Decimal), linkage_type, fallback_order, created_at, created_by |
| Binding | `GW_ENDPOINT#<endpoint_id>` | `GW#BIND#<resource_type>#<resource_id>` | endpoint_id, resource_type, resource_id, created_at, last_updated_at, created_by, last_updated_by, display_name |
| Tag | `GW_ENDPOINT#<endpoint_id>` | `GW#TAG#<key>` | key, value |

### ID Generation

| Entity | Format | Example |
|--------|--------|---------|
| secret_id | `s-<ulid>` | `s-01JARW3QXP000000000000000` |
| model_definition_id | `d-<ulid>` | `d-01JARW3QXP000000000000000` |
| endpoint_id | `e-<ulid>` | `e-01JARW3QXP000000000000000` |
| mapping_id | (deterministic, see below) | `GW#MAP#<model_def_id>#<linkage_type>` |
| online_scoring_config_id | ULID | (from Phase 1c, used in bindings) |

**Note on ID format divergence**: The SQLAlchemy store generates IDs as `s-{uuid4().hex}` (32 hex chars). This spec uses `s-<ulid>` (26 base32 chars) for time-sortability, consistent with all other phases. No MLflow REST handler validates gateway ID length or format — IDs are opaque strings.

### GSI Usage

GSIs are shared across all entity types using PK prefix namespacing. Each GSI uses the existing DynamoDB attribute names (`gsi1pk`/`gsi1sk`, `gsi2pk`/`gsi2sk`, etc.) defined in `dynamodb/schema.py`. Gateway items write to these same attributes with distinct PK prefixes — no new GSI attribute names or provisioner changes needed. All 5 GSIs already exist with `ALL` projection.

**Existing GSI co-tenants** (for reference):
- GSI1: `RUN#`, `TRACE#`, `CLIENT#`, `DS#` (base), `DS_EXP#` (Phase 1a), `LM#` (Phase 1b), `SCOR#` (Phase 1c)
- GSI2: `EXPERIMENTS#`, `MODELS#`, `AUTH_USERS`, `WORKSPACES`, `FTS_NAMES#` (base), `DS_LIST#` (Phase 1a), `ACTIVE_SCORERS#` (Phase 1c), `SESSIONS#` (Phase 3)
- GSI3: `EXP_NAME#`, `MODEL_NAME#`, `ALIAS#` (base), `DS_NAME#` (Phase 1a), `SCOR_NAME#` (Phase 1c)
- GSI4: `PERM#` (base)
- GSI5: `EXP_NAMES#`, `MODEL_NAMES#` (base)

| GSI | Attribute | PK Value | SK Value | Purpose | Written On |
|-----|-----------|----------|----------|---------|------------|
| GSI1 | `gsi1pk`/`gsi1sk` | `GW_SECRET_NAME#<workspace>#<secret_name>` | `<secret_id>` | Unique secret name lookup | Secret META |
| GSI2 | `gsi2pk`/`gsi2sk` | `GW_SECRETS#<workspace>` | `<secret_id>` | List all secrets in workspace | Secret META |
| GSI2 | `gsi2pk`/`gsi2sk` | `GW_MODELDEFS#<workspace>` | `<model_def_id>` | List all model defs in workspace | ModelDef META |
| GSI2 | `gsi2pk`/`gsi2sk` | `GW_ENDPOINTS#<workspace>` | `<endpoint_id>` | List all endpoints in workspace | Endpoint META |
| GSI2 | `gsi2pk`/`gsi2sk` | `GW_BIND#<resource_type>#<resource_id>` | `<endpoint_id>` | Reverse lookup: find endpoints by bound resource | Binding items (only when resource_type + resource_id are known) |
| GSI3 | `gsi3pk`/`gsi3sk` | `GW_MODELDEF_NAME#<workspace>#<name>` | `<model_def_id>` | Unique model def name lookup | ModelDef META |
| GSI3 | `gsi3pk`/`gsi3sk` | `GW_ENDPOINT_NAME#<workspace>#<name>` | `<endpoint_id>` | Unique endpoint name lookup | Endpoint META |
| GSI4 | `gsi4pk`/`gsi4sk` | `GW_MODELDEF_SECRET#<secret_id>` | `<model_def_id>` | Find model defs by secret_id | ModelDef META |
| GSI5 | `gsi5pk`/`gsi5sk` | `GW_ENDPOINT_MODELDEF#<model_def_id>` | `<endpoint_id>` | Find endpoints using a model def | Model Mapping |

### LSI Usage

No LSIs used for gateway items — gateway entities are in their own partitions (not under `EXP#`), and the access patterns are served by GSIs.

## Access Patterns

Every store method maps to Query, GetItem, PutItem, UpdateItem, or DeleteItem — never Scan.

### Secrets

| # | Access Pattern | Caller | Operation | Table/Index | Key Condition | Notes |
|---|---------------|--------|-----------|-------------|---------------|-------|
| AP1 | Get secret by ID | `get_secret_info(secret_id=)` | GetItem | Table | `PK=GW_SECRET#<id>, SK=GW#META` | O(1) |
| AP2 | Get secret by name | `get_secret_info(secret_name=)` | Query | GSI1 | `PK=GW_SECRET_NAME#<ws>#<name>` | Returns secret_id, then AP1 |
| AP3 | Check name uniqueness | `create_gateway_secret` | Query | GSI1 | `PK=GW_SECRET_NAME#<ws>#<name>` | Expect 0 results |
| AP4 | List all secrets | `list_secret_infos` | Query | GSI2 | `PK=GW_SECRETS#<ws>` | Optional FilterExpression on provider |
| AP5 | Find model defs using secret | `delete_gateway_secret` (impact check) | Query | GSI4 | `PK=GW_MODELDEF_SECRET#<secret_id>` | Check before delete |

### Model Definitions

| # | Access Pattern | Caller | Operation | Table/Index | Key Condition | Notes |
|---|---------------|--------|-----------|-------------|---------------|-------|
| AP6 | Get model def by ID | `get_gateway_model_definition(id=)` | GetItem | Table | `PK=GW_MODELDEF#<id>, SK=GW#META` | O(1) |
| AP7 | Get model def by name | `get_gateway_model_definition(name=)` | Query | GSI3 | `PK=GW_MODELDEF_NAME#<ws>#<name>` | Returns ID, then AP6 |
| AP8 | Check name uniqueness | `create_gateway_model_definition` | Query | GSI3 | `PK=GW_MODELDEF_NAME#<ws>#<name>` | Expect 0 results |
| AP9 | List all model defs | `list_gateway_model_definitions` | Query | GSI2 | `PK=GW_MODELDEFS#<ws>` | Optional filter on provider, secret_id |
| AP10 | Find model defs by secret | `list_gateway_model_definitions(secret_id=)` | Query | GSI4 | `PK=GW_MODELDEF_SECRET#<secret_id>` | Efficient filter by secret |
| AP11 | Find endpoints using model def | `delete_gateway_model_definition` (RESTRICT check) | Query | GSI5 | `PK=GW_ENDPOINT_MODELDEF#<model_def_id>` | Must be empty to allow delete |

### Endpoints

| # | Access Pattern | Caller | Operation | Table/Index | Key Condition | Notes |
|---|---------------|--------|-----------|-------------|---------------|-------|
| AP12 | Get endpoint by ID | `get_gateway_endpoint(id=)` | Query | Table | `PK=GW_ENDPOINT#<id>` (all items) | Returns META + mappings + bindings + tags |
| AP13 | Get endpoint by name | `get_gateway_endpoint(name=)` | Query | GSI3 | `PK=GW_ENDPOINT_NAME#<ws>#<name>` | Returns ID, then AP12 |
| AP14 | Check name uniqueness | `create_gateway_endpoint` | Query | GSI3 | `PK=GW_ENDPOINT_NAME#<ws>#<name>` | Expect 0 results |
| AP15 | List all endpoints | `list_gateway_endpoints` | Query | GSI2 | `PK=GW_ENDPOINTS#<ws>` | Returns endpoint IDs, batch-get META+children |
| AP16 | List endpoints by provider | `list_gateway_endpoints(provider=)` | AP15 → filter | — | — | In-memory: filter endpoints having at least one mapping whose model_def has matching provider |
| AP17 | List endpoints by secret | `list_gateway_endpoints(secret_id=)` | AP10 → AP11 | — | — | Find model_defs by secret → find endpoints by model_def → deduplicate |
| AP18 | Delete endpoint (cascade) | `delete_gateway_endpoint` | Query + BatchDelete | Table | `PK=GW_ENDPOINT#<id>` (all items) | Deletes META + all children |
| AP19 | Update endpoint model configs | `update_gateway_endpoint` | BatchDelete + BatchWrite | Table | `SK begins_with GW#MAP#` under endpoint PK | Delete old mappings, write new ones |

### Model Attachment

| # | Access Pattern | Caller | Operation | Table/Index | Key Condition | Notes |
|---|---------------|--------|-----------|-------------|---------------|-------|
| AP20 | Attach model to endpoint | `attach_model_to_endpoint` | PutItem (conditional) | Table | `PK=GW_ENDPOINT#<ep_id>, SK=GW#MAP#<model_def_id>#<linkage_type>` | `ConditionExpression: attribute_not_exists(SK)` for atomic uniqueness. Writes GSI5 projection. |
| AP21 | Detach model from endpoint | `detach_model_from_endpoint` | DeleteItem | Table | `PK=GW_ENDPOINT#<ep_id>, SK=GW#MAP#<model_def_id>#<linkage_type>` | Deterministic SK — no query needed. If `linkage_type` not specified, query `SK begins_with GW#MAP#<model_def_id>#` to find all linkages and delete all. |

### Bindings

| # | Access Pattern | Caller | Operation | Table/Index | Key Condition | Notes |
|---|---------------|--------|-----------|-------------|---------------|-------|
| AP22 | Create binding | `create_endpoint_binding` | PutItem | Table | `PK=GW_ENDPOINT#<ep_id>, SK=GW#BIND#<type>#<res_id>` | Composite SK = unique |
| AP23 | Delete binding | `delete_endpoint_binding` | DeleteItem | Table | Same as AP22 | O(1) |
| AP24 | List bindings (by endpoint) | `list_endpoint_bindings(endpoint_id=)` | Query | Table | `PK=GW_ENDPOINT#<ep_id>, SK begins_with GW#BIND#` | Within endpoint partition |
| AP25 | List bindings (by resource) | `list_endpoint_bindings(resource_type=, resource_id=)` | Query | GSI2 | `PK=GW_BIND#<resource_type>#<resource_id>` | Direct reverse lookup via GSI2. Returns endpoint_ids; batch-get endpoint META for each. |

### Tags

| # | Access Pattern | Caller | Operation | Table/Index | Key Condition | Notes |
|---|---------------|--------|-----------|-------------|---------------|-------|
| AP26 | Set tag | `set_gateway_endpoint_tag` | PutItem + UpdateItem | Table | `PK=GW_ENDPOINT#<ep_id>, SK=GW#TAG#<key>` | Also update denormalized tags on META |
| AP27 | Delete tag | `delete_gateway_endpoint_tag` | DeleteItem + UpdateItem | Table | Same as AP26 | Remove from denormalized tags |

### Scan Risk Assessment

**All 27 access patterns are efficient.** No table scans. Every operation uses GetItem, Query with precise key conditions, PutItem, UpdateItem, or DeleteItem. Cross-entity lookups (e.g., "endpoints using model def") use dedicated GSI indexes. Binding reverse lookups (AP25) use a GSI2 projection for direct access by resource_type + resource_id. The only in-memory filtering is for `list_gateway_endpoints(provider=)` which is bounded by the total endpoint count per workspace (typically small — tens, not thousands).

## Encryption Design

The SQLAlchemy store uses envelope encryption with a Key Encryption Key (KEK) and Data Encryption Key (DEK). The DynamoDB store reuses the same encryption infrastructure.

### KEK Management

Reuse MLflow's `KEKManager` class, which:
- Generates/rotates the KEK
- Stores the KEK in the MLflow tracking server's filesystem or environment
- Is independent of the backend store

The DynamoDB store calls `KEKManager` the same way `SqlAlchemyGatewayStoreMixin` does.

### Encryption Flow (create_gateway_secret)

1. Generate a random DEK (32 bytes).
2. Encrypt secret_value JSON with DEK using AES-GCM.
   - AAD (Additional Authenticated Data) = `f"{secret_id}:{secret_name}"`.
   - Output: nonce (12 bytes) + ciphertext + auth tag (16 bytes) → stored as `encrypted_value` (Binary).
3. Wrap DEK with KEK → stored as `wrapped_dek` (Binary).
4. Store `kek_version` for future KEK rotation.
5. Compute masked_value: `{"key": "prefix...suffix"}` for each key-value pair.

### Decryption Flow (internal, for endpoint proxy)

1. Read `wrapped_dek` and `kek_version`.
2. Unwrap DEK using KEK (matching version).
3. Decrypt `encrypted_value` with DEK using AES-GCM and AAD.
4. Return plaintext secret_value dict.

### DynamoDB Binary Attribute

DynamoDB natively supports Binary (`B`) attributes. `encrypted_value` and `wrapped_dek` are stored as Binary, not Base64-encoded strings. boto3 handles `bytes ↔ B` conversion automatically.

## Store Methods

### Secrets

**`create_gateway_secret(secret_name, secret_value, provider, auth_config, created_by)`**
1. Check name uniqueness via GSI1 (AP3). Raise `RESOURCE_ALREADY_EXISTS` if taken.
2. Generate `s-<ulid>` secret_id.
3. Encrypt: generate DEK, encrypt secret_value with AES-GCM (AAD = `secret_id:secret_name`), wrap DEK with KEK.
4. Compute masked_value.
5. PutItem: `PK=GW_SECRET#<id>, SK=GW#META` with GSI1 and GSI2 projections.
6. Return `GatewaySecretInfo`.

**`get_secret_info(secret_id, secret_name)`**
1. If `secret_id`: GetItem (AP1).
2. If `secret_name`: Query GSI1 (AP2), then GetItem.
3. Return `GatewaySecretInfo` (masked_value, not decrypted).

**`update_gateway_secret(secret_id, secret_value, auth_config, updated_by)`**
1. GetItem to verify exists and get secret_name (needed for AAD).
2. If `secret_value`: re-encrypt with new DEK, same AAD, recompute masked_value.
3. If `auth_config`: update auth_config attribute.
4. UpdateItem on META: set changed attributes, update `last_updated_at`, `last_updated_by`.
5. Invalidate secret cache.
6. Return updated `GatewaySecretInfo`.

**`delete_gateway_secret(secret_id)`**
1. Verify exists (AP1).
2. Query model defs using this secret (AP5/GSI4). For each found: UpdateItem to set `secret_id=NULL` (orphan, don't block delete — matches SQLAlchemy `ondelete=SET NULL`).
3. DeleteItem: `PK=GW_SECRET#<id>, SK=GW#META`.
4. Invalidate secret cache.

**`list_secret_infos(provider)`**
1. Query GSI2: `PK=GW_SECRETS#<ws>` (AP4).
2. If `provider`: FilterExpression on `provider` attribute.
3. Return `list[GatewaySecretInfo]`.

### Model Definitions

**`create_gateway_model_definition(name, secret_id, provider, model_name, created_by)`**
1. Check name uniqueness via GSI3 (AP8).
2. Verify secret exists (AP1). Raise `RESOURCE_DOES_NOT_EXIST` if not.
3. Generate `d-<ulid>` model_definition_id.
4. PutItem: `PK=GW_MODELDEF#<id>, SK=GW#META` with GSI2, GSI3, GSI4 projections.
5. Return `GatewayModelDefinition` with `secret_name` resolved from step 2.

**`get_gateway_model_definition(model_definition_id, name)`**
1. If `model_definition_id`: GetItem (AP6).
2. If `name`: Query GSI3 (AP7), then GetItem.
3. Resolve `secret_name` by reading secret META if `secret_id` is not NULL.
4. Return `GatewayModelDefinition`.

**`list_gateway_model_definitions(provider, secret_id)`**
1. If `secret_id`: Query GSI4 (AP10) for direct lookup.
2. Otherwise: Query GSI2 (AP9) for all model defs.
3. Apply `provider` filter in-memory if specified.
4. Batch-resolve `secret_name` for each model def's `secret_id`.
5. Return `list[GatewayModelDefinition]`.

**`update_gateway_model_definition(model_definition_id, name, secret_id, model_name, updated_by, provider)`**
1. GetItem to verify exists (AP6).
2. If `name` changed: check new name uniqueness via GSI3. Update GSI3 projection.
3. If `secret_id` is explicitly provided:
   a. If `secret_id` is `None`: set `secret_id=NULL` on META (intentional orphan — removes GSI4 projection). This matches SQLAlchemy's nullable FK behavior.
   b. If `secret_id` is a non-empty string: verify new secret exists (AP1). Update GSI4 projection.
4. UpdateItem on META with changed attributes.
5. Invalidate secret cache.
6. Return updated `GatewayModelDefinition`.

**`delete_gateway_model_definition(model_definition_id)`**
1. Query GSI5 (AP11) to check if any endpoints use this model def.
2. If endpoints found: raise `INVALID_STATE` ("model definition is attached to endpoints").
3. DeleteItem: `PK=GW_MODELDEF#<id>, SK=GW#META`.
4. Invalidate secret cache.

### Endpoints

**`create_gateway_endpoint(name, model_configs, created_by, routing_strategy, fallback_config, experiment_id, usage_tracking)`**
1. Check name uniqueness via GSI3 (AP14).
2. If `usage_tracking=True` and no `experiment_id`: create a new experiment via `self.create_experiment()`, tag it with `MLFLOW_EXPERIMENT_SOURCE_TYPE=GATEWAY`, `MLFLOW_EXPERIMENT_SOURCE_ID=<endpoint_id>`, `MLFLOW_EXPERIMENT_IS_GATEWAY=true`.
3. Generate `e-<ulid>` endpoint_id.
4. For each `model_config` in `model_configs`:
   a. Verify model_definition exists (AP6).
   b. Create mapping item: `SK=GW#MAP#<model_def_id>#<linkage_type>` with GSI5 projection.
5. PutItem META: `PK=GW_ENDPOINT#<id>, SK=GW#META` with GSI2, GSI3 projections.
6. Batch write mapping items.
7. Return `GatewayEndpoint` with `model_mappings` populated.

**`get_gateway_endpoint(endpoint_id, name)`**
1. If `name`: resolve to endpoint_id via GSI3 (AP13).
2. Query all items in partition (AP12): META + mappings + bindings + tags.
3. For each mapping item: resolve `model_definition` by reading ModelDef META.
4. Construct `GatewayEndpoint` with `model_mappings`, `tags`.
5. Return `GatewayEndpoint`.

**`list_gateway_endpoints(provider, secret_id)`**
1. If `secret_id` provided (targeted path — skip full list):
   a. Find model_defs by secret (GSI4 → AP10). Collect model_def_ids.
   b. For each model_def_id: find endpoints via GSI5 (AP11). Collect endpoint_ids, deduplicate.
   c. For each endpoint_id: query partition (AP12) to get META + children.
   d. Return `list[GatewayEndpoint]`.
2. Otherwise (no secret_id filter):
   a. Query GSI2 (AP15) for all endpoint IDs in workspace.
   b. For each endpoint_id: query partition (AP12) to get META + children.
   c. If `provider`: filter endpoints having at least one mapping whose model_def has matching provider (in-memory, bounded by endpoint count).
   d. Return `list[GatewayEndpoint]`.

**`update_gateway_endpoint(endpoint_id, name, updated_by, routing_strategy, fallback_config, model_configs, experiment_id, usage_tracking)`**
1. Get existing endpoint (AP12).
2. If `name` changed: check uniqueness, update GSI3 projection.
3. If `usage_tracking` changed to True and no experiment_id: auto-create experiment (same as create).
4. If `model_configs` provided (full replacement):
   a. Query existing mappings (`SK begins_with GW#MAP#`).
   b. Batch delete all existing mapping items.
   c. For each new config: verify model_def exists, create mapping item `SK=GW#MAP#<model_def_id>#<linkage_type>` with GSI5 projection.
   d. Batch write new mapping items.
5. UpdateItem on META with changed attributes.
6. Invalidate secret cache.
7. Return updated `GatewayEndpoint`.

**`delete_gateway_endpoint(endpoint_id)`**
1. Query ALL items with `PK=GW_ENDPOINT#<id>` (AP18).
2. Batch delete all items (META, mappings, bindings, tags).
3. GSI entries auto-cleaned.
4. Invalidate secret cache.

### Model Attachment

**`attach_model_to_endpoint(endpoint_id, model_config, created_by)`**
1. Verify endpoint exists (GetItem META).
2. Verify model_definition exists (AP6).
3. Construct deterministic SK: `GW#MAP#<model_definition_id>#<linkage_type>`.
4. PutItem with `ConditionExpression: attribute_not_exists(SK)` (AP20). If condition fails: raise `RESOURCE_ALREADY_EXISTS`. This atomically enforces the `(endpoint_id, model_definition_id, linkage_type)` uniqueness constraint — no read-then-write race.
5. Write GSI5 projection on the mapping item.
6. Update endpoint META `last_updated_at`.
7. Invalidate secret cache.
8. Return `GatewayEndpointModelMapping`.

**`detach_model_from_endpoint(endpoint_id, model_definition_id, linkage_type=None)`**

The SQLAlchemy store accepts an optional `linkage_type` to disambiguate when the same model definition is attached with both PRIMARY and FALLBACK linkages. The unique constraint is on `(endpoint_id, model_definition_id, linkage_type)`.

1. If `linkage_type` specified: construct deterministic SK `GW#MAP#<model_definition_id>#<linkage_type>`, DeleteItem directly (AP21). If item doesn't exist: raise `RESOURCE_DOES_NOT_EXIST`.
2. If `linkage_type` not specified: query `SK begins_with GW#MAP#<model_definition_id>#` to find all linkage variants. If none found: raise `RESOURCE_DOES_NOT_EXIST`. Delete all matching items.
3. Update endpoint META `last_updated_at`.
4. Invalidate secret cache.

### Bindings

**`create_endpoint_binding(endpoint_id, resource_type, resource_id, created_by)`**
1. Verify endpoint exists.
2. PutItem: `SK=GW#BIND#<resource_type>#<resource_id>` (AP22). Composite SK ensures uniqueness. Write GSI2 projection `PK=GW_BIND#<resource_type>#<resource_id>` for reverse lookup (AP25).
3. Invalidate secret cache.
4. Return `GatewayEndpointBinding`.

**`delete_endpoint_binding(endpoint_id, resource_type, resource_id)`**
1. GetItem to verify binding exists: `SK=GW#BIND#<resource_type>#<resource_id>`. If not found: raise `RESOURCE_DOES_NOT_EXIST` (matches SQLAlchemy behavior — not idempotent).
2. DeleteItem (AP23).
3. Invalidate secret cache.

**`list_endpoint_bindings(endpoint_id, resource_type, resource_id)`**
1. If `endpoint_id`: Query `SK begins_with GW#BIND#` (AP24).
   - If `resource_type` also specified: narrow prefix to `GW#BIND#<resource_type>#`.
   - If `resource_id` also specified: narrow to exact SK.
2. If no `endpoint_id` but `resource_type` and `resource_id` specified (cross-endpoint query): Query GSI2 `PK=GW_BIND#<resource_type>#<resource_id>` (AP25) for direct reverse lookup. Returns endpoint_ids; batch-get binding items from each endpoint partition.
3. If no `endpoint_id` and no resource filter: Query GSI2 for all endpoints (`PK=GW_ENDPOINTS#<ws>`), then query bindings per endpoint. Bounded by endpoint count.
4. For each binding: resolve `endpoint_name` from META, attach `model_mappings`.
5. Return `list[GatewayEndpointBinding]`.

### Tags

**`set_gateway_endpoint_tag(endpoint_id, tag)`**
1. PutItem: `SK=GW#TAG#<key>` (upsert) (AP26).
2. Update denormalized tags dict on META, update `last_updated_at`.

**`delete_gateway_endpoint_tag(endpoint_id, key)`**
1. DeleteItem: `SK=GW#TAG#<key>` (AP27).
2. Remove key from denormalized tags on META, update `last_updated_at`.

## Schema Constants

New constants to add to `dynamodb/schema.py`:

```python
# Gateway partition prefixes
PK_GW_SECRET_PREFIX = "GW_SECRET#"
PK_GW_MODELDEF_PREFIX = "GW_MODELDEF#"
PK_GW_ENDPOINT_PREFIX = "GW_ENDPOINT#"

# Gateway sort keys
SK_GW_META = "GW#META"
SK_GW_MAP_PREFIX = "GW#MAP#"
SK_GW_BIND_PREFIX = "GW#BIND#"
SK_GW_TAG_PREFIX = "GW#TAG#"

# GSI prefixes for gateway (logical labels — all use shared gsiNpk/gsiNsk attributes)
GSI1_GW_SECRET_NAME_PREFIX = "GW_SECRET_NAME#"
GSI2_GW_SECRETS_PREFIX = "GW_SECRETS#"
GSI2_GW_MODELDEFS_PREFIX = "GW_MODELDEFS#"
GSI2_GW_ENDPOINTS_PREFIX = "GW_ENDPOINTS#"
GSI2_GW_BIND_PREFIX = "GW_BIND#"
GSI3_GW_MODELDEF_NAME_PREFIX = "GW_MODELDEF_NAME#"
GSI3_GW_ENDPOINT_NAME_PREFIX = "GW_ENDPOINT_NAME#"
GSI4_GW_MODELDEF_SECRET_PREFIX = "GW_MODELDEF_SECRET#"
GSI5_GW_ENDPOINT_MODELDEF_PREFIX = "GW_ENDPOINT_MODELDEF#"
```

## Entity Mapping

| MLflow Entity | DynamoDB Source |
|---------------|----------------|
| `GatewaySecretInfo.secret_id` | Extracted from PK (`GW_SECRET#<id>`) |
| `GatewaySecretInfo.secret_name` | META `secret_name` attribute |
| `GatewaySecretInfo.masked_values` | META `masked_value` attribute (singular in DB, plural in entity — JSON → dict). When constructing entity, pass DB's `masked_value` to entity's `masked_values` parameter. |
| `GatewaySecretInfo.provider` | META `provider` attribute |
| `GatewaySecretInfo.auth_config` | META `auth_config` attribute (JSON → dict) |
| `GatewayModelDefinition.model_definition_id` | Extracted from PK |
| `GatewayModelDefinition.name` | META `name` attribute |
| `GatewayModelDefinition.secret_id` | META `secret_id` attribute (nullable) |
| `GatewayModelDefinition.secret_name` | Resolved from Secret META (lazy) |
| `GatewayModelDefinition.provider` | META `provider` attribute |
| `GatewayModelDefinition.model_name` | META `model_name` attribute |
| `GatewayEndpoint.endpoint_id` | Extracted from PK |
| `GatewayEndpoint.name` | META `name` attribute |
| `GatewayEndpoint.model_mappings` | Mapping items (`SK begins_with GW#MAP#`) |
| `GatewayEndpoint.tags` | Tag items (`SK begins_with GW#TAG#`) + denormalized |
| `GatewayEndpoint.routing_strategy` | META `routing_strategy` attribute → `RoutingStrategy` enum |
| `GatewayEndpoint.fallback_config` | META `fallback_config` attribute (JSON → `FallbackConfig`) |
| `GatewayEndpoint.experiment_id` | META `experiment_id` attribute |
| `GatewayEndpoint.usage_tracking` | META `usage_tracking` attribute (bool) |
| `GatewayEndpointModelMapping.mapping_id` | Synthesized as `<model_def_id>:<linkage_type>` from SK (`GW#MAP#<model_def_id>#<linkage_type>`) |
| `GatewayEndpointModelMapping.model_definition` | Resolved from ModelDef META |
| `GatewayEndpointBinding.resource_type` | Extracted from SK (`GW#BIND#<type>#<id>`) |
| `GatewayEndpointBinding.resource_id` | Extracted from SK |

## Secret Cache Invalidation

The SQLAlchemy store calls `_invalidate_secret_cache()` after every gateway mutation — not just secret mutations. The cache (`config_resolver.py`) builds `GatewayEndpointConfig` objects that include resolved secrets, model definitions, and endpoint configurations. Any change to any gateway entity can invalidate this cache.

The DynamoDB store calls the same inherited `_invalidate_secret_cache()` method after **all 11 mutation methods**:

| Method | Reason |
|--------|--------|
| `create_gateway_secret` | New secret available |
| `update_gateway_secret` | Secret value or auth_config changed |
| `delete_gateway_secret` | Secret removed; model defs orphaned |
| `update_gateway_model_definition` | Model def may now point to different secret |
| `delete_gateway_model_definition` | Model def removed from endpoint configs |
| `update_gateway_endpoint` | Model mappings or routing config changed |
| `delete_gateway_endpoint` | Endpoint removed |
| `attach_model_to_endpoint` | New model mapping affects endpoint config |
| `detach_model_from_endpoint` | Model mapping removed |
| `create_endpoint_binding` | New binding affects config resolution |
| `delete_endpoint_binding` | Binding removed |

No additional cache infrastructure is needed — the inherited `_invalidate_secret_cache()` method handles it.

## Testing Strategy

### Unit Tests (moto, direct store)

**Secrets:**
- CRUD lifecycle: create → get (by ID) → get (by name) → update → delete
- Name uniqueness: duplicate name in workspace rejected
- Update secret_value: re-encryption, masked_value changes
- Update auth_config only: secret_value unchanged
- Delete cascades to model_defs: orphaned model_defs get `secret_id=NULL`
- List with provider filter
- Encryption round-trip: create with known value, verify decryption matches (via internal method)
- Binary storage: verify `encrypted_value` and `wrapped_dek` stored as Binary

**Model Definitions:**
- CRUD lifecycle: create → get (by ID) → get (by name) → update → delete
- Name uniqueness within workspace
- RESTRICT delete: model def attached to endpoint cannot be deleted
- Update name: GSI3 projection updated
- Update secret_id: GSI4 projection updated, old secret orphaned
- secret_name resolution: resolved from Secret META
- Orphaned model def (secret deleted): `secret_id=NULL`, `secret_name=None`

**Endpoints:**
- CRUD lifecycle: create with model configs → get → update → delete
- Name uniqueness within workspace
- Cascade delete: META + mappings + bindings + tags all deleted
- Model config replacement on update: old mappings deleted, new created
- Auto-experiment creation: `usage_tracking=True` creates tagged experiment
- Get returns full entity: mappings + bindings + tags
- Routing strategy and fallback config serialization

**Model Attachment:**
- Attach and detach model
- Duplicate attachment rejected (atomic via ConditionExpression)
- Concurrent duplicate attachment — both PutItem race, one fails with ConditionalCheckFailedException → RESOURCE_ALREADY_EXISTS
- Detach with linkage_type — removes specific linkage
- Detach without linkage_type — removes all linkages for that model_def
- Detach non-existent raises error
- Same model_def with PRIMARY and FALLBACK — two separate mapping items
- GSI5 projection: find endpoints by model_def works after attach

**Bindings:**
- Create and delete binding
- Delete non-existent binding raises RESOURCE_DOES_NOT_EXIST (not idempotent — matches SQLAlchemy)
- List by endpoint_id
- List by resource_type + resource_id (cross-endpoint via GSI2 reverse lookup)
- Composite SK uniqueness: same resource can't bind twice to same endpoint

**Tags:**
- Set, overwrite, delete
- Denormalized tags dict on META stays in sync

### Integration Tests (moto server, REST)

- Gateway secret CRUD via REST endpoints
- Model definition CRUD with secret resolution
- Endpoint CRUD with model configs
- Binding operations via REST

### E2E Tests (full server)

- Create secret → create model def → create endpoint → verify proxy works
- Endpoint with usage_tracking creates experiment automatically

### Coverage

100% patch coverage on new code.

## Implementation Sub-Phases

Given the size (22 methods), this phase should be implemented in 3 sub-phases:

### Phase 5a: Secrets (5 methods)
- `create_gateway_secret`, `get_secret_info`, `update_gateway_secret`, `delete_gateway_secret`, `list_secret_infos`
- Foundation — model defs and endpoints depend on secrets

### Phase 5b: Model Definitions (5 methods)
- `create_gateway_model_definition`, `get_gateway_model_definition`, `list_gateway_model_definitions`, `update_gateway_model_definition`, `delete_gateway_model_definition`
- Depends on Phase 5a for secret resolution

### Phase 5c: Endpoints, Attachments, Bindings, Tags (12 methods)
- All endpoint CRUD, attach/detach, binding CRUD, tag operations
- Depends on Phase 5b for model_definition resolution

## Out of Scope

- KEK management infrastructure (reuses MLflow's built-in `KEKManager`)
- Gateway proxy/routing logic (MLflow server, not tracking store)
- AI Gateway configuration file format
- Secret rotation scheduler
- Endpoint health checks / circuit breaker
