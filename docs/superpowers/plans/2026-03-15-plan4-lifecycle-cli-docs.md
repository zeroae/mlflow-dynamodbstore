# Plan 4: Lifecycle, CLI & Documentation — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the plugin with TTL-based data lifecycle management, admin CLI commands, comprehensive documentation, and MLflow compatibility test gap closure. After this plan, `mlflow-dynamodbstore` is ready for v0.1.0 release.

**Architecture:** TTL attributes on DynamoDB items for automatic expiration. Background cleanup CLI for orphaned experiment children. Admin CLI for tag denormalization config, FTS trigram config, TTL policy, span pre-caching, workspace deletion. MkDocs documentation covering quickstart, configuration, architecture, and API reference.

**Tech Stack:** Same as Plans 1-3, plus `click` for CLI (new dependency).

**Spec:** `docs/superpowers/specs/2026-03-15-mlflow-dynamodbstore-design.md`

**Depends on:** Plans 1 + 2 + 3 (foundation, CRUD, search, FTS, traces, X-Ray, auth)

**What Plan 2 provides (used by Plan 4):**
- `dynamodb/config.py`: `ConfigReader` with methods for denormalize patterns + FTS trigram fields. **Plan 4 extends this** with TTL policy methods (`get_ttl_policy`, `set_ttl_policy`, `reconcile` for TTL env vars).
- `dynamodb/fts.py`: `fts_items_for_text()`, `fts_diff()` — used by backfill commands
- `dynamodb/table.py`: `DynamoDBTable` — used by CLI commands for direct table operations

**What Plan 3 provides (used by Plan 4):**
- Trace CRUD with TTL on trace creation (if Plan 3 implemented TTL on `start_trace`, Plan 4 Task 3 is already done — verify and skip)
- `get_trace` with X-Ray span proxy + caching — used by `cache-spans` CLI
- Auth store — used by lifecycle integration test
- `XRayClient` — used by `cache-spans` CLI

**`click` is NOT in pyproject.toml yet** — must be added in Task 6.

---

## File Structure (new/modified files only)

```
src/mlflow_dynamodbstore/
├── tracking_store.py               # MODIFY: Add TTL to soft-delete/restore operations
├── dynamodb/
│   └── config.py                   # MODIFY: Add TTL policy methods to ConfigReader
├── cli/
│   ├── __init__.py                 # NEW: Click group
│   ├── denormalize_tags.py         # NEW: denormalize-tags list/add/remove/backfill
│   ├── fts_trigrams.py             # NEW: fts-trigrams list/add/backfill
│   ├── ttl_policy.py               # NEW: ttl-policy show/set
│   ├── cleanup_expired.py          # NEW: cleanup-expired
│   ├── cache_spans.py              # NEW: cache-spans
│   └── delete_workspace.py         # NEW: delete-workspace

pyproject.toml                      # MODIFY: Add click dep + CLI entry point

tests/
├── unit/
│   ├── test_ttl.py                 # NEW: TTL lifecycle tests
│   └── cli/
│       ├── test_denormalize_tags.py # NEW
│       ├── test_fts_trigrams.py     # NEW
│       ├── test_ttl_policy.py       # NEW
│       ├── test_cleanup_expired.py  # NEW
│       ├── test_cache_spans.py      # NEW
│       └── test_delete_workspace.py # NEW
├── integration/
│   └── test_lifecycle.py           # NEW: Full lifecycle integration
├── compatibility/
│   ├── test_mlflow_tracking.py     # MODIFY: Close test gaps from Plans 1-3
│   └── test_mlflow_registry.py     # MODIFY: Close test gaps

docs/
├── index.md                        # MODIFY: Expand with feature overview
├── user-guide/
│   ├── quickstart.md               # MODIFY: Full getting started guide
│   ├── configuration.md            # NEW: URI format, env vars, all CONFIG items
│   ├── workspaces.md               # NEW: Workspace setup and usage
│   └── xray-integration.md         # NEW: OTel dual-export + X-Ray setup
├── operator-guide/
│   ├── cli-reference.md            # NEW: All admin CLI commands
│   ├── ttl-lifecycle.md            # NEW: TTL policies, retention, cleanup
│   ├── monitoring.md               # NEW: CloudWatch metrics, partition size
│   └── upgrading.md                # NEW: v2 upgrade path (streams, OpenSearch)
├── reference/
│   ├── api.md                      # MODIFY: Complete API reference
│   ├── schema.md                   # NEW: DynamoDB table schema reference
│   └── access-patterns.md          # NEW: All 45+ access patterns documented
├── contributing/
│   └── development.md              # NEW: Dev setup, testing, PR workflow
└── adr/
    └── 001-single-table-design.md  # NEW: Architecture Decision Record
```

---

## Chunk 1: TTL Lifecycle Management

### Task 1: TTL policy config reader

Extend `dynamodb/config.py` `ConfigReader` with TTL policy methods.

**Files:**
- Modify: `src/mlflow_dynamodbstore/dynamodb/config.py`
- Modify: `tests/unit/dynamodb/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/dynamodb/test_config.py (extend existing)
class TestTTLPolicy:
    def test_default_ttl_policy(self, config_reader):
        policy = config_reader.get_ttl_policy()
        assert policy["soft_deleted_retention_days"] == 90
        assert policy["trace_retention_days"] == 30
        assert policy["metric_history_retention_days"] == 365

    def test_set_ttl_policy(self, config_reader):
        config_reader.set_ttl_policy(soft_deleted_retention_days=60)
        policy = config_reader.get_ttl_policy()
        assert policy["soft_deleted_retention_days"] == 60

    def test_ttl_zero_means_disabled(self, config_reader):
        config_reader.set_ttl_policy(trace_retention_days=0)
        assert config_reader.get_trace_ttl_seconds() is None  # disabled

    def test_get_trace_ttl_seconds(self, config_reader):
        ttl = config_reader.get_trace_ttl_seconds()
        assert ttl == 30 * 86400  # 30 days in seconds

    def test_get_soft_deleted_ttl_seconds(self, config_reader):
        ttl = config_reader.get_soft_deleted_ttl_seconds()
        assert ttl == 90 * 86400

    def test_get_metric_history_ttl_seconds(self, config_reader):
        ttl = config_reader.get_metric_history_ttl_seconds()
        assert ttl == 365 * 86400

    def test_reconcile_ttl_from_env(self, config_reader, monkeypatch):
        monkeypatch.setenv("MLFLOW_DYNAMODB_SOFT_DELETED_RETENTION_DAYS", "60")
        config_reader.reconcile()
        policy = config_reader.get_ttl_policy()
        assert policy["soft_deleted_retention_days"] == 60
```

- [ ] **Step 2: Implement TTL policy methods on ConfigReader**

Add to `ConfigReader`:
- `get_ttl_policy() -> dict` — reads `CONFIG#TTL_POLICY`
- `set_ttl_policy(**kwargs)` — updates individual fields
- `get_trace_ttl_seconds() -> int | None` — returns seconds or None if disabled (0)
- `get_soft_deleted_ttl_seconds() -> int | None`
- `get_metric_history_ttl_seconds() -> int | None`
- Extend `reconcile()` to read `MLFLOW_DYNAMODB_SOFT_DELETED_RETENTION_DAYS`, `MLFLOW_DYNAMODB_TRACE_RETENTION_DAYS`, `MLFLOW_DYNAMODB_METRIC_HISTORY_RETENTION_DAYS`

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Run existing config tests to verify no regressions**

```bash
uv run pytest tests/unit/dynamodb/test_config.py -v
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add TTL policy methods to ConfigReader"
```

### Task 2: TTL on soft-delete (delete_run + restore_run)

Modify `tracking_store.py`: `delete_run` sets TTL on run META + all children. `restore_run` removes TTL.

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_ttl.py
import time

class TestRunTTL:
    def test_delete_run_sets_ttl_on_meta(self, tracking_store, table):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.delete_run(run.info.run_id)
        item = table.get_item(f"EXP#{exp_id}", f"R#{run.info.run_id}")
        assert "ttl" in item
        assert item["ttl"] > time.time()

    def test_delete_run_sets_ttl_on_children(self, tracking_store, table):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.log_batch(run.info.run_id, metrics=[Metric("acc", 0.9, 0, 0)], params=[], tags=[])
        tracking_store.delete_run(run.info.run_id)
        children = table.query(pk=f"EXP#{exp_id}", sk_prefix=f"R#{run.info.run_id}#")
        for child in children:
            assert "ttl" in child

    def test_restore_run_removes_ttl(self, tracking_store, table):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.delete_run(run.info.run_id)
        tracking_store.restore_run(run.info.run_id)
        item = table.get_item(f"EXP#{exp_id}", f"R#{run.info.run_id}")
        assert "ttl" not in item
        children = table.query(pk=f"EXP#{exp_id}", sk_prefix=f"R#{run.info.run_id}#")
        for child in children:
            assert "ttl" not in child

    def test_rank_items_get_ttl_not_deleted(self, tracking_store, table):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.log_batch(run.info.run_id, metrics=[Metric("acc", 0.9, 0, 0)], params=[], tags=[])
        tracking_store.delete_run(run.info.run_id)
        rank_items = table.query(pk=f"EXP#{exp_id}", sk_prefix="RANK#m#acc#")
        assert len(rank_items) == 1
        assert "ttl" in rank_items[0]

    def test_fts_items_get_ttl(self, tracking_store, table):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="my-pipeline")
        tracking_store.delete_run(run.info.run_id)
        fts_items = table.query(pk=f"EXP#{exp_id}", sk_prefix="FTS#")
        for item in fts_items:
            if run.info.run_id in item.get("SK", ""):
                assert "ttl" in item

    def test_dlink_items_get_ttl(self, tracking_store, table):
        # Create run with dataset inputs, delete, verify DLINK has TTL
        ...

    def test_ttl_disabled_when_zero(self, tracking_store, table, monkeypatch):
        """When soft_deleted_retention_days=0, no TTL set."""
        tracking_store._config.set_ttl_policy(soft_deleted_retention_days=0)
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.delete_run(run.info.run_id)
        item = table.get_item(f"EXP#{exp_id}", f"R#{run.info.run_id}")
        assert "ttl" not in item  # disabled
```

- [ ] **Step 2: Modify delete_run**

Current Plan 1 `delete_run` changes lifecycle_stage. Add:
1. Read `soft_deleted_ttl_seconds` from ConfigReader
2. If not None: compute `ttl = int(time.time()) + ttl_seconds`
3. Set `ttl` on run META via UpdateItem
4. Query all children `SK begins_with(f"R#{run_id}#")` + RANK items + DLINK items + FTS items
5. BatchWriteItem: update each with `ttl` attribute
6. Do NOT delete RANK items (they get TTL like everything else)

- [ ] **Step 3: Modify restore_run**

1. Remove `ttl` from run META via UpdateItem `REMOVE ttl`
2. Query all children + RANK + DLINK + FTS items
3. BatchWriteItem: update each to `REMOVE ttl`

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Run Plan 1+2 tests to verify no regressions**

```bash
uv run pytest tests/unit/test_tracking_store.py tests/integration/test_tracking_crud.py -v
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: add TTL to delete_run/restore_run (no immediate RANK deletion)"
```

### Task 3: TTL on delete_experiment + restore_experiment

- [ ] **Step 1: Write failing tests**

Test delete_experiment sets TTL on experiment META only (children untouched). Test restore_experiment removes TTL from META.

- [ ] **Step 2: Implement**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add TTL to delete_experiment/restore_experiment"
```

### Task 4: TTL on metric history

Modify `log_batch`: set TTL on metric history items from `config.get_metric_history_ttl_seconds()`. Metric latest items do NOT get TTL.

- [ ] **Step 1: Write failing tests**

```python
class TestMetricHistoryTTL:
    def test_metric_history_has_ttl(self, tracking_store, table):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.log_batch(run.info.run_id, metrics=[Metric("loss", 0.5, 1, 100)], params=[], tags=[])
        # History item should have TTL
        history = table.query(pk=f"EXP#{exp_id}", sk_prefix=f"R#{run.info.run_id}#MHIST#")
        assert len(history) > 0
        assert "ttl" in history[0]

    def test_metric_latest_no_ttl(self, tracking_store, table):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.log_batch(run.info.run_id, metrics=[Metric("loss", 0.5, 1, 100)], params=[], tags=[])
        latest = table.get_item(f"EXP#{exp_id}", f"R#{run.info.run_id}#METRIC#loss")
        assert "ttl" not in latest
```

- [ ] **Step 2: Implement**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add TTL to metric history items"
```

### Task 5: Verify trace TTL (from Plan 3)

If Plan 3 already sets TTL on `start_trace` and trace children, verify and skip. If not, implement here.

- [ ] **Step 1: Write/run verification test**

```python
class TestTraceTTL:
    def test_trace_meta_has_ttl(self, tracking_store, table):
        # start_trace should set TTL from trace_retention_days
        ...

    def test_trace_children_inherit_ttl(self, tracking_store, table):
        # Tags, metadata, assessments should have same TTL
        ...
```

- [ ] **Step 2: If tests pass → skip (already done in Plan 3). If fail → implement.**
- [ ] **Step 3: Commit if changes made**

```bash
git commit -m "feat: verify/add TTL to trace creation"
```

---

## Chunk 2: Admin CLI

### Task 6: CLI framework

**Files:**
- Create: `src/mlflow_dynamodbstore/cli/__init__.py`
- Modify: `pyproject.toml` (add `click>=8.0.0` dependency + CLI entry point)

- [ ] **Step 1: Add click dependency**

```toml
# pyproject.toml
dependencies = [
    ...
    "click>=8.0.0",
]

[project.scripts]
mlflow-dynamodbstore = "mlflow_dynamodbstore.cli:cli"
```

- [ ] **Step 2: Set up Click group**

```python
# src/mlflow_dynamodbstore/cli/__init__.py
"""Admin CLI for mlflow-dynamodbstore."""
import click

@click.group()
def cli():
    """mlflow-dynamodbstore admin commands."""
    pass
```

- [ ] **Step 3: Install and verify**

```bash
uv sync
uv run mlflow-dynamodbstore --help
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add CLI framework with Click"
```

### Task 7: denormalize-tags CLI

**Files:**
- Create: `src/mlflow_dynamodbstore/cli/denormalize_tags.py`
- Create: `tests/unit/cli/test_denormalize_tags.py`

- [ ] **Step 1: Write failing tests**

Test via `CliRunner`: `list` shows `mlflow.*`, `add` appends patterns, `remove` removes patterns (but `mlflow.*` re-added on next reconcile), `add --experiment-id` writes per-experiment config, `backfill` scans tag items and denormalizes onto META items. Use `@mock_aws` for DynamoDB.

- [ ] **Step 2: Implement**

CLI commands use `ConfigReader` from Plan 2 for pattern management. `backfill` command:
1. Read effective patterns for experiment
2. Scan all tag items in experiment partition
3. For each tag matching a pattern: UpdateItem META `SET tags.<key> = <value>`
4. Report progress

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add denormalize-tags CLI commands"
```

### Task 8: fts-trigrams CLI

- [ ] **Step 1: Write failing tests**

`list` shows current fields, `add` appends fields, `backfill` re-tokenizes existing text and writes FTS items.

- [ ] **Step 2: Implement (same pattern as denormalize-tags, uses ConfigReader)**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add fts-trigrams CLI commands"
```

### Task 9: ttl-policy CLI

- [ ] **Step 1: Write failing tests**

`show` displays current policy, `set` updates individual fields.

- [ ] **Step 2: Implement (uses ConfigReader.get_ttl_policy/set_ttl_policy)**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add ttl-policy CLI commands"
```

### Task 10: cleanup-expired CLI

- [ ] **Step 1: Write failing tests**

Create experiment, soft-delete, manually delete META item (simulating TTL expiry), run `cleanup-expired`, verify orphaned children get `ttl = now`. Test `--dry-run`.

- [ ] **Step 2: Implement**

1. List all known experiment IDs (from GSI2 or scan CONFIG)
2. For each: check if `E#META` exists
3. If missing (TTL-deleted): query all remaining items in partition, set `ttl = now`
4. Report progress

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add cleanup-expired CLI command"
```

### Task 11: cache-spans CLI

- [ ] **Step 1: Write failing tests**

Create traces, mock X-Ray to return spans, run `cache-spans --experiment-id <id>`, verify `T#<trace_id>#SPANS` items written + span attributes denormalized + FTS items. Test `--days` option. Test idempotent re-run.

Note: X-Ray must be mocked (`unittest.mock.patch`).

- [ ] **Step 2: Implement**

Iterate traces in experiment(s) that lack `T#<trace_id>#SPANS`, call `tracking_store.get_trace()` for each (triggers caching + indexing from Plan 3). Report progress.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add cache-spans CLI command"
```

### Task 12: delete-workspace CLI

- [ ] **Step 1: Write failing tests**

Test `--mode soft` (workspace META status='deleted'). Test `--mode cascade` (all experiments + models deleted). Test default workspace rejected. Test `--yes` skips confirmation.

- [ ] **Step 2: Implement**

Soft mode: UpdateItem workspace META. Cascade mode: query GSI2 for all experiments + models in workspace, delete each partition, then delete workspace META. Report progress. Require `--yes` or interactive confirmation for cascade.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add delete-workspace CLI command"
```

---

## Chunk 3: Documentation

### Task 13: User guide documentation

**Files:**
- Modify: `docs/index.md`
- Modify: `docs/user-guide/quickstart.md`
- Create: `docs/user-guide/configuration.md`
- Create: `docs/user-guide/workspaces.md`
- Create: `docs/user-guide/xray-integration.md`

- [ ] **Step 1: Write quickstart**

Full getting started: `uv pip install`, configure URI, run `mlflow server`, create experiment, log run, view in UI.

- [ ] **Step 2: Write configuration guide**

URI format (`dynamodb://region/table`, `dynamodb://endpoint/table`), all environment variables, CONFIG items (denormalize patterns, TTL policy, FTS trigrams, X-Ray annotation mapping), config reconciliation behavior.

- [ ] **Step 3: Write workspace guide**

Enable workspaces (`MLFLOW_ENABLE_WORKSPACES=1`), create workspaces, scope experiments/models, workspace permissions, artifact isolation.

- [ ] **Step 4: Write X-Ray integration guide**

OTel dual-export setup, annotation processor configuration, span search, lazy caching, pre-caching with CLI, retention considerations.

- [ ] **Step 5: Verify docs build**

```bash
uv run mkdocs build --strict
```

- [ ] **Step 6: Commit**

```bash
git commit -m "docs: add user guide (quickstart, configuration, workspaces, X-Ray)"
```

### Task 14: Operator guide documentation

**Files:**
- Create: `docs/operator-guide/cli-reference.md`
- Create: `docs/operator-guide/ttl-lifecycle.md`
- Create: `docs/operator-guide/monitoring.md`
- Create: `docs/operator-guide/upgrading.md`

- [ ] **Step 1: Write CLI reference**

Document all CLI commands with examples: denormalize-tags, fts-trigrams, ttl-policy, cleanup-expired, cache-spans, delete-workspace.

- [ ] **Step 2: Write TTL lifecycle guide**

Retention policies, soft-delete flow, automatic expiration, background cleanup, data recovery window, interaction between TTL and restore.

- [ ] **Step 3: Write monitoring guide**

CloudWatch metrics to watch (ConsumedReadCapacity, ConsumedWriteCapacity, ThrottledRequests), partition size monitoring, DynamoDB item count, X-Ray trace count, 10GB LSI partition limit.

- [ ] **Step 4: Write upgrade guide**

v2 path: DynamoDB Streams + Lambda for async materialization (zae-mlflow CDK), OpenSearch Serverless for FTS upgrade, EventBridge-scheduled cleanup.

- [ ] **Step 5: Verify docs build**
- [ ] **Step 6: Commit**

```bash
git commit -m "docs: add operator guide (CLI, TTL, monitoring, upgrading)"
```

### Task 15: Reference documentation

**Files:**
- Modify: `docs/reference/api.md`
- Create: `docs/reference/schema.md`
- Create: `docs/reference/access-patterns.md`

- [ ] **Step 1: Write schema reference**

Complete DynamoDB table schema from the spec: all entity types, PK/SK patterns, LSI/GSI attributes, partition families.

- [ ] **Step 2: Write access pattern reference**

All 45+ access patterns with mechanism descriptions from the spec's Access Pattern Coverage section.

- [ ] **Step 3: Expand API reference**

Ensure mkdocstrings generates docs for all public modules: `tracking_store`, `registry_store`, `workspace_store`, `auth.store`, `xray.client`, `dynamodb.table`, `dynamodb.config`, `dynamodb.fts`, `dynamodb.search`.

- [ ] **Step 4: Verify docs build**
- [ ] **Step 5: Commit**

```bash
git commit -m "docs: add reference docs (schema, access patterns, API)"
```

### Task 16: Contributing guide + ADR

**Files:**
- Create: `docs/contributing/development.md`
- Create: `docs/adr/001-single-table-design.md`

- [ ] **Step 1: Write development guide**

Dev setup (uv, pre-commit), testing (markers: unit/integration/compatibility/smoke, moto), PR workflow, commit conventions (conventional commits for cliff).

- [ ] **Step 2: Write ADR for single-table design**

Decision, context, alternatives considered (multi-table, hybrid), consequences, trade-offs.

- [ ] **Step 3: Commit**

```bash
git commit -m "docs: add contributing guide and architecture decision record"
```

---

## Chunk 4: MLflow Compatibility + Release Prep

### Task 17: MLflow compatibility test gap closure

**Files:**
- Modify: `tests/compatibility/test_mlflow_tracking.py`
- Modify: `tests/compatibility/test_mlflow_registry.py`

- [ ] **Step 1: Run MLflow's test suite, identify remaining failures**

```bash
uv run pytest tests/compatibility/ -v --tb=short 2>&1 | tee compatibility-report.txt
grep FAILED compatibility-report.txt
```

- [ ] **Step 2: Fix each failure category**

Common categories:
- Missing method implementations → implement or raise appropriate MlflowException
- Return type mismatches → ensure entities match MLflow's expected types
- Edge cases in search behavior → adjust query planner
- Default experiment handling → verify experiment "0" works
- Trace method signatures → verify match MLflow's expected interface

- [ ] **Step 3: Re-run, iterate until all applicable tests pass**

Document skipped tests (Databricks-specific, gateway-specific) with `@pytest.mark.skip(reason="...")`.

- [ ] **Step 4: Commit**

```bash
git commit -m "test: close MLflow compatibility test gaps"
```

### Task 18: CI workflow updates

**Files:**
- Modify: `.github/workflows/ci-tests.yml`
- Create: `.github/workflows/docs.yml`
- Create: `.github/workflows/release.yml`

Reference: `/home/sodre/ghq/github.com/zeroae/zae-limiter/.github/workflows/`

- [ ] **Step 1: Update ci-tests.yml**

Add stages: unit → integration → compatibility. Matrix: Python 3.11, 3.12. Use `uv` for setup.

- [ ] **Step 2: Create docs.yml**

Build + deploy docs with mike versioning on tag push. Follow zae-limiter pattern.

- [ ] **Step 3: Create release.yml**

Tag push → `uv build` → PyPI publish (OIDC) → GitHub Release with cliff changelog. Follow zae-limiter pattern.

- [ ] **Step 4: Commit**

```bash
git commit -m "ci: add docs and release workflows"
```

### Task 19: Full lifecycle integration test

**Files:**
- Create: `tests/integration/test_lifecycle.py`

- [ ] **Step 1: Write full lifecycle integration test via moto server**

End-to-end scenario:
1. Create workspace
2. Create experiment in workspace
3. Create runs with metrics, params, tags
4. Search runs with filters (status, metric order, tag filter, LIKE)
5. Register model, create versions, set aliases
6. Search models with filters
7. Soft-delete run → verify TTL set, RANK items preserved with TTL, excluded from active search
8. Restore run → verify TTL removed, back in active search
9. Soft-delete experiment → verify TTL on META only
10. Create traces with tags, assessments → verify TTL from trace_retention
11. Search traces by status, FTS keyword
12. Auth: create user, set experiment permission, verify access control
13. Cleanup: delete workspace cascade → verify all items removed

- [ ] **Step 2: Run integration test**
- [ ] **Step 3: Commit**

```bash
git commit -m "test: add full lifecycle integration test"
```

### Task 20: Release preparation

- [ ] **Step 1: Run full test suite with coverage**

```bash
uv run pytest tests/ -v --cov=mlflow_dynamodbstore --cov-report=html --cov-report=term-missing
```

Target: >90% coverage on `src/mlflow_dynamodbstore/`

- [ ] **Step 2: Run all linters**

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/mlflow_dynamodbstore/
```

- [ ] **Step 3: Build docs**

```bash
uv run mkdocs build --strict
```

- [ ] **Step 4: Build package**

```bash
uv build
```

- [ ] **Step 5: Test package install**

```bash
uv pip install dist/mlflow_dynamodbstore-*.whl
uv run python -c "import mlflow_dynamodbstore; print(mlflow_dynamodbstore.__version__)"
uv run mlflow-dynamodbstore --help
```

- [ ] **Step 6: Create git tag and push**

```bash
git tag v0.1.0
git push origin main --tags
```

- [ ] **Step 7: Verify CI release pipeline**

Watch GitHub Actions release workflow: build → PyPI publish → GitHub Release.

---

## Implementation Notes

### What's NOT in Plan 4

Deferred to future versions:
- **v2: DynamoDB Streams + Lambda** for async materialization (zae-mlflow CDK repo)
- **v2: OpenSearch Serverless** for FTS upgrade (replace trigram index)
- **v2: EventBridge-scheduled cleanup** (replaces manual `cleanup-expired` CLI)
- **Benchmarks** (pytest-benchmark, locust load tests)

### CLI testing pattern

All CLI commands tested via Click's `CliRunner` with `@mock_aws`:

```python
from click.testing import CliRunner
from moto import mock_aws
from mlflow_dynamodbstore.cli import cli
from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists

@mock_aws
def test_denormalize_tags_list():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    result = runner.invoke(cli, ["denormalize-tags", "list", "--table", "test", "--region", "us-east-1"])
    assert result.exit_code == 0
    assert "mlflow.*" in result.output
```

### Documentation standards

- Use google docstring style (matches mkdocstrings config from Plan 1)
- All public functions and classes must have docstrings
- Include type hints on all public APIs
- Use Mermaid for architecture diagrams in docs
- Reference the spec for design rationale
