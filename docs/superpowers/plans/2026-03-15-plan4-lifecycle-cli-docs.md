# Plan 4: Lifecycle, CLI & Documentation — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the plugin with TTL-based data lifecycle management, admin CLI commands, comprehensive documentation, and MLflow compatibility test gap closure. After this plan, `mlflow-dynamodbstore` is ready for v0.1.0 release.

**Architecture:** TTL attributes on DynamoDB items for automatic expiration. Background cleanup CLI for orphaned experiment children. Admin CLI for tag denormalization config, FTS trigram config, TTL policy, span pre-caching, workspace deletion. MkDocs documentation covering quickstart, configuration, architecture, and API reference.

**Tech Stack:** Same as Plans 1-3, plus `click` for CLI.

**Spec:** `docs/superpowers/specs/2026-03-15-mlflow-dynamodbstore-design.md`

**Depends on:** Plans 1 + 2 + 3 (foundation, CRUD, search, FTS, traces, X-Ray, auth)

---

## File Structure (new/modified files only)

```
src/mlflow_dynamodbstore/
├── tracking_store.py               # MODIFY: Add TTL to soft-delete/restore operations
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

### Task 1: TTL on soft-delete (delete_run + restore_run)

Modify `tracking_store.py`: `delete_run` sets TTL on run META + all children. `restore_run` removes TTL.

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_ttl.py
import time
from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable

class TestRunTTL:
    def test_delete_run_sets_ttl_on_meta(self, tracking_store, table):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.delete_run(run.info.run_id)
        item = table.get_item(f"EXP#{exp_id}", f"R#{run.info.run_id}")
        assert "ttl" in item
        assert item["ttl"] > time.time()  # TTL is in the future

    def test_delete_run_sets_ttl_on_children(self, tracking_store, table):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.log_batch(run.info.run_id, metrics=[...], params=[...], tags=[...])
        tracking_store.delete_run(run.info.run_id)
        # All child items should have TTL
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
        # RANK items should get TTL, NOT be immediately deleted
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.log_batch(run.info.run_id, metrics=[Metric("acc", 0.9, 0, 0)], params=[], tags=[])
        tracking_store.delete_run(run.info.run_id)
        # RANK item should still exist but with TTL
        rank_items = table.query(pk=f"EXP#{exp_id}", sk_prefix="RANK#m#acc#")
        assert len(rank_items) == 1
        assert "ttl" in rank_items[0]
```

- [ ] **Step 2: Modify delete_run to set TTL on all children instead of deleting RANK items**
- [ ] **Step 3: Modify restore_run to remove TTL from all children**
- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add TTL to delete_run/restore_run (no immediate RANK deletion)"
```

### Task 2: TTL on delete_experiment + restore_experiment

- [ ] **Step 1: Write failing tests**

Test that delete_experiment sets TTL on experiment META only (children untouched — too many to walk synchronously). Test restore_experiment removes TTL from META.

- [ ] **Step 2: Implement**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add TTL to delete_experiment/restore_experiment"
```

### Task 3: TTL on trace creation

Modify `start_trace`: set TTL on trace META from `CONFIG#TTL_POLICY.trace_retention_days`. Modify all trace child writes (tags, metadata, assessments) to inherit same TTL.

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Implement**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add TTL to trace creation from trace_retention_days config"
```

### Task 4: TTL on metric history

Modify `log_batch`: set TTL on metric history items from `CONFIG#TTL_POLICY.metric_history_retention_days`. Metric latest items do NOT get TTL.

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Implement**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add TTL to metric history items"
```

### Task 5: TTL policy config reader

- [ ] **Step 1: Write failing tests**

Test reading `CONFIG#TTL_POLICY` item, default values, `0` means disabled (no TTL set).

- [ ] **Step 2: Implement**

Cached reader for TTL policy. Used by delete_run, start_trace, log_batch to determine TTL values.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add TTL policy config reader"
```

---

## Chunk 2: Admin CLI

### Task 6: CLI framework

**Files:**
- Create: `src/mlflow_dynamodbstore/cli/__init__.py`
- Modify: `pyproject.toml` (add `click` dependency + CLI entry point)

- [ ] **Step 1: Set up Click group**

```python
# src/mlflow_dynamodbstore/cli/__init__.py
"""Admin CLI for mlflow-dynamodbstore."""
import click

@click.group()
def cli():
    """mlflow-dynamodbstore admin commands."""
    pass
```

```toml
# pyproject.toml addition
[project.scripts]
mlflow-dynamodbstore = "mlflow_dynamodbstore.cli:cli"
```

- [ ] **Step 2: Verify CLI is accessible**

```bash
uv run mlflow-dynamodbstore --help
```

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: add CLI framework with Click"
```

### Task 7: denormalize-tags CLI

**Files:**
- Create: `src/mlflow_dynamodbstore/cli/denormalize_tags.py`
- Create: `tests/unit/cli/test_denormalize_tags.py`

- [ ] **Step 1: Write failing tests**

Test `list`, `add`, `remove` subcommands modifying `CONFIG#DENORMALIZE_TAGS`. Test `add` with `--experiment-id` writing per-experiment config. Test `backfill` scanning tag items and updating META items. Test that `mlflow.*` cannot be removed.

- [ ] **Step 2: Implement denormalize-tags commands**

```python
@cli.group("denormalize-tags")
def denormalize_tags():
    """Manage tag denormalization patterns."""
    pass

@denormalize_tags.command("list")
@click.option("--table", required=True)
@click.option("--region", default="us-east-1")
@click.option("--experiment-id")
def list_patterns(table, region, experiment_id): ...

@denormalize_tags.command("add")
@click.argument("patterns", nargs=-1)
@click.option("--table", required=True)
@click.option("--region", default="us-east-1")
@click.option("--experiment-id")
def add_patterns(patterns, table, region, experiment_id): ...

@denormalize_tags.command("remove")
@click.argument("patterns", nargs=-1)
@click.option("--table", required=True)
@click.option("--region", default="us-east-1")
def remove_patterns(patterns, table, region): ...

@denormalize_tags.command("backfill")
@click.option("--table", required=True)
@click.option("--region", default="us-east-1")
@click.option("--experiment-id")
def backfill(table, region, experiment_id): ...
```

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add denormalize-tags CLI commands"
```

### Task 8: fts-trigrams CLI

- [ ] **Step 1: Write failing tests for list/add/backfill**
- [ ] **Step 2: Implement fts-trigrams commands (same pattern as denormalize-tags)**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add fts-trigrams CLI commands"
```

### Task 9: ttl-policy CLI

- [ ] **Step 1: Write failing tests for show/set**
- [ ] **Step 2: Implement**

```python
@cli.group("ttl-policy")
def ttl_policy(): ...

@ttl_policy.command("show")
def show(table, region): ...

@ttl_policy.command("set")
@click.option("--soft-deleted-retention-days", type=int)
@click.option("--trace-retention-days", type=int)
@click.option("--metric-history-retention-days", type=int)
def set_policy(table, region, **kwargs): ...
```

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add ttl-policy CLI commands"
```

### Task 10: cleanup-expired CLI

- [ ] **Step 1: Write failing tests**

Test: create experiment, soft-delete it, simulate TTL expiry of META (delete META item manually), run cleanup-expired, verify children get TTL or are deleted. Test `--dry-run` mode.

- [ ] **Step 2: Implement**

Scan for experiment partitions where `E#META` has been TTL-deleted. For each, set `ttl = now` on all remaining items. Report progress.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add cleanup-expired CLI command"
```

### Task 11: cache-spans CLI

- [ ] **Step 1: Write failing tests**

Test: create traces, mock X-Ray to return spans, run cache-spans, verify SPANS items + span attribute denormalization + FTS items. Test `--experiment-id` and `--days` options. Test idempotent (re-run doesn't duplicate).

Note: X-Ray calls mocked since moto doesn't support X-Ray.

- [ ] **Step 2: Implement**

Iterate traces in experiment(s), check for missing `T#<trace_id>#SPANS` items, call `get_trace` for each (which triggers caching + indexing). Report progress.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add cache-spans CLI command"
```

### Task 12: delete-workspace CLI

- [ ] **Step 1: Write failing tests**

Test soft-delete mode (workspace META status='deleted'). Test cascade mode (all experiments + models deleted). Test default workspace cannot be deleted. Test progress reporting.

- [ ] **Step 2: Implement**

```python
@cli.command("delete-workspace")
@click.argument("workspace_name")
@click.option("--mode", type=click.Choice(["soft", "cascade"]), default="soft")
@click.option("--table", required=True)
@click.option("--region", default="us-east-1")
@click.option("--yes", is_flag=True, help="Skip confirmation")
def delete_workspace(workspace_name, mode, table, region, yes): ...
```

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

Full getting started: install, configure, run mlflow server, create experiment, log run, view in UI.

- [ ] **Step 2: Write configuration guide**

URI format, all environment variables, CONFIG items (denormalize patterns, TTL policy, FTS trigrams, annotation mapping).

- [ ] **Step 3: Write workspace guide**

Enable workspaces, create workspaces, scope experiments/models, workspace permissions.

- [ ] **Step 4: Write X-Ray integration guide**

OTel dual-export setup, annotation mapping, span search, pre-caching spans.

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

Retention policies, soft-delete flow, automatic expiration, background cleanup, data recovery window.

- [ ] **Step 3: Write monitoring guide**

CloudWatch metrics to watch, partition size monitoring, DynamoDB capacity, X-Ray trace count.

- [ ] **Step 4: Write upgrade guide**

v2 path: DynamoDB Streams + Lambda for async materialization, OpenSearch Serverless for FTS upgrade, CDK deployment.

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

Complete DynamoDB table schema: all entity types, PK/SK patterns, LSI/GSI attributes. Based on the spec tables.

- [ ] **Step 2: Write access pattern reference**

All 45+ access patterns with mechanism descriptions. Based on the spec's Access Pattern Coverage section.

- [ ] **Step 3: Expand API reference**

Ensure mkdocstrings generates docs for all public modules: tracking_store, registry_store, workspace_store, auth.store, xray.client, dynamodb.table.

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

Dev setup (uv, pre-commit), testing (markers, moto), PR workflow, commit conventions.

- [ ] **Step 2: Write ADR for single-table design**

Decision, context, alternatives considered (multi-table, hybrid), consequences.

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

Common failure categories and fixes:
- Missing method implementations → implement stubs or raise appropriate MlflowException
- Return type mismatches → ensure entities match MLflow's expected types
- Edge cases in search behavior → adjust query planner
- Default experiment handling → verify experiment "0" works

- [ ] **Step 3: Re-run compatibility tests**

Iterate until all applicable tests pass. Document any intentionally skipped tests (e.g., Databricks-specific, gateway-specific) with `@pytest.mark.skip(reason="...")`.

- [ ] **Step 4: Commit**

```bash
git commit -m "test: close MLflow compatibility test gaps"
```

### Task 18: CI workflow updates

**Files:**
- Modify: `.github/workflows/ci-tests.yml`
- Create: `.github/workflows/docs.yml`
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Update ci-tests.yml**

Add stages: unit → integration → compatibility. Matrix: Python 3.11, 3.12.

- [ ] **Step 2: Create docs.yml**

Build + deploy docs with mike versioning on tag push.

- [ ] **Step 3: Create release.yml**

Tag push → uv build → PyPI publish (OIDC) → GitHub Release with cliff changelog.

- [ ] **Step 4: Commit**

```bash
git commit -m "ci: add docs and release workflows"
```

### Task 19: Final integration test

**Files:**
- Create: `tests/integration/test_lifecycle.py`

- [ ] **Step 1: Write full lifecycle integration test**

End-to-end scenario via moto server:
1. Create workspace
2. Create experiment in workspace
3. Create runs with metrics, params, tags
4. Search runs with filters
5. Register model, create versions
6. Set aliases, resolve by alias
7. Soft-delete run → verify TTL set, RANK items preserved, excluded from search
8. Restore run → verify TTL removed, RANK items still exist
9. Soft-delete experiment → verify TTL on META
10. Search traces with span filters (mock X-Ray)
11. Auth: create user, set permissions, verify access control

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

Target: >90% coverage on src/mlflow_dynamodbstore/

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

All CLI commands are tested via Click's `CliRunner`:

```python
from click.testing import CliRunner
from mlflow_dynamodbstore.cli import cli

def test_denormalize_tags_list(mock_dynamodb):
    runner = CliRunner()
    result = runner.invoke(cli, ["denormalize-tags", "list", "--table", "test", "--region", "us-east-1"])
    assert result.exit_code == 0
    assert "mlflow.*" in result.output
```

### Documentation standards

- Use google docstring style (matches mkdocstrings config)
- All public functions and classes must have docstrings
- Include type hints on all public APIs
- Use Mermaid for architecture diagrams in docs
- Reference the spec for design rationale
