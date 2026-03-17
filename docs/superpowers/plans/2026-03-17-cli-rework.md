# CLI Rework Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the mlflow-dynamodbstore CLI to use noun-verb groups, add deploy/destroy lifecycle commands, simplify stack naming, and auto-generate docs with mkdocs-click.

**Architecture:** Global `--name`/`--region`/`--endpoint-url` options on the top-level cloup group, passed via `click.Context.obj`. Commands organized into "Stack Lifecycle" and "Configuration" sections. Existing commands renamed and reorganized into noun groups (`tag`, `ttl`, `fts`, `trace`, `workspace`). New `deploy`/`destroy` commands wrap provisioner functions. URI parser extended with `deploy` flag and sensible defaults.

**Tech Stack:** cloup 3+, boto3, moto (testing), mkdocs-click

**Status:** COMPLETED — all 14 tasks implemented plus post-plan refinements.

**Spec:** `docs/superpowers/specs/2026-03-17-cli-rework-design.md`

---

### Task 1: Update URI parser — add `deploy` field

**Files:**
- Modify: `src/mlflow_dynamodbstore/dynamodb/uri.py`
- Modify: `tests/unit/dynamodb/test_uri.py`

- [ ] **Step 1: Write failing tests for `deploy` query param parsing**

Add to `tests/unit/dynamodb/test_uri.py`:

```python
def test_deploy_default_true(self):
    result = parse_dynamodb_uri("dynamodb://us-east-1/my-table")
    assert result.deploy is True

def test_deploy_explicit_true(self):
    result = parse_dynamodb_uri("dynamodb://us-east-1/my-table?deploy=true")
    assert result.deploy is True

def test_deploy_false(self):
    result = parse_dynamodb_uri("dynamodb://us-east-1/my-table?deploy=false")
    assert result.deploy is False

def test_deploy_false_with_localhost(self):
    result = parse_dynamodb_uri("dynamodb://localhost:5000/test-table?deploy=false")
    assert result.deploy is False
    assert result.endpoint_url == "http://localhost:5000"
    assert result.table_name == "test-table"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/dynamodb/test_uri.py -v`
Expected: FAIL — `DynamoDBUriComponents` has no `deploy` attribute

- [ ] **Step 3: Add `deploy` field to `DynamoDBUriComponents` and parse query params**

In `src/mlflow_dynamodbstore/dynamodb/uri.py`:

1. Add `deploy: bool = True` to the `DynamoDBUriComponents` dataclass
2. Before parsing the host/table, strip query params from the URI using `urllib.parse.urlparse` or simple string splitting
3. Parse `?deploy=true|false` and pass to the dataclass constructor

The key change: split `table_name` on `?` to extract query string, then parse `deploy` param. Apply to all three return paths (explicit http endpoint, region, host:port).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/dynamodb/test_uri.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/dynamodb/uri.py tests/unit/dynamodb/test_uri.py
git commit -m "feat: add deploy query param to DynamoDB URI parser"
```

---

### Task 2: Update provisioner — remove stack prefix, add destroy

**Files:**
- Modify: `src/mlflow_dynamodbstore/dynamodb/provisioner.py`
- Modify: `tests/unit/dynamodb/test_provisioner.py`

- [ ] **Step 1: Write failing tests for new stack naming and destroy**

Replace/update tests in `tests/unit/dynamodb/test_provisioner.py`:

```python
import boto3
import pytest
from moto import mock_aws

from mlflow_dynamodbstore.dynamodb.provisioner import (
    destroy_stack,
    ensure_stack_exists,
)


class TestProvisioner:
    @mock_aws
    def test_stack_name_equals_table_name(self):
        """Stack name should be the table name directly, no prefix."""
        ensure_stack_exists(table_name="my-table", region="us-east-1")
        cfn = boto3.client("cloudformation", region_name="us-east-1")
        stacks = cfn.list_stacks(StackStatusFilter=["CREATE_COMPLETE"])["StackSummaries"]
        stack_names = [s["StackName"] for s in stacks]
        assert "my-table" in stack_names

    @mock_aws
    def test_creates_stack_if_not_exists(self):
        ensure_stack_exists(table_name="test-table", region="us-east-1")
        cfn = boto3.client("cloudformation", region_name="us-east-1")
        stacks = cfn.list_stacks(StackStatusFilter=["CREATE_COMPLETE"])["StackSummaries"]
        stack_names = [s["StackName"] for s in stacks]
        assert "test-table" in stack_names

    @mock_aws
    def test_table_created_by_stack(self):
        ensure_stack_exists(table_name="test-table", region="us-east-1")
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        tables = ddb.list_tables()["TableNames"]
        assert "test-table" in tables

    @mock_aws
    def test_table_has_5_lsis(self):
        ensure_stack_exists(table_name="test-table", region="us-east-1")
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        desc = ddb.describe_table(TableName="test-table")["Table"]
        assert len(desc.get("LocalSecondaryIndexes", [])) == 5

    @mock_aws
    def test_table_has_5_gsis(self):
        ensure_stack_exists(table_name="test-table", region="us-east-1")
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        desc = ddb.describe_table(TableName="test-table")["Table"]
        assert len(desc.get("GlobalSecondaryIndexes", [])) == 5

    @mock_aws
    def test_idempotent_if_stack_exists(self):
        ensure_stack_exists(table_name="test-table", region="us-east-1")
        ensure_stack_exists(table_name="test-table", region="us-east-1")

    @mock_aws
    def test_creates_default_workspace(self):
        ensure_stack_exists(table_name="test-table", region="us-east-1")
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        result = ddb.get_item(
            TableName="test-table",
            Key={"PK": {"S": "WORKSPACE#default"}, "SK": {"S": "META"}},
        )
        assert "Item" in result

    @mock_aws
    def test_creates_default_experiment(self):
        ensure_stack_exists(table_name="test-table", region="us-east-1")
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        result = ddb.get_item(
            TableName="test-table",
            Key={"PK": {"S": "EXP#0"}, "SK": {"S": "E#META"}},
        )
        assert "Item" in result
        assert result["Item"]["name"]["S"] == "Default"


class TestDestroyStack:
    @mock_aws
    def test_destroy_removes_stack_and_table(self):
        ensure_stack_exists(table_name="test-table", region="us-east-1")
        destroy_stack(table_name="test-table", region="us-east-1")
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        tables = ddb.list_tables()["TableNames"]
        assert "test-table" not in tables

    @mock_aws
    def test_destroy_retain_keeps_table(self):
        ensure_stack_exists(table_name="test-table", region="us-east-1")
        destroy_stack(table_name="test-table", region="us-east-1", retain=True)
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        tables = ddb.list_tables()["TableNames"]
        assert "test-table" in tables

    @mock_aws
    def test_destroy_nonexistent_raises(self):
        with pytest.raises(Exception):
            destroy_stack(table_name="nope", region="us-east-1")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/dynamodb/test_provisioner.py -v`
Expected: FAIL — `destroy_stack` doesn't exist, stack naming tests fail

- [ ] **Step 3: Update provisioner implementation**

In `src/mlflow_dynamodbstore/dynamodb/provisioner.py`:

1. Remove `_STACK_PREFIX` and `get_stack_name()` — use `table_name` directly as the stack name
2. Update `ensure_stack_exists()` to use `table_name` as stack name
3. Add `destroy_stack(table_name, region, endpoint_url, retain)` function:
   - If `retain=True`: `cfn.delete_stack(StackName=table_name, RetainResources=['MlflowTable'])`
   - If `retain=False`: `cfn.delete_stack(StackName=table_name)`
   - Wait for `stack_delete_complete`

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/dynamodb/test_provisioner.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/dynamodb/provisioner.py tests/unit/dynamodb/test_provisioner.py
git commit -m "feat: simplify stack naming, add destroy_stack function"
```

---

### Task 3: Update stores — conditional deploy based on URI param

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py` (line 222)
- Modify: `src/mlflow_dynamodbstore/registry_store.py` (find `ensure_stack_exists` call)
- Modify: `src/mlflow_dynamodbstore/workspace_store.py` (line 39)
- Modify: `src/mlflow_dynamodbstore/auth/store.py` (find `ensure_stack_exists` call)

- [ ] **Step 1: Update all four stores with conditional deploy**

In each store's `__init__`, replace the bare `ensure_stack_exists(...)` call with:

```python
uri = parse_dynamodb_uri(store_uri)
if uri.deploy:
    ensure_stack_exists(uri.table_name, uri.region, uri.endpoint_url)
```

Apply to: `tracking_store.py`, `registry_store.py`, `workspace_store.py`, `auth/store.py`.

- [ ] **Step 2: Write test for `deploy=False` behavior**

Add a test (e.g., in `tests/unit/dynamodb/test_provisioner.py` or a new `tests/unit/test_store_deploy.py`):

```python
import pytest
from moto import mock_aws

from mlflow_dynamodbstore.dynamodb.uri import parse_dynamodb_uri


class TestDeployFlag:
    def test_deploy_false_parsed_from_uri(self):
        uri = parse_dynamodb_uri("dynamodb://us-east-1/my-table?deploy=false")
        assert uri.deploy is False

    @mock_aws
    def test_workspace_store_deploy_false_skips_provisioning(self):
        """With deploy=false, store init should not create a CFn stack."""
        from mlflow_dynamodbstore.workspace_store import DynamoDBWorkspaceStore

        with pytest.raises(Exception):
            # Table doesn't exist and deploy=false, so this should fail
            DynamoDBWorkspaceStore(
                workspace_uri="dynamodb://us-east-1/nonexistent?deploy=false"
            )
```

- [ ] **Step 3: Run tests to verify**

Run: `uv run pytest tests/unit/ -v --timeout=60`
Expected: All PASS (existing tests use default `deploy=True` behavior, new test validates `deploy=False`)

- [ ] **Step 4: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py src/mlflow_dynamodbstore/registry_store.py src/mlflow_dynamodbstore/workspace_store.py src/mlflow_dynamodbstore/auth/store.py tests/
git commit -m "feat: conditional auto-deploy based on URI deploy param"
```

---

### Task 4: Create `CliContext` in `cli/__init__.py` (phase 1)

**Files:**
- Modify: `src/mlflow_dynamodbstore/cli/__init__.py`

This task adds `CliContext` and `pass_context` to the CLI module without changing the group or imports yet. This allows new command modules (Tasks 5-6) and renamed modules (Tasks 7-11) to import `CliContext` and `pass_context` while the old commands still work.

- [ ] **Step 1: Add `CliContext` and `pass_context` to `cli/__init__.py`**

Add at the top of the file (after existing imports):

```python
class CliContext:
    """Shared context for all CLI commands."""

    def __init__(self, name: str, region: str, endpoint_url: str | None) -> None:
        self.name = name
        self.region = region
        self.endpoint_url = endpoint_url


pass_context = click.make_pass_decorator(CliContext)
```

Do NOT change the `cli` group or command registrations yet.

- [ ] **Step 2: Commit**

```bash
git add src/mlflow_dynamodbstore/cli/__init__.py
git commit -m "refactor: add CliContext and pass_context to CLI module"
```

---

### Task 5: Create `deploy` command

**Files:**
- Create: `src/mlflow_dynamodbstore/cli/deploy.py`
- Create: `tests/unit/cli/test_deploy.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/cli/test_deploy.py`. Since the full CLI entry point isn't rewritten yet, test the command directly:

```python
import boto3
import click
from click.testing import CliRunner
from moto import mock_aws

from mlflow_dynamodbstore.cli import CliContext


@click.group()
@click.pass_context
def _test_cli(ctx):
    ctx.obj = CliContext(name="test-table", region="us-east-1", endpoint_url=None)


from mlflow_dynamodbstore.cli.deploy import deploy  # noqa: E402

_test_cli.add_command(deploy)


class TestDeploy:
    @mock_aws
    def test_deploy_creates_stack(self):
        runner = CliRunner()
        result = runner.invoke(_test_cli, ["deploy"])
        assert result.exit_code == 0
        cfn = boto3.client("cloudformation", region_name="us-east-1")
        stacks = cfn.list_stacks(StackStatusFilter=["CREATE_COMPLETE"])["StackSummaries"]
        assert any(s["StackName"] == "test-table" for s in stacks)

    @mock_aws
    def test_deploy_idempotent(self):
        runner = CliRunner()
        runner.invoke(_test_cli, ["deploy"])
        result = runner.invoke(_test_cli, ["deploy"])
        assert result.exit_code == 0

    @mock_aws
    def test_deploy_seeds_default_data(self):
        runner = CliRunner()
        runner.invoke(_test_cli, ["deploy"])
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        result = ddb.get_item(
            TableName="test-table",
            Key={"PK": {"S": "WORKSPACE#default"}, "SK": {"S": "META"}},
        )
        assert "Item" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/cli/test_deploy.py -v`
Expected: FAIL — module `mlflow_dynamodbstore.cli.deploy` not found

- [ ] **Step 3: Create `deploy.py`**

Create `src/mlflow_dynamodbstore/cli/deploy.py`:

```python
"""deploy CLI command."""

from __future__ import annotations

import click

from mlflow_dynamodbstore.cli import CliContext, pass_context
from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists


@click.command()
@pass_context
def deploy(ctx: CliContext) -> None:
    """Create the CloudFormation stack and seed initial data."""
    ensure_stack_exists(ctx.name, ctx.region, ctx.endpoint_url)
    click.echo(f"Stack '{ctx.name}' deployed successfully.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/cli/test_deploy.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/cli/deploy.py tests/unit/cli/test_deploy.py
git commit -m "feat: add deploy CLI command"
```

---

### Task 6: Create `destroy` command

**Files:**
- Create: `src/mlflow_dynamodbstore/cli/destroy.py`
- Create: `tests/unit/cli/test_destroy.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/cli/test_destroy.py`. Test the command directly (same pattern as deploy):

```python
import boto3
import click
from click.testing import CliRunner
from moto import mock_aws

from mlflow_dynamodbstore.cli import CliContext
from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists


@click.group()
@click.pass_context
def _test_cli(ctx):
    ctx.obj = CliContext(name="test-table", region="us-east-1", endpoint_url=None)


from mlflow_dynamodbstore.cli.destroy import destroy  # noqa: E402

_test_cli.add_command(destroy)


class TestDestroy:
    @mock_aws
    def test_destroy_with_yes(self):
        ensure_stack_exists("test-table", "us-east-1")
        runner = CliRunner()
        result = runner.invoke(_test_cli, ["destroy", "--yes"])
        assert result.exit_code == 0
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        tables = ddb.list_tables()["TableNames"]
        assert "test-table" not in tables

    @mock_aws
    def test_destroy_prompts_confirmation(self):
        ensure_stack_exists("test-table", "us-east-1")
        runner = CliRunner()
        result = runner.invoke(_test_cli, ["destroy"], input="y\n")
        assert result.exit_code == 0

    @mock_aws
    def test_destroy_aborts_on_no(self):
        ensure_stack_exists("test-table", "us-east-1")
        runner = CliRunner()
        result = runner.invoke(_test_cli, ["destroy"], input="n\n")
        assert result.exit_code != 0

    @mock_aws
    def test_destroy_retain_keeps_table(self):
        ensure_stack_exists("test-table", "us-east-1")
        runner = CliRunner()
        result = runner.invoke(_test_cli, ["destroy", "--yes", "--retain"])
        assert result.exit_code == 0
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        tables = ddb.list_tables()["TableNames"]
        assert "test-table" in tables
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/cli/test_destroy.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Create `destroy.py`**

Create `src/mlflow_dynamodbstore/cli/destroy.py`:

```python
"""destroy CLI command."""

from __future__ import annotations

import click

from mlflow_dynamodbstore.cli import CliContext, pass_context
from mlflow_dynamodbstore.dynamodb.provisioner import destroy_stack


@click.command()
@click.option("--yes", "confirmed", is_flag=True, help="Skip confirmation prompt")
@click.option("--retain", is_flag=True, help="Delete stack but keep the DynamoDB table")
@pass_context
def destroy(ctx: CliContext, confirmed: bool, retain: bool) -> None:
    """Delete the CloudFormation stack."""
    if not confirmed:
        click.confirm(
            f"Destroy stack '{ctx.name}'? This will delete the CloudFormation stack"
            + (" (table will be retained)" if retain else " and the DynamoDB table"),
            abort=True,
        )
    destroy_stack(ctx.name, ctx.region, ctx.endpoint_url, retain=retain)
    click.echo(f"Stack '{ctx.name}' destroyed.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/cli/test_destroy.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/cli/destroy.py tests/unit/cli/test_destroy.py
git commit -m "feat: add destroy CLI command"
```

---

### Task 7: Rename `denormalize_tags` → `tag`

**Files:**
- Rename: `src/mlflow_dynamodbstore/cli/denormalize_tags.py` → `src/mlflow_dynamodbstore/cli/tag.py`
- Rename: `tests/unit/cli/test_denormalize_tags.py` → `tests/unit/cli/test_tag.py`

- [ ] **Step 1: Rename files**

```bash
git mv src/mlflow_dynamodbstore/cli/denormalize_tags.py src/mlflow_dynamodbstore/cli/tag.py
git mv tests/unit/cli/test_denormalize_tags.py tests/unit/cli/test_tag.py
```

- [ ] **Step 2: Update `tag.py`**

In `src/mlflow_dynamodbstore/cli/tag.py`:

1. Change group name: `@click.group("denormalize-tags")` → `@click.group("tag")`
2. Rename group function: `def denormalize_tags()` → `def tag()`
3. Remove per-command `--table`/`--region` options — use `pass_context` instead
4. Rename subcommand functions: `list_patterns` → `list_`, `add_pattern` → `add`, `remove_pattern` → `remove` (keep `backfill`)
5. Replace `@denormalize_tags.command(...)` → `@tag.command(...)`
6. Each subcommand receives `ctx: CliContext` via `@pass_context` and constructs `DynamoDBTable(ctx.name, ctx.region, ctx.endpoint_url)`
7. Fix `backfill` command: the raw `boto3.resource("dynamodb", region_name=region)` call (line 96) must also pass `endpoint_url` from context, otherwise backfill will break with `--endpoint-url`

- [ ] **Step 3: Update `test_tag.py`**

Update all test invocations from:
```python
runner.invoke(cli, ["denormalize-tags", "list", "--table", "t", "--region", "us-east-1"])
```
to:
```python
runner.invoke(cli, ["--name", "t", "--region", "us-east-1", "tag", "list"])
```

Update imports from `denormalize_tags` to `tag`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/cli/test_tag.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add -A src/mlflow_dynamodbstore/cli/tag.py tests/unit/cli/test_tag.py
git commit -m "refactor: rename denormalize-tags CLI group to tag"
```

---

### Task 8: Rename `ttl_policy` → `ttl`, absorb `cleanup_expired`

**Files:**
- Rename: `src/mlflow_dynamodbstore/cli/ttl_policy.py` → `src/mlflow_dynamodbstore/cli/ttl.py`
- Delete: `src/mlflow_dynamodbstore/cli/cleanup_expired.py`
- Rename: `tests/unit/cli/test_ttl_policy.py` → `tests/unit/cli/test_ttl.py`
- Delete: `tests/unit/cli/test_cleanup_expired.py` (merge into `test_ttl.py`)

- [ ] **Step 1: Rename and merge files**

```bash
git mv src/mlflow_dynamodbstore/cli/ttl_policy.py src/mlflow_dynamodbstore/cli/ttl.py
git mv tests/unit/cli/test_ttl_policy.py tests/unit/cli/test_ttl.py
```

- [ ] **Step 2: Update `ttl.py`**

1. Change group name: `@click.group("ttl-policy")` → `@click.group("ttl")`
2. Rename group function: `def ttl_policy()` → `def ttl()`
3. Remove per-command `--table`/`--region` — use `pass_context`
4. Rename: `show_policy` → `show`, `set_policy` → `set_`
5. Move `cleanup_expired` logic into this file as `@ttl.command("cleanup")`
6. Replace `@ttl_policy.command(...)` → `@ttl.command(...)`

- [ ] **Step 3: Update `test_ttl.py` — merge cleanup tests**

1. Update invocations to use global options
2. Merge tests from `test_cleanup_expired.py` into this file, updating invocations from `cleanup-expired` to `ttl cleanup`

- [ ] **Step 4: Delete old files**

```bash
git rm src/mlflow_dynamodbstore/cli/cleanup_expired.py
git rm tests/unit/cli/test_cleanup_expired.py
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/cli/test_ttl.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: rename ttl-policy to ttl, absorb cleanup-expired as ttl cleanup"
```

---

### Task 9: Rename `fts_trigrams` → `fts`

**Files:**
- Rename: `src/mlflow_dynamodbstore/cli/fts_trigrams.py` → `src/mlflow_dynamodbstore/cli/fts.py`
- Rename: `tests/unit/cli/test_fts_trigrams.py` → `tests/unit/cli/test_fts.py`

- [ ] **Step 1: Rename files**

```bash
git mv src/mlflow_dynamodbstore/cli/fts_trigrams.py src/mlflow_dynamodbstore/cli/fts.py
git mv tests/unit/cli/test_fts_trigrams.py tests/unit/cli/test_fts.py
```

- [ ] **Step 2: Update `fts.py`**

1. Change group name: `@click.group("fts-trigrams")` → `@click.group("fts")`
2. Rename group function: `def fts_trigrams()` → `def fts()`
3. Remove per-command `--table`/`--region` — use `pass_context`
4. Rename: `list_fields` → `list_`, `add_field` → `add`
5. Replace `@fts_trigrams.command(...)` → `@fts.command(...)`

- [ ] **Step 3: Update `test_fts.py`**

Update invocations to use global options format.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/cli/test_fts.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add -A src/mlflow_dynamodbstore/cli/fts.py tests/unit/cli/test_fts.py
git commit -m "refactor: rename fts-trigrams CLI group to fts"
```

---

### Task 10: Rename `cache_spans` → `trace`

**Files:**
- Rename: `src/mlflow_dynamodbstore/cli/cache_spans.py` → `src/mlflow_dynamodbstore/cli/trace.py`
- Rename: `tests/unit/cli/test_cache_spans.py` → `tests/unit/cli/test_trace.py`

- [ ] **Step 1: Rename files**

```bash
git mv src/mlflow_dynamodbstore/cli/cache_spans.py src/mlflow_dynamodbstore/cli/trace.py
git mv tests/unit/cli/test_cache_spans.py tests/unit/cli/test_trace.py
```

- [ ] **Step 2: Update `trace.py`**

1. Convert from `@click.command("cache-spans")` to a group:
   ```python
   @click.group("trace")
   def trace() -> None:
       """Trace operations."""
       pass

   @trace.command("cache")
   @pass_context
   def cache(ctx: CliContext, ...) -> None:
       ...
   ```
2. Remove per-command `--table`/`--region` — use `pass_context`

- [ ] **Step 3: Update `test_trace.py`**

Update invocations from `cache-spans` to `trace cache` with global options.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/cli/test_trace.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add -A src/mlflow_dynamodbstore/cli/trace.py tests/unit/cli/test_trace.py
git commit -m "refactor: rename cache-spans to trace cache"
```

---

### Task 11: Rename `delete_workspace` → `workspace`

**Files:**
- Rename: `src/mlflow_dynamodbstore/cli/delete_workspace.py` → `src/mlflow_dynamodbstore/cli/workspace.py`
- Rename: `tests/unit/cli/test_delete_workspace.py` → `tests/unit/cli/test_workspace.py`

- [ ] **Step 1: Rename files**

```bash
git mv src/mlflow_dynamodbstore/cli/delete_workspace.py src/mlflow_dynamodbstore/cli/workspace.py
git mv tests/unit/cli/test_delete_workspace.py tests/unit/cli/test_workspace.py
```

- [ ] **Step 2: Update `workspace.py`**

1. Convert from `@click.command("delete-workspace")` to a group:
   ```python
   @click.group("workspace")
   def workspace() -> None:
       """Workspace operations."""
       pass

   @workspace.command("delete")
   @pass_context
   def delete(ctx: CliContext, ...) -> None:
       ...
   ```
2. Remove per-command `--table`/`--region` — use `pass_context`

- [ ] **Step 3: Update `test_workspace.py`**

Update invocations from `delete-workspace` to `workspace delete` with global options.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/cli/test_workspace.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add -A src/mlflow_dynamodbstore/cli/workspace.py tests/unit/cli/test_workspace.py
git commit -m "refactor: rename delete-workspace to workspace delete"
```

---

### Task 12: Rewrite CLI entry point (phase 2) and full validation

**Files:**
- Rewrite: `src/mlflow_dynamodbstore/cli/__init__.py`

All command modules now exist. Rewrite the entry point to use global options and register the new commands.

- [ ] **Step 1: Rewrite `cli/__init__.py`**

```python
"""Admin CLI for mlflow-dynamodbstore."""

from __future__ import annotations

import click


class CliContext:
    """Shared context for all CLI commands."""

    def __init__(self, name: str, region: str, endpoint_url: str | None) -> None:
        self.name = name
        self.region = region
        self.endpoint_url = endpoint_url


pass_context = click.make_pass_decorator(CliContext)


@click.group()
@click.option("--name", required=True, help="Stack/table name")
@click.option("--region", required=True, help="AWS region (e.g. us-east-1)")
@click.option("--endpoint-url", default=None, help="Custom endpoint URL (for LocalStack/testing)")
@click.pass_context
def cli(ctx: click.Context, name: str, region: str, endpoint_url: str | None) -> None:
    """mlflow-dynamodbstore admin commands."""
    ctx.ensure_object(dict)
    ctx.obj = CliContext(name=name, region=region, endpoint_url=endpoint_url)


from mlflow_dynamodbstore.cli.deploy import deploy  # noqa: E402
from mlflow_dynamodbstore.cli.destroy import destroy  # noqa: E402
from mlflow_dynamodbstore.cli.fts import fts  # noqa: E402
from mlflow_dynamodbstore.cli.tag import tag  # noqa: E402
from mlflow_dynamodbstore.cli.trace import trace  # noqa: E402
from mlflow_dynamodbstore.cli.ttl import ttl  # noqa: E402
from mlflow_dynamodbstore.cli.workspace import workspace  # noqa: E402

cli.add_command(deploy)
cli.add_command(destroy)
cli.add_command(fts)
cli.add_command(tag)
cli.add_command(trace)
cli.add_command(ttl)
cli.add_command(workspace)
```

- [ ] **Step 2: Update tests to use the real CLI group**

Update `test_deploy.py` and `test_destroy.py` to import `from mlflow_dynamodbstore.cli import cli` and use global `--name`/`--region` options instead of the `_test_cli` helper group.

- [ ] **Step 3: Test that `cli --help` works**

Run: `uv run mlflow-dynamodbstore --help`
Expected: Shows `--name`, `--region`, `--endpoint-url` and all subcommands (deploy, destroy, fts, tag, trace, ttl, workspace)

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/unit/ -v --timeout=60`
Expected: All PASS

- [ ] **Step 5: Run linting**

Run: `uv run ruff check src/mlflow_dynamodbstore/cli/`
Expected: Clean

- [ ] **Step 6: Run type checking**

Run: `uv run mypy src/mlflow_dynamodbstore/cli/`
Expected: Clean (or pre-existing issues only)

- [ ] **Step 7: Verify pyproject.toml entry point**

Confirm `pyproject.toml` still has:
```toml
[project.scripts]
mlflow-dynamodbstore = "mlflow_dynamodbstore.cli:cli"
```

- [ ] **Step 8: Commit**

```bash
git add src/mlflow_dynamodbstore/cli/__init__.py tests/unit/cli/test_deploy.py tests/unit/cli/test_destroy.py
git commit -m "refactor: rewrite CLI entry point with global --name/--region/--endpoint-url"
```

---

### Task 13: Rewrite CLI reference documentation

**Files:**
- Rewrite: `docs/operator-guide/cli-reference.md`

- [ ] **Step 1: Rewrite `cli-reference.md`**

Replace the entire file with the new structure using a single `mkdocs-click` directive (matching zae-limiter pattern):

```markdown
# CLI Reference

The `mlflow-dynamodbstore` CLI provides admin commands for managing your
DynamoDB-backed MLflow deployment.

```bash
mlflow-dynamodbstore --help
```

## Global Options

Every subcommand inherits:

| Option           | Description                          | Required |
|------------------|--------------------------------------|----------|
| `--name`         | Stack/table name                     | Yes      |
| `--region`       | AWS region (e.g. `us-east-1`)        | Yes      |
| `--endpoint-url` | Custom endpoint (LocalStack/testing) | No       |

---

## Commands

::: mkdocs-click
    :module: mlflow_dynamodbstore.cli
    :command: cli
    :prog_name: mlflow-dynamodbstore
    :depth: 2
    :style: table
    :list_subcommands: true

---

## Examples

### Deploy a new stack

```bash
mlflow-dynamodbstore --name mlflow --region us-east-1 deploy
```

### Destroy a stack (retain table)

```bash
mlflow-dynamodbstore --name mlflow --region us-east-1 destroy --yes --retain
```

### Manage TTL policies

```bash
# Show current policy
mlflow-dynamodbstore --name mlflow --region us-east-1 ttl show

# Set retention
mlflow-dynamodbstore --name mlflow --region us-east-1 ttl set \
    --soft-deleted-retention-days 90 \
    --trace-retention-days 30

# Cleanup orphaned items
mlflow-dynamodbstore --name mlflow --region us-east-1 ttl cleanup --dry-run
```

### Manage tag denormalization

```bash
mlflow-dynamodbstore --name mlflow --region us-east-1 tag list
mlflow-dynamodbstore --name mlflow --region us-east-1 tag add "mlflow.user"
mlflow-dynamodbstore --name mlflow --region us-east-1 tag backfill
```

### Full-text search configuration

```bash
mlflow-dynamodbstore --name mlflow --region us-east-1 fts list
mlflow-dynamodbstore --name mlflow --region us-east-1 fts add "description"
```

### Cache trace spans

```bash
mlflow-dynamodbstore --name mlflow --region us-east-1 trace cache \
    --experiment-id 1 --days 7
```

### Delete a workspace

```bash
# Soft delete
mlflow-dynamodbstore --name mlflow --region us-east-1 workspace delete \
    --workspace staging --mode soft

# Cascade (with confirmation)
mlflow-dynamodbstore --name mlflow --region us-east-1 workspace delete \
    --workspace staging --mode cascade --yes
```

!!! danger
    `destroy` permanently deletes the CloudFormation stack. `workspace delete --mode cascade`
    permanently removes all data. These actions cannot be undone.

!!! tip
    Use `--retain` with `destroy` to remove the CloudFormation stack while keeping the
    DynamoDB table and its data intact.

- [ ] **Step 2: Verify docs build**

Run: `uv run mkdocs build --strict 2>&1 | tail -20`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add docs/operator-guide/cli-reference.md
git commit -m "docs: rewrite CLI reference with mkdocs-click auto-generation"
```

---

### Task 14: Update mkdocs.yml nav if needed

**Files:**
- Check: `mkdocs.yml`

- [ ] **Step 1: Verify mkdocs.yml nav references are correct**

The nav should already reference `cli-reference.md` under Operator Guide. No changes needed unless the file path changed.

- [ ] **Step 2: Verify docs build end-to-end**

Run: `uv run mkdocs build --strict`
Expected: Clean build, no warnings about missing references

---

## Post-Plan Refinements

The following changes were made after the 14 planned tasks were completed,
based on review feedback and alignment with AWS ecosystem conventions:

### Task 15: Default `--name` and delegate `--region` to boto3

- `--name` defaults to `mlflow` (not required)
- `--region` is optional — when omitted, boto3 resolves from its standard chain:
  `AWS_REGION` → `AWS_DEFAULT_REGION` → `~/.aws/config` profile
- Updated `DynamoDBTable`, `XRayClient`, provisioner, and URI parser to accept
  `region=None` and only pass `region_name` to boto3 when explicitly set
- Matches behavior of AWS CLI, CDK, SAM, and Terraform

### Task 16: Use cloup for CLI command sections

- Replaced `click>=8.0.0` dependency with `cloup>=3.0.0` (conda-forge available)
- Commands organized into labeled sections in help output:
  - "Stack Lifecycle": `deploy`, `destroy`
  - "Configuration": `tag`, `ttl`, `fts`, `trace`, `workspace`
- Extracted `CliContext`/`pass_context` to `cli/_context.py` to avoid circular
  imports between `__init__.py` and subcommand modules
- Added mypy overrides for cloup (no type stubs available)

### Task 17: URI parser defaults

- `dynamodb://` now works with no arguments → table `mlflow`, region from boto3
- `dynamodb://us-east-1` → table `mlflow`, explicit region
- `dynamodb://localhost:5000` → table `mlflow`, local endpoint
- Added `DEFAULT_TABLE_NAME = "mlflow"` constant to `dynamodb/uri.py`
