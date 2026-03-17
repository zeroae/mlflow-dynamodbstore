# CLI Rework Design

## Problem

The `mlflow-dynamodbstore` CLI has grown organically with flat command names
(`delete-workspace`, `cache-spans`, `cleanup-expired`, `denormalize-tags`,
`fts-trigrams`, `ttl-policy`). The CloudFormation stack is auto-created on first
use with a `mlflow-dynamodbstore-` prefix on the stack name, offering no direct
lifecycle control (no way to delete a stack via CLI) and an opaque naming
convention.

## Goals

1. **Noun-verb CLI structure** — group related commands under nouns for
   discoverability and consistency with robust CLIs. Commands organized into
   labeled sections using `cloup`.
2. **Explicit stack lifecycle** — `deploy` and `destroy` commands for
   CloudFormation management.
3. **Simplified stack naming** — stack name = table name (drop the
   `mlflow-dynamodbstore-` prefix). Default table name: `mlflow`.
4. **Auto-deploy by default** — consistent with MLflow SQLAlchemy stores that
   auto-create schema on first connect. Opt-out via `?deploy=false` URI param.
5. **Auto-generated CLI docs** — use `mkdocs-click` (same pattern as
   zae-limiter) for a single-directive reference page.
6. **Sensible defaults** — `--name` defaults to `mlflow`, `--region` defers to
   boto3's resolution chain (`AWS_REGION` → `AWS_DEFAULT_REGION` →
   `~/.aws/config`). URI `dynamodb://` works with no arguments.

## CLI Structure

### Global Options

All commands inherit these from the top-level `cloup` group:

| Option           | Description                                    | Default     |
|------------------|------------------------------------------------|-------------|
| `--name`         | Stack/table name                               | `mlflow`    |
| `--region`       | AWS region (omit to use boto3 default chain)   | from boto3  |
| `--endpoint-url` | Custom endpoint (LocalStack/testing)           | None        |

### Command Tree

```
mlflow-dynamodbstore [--name NAME] [--region REGION] [--endpoint-url URL]

Stack Lifecycle:
  deploy                              # Create CFn stack + seed data
  destroy [--yes] [--retain]          # Delete CFn stack

Configuration:
  tag list/add/remove/backfill        # Tag denormalization patterns
  ttl show/set/cleanup                # TTL policies + orphan cleanup
  fts list/add                        # Full-text search trigram fields
  trace cache                         # Cache X-Ray spans for traces
  workspace delete                    # Soft or cascade delete workspace
```

### Command Details

#### `deploy`

Creates the CloudFormation stack (stack name = `--name`) and seeds initial data
(default workspace, default experiment, config items). Idempotent — skips if the
stack already exists in a good state.

#### `destroy`

Deletes the CloudFormation stack.

| Flag       | Description                                            |
|------------|--------------------------------------------------------|
| `--yes`    | Skip confirmation prompt                               |
| `--retain` | Delete stack but keep the DynamoDB table (via CFn DeletionPolicy Retain) |

Operator responsibility to stop the MLflow server before destroying — auto-deploy
will re-create the stack if the server is still running.

#### `workspace delete`

Replaces `delete-workspace`. Same functionality: soft or cascade mode, `--yes`
to skip confirmation.

#### `tag list/add/remove/backfill`

Replaces `denormalize-tags`. Subcommand names simplified from `list_patterns` →
`list`, `add_pattern` → `add`, `remove_pattern` → `remove`, `backfill` stays.

#### `ttl show/set/cleanup`

Replaces `ttl-policy` and absorbs `cleanup-expired`. Subcommand names simplified
from `show_policy` → `show`, `set_policy` → `set`, plus new `cleanup`.

#### `fts list/add`

Replaces `fts-trigrams`. Subcommand names simplified from `list_fields` → `list`,
`add_field` → `add`.

#### `trace cache`

Replaces `cache-spans`. Caches X-Ray spans for traces in DynamoDB.

## Stack Naming Change

- Remove `_STACK_PREFIX = "mlflow-dynamodbstore-"` from `provisioner.py`
- Stack name = table name directly
- `get_stack_name()` removed
- Default table name: `mlflow` (matching CLI `--name` default and URI parser
  `DEFAULT_TABLE_NAME`)

### Migration

Existing deployments have stacks named `mlflow-dynamodbstore-<table>`. After
upgrading, the new code looks for stacks named `<table>`. Existing users must
either:

1. Delete the old stack (`aws cloudformation delete-stack --stack-name mlflow-dynamodbstore-<table>`)
   and let auto-deploy create the new one, or
2. Manually rename by creating a new stack and migrating (the DynamoDB table
   itself is unchanged — only the CloudFormation wrapper changes).

The DynamoDB table name does not change, so data is preserved in both paths.
The `deploy` command will fail if the table already exists without a matching
stack — in that case, use CloudFormation resource import (`aws cloudformation
create-change-set --change-set-type IMPORT`) to adopt the existing table into
a new stack, or simply delete the old stack first (path 1).

## URI Parser Defaults

The URI parser now supports minimal URIs with sensible defaults:

| URI                                  | Table    | Region      | Endpoint             |
|--------------------------------------|----------|-------------|----------------------|
| `dynamodb://`                        | `mlflow` | from boto3  | None                 |
| `dynamodb://us-east-1`               | `mlflow` | `us-east-1` | None                 |
| `dynamodb://us-east-1/my-table`      | `my-table` | `us-east-1` | None               |
| `dynamodb://localhost:5000`          | `mlflow` | from boto3  | `http://localhost:5000` |
| `dynamodb://localhost:5000/my-table` | `my-table` | from boto3 | `http://localhost:5000` |

Query params: `?deploy=true|false` controls auto-deploy (default: `true`).

## Auto-Deploy Behavior

- **Default (no query param or `?deploy=true`)**: stores call
  `ensure_stack_exists()` during `__init__()` — same as today, consistent with
  MLflow SQLAlchemy auto-schema-creation.
- **Opt-out (`?deploy=false`)**: stores skip auto-deploy. If the table doesn't
  exist, the store will fail on first operation.
- URI parsing in `dynamodb/uri.py` handles `?deploy=true|false` query parameter.
  `deploy: bool = True` field on `DynamoDBUriComponents`.

## Region Resolution

Region is resolved by boto3's standard chain — the CLI and provisioner do NOT
manually check environment variables. When `--region` is omitted (or URI has no
region), `region=None` flows through to boto3 which resolves:

1. `AWS_REGION` env var
2. `AWS_DEFAULT_REGION` env var
3. `~/.aws/config` profile `region`

This matches the behavior of AWS CLI, CDK, SAM, and Terraform.

## File Changes

| Action    | File                                | Notes                                                  |
|-----------|-------------------------------------|--------------------------------------------------------|
| Rewrite   | `cli/__init__.py`                   | `cloup.group` with sections, global options with defaults |
| New       | `cli/_context.py`                   | `CliContext` and `pass_context` (avoids circular imports) |
| New       | `cli/deploy.py`                     | `deploy` command (extracted from provisioner)           |
| New       | `cli/destroy.py`                    | `destroy` command                                      |
| Rename    | `cli/denormalize_tags.py` → `cli/tag.py` | Group name `tag`                                  |
| Rename    | `cli/ttl_policy.py` → `cli/ttl.py` | Group name `ttl`, absorbs cleanup-expired              |
| Delete    | `cli/cleanup_expired.py`            | Merged into `cli/ttl.py` as `cleanup`                  |
| Rename    | `cli/fts_trigrams.py` → `cli/fts.py` | Group name `fts`                                     |
| Rename    | `cli/cache_spans.py` → `cli/trace.py` | Group name `trace`, subcommand `cache`              |
| Rename    | `cli/delete_workspace.py` → `cli/workspace.py` | Group name `workspace`                      |
| Update    | `dynamodb/provisioner.py`           | Remove prefix, add `destroy_stack()`, `region=None` support |
| Update    | `dynamodb/uri.py`                   | Default table `mlflow`, `region=None`, `?deploy` param |
| Update    | `dynamodb/table.py`                 | `region=None` support (defer to boto3)                 |
| Update    | `xray/client.py`                    | `region=None` support (defer to boto3)                 |
| Update    | `tracking_store.py`                 | Conditional deploy based on URI param                  |
| Update    | `registry_store.py`                 | Conditional deploy based on URI param                  |
| Update    | `workspace_store.py`                | Conditional deploy based on URI param                  |
| Update    | `auth/store.py`                     | Conditional deploy based on URI param                  |
| Rewrite   | `docs/operator-guide/cli-reference.md` | Single mkdocs-click directive + examples            |
| Update    | `pyproject.toml`                    | Replace `click>=8.0.0` with `cloup>=3.0.0`            |
| Update    | `tests/unit/cli/`                   | Rename test files, update command invocations          |

## Dependencies

- Replaced `click>=8.0.0` with `cloup>=3.0.0` (cloup re-exports click and adds
  command sections, option groups). Available on conda-forge.

## Documentation

Replace per-command `mkdocs-click` directives with a single top-level directive
(matching zae-limiter pattern):

```markdown
::: mkdocs-click
    :module: mlflow_dynamodbstore.cli
    :command: cli
    :prog_name: mlflow-dynamodbstore
    :depth: 2
    :style: table
    :list_subcommands: true
```

Supplement with manual examples and admonitions for destructive operations
(`destroy`, `workspace delete cascade`, `ttl cleanup`).

## Out of Scope

- Renaming the PyPI package or entry point (`mlflow-dynamodbstore` stays)
- Changes to the DynamoDB table schema itself
- Changes to the auth CLI (if any)
