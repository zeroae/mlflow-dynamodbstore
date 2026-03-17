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
   discoverability and consistency with robust CLIs.
2. **Explicit stack lifecycle** — `deploy` and `destroy` commands for
   CloudFormation management.
3. **Simplified stack naming** — stack name = table name (drop the
   `mlflow-dynamodbstore-` prefix).
4. **Auto-deploy by default** — consistent with MLflow SQLAlchemy stores that
   auto-create schema on first connect. Opt-out via `?deploy=false` URI param.
5. **Auto-generated CLI docs** — use `mkdocs-click` (same pattern as
   zae-limiter) for a single-directive reference page.

## CLI Structure

### Global Options

All commands inherit these from the top-level group:

| Option           | Description                          | Required |
|------------------|--------------------------------------|----------|
| `--name`         | Stack/table name                     | Yes      |
| `--region`       | AWS region (e.g. `us-east-1`)        | Yes      |
| `--endpoint-url` | Custom endpoint (LocalStack/testing) | No       |

### Command Tree

```
mlflow-dynamodbstore [--name NAME] [--region REGION] [--endpoint-url URL]

  deploy                              # Create CFn stack + seed data
  destroy [--yes] [--retain]          # Delete CFn stack

  workspace delete                    # Soft or cascade delete workspace
  tag list/add/remove/backfill        # Tag denormalization patterns
  ttl show/set/cleanup                # TTL policies + orphan cleanup
  fts list/add                        # Full-text search trigram fields
  trace cache                         # Cache X-Ray spans for traces
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
| `--retain` | Delete stack but keep the DynamoDB table via `cfn.delete_stack(StackName=..., RetainResources=['MlflowTable'])` |

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
- `get_stack_name()` becomes identity function (or is removed)

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

## Auto-Deploy Behavior

- **Default (no query param or `?deploy=true`)**: stores call
  `ensure_stack_exists()` during `__init__()` — same as today, consistent with
  MLflow SQLAlchemy auto-schema-creation.
- **Opt-out (`?deploy=false`)**: stores skip auto-deploy. If the table doesn't
  exist, fail with a clear error:
  `Table 'X' not found. Run 'mlflow-dynamodbstore --name X --region R deploy' first.`
- URI parsing in `dynamodb/uri.py` extended to handle `?deploy=true|false` query
  parameter. Add `deploy: bool = True` field to `DynamoDBUriComponents`.

## File Changes

| Action    | File                                | Notes                                                  |
|-----------|-------------------------------------|--------------------------------------------------------|
| Rewrite   | `cli/__init__.py`                   | Global `--name`/`--region`/`--endpoint-url`, new registration |
| New       | `cli/deploy.py`                     | `deploy` command (extracted from provisioner)           |
| New       | `cli/destroy.py`                    | `destroy` command                                      |
| Rename    | `cli/denormalize_tags.py` → `cli/tag.py` | Group name `tag`                                  |
| Rename    | `cli/ttl_policy.py` → `cli/ttl.py` | Group name `ttl`, absorbs cleanup-expired              |
| Delete    | `cli/cleanup_expired.py`            | Merged into `cli/ttl.py` as `cleanup`                  |
| Rename    | `cli/fts_trigrams.py` → `cli/fts.py` | Group name `fts`                                     |
| Rename    | `cli/cache_spans.py` → `cli/trace.py` | Group name `trace`, subcommand `cache`              |
| Rename    | `cli/delete_workspace.py` → `cli/workspace.py` | Group name `workspace`                      |
| Update    | `dynamodb/provisioner.py`           | Remove `_STACK_PREFIX`, add `destroy_stack()`          |
| Update    | `dynamodb/uri.py`                   | Parse `?deploy=true\|false` query param                |
| Update    | `tracking_store.py`                 | Conditional deploy based on URI param                  |
| Update    | `registry_store.py`                 | Conditional deploy based on URI param                  |
| Update    | `workspace_store.py`                | Conditional deploy based on URI param                  |
| Update    | `auth/store.py`                     | Conditional deploy based on URI param                  |
| Rewrite   | `docs/operator-guide/cli-reference.md` | Single mkdocs-click directive + examples            |
| Update    | `tests/unit/cli/`                   | Rename test files, update command invocations          |

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
