# S3 Artifact Bucket Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an S3 bucket to the CloudFormation stack for artifact storage and large-item overflow.

**Architecture:** Extend the existing CloudFormation provisioner to create an S3 bucket alongside the DynamoDB table. The tracking store auto-discovers the bucket from stack outputs and uses it for artifacts and overflow. Large dataset fields (schema, profile) exceeding ~350KB are transparently stored in S3.

**Tech Stack:** Python, boto3, CloudFormation, S3, moto (testing)

**Spec:** `docs/superpowers/specs/2026-03-22-s3-artifact-bucket-design.md`

**moto limitation:** moto does not invoke Lambda-backed CloudFormation custom resources. The Lambda cleanup code cannot be tested under moto — only its template structure can be verified. Bucket emptying is tested directly via S3 API calls.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/mlflow_dynamodbstore/dynamodb/uri.py` | Modify | Add `bucket`, `iam_format`, `permission_boundary` to URI parsing |
| `src/mlflow_dynamodbstore/dynamodb/provisioner.py` | Modify | Add S3 bucket, Lambda, IAM role, custom resource to CloudFormation template |
| `src/mlflow_dynamodbstore/dynamodb/overflow.py` | Create | S3 overflow read/write helpers |
| `src/mlflow_dynamodbstore/tracking_store.py` | Modify | Auto-discover bucket, use overflow for large dataset fields |
| `src/mlflow_dynamodbstore/cli/__init__.py` | Modify | Add `--bucket`, `--iam-format`, `--permission-boundary` options |
| `src/mlflow_dynamodbstore/cli/_context.py` | Modify | Add new fields to `CliContext` |
| `src/mlflow_dynamodbstore/cli/deploy.py` | Modify | Pass new params to `ensure_stack_exists` |
| `src/mlflow_dynamodbstore/cli/destroy.py` | Modify | Update confirmation message for bucket |
| `tests/unit/test_uri.py` | Modify | Test new URI query params |
| `tests/unit/test_provisioner.py` | Create | Test CloudFormation template generation (pure dict assertions, no moto) |
| `tests/unit/test_overflow.py` | Create | Test overflow write/read helpers (moto) |
| `tests/integration/test_provisioner_s3.py` | Create | Test stack creates bucket, destroy empties+deletes (moto) |
| `tests/compatibility/test_tracking_compat.py` | Modify | Remove `test_log_inputs_with_large_inputs_limit_check` xfail |

---

### Task 1: URI Parsing — New Query Params

**Files:**
- Modify: `src/mlflow_dynamodbstore/dynamodb/uri.py`
- Modify: `tests/unit/test_uri.py`

**Note:** The current `uri.py` has 5 `return DynamoDBUriComponents(...)` call sites across different branches (bare scheme, http endpoint with/without path, region, host:port). ALL must be updated to pass the new params.

- [ ] **Step 1: Write failing tests for new URI params**

Include tests for all URI variants: region-based, http-endpoint, bare scheme, and combined params:

```python
def test_parse_uri_with_bucket():
    result = parse_dynamodb_uri("dynamodb://us-east-1/my-table?bucket=my-bucket")
    assert result.bucket == "my-bucket"

def test_parse_uri_with_iam_format():
    result = parse_dynamodb_uri("dynamodb://us-east-1/my-table?iam_format=PowerUserPB-{}")
    assert result.iam_format == "PowerUserPB-{}"

def test_parse_uri_with_permission_boundary():
    result = parse_dynamodb_uri("dynamodb://us-east-1/t?permission_boundary=PowerUserAccess")
    assert result.permission_boundary == "PowerUserAccess"

def test_parse_uri_defaults_no_new_params():
    result = parse_dynamodb_uri("dynamodb://us-east-1/my-table")
    assert result.bucket is None
    assert result.iam_format == "{}"
    assert result.permission_boundary is None

def test_parse_uri_all_params_combined():
    uri = "dynamodb://us-east-1/t?bucket=b&iam_format=PB-{}&permission_boundary=PUA&deploy=false"
    result = parse_dynamodb_uri(uri)
    assert result.bucket == "b"
    assert result.iam_format == "PB-{}"
    assert result.permission_boundary == "PUA"
    assert result.deploy is False

def test_parse_uri_http_endpoint_with_bucket():
    result = parse_dynamodb_uri("dynamodb://http://localhost:5000/table?bucket=my-bucket")
    assert result.bucket == "my-bucket"
    assert result.endpoint_url == "http://localhost:5000"

def test_parse_uri_bare_scheme_defaults():
    result = parse_dynamodb_uri("dynamodb://")
    assert result.bucket is None
    assert result.iam_format == "{}"

def test_parse_uri_host_port_with_params():
    result = parse_dynamodb_uri("dynamodb://localhost:8000/t?iam_format={}-managed")
    assert result.iam_format == "{}-managed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_uri.py -v -k "bucket or iam_format or permission_boundary"`

- [ ] **Step 3: Add fields and refactor parser**

Add fields to `DynamoDBUriComponents`. Refactor `_parse_query_params` to return all params as a dict. Update ALL 5 return sites to extract and pass the new fields.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_uri.py -v`

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add bucket, iam_format, permission_boundary to URI parsing"
```

---

### Task 2: CLI — New Options

**Files:**
- Modify: `src/mlflow_dynamodbstore/cli/__init__.py`
- Modify: `src/mlflow_dynamodbstore/cli/_context.py`
- Modify: `src/mlflow_dynamodbstore/cli/deploy.py`
- Modify: `src/mlflow_dynamodbstore/cli/destroy.py`

- [ ] **Step 1: Add fields to `CliContext`**

Add `bucket: str | None`, `iam_format: str`, `permission_boundary: str | None` with defaults.

- [ ] **Step 2: Add options to CLI group in `__init__.py`**

```python
@cloup.option("--bucket", default=None, help="S3 bucket name (default: {stack}-artifacts-{account-id})")
@cloup.option("--iam-format", default="{}", show_default=True, help="Format string for IAM role/policy names")
@cloup.option("--permission-boundary", default=None, help="IAM permissions boundary policy name")
```

- [ ] **Step 3: Update `deploy.py`**

Pass `bucket_name=ctx.bucket`, `iam_format=ctx.iam_format`, `permission_boundary=ctx.permission_boundary` to `ensure_stack_exists`.

- [ ] **Step 4: Update `destroy.py` confirmation message**

```python
click.confirm(
    f"Destroy stack '{ctx.name}'? This will delete the CloudFormation stack"
    + (" (table and bucket will be retained)" if retain else " and all resources (table + bucket)"),
    abort=True,
)
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add --bucket, --iam-format, --permission-boundary CLI options"
```

---

### Task 3: CloudFormation Template — S3 Bucket + Lambda Cleanup

**Files:**
- Modify: `src/mlflow_dynamodbstore/dynamodb/provisioner.py`
- Create: `tests/unit/test_provisioner.py`

- [ ] **Step 1: Write failing unit tests (pure template dict assertions, no moto)**

```python
from mlflow_dynamodbstore.dynamodb.provisioner import _build_template

def test_template_includes_s3_bucket():
    t = _build_template("tbl", bucket_name="my-bucket")
    assert "ArtifactBucket" in t["Resources"]
    assert t["Resources"]["ArtifactBucket"]["Properties"]["BucketName"] == "my-bucket"

def test_template_includes_cleanup_resources():
    t = _build_template("tbl", bucket_name="b")
    assert "BucketCleanupFunction" in t["Resources"]
    assert "BucketCleanupRole" in t["Resources"]
    assert "BucketCleanupCustomResource" in t["Resources"]

def test_template_iam_format():
    t = _build_template("tbl", bucket_name="b", iam_format="PowerUserPB-{}")
    role_name = t["Resources"]["BucketCleanupRole"]["Properties"]["RoleName"]
    # RoleName uses Fn::Sub — verify the format is in the template string
    assert "PowerUserPB-" in str(role_name)

def test_template_permission_boundary_present():
    t = _build_template("tbl", bucket_name="b", permission_boundary="PowerUserAccess")
    role = t["Resources"]["BucketCleanupRole"]["Properties"]
    assert "PermissionsBoundary" in role

def test_template_no_permission_boundary():
    t = _build_template("tbl", bucket_name="b")
    role = t["Resources"]["BucketCleanupRole"]["Properties"]
    assert "PermissionsBoundary" not in role

def test_template_retain_bucket():
    t = _build_template("tbl", bucket_name="b", retain_bucket=True)
    assert t["Resources"]["ArtifactBucket"].get("DeletionPolicy") == "Retain"
    assert "BucketCleanupCustomResource" not in t["Resources"]

def test_template_outputs():
    t = _build_template("tbl", bucket_name="b")
    assert "ArtifactBucketName" in t["Outputs"]
    assert "ArtifactBucketArn" in t["Outputs"]

def test_template_lambda_has_logs_permissions():
    t = _build_template("tbl", bucket_name="b")
    role = t["Resources"]["BucketCleanupRole"]["Properties"]
    policy_doc = str(role["Policies"])
    assert "logs:CreateLogGroup" in policy_doc
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_provisioner.py -v`

- [ ] **Step 3: Update `_build_template` with new resources**

New signature:
```python
def _build_template(
    table_name: str,
    retain_table: bool = False,
    retain_bucket: bool = False,
    bucket_name: str | None = None,
    iam_format: str = "{}",
    permission_boundary: str | None = None,
) -> dict[str, Any]:
```

Add: S3 Bucket, IAM Role (with CloudWatch Logs + S3 permissions), Lambda Function (inline Python), Custom Resource, Outputs. Conditionally exclude cleanup resources when `retain_bucket=True`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_provisioner.py -v`

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add S3 bucket and Lambda cleanup to CloudFormation template"
```

---

### Task 4: Provisioner — `ensure_stack_exists` and `destroy_stack` Updates

**Files:**
- Modify: `src/mlflow_dynamodbstore/dynamodb/provisioner.py`

- [ ] **Step 1: Update `ensure_stack_exists` signature**

```python
def ensure_stack_exists(
    table_name: str,
    region: str | None = None,
    endpoint_url: str | None = None,
    bucket_name: str | None = None,
    iam_format: str = "{}",
    permission_boundary: str | None = None,
) -> None:
```

- If `bucket_name` is None, compute default via STS: `f"{table_name}-artifacts-{account_id}"`
- Pass `Capabilities=["CAPABILITY_NAMED_IAM"]` to `create_stack`
- Pass all new params to `_build_template`

- [ ] **Step 2: Add `get_stack_outputs` helper**

```python
def get_stack_outputs(
    stack_name: str,
    region: str | None = None,
    endpoint_url: str | None = None,
) -> dict[str, str]:
    cfn = boto3.client("cloudformation", **_boto_kwargs(region, endpoint_url))
    response = cfn.describe_stacks(StackName=stack_name)
    outputs = response["Stacks"][0].get("Outputs", [])
    return {o["OutputKey"]: o["OutputValue"] for o in outputs}
```

- [ ] **Step 3: Update `destroy_stack`**

When `retain=True`:
- Read `ArtifactBucketName` from stack outputs via `get_stack_outputs`
- Pass `retain_bucket=True` and `bucket_name` to `_build_template`
- Pass `Capabilities=["CAPABILITY_NAMED_IAM"]` to `update_stack`

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: update ensure_stack_exists and destroy_stack for S3 bucket"
```

---

### Task 5: S3 Overflow Helpers

**Files:**
- Create: `src/mlflow_dynamodbstore/dynamodb/overflow.py`
- Create: `tests/unit/test_overflow.py`

- [ ] **Step 1: Write failing tests**

```python
import pytest
from moto import mock_aws
import boto3

@pytest.fixture
def s3_bucket():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        yield "test-bucket"

def test_overflow_write_and_read(s3_bucket):
    from mlflow_dynamodbstore.dynamodb.overflow import overflow_write, overflow_read
    large_value = "x" * 400_000
    ref = overflow_write(s3_bucket, "EXP#123", "D#n#d", "schema", large_value, region="us-east-1")
    result = overflow_read(ref, region="us-east-1")
    assert result == large_value

def test_overflow_key_url_encodes():
    from mlflow_dynamodbstore.dynamodb.overflow import _overflow_key
    key = _overflow_key("EXP#123", "R#abc#INPUT#xyz", "schema")
    # Hash chars are URL-encoded in path segments
    assert "EXP%23123" in key

def test_prepare_item_overflows_large_fields(s3_bucket):
    from mlflow_dynamodbstore.dynamodb.overflow import prepare_item_for_write
    item = {"PK": "EXP#1", "SK": "D#n#d", "schema": "x" * 400_000, "name": "small"}
    prepared = prepare_item_for_write(item, s3_bucket, region="us-east-1")
    assert isinstance(prepared["schema"], dict) and "_s3_ref" in prepared["schema"]
    assert prepared["name"] == "small"

def test_resolve_item_fetches_overflow(s3_bucket):
    from mlflow_dynamodbstore.dynamodb.overflow import prepare_item_for_write, resolve_item_overflows
    item = {"PK": "EXP#1", "SK": "D#n#d", "schema": "x" * 400_000}
    prepared = prepare_item_for_write(item, s3_bucket, region="us-east-1")
    resolved = resolve_item_overflows(prepared, region="us-east-1")
    assert resolved["schema"] == "x" * 400_000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_overflow.py -v`

- [ ] **Step 3: Implement `overflow.py`**

Key design decisions:
- `overflow_write` returns the full `s3://bucket/key` reference
- `overflow_read(s3_ref)` parses the `s3://` URL to extract bucket and key — does NOT recompute from PK/SK
- `resolve_item_overflows` calls `overflow_read(value["_s3_ref"])` directly
- `_OVERFLOW_THRESHOLD = 350_000` bytes

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_overflow.py -v`

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add S3 overflow helpers for large DynamoDB items"
```

---

### Task 6: Tracking Store — Auto-Discover Bucket + Overflow Integration

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`

- [ ] **Step 1: Update `__init__` to auto-discover bucket**

After `ensure_stack_exists`:
1. If `uri.bucket` is set, use it directly
2. Otherwise, try `get_stack_outputs` to read `ArtifactBucketName`
3. Fall back to provided `artifact_uri` or `./mlartifacts`

Store `self._overflow_bucket` and `self._uri` for overflow calls.

- [ ] **Step 2: Integrate overflow into `log_inputs` write path (~line 3268)**

After building the dataset item dict, before appending to `items`:

```python
dataset_item = {"PK": pk, "SK": ..., "schema": ds.schema, "profile": ds.profile, ...}
if self._overflow_bucket:
    from mlflow_dynamodbstore.dynamodb.overflow import prepare_item_for_write
    dataset_item = prepare_item_for_write(
        dataset_item, self._overflow_bucket,
        self._uri.region, self._uri.endpoint_url,
    )
items.append(dataset_item)
```

- [ ] **Step 3: Integrate overflow into dataset read path (~line 341-366 in `_item_to_run`)**

`_item_to_run` is a module-level function and doesn't have access to `self._overflow_bucket`. Two options:
1. Pass `overflow_bucket` as a parameter to `_item_to_run`
2. Resolve overflows on `dataset_items` before passing them to `_item_to_run`

Option 2 is simpler — in the caller (tracking store methods that call `_item_to_run`), resolve overflows on the dataset items before passing them in. Find all call sites of `_item_to_run` and add:

```python
if self._overflow_bucket and dataset_items:
    from mlflow_dynamodbstore.dynamodb.overflow import resolve_item_overflows
    dataset_items = [
        resolve_item_overflows(d, self._uri.region, self._uri.endpoint_url)
        for d in dataset_items
    ]
```

Check `_build_run_from_meta_item` and any other method that fetches dataset items and passes them to `_item_to_run`.

- [ ] **Step 4: Run unit tests**

Run: `uv run pytest tests/unit/ -x -q`

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: auto-discover S3 bucket and integrate overflow for datasets"
```

---

### Task 7: Other Stores — Pass New URI Params

**Files:**
- Modify: `src/mlflow_dynamodbstore/registry_store.py`
- Modify: `src/mlflow_dynamodbstore/workspace_store.py`
- Modify: `src/mlflow_dynamodbstore/auth/store.py`
- Modify: `src/mlflow_dynamodbstore/job_store.py`

- [ ] **Step 1: Update all `ensure_stack_exists` calls**

Each store's `__init__` calls `ensure_stack_exists`. Add the new params from `uri`:

```python
ensure_stack_exists(
    uri.table_name, uri.region, uri.endpoint_url,
    bucket_name=uri.bucket,
    iam_format=uri.iam_format,
    permission_boundary=uri.permission_boundary,
)
```

- [ ] **Step 2: Run unit tests**

Run: `uv run pytest tests/unit/ -x -q`

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: pass bucket/IAM params through all store constructors"
```

---

### Task 8: Integration Tests

**Files:**
- Create: `tests/integration/test_provisioner_s3.py`

- [ ] **Step 1: Write integration tests (moto)**

```python
import boto3
import pytest
from moto import mock_aws
from mlflow_dynamodbstore.dynamodb.provisioner import (
    destroy_stack, ensure_stack_exists, get_stack_outputs,
)

@pytest.fixture
def aws_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

@mock_aws
def test_deploy_creates_bucket_and_table(aws_env):
    ensure_stack_exists("test", region="us-east-1", bucket_name="test-artifacts")
    outputs = get_stack_outputs("test", region="us-east-1")
    assert outputs["ArtifactBucketName"] == "test-artifacts"
    ddb = boto3.client("dynamodb", region_name="us-east-1")
    assert "test" in ddb.list_tables()["TableNames"]

@mock_aws
def test_deploy_default_bucket_name(aws_env):
    ensure_stack_exists("mystack", region="us-east-1")
    outputs = get_stack_outputs("mystack", region="us-east-1")
    # moto returns account 123456789012
    assert outputs["ArtifactBucketName"] == "mystack-artifacts-123456789012"

@mock_aws
def test_destroy_with_retain(aws_env):
    ensure_stack_exists("test", region="us-east-1", bucket_name="test-artifacts")
    destroy_stack("test", region="us-east-1", retain=True)
    s3 = boto3.client("s3", region_name="us-east-1")
    assert "test-artifacts" in [b["Name"] for b in s3.list_buckets()["Buckets"]]

@mock_aws
def test_store_autodiscover_bucket(aws_env):
    from mlflow_dynamodbstore.tracking_store import DynamoDBTrackingStore
    store = DynamoDBTrackingStore("dynamodb://us-east-1/test?bucket=test-artifacts")
    assert store._overflow_bucket == "test-artifacts"
    assert store._artifact_uri == "s3://test-artifacts"

@mock_aws
def test_store_fallback_without_stack(aws_env):
    from mlflow_dynamodbstore.tracking_store import DynamoDBTrackingStore
    store = DynamoDBTrackingStore("dynamodb://us-east-1/test?deploy=false")
    assert store._artifact_uri == "./mlartifacts"
    assert store._overflow_bucket is None
```

**Note:** moto does not invoke Lambda custom resources, so `test_destroy_deletes_bucket_with_objects` cannot be tested end-to-end. The Lambda template structure is verified by unit tests in Task 3.

- [ ] **Step 2: Run integration tests**

Run: `uv run pytest tests/integration/test_provisioner_s3.py -v`

- [ ] **Step 3: Commit**

```bash
git commit -m "test: integration tests for S3 bucket provisioning"
```

---

### Task 9: Remove Compat Test xfail

**Files:**
- Modify: `tests/compatibility/test_tracking_compat.py`

- [ ] **Step 1: Remove the xfail**

Remove:
```python
test_log_inputs_with_large_inputs_limit_check = pytest.mark.xfail(
    reason="DynamoDB 400KB item size limit vs MLflow's 1MB schema / 16MB profile limits"
)(test_log_inputs_with_large_inputs_limit_check)
```

- [ ] **Step 2: Run the compat test**

Run: `uv run pytest tests/compatibility/test_tracking_compat.py::test_log_inputs_with_large_inputs_limit_check -x -v`
Expected: PASS

- [ ] **Step 3: Run full compat suite**

Run: `uv run pytest tests/compatibility/test_tracking_compat.py -n auto`
Expected: 0 xfails

- [ ] **Step 4: Commit**

```bash
git commit -m "fix: remove last tracking compat xfail — all tests passing"
```
