# S3 Artifact Bucket in CloudFormation Stack

## Problem

DynamoDB has a 400KB item size limit. MLflow dataset fields `schema` (1MB max) and `profile` (16MB max) exceed this limit. The `test_log_inputs_with_large_inputs_limit_check` compat test fails because DynamoDB rejects items before MLflow's validation runs.

Additionally, users deploying with `dynamodb://` must separately configure an S3 bucket for artifact storage. This requires extra setup that could be eliminated.

## Solution

Add an S3 bucket to the existing CloudFormation stack created by `ensure_stack_exists`. The bucket serves two purposes:

1. **Default artifact root** — MLflow stores run artifacts (model files, plots, exports)
2. **DynamoDB overflow** — large dataset fields that exceed 400KB are stored in S3 transparently

## CloudFormation Resources

### S3 Bucket (`AWS::S3::Bucket`)

- **Name**: `{stack-name}-artifacts-{account-id}` (default), overridable via `bucket` URI param
- **Encryption**: SSE-S3 (default encryption)
- **Public access**: Blocked (all four `PublicAccessBlock` settings enabled)
- **Versioning**: Disabled
- **Lifecycle**: None
- **DeletionPolicy**: `Delete` (bucket emptied by custom resource before deletion)

### IAM Role (`AWS::IAM::Role`)

- **RoleName**: `{iam_format.format(f"{stack}-BucketCleanup")}`
- **PermissionsBoundary**: `arn:aws:iam::{account}:policy/{permission_boundary}` (omitted if not set)
- **AssumeRolePolicy**: `lambda.amazonaws.com`
- **Inline policy** (two statements):
  1. S3: `s3:ListBucket`, `s3:DeleteObject`, `s3:ListObjectVersions`, `s3:DeleteObjectVersion` scoped to the bucket
  2. CloudWatch Logs: `logs:CreateLogGroup`, `logs:CreateLogStream`, `logs:PutLogEvents` scoped to the function's log group ARN

When no `iam_format` is specified, the default `"{}"` passes through the name unchanged. When `permission_boundary` is omitted, no boundary is attached.

### Lambda Function (`AWS::Lambda::Function`)

- **Purpose**: Empty the S3 bucket on stack deletion
- **Runtime**: python3.12
- **Timeout**: 300 seconds
- **Code**: Inline Python (~20 lines) responding to CloudFormation DELETE signal
- **Role**: References the IAM role above

### Custom Resource (`AWS::CloudFormation::CustomResource`)

- **ServiceToken**: References the Lambda function
- **Trigger**: Fires on stack deletion to empty the bucket

### Stack Outputs

- `ArtifactBucketName` — the bucket name
- `ArtifactBucketArn` — for IAM policy composition

### Capability

Both `create_stack` and `update_stack` calls must pass `Capabilities=["CAPABILITY_NAMED_IAM"]` (required for explicit role names).

## Provisioner Changes

### `_build_template` Signature

New parameters:

```python
def _build_template(
    table_name: str,
    retain_table: bool = False,
    retain_bucket: bool = False,
    bucket_name: str | None = None,
    iam_format: str = "{}",
    permission_boundary: str | None = None,
) -> dict:
```

- `bucket_name`: if provided, used as the S3 bucket name; otherwise defaults to `{stack}-artifacts-{account-id}`
- `iam_format`: format string applied to role/policy names
- `permission_boundary`: IAM policy name (not ARN); the template constructs the full ARN using `!Sub "arn:aws:iam::${AWS::AccountId}:policy/{name}"`
- `retain_bucket`: when True, sets `DeletionPolicy: Retain` on the S3 bucket and suppresses the custom resource

### `ensure_stack_exists` Signature

New parameters matching `_build_template`, plus:

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

### Account ID

Retrieved via `boto3.client("sts").get_caller_identity()["Account"]` when needed for default bucket naming. Not needed when `bucket_name` is explicitly provided.

### `destroy_stack` Changes

Updated signature:

```python
def destroy_stack(
    table_name: str,
    region: str | None = None,
    endpoint_url: str | None = None,
    retain: bool = False,
) -> None:
```

When `retain=True`:
- Sets `DeletionPolicy: Retain` on both the DynamoDB table AND the S3 bucket
- Suppresses the bucket cleanup custom resource (so the Lambda doesn't empty the bucket)
- Then deletes the stack — both resources are orphaned but preserved

When `retain=False` (default):
- The custom resource fires, Lambda empties the bucket
- CloudFormation deletes both the bucket and the table

## URI Changes

### New Query Parameters on `dynamodb://` URI

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bucket` | `{stack}-artifacts-{account-id}` | Override S3 bucket name |
| `iam_format` | `"{}"` | Format string for IAM role/policy names. `{}` is replaced with the generated name. Examples: `PowerUserPB-{}`, `{}-managed` |
| `permission_boundary` | (none) | IAM policy name for permissions boundary. Store constructs full ARN: `arn:aws:iam::{account}:policy/{name}` |

### `DynamoDBUriComponents` Dataclass

Three new fields: `bucket`, `iam_format`, `permission_boundary` — all optional, parsed from query params.

## CLI Changes

### New Options on Group Command

```
mlflow-dynamodbstore --bucket my-bucket --iam-format "PowerUserPB-{}" --permission-boundary PowerUserAccess deploy
```

### `CliContext` Dataclass

Three new fields matching the URI params, passed through to `ensure_stack_exists`.

### Destroy Command

- `--retain` keeps both the DynamoDB table AND the S3 bucket (suppresses cleanup Lambda)
- Without `--retain`, the Lambda custom resource empties the bucket before CloudFormation deletes it

## Store Integration

### Tracking Store `__init__`

After `ensure_stack_exists` (or when `deploy=false`):

1. Read `ArtifactBucketName` from CloudFormation stack outputs via `describe_stacks` (works regardless of `deploy` flag, as long as the stack exists)
2. Set `self._artifact_uri = f"s3://{bucket_name}"` unless caller explicitly passed `artifact_uri`
3. Store bucket name for overflow: `self._overflow_bucket = bucket_name`
4. If `describe_stacks` fails (no stack), fall back to the provided `artifact_uri` or `./mlartifacts`

When `bucket` is provided in the URI, use that value directly without consulting CloudFormation outputs.

### Overflow Write

Before writing a DynamoDB item, check if any single attribute exceeds ~350KB:

1. Upload the value to `s3://{bucket}/.overflow/{PK}/{SK}/{field}`
2. Replace the attribute value in the item with `{"_s3_ref": "s3://..."}`
3. Write the item to DynamoDB (now under 400KB)

PK and SK values are URL-encoded before use as S3 key components to avoid issues with `/` characters in SK values.

### Overflow Read

When reading a DynamoDB item, check each attribute for `_s3_ref`:

1. If found, fetch the value from S3
2. Replace the `_s3_ref` with the actual value transparently

### Scope

Only the tracking store handles overflow initially (datasets are tracking-store entities). Registry, workspace, auth, and job stores don't need overflow.

## Overflow S3 Key Convention

```
.overflow/{url_encoded(PK)}/{url_encoded(SK)}/{field}
```

Example: `.overflow/EXP%2301ABC.../R%2301DEF...%23INPUT%23digest1/schema`

The PK embeds the workspace context (experiments are workspace-scoped), so overflow items are inherently workspace-isolated.

## Testing

### Unit Tests

- URI parsing with new query params (`iam_format`, `permission_boundary`, `bucket`)
- Overflow write/read helpers (`_overflow_write` / `_overflow_read`) for S3 roundtrip
- `CliContext` with new fields

### Integration Tests (moto)

- `ensure_stack_exists` creates bucket alongside table
- Stack outputs include `ArtifactBucketName`
- `destroy_stack` empties and deletes bucket
- `destroy_stack --retain` preserves both table and bucket
- Store auto-discovers artifact root from stack output
- Store falls back to `./mlartifacts` when no stack exists (`deploy=false`)
- Overflow write/read roundtrip for large dataset fields

### Compat Test

- `test_log_inputs_with_large_inputs_limit_check` — remove xfail, should pass with overflow

## Future Considerations

- **Import existing bucket**: `--import-bucket` flag to adopt an existing S3 bucket into the stack
- **CDK layer**: The CDK deployment can override this bucket with its own, more customizable S3 configuration
- **Other stores**: Overflow could extend to registry/auth if needed
