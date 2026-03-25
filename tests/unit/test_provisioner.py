"""Unit tests for CloudFormation template builder."""

from __future__ import annotations

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


def test_template_no_bucket_no_s3_resources():
    """When bucket_name is not provided, no S3/Lambda resources are added."""
    t = _build_template("tbl")
    assert "ArtifactBucket" not in t["Resources"]
    assert "BucketCleanupFunction" not in t["Resources"]
    assert "BucketCleanupRole" not in t["Resources"]
    assert "Outputs" not in t


def test_template_bucket_encryption():
    t = _build_template("tbl", bucket_name="b")
    bucket = t["Resources"]["ArtifactBucket"]["Properties"]
    assert "BucketEncryption" in bucket


def test_template_public_access_block():
    t = _build_template("tbl", bucket_name="b")
    bucket = t["Resources"]["ArtifactBucket"]["Properties"]
    pac = bucket["PublicAccessBlockConfiguration"]
    assert pac["BlockPublicAcls"] is True
    assert pac["BlockPublicPolicy"] is True
    assert pac["IgnorePublicAcls"] is True
    assert pac["RestrictPublicBuckets"] is True


def test_template_lambda_runtime():
    t = _build_template("tbl", bucket_name="b")
    fn = t["Resources"]["BucketCleanupFunction"]["Properties"]
    assert fn["Runtime"] == "python3.12"
    assert fn["Timeout"] == 300


def test_template_permission_boundary_full_arn():
    """Full ARN permission boundary is used as-is, not wrapped in Fn::Sub."""
    arn = "arn:aws:iam::aws:policy/PowerUserAccess"
    t = _build_template("tbl", bucket_name="b", permission_boundary=arn)
    role = t["Resources"]["BucketCleanupRole"]["Properties"]
    assert role["PermissionsBoundary"] == arn


def test_template_permission_boundary_short_name_uses_fn_sub():
    """Short policy name is wrapped in Fn::Sub with account ID."""
    t = _build_template("tbl", bucket_name="b", permission_boundary="PowerUserAccess")
    role = t["Resources"]["BucketCleanupRole"]["Properties"]
    assert "Fn::Sub" in role["PermissionsBoundary"]
    assert "PowerUserAccess" in role["PermissionsBoundary"]["Fn::Sub"]


def test_template_lambda_cfn_response_uses_put():
    """Lambda _send function must use PUT for CloudFormation pre-signed S3 URL."""
    t = _build_template("tbl", bucket_name="b")
    code = t["Resources"]["BucketCleanupFunction"]["Properties"]["Code"]["ZipFile"]
    assert 'method="PUT"' in code
