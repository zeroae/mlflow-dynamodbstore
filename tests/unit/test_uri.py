"""Tests for dynamodb:// URI parsing."""

from __future__ import annotations

from mlflow_dynamodbstore.dynamodb.uri import parse_dynamodb_uri


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
