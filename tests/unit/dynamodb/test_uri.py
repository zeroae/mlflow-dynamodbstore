import pytest

from mlflow_dynamodbstore.dynamodb.uri import parse_dynamodb_uri


class TestParseDynamoDBUri:
    def test_region_and_table(self):
        result = parse_dynamodb_uri("dynamodb://us-east-1/my-table")
        assert result.region == "us-east-1"
        assert result.table_name == "my-table"
        assert result.endpoint_url is None

    def test_localhost_endpoint(self):
        result = parse_dynamodb_uri("dynamodb://localhost:5000/test-table")
        assert result.endpoint_url == "http://localhost:5000"
        assert result.table_name == "test-table"
        assert result.region is None  # no region in URI, defer to boto3

    def test_custom_endpoint(self):
        result = parse_dynamodb_uri("dynamodb://http://localhost:8000/test-table")
        assert result.endpoint_url == "http://localhost:8000"
        assert result.table_name == "test-table"

    def test_invalid_uri_no_table(self):
        with pytest.raises(ValueError, match="table name"):
            parse_dynamodb_uri("dynamodb://us-east-1")

    def test_invalid_scheme(self):
        with pytest.raises(ValueError, match="scheme"):
            parse_dynamodb_uri("postgresql://localhost/db")


class TestDeployQueryParam:
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
