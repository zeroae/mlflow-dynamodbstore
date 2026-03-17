"""Parse dynamodb:// URIs into connection parameters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs

DEFAULT_TABLE_NAME = "mlflow"

_REGION_RE = re.compile(r"^[a-z]{2}-[a-z]+-\d+$")


@dataclass(frozen=True)
class DynamoDBUriComponents:
    """Parsed components of a dynamodb:// URI."""

    table_name: str = DEFAULT_TABLE_NAME
    region: str | None = None
    endpoint_url: str | None = None
    deploy: bool = True


def _parse_query_params(raw: str) -> tuple[str, bool]:
    """Split a string from its query params and extract deploy flag.

    Returns:
        Tuple of (value, deploy).
    """
    if "?" in raw:
        value, query_string = raw.split("?", 1)
        params = parse_qs(query_string)
        deploy_values = params.get("deploy", ["true"])
        deploy = deploy_values[0].lower() != "false"
        return value, deploy
    return raw, True


def parse_dynamodb_uri(uri: str) -> DynamoDBUriComponents:
    """Parse a dynamodb:// URI into components.

    Supported formats:
        dynamodb://                            — default table, region from boto3
        dynamodb://us-east-1                   — default table, explicit region
        dynamodb://us-east-1/my-table          — AWS region + table
        dynamodb://localhost:5000/test-table   — local endpoint + table
        dynamodb://localhost:5000              — local endpoint, default table
        dynamodb://http://host:port/table      — explicit endpoint + table
        dynamodb://http://host:port            — explicit endpoint, default table

    Query params:
        deploy=true|false  — whether to auto-deploy CloudFormation stack (default: true)

    Args:
        uri: The DynamoDB URI to parse.

    Returns:
        Parsed URI components.

    Raises:
        ValueError: If the URI is malformed.
    """
    if not uri.startswith("dynamodb://"):
        raise ValueError(f"URI must use dynamodb:// scheme, got: {uri}")

    rest = uri[len("dynamodb://") :]

    # Bare scheme: dynamodb://
    if not rest:
        return DynamoDBUriComponents()

    # Check for explicit http(s):// endpoint
    if rest.startswith("http://") or rest.startswith("https://"):
        last_slash = rest.rfind("/")
        # No path after endpoint: dynamodb://http://host:port
        if last_slash == rest.index("//") + 1:
            endpoint_url, deploy = _parse_query_params(rest)
            return DynamoDBUriComponents(endpoint_url=endpoint_url, deploy=deploy)
        endpoint_url = rest[:last_slash]
        raw_table_name = rest[last_slash + 1 :]
        table_name, deploy = _parse_query_params(raw_table_name)
        return DynamoDBUriComponents(
            table_name=table_name or DEFAULT_TABLE_NAME,
            endpoint_url=endpoint_url,
            deploy=deploy,
        )

    # Split on first /
    if "/" in rest:
        host, raw_table_name = rest.split("/", 1)
        table_name, deploy = _parse_query_params(raw_table_name)
        table_name = table_name or DEFAULT_TABLE_NAME
    else:
        host, deploy = _parse_query_params(rest)
        table_name = DEFAULT_TABLE_NAME

    # Check if host looks like a region (e.g., us-east-1)
    if _REGION_RE.match(host):
        return DynamoDBUriComponents(table_name=table_name, region=host, deploy=deploy)

    # host:port or localhost
    endpoint_url = f"http://{host}"
    return DynamoDBUriComponents(
        table_name=table_name,
        endpoint_url=endpoint_url,
        deploy=deploy,
    )
