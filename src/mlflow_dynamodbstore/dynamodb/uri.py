"""Parse dynamodb:// URIs into connection parameters."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DynamoDBUriComponents:
    """Parsed components of a dynamodb:// URI."""

    table_name: str
    region: str = "us-east-1"
    endpoint_url: str | None = None


def parse_dynamodb_uri(uri: str) -> DynamoDBUriComponents:
    """Parse a dynamodb:// URI into components.

    Supported formats:
        dynamodb://us-east-1/my-table          — AWS region + table
        dynamodb://localhost:5000/test-table    — local endpoint + table
        dynamodb://http://host:port/table       — explicit endpoint + table

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

    if not rest or "/" not in rest:
        raise ValueError(f"URI must include a table name: {uri}")

    # Check for explicit http(s):// endpoint
    if rest.startswith("http://") or rest.startswith("https://"):
        last_slash = rest.rfind("/")
        endpoint_url = rest[:last_slash]
        table_name = rest[last_slash + 1 :]
        return DynamoDBUriComponents(
            table_name=table_name,
            endpoint_url=endpoint_url,
        )

    host, table_name = rest.split("/", 1)

    if not table_name:
        raise ValueError(f"URI must include a table name: {uri}")

    # Check if host looks like a region (e.g., us-east-1) or a host:port
    if re.match(r"^[a-z]{2}-[a-z]+-\d+$", host):
        return DynamoDBUriComponents(table_name=table_name, region=host)

    # host:port or localhost
    endpoint_url = f"http://{host}"
    return DynamoDBUriComponents(
        table_name=table_name,
        endpoint_url=endpoint_url,
    )
