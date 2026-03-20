"""DynamoDB table client wrapping boto3 resource API."""

from __future__ import annotations

from typing import Any

import boto3
from boto3.dynamodb.conditions import ConditionBase, Key

# Mapping from index name to (pk_attr, sk_attr)
_INDEX_KEY_ATTRS: dict[str, tuple[str, str]] = {
    "gsi1": ("gsi1pk", "gsi1sk"),
    "gsi2": ("gsi2pk", "gsi2sk"),
    "gsi3": ("gsi3pk", "gsi3sk"),
    "gsi4": ("gsi4pk", "gsi4sk"),
    "gsi5": ("gsi5pk", "gsi5sk"),
    "lsi1": ("PK", "lsi1sk"),
    "lsi2": ("PK", "lsi2sk"),
    "lsi3": ("PK", "lsi3sk"),
    "lsi4": ("PK", "lsi4sk"),
    "lsi5": ("PK", "lsi5sk"),
}

_BATCH_WRITE_CHUNK_SIZE = 25


def convert_decimals(obj: Any) -> Any:
    """Recursively convert Decimal values to int/float for JSON serialization.

    DynamoDB returns numeric values as ``decimal.Decimal``.  Most MLflow entity
    classes call ``json.dumps`` on stored dicts, which fails on ``Decimal``.
    """
    from decimal import Decimal

    if isinstance(obj, Decimal):
        return int(obj) if obj == int(obj) else float(obj)
    if isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_decimals(v) for v in obj]
    return obj


def convert_floats(obj: Any) -> Any:
    """Recursively convert float values to Decimal for DynamoDB storage.

    DynamoDB's boto3 resource API rejects Python ``float`` values.
    This converts them to ``Decimal`` before writing.
    """
    from decimal import Decimal

    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_floats(v) for v in obj]
    return obj


# Attributes that must be numeric (DynamoDB N type) per table schema.
# All other index key attributes are String (S).
_NUMERIC_ATTRS: frozenset[str] = frozenset({"lsi2sk"})


def _validate_index_key_types(item: dict[str, Any]) -> None:
    """Raise TypeError if an index key attribute has the wrong Python type."""
    for attr in _NUMERIC_ATTRS:
        if attr in item and not isinstance(item[attr], int | float):
            val = item[attr]
            raise TypeError(
                f"Attribute {attr!r} must be numeric, got {type(val).__name__}: {val!r}"
            )
    # String index keys: all LSI sk attrs except the numeric ones
    for attr_key in ("lsi1sk", "lsi3sk", "lsi4sk", "lsi5sk"):
        if attr_key in item and not isinstance(item[attr_key], str):
            val = item[attr_key]
            raise TypeError(
                f"Attribute {attr_key!r} must be str, got {type(val).__name__}: {val!r}"
            )


class DynamoDBTable:
    """High-level DynamoDB table client using boto3 resource API."""

    def __init__(
        self,
        table_name: str,
        region: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if region:
            kwargs["region_name"] = region
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        resource = boto3.resource("dynamodb", **kwargs)
        self._table = resource.Table(table_name)

    # ------------------------------------------------------------------
    # Single-item operations
    # ------------------------------------------------------------------

    def put_item(self, item: dict[str, Any], condition: str | None = None) -> None:
        """Write an item, with optional ConditionExpression."""
        _validate_index_key_types(item)
        item = convert_floats(item)
        kwargs: dict[str, Any] = {"Item": item}
        if condition:
            kwargs["ConditionExpression"] = condition
        self._table.put_item(**kwargs)

    def get_item(self, pk: str, sk: str, consistent: bool = False) -> dict[str, Any] | None:
        """Return item by PK+SK, or None if not found."""
        kwargs: dict[str, Any] = {
            "Key": {"PK": pk, "SK": sk},
            "ConsistentRead": consistent,
        }
        response = self._table.get_item(**kwargs)
        item: dict[str, Any] | None = response.get("Item")
        return item

    def delete_item(self, pk: str, sk: str) -> None:
        """Delete item by PK+SK."""
        self._table.delete_item(Key={"PK": pk, "SK": sk})

    def update_item(
        self,
        pk: str,
        sk: str,
        updates: dict[str, Any] | None = None,
        removes: list[str] | None = None,
        condition: str | None = None,
    ) -> dict[str, Any] | None:
        """Update attributes on an item using SET and/or REMOVE expressions."""
        expression_parts: list[str] = []
        expr_names: dict[str, str] = {}
        expr_values: dict[str, Any] = {}

        if updates:
            _validate_index_key_types(updates)
            set_clauses = []
            for i, (attr, val) in enumerate(updates.items()):
                name_token = f"#u{i}"
                value_token = f":u{i}"
                expr_names[name_token] = attr
                expr_values[value_token] = val
                set_clauses.append(f"{name_token} = {value_token}")
            expression_parts.append("SET " + ", ".join(set_clauses))

        if removes:
            remove_tokens = []
            for i, attr in enumerate(removes):
                name_token = f"#r{i}"
                expr_names[name_token] = attr
                remove_tokens.append(name_token)
            expression_parts.append("REMOVE " + ", ".join(remove_tokens))

        if not expression_parts:
            return None

        kwargs: dict[str, Any] = {
            "Key": {"PK": pk, "SK": sk},
            "UpdateExpression": " ".join(expression_parts),
            "ReturnValues": "ALL_NEW",
        }
        if expr_names:
            kwargs["ExpressionAttributeNames"] = expr_names
        if expr_values:
            kwargs["ExpressionAttributeValues"] = expr_values
        if condition:
            kwargs["ConditionExpression"] = condition

        response = self._table.update_item(**kwargs)
        attributes: dict[str, Any] | None = response.get("Attributes")
        return attributes

    def add_attribute(
        self,
        pk: str,
        sk: str,
        attribute: str,
        value: int | float,
    ) -> dict[str, Any]:
        """Atomically increment an attribute using ADD expression. Returns updated item."""
        response = self._table.update_item(
            Key={"PK": pk, "SK": sk},
            UpdateExpression="ADD #attr :val",
            ExpressionAttributeNames={"#attr": attribute},
            ExpressionAttributeValues={":val": value},
            ReturnValues="UPDATED_NEW",
        )
        attributes: dict[str, Any] = response.get("Attributes", {})
        return attributes

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        pk: str,
        sk_prefix: str | None = None,
        sk_gte: str | None = None,
        sk_lte: str | None = None,
        index_name: str | None = None,
        limit: int | None = None,
        scan_forward: bool = True,
        consistent: bool = False,
        filter_expression: ConditionBase | None = None,
    ) -> list[dict[str, Any]]:
        """Query the table or an index with flexible key conditions.

        For index queries the PK attribute name is derived from index_name
        (e.g. "gsi1" -> "gsi1pk", "lsi1" -> "PK").
        """
        # Determine key attribute names
        if index_name:
            pk_attr, sk_attr = _INDEX_KEY_ATTRS[index_name]
        else:
            pk_attr, sk_attr = "PK", "SK"

        # Build key condition
        key_cond: ConditionBase = Key(pk_attr).eq(pk)
        if sk_prefix:
            key_cond = key_cond & Key(sk_attr).begins_with(sk_prefix)
        elif sk_gte is not None and sk_lte is not None:
            key_cond = key_cond & Key(sk_attr).between(sk_gte, sk_lte)

        kwargs: dict[str, Any] = {
            "KeyConditionExpression": key_cond,
            "ScanIndexForward": scan_forward,
            "ConsistentRead": consistent,
        }
        if index_name:
            kwargs["IndexName"] = index_name
        if filter_expression is not None:
            kwargs["FilterExpression"] = filter_expression

        # Collect items, handling pagination
        items: list[dict[str, Any]] = []
        remaining = limit  # None means unlimited

        while True:
            if remaining is not None:
                kwargs["Limit"] = remaining

            response = self._table.query(**kwargs)
            batch: list[dict[str, Any]] = response.get("Items", [])
            items.extend(batch)

            if remaining is not None:
                remaining -= len(batch)
                if remaining <= 0:
                    break

            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            kwargs["ExclusiveStartKey"] = last_key

        return items

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def batch_write(self, items: list[dict[str, Any]]) -> None:
        """Batch write items, chunking into groups of 25."""
        for item in items:
            _validate_index_key_types(item)
        with self._table.batch_writer() as batch:
            for item in items:
                batch.put_item(Item=convert_floats(item))

    def batch_delete(self, keys: list[dict[str, Any]]) -> None:
        """Batch delete items by PK+SK key dicts."""
        with self._table.batch_writer() as batch:
            for key in keys:
                batch.delete_item(Key=key)

    # ------------------------------------------------------------------
    # Paged query
    # ------------------------------------------------------------------

    def query_page(
        self,
        pk: str,
        sk_prefix: str | None = None,
        index_name: str | None = None,
        limit: int | None = None,
        scan_forward: bool = True,
        consistent: bool = False,
        exclusive_start_key: dict[str, Any] | None = None,
        filter_expression: ConditionBase | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """Query a single page, returning (items, last_evaluated_key).

        Unlike query(), this does NOT auto-exhaust pagination.
        Returns the raw LastEvaluatedKey for caller-managed cursors.
        """
        if index_name:
            pk_attr, sk_attr = _INDEX_KEY_ATTRS[index_name]
        else:
            pk_attr, sk_attr = "PK", "SK"

        key_cond: ConditionBase = Key(pk_attr).eq(pk)
        if sk_prefix:
            key_cond = key_cond & Key(sk_attr).begins_with(sk_prefix)

        kwargs: dict[str, Any] = {
            "KeyConditionExpression": key_cond,
            "ScanIndexForward": scan_forward,
            "ConsistentRead": consistent,
        }
        if index_name:
            kwargs["IndexName"] = index_name
        if limit is not None:
            kwargs["Limit"] = limit
        if exclusive_start_key is not None:
            kwargs["ExclusiveStartKey"] = exclusive_start_key
        if filter_expression is not None:
            kwargs["FilterExpression"] = filter_expression

        response = self._table.query(**kwargs)
        items: list[dict[str, Any]] = response.get("Items", [])
        lek: dict[str, Any] | None = response.get("LastEvaluatedKey")
        return items, lek
