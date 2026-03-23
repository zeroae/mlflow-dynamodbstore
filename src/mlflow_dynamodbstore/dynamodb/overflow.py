"""S3 overflow for DynamoDB items exceeding 400KB."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote, urlparse

import boto3

_OVERFLOW_PREFIX = ".overflow"
_OVERFLOW_THRESHOLD = 350_000


def _overflow_key(pk: str, sk: str, field: str) -> str:
    """Return the S3 key for an overflow field."""
    return f"{_OVERFLOW_PREFIX}/{quote(pk, safe='')}/{quote(sk, safe='')}/{field}"


def _s3_kwargs(region: str | None = None, endpoint_url: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if region:
        kwargs["region_name"] = region
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    return kwargs


def overflow_write(
    bucket: str,
    pk: str,
    sk: str,
    field: str,
    value: str,
    region: str | None = None,
    endpoint_url: str | None = None,
) -> str:
    """Upload *value* to S3 and return a ``s3://bucket/key`` reference string."""
    s3 = boto3.client("s3", **_s3_kwargs(region, endpoint_url))
    key = _overflow_key(pk, sk, field)
    s3.put_object(Bucket=bucket, Key=key, Body=value.encode("utf-8"))
    return f"s3://{bucket}/{key}"


def overflow_read(
    s3_ref: str,
    region: str | None = None,
    endpoint_url: str | None = None,
) -> str:
    """Fetch and return the string value stored at *s3_ref* (``s3://bucket/key``)."""
    parsed = urlparse(s3_ref)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    s3 = boto3.client("s3", **_s3_kwargs(region, endpoint_url))
    response = s3.get_object(Bucket=bucket, Key=key)
    return str(response["Body"].read().decode("utf-8"))


def prepare_item_for_write(
    item: dict[str, Any],
    bucket: str,
    region: str | None = None,
    endpoint_url: str | None = None,
) -> dict[str, Any]:
    """Return a copy of *item* with large string fields replaced by S3 refs.

    Any string attribute whose UTF-8 size exceeds ``_OVERFLOW_THRESHOLD`` bytes
    is uploaded to S3 and replaced with ``{"_s3_ref": "s3://..."}``.  The
    ``PK`` and ``SK`` keys are never overflowed.
    """
    result = dict(item)
    pk: str = result["PK"]
    sk: str = result["SK"]
    for field, value in item.items():
        if field in ("PK", "SK") or not isinstance(value, str):
            continue
        if len(value.encode("utf-8")) > _OVERFLOW_THRESHOLD:
            ref = overflow_write(bucket, pk, sk, field, value, region, endpoint_url)
            result[field] = {"_s3_ref": ref}
    return result


def resolve_item_overflows(
    item: dict[str, Any],
    region: str | None = None,
    endpoint_url: str | None = None,
) -> dict[str, Any]:
    """Return a copy of *item* with S3 overflow refs replaced by their actual values."""
    result = dict(item)
    for field, value in item.items():
        if isinstance(value, dict) and "_s3_ref" in value:
            result[field] = overflow_read(value["_s3_ref"], region, endpoint_url)
    return result
