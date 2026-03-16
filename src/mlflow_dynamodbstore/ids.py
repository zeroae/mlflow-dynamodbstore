"""ULID generation with custom timestamp support."""

from __future__ import annotations


def generate_ulid() -> str:
    """Generate a new ULID with current timestamp.

    Returns lowercase string representation.
    """
    from ulid import ULID

    return str(ULID()).lower()


def ulid_from_timestamp(timestamp_ms: int) -> str:
    """Generate a ULID with a specific timestamp (milliseconds).

    The timestamp is encoded in the ULID's first 10 characters.
    Random suffix ensures uniqueness for same-timestamp ULIDs.

    Returns lowercase string representation.
    """
    from ulid import ULID

    return str(ULID.from_timestamp(timestamp_ms / 1000.0)).lower()
