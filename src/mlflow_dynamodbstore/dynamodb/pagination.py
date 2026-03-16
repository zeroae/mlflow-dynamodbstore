from __future__ import annotations

import base64
import json
from typing import Any


def encode_page_token(token_data: dict[str, Any]) -> str:
    """Encode pagination state as an opaque base64url token.

    Args:
        token_data: Dictionary containing pagination state, typically:
            - lek: DynamoDB LastEvaluatedKey (or None)
            - exp_idx: experiment index for multi-experiment queries
            - accumulated: count of results accumulated so far

    Returns:
        A base64url-encoded string (without padding) suitable for use as a
        page_token in MLflow search APIs.
    """
    serialized = json.dumps(token_data, separators=(",", ":"))
    encoded = base64.urlsafe_b64encode(serialized.encode()).rstrip(b"=")
    return encoded.decode()


def decode_page_token(token: str | None) -> dict[str, Any] | None:
    """Decode an opaque base64url page token back to pagination state.

    Args:
        token: A base64url-encoded page token previously returned by
            encode_page_token, or None/empty string to indicate the first page.

    Returns:
        The decoded pagination state dictionary, or None if token is None or
        empty (indicating the first page with no prior state).
    """
    if not token:
        return None
    # Re-add stripped padding
    padding = 4 - len(token) % 4
    if padding != 4:
        token = token + "=" * padding
    decoded_bytes = base64.urlsafe_b64decode(token)
    result: dict[str, Any] = json.loads(decoded_bytes.decode())
    return result
