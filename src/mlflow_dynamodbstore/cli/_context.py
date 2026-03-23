"""Shared CLI context — imported by subcommand modules to avoid circular imports."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import click


class CliContext:
    """Shared context for all CLI commands."""

    def __init__(
        self,
        name: str,
        region: str | None,
        endpoint_url: str | None,
        bucket: str | None = None,
        iam_format: str = "{}",
        permission_boundary: str | None = None,
    ) -> None:
        self.name = name
        self.region = region
        self.endpoint_url = endpoint_url
        self.bucket = bucket
        self.iam_format = iam_format
        self.permission_boundary = permission_boundary


pass_context: Callable[..., Any] = click.make_pass_decorator(CliContext)
