"""Shared CLI context — imported by subcommand modules to avoid circular imports."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import click


class CliContext:
    """Shared context for all CLI commands."""

    def __init__(self, name: str, region: str, endpoint_url: str | None) -> None:
        self.name = name
        self.region = region
        self.endpoint_url = endpoint_url


pass_context: Callable[..., Any] = click.make_pass_decorator(CliContext)
