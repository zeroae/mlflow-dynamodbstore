from __future__ import annotations

import datetime
from typing import Any

import boto3

_MAX_WINDOW_HOURS = 6
_MAX_BATCH_SIZE = 5


class XRayClient:
    """Wrapper around boto3 X-Ray client with automatic chunking and pagination."""

    def __init__(self, region: str = "us-east-1", endpoint_url: str | None = None):
        kwargs: dict[str, Any] = {"region_name": region}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        self._client = boto3.client("xray", **kwargs)

    def get_trace_summaries(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        filter_expression: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get trace summaries with automatic time window chunking and pagination.

        X-Ray limits GetTraceSummaries to 6-hour windows, so longer ranges are
        automatically split into consecutive chunks.
        """
        summaries: list[dict[str, Any]] = []
        chunk_start = start_time
        while chunk_start < end_time:
            chunk_end = min(chunk_start + datetime.timedelta(hours=_MAX_WINDOW_HOURS), end_time)
            kwargs: dict[str, Any] = {"StartTime": chunk_start, "EndTime": chunk_end}
            if filter_expression:
                kwargs["FilterExpression"] = filter_expression
            # Paginate within chunk
            while True:
                response = self._client.get_trace_summaries(**kwargs)
                summaries.extend(response.get("TraceSummaries", []))
                next_token = response.get("NextToken")
                if not next_token:
                    break
                kwargs["NextToken"] = next_token
            chunk_start = chunk_end
        return summaries

    def batch_get_traces(self, trace_ids: list[str]) -> list[dict[str, Any]]:
        """Get full traces, batching in groups of 5 (X-Ray API limit)."""
        traces: list[dict[str, Any]] = []
        for i in range(0, len(trace_ids), _MAX_BATCH_SIZE):
            batch = trace_ids[i : i + _MAX_BATCH_SIZE]
            response = self._client.batch_get_traces(TraceIds=batch)
            traces.extend(response.get("Traces", []))
        return traces
