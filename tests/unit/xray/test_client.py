from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pytest

from mlflow_dynamodbstore.xray.client import XRayClient


@pytest.fixture
def mock_boto3_client():
    """Patch boto3.client so XRayClient uses a mock."""
    with patch("mlflow_dynamodbstore.xray.client.boto3.client") as mock_client_ctor:
        mock_xray = MagicMock()
        mock_client_ctor.return_value = mock_xray
        yield mock_xray


class TestGetTraceSummaries:
    def test_single_window(self, mock_boto3_client):
        """Time window <= 6 hours results in a single API call."""
        mock_boto3_client.get_trace_summaries.return_value = {
            "TraceSummaries": [{"Id": "1-abc"}, {"Id": "1-def"}],
        }
        client = XRayClient()
        start = datetime.datetime(2026, 1, 1, 0, 0, tzinfo=datetime.UTC)
        end = datetime.datetime(2026, 1, 1, 5, 0, tzinfo=datetime.UTC)

        result = client.get_trace_summaries(start, end)

        assert len(result) == 2
        assert result[0]["Id"] == "1-abc"
        mock_boto3_client.get_trace_summaries.assert_called_once_with(StartTime=start, EndTime=end)

    def test_single_window_with_filter(self, mock_boto3_client):
        """Filter expression is passed through to the API call."""
        mock_boto3_client.get_trace_summaries.return_value = {
            "TraceSummaries": [{"Id": "1-abc"}],
        }
        client = XRayClient()
        start = datetime.datetime(2026, 1, 1, 0, 0, tzinfo=datetime.UTC)
        end = datetime.datetime(2026, 1, 1, 2, 0, tzinfo=datetime.UTC)
        expr = 'annotation.mlflow_experiment_id = "42"'

        result = client.get_trace_summaries(start, end, filter_expression=expr)

        assert len(result) == 1
        mock_boto3_client.get_trace_summaries.assert_called_once_with(
            StartTime=start, EndTime=end, FilterExpression=expr
        )

    def test_chunked(self, mock_boto3_client):
        """Time window > 6 hours is split into multiple chunks."""
        mock_boto3_client.get_trace_summaries.side_effect = [
            {"TraceSummaries": [{"Id": "1-chunk1"}]},
            {"TraceSummaries": [{"Id": "1-chunk2"}]},
            {"TraceSummaries": [{"Id": "1-chunk3"}]},
        ]
        client = XRayClient()
        start = datetime.datetime(2026, 1, 1, 0, 0, tzinfo=datetime.UTC)
        end = datetime.datetime(2026, 1, 1, 13, 0, tzinfo=datetime.UTC)

        result = client.get_trace_summaries(start, end)

        assert len(result) == 3
        assert mock_boto3_client.get_trace_summaries.call_count == 3
        # Verify chunk boundaries
        calls = mock_boto3_client.get_trace_summaries.call_args_list
        assert calls[0].kwargs["StartTime"] == start
        assert calls[0].kwargs["EndTime"] == start + datetime.timedelta(hours=6)
        assert calls[1].kwargs["StartTime"] == start + datetime.timedelta(hours=6)
        assert calls[1].kwargs["EndTime"] == start + datetime.timedelta(hours=12)
        assert calls[2].kwargs["StartTime"] == start + datetime.timedelta(hours=12)
        assert calls[2].kwargs["EndTime"] == end

    def test_pagination(self, mock_boto3_client):
        """Single chunk with multiple pages via NextToken."""
        mock_boto3_client.get_trace_summaries.side_effect = [
            {"TraceSummaries": [{"Id": "1-page1"}], "NextToken": "token-1"},
            {"TraceSummaries": [{"Id": "1-page2"}], "NextToken": "token-2"},
            {"TraceSummaries": [{"Id": "1-page3"}]},
        ]
        client = XRayClient()
        start = datetime.datetime(2026, 1, 1, 0, 0, tzinfo=datetime.UTC)
        end = datetime.datetime(2026, 1, 1, 5, 0, tzinfo=datetime.UTC)

        result = client.get_trace_summaries(start, end)

        assert len(result) == 3
        assert mock_boto3_client.get_trace_summaries.call_count == 3
        # Verify NextToken was passed
        second_call = mock_boto3_client.get_trace_summaries.call_args_list[1]
        assert second_call.kwargs["NextToken"] == "token-1"
        third_call = mock_boto3_client.get_trace_summaries.call_args_list[2]
        assert third_call.kwargs["NextToken"] == "token-2"

    def test_empty_result(self, mock_boto3_client):
        """No summaries returned yields empty list."""
        mock_boto3_client.get_trace_summaries.return_value = {"TraceSummaries": []}
        client = XRayClient()
        start = datetime.datetime(2026, 1, 1, 0, 0, tzinfo=datetime.UTC)
        end = datetime.datetime(2026, 1, 1, 1, 0, tzinfo=datetime.UTC)

        result = client.get_trace_summaries(start, end)

        assert result == []


class TestBatchGetTraces:
    def test_small_batch(self, mock_boto3_client):
        """< 5 IDs results in a single batch call."""
        mock_boto3_client.batch_get_traces.return_value = {
            "Traces": [{"Id": "1-a"}, {"Id": "1-b"}],
        }
        client = XRayClient()

        result = client.batch_get_traces(["1-a", "1-b"])

        assert len(result) == 2
        mock_boto3_client.batch_get_traces.assert_called_once_with(TraceIds=["1-a", "1-b"])

    def test_large_batch(self, mock_boto3_client):
        """> 5 IDs are split into multiple batches of 5."""
        ids = [f"1-{i}" for i in range(12)]
        mock_boto3_client.batch_get_traces.side_effect = [
            {"Traces": [{"Id": tid} for tid in ids[0:5]]},
            {"Traces": [{"Id": tid} for tid in ids[5:10]]},
            {"Traces": [{"Id": tid} for tid in ids[10:12]]},
        ]
        client = XRayClient()

        result = client.batch_get_traces(ids)

        assert len(result) == 12
        assert mock_boto3_client.batch_get_traces.call_count == 3
        # Verify batch sizes
        calls = mock_boto3_client.batch_get_traces.call_args_list
        assert len(calls[0].kwargs["TraceIds"]) == 5
        assert len(calls[1].kwargs["TraceIds"]) == 5
        assert len(calls[2].kwargs["TraceIds"]) == 2

    def test_empty(self, mock_boto3_client):
        """Empty list returns empty without any API call."""
        client = XRayClient()

        result = client.batch_get_traces([])

        assert result == []
        mock_boto3_client.batch_get_traces.assert_not_called()

    def test_exact_batch_size(self, mock_boto3_client):
        """Exactly 5 IDs results in a single batch call."""
        ids = [f"1-{i}" for i in range(5)]
        mock_boto3_client.batch_get_traces.return_value = {
            "Traces": [{"Id": tid} for tid in ids],
        }
        client = XRayClient()

        result = client.batch_get_traces(ids)

        assert len(result) == 5
        mock_boto3_client.batch_get_traces.assert_called_once()
