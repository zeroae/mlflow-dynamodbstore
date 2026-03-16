import json

from mlflow_dynamodbstore.xray.span_converter import convert_xray_trace


def _make_trace(segments_docs, trace_id="1-abc123"):
    """Helper to build an X-Ray trace dict from a list of document dicts."""
    return {
        "Id": trace_id,
        "Duration": 1.5,
        "Segments": [{"Id": doc["id"], "Document": json.dumps(doc)} for doc in segments_docs],
    }


def _base_segment(
    seg_id="seg-001",
    name="ChatModel",
    trace_id="1-abc123",
    parent_id=None,
    start_time=1709251200.0,
    end_time=1709251201.5,
    annotations=None,
    metadata=None,
):
    doc = {
        "id": seg_id,
        "name": name,
        "trace_id": trace_id,
        "parent_id": parent_id,
        "start_time": start_time,
        "end_time": end_time,
    }
    if annotations is not None:
        doc["annotations"] = annotations
    if metadata is not None:
        doc["metadata"] = metadata
    return doc


class TestSpanConverter:
    def test_basic_conversion(self):
        """Single segment -> single Span with correct fields."""
        seg = _base_segment(
            annotations={
                "mlflow_spanType": "LLM",
                "mlflow_spanName": "ChatModel",
                "mlflow_spanStatus": "OK",
            },
        )
        spans = convert_xray_trace(_make_trace([seg]))

        assert len(spans) == 1
        span = spans[0]
        assert span["span_id"] == "seg-001"
        assert span["trace_id"] == "1-abc123"
        assert span["parent_span_id"] is None
        assert span["name"] == "ChatModel"
        assert span["status"] == "OK"
        assert span["span_type"] == "LLM"

    def test_timing_conversion(self):
        """X-Ray seconds -> MLflow nanoseconds."""
        seg = _base_segment(start_time=1709251200.0, end_time=1709251201.5)
        spans = convert_xray_trace(_make_trace([seg]))

        span = spans[0]
        assert span["start_time_ns"] == 1709251200_000000000
        assert span["end_time_ns"] == 1709251201_500000000

    def test_annotation_to_span_type(self):
        """mlflow_spanType annotation -> span type attribute."""
        seg = _base_segment(annotations={"mlflow_spanType": "RETRIEVER"})
        spans = convert_xray_trace(_make_trace([seg]))

        assert spans[0]["span_type"] == "RETRIEVER"
        # span_type should also appear in attributes via reverse config
        assert spans[0]["attributes"]["mlflow.spanType"] == "RETRIEVER"

    def test_parent_child(self):
        """Two segments with parent_id -> correct parent-child relationship."""
        parent = _base_segment(seg_id="seg-001", name="Chain")
        child = _base_segment(seg_id="seg-002", name="LLM", parent_id="seg-001")
        spans = convert_xray_trace(_make_trace([parent, child]))

        assert len(spans) == 2
        parent_span = next(s for s in spans if s["span_id"] == "seg-001")
        child_span = next(s for s in spans if s["span_id"] == "seg-002")
        assert parent_span["parent_span_id"] is None
        assert child_span["parent_span_id"] == "seg-001"

    def test_inputs_outputs(self):
        """Metadata mlflow.inputs/outputs -> span inputs/outputs."""
        inputs = {"messages": [{"role": "user", "content": "hi"}]}
        outputs = {"choices": [{"message": {"content": "hello"}}]}
        seg = _base_segment(
            metadata={"mlflow": {"inputs": inputs, "outputs": outputs}},
        )
        spans = convert_xray_trace(_make_trace([seg]))

        assert spans[0]["inputs"] == inputs
        assert spans[0]["outputs"] == outputs

    def test_empty_segments(self):
        """Trace with no segments -> empty list."""
        trace = {"Id": "1-abc123", "Duration": 0, "Segments": []}
        spans = convert_xray_trace(trace)
        assert spans == []

    def test_missing_annotations(self):
        """Segments without MLflow annotations get defaults."""
        seg = _base_segment()  # no annotations, no metadata
        spans = convert_xray_trace(_make_trace([seg]))

        span = spans[0]
        assert span["status"] == "UNSET"
        assert span["span_type"] == "UNKNOWN"
        assert span["inputs"] is None
        assert span["outputs"] is None
        assert span["attributes"] == {}
