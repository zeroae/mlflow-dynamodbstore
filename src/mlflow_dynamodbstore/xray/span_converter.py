from __future__ import annotations

import json
from typing import Any

from mlflow_dynamodbstore.xray.annotation_config import DEFAULT_ANNOTATION_CONFIG

# Reverse mapping: X-Ray annotation name → MLflow attribute name
_REVERSE_CONFIG = {v: k for k, v in DEFAULT_ANNOTATION_CONFIG.items()}


def convert_xray_trace(xray_trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert an X-Ray trace (from BatchGetTraces) to a list of span dicts.

    Each span dict has: span_id, trace_id, parent_span_id, name,
    start_time_ns, end_time_ns, status, span_type, inputs, outputs, attributes.

    Returns list of span dicts (not MLflow Span objects — let the caller
    construct the actual entities, since Span construction may vary across
    MLflow versions).
    """
    spans = []
    for segment_wrapper in xray_trace.get("Segments", []):
        doc_str = segment_wrapper.get("Document", "{}")
        doc = json.loads(doc_str) if isinstance(doc_str, str) else doc_str

        annotations = doc.get("annotations", {})
        metadata = doc.get("metadata", {}).get("mlflow", {})

        span = {
            "span_id": doc.get("id", ""),
            "trace_id": doc.get("trace_id", xray_trace.get("Id", "")),
            "parent_span_id": doc.get("parent_id"),
            "name": doc.get("name", ""),
            "start_time_ns": int(doc.get("start_time", 0) * 1e9),
            "end_time_ns": int(doc.get("end_time", 0) * 1e9),
            "status": annotations.get("mlflow_spanStatus", "UNSET"),
            "span_type": annotations.get("mlflow_spanType", "UNKNOWN"),
            "inputs": metadata.get("inputs"),
            "outputs": metadata.get("outputs"),
            "attributes": {},
        }

        # Map remaining annotations to attributes
        for ann_key, ann_value in annotations.items():
            mlflow_key = _REVERSE_CONFIG.get(ann_key)
            if mlflow_key and mlflow_key not in ("name", "status"):
                span["attributes"][mlflow_key] = ann_value

        spans.append(span)

    return spans
