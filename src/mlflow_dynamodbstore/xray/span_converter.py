from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlflow.entities.span import Span

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


def _stable_hex_id(s: str, length: int = 16) -> str:
    """Produce a deterministic hex ID from an arbitrary string."""
    h = hashlib.sha256(s.encode()).hexdigest()
    return h[:length]


def span_dicts_to_mlflow_spans(
    span_dicts: list[dict[str, Any]],
    trace_id: str,
) -> list[Span]:
    """Convert span dicts (from convert_xray_trace) to MLflow Span objects.

    This builds OTel ReadableSpan instances and wraps them as MLflow Span
    objects.  Span and trace IDs are hashed to valid 16-/32-char hex strings
    so that the OTel context layer accepts them.
    """
    from mlflow.entities.span import Span
    from mlflow.entities.span_status import SpanStatus
    from mlflow.tracing.constant import SpanAttributeKey
    from opentelemetry.sdk.resources import Resource as _OTelResource
    from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
    from opentelemetry.trace import SpanContext, TraceFlags

    otel_trace_id = int(_stable_hex_id(trace_id, 32), 16)

    spans: list[Span] = []
    for sd in span_dicts:
        otel_span_id = int(_stable_hex_id(sd["span_id"], 16), 16)

        parent_ctx = None
        if sd.get("parent_span_id"):
            parent_span_id = int(_stable_hex_id(sd["parent_span_id"], 16), 16)
            parent_ctx = SpanContext(
                trace_id=otel_trace_id,
                span_id=parent_span_id,
                is_remote=False,
                trace_flags=TraceFlags(1),
            )

        ctx = SpanContext(
            trace_id=otel_trace_id,
            span_id=otel_span_id,
            is_remote=False,
            trace_flags=TraceFlags(1),
        )

        # Build attributes dict with MLflow-required keys
        attrs: dict[str, Any] = {
            SpanAttributeKey.REQUEST_ID: trace_id,
            SpanAttributeKey.SPAN_TYPE: sd.get("span_type", "UNKNOWN"),
        }
        if sd.get("inputs") is not None:
            attrs[SpanAttributeKey.INPUTS] = json.dumps(sd["inputs"])
        if sd.get("outputs") is not None:
            attrs[SpanAttributeKey.OUTPUTS] = json.dumps(sd["outputs"])
        # Copy extra attributes
        for k, v in sd.get("attributes", {}).items():
            attrs[k] = v

        status = SpanStatus(sd.get("status", "UNSET"), "")

        otel_span = OTelReadableSpan(
            name=sd.get("name", ""),
            context=ctx,
            parent=parent_ctx,
            start_time=sd.get("start_time_ns", 0),
            end_time=sd.get("end_time_ns", 0),
            attributes=attrs,
            status=status.to_otel_status(),
            resource=_OTelResource.get_empty(),
            events=[],
        )
        spans.append(Span(otel_span))

    return spans
