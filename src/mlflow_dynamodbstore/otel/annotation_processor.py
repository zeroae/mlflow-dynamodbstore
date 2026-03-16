from __future__ import annotations

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from mlflow_dynamodbstore.xray.annotation_config import DEFAULT_ANNOTATION_CONFIG


class AnnotationSpanProcessor(SpanProcessor):
    """OTel SpanProcessor that maps MLflow span attributes to X-Ray annotations.

    X-Ray annotations are indexed key-value pairs used for filtering traces.
    This processor copies configured MLflow span attributes into annotation
    attributes on the span so they are available for X-Ray indexing.

    The processor works in ``on_end`` by reading final attribute values and
    writing the corresponding annotation attributes onto the span's internal
    ``BoundedAttributes``. At ``on_end`` time the attributes are still mutable
    (the ``_immutable`` flag has not been set), so direct item assignment works.

    For the special keys ``"name"`` and ``"status"`` in the config, the
    processor reads ``span.name`` and ``span.status.status_code.name``
    respectively, rather than looking them up in ``span.attributes``.
    """

    def __init__(self, config: dict[str, str] | None = None):
        self._config = config if config is not None else DEFAULT_ANNOTATION_CONFIG

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        """No-op on start."""

    def on_end(self, span: ReadableSpan) -> None:
        """Add X-Ray annotation attributes from MLflow attributes.

        Writes annotation key-value pairs into the span's internal attributes
        collection. This runs before downstream processors/exporters see the
        span, so annotations are available for X-Ray indexing.
        """
        attrs = span._attributes  # noqa: SLF001
        if attrs is None:
            return

        for mlflow_attr, annotation_name in self._config.items():
            if mlflow_attr == "name":
                attrs[annotation_name] = span.name  # type: ignore[index]
            elif mlflow_attr == "status":
                attrs[annotation_name] = span.status.status_code.name  # type: ignore[index]
            else:
                value = (span.attributes or {}).get(mlflow_attr)
                if value is not None:
                    attrs[annotation_name] = str(value)  # type: ignore[index]

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
