from __future__ import annotations

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import StatusCode

from mlflow_dynamodbstore.otel.annotation_processor import AnnotationSpanProcessor
from mlflow_dynamodbstore.xray.annotation_config import DEFAULT_ANNOTATION_CONFIG


def _make_tracer(processor: AnnotationSpanProcessor):
    """Create a TracerProvider with the annotation processor and return a tracer."""
    provider = TracerProvider()
    provider.add_span_processor(processor)
    return provider.get_tracer("test")


class TestAnnotationSpanProcessor:
    def test_maps_span_type(self):
        """mlflow.spanType attribute -> mlflow_spanType annotation on OTel span."""
        processor = AnnotationSpanProcessor()
        tracer = _make_tracer(processor)

        with tracer.start_as_current_span("my-span") as span:
            span.set_attribute("mlflow.spanType", "LLM")

        # After the span ends, on_end should have added the annotation
        assert span.attributes.get("mlflow_spanType") == "LLM"

    def test_maps_span_name(self):
        """Span name -> mlflow_spanName annotation."""
        processor = AnnotationSpanProcessor()
        tracer = _make_tracer(processor)

        with tracer.start_as_current_span("predict") as span:
            pass

        assert span.attributes.get("mlflow_spanName") == "predict"

    def test_maps_span_status(self):
        """Span status -> mlflow_spanStatus annotation."""
        processor = AnnotationSpanProcessor()
        tracer = _make_tracer(processor)

        with tracer.start_as_current_span("my-span") as span:
            span.set_status(StatusCode.OK)

        assert span.attributes.get("mlflow_spanStatus") == "OK"

    def test_configurable_mapping(self):
        """Custom mapping can be provided."""
        custom_config = {"my.custom.attr": "custom_annotation"}
        processor = AnnotationSpanProcessor(config=custom_config)
        tracer = _make_tracer(processor)

        with tracer.start_as_current_span("my-span") as span:
            span.set_attribute("my.custom.attr", "hello")

        assert span.attributes.get("custom_annotation") == "hello"

    def test_unmapped_attributes_pass_through(self):
        """Attributes not in mapping are left unchanged."""
        processor = AnnotationSpanProcessor()
        tracer = _make_tracer(processor)

        with tracer.start_as_current_span("my-span") as span:
            span.set_attribute("some.other.attr", "value123")

        # The original attribute should be present and unchanged
        assert span.attributes.get("some.other.attr") == "value123"
        # No annotation should have been created for it
        annotation_names = set(DEFAULT_ANNOTATION_CONFIG.values())
        for key in span.attributes:
            if key == "some.other.attr":
                continue
            # Any other keys should only be from the annotation config
            # (name and status are always mapped)
            if key in annotation_names:
                continue
            raise AssertionError(f"Unexpected attribute: {key}")

    def test_on_start_is_noop(self):
        """on_start should not modify the span's attributes."""
        processor = AnnotationSpanProcessor()
        tracer = _make_tracer(processor)

        span = tracer.start_span("my-span")
        # After on_start, no annotation attributes should be present yet
        annotation_names = set(DEFAULT_ANNOTATION_CONFIG.values())
        for key in span.attributes or {}:
            assert key not in annotation_names, f"on_start should not add annotation {key}"
        span.end()

    def test_missing_attribute_not_annotated(self):
        """If a configured MLflow attribute is absent, no annotation is created."""
        # Config maps 'mlflow.chat_model' but we never set it
        processor = AnnotationSpanProcessor()
        tracer = _make_tracer(processor)

        with tracer.start_as_current_span("my-span") as span:
            pass

        assert "mlflow_chatModel" not in span.attributes

    def test_shutdown_is_safe(self):
        """shutdown() should be callable without error."""
        processor = AnnotationSpanProcessor()
        processor.shutdown()  # no-op, should not raise

    def test_force_flush_returns_true(self):
        processor = AnnotationSpanProcessor()
        assert processor.force_flush() is True
