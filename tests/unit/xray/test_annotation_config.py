from mlflow_dynamodbstore.xray.annotation_config import (
    DEFAULT_ANNOTATION_CONFIG,
    get_xray_annotation_name,
)


class TestAnnotationConfig:
    def test_default_config_maps_span_type(self):
        assert DEFAULT_ANNOTATION_CONFIG["mlflow.spanType"] == "mlflow_spanType"

    def test_default_config_maps_span_name(self):
        assert DEFAULT_ANNOTATION_CONFIG["name"] == "mlflow_spanName"

    def test_default_config_maps_span_status(self):
        assert DEFAULT_ANNOTATION_CONFIG["status"] == "mlflow_spanStatus"

    def test_default_config_maps_model_name(self):
        # For LLM spans, mlflow.chat_model -> mlflow_chatModel annotation
        assert "mlflow.chat_model" in DEFAULT_ANNOTATION_CONFIG

    def test_get_xray_annotation_name_mapped(self):
        assert get_xray_annotation_name("mlflow.spanType") == "mlflow_spanType"

    def test_get_xray_annotation_name_unmapped(self):
        assert get_xray_annotation_name("unknown.attr") is None

    def test_custom_config(self):
        custom = {"my.attr": "myAnnotation"}
        assert get_xray_annotation_name("my.attr", config=custom) == "myAnnotation"
