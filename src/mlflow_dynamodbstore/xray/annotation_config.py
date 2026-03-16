# Default mapping: MLflow span attribute -> X-Ray annotation name
# X-Ray annotations must use only alphanumeric + underscore
DEFAULT_ANNOTATION_CONFIG: dict[str, str] = {
    "mlflow.spanType": "mlflow_spanType",
    "name": "mlflow_spanName",  # span name
    "status": "mlflow_spanStatus",  # span status
    "mlflow.chat_model": "mlflow_chatModel",
    "mlflow.invocation_params.model_name": "mlflow_modelName",
}


def get_xray_annotation_name(
    mlflow_attr: str,
    config: dict[str, str] | None = None,
) -> str | None:
    """Look up X-Ray annotation name for an MLflow span attribute."""
    cfg = config if config is not None else DEFAULT_ANNOTATION_CONFIG
    return cfg.get(mlflow_attr)
