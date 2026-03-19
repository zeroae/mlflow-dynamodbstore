"""Phase 1: Contract fidelity tests for DynamoDB vs SqlAlchemy registry stores."""

import pytest

from tests.compatibility.comparison import assert_entities_match
from tests.compatibility.field_policy import MODEL_VERSION, REGISTERED_MODEL


def test_create_and_get_registered_model(registry_stores):
    """Model returned by create/get must match between backends."""
    sql_model = registry_stores.sql.create_registered_model("test-model")
    ddb_model = registry_stores.ddb.create_registered_model("test-model")
    assert_entities_match(sql_model, ddb_model, REGISTERED_MODEL)

    sql_get = registry_stores.sql.get_registered_model("test-model")
    ddb_get = registry_stores.ddb.get_registered_model("test-model")
    assert_entities_match(sql_get, ddb_get, REGISTERED_MODEL)


def test_update_registered_model(registry_stores):
    """Updated model fields must match."""
    registry_stores.sql.create_registered_model("upd-model")
    registry_stores.ddb.create_registered_model("upd-model")

    sql_upd = registry_stores.sql.update_registered_model("upd-model", "new desc")
    ddb_upd = registry_stores.ddb.update_registered_model("upd-model", "new desc")
    assert_entities_match(sql_upd, ddb_upd, REGISTERED_MODEL)


def test_delete_registered_model(registry_stores):
    """Deleting a model should not raise and model should be gone."""
    from mlflow.exceptions import MlflowException

    registry_stores.sql.create_registered_model("del-model")
    registry_stores.ddb.create_registered_model("del-model")

    registry_stores.sql.delete_registered_model("del-model")
    registry_stores.ddb.delete_registered_model("del-model")

    with pytest.raises(MlflowException):
        registry_stores.sql.get_registered_model("del-model")
    with pytest.raises(MlflowException):
        registry_stores.ddb.get_registered_model("del-model")


def test_search_registered_models(registry_stores):
    """Search results must return same models (ignoring order)."""
    for name in ["search-a", "search-b", "search-c"]:
        registry_stores.sql.create_registered_model(name)
        registry_stores.ddb.create_registered_model(name)

    sql_results = registry_stores.sql.search_registered_models()
    ddb_results = registry_stores.ddb.search_registered_models()

    assert len(sql_results) == len(ddb_results)
    sql_sorted = sorted(sql_results, key=lambda m: m.name)
    ddb_sorted = sorted(ddb_results, key=lambda m: m.name)
    for sql_m, ddb_m in zip(sql_sorted, ddb_sorted):
        assert_entities_match(sql_m, ddb_m, REGISTERED_MODEL)


def test_create_and_get_model_version(registry_stores):
    """Model version fields must match between backends."""
    registry_stores.sql.create_registered_model("mv-model")
    registry_stores.ddb.create_registered_model("mv-model")

    sql_mv = registry_stores.sql.create_model_version("mv-model", "s3://source", "run123")
    ddb_mv = registry_stores.ddb.create_model_version("mv-model", "s3://source", "run123")
    assert_entities_match(sql_mv, ddb_mv, MODEL_VERSION)


def test_set_and_get_registered_model_tag(registry_stores):
    """Tags must round-trip identically."""
    from mlflow.entities.model_registry import RegisteredModelTag

    registry_stores.sql.create_registered_model("tag-model")
    registry_stores.ddb.create_registered_model("tag-model")

    tag = RegisteredModelTag("env", "prod")
    registry_stores.sql.set_registered_model_tag("tag-model", tag)
    registry_stores.ddb.set_registered_model_tag("tag-model", tag)

    sql_model = registry_stores.sql.get_registered_model("tag-model")
    ddb_model = registry_stores.ddb.get_registered_model("tag-model")
    assert_entities_match(sql_model, ddb_model, REGISTERED_MODEL)


def test_set_registered_model_alias(registry_stores):
    """Alias operations must match."""
    registry_stores.sql.create_registered_model("alias-model")
    registry_stores.ddb.create_registered_model("alias-model")

    registry_stores.sql.create_model_version("alias-model", "s3://src", "run1")
    registry_stores.ddb.create_model_version("alias-model", "s3://src", "run1")

    registry_stores.sql.set_registered_model_alias("alias-model", "champion", "1")
    registry_stores.ddb.set_registered_model_alias("alias-model", "champion", "1")

    sql_mv = registry_stores.sql.get_model_version_by_alias("alias-model", "champion")
    ddb_mv = registry_stores.ddb.get_model_version_by_alias("alias-model", "champion")
    assert_entities_match(sql_mv, ddb_mv, MODEL_VERSION)
