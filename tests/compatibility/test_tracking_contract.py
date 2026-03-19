"""Phase 1: Contract fidelity tests for DynamoDB vs SqlAlchemy tracking stores.

Focused on operations that have historically produced contract mismatches:
Decimal vs float, missing proto fields, entity type naming differences.
"""

import time

from tests.compatibility.comparison import assert_entities_match
from tests.compatibility.field_policy import EXPERIMENT, RUN_INFO


def test_create_and_get_experiment(tracking_stores):
    """Experiment fields must match between backends."""
    sql_id = tracking_stores.sql.create_experiment("test-exp")
    ddb_id = tracking_stores.ddb.create_experiment("test-exp")

    sql_exp = tracking_stores.sql.get_experiment(sql_id)
    ddb_exp = tracking_stores.ddb.get_experiment(ddb_id)
    assert_entities_match(sql_exp, ddb_exp, EXPERIMENT)


def test_search_experiments(tracking_stores):
    """Search should return matching experiments."""
    from mlflow.entities import ViewType

    tracking_stores.sql.create_experiment("search-a")
    tracking_stores.sql.create_experiment("search-b")
    tracking_stores.ddb.create_experiment("search-a")
    tracking_stores.ddb.create_experiment("search-b")

    sql_results = tracking_stores.sql.search_experiments(view_type=ViewType.ALL)
    ddb_results = tracking_stores.ddb.search_experiments(view_type=ViewType.ALL)

    sql_names = sorted([e.name for e in sql_results if e.name.startswith("search-")])
    ddb_names = sorted([e.name for e in ddb_results if e.name.startswith("search-")])
    assert sql_names == ddb_names


def test_create_run_and_log_metrics(tracking_stores):
    """Run info and logged metrics must match."""
    from mlflow.entities import Metric

    sql_exp_id = tracking_stores.sql.create_experiment("metric-exp")
    ddb_exp_id = tracking_stores.ddb.create_experiment("metric-exp")

    now = int(time.time() * 1000)
    sql_run = tracking_stores.sql.create_run(sql_exp_id, "user1", now, [], "test-run")
    ddb_run = tracking_stores.ddb.create_run(ddb_exp_id, "user1", now, [], "test-run")
    assert_entities_match(sql_run.info, ddb_run.info, RUN_INFO)

    # Log a metric — this is where Decimal vs float bugs live
    sql_metric = Metric("accuracy", 0.95, 1000, 0)
    ddb_metric = Metric("accuracy", 0.95, 1000, 0)
    tracking_stores.sql.log_metric(sql_run.info.run_id, sql_metric)
    tracking_stores.ddb.log_metric(ddb_run.info.run_id, ddb_metric)

    sql_run_data = tracking_stores.sql.get_run(sql_run.info.run_id)
    ddb_run_data = tracking_stores.ddb.get_run(ddb_run.info.run_id)

    assert len(sql_run_data.data.metrics) == len(ddb_run_data.data.metrics)
    for key in sql_run_data.data.metrics:
        sql_val = sql_run_data.data.metrics[key]
        ddb_val = ddb_run_data.data.metrics[key]
        assert type(sql_val) is type(ddb_val), (
            f"Metric '{key}': type mismatch sql={type(sql_val).__name__} "
            f"ddb={type(ddb_val).__name__}"
        )
        assert sql_val == ddb_val


def test_log_params_and_tags(tracking_stores):
    """Params and tags must round-trip identically."""
    from mlflow.entities import Param, RunTag

    sql_exp_id = tracking_stores.sql.create_experiment("param-exp")
    ddb_exp_id = tracking_stores.ddb.create_experiment("param-exp")

    now = int(time.time() * 1000)
    sql_run = tracking_stores.sql.create_run(sql_exp_id, "user1", now, [], "test-run")
    ddb_run = tracking_stores.ddb.create_run(ddb_exp_id, "user1", now, [], "test-run")

    tracking_stores.sql.log_param(sql_run.info.run_id, Param("lr", "0.001"))
    tracking_stores.ddb.log_param(ddb_run.info.run_id, Param("lr", "0.001"))

    tracking_stores.sql.set_tag(sql_run.info.run_id, RunTag("env", "prod"))
    tracking_stores.ddb.set_tag(ddb_run.info.run_id, RunTag("env", "prod"))

    sql_data = tracking_stores.sql.get_run(sql_run.info.run_id)
    ddb_data = tracking_stores.ddb.get_run(ddb_run.info.run_id)

    assert sql_data.data.params == ddb_data.data.params
    sql_tags = {k: v for k, v in sql_data.data.tags.items() if not k.startswith("mlflow.")}
    ddb_tags = {k: v for k, v in ddb_data.data.tags.items() if not k.startswith("mlflow.")}
    assert sql_tags == ddb_tags
