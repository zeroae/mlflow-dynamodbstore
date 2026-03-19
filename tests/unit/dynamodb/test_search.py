from __future__ import annotations

import pytest
from moto import mock_aws

from mlflow_dynamodbstore.dynamodb.search import (
    FilterPredicate,
    QueryPlan,
    execute_query,
    parse_experiment_filter,
    parse_run_filter,
)


class TestRunFilterParser:
    def test_empty_filter(self):
        predicates = parse_run_filter("")
        assert predicates == []

    def test_attribute_equals(self):
        predicates = parse_run_filter("status = 'FINISHED'")
        assert len(predicates) == 1
        assert predicates[0].field_type == "attribute"
        assert predicates[0].key == "status"
        assert predicates[0].op == "="
        assert predicates[0].value == "FINISHED"

    def test_metric_comparison(self):
        predicates = parse_run_filter("metric.accuracy > 0.9")
        assert predicates[0].field_type == "metric"
        assert predicates[0].key == "accuracy"
        assert predicates[0].op == ">"
        assert predicates[0].value == 0.9

    def test_param_like(self):
        predicates = parse_run_filter("param.model LIKE '%transformer%'")
        assert predicates[0].field_type == "param"
        assert predicates[0].op == "LIKE"

    def test_tag_equals(self):
        predicates = parse_run_filter("tag.env = 'prod'")
        assert predicates[0].field_type == "tag"

    def test_compound_and(self):
        predicates = parse_run_filter("metric.acc > 0.9 AND param.lr = '0.01'")
        assert len(predicates) == 2

    def test_dataset_filter(self):
        predicates = parse_run_filter("dataset.name = 'my_data'")
        assert predicates[0].field_type == "dataset"

    def test_run_name_like(self):
        predicates = parse_run_filter("run_name LIKE '%pipeline%'")
        assert predicates[0].field_type == "attribute"
        assert predicates[0].key == "run_name"

    def test_metric_value_is_float(self):
        predicates = parse_run_filter("metric.loss < 0.5")
        assert isinstance(predicates[0].value, float)

    def test_none_filter(self):
        predicates = parse_run_filter(None)
        assert predicates == []

    def test_filter_predicate_immutable(self):
        predicates = parse_run_filter("status = 'FINISHED'")
        p = predicates[0]
        assert isinstance(p, FilterPredicate)
        # frozen dataclass - cannot modify
        import pytest

        with pytest.raises((AttributeError, TypeError)):
            p.key = "other"  # type: ignore[misc]


class TestRunQueryPlanner:
    def test_no_filter_uses_base_sk(self):
        from mlflow.entities import ViewType

        from mlflow_dynamodbstore.dynamodb.search import plan_run_query

        plan = plan_run_query(predicates=[], order_by=None, view_type=ViewType.ACTIVE_ONLY)
        assert plan.index == "lsi1"  # lifecycle filter
        assert plan.sk_prefix == "active#"

    def test_status_filter_uses_lsi3(self):
        from mlflow.entities import ViewType

        from mlflow_dynamodbstore.dynamodb.search import plan_run_query

        preds = parse_run_filter("status = 'FINISHED'")
        plan = plan_run_query(predicates=preds, order_by=None, view_type=ViewType.ACTIVE_ONLY)
        assert plan.index == "lsi3"
        assert plan.sk_prefix == "FINISHED#"

    def test_metric_order_uses_rank(self):
        from mlflow.entities import ViewType

        from mlflow_dynamodbstore.dynamodb.search import plan_run_query

        plan = plan_run_query(
            predicates=[], order_by=["metric.accuracy DESC"], view_type=ViewType.ACTIVE_ONLY
        )
        assert plan.strategy == "rank"
        assert plan.rank_key == "accuracy"

    def test_dataset_filter_uses_dlink(self):
        from mlflow.entities import ViewType

        from mlflow_dynamodbstore.dynamodb.search import plan_run_query

        preds = parse_run_filter("dataset.name = 'my_data'")
        plan = plan_run_query(predicates=preds, order_by=None, view_type=ViewType.ACTIVE_ONLY)
        assert plan.strategy == "dlink"

    def test_run_name_like_uses_fts(self):
        from mlflow.entities import ViewType

        from mlflow_dynamodbstore.dynamodb.search import plan_run_query

        preds = parse_run_filter("run_name LIKE '%pipeline%'")
        plan = plan_run_query(predicates=preds, order_by=None, view_type=ViewType.ACTIVE_ONLY)
        assert plan.strategy == "fts"

    def test_tag_filter_denormalized_uses_filter_expression(self):
        from mlflow.entities import ViewType

        from mlflow_dynamodbstore.dynamodb.search import plan_run_query

        preds = parse_run_filter("tag.mlflow.user = 'alice'")
        plan = plan_run_query(
            predicates=preds,
            order_by=None,
            view_type=ViewType.ACTIVE_ONLY,
            denormalized_patterns=["mlflow.*"],
        )
        assert any(fe for fe in plan.filter_expressions)

    def test_order_by_end_time_uses_lsi2(self):
        from mlflow.entities import ViewType

        from mlflow_dynamodbstore.dynamodb.search import plan_run_query

        plan = plan_run_query(
            predicates=[], order_by=["end_time ASC"], view_type=ViewType.ACTIVE_ONLY
        )
        assert plan.index == "lsi2"

    def test_order_by_duration_uses_lsi5(self):
        from mlflow.entities import ViewType

        from mlflow_dynamodbstore.dynamodb.search import plan_run_query

        plan = plan_run_query(
            predicates=[], order_by=["duration DESC"], view_type=ViewType.ACTIVE_ONLY
        )
        assert plan.index == "lsi5"


class TestQueryExecutor:
    @pytest.fixture
    def table(self):
        """Create a DynamoDB table with the full schema for testing."""
        with mock_aws():
            from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists
            from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable

            ensure_stack_exists("test-table", "us-east-1")
            table = DynamoDBTable("test-table", "us-east-1")
            yield table

    def test_index_strategy_basic(self, table):
        """Index strategy returns META items matching the LSI sk_prefix."""
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "R#01JR",
                "run_id": "01JR",
                "experiment_id": "01JQ",
                "status": "RUNNING",
                "lifecycle_stage": "active",
                "lsi1sk": "active#01JR",
            }
        )
        plan = QueryPlan(strategy="index", index="lsi1", sk_prefix="active#", scan_forward=False)
        items, token = execute_query(table, plan, pk="EXP#01JQ", max_results=10)
        assert len(items) == 1
        assert items[0]["run_id"] == "01JR"
        assert token is None

    def test_index_strategy_pagination(self, table):
        """Index strategy paginates when results exceed max_results."""
        for i in range(5):
            table.put_item(
                {
                    "PK": "EXP#01JQ",
                    "SK": f"R#RUN{i:02d}",
                    "run_id": f"RUN{i:02d}",
                    "experiment_id": "01JQ",
                    "lsi1sk": f"active#RUN{i:02d}",
                }
            )
        plan = QueryPlan(strategy="index", index="lsi1", sk_prefix="active#", scan_forward=True)
        items, token = execute_query(table, plan, pk="EXP#01JQ", max_results=3)
        assert len(items) == 3
        assert token is not None

        # Fetch next page
        items2, token2 = execute_query(table, plan, pk="EXP#01JQ", max_results=3, page_token=token)
        assert len(items2) == 2
        assert token2 is None

    def test_index_strategy_no_results(self, table):
        """Index strategy returns empty list when no items match."""
        plan = QueryPlan(strategy="index", index="lsi1", sk_prefix="active#", scan_forward=False)
        items, token = execute_query(table, plan, pk="EXP#EMPTY", max_results=10)
        assert items == []
        assert token is None

    def test_rank_strategy(self, table):
        """Rank strategy fetches RANK items and resolves to META items."""
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "R#RUN1",
                "run_id": "RUN1",
                "experiment_id": "01JQ",
                "lsi1sk": "active#RUN1",
            }
        )
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "R#RUN2",
                "run_id": "RUN2",
                "experiment_id": "01JQ",
                "lsi1sk": "active#RUN2",
            }
        )
        # RANK items with inverted metric values (lower = higher actual)
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "RANK#m#accuracy#9999999899.8000#RUN1",
                "key": "accuracy",
                "run_id": "RUN1",
            }
        )
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "RANK#m#accuracy#9999999899.9500#RUN2",
                "key": "accuracy",
                "run_id": "RUN2",
            }
        )
        plan = QueryPlan(
            strategy="rank",
            index=None,
            sk_prefix=None,
            scan_forward=True,
            rank_key="accuracy",
        )
        items, token = execute_query(table, plan, pk="EXP#01JQ", max_results=10)
        assert len(items) == 2
        # scan_forward=True means ascending SK order:
        # RUN1 (9899.8000) comes before RUN2 (9899.9500)
        assert items[0]["run_id"] == "RUN1"
        assert items[1]["run_id"] == "RUN2"

    def test_rank_strategy_no_rank_items(self, table):
        """Rank strategy returns empty when no RANK items exist."""
        plan = QueryPlan(
            strategy="rank",
            index=None,
            sk_prefix=None,
            scan_forward=True,
            rank_key="nonexistent",
        )
        items, token = execute_query(table, plan, pk="EXP#01JQ", max_results=10)
        assert items == []

    def test_dlink_strategy(self, table):
        """DLINK strategy resolves dataset links to META items."""
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "R#RUN1",
                "run_id": "RUN1",
                "experiment_id": "01JQ",
            }
        )
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "DLINK#my_data#abc123#RUN1",
                "run_id": "RUN1",
            }
        )
        preds = [
            FilterPredicate(field_type="dataset", key="name", op="=", value="my_data"),
            FilterPredicate(field_type="dataset", key="digest", op="=", value="abc123"),
        ]
        plan = QueryPlan(strategy="dlink", index=None, sk_prefix=None, scan_forward=True)
        items, token = execute_query(table, plan, pk="EXP#01JQ", max_results=10, predicates=preds)
        assert len(items) == 1
        assert items[0]["run_id"] == "RUN1"

    def test_dlink_strategy_name_only(self, table):
        """DLINK strategy works with only dataset name."""
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "R#RUN1",
                "run_id": "RUN1",
                "experiment_id": "01JQ",
            }
        )
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "DLINK#my_data#digest1#RUN1",
                "run_id": "RUN1",
            }
        )
        preds = [
            FilterPredicate(field_type="dataset", key="name", op="=", value="my_data"),
        ]
        plan = QueryPlan(strategy="dlink", index=None, sk_prefix=None, scan_forward=True)
        items, token = execute_query(table, plan, pk="EXP#01JQ", max_results=10, predicates=preds)
        assert len(items) == 1

    def test_fts_strategy_word_match(self, table):
        """FTS strategy finds items via word-level tokens."""
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "R#RUN1",
                "run_id": "RUN1",
                "experiment_id": "01JQ",
                "lsi1sk": "active#RUN1",
            }
        )
        # The stemmed form of "pipeline" is "pipelin"
        table.put_item({"PK": "EXP#01JQ", "SK": "FTS#W#R#pipelin#RUN1"})
        plan = QueryPlan(
            strategy="fts",
            index=None,
            sk_prefix=None,
            scan_forward=True,
            fts_query="pipeline",
        )
        items, token = execute_query(table, plan, pk="EXP#01JQ", max_results=10)
        assert len(items) == 1
        assert items[0]["run_id"] == "RUN1"

    def test_fts_strategy_no_match(self, table):
        """FTS strategy returns empty when no tokens match."""
        plan = QueryPlan(
            strategy="fts",
            index=None,
            sk_prefix=None,
            scan_forward=True,
            fts_query="nonexistent",
        )
        items, token = execute_query(table, plan, pk="EXP#01JQ", max_results=10)
        assert items == []

    def test_fts_strategy_trigram_fallback(self, table):
        """FTS strategy falls back to trigrams when word tokens yield no results."""
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "R#RUN1",
                "run_id": "RUN1",
                "experiment_id": "01JQ",
            }
        )
        # "xy" is too short for word tokens but we add trigram tokens for "xyz"
        table.put_item({"PK": "EXP#01JQ", "SK": "FTS#3#R#xyz#RUN1"})
        plan = QueryPlan(
            strategy="fts",
            index=None,
            sk_prefix=None,
            scan_forward=True,
            fts_query="xyz",
        )
        items, token = execute_query(table, plan, pk="EXP#01JQ", max_results=10)
        assert len(items) == 1
        assert items[0]["run_id"] == "RUN1"

    def test_denormalized_tag_filter(self, table):
        """Filter expressions apply denormalized tag checks on META items."""
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "R#RUN1",
                "run_id": "RUN1",
                "experiment_id": "01JQ",
                "lsi1sk": "active#RUN1",
                "tags": {"mlflow.user": "alice"},
            }
        )
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "R#RUN2",
                "run_id": "RUN2",
                "experiment_id": "01JQ",
                "lsi1sk": "active#RUN2",
                "tags": {"mlflow.user": "bob"},
            }
        )
        preds = [
            FilterPredicate(field_type="tag", key="mlflow.user", op="=", value="alice"),
        ]
        plan = QueryPlan(
            strategy="index",
            index="lsi1",
            sk_prefix="active#",
            scan_forward=False,
            filter_expressions=["tags.#mlflow_user = :mlflow_userv"],
        )
        items, token = execute_query(table, plan, pk="EXP#01JQ", max_results=10, predicates=preds)
        assert len(items) == 1
        assert items[0]["run_id"] == "RUN1"

    def test_post_filter_tag_lookup(self, table):
        """Post-filters look up sub-items for non-denormalized tags."""
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "R#RUN1",
                "run_id": "RUN1",
                "experiment_id": "01JQ",
                "lsi1sk": "active#RUN1",
            }
        )
        # Tag sub-item
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "R#RUN1#TAG#env",
                "value": "prod",
            }
        )
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "R#RUN2",
                "run_id": "RUN2",
                "experiment_id": "01JQ",
                "lsi1sk": "active#RUN2",
            }
        )
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "R#RUN2#TAG#env",
                "value": "dev",
            }
        )
        plan = QueryPlan(
            strategy="index",
            index="lsi1",
            sk_prefix="active#",
            scan_forward=False,
            post_filters=[
                FilterPredicate(field_type="tag", key="env", op="=", value="prod"),
            ],
        )
        items, token = execute_query(table, plan, pk="EXP#01JQ", max_results=10)
        assert len(items) == 1
        assert items[0]["run_id"] == "RUN1"

    def test_post_filter_metric_comparison(self, table):
        """Post-filters support metric comparison operators."""
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "R#RUN1",
                "run_id": "RUN1",
                "experiment_id": "01JQ",
                "lsi1sk": "active#RUN1",
            }
        )
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "R#RUN1#METRIC#accuracy",
                "value": "0.95",
            }
        )
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "R#RUN2",
                "run_id": "RUN2",
                "experiment_id": "01JQ",
                "lsi1sk": "active#RUN2",
            }
        )
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "R#RUN2#METRIC#accuracy",
                "value": "0.5",
            }
        )
        plan = QueryPlan(
            strategy="index",
            index="lsi1",
            sk_prefix="active#",
            scan_forward=False,
            post_filters=[
                FilterPredicate(field_type="metric", key="accuracy", op=">", value=0.9),
            ],
        )
        items, token = execute_query(table, plan, pk="EXP#01JQ", max_results=10)
        assert len(items) == 1
        assert items[0]["run_id"] == "RUN1"

    def test_post_filter_attribute(self, table):
        """Post-filters check attributes directly on META items."""
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "R#RUN1",
                "run_id": "RUN1",
                "experiment_id": "01JQ",
                "lsi1sk": "active#RUN1",
                "run_name": "training-v1",
            }
        )
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "R#RUN2",
                "run_id": "RUN2",
                "experiment_id": "01JQ",
                "lsi1sk": "active#RUN2",
                "run_name": "evaluation-v1",
            }
        )
        plan = QueryPlan(
            strategy="index",
            index="lsi1",
            sk_prefix="active#",
            scan_forward=False,
            post_filters=[
                FilterPredicate(
                    field_type="attribute",
                    key="run_name",
                    op="=",
                    value="training-v1",
                ),
            ],
        )
        items, token = execute_query(table, plan, pk="EXP#01JQ", max_results=10)
        assert len(items) == 1
        assert items[0]["run_id"] == "RUN1"

    def test_fts_strategy_and_semantics(self, table):
        """FTS with multiple word tokens uses AND semantics."""
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "R#RUN1",
                "run_id": "RUN1",
                "experiment_id": "01JQ",
            }
        )
        table.put_item(
            {
                "PK": "EXP#01JQ",
                "SK": "R#RUN2",
                "run_id": "RUN2",
                "experiment_id": "01JQ",
            }
        )
        # RUN1 has both tokens
        table.put_item({"PK": "EXP#01JQ", "SK": "FTS#W#R#train#RUN1"})
        table.put_item({"PK": "EXP#01JQ", "SK": "FTS#W#R#model#RUN1"})
        # RUN2 has only one token
        table.put_item({"PK": "EXP#01JQ", "SK": "FTS#W#R#train#RUN2"})

        plan = QueryPlan(
            strategy="fts",
            index=None,
            sk_prefix=None,
            scan_forward=True,
            fts_query="training model",
        )
        items, token = execute_query(table, plan, pk="EXP#01JQ", max_results=10)
        # Only RUN1 has both "train" (stemmed from "training") and "model"
        assert len(items) == 1
        assert items[0]["run_id"] == "RUN1"


class TestExperimentFilterParser:
    def test_name_like(self):
        predicates = parse_experiment_filter("name LIKE 'prod%'")
        assert predicates[0].field_type == "attribute"
        assert predicates[0].key == "name"
        assert predicates[0].op == "LIKE"

    def test_tag_filter(self):
        predicates = parse_experiment_filter("tag.team = 'ml'")
        assert predicates[0].field_type == "tag"

    def test_empty_filter(self):
        predicates = parse_experiment_filter("")
        assert predicates == []

    def test_none_filter(self):
        predicates = parse_experiment_filter(None)
        assert predicates == []
