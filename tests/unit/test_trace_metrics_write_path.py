"""Tests for the log_spans write path: individual span items, trace/span metrics, META denorm."""

import json
import time

import pytest
from mlflow.entities import TraceInfo, TraceLocation, TraceLocationType, TraceState
from mlflow.entities.assessment import AssessmentSource, Feedback
from mlflow.entities.trace_location import MlflowExperimentLocation
from moto import mock_aws


class _FakeSpan:
    def __init__(
        self,
        trace_id,
        span_id,
        name="span",
        span_type="CHAIN",
        status="OK",
        start_time_ns=1_000_000_000,
        end_time_ns=2_000_000_000,
        parent_id=None,
        attributes=None,
    ):
        self.trace_id = trace_id
        self.span_id = span_id
        self.name = name
        self.span_type = span_type
        self.status = status
        self.start_time_ns = start_time_ns
        self.end_time_ns = end_time_ns
        self.parent_id = parent_id
        self._attributes = attributes or {}

    def to_dict(self):
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "name": self.name,
            "span_type": self.span_type,
            "status": self.status,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "parent_id": self.parent_id,
            "attributes": self._attributes,
        }


def _make_trace_info(experiment_id, trace_id="tr-test"):
    return TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation(
            type=TraceLocationType.MLFLOW_EXPERIMENT,
            mlflow_experiment=MlflowExperimentLocation(experiment_id=experiment_id),
        ),
        request_time=int(time.time() * 1000),
        execution_duration=500,
        state=TraceState.OK,
        trace_metadata={},
        tags={},
    )


@mock_aws
class TestLogSpansWritePath:
    """Verify log_spans writes individual span items with correct attributes."""

    def test_creates_individual_span_items(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp")
        trace_id = "tr-span-items"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))

        tracking_store.log_spans(
            exp_id,
            [
                _FakeSpan(trace_id, "s1", name="embed", span_type="LLM"),
                _FakeSpan(trace_id, "s2", name="retrieve", span_type="RETRIEVER"),
            ],
        )

        items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix=f"T#{trace_id}#SPAN#")
        assert len(items) == 2
        span_ids = {item["SK"].split("#SPAN#")[1] for item in items}
        assert span_ids == {"s1", "s2"}

        s1 = next(i for i in items if "s1" in i["SK"])
        assert s1["name"] == "embed"
        assert s1["type"] == "LLM"
        assert s1["status"] == "OK"

    def test_span_duration_ms_computed_correctly(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp")
        trace_id = "tr-duration"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))

        # 1.5 seconds = 1500 ms
        tracking_store.log_spans(
            exp_id,
            [
                _FakeSpan(
                    trace_id,
                    "s1",
                    start_time_ns=1_000_000_000,
                    end_time_ns=2_500_000_000,
                ),
            ],
        )

        items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix=f"T#{trace_id}#SPAN#")
        assert len(items) == 1
        assert items[0]["duration_ms"] == 1500

    def test_span_extracts_model_name_and_provider(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp")
        trace_id = "tr-model"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))

        tracking_store.log_spans(
            exp_id,
            [
                _FakeSpan(
                    trace_id,
                    "s1",
                    attributes={
                        "mlflow.llm.model": json.dumps("gpt-4"),
                        "mlflow.llm.provider": "openai",
                    },
                ),
            ],
        )

        items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix=f"T#{trace_id}#SPAN#")
        assert len(items) == 1
        assert items[0]["model_name"] == "gpt-4"
        assert items[0]["model_provider"] == "openai"

    def test_span_omits_none_model_fields(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp")
        trace_id = "tr-no-model"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))

        tracking_store.log_spans(
            exp_id,
            [
                _FakeSpan(trace_id, "s1"),
            ],
        )

        items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix=f"T#{trace_id}#SPAN#")
        assert len(items) == 1
        assert "model_name" not in items[0]
        assert "model_provider" not in items[0]

    def test_meta_denormalized_on_write(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp")
        trace_id = "tr-meta"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))

        tracking_store.log_spans(
            exp_id,
            [
                _FakeSpan(trace_id, "s1", name="embed", span_type="LLM", status="OK"),
                _FakeSpan(trace_id, "s2", name="retrieve", span_type="RETRIEVER", status="ERROR"),
            ],
        )

        meta = tracking_store._table.get_item(pk=f"EXP#{exp_id}", sk=f"T#{trace_id}")
        assert set(meta["span_types"]) == {"LLM", "RETRIEVER"}
        assert set(meta["span_statuses"]) == {"OK", "ERROR"}
        assert set(meta["span_names"]) == {"embed", "retrieve"}

    def test_trace_metric_items_from_token_usage(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp")
        trace_id = "tr-tokens"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))

        tracking_store.log_spans(
            exp_id,
            [
                _FakeSpan(
                    trace_id,
                    "s1",
                    attributes={
                        "mlflow.chat.tokenUsage": json.dumps(
                            {
                                "input_tokens": 100,
                                "output_tokens": 50,
                                "total_tokens": 150,
                            }
                        ),
                    },
                ),
                _FakeSpan(
                    trace_id,
                    "s2",
                    attributes={
                        "mlflow.chat.tokenUsage": json.dumps(
                            {
                                "input_tokens": 200,
                                "output_tokens": 80,
                                "total_tokens": 280,
                            }
                        ),
                    },
                ),
            ],
        )

        items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix=f"T#{trace_id}#TMETRIC#")
        assert len(items) == 3
        metrics = {item["SK"].split("#TMETRIC#")[1]: item for item in items}
        assert metrics["input_tokens"]["value"] == 300
        assert metrics["output_tokens"]["value"] == 130
        assert metrics["total_tokens"]["value"] == 430
        # Verify key attribute is present
        for metric_key, item in metrics.items():
            assert item["key"] == metric_key

    def test_no_trace_metrics_without_token_usage(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp")
        trace_id = "tr-no-tokens"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))

        tracking_store.log_spans(
            exp_id,
            [
                _FakeSpan(trace_id, "s1"),
            ],
        )

        items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix=f"T#{trace_id}#TMETRIC#")
        assert len(items) == 0

    def test_span_metric_items_from_cost(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp")
        trace_id = "tr-cost"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))

        tracking_store.log_spans(
            exp_id,
            [
                _FakeSpan(
                    trace_id,
                    "s1",
                    attributes={
                        "mlflow.llm.cost": json.dumps(
                            {
                                "input_cost": 0.001,
                                "output_cost": 0.002,
                                "total_cost": 0.003,
                            }
                        ),
                    },
                ),
            ],
        )

        items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix=f"T#{trace_id}#SMETRIC#")
        assert len(items) == 3
        metrics = {}
        for item in items:
            # SK: T#<trace_id>#SMETRIC#<span_id>#<key>
            parts = item["SK"].split("#SMETRIC#")[1]
            span_id, key = parts.split("#", 1)
            metrics[(span_id, key)] = item
        assert float(metrics[("s1", "input_cost")]["value"]) == pytest.approx(0.001)
        assert float(metrics[("s1", "output_cost")]["value"]) == pytest.approx(0.002)
        assert float(metrics[("s1", "total_cost")]["value"]) == pytest.approx(0.003)
        # Verify key and span_id attributes are present
        for (span_id, key), item in metrics.items():
            assert item["key"] == key
            assert item["span_id"] == span_id

    def test_no_span_metrics_without_costs(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp")
        trace_id = "tr-no-cost"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))

        tracking_store.log_spans(
            exp_id,
            [
                _FakeSpan(trace_id, "s1"),
            ],
        )

        items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix=f"T#{trace_id}#SMETRIC#")
        assert len(items) == 0


def _make_feedback(trace_id, name="quality", value="good"):
    return Feedback(
        name=name,
        source=AssessmentSource(source_type="HUMAN", source_id="user1"),
        trace_id=trace_id,
        value=value,
    )


@mock_aws
class TestAssessmentDenormalization:
    def test_create_assessment_denormalizes(self, tracking_store):
        exp_id = tracking_store.create_experiment("assess-denorm-exp")
        trace_id = "tr-assess-denorm"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))
        created = tracking_store.create_assessment(_make_feedback(trace_id))
        pk = f"EXP#{exp_id}"
        sk = f"T#{trace_id}#ASSESS#{created.assessment_id}"
        item = tracking_store._table.get_item(pk=pk, sk=sk)
        assert item["name"] == "quality"
        assert item["assessment_type"] == "feedback"
        assert "created_timestamp" in item

    def test_numeric_value_from_yes(self, tracking_store):
        exp_id = tracking_store.create_experiment("assess-yes-exp")
        trace_id = "tr-assess-yes"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))
        created = tracking_store.create_assessment(_make_feedback(trace_id, value="yes"))
        pk = f"EXP#{exp_id}"
        sk = f"T#{trace_id}#ASSESS#{created.assessment_id}"
        item = tracking_store._table.get_item(pk=pk, sk=sk)
        assert item.get("numeric_value") == 1.0

    def test_numeric_value_non_numeric(self, tracking_store):
        exp_id = tracking_store.create_experiment("assess-str-exp")
        trace_id = "tr-assess-str"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))
        created = tracking_store.create_assessment(_make_feedback(trace_id, value="some text"))
        pk = f"EXP#{exp_id}"
        sk = f"T#{trace_id}#ASSESS#{created.assessment_id}"
        item = tracking_store._table.get_item(pk=pk, sk=sk)
        assert "numeric_value" not in item

    def test_update_assessment_updates_denormalized(self, tracking_store):
        exp_id = tracking_store.create_experiment("assess-update-exp")
        trace_id = "tr-assess-update"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))
        created = tracking_store.create_assessment(_make_feedback(trace_id))
        tracking_store.update_assessment(
            trace_id=trace_id,
            assessment_id=created.assessment_id,
            name="renamed",
            feedback="excellent",
        )
        pk = f"EXP#{exp_id}"
        sk = f"T#{trace_id}#ASSESS#{created.assessment_id}"
        item = tracking_store._table.get_item(pk=pk, sk=sk)
        assert item["name"] == "renamed"
        assert item["assessment_type"] == "feedback"
