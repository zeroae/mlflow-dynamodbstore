"""Integration tests for search_runs via moto HTTP server."""

import pytest
from mlflow.entities import Metric, RunTag, ViewType


@pytest.mark.integration
class TestSearchRunsIntegration:
    def test_search_no_filter(self, tracking_store):
        exp_id = tracking_store.create_experiment("search-test", artifact_location="s3://b")
        for i in range(5):
            tracking_store.create_run(
                exp_id, user_id="u", start_time=1000 + i, tags=[], run_name=f"run-{i}"
            )
        runs, _ = tracking_store._search_runs([exp_id], "", ViewType.ACTIVE_ONLY, 100, None, None)
        assert len(runs) == 5

    def test_search_by_status(self, tracking_store):
        exp_id = tracking_store.create_experiment("status-test", artifact_location="s3://b")
        r1 = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r1")
        tracking_store.create_run(exp_id, user_id="u", start_time=1001, tags=[], run_name="r2")
        tracking_store.update_run_info(r1.info.run_id, "FINISHED", end_time=2000, run_name="r1")
        runs, _ = tracking_store._search_runs(
            [exp_id], "status = 'FINISHED'", ViewType.ACTIVE_ONLY, 100, None, None
        )
        assert len(runs) == 1
        assert runs[0].info.run_id == r1.info.run_id

    def test_search_order_by_metric(self, tracking_store):
        exp_id = tracking_store.create_experiment("metric-test", artifact_location="s3://b")
        r1 = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r1")
        r2 = tracking_store.create_run(exp_id, user_id="u", start_time=1001, tags=[], run_name="r2")
        r3 = tracking_store.create_run(exp_id, user_id="u", start_time=1002, tags=[], run_name="r3")
        tracking_store.log_batch(
            r1.info.run_id, metrics=[Metric("acc", 0.7, 0, 0)], params=[], tags=[]
        )
        tracking_store.log_batch(
            r2.info.run_id, metrics=[Metric("acc", 0.95, 0, 0)], params=[], tags=[]
        )
        tracking_store.log_batch(
            r3.info.run_id, metrics=[Metric("acc", 0.8, 0, 0)], params=[], tags=[]
        )
        runs, _ = tracking_store._search_runs(
            [exp_id], "", ViewType.ACTIVE_ONLY, 100, ["metric.acc ASC"], None
        )
        # Verify that ordering returns all 3 runs and they are sorted by metric value
        assert len(runs) == 3
        values = [float(r.data.metrics.get("acc", 0)) for r in runs]
        assert values == sorted(values)

    def test_search_by_tag(self, tracking_store):
        exp_id = tracking_store.create_experiment("tag-test", artifact_location="s3://b")
        r1 = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r1")
        r2 = tracking_store.create_run(exp_id, user_id="u", start_time=1001, tags=[], run_name="r2")
        tracking_store.set_tag(r1.info.run_id, RunTag("mlflow.user", "alice"))
        tracking_store.set_tag(r2.info.run_id, RunTag("mlflow.user", "bob"))
        runs, _ = tracking_store._search_runs(
            [exp_id], "tag.mlflow.user = 'alice'", ViewType.ACTIVE_ONLY, 100, None, None
        )
        assert len(runs) == 1
        assert runs[0].info.run_id == r1.info.run_id

    def test_search_run_name_like(self, tracking_store):
        exp_id = tracking_store.create_experiment("name-like-test", artifact_location="s3://b")
        tracking_store.create_run(
            exp_id, user_id="u", start_time=1000, tags=[], run_name="my-pipeline-v1"
        )
        tracking_store.create_run(
            exp_id, user_id="u", start_time=1001, tags=[], run_name="other-job"
        )
        runs, _ = tracking_store._search_runs(
            [exp_id], "run_name LIKE '%pipeline%'", ViewType.ACTIVE_ONLY, 100, None, None
        )
        assert len(runs) == 1

    def test_search_pagination(self, tracking_store):
        exp_id = tracking_store.create_experiment("pagination-test", artifact_location="s3://b")
        for i in range(20):
            tracking_store.create_run(
                exp_id, user_id="u", start_time=1000 + i, tags=[], run_name=f"r{i}"
            )
        runs1, token = tracking_store._search_runs(
            [exp_id], "", ViewType.ACTIVE_ONLY, 10, None, None
        )
        assert len(runs1) == 10
        assert token is not None
        runs2, _ = tracking_store._search_runs([exp_id], "", ViewType.ACTIVE_ONLY, 10, None, token)
        assert len(runs2) == 10
        ids1 = {r.info.run_id for r in runs1}
        ids2 = {r.info.run_id for r in runs2}
        assert ids1.isdisjoint(ids2)

    def test_multi_experiment_search(self, tracking_store):
        e1 = tracking_store.create_experiment("multi-exp-1", artifact_location="s3://b")
        e2 = tracking_store.create_experiment("multi-exp-2", artifact_location="s3://b")
        tracking_store.create_run(e1, user_id="u", start_time=1000, tags=[], run_name="r1")
        tracking_store.create_run(e2, user_id="u", start_time=1001, tags=[], run_name="r2")
        runs, _ = tracking_store._search_runs([e1, e2], "", ViewType.ACTIVE_ONLY, 100, None, None)
        assert len(runs) == 2
