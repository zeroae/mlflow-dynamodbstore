"""Unit tests for DynamoDBJobStore."""

from __future__ import annotations

import json
import time

import pytest
from mlflow.entities._job_status import JobStatus
from mlflow.exceptions import MlflowException


@pytest.fixture
def job_store(mock_dynamodb):
    from mlflow_dynamodbstore.job_store import DynamoDBJobStore

    return DynamoDBJobStore("dynamodb://us-east-1/test_table")


class TestCreateAndGetJob:
    def test_create_job(self, job_store):
        job = job_store.create_job("my_job", '{"a": 1}')
        assert job.job_name == "my_job"
        assert job.params == '{"a": 1}'
        assert job.status == JobStatus.PENDING
        assert job.retry_count == 0
        assert job.result is None
        assert job.timeout is None

    def test_create_job_with_timeout(self, job_store):
        job = job_store.create_job("my_job", "{}", timeout=30.0)
        assert job.timeout == 30.0

    def test_get_job(self, job_store):
        created = job_store.create_job("my_job", '{"x": 2}')
        fetched = job_store.get_job(created.job_id)
        assert fetched.job_id == created.job_id
        assert fetched.job_name == "my_job"
        assert fetched.params == '{"x": 2}'
        assert fetched.status == JobStatus.PENDING

    def test_get_nonexistent_job_raises(self, job_store):
        with pytest.raises(MlflowException, match="not found"):
            job_store.get_job("nonexistent-id")


class TestStatusTransitions:
    def test_start_job(self, job_store):
        job = job_store.create_job("my_job", "{}")
        job_store.start_job(job.job_id)
        fetched = job_store.get_job(job.job_id)
        assert fetched.status == JobStatus.RUNNING

    def test_start_job_is_atomic(self, job_store):
        """Second start_job on the same job should fail."""
        job = job_store.create_job("my_job", "{}")
        job_store.start_job(job.job_id)
        with pytest.raises(MlflowException, match="cannot start"):
            job_store.start_job(job.job_id)

    def test_start_nonexistent_job_raises(self, job_store):
        with pytest.raises(MlflowException, match="not found"):
            job_store.start_job("nonexistent-id")

    def test_finish_job(self, job_store):
        job = job_store.create_job("my_job", "{}")
        job_store.start_job(job.job_id)
        job_store.finish_job(job.job_id, '{"output": 42}')
        fetched = job_store.get_job(job.job_id)
        assert fetched.status == JobStatus.SUCCEEDED
        assert fetched.result == '{"output": 42}'

    def test_fail_job(self, job_store):
        job = job_store.create_job("my_job", "{}")
        job_store.start_job(job.job_id)
        job_store.fail_job(job.job_id, "something went wrong")
        fetched = job_store.get_job(job.job_id)
        assert fetched.status == JobStatus.FAILED
        assert fetched.result == "something went wrong"

    def test_mark_job_timed_out(self, job_store):
        job = job_store.create_job("my_job", "{}")
        job_store.start_job(job.job_id)
        job_store.mark_job_timed_out(job.job_id)
        fetched = job_store.get_job(job.job_id)
        assert fetched.status == JobStatus.TIMEOUT

    def test_cancel_job(self, job_store):
        job = job_store.create_job("my_job", "{}")
        result = job_store.cancel_job(job.job_id)
        assert result.status == JobStatus.CANCELED

    def test_reset_job(self, job_store):
        job = job_store.create_job("my_job", "{}")
        job_store.start_job(job.job_id)
        job_store.reset_job(job.job_id)
        fetched = job_store.get_job(job.job_id)
        assert fetched.status == JobStatus.PENDING

    def test_cannot_update_finalized_job(self, job_store):
        job = job_store.create_job("my_job", "{}")
        job_store.start_job(job.job_id)
        job_store.finish_job(job.job_id, "done")
        with pytest.raises(MlflowException, match="already finalized"):
            job_store.fail_job(job.job_id, "error")


class TestRetryOrFail:
    def test_retry_increments_count(self, job_store, monkeypatch):
        monkeypatch.setenv("MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES", "3")
        job = job_store.create_job("my_job", "{}")
        job_store.start_job(job.job_id)

        result = job_store.retry_or_fail_job(job.job_id, "transient error")
        assert result == 1
        fetched = job_store.get_job(job.job_id)
        assert fetched.status == JobStatus.PENDING
        assert fetched.retry_count == 1

    def test_retry_exceeds_max_fails(self, job_store, monkeypatch):
        monkeypatch.setenv("MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES", "1")
        job = job_store.create_job("my_job", "{}")
        job_store.start_job(job.job_id)

        # First retry succeeds
        result = job_store.retry_or_fail_job(job.job_id, "error1")
        assert result == 1

        # Start again after retry
        job_store.start_job(job.job_id)

        # Second retry exceeds max → fails
        result = job_store.retry_or_fail_job(job.job_id, "error2")
        assert result is None
        fetched = job_store.get_job(job.job_id)
        assert fetched.status == JobStatus.FAILED
        assert fetched.result == "error2"


class TestListJobs:
    def _create_jobs(self, job_store, count=5):
        jobs = []
        for i in range(count):
            job = job_store.create_job(f"job_{i % 2}", json.dumps({"idx": i}))
            jobs.append(job)
            time.sleep(0.001)  # ensure different creation_time
        return jobs

    def test_list_all_jobs(self, job_store):
        created = self._create_jobs(job_store, 3)
        result = list(job_store.list_jobs())
        assert len(result) == 3
        # Should be ordered by creation_time ASC
        assert [j.job_id for j in result] == [c.job_id for c in created]

    def test_list_jobs_filter_by_name(self, job_store):
        self._create_jobs(job_store, 4)
        result = list(job_store.list_jobs(job_name="job_0"))
        assert all(j.job_name == "job_0" for j in result)
        assert len(result) == 2

    def test_list_jobs_filter_by_status(self, job_store):
        jobs = self._create_jobs(job_store, 3)
        job_store.start_job(jobs[0].job_id)
        job_store.start_job(jobs[1].job_id)

        result = list(job_store.list_jobs(statuses=[JobStatus.RUNNING]))
        assert len(result) == 2
        assert all(j.status == JobStatus.RUNNING for j in result)

        result = list(job_store.list_jobs(statuses=[JobStatus.PENDING]))
        assert len(result) == 1

    def test_list_jobs_filter_by_multiple_statuses(self, job_store):
        jobs = self._create_jobs(job_store, 3)
        job_store.start_job(jobs[0].job_id)

        result = list(job_store.list_jobs(statuses=[JobStatus.PENDING, JobStatus.RUNNING]))
        assert len(result) == 3

    def test_list_jobs_filter_by_timestamp(self, job_store):
        jobs = self._create_jobs(job_store, 3)
        mid_time = jobs[1].creation_time

        result = list(job_store.list_jobs(end_timestamp=mid_time))
        assert all(j.creation_time <= mid_time for j in result)

        result = list(job_store.list_jobs(begin_timestamp=mid_time))
        assert all(j.creation_time >= mid_time for j in result)

    def test_list_jobs_filter_by_params(self, job_store):
        job_store.create_job("j", json.dumps({"a": 1, "b": 2}))
        job_store.create_job("j", json.dumps({"a": 1, "b": 3}))
        job_store.create_job("j", json.dumps({"a": 2, "b": 2}))

        result = list(job_store.list_jobs(params={"a": 1}))
        assert len(result) == 2

        result = list(job_store.list_jobs(params={"a": 1, "b": 2}))
        assert len(result) == 1

    def test_list_jobs_empty(self, job_store):
        result = list(job_store.list_jobs())
        assert result == []


class TestDeleteJobs:
    def test_delete_only_finalized_jobs(self, job_store):
        j1 = job_store.create_job("j1", "{}")
        j2 = job_store.create_job("j2", "{}")
        j3 = job_store.create_job("j3", "{}")

        job_store.start_job(j1.job_id)
        job_store.finish_job(j1.job_id, "done")
        # j2 stays PENDING, j3 gets canceled
        job_store.cancel_job(j3.job_id)

        deleted = job_store.delete_jobs()
        assert j1.job_id in deleted
        assert j3.job_id in deleted
        assert j2.job_id not in deleted

        # j2 should still exist
        assert job_store.get_job(j2.job_id).status == JobStatus.PENDING

    def test_delete_with_job_ids_filter(self, job_store):
        j1 = job_store.create_job("j1", "{}")
        j2 = job_store.create_job("j2", "{}")

        job_store.start_job(j1.job_id)
        job_store.finish_job(j1.job_id, "done")
        job_store.start_job(j2.job_id)
        job_store.finish_job(j2.job_id, "done")

        deleted = job_store.delete_jobs(job_ids=[j1.job_id])
        assert deleted == [j1.job_id]
        # j2 should still exist
        assert job_store.get_job(j2.job_id) is not None

    def test_delete_with_older_than(self, job_store):
        j1 = job_store.create_job("j1", "{}")
        job_store.start_job(j1.job_id)
        job_store.finish_job(j1.job_id, "done")

        # older_than=0 disables time filter — deletes all finalized
        deleted = job_store.delete_jobs(older_than=0)
        assert j1.job_id in deleted

    def test_delete_skips_non_finalized_even_with_job_ids(self, job_store):
        j1 = job_store.create_job("j1", "{}")
        # j1 is PENDING — not finalized
        deleted = job_store.delete_jobs(job_ids=[j1.job_id])
        assert deleted == []
