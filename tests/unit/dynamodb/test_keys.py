from mlflow_dynamodbstore.dynamodb.keys import (
    experiment_gsi2,
    experiment_gsi3,
    experiment_meta_lsi,
    experiment_meta_sk,
    experiment_pk,
    metric_history_sk,
    model_pk,
    run_meta_lsi,
    version_meta_lsi,
)


class TestExperimentKeys:
    def test_experiment_pk(self):
        assert experiment_pk("01JQXYZ") == {"PK": {"S": "EXP#01JQXYZ"}}

    def test_experiment_meta_sk(self):
        assert experiment_meta_sk() == {"SK": {"S": "E#META"}}

    def test_experiment_meta_lsi(self):
        lsi = experiment_meta_lsi(
            lifecycle="active",
            ulid="01JQXYZ",
            last_update_time=1709251200000,
            name="my-experiment",
        )
        assert lsi["lsi1sk"]["S"] == "active#01JQXYZ"
        assert lsi["lsi2sk"]["N"] == "1709251200000"
        assert lsi["lsi3sk"]["S"] == "my-experiment"
        assert lsi["lsi4sk"]["S"] == "tnemirepxe-ym"  # reversed

    def test_experiment_gsi2(self):
        keys = experiment_gsi2(workspace="default", lifecycle="active", ulid="01JQXYZ")
        assert keys["gsi2pk"]["S"] == "EXPERIMENTS#default#active"
        assert keys["gsi2sk"]["S"] == "01JQXYZ"

    def test_experiment_gsi3(self):
        keys = experiment_gsi3(workspace="default", name="my-experiment", exp_id="01JQXYZ")
        assert keys["gsi3pk"]["S"] == "EXP_NAME#default#my-experiment"
        assert keys["gsi3sk"]["S"] == "01JQXYZ"


class TestRunKeys:
    def test_run_meta_lsi_includes_lifecycle(self):
        lsi = run_meta_lsi(
            lifecycle="active",
            ulid="01JRABC",
            status="RUNNING",
            run_name="training-v1",
        )
        assert lsi["lsi1sk"]["S"] == "active#01JRABC"
        assert lsi["lsi3sk"]["S"] == "RUNNING#01JRABC"
        assert lsi["lsi4sk"]["S"] == "training-v1"  # lowercased


class TestMetricKeys:
    def test_metric_history_sk_zero_padded(self):
        sk = metric_history_sk(run_ulid="01JRABC", key="loss", step=42, timestamp=1709251200000)
        assert "P#00000000000000000042" in sk["SK"]["S"]

    def test_metric_history_sk_negative_step(self):
        sk = metric_history_sk(run_ulid="01JRABC", key="loss", step=-1, timestamp=1709251200000)
        assert sk["SK"]["S"].startswith("R#01JRABC#MHIST#loss#N#")


class TestModelKeys:
    def test_model_pk_uses_ulid(self):
        assert model_pk("01JMXYZ") == {"PK": {"S": "RM#01JMXYZ"}}

    def test_version_meta_lsi_includes_run_id(self):
        lsi = version_meta_lsi(
            creation_time=1709251200000,
            last_update_time=1709251200000,
            stage="Production",
            padded_ver="00000003",
            source_path="s3://bucket/model",
            run_id="01JRABC",
        )
        assert lsi["lsi5sk"]["S"] == "01JRABC#00000003"
