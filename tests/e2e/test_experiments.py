import pytest

pytestmark = pytest.mark.e2e


class TestExperiments:
    def test_default_experiment_exists(self, client):
        exp = client.get_experiment("0")
        assert exp.name == "Default"
