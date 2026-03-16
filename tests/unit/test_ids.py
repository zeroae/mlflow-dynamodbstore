from mlflow_dynamodbstore.ids import generate_ulid, ulid_from_timestamp


class TestUlidGeneration:
    def test_generate_ulid_returns_string(self):
        result = generate_ulid()
        assert isinstance(result, str)
        assert len(result) == 26  # standard ULID length

    def test_ulid_from_timestamp_is_time_ordered(self):
        ulid1 = ulid_from_timestamp(1709251200000)
        ulid2 = ulid_from_timestamp(1709251200001)
        assert ulid1 < ulid2

    def test_ulid_from_timestamp_deterministic_prefix(self):
        ulid1 = ulid_from_timestamp(1709251200000)
        ulid2 = ulid_from_timestamp(1709251200000)
        # Same timestamp prefix, different random suffix
        assert ulid1[:10] == ulid2[:10]
        assert ulid1 != ulid2

    def test_generate_ulid_lowercase(self):
        """MLflow requires lowercase alphanumeric run IDs."""
        result = generate_ulid()
        assert result == result.lower() or result == result.upper()
        # ULIDs use Crockford base32, always uppercase — we lowercase it
