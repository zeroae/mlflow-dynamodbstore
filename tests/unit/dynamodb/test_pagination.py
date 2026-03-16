from __future__ import annotations

from mlflow_dynamodbstore.dynamodb.pagination import decode_page_token, encode_page_token


class TestPagination:
    def test_encode_decode_round_trip(self):
        token_data = {
            "lek": {"PK": {"S": "EXP#01JQ"}, "SK": {"S": "R#01JR"}},
            "exp_idx": 0,
            "accumulated": 25,
        }
        token = encode_page_token(token_data)
        assert isinstance(token, str)
        decoded = decode_page_token(token)
        assert decoded == token_data

    def test_none_token(self):
        assert decode_page_token(None) is None
        assert decode_page_token("") is None

    def test_empty_dict_round_trip(self):
        token_data = {}
        token = encode_page_token(token_data)
        assert isinstance(token, str)
        decoded = decode_page_token(token)
        assert decoded == token_data

    def test_token_is_valid_base64(self):
        import base64

        token_data = {"lek": None, "exp_idx": 1, "accumulated": 100}
        token = encode_page_token(token_data)
        # Should not raise
        decoded_bytes = base64.urlsafe_b64decode(token + "==")
        assert len(decoded_bytes) > 0

    def test_nested_lek_preserved(self):
        token_data = {
            "lek": {
                "PK": {"S": "EXP#01JQ"},
                "SK": {"S": "R#01JR"},
                "GSI1PK": {"S": "RUNS"},
                "GSI1SK": {"S": "2024-01-01T00:00:00"},
            },
            "exp_idx": 3,
            "accumulated": 999,
        }
        token = encode_page_token(token_data)
        decoded = decode_page_token(token)
        assert decoded == token_data
