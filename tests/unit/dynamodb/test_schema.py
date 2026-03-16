from mlflow_dynamodbstore.dynamodb import schema


class TestSchemaConstants:
    def test_sk_prefixes_defined(self):
        assert schema.SK_EXPERIMENT_META == "E#META"
        assert schema.SK_EXPERIMENT_TAG_PREFIX == "E#TAG#"
        assert schema.SK_RUN_PREFIX == "R#"
        assert schema.SK_TRACE_PREFIX == "T#"

    def test_gsi_names(self):
        assert schema.GSI1_NAME == "gsi1"
        assert schema.GSI2_NAME == "gsi2"
        assert schema.GSI3_NAME == "gsi3"
        assert schema.GSI4_NAME == "gsi4"
        assert schema.GSI5_NAME == "gsi5"

    def test_lsi_attributes(self):
        assert schema.LSI1_SK == "lsi1sk"
        assert schema.LSI2_SK == "lsi2sk"
        assert schema.LSI3_SK == "lsi3sk"
        assert schema.LSI4_SK == "lsi4sk"
        assert schema.LSI5_SK == "lsi5sk"

    def test_pk_prefixes(self):
        assert schema.PK_EXPERIMENT_PREFIX == "EXP#"
        assert schema.PK_MODEL_PREFIX == "RM#"
        assert schema.PK_WORKSPACE_PREFIX == "WORKSPACE#"
        assert schema.PK_USER_PREFIX == "USER#"
        assert schema.PK_CONFIG == "CONFIG"
