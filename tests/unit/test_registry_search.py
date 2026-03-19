"""Tests for tag denormalization and FTS writes in DynamoDBRegistryStore."""

from __future__ import annotations

from mlflow.entities.model_registry import RegisteredModelTag
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag

from mlflow_dynamodbstore.dynamodb.schema import (
    PK_MODEL_PREFIX,
    SK_FTS_PREFIX,
    SK_MODEL_META,
    SK_MODEL_NAME_REV,
    SK_VERSION_PREFIX,
)


class TestRegisteredModelTagDenormalization:
    """Test that set/delete_registered_model_tag denormalizes tags into the META item."""

    def test_set_registered_model_tag_denormalizes_mlflow_tag(self, registry_store):
        """After set_registered_model_tag with mlflow.* key, META item has tags map entry."""
        registry_store.create_registered_model("my-model")
        tag = RegisteredModelTag("mlflow.note", "hello")
        registry_store.set_registered_model_tag("my-model", tag)

        # Read the META item directly from DynamoDB
        model_ulid = registry_store._resolve_model_ulid("my-model")
        item = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_META,
        )
        assert item is not None
        assert "tags" in item
        assert item["tags"]["mlflow.note"] == "hello"

    def test_set_registered_model_tag_overwrites_existing(self, registry_store):
        """Setting the same tag twice updates the value in the META item."""
        registry_store.create_registered_model("my-model")
        registry_store.set_registered_model_tag("my-model", RegisteredModelTag("mlflow.note", "v1"))
        registry_store.set_registered_model_tag("my-model", RegisteredModelTag("mlflow.note", "v2"))

        model_ulid = registry_store._resolve_model_ulid("my-model")
        item = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_META,
        )
        assert item["tags"]["mlflow.note"] == "v2"

    def test_delete_registered_model_tag_removes_denormalized_entry(self, registry_store):
        """delete_registered_model_tag removes the key from the META tags map."""
        registry_store.create_registered_model("my-model")
        registry_store.set_registered_model_tag(
            "my-model", RegisteredModelTag("mlflow.note", "hello")
        )
        registry_store.delete_registered_model_tag("my-model", "mlflow.note")

        model_ulid = registry_store._resolve_model_ulid("my-model")
        item = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_META,
        )
        assert item is not None
        tags = item.get("tags", {})
        assert "mlflow.note" not in tags

    def test_non_matching_tag_not_denormalized(self, registry_store):
        """Tags that don't match mlflow.* patterns are NOT added to META tags map."""
        registry_store.create_registered_model("my-model")
        registry_store.set_registered_model_tag(
            "my-model", RegisteredModelTag("custom.key", "value")
        )

        model_ulid = registry_store._resolve_model_ulid("my-model")
        item = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_META,
        )
        assert item is not None
        tags = item.get("tags", {})
        assert "custom.key" not in tags

    def test_create_registered_model_has_empty_tags_map(self, registry_store):
        """create_registered_model initializes the META item with an empty tags map."""
        registry_store.create_registered_model("my-model")

        model_ulid = registry_store._resolve_model_ulid("my-model")
        item = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_META,
        )
        assert item is not None
        assert "tags" in item
        assert item["tags"] == {}


class TestRegistryFTSWrites:
    """Test that create/rename_registered_model writes FTS items."""

    def test_create_model_writes_fts_items(self, registry_store):
        """create_registered_model should produce FTS# items for the model name."""
        registry_store.create_registered_model("my-pipeline-model")
        model_ulid = registry_store._resolve_model_ulid("my-pipeline-model")
        fts_items = registry_store._table.query(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}", sk_prefix=SK_FTS_PREFIX
        )
        assert len(fts_items) > 0
        word_items = [i for i in fts_items if i["SK"].startswith("FTS#W#")]
        trigram_items = [i for i in fts_items if i["SK"].startswith("FTS#3#")]
        assert len(word_items) > 0
        assert len(trigram_items) > 0

    def test_create_model_fts_has_gsi2(self, registry_store):
        """FTS forward items for model names must carry gsi2pk for cross-partition search."""
        registry_store.create_registered_model("test-model")
        model_ulid = registry_store._resolve_model_ulid("test-model")
        fts_items = registry_store._table.query(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}", sk_prefix=SK_FTS_PREFIX
        )
        assert len(fts_items) > 0
        for item in fts_items:
            assert "gsi2pk" in item

    def test_rename_model_updates_fts(self, registry_store):
        """rename_registered_model should replace old FTS items with new ones."""
        registry_store.create_registered_model("old-model")
        registry_store.rename_registered_model("old-model", "new-model")
        model_ulid = registry_store._resolve_model_ulid("new-model")
        fts_items = registry_store._table.query(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}", sk_prefix=SK_FTS_PREFIX
        )
        sks = [i["SK"] for i in fts_items]
        # Should not have "old" word tokens
        assert not any("old" in sk.lower() for sk in sks if sk.startswith("FTS#W#"))


class TestModelVersionTagDenormalization:
    """Test that set/delete_model_version_tag denormalizes tags into the version item."""

    def test_set_model_version_tag_denormalizes_mlflow_tag(self, registry_store):
        """After set_model_version_tag with mlflow.* key, version item has tags map entry."""
        registry_store.create_registered_model("my-model")
        mv = registry_store.create_model_version("my-model", source="s3://bucket/model")
        version = mv.version

        tag = ModelVersionTag("mlflow.note", "hello")
        registry_store.set_model_version_tag("my-model", version, tag)

        model_ulid = registry_store._resolve_model_ulid("my-model")
        padded = f"{int(version):08d}"
        item = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=f"{SK_VERSION_PREFIX}{padded}",
        )
        assert item is not None
        assert "tags" in item
        assert item["tags"]["mlflow.note"] == "hello"

    def test_set_model_version_tag_overwrites_existing(self, registry_store):
        """Setting the same version tag twice updates the value."""
        registry_store.create_registered_model("my-model")
        mv = registry_store.create_model_version("my-model", source="s3://bucket/model")
        version = mv.version

        registry_store.set_model_version_tag(
            "my-model", version, ModelVersionTag("mlflow.note", "v1")
        )
        registry_store.set_model_version_tag(
            "my-model", version, ModelVersionTag("mlflow.note", "v2")
        )

        model_ulid = registry_store._resolve_model_ulid("my-model")
        padded = f"{int(version):08d}"
        item = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=f"{SK_VERSION_PREFIX}{padded}",
        )
        assert item["tags"]["mlflow.note"] == "v2"

    def test_delete_model_version_tag_removes_denormalized_entry(self, registry_store):
        """delete_model_version_tag removes the key from the version item tags map."""
        registry_store.create_registered_model("my-model")
        mv = registry_store.create_model_version("my-model", source="s3://bucket/model")
        version = mv.version

        registry_store.set_model_version_tag(
            "my-model", version, ModelVersionTag("mlflow.note", "hello")
        )
        registry_store.delete_model_version_tag("my-model", version, "mlflow.note")

        model_ulid = registry_store._resolve_model_ulid("my-model")
        padded = f"{int(version):08d}"
        item = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=f"{SK_VERSION_PREFIX}{padded}",
        )
        assert item is not None
        tags = item.get("tags", {})
        assert "mlflow.note" not in tags

    def test_version_non_matching_tag_not_denormalized(self, registry_store):
        """Version tags that don't match mlflow.* patterns are NOT added to version tags map."""
        registry_store.create_registered_model("my-model")
        mv = registry_store.create_model_version("my-model", source="s3://bucket/model")
        version = mv.version

        registry_store.set_model_version_tag(
            "my-model", version, ModelVersionTag("custom.key", "value")
        )

        model_ulid = registry_store._resolve_model_ulid("my-model")
        padded = f"{int(version):08d}"
        item = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=f"{SK_VERSION_PREFIX}{padded}",
        )
        assert item is not None
        tags = item.get("tags", {})
        assert "custom.key" not in tags

    def test_create_model_version_has_empty_tags_map(self, registry_store):
        """create_model_version initializes the version item with an empty tags map."""
        registry_store.create_registered_model("my-model")
        mv = registry_store.create_model_version("my-model", source="s3://bucket/model")
        version = mv.version

        model_ulid = registry_store._resolve_model_ulid("my-model")
        padded = f"{int(version):08d}"
        item = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=f"{SK_VERSION_PREFIX}{padded}",
        )
        assert item is not None
        assert "tags" in item
        assert item["tags"] == {}


class TestModelNameRev:
    """Test that create/rename_registered_model writes NAME_REV items for suffix ILIKE."""

    def test_create_model_writes_name_rev(self, registry_store):
        """create_registered_model should write a NAME_REV item with reversed lowercase name."""
        registry_store.create_registered_model("MyModel")
        model_ulid = registry_store._resolve_model_ulid("MyModel")
        name_rev = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_NAME_REV,
        )
        assert name_rev is not None
        assert name_rev["gsi5sk"].startswith("REV#")

    def test_create_model_name_rev_has_correct_gsi5pk(self, registry_store):
        """NAME_REV item should have gsi5pk set to MODEL_NAMES#<workspace>."""
        registry_store.create_registered_model("MyModel")
        model_ulid = registry_store._resolve_model_ulid("MyModel")
        name_rev = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_NAME_REV,
        )
        assert name_rev is not None
        assert name_rev["gsi5pk"] == "MODEL_NAMES#default"

    def test_create_model_name_rev_has_reversed_name(self, registry_store):
        """NAME_REV gsi5sk should contain the reversed lowercase model name."""
        registry_store.create_registered_model("MyModel")
        model_ulid = registry_store._resolve_model_ulid("MyModel")
        name_rev = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_NAME_REV,
        )
        assert name_rev is not None
        # "MyModel" reversed lowercase is "ledomym"
        assert "ledomym" in name_rev["gsi5sk"].lower()

    def test_rename_model_updates_name_rev(self, registry_store):
        """rename_registered_model should update the NAME_REV item with the new reversed name."""
        registry_store.create_registered_model("OldModel")
        registry_store.rename_registered_model("OldModel", "NewModel")
        model_ulid = registry_store._resolve_model_ulid("NewModel")
        name_rev = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_NAME_REV,
        )
        assert name_rev is not None
        # "NewModel" reversed lowercase is "ledomwen"
        assert "ledomwen" in name_rev["gsi5sk"].lower()
        # "OldModel" reversed lowercase is "ledomdlo" — should NOT be present
        assert "ledomdlo" not in name_rev["gsi5sk"].lower()

    def test_create_model_name_rev_contains_ulid(self, registry_store):
        """NAME_REV gsi5sk should contain the model ULID as a suffix."""
        registry_store.create_registered_model("MyModel")
        model_ulid = registry_store._resolve_model_ulid("MyModel")
        name_rev = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_NAME_REV,
        )
        assert name_rev is not None
        assert model_ulid in name_rev["gsi5sk"]


class TestSearchRegisteredModels:
    """Test search_registered_models with filter and order_by support."""

    def test_search_no_filter(self, registry_store):
        registry_store.create_registered_model("model1")
        registry_store.create_registered_model("model2")
        models = registry_store.search_registered_models()
        assert len(models) >= 2

    def test_search_by_name_equals(self, registry_store):
        registry_store.create_registered_model("target-model")
        registry_store.create_registered_model("other-model")
        models = registry_store.search_registered_models(filter_string="name = 'target-model'")
        assert len(models) == 1
        assert models[0].name == "target-model"

    def test_search_by_name_like_prefix(self, registry_store):
        registry_store.create_registered_model("prod-model")
        registry_store.create_registered_model("dev-model")
        models = registry_store.search_registered_models(filter_string="name LIKE 'prod%'")
        assert len(models) == 1
        assert models[0].name == "prod-model"

    def test_search_by_name_like_contains(self, registry_store):
        registry_store.create_registered_model("my-pipeline-model")
        registry_store.create_registered_model("other-job")
        models = registry_store.search_registered_models(filter_string="name LIKE '%pipeline%'")
        assert len(models) == 1
        assert models[0].name == "my-pipeline-model"

    def test_search_by_tag(self, registry_store):
        registry_store.create_registered_model("model-a")
        registry_store.set_registered_model_tag("model-a", RegisteredModelTag("env", "prod"))
        registry_store.create_registered_model("model-b")
        registry_store.set_registered_model_tag("model-b", RegisteredModelTag("env", "dev"))
        models = registry_store.search_registered_models(filter_string="tag.env = 'prod'")
        assert len(models) == 1
        assert models[0].name == "model-a"

    def test_search_order_by_name_asc(self, registry_store):
        registry_store.create_registered_model("zebra-model")
        registry_store.create_registered_model("alpha-model")
        models = registry_store.search_registered_models(order_by=["name ASC"])
        names = [m.name for m in models]
        assert names == sorted(names)

    def test_search_order_by_name_desc(self, registry_store):
        registry_store.create_registered_model("alpha-model")
        registry_store.create_registered_model("zebra-model")
        models = registry_store.search_registered_models(order_by=["name DESC"])
        names = [m.name for m in models]
        assert names == sorted(names, reverse=True)

    def test_search_max_results(self, registry_store):
        for i in range(5):
            registry_store.create_registered_model(f"model-{i}")
        models = registry_store.search_registered_models(max_results=3)
        assert len(models) == 3

    def test_search_name_equals_not_found(self, registry_store):
        registry_store.create_registered_model("existing-model")
        models = registry_store.search_registered_models(filter_string="name = 'nonexistent'")
        assert len(models) == 0


class TestSearchRegisteredModelOrderBy:
    """Test order_by with tiebreaking and pagination edge cases."""

    def _create_models_with_timestamps(self, store, models_and_timestamps):
        """Create models with specific timestamps using mocks.

        Args:
            models_and_timestamps: list of (name, timestamp_ms) tuples
        """
        from unittest import mock

        for name, ts in models_and_timestamps:
            with mock.patch(
                "mlflow_dynamodbstore.registry_store.get_current_time_millis",
                return_value=ts,
            ):
                store.create_registered_model(name)

    def test_order_by_timestamp_asc_with_name_desc_tiebreak(self, registry_store):
        """When timestamps tie, secondary sort by name DESC should break ties."""
        self._create_models_with_timestamps(
            registry_store,
            [("MR1", 1), ("MR2", 1), ("MR3", 2), ("MR4", 2)],
        )
        models = registry_store.search_registered_models(
            order_by=["last_updated_timestamp ASC", "name DESC"]
        )
        names = [m.name for m in models]
        assert names == ["MR2", "MR1", "MR4", "MR3"]

    def test_order_by_timestamp_desc_implicit_name_asc_tiebreak(self, registry_store):
        """When only timestamp DESC is specified, ties break by name ASC (default)."""
        self._create_models_with_timestamps(
            registry_store,
            [("MR1", 1), ("MR2", 1), ("MR3", 2), ("MR4", 2)],
        )
        models = registry_store.search_registered_models(order_by=["last_updated_timestamp DESC"])
        names = [m.name for m in models]
        assert names == ["MR3", "MR4", "MR1", "MR2"]

    def test_empty_order_by_defaults_to_name_asc(self, registry_store):
        """Empty order_by defaults to name ASC regardless of timestamps."""
        self._create_models_with_timestamps(
            registry_store,
            [("MR3", 2), ("MR1", 1), ("MR4", 2), ("MR2", 1)],
        )
        models = registry_store.search_registered_models(order_by=[])
        names = [m.name for m in models]
        assert names == ["MR1", "MR2", "MR3", "MR4"]

    def test_pagination_with_tiebreaking_no_duplicates(self, registry_store):
        """Paginating through tied-timestamp results must not skip or duplicate items."""
        # Create 6 models: 3 pairs with same timestamp
        self._create_models_with_timestamps(
            registry_store,
            [
                ("A1", 10),
                ("A2", 10),
                ("B1", 20),
                ("B2", 20),
                ("C1", 30),
                ("C2", 30),
            ],
        )
        # Fetch page by page with max_results=2
        all_names: list[str] = []
        token = None
        for _ in range(10):  # safety limit
            result = registry_store.search_registered_models(
                order_by=["last_updated_timestamp ASC", "name ASC"],
                max_results=2,
                page_token=token,
            )
            all_names.extend(m.name for m in result)
            token = result.token
            if not token:
                break

        # All 6 models returned exactly once, in correct order
        assert all_names == ["A1", "A2", "B1", "B2", "C1", "C2"]

    def test_pagination_tiebreak_at_page_boundary(self, registry_store):
        """When a page boundary falls in the middle of tied items, ordering is preserved."""
        # 4 models all with the same timestamp — tiebreak is purely by name
        self._create_models_with_timestamps(
            registry_store,
            [("D", 100), ("B", 100), ("C", 100), ("A", 100)],
        )
        # Page size 2: should get [A, B] then [C, D]
        all_names: list[str] = []
        token = None
        for _ in range(10):
            result = registry_store.search_registered_models(
                order_by=["last_updated_timestamp ASC", "name ASC"],
                max_results=2,
                page_token=token,
            )
            all_names.extend(m.name for m in result)
            token = result.token
            if not token:
                break

        assert all_names == ["A", "B", "C", "D"]

    def test_pagination_timestamp_desc_name_desc(self, registry_store):
        """Both directions reversed: timestamp DESC, name DESC."""
        self._create_models_with_timestamps(
            registry_store,
            [("A", 1), ("B", 1), ("C", 2), ("D", 2)],
        )
        all_names: list[str] = []
        token = None
        for _ in range(10):
            result = registry_store.search_registered_models(
                order_by=["last_updated_timestamp DESC", "name DESC"],
                max_results=2,
                page_token=token,
            )
            all_names.extend(m.name for m in result)
            token = result.token
            if not token:
                break

        assert all_names == ["D", "C", "B", "A"]

    def test_pagination_tie_group_exceeds_page_size(self, registry_store):
        """When a tie group is larger than max_results, overflow cache is used."""
        self._create_models_with_timestamps(
            registry_store,
            [("E", 1), ("D", 1), ("C", 1), ("B", 1), ("A", 1)],
        )
        all_names: list[str] = []
        token = None
        for _ in range(10):
            result = registry_store.search_registered_models(
                order_by=["last_updated_timestamp ASC", "name ASC"],
                max_results=2,
                page_token=token,
            )
            all_names.extend(m.name for m in result)
            token = result.token
            if not token:
                break

        assert all_names == ["A", "B", "C", "D", "E"]

    def test_pagination_mixed_tie_groups(self, registry_store):
        """Mix of tie groups and unique timestamps paginate correctly."""
        self._create_models_with_timestamps(
            registry_store,
            [
                ("A1", 10),
                ("A2", 10),
                ("A3", 10),  # tie group of 3
                ("B1", 20),  # singleton
                ("C1", 30),
                ("C2", 30),  # tie group of 2
            ],
        )
        all_names: list[str] = []
        token = None
        for _ in range(10):
            result = registry_store.search_registered_models(
                order_by=["last_updated_timestamp ASC", "name DESC"],
                max_results=2,
                page_token=token,
            )
            all_names.extend(m.name for m in result)
            token = result.token
            if not token:
                break

        # ts=10 group sorted name DESC: A3, A2, A1; ts=20: B1; ts=30 name DESC: C2, C1
        assert all_names == ["A3", "A2", "A1", "B1", "C2", "C1"]

    def test_pagination_tiebreak_with_prompt_filter(self, registry_store):
        """Tiebreak pagination works when prompt filter is active (rejects prompts)."""
        from unittest import mock

        from mlflow.prompt.constants import IS_PROMPT_TAG_KEY

        # Create mix of models and prompts, all with same timestamp
        for name in ["m1", "m2", "m3", "m4"]:
            with mock.patch(
                "mlflow_dynamodbstore.registry_store.get_current_time_millis",
                return_value=100,
            ):
                registry_store.create_registered_model(name)

        # Mark two as prompts
        registry_store.set_registered_model_tag("m2", RegisteredModelTag(IS_PROMPT_TAG_KEY, "true"))
        registry_store.set_registered_model_tag("m4", RegisteredModelTag(IS_PROMPT_TAG_KEY, "true"))

        # Search non-prompts with tiebreak ordering
        all_names: list[str] = []
        token = None
        for _ in range(10):
            result = registry_store.search_registered_models(
                order_by=["last_updated_timestamp ASC", "name ASC"],
                max_results=1,
                page_token=token,
            )
            all_names.extend(m.name for m in result)
            token = result.token
            if not token:
                break

        # Only non-prompts: m1, m3
        assert all_names == ["m1", "m3"]

    def test_pagination_tiebreak_with_filter_no_data_loss(self, registry_store):
        """When filter_fn is active with tiebreak, all items must be returned.

        Verifies that 2x batch_size optimization doesn't cause data loss
        when combined with tiebreak tie-group detection. With filter_fn active,
        the paginated path fetches 2x items. If tiebreak code breaks early at
        max_results+1 and uses the LEK (which points past the full batch),
        unprocessed items would be lost.
        """
        # Create enough models that DDB paginates (2x batch won't get all)
        models = []
        for i in range(15):
            models.append((f"m{i:02d}", 10))  # all same timestamp
        self._create_models_with_timestamps(registry_store, models)

        all_names: list[str] = []
        token = None
        for _ in range(20):
            result = registry_store.search_registered_models(
                order_by=["last_updated_timestamp ASC", "name ASC"],
                max_results=3,
                page_token=token,
            )
            all_names.extend(m.name for m in result)
            token = result.token
            if not token:
                break

        expected = [f"m{i:02d}" for i in range(15)]
        assert all_names == expected

    def test_pagination_consecutive_overflow_groups(self, registry_store):
        """Two consecutive tie groups both larger than page size."""
        # max_results=2, groups: x(t=10) has 4 items, y(t=20) has 3 items, z(t=30) has 1
        self._create_models_with_timestamps(
            registry_store,
            [
                ("x1", 10),
                ("x2", 10),
                ("x3", 10),
                ("x4", 10),
                ("y1", 20),
                ("y2", 20),
                ("y3", 20),
                ("z0", 30),
            ],
        )
        all_names: list[str] = []
        token = None
        for _ in range(20):
            result = registry_store.search_registered_models(
                order_by=["last_updated_timestamp ASC", "name ASC"],
                max_results=2,
                page_token=token,
            )
            all_names.extend(m.name for m in result)
            token = result.token
            if not token:
                break

        assert all_names == ["x1", "x2", "x3", "x4", "y1", "y2", "y3", "z0"]


class TestSearchModelVersions:
    """Test search_model_versions with filter support."""

    def test_search_by_model_name(self, registry_store):
        registry_store.create_registered_model("model-a")
        registry_store.create_model_version("model-a", source="s3://a")
        registry_store.create_registered_model("model-b")
        registry_store.create_model_version("model-b", source="s3://b")
        versions = registry_store.search_model_versions(filter_string="name = 'model-a'")
        assert len(versions) == 1
        assert versions[0].name == "model-a"

    def test_search_all_versions(self, registry_store):
        registry_store.create_registered_model("model-c")
        registry_store.create_model_version("model-c", source="s3://c")
        registry_store.create_model_version("model-c", source="s3://c2")
        versions = registry_store.search_model_versions()
        assert len(versions) >= 2

    def test_search_versions_max_results(self, registry_store):
        registry_store.create_registered_model("model-d")
        for i in range(5):
            registry_store.create_model_version("model-d", source=f"s3://d{i}")
        versions = registry_store.search_model_versions(max_results=3)
        assert len(versions) == 3

    def test_search_versions_by_name_multiple_versions(self, registry_store):
        registry_store.create_registered_model("model-e")
        registry_store.create_model_version("model-e", source="s3://e1")
        registry_store.create_model_version("model-e", source="s3://e2")
        registry_store.create_registered_model("model-f")
        registry_store.create_model_version("model-f", source="s3://f1")
        versions = registry_store.search_model_versions(filter_string="name = 'model-e'")
        assert len(versions) == 2
        assert all(v.name == "model-e" for v in versions)

    def test_search_versions_by_run_id(self, registry_store):
        registry_store.create_registered_model("model-g")
        registry_store.create_model_version("model-g", source="s3://g1", run_id="run-abc")
        registry_store.create_model_version("model-g", source="s3://g2", run_id="run-def")
        versions = registry_store.search_model_versions(filter_string="run_id = 'run-abc'")
        assert len(versions) == 1
        assert versions[0].run_id == "run-abc"
