"""Tests for the comparison engine itself."""

from tests.compatibility.comparison import ComparisonError, assert_entities_match
from tests.compatibility.field_policy import IGNORE, MUST_MATCH, TYPE_MUST_MATCH


class FakeEntity:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_must_match_equal_passes():
    policy = {"name": MUST_MATCH}
    a = FakeEntity(name="foo")
    b = FakeEntity(name="foo")
    assert_entities_match(a, b, policy)


def test_must_match_unequal_fails():
    policy = {"name": MUST_MATCH}
    a = FakeEntity(name="foo")
    b = FakeEntity(name="bar")
    try:
        assert_entities_match(a, b, policy)
        assert False, "Should have raised"
    except ComparisonError as e:
        assert "name" in str(e)


def test_type_must_match_same_type_passes():
    policy = {"ts": TYPE_MUST_MATCH}
    a = FakeEntity(ts=100)
    b = FakeEntity(ts=200)
    assert_entities_match(a, b, policy)


def test_type_must_match_different_type_fails():
    policy = {"ts": TYPE_MUST_MATCH}
    a = FakeEntity(ts=100)
    b = FakeEntity(ts=100.0)
    try:
        assert_entities_match(a, b, policy)
        assert False, "Should have raised"
    except ComparisonError as e:
        assert "ts" in str(e)
        assert "int" in str(e)
        assert "float" in str(e)


def test_ignore_skips_field():
    policy = {"id": IGNORE, "name": MUST_MATCH}
    a = FakeEntity(id="abc", name="same")
    b = FakeEntity(id="xyz", name="same")
    assert_entities_match(a, b, policy)


def test_unknown_field_defaults_to_must_match():
    policy = {"name": MUST_MATCH}
    a = FakeEntity(name="same", new_field="a")
    b = FakeEntity(name="same", new_field="b")
    try:
        assert_entities_match(a, b, policy)
        assert False, "Should have raised"
    except ComparisonError as e:
        assert "new_field" in str(e)


def test_missing_field_on_one_side_fails():
    policy = {"name": MUST_MATCH, "desc": MUST_MATCH}
    a = FakeEntity(name="same", desc="hello")
    b = FakeEntity(name="same")
    try:
        assert_entities_match(a, b, policy)
        assert False, "Should have raised"
    except ComparisonError as e:
        assert "desc" in str(e)
