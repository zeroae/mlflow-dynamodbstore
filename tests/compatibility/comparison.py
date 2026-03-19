"""Field-by-field comparison engine for store entity contract testing."""

from tests.compatibility.field_policy import DEFAULT_POLICY, FieldPolicy

# Attributes that are methods/classmethods, not data fields
_NON_DATA_ATTRS = frozenset({"from_proto", "to_proto", "from_dictionary", "from_dict", "to_dict"})


class ComparisonError(AssertionError):
    """Raised when entity comparison finds mismatches."""

    def __init__(self, diffs: list[str]):
        self.diffs = diffs
        super().__init__("\n".join(diffs))


def _get_public_fields(obj) -> set[str]:
    """Get public data fields from an entity via properties and plain attributes."""
    cls = type(obj)
    fields = set()
    # Properties defined on the class
    for name in dir(cls):
        if name.startswith("_") or name in _NON_DATA_ATTRS:
            continue
        if isinstance(getattr(cls, name, None), property):
            fields.add(name)
    # Plain public attributes set on the instance
    for name in getattr(obj, "__dict__", {}):
        if not name.startswith("_") and name not in _NON_DATA_ATTRS:
            fields.add(name)
    return fields


def assert_entities_match(
    entity_a,
    entity_b,
    policy: dict[str, FieldPolicy],
    label_a: str = "sql",
    label_b: str = "ddb",
) -> None:
    """Compare two entity objects field-by-field using the given policy.

    Fields are driven by the policy dict, but any public data field present
    on either entity that is NOT in the policy triggers an error so that
    new fields are never silently ignored.
    """
    discovered = _get_public_fields(entity_a) | _get_public_fields(entity_b)
    all_fields = set(policy.keys()) | discovered

    diffs: list[str] = []

    for field in sorted(all_fields):
        field_policy = policy.get(field, DEFAULT_POLICY)

        if field_policy == FieldPolicy.IGNORE:
            continue

        has_a = hasattr(entity_a, field)
        has_b = hasattr(entity_b, field)

        if field not in policy:
            diffs.append(f"  {field}: not in policy (found on entity, add to field_policy)")
            continue

        if has_a and not has_b:
            diffs.append(f"  {field}: present in {label_a} but missing in {label_b}")
            continue
        if has_b and not has_a:
            diffs.append(f"  {field}: present in {label_b} but missing in {label_a}")
            continue

        val_a = getattr(entity_a, field)
        val_b = getattr(entity_b, field)

        if field_policy == FieldPolicy.MUST_MATCH:
            if val_a != val_b:
                diffs.append(f"  {field}: {label_a}={val_a!r} != {label_b}={val_b!r}")
        elif field_policy == FieldPolicy.TYPE_MUST_MATCH:
            if type(val_a) is not type(val_b):
                diffs.append(
                    f"  {field}: type mismatch — {label_a}={type(val_a).__name__}, "
                    f"{label_b}={type(val_b).__name__}"
                )

    if diffs:
        raise ComparisonError(diffs)
