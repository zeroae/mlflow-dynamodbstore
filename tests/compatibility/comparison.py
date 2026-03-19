"""Field-by-field comparison engine for store entity contract testing."""

from tests.compatibility.field_policy import DEFAULT_POLICY, FieldPolicy


class ComparisonError(AssertionError):
    """Raised when entity comparison finds mismatches."""

    def __init__(self, diffs: list[str]):
        self.diffs = diffs
        super().__init__("\n".join(diffs))


def _get_fields(obj) -> set[str]:
    """Get comparable fields from an entity object."""
    if hasattr(obj, "__dict__"):
        return {k for k in obj.__dict__ if not k.startswith("_")}
    return set()


def assert_entities_match(
    entity_a,
    entity_b,
    policy: dict[str, FieldPolicy],
    label_a: str = "sql",
    label_b: str = "ddb",
) -> None:
    """Compare two entity objects field-by-field using the given policy."""
    fields_a = _get_fields(entity_a)
    fields_b = _get_fields(entity_b)
    all_fields = fields_a | fields_b

    diffs: list[str] = []

    for field in sorted(all_fields):
        field_policy = policy.get(field, DEFAULT_POLICY)

        if field_policy == FieldPolicy.IGNORE:
            continue

        has_a = hasattr(entity_a, field)
        has_b = hasattr(entity_b, field)

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
