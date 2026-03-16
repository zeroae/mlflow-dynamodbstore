import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on directory."""
    for item in items:
        if "/unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "/compatibility/" in str(item.fspath):
            item.add_marker(pytest.mark.compatibility)
