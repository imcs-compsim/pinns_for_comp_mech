import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if "integration_tests/" in item.nodeid:
            item.add_marker(pytest.mark.integration_tests)
        elif "unit_tests/" in item.nodeid:
            item.add_marker(pytest.mark.unit_tests)