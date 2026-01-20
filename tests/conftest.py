from deepxde.backend import get_preferred_backend
import pytest


# Utility function to mark tests based on their directory
def pytest_collection_modifyitems(items):
    for item in items:
        if "integration_tests/" in item.nodeid:
            item.add_marker(pytest.mark.integration_tests)
        elif "unit_tests/" in item.nodeid:
            item.add_marker(pytest.mark.unit_tests)


# Utility function to convert a tensor to NumPy across backends 
# (currently only supported for PyTorch)
def to_numpy(tensor):
    backend_name = get_preferred_backend()
    if backend_name == "pytorch":
        return tensor.detach().cpu().numpy()
    else:
        raise NotImplementedError(
            f"Testing for backend {backend_name} currently not supported"
        )
    
