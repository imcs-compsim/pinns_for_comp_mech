import deepxde as dde
import deepxde.backend as bkd
import numpy as np
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if "integration_tests/" in item.nodeid:
            item.add_marker(pytest.mark.integration_tests)
        elif "unit_tests/" in item.nodeid:
            item.add_marker(pytest.mark.unit_tests)


# Utility to enable eager execution in TensorFlow 
# (required for tests that involve gradient computations)
def setup_backend():
    if bkd.backend_name == "tensorflow":
        import tensorflow as tf
        if tf.executing_eagerly():
            tf.compat.v1.disable_eager_execution()


# Utility to convert tensor to NumPy across backends
def to_numpy(tensor):
    backend = bkd.backend_name
    if backend == "tensorflow":
        import tensorflow as tf
        if tf.executing_eagerly():
            return tensor.numpy()
        else:
            with tf.compat.v1.Session() as sess:
                # Initialize all variables in the session (which does not 
                # happen automatically when running in tf.compat.v1 mode)
                sess.run(tf.compat.v1.global_variables_initializer())
                return sess.run(tensor)
    elif backend == "pytorch":
        return tensor.detach().cpu().numpy()
    elif backend == "jax":
        return np.array(tensor)
    else:
        raise NotImplementedError(f"Backend {backend} not supported")
    

setup_backend()