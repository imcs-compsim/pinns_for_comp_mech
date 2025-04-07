import deepxde.backend as bkd
import numpy as np
import pytest

from utils.math.differential_utils import tensor_divergence, vector_jacobian

from conftest import to_numpy


# --------------------- Functions ---------------------

def vector_field_2d(coords):
    """A simple 2D vector field with non-trivial jacobian."""
    x, y = coords[..., 0], coords[..., 1]
    v1 = x**2 * y + bkd.cos(y)
    v2 = bkd.sin(x * y) + y**3
    return bkd.stack([v1, v2], axis=-1)  # shape (..., 2)

def tensor_map_2d(coords):
    """A simple 2D tensor map with non-trivial divergence."""
    tensor_list = []
    for _i in range(bkd.shape(coords)[0]):
        x, y = coords[_i,0], coords[_i,1]
        t11 = x**2 * y
        t12 = bkd.sin(x)
        t21 = bkd.exp(y)
        t22 = x * y + y**2
        tensor_list.append([[t11, t12], [t21, t22]])
    return bkd.as_tensor(tensor_list)  # shape (..., 2, 2)


# --------------------- Fixtures ---------------------

@pytest.fixture
def list_of_2d_coords():
    temp = bkd.as_tensor(
        [
            bkd.Variable([1.0, 2.0]),
            bkd.Variable([0.0, 1.0]),
            bkd.Variable([3.0, -1.0]),
        ]
    )
    return temp

@pytest.fixture
def list_of_2d_vector_jacobians():
    temp = bkd.as_tensor(
        [
            [[4.0, 1.0-np.sin(2.0)], [2.0*np.cos(2.0), np.cos(2.0)+12.0]],
            [[0.0, -np.sin(1.0)], [1.0, 3.0]],
            [[-6.0, 9.0+np.sin(1.0)], [-1.0*np.cos(-3.0), 3.0*np.cos(-3.0)+3.0]],
        ]
    )
    return temp

@pytest.fixture
def list_of_2d_tensor_divergences():
    temp = bkd.as_tensor(
        [
            [4.0, 5.0],
            [0.0, 2.0],
            [-6.0, 1.0],
        ]
    )
    return temp


# --------------------- Tests ---------------------

@pytest.mark.parametrize(
    "batch_of_coords, batch_of_results",
    [
        ("list_of_2d_coords", "list_of_2d_vector_jacobians"),
    ]
)
def test_math_differential_vector_jacobian(batch_of_coords, batch_of_results, request):
    # since fixtures can't be passed in parameterized tests directly, we need 
    # to provide the name of the fixture as string and then retrieve their 
    # values by using the special request fixture which is provided by pytest
    batch_of_coords = request.getfixturevalue(batch_of_coords)
    batch_of_results = request.getfixturevalue(batch_of_results)

    # evaluate the vector field
    batch_of_vectors = vector_field_2d(batch_of_coords)

    # compute the jacobians of the batch of tensors
    batch_of_vector_jacobians = vector_jacobian(batch_of_vectors, batch_of_coords)

    # convert the tensors to NumPy arrays
    computed_jacobians = to_numpy(batch_of_vector_jacobians)
    expected_jacobians = to_numpy(batch_of_results)

    # # make sure the transposed tensors are correct
    np.testing.assert_allclose(computed_jacobians, expected_jacobians, rtol=5e-6)


@pytest.mark.parametrize(
    "batch_of_coords, batch_of_results",
    [
        ("list_of_2d_coords", "list_of_2d_tensor_divergences"),
    ]
)
def test_math_differential_tensor_divergence(batch_of_coords, batch_of_results, request):
    # since fixtures can't be passed in parameterized tests directly, we need 
    # to provide the name of the fixture as string and then retrieve their 
    # values by using the special request fixture which is provided by pytest
    batch_of_coords = request.getfixturevalue(batch_of_coords)
    batch_of_results = request.getfixturevalue(batch_of_results)

    # evaluate the tensor map
    batch_of_tensors = tensor_map_2d(batch_of_coords)

    # transpose the batch of tensors
    batch_of_tensor_divergences = tensor_divergence(batch_of_tensors, batch_of_coords)

    # convert the tensors to NumPy arrays
    computed_divergences = to_numpy(batch_of_tensor_divergences)
    expected_divergences = to_numpy(batch_of_results)

    # make sure the transposed tensors are correct
    np.testing.assert_array_equal(computed_divergences, expected_divergences)


