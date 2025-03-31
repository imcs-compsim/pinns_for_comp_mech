import deepxde.backend as bkd
import numpy as np
import pytest

from utils.linalg.linalg_utils import determinant, identity, identity_like, \
    inverse, outer_vec_mat_prod, trace, transpose

from conftest import to_numpy


# --------------------- Fixtures ---------------------

@pytest.fixture
def identity2d():
    temp = bkd.as_tensor([[1.0, 0.0], [0.0, 1.0]])
    return temp

@pytest.fixture
def identity3d():
    temp = bkd.as_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return temp

@pytest.fixture
def vector2d():
    return bkd.as_tensor([-1.0, 2.0])

@pytest.fixture
def tensor2d(list_of_2d_tensors):
    return list_of_2d_tensors[0]

@pytest.fixture
def list_of_2d_tensors():
    temp = bkd.as_tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[1.0, 1.5], [0.0, 1.5]],
        ]
    )
    return temp

@pytest.fixture
def list_of_2d_tensors_determinant():
    temp = bkd.as_tensor([-2.0, -2.0, 1.5,])
    return temp

@pytest.fixture
def list_of_2d_tensors_inverted():
    temp = bkd.as_tensor(
        [
            [[-2.0, 1.0],[1.5, -0.5]],
            [[-4.0, 3.0],[3.5, -2.5]],
            [[1.0, -1.0],[0.0, 2.0/3.0]],
        ]
    )
    return temp

@pytest.fixture
def list_of_2d_tensor_traces():
    temp = bkd.as_tensor([5.0, 13.0, 2.5,])
    return temp

@pytest.fixture
def list_of_2d_tensors_transposed():
    temp = bkd.as_tensor(
        [
            [[1.0, 3.0], [2.0, 4.0]],
            [[5.0, 7.0], [6.0, 8.0]],
            [[1.0, 0.0], [1.5, 1.5]],
        ]
    )
    return temp

@pytest.fixture
def list_of_3d_tensors():
    temp = bkd.as_tensor(
        [
            [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]],
            [[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]],
        ]
    )
    return temp

@pytest.fixture
def list_of_3d_tensors_determinant():
    temp = bkd.as_tensor(
        [
            -9.0,
            24.0,
        ]
    )
    return temp

@pytest.fixture
def list_of_3d_tensors_inverted(list_of_3d_tensors_determinant):
    # tensor with the transposed tensors of co-factors
    # https://metric.ma.ic.ac.uk/metric_public/matrices/inverses/inverses2.html
    temp = bkd.as_tensor(
        [
            [[15.0, 0.0, -6.0], [0.0, -3.0, 0.0], [-12.0, 0.0, 3.0]],
            [[24.0, -12.0, -2.0], [0.0, 6.0, -5.0], [0.0, 0.0, 4.0]],
        ]
    )
    # scale with inverse of determinant to obtain the inverted tensor
    temp /= bkd.reshape(list_of_3d_tensors_determinant, (-1, 1, 1))
    return temp

@pytest.fixture
def list_of_3d_tensors_traces():
    temp = bkd.as_tensor([9.0, 11.0])
    return temp

@pytest.fixture
def list_of_3d_tensors_transposed():
    temp = bkd.as_tensor(
        [
            [[1.0, 0.0, 4.0], [0.0, 3.0, 0.0], [2.0, 0.0, 5.0]],
            [[1.0, 0.0, 0.0], [2.0, 4.0, 0.0], [3.0, 5.0, 6.0]],
        ]
    )
    return temp

@pytest.fixture
def outer_vec_mat_prod_res():
    temp = bkd.as_tensor(
        [
            [[-1.0, -2.0], [-3.0, -4.0]],
            [[2.0, 4.0], [6.0, 8.0]]
        ]
    )
    return temp


# --------------------- Tests ---------------------

@pytest.mark.parametrize(
    "batch_of_tensors, batch_of_results",
    [
        ("list_of_2d_tensors", "list_of_2d_tensors_transposed"),
        ("list_of_3d_tensors", "list_of_3d_tensors_transposed"),
    ]
)
def test_linalg_transpose(batch_of_tensors, batch_of_results, request):
    # since fixtures can't be passed in parameterized tests directly, we need 
    # to provide the name of the fixture as string and then retrieve their 
    # values by using the special request fixture which is provided by pytest
    batch_of_tensors = request.getfixturevalue(batch_of_tensors)
    batch_of_results = request.getfixturevalue(batch_of_results)

    # transpose the batch of tensors
    batch_of_transposed_tensors = transpose(batch_of_tensors)

    # convert the tensors to NumPy arrays
    computed_transposed = to_numpy(batch_of_transposed_tensors)
    expected_transposed = to_numpy(batch_of_results)

    # make sure the transposed tensors are correct
    np.testing.assert_array_equal(computed_transposed, expected_transposed)


@pytest.mark.parametrize(
    "batch_of_tensors, batch_of_results",
    [
        ("list_of_2d_tensors", "list_of_2d_tensors_determinant"),
        ("list_of_3d_tensors", "list_of_3d_tensors_determinant"),
    ]
)
def test_linalg_determinant(batch_of_tensors, batch_of_results, request):
    # since fixtures can't be passed in parameterized tests directly, we need 
    # to provide the name of the fixture as string and then retrieve their 
    # values by using the special request fixture which is provided by pytest
    batch_of_tensors = request.getfixturevalue(batch_of_tensors)
    batch_of_results = request.getfixturevalue(batch_of_results)

    # compute the determinants of the batch of tensors
    batch_of_tensor_determinants = determinant(batch_of_tensors)

    # convert the tensors to NumPy arrays
    computed_determinants = to_numpy(batch_of_tensor_determinants)
    expected_determinants = to_numpy(batch_of_results)

    # make sure the determinants are correct
    np.testing.assert_allclose(computed_determinants, expected_determinants, rtol=1e-6)


@pytest.mark.parametrize(
    "batch_of_tensors, batch_of_results",
    [
        ("list_of_2d_tensors", "list_of_2d_tensors_inverted"),
        ("list_of_3d_tensors", "list_of_3d_tensors_inverted"),
    ]
)
def test_linalg_inverse(batch_of_tensors, batch_of_results, request):
    # since fixtures can't be passed in parameterized tests directly, we need 
    # to provide the name of the fixture as string and then retrieve their 
    # values by using the special request fixture which is provided by pytest
    batch_of_tensors = request.getfixturevalue(batch_of_tensors)
    batch_of_results = request.getfixturevalue(batch_of_results)

    # invert the batch of tensors
    batch_of_inverted_tensors = inverse(batch_of_tensors)

    # convert the tensors to NumPy arrays
    computed_inverse = to_numpy(batch_of_inverted_tensors)
    expected_inverse = to_numpy(batch_of_results)

    # make sure the transposed tensors are correct
    np.testing.assert_allclose(computed_inverse, expected_inverse, rtol=1e-6)


@pytest.mark.parametrize(
    "dim, result",
    [
        (2, "identity2d"),
        (3, "identity3d"),
    ]
)
def test_linalg_identity(dim, result, request):
    # since fixtures can't be passed in parameterized tests directly, we need 
    # to provide the name of the fixture as string and then retrieve their 
    # values by using the special request fixture which is provided by pytest
    result = request.getfixturevalue(result)

    # generate the identity tensor
    identity_tensor = identity(dim)

    # convert the tensors to NumPy arrays
    computed_identity = to_numpy(identity_tensor)
    expected_identity = to_numpy(result)

    # make sure the identity tensor is correct
    np.testing.assert_array_equal(computed_identity, expected_identity)


@pytest.mark.parametrize(
    "batch_of_tensors, result",
    [
        ("list_of_2d_tensors", "identity2d"),
        ("list_of_3d_tensors", "identity3d"),
    ]
)
def test_linalg_identity_like(batch_of_tensors, result, request):
    # since fixtures can't be passed in parameterized tests directly, we need 
    # to provide the name of the fixture as string and then retrieve their 
    # values by using the special request fixture which is provided by pytest
    batch_of_tensors = request.getfixturevalue(batch_of_tensors)
    result = request.getfixturevalue(result)
    
    # generate the identity tensor
    identity_tensor = identity_like(batch_of_tensors)

    # convert the tensors to NumPy arrays
    computed_identity = to_numpy(identity_tensor)
    expected_identity = to_numpy(result)

    # make sure the identity tensor is correct
    np.testing.assert_array_equal(computed_identity, expected_identity)


@pytest.mark.parametrize(
    "batch_of_tensors, batch_of_results",
    [
        ("list_of_2d_tensors", "list_of_2d_tensor_traces"),
        ("list_of_3d_tensors", "list_of_3d_tensors_traces"),
    ]
)
def test_linalg_trace(batch_of_tensors, batch_of_results, request):
    # since fixtures can't be passed in parameterized tests directly, we need 
    # to provide the name of the fixture as string and then retrieve their 
    # values by using the special request fixture which is provided by pytest
    batch_of_tensors = request.getfixturevalue(batch_of_tensors)
    batch_of_results = request.getfixturevalue(batch_of_results)

    # compute the traces of the batch of tensors
    batch_of_tensor_traces = trace(batch_of_tensors)

    # convert the tensors to NumPy arrays
    computed_traces = to_numpy(batch_of_tensor_traces)
    expected_traces = to_numpy(batch_of_results)

    # make sure the identity tensor is correct
    np.testing.assert_array_equal(computed_traces, expected_traces)


@pytest.mark.parametrize(
    "vector, matrix, result",
    [
        ("vector2d", "tensor2d", "outer_vec_mat_prod_res"),
    ]
)
def test_linalg_outer_vec_mat_prod(vector, matrix, result, request):
    # since fixtures can't be passed in parameterized tests directly, we need 
    # to provide the name of the fixture as string and then retrieve their 
    # values by using the special request fixture which is provided by pytest
    vector = request.getfixturevalue(vector)
    matrix = request.getfixturevalue(matrix)
    result = request.getfixturevalue(result)

    # compute the outer product of the batch of vectors with the batch of matrices
    outer_product = outer_vec_mat_prod(vector, matrix)

    # convert the tensors to NumPy arrays
    computed_outer_product = to_numpy(outer_product)
    expected_outer_product = to_numpy(result)

    # make sure the outer products are correct
    np.testing.assert_array_equal(computed_outer_product, expected_outer_product)