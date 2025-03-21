import deepxde.backend as bkd
import numpy as np
import pytest

from utils.linalg.linalg_utils import determinant, identity, identity_like, \
    inverse, transpose


# --------------------- Fixtures ---------------------

@pytest.fixture
def identity_2d():
    temp = bkd.as_tensor([[1.0, 0.0], [0.0, 1.0]])
    return temp

@pytest.fixture
def identity_3d():
    temp = bkd.as_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return temp

@pytest.fixture
def list_of_2d_tensors():
    temp = bkd.as_tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ]
    )
    return temp

@pytest.fixture
def list_of_2d_tensors_determinant():
    temp = bkd.as_tensor(
        [
            -2.0,
            -2.0,
            -2.0,
        ]
    )
    return temp

@pytest.fixture
def list_of_2d_tensors_inverted(list_of_2d_tensors_determinant):
    # tensor with flipped signs on the off-diagonal entries
    temp = bkd.as_tensor(
        [
            [[4.0, -2.0],[-3.0, 1.0]],
            [[8.0, -6.0],[-7.0, 5.0]],
            [[12.0, -10.0],[-11.0, 9.0]],
        ]
    )
    # scale with inverse of determinant to obtain the inverted tensor
    temp /= bkd.reshape(list_of_2d_tensors_determinant, (-1, 1, 1))
    return temp

@pytest.fixture
def list_of_2d_tensors_transposed():
    temp = bkd.as_tensor(
        [
            [[1.0, 3.0], [2.0, 4.0]],
            [[5.0, 7.0], [6.0, 8.0]],
            [[9.0, 11.0], [10.0, 12.0]],
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
def list_of_3d_tensors_transposed():
    temp = bkd.as_tensor(
        [
            [[1.0, 0.0, 4.0], [0.0, 3.0, 0.0], [2.0, 0.0, 5.0]],
            [[1.0, 0.0, 0.0], [2.0, 4.0, 0.0], [3.0, 5.0, 6.0]],
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

    # make sure the transposed tensors are correct
    np.testing.assert_array_equal(batch_of_transposed_tensors, batch_of_results)


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

    # make sure the determinants are correct
    np.testing.assert_allclose(batch_of_tensor_determinants, batch_of_results, rtol=1e-6)


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

    # make sure the transposed tensors are correct
    np.testing.assert_allclose(batch_of_inverted_tensors, batch_of_results, rtol=1e-6)


@pytest.mark.parametrize(
    "dim, result",
    [
        (2, "identity_2d"),
        (3, "identity_3d"),
    ]
)
def test_linalg_identity(dim, result, request):
    # since fixtures can't be passed in parameterized tests directly, we need 
    # to provide the name of the fixture as string and then retrieve their 
    # values by using the special request fixture which is provided by pytest
    result = request.getfixturevalue(result)

    # generate the identity tensor
    identity_tensor = identity(dim)

    # make sure the identity tensor is correct
    np.testing.assert_array_equal(identity_tensor, result)


@pytest.mark.parametrize(
    "batch_of_tensors, result",
    [
        ("list_of_2d_tensors", "identity_2d"),
        ("list_of_3d_tensors", "identity_3d"),
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

    # make sure the identity tensor is correct
    np.testing.assert_array_equal(identity_tensor, result)