import deepxde.backend as bkd
import numpy as np
import pytest

from utils.elasticity.elasticity_utils_new import strain
from utils.linalg.linalg_utils import transpose


# --------------------- Fixtures ---------------------

@pytest.fixture
def list_of_2d_tensors():
    temp = bkd.as_tensor(
        [
            [[1.0, 3.0], [2.0, 1.0]],
            # [[5.0, 6.0], [7.0, 8.0]],
            # [[9.0, 10.0], [11.0, 12.0]],
        ]
    )
    return temp

@pytest.fixture
def list_of_2d_strains():
    temp = bkd.as_tensor(
        [
            [[3.5, 5.0], [5.0, 6.0]],
            # [[5.0, 6.0], [7.0, 8.0]],
            # [[9.0, 10.0], [11.0, 12.0]],
        ]
    )
    return temp

@pytest.fixture
def list_of_2d_linearized_strains():
    temp = bkd.as_tensor(
        [
            [[1.0, 2.5], [2.5, 1.0]],
            # [[5.0, 6.0], [7.0, 8.0]],
            # [[9.0, 10.0], [11.0, 12.0]],
        ]
    )
    return temp


# --------------------- Tests ---------------------

@pytest.mark.parametrize(
        "batch_of_tensors, batch_of_results",
    [
        ("list_of_2d_tensors", "list_of_2d_strains"),
        # ("list_of_3d_tensors", "list_of_3d_strains"),
    ]
)
def test_elasticity_strain(batch_of_tensors, batch_of_results, request):
    # since fixtures can't be passed in parameterized tests directly, we need 
    # to provide the name of the fixture as string and then retrieve their 
    # values by using the special request fixture which is provided by pytest
    batch_of_tensors = request.getfixturevalue(batch_of_tensors)
    batch_of_results = request.getfixturevalue(batch_of_results)

    # compute the strains
    batch_of_strains = strain(batch_of_tensors)

    # check whether the results are correct
    np.testing.assert_allclose(batch_of_strains, batch_of_results)


@pytest.mark.parametrize(
        "batch_of_tensors, batch_of_results",
    [
        ("list_of_2d_tensors", "list_of_2d_linearized_strains"),
        # ("list_of_3d_tensors", "list_of_3d_strains"),
    ]
)
def test_elasticity_strain_linearized(batch_of_tensors, batch_of_results, request):
    # since fixtures can't be passed in parameterized tests directly, we need 
    # to provide the name of the fixture as string and then retrieve their 
    # values by using the special request fixture which is provided by pytest
    batch_of_tensors = request.getfixturevalue(batch_of_tensors)
    batch_of_results = request.getfixturevalue(batch_of_results)

    # compute the strains
    batch_of_strains = strain(batch_of_tensors, linearize=True)

    # check whether the results are correct
    np.testing.assert_allclose(batch_of_strains, batch_of_results)

    # make sure the results are symmetric
    np.testing.assert_array_equal(batch_of_strains, transpose(batch_of_strains))