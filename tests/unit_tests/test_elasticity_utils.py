import deepxde as dde
import deepxde.backend as bkd
import numpy as np
import pytest

from utils.elasticity.elasticity_utils_new import deformation_gradient, \
    displacement_gradient, strain
from utils.linalg.linalg_utils import transpose

from conftest import to_numpy


# --------------------- Functions ---------------------    

def linear_motion(coords):
    """A simple linear motion field.
    
    Example 4.2 from "Nonlinear continuum mechanics for finite element analysis" 
    by J. Bonet and R. D. Wood.
    """
    x1 = 0.25*(18.0 + 4.0*coords[:,0] + 6.0*coords[:,1])
    x2 = 0.25*(14.0 + 6.0*coords[:,1])
    return bkd.stack([x1, x2], axis=1)


def displacement(coords):
    if bkd.backend_name in ["tensorflow.compat.v1", "tensorflow"]:
        with bkd.lib.GradientTape():
            _disp = linear_motion(coords) - coords
    return _disp


# --------------------- Fixtures ---------------------

@pytest.fixture
def list_of_coords():
    temp = bkd.as_tensor(
        [
            bkd.Variable([1.0, -1.0]),
            bkd.Variable([1.0, 1.0]),
            bkd.Variable([-1.0, 1.0]),
            bkd.Variable([-1.0, -1.0]),
        ]
    )
    return temp

@pytest.fixture
def list_of_2d_displacement_gradients():
    """The displacement gradients as resulting from the motion defined above 
    independent of the displacement (as it's a linear uniform motion)."""
    temp = bkd.as_tensor(
        [
            [[0.0, 1.5], [0.0, 0.5]],
            [[0.0, 1.5], [0.0, 0.5]],
            [[0.0, 1.5], [0.0, 0.5]],
            [[0.0, 1.5], [0.0, 0.5]],
        ]
    )
    return temp

@pytest.fixture
def list_of_2d_deformation_gradients():
    """The deformation gradients as resulting from the motion defined above 
    independent of the displacement (as it's a linear uniform motion)."""
    temp = bkd.as_tensor(
        [
            [[1.0, 1.5], [0.0, 1.5]],
            [[1.0, 1.5], [0.0, 1.5]],
            [[1.0, 1.5], [0.0, 1.5]],
            [[1.0, 1.5], [0.0, 1.5]],
        ]
    )
    return temp

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

def test_elasticity_displacement_gradient(list_of_coords, list_of_2d_displacement_gradients):
    # compute the displacement
    disp = linear_motion(list_of_coords) - list_of_coords
    # compute the displacement gradient
    disp_grad = displacement_gradient(disp, list_of_coords)

    # convert the tensors to NumPy arrays
    computed_disp_grad = to_numpy(disp_grad)
    expected_disp_grad = to_numpy(list_of_2d_displacement_gradients)
    
    # check whether the results are correct
    np.testing.assert_allclose(computed_disp_grad, expected_disp_grad)


def test_elasticity_deformation_gradient(list_of_coords, list_of_2d_deformation_gradients):
    # compute the displacement
    disp = linear_motion(list_of_coords) - list_of_coords
    # compute the displacement gradient
    def_grad = deformation_gradient(disp, list_of_coords)

    # convert the tensors to NumPy arrays
    computed_deformation_gradient = to_numpy(def_grad)
    expected_deformation_gradient = to_numpy(list_of_2d_deformation_gradients)
    
    # check whether the results are correct
    np.testing.assert_allclose(computed_deformation_gradient, expected_deformation_gradient)


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

    # convert the tensors to NumPy arrays
    computed_strains = to_numpy(batch_of_strains)
    expected_strains = to_numpy(batch_of_results)

    # check whether the results are correct
    np.testing.assert_allclose(computed_strains, expected_strains)

    # make sure the results are symmetric
    batch_of_transposed_strains = transpose(batch_of_strains)
    computed_transposed_strains = to_numpy(batch_of_transposed_strains)
    np.testing.assert_array_equal(computed_strains, computed_transposed_strains)


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

    # convert the tensors to NumPy arrays
    computed_strains = to_numpy(batch_of_strains)
    expected_strains = to_numpy(batch_of_results)

    # check whether the results are correct
    np.testing.assert_allclose(computed_strains, expected_strains)

    # make sure the results are symmetric
    batch_of_transposed_strains = transpose(batch_of_strains)
    computed_transposed_strains = to_numpy(batch_of_transposed_strains)
    np.testing.assert_array_equal(computed_strains, computed_transposed_strains)