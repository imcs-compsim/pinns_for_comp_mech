import numpy as np
import pytest

import compsim_pinns.hyperelasticity.hyperelasticity_utils as utils

from compsim_pinns.hyperelasticity.hyperelasticity_utils import (
    compute_elastic_properties,
    matrix_determinant_2D,
    matrix_inverse_2D,
)

# --------------------- Tests ---------------------


# Testing matrix_determinant_2D()


@pytest.mark.parametrize(
    "tensor_2d, determinant_2d",
    [
        (
            # Identity Matrix (No deformation)
            # tensor([[1.0, 0.0], [0.0, 1.0]]),
            [1.0, 1.0, 0.0, 0.0],
            1.0,
        ),
        (
            # Zero Matrix (Empty)
            # tensor([[0.0, 0.0], [0.0, 0.0]]),
            [0.0, 0.0, 0.0, 0.0],
            0.0,
        ),
        (
            # Singular Matrix (Linearly dependent rows/cols - Determinant is 0)
            # tensor([[2.0, 4.0], [1.0, 2.0]]),
            [2.0, 2.0, 4.0, 1.0],
            0.0,
        ),
        (
            # All Negative Components (Negative determinant result)
            # tensor([[-1.0, -2.0], [-3.0, -4.0]]),
            [-1.0, -4.0, -2.0, -3.0],
            -2.0,
        ),
        (
            # Floating Point / Decimal Precision Case
            # tensor([[0.5, 0.25], [0.2, 0.8]]),
            [0.5, 0.8, 0.25, 0.2],
            0.35,
        ),
    ],
)
def test_matrix_determinant_2D(tensor_2d, determinant_2d):
    result = matrix_determinant_2D(*tensor_2d)
    np_result = result

    np.testing.assert_almost_equal(np_result, determinant_2d)


# Testing matrix_inverse_2D() for solutions
@pytest.mark.parametrize(
    "tensor_2d, inverse_2d",
    [
        (
            # Identity Matrix (No deformation)
            # tensor([[1.0, 0.0], [0.0, 1.0]]),
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ),
        (
            # All Negative Components (Negative determinant result)
            # tensor([[-1.0, -2.0], [-3.0, -4.0]]),
            [-1.0, -4.0, -2.0, -3.0],
            [2.0, 0.5, -1.0, -1.5],
        ),
        (
            # Floating Point / Decimal Precision Case
            # tensor([[0.5, 0.25], [0.2, 0.8]]),
            [0.5, 0.8, 0.25, 0.2],
            [
                2.2857142857142856,
                1.4285714285714286,
                -0.7142857142857143,
                -0.5714285714285714,
            ],
        ),
    ],
)
def test_matrix_inverse_2D(tensor_2d, inverse_2d):
    result = matrix_inverse_2D(*tensor_2d)
    np_result = result

    np.testing.assert_almost_equal(np_result, inverse_2d)


# Testing matrix_inverse_2D() for error case: Determinant is zero
@pytest.mark.parametrize(
    "tensor_error_inverse_2d",
    [
        # Zero Matrix (Empty)
        # tensor([[0.0, 0.0], [0.0, 0.0]]),
        [0.0, 0.0, 0.0, 0.0],
        # Singular Matrix (Linearly dependent rows/cols - Determinant is 0)
        # tensor([[2.0, 4.0], [1.0, 2.0]]),
        [2.0, 2.0, 4.0, 1.0],
    ],
)
def test_matrix_inverse_2D_error(tensor_error_inverse_2d):

    with pytest.raises(ValueError):
        matrix_inverse_2D(*tensor_error_inverse_2d)


# Change the global variables with values from an input array
def change_global_variables(input_globals):
    utils.youngs_modulus = input_globals[0]
    utils.nu = input_globals[1]
    utils.shear = input_globals[2]
    utils.lame = input_globals[3]
    print(
        "Youngs modulus: ",
        utils.youngs_modulus,
        "Nu: ",
        utils.nu,
        "Shear: ",
        utils.shear,
        "Lame: ",
        utils.lame,
    )


# Combine the result after compute_elastic_properties function in an array
def result_global_variables():
    result = [utils.youngs_modulus, utils.nu, utils.shear, utils.lame]
    return result


# Testing compute_elastic_properties for error case: Less than two parameters are known
@pytest.mark.parametrize(
    "inputs_error_elastic_properties",
    [
        # No known parameter
        [None, None, None, None],
        # One known parameter
        [1.0, None, None, None],
        [None, 1.0, None, None],
        [None, None, 1.0, None],
        [None, None, None, 1.0],
    ],
)
def test_compute_elastic_properties_error_raised(inputs_error_elastic_properties):
    change_global_variables(inputs_error_elastic_properties)
    with pytest.raises(ValueError):
        compute_elastic_properties()


# Test compute_elastic_properties for real results, especially edge cases near errors e.g. division by zero
# Parameters: [youngs_modulus, nu, shear, lame]
@pytest.mark.parametrize(
    "input_known, output_all",
    [
        # Case 1: youngs_modulus and nu known (Inputs: [E, nu, None, None])
        (
            # Input:
            [1.0, 0.0, None, None],
            # Result:
            [1.0, 0.0, 0.5, 0.0],
        ),
        (
            [1.0, 0.49999999, None, None],
            [1.0, 0.49999999, 0.333333356, 16666666.444],
        ),
        (
            [1.0, -0.99999999, None, None],
            [1.0, -0.99999999, 50000000.0, -33333333.05],
        ),
        (
            [0.0, 0.0, None, None],
            [0.0, 0.0, 0.0, 0.0],
        ),
        (
            # Should be right, but it is not
            [1.0, 1.0, None, None],
            [1.0, 1.0, 0.25, -0.5],
        ),
        # Case 2: nu and shear known (Inputs: [None, nu, shear, None])
        (
            [None, 0.4999999, 1.0, None],
            [2.9999998, 0.4999999, 1.0, 4999999.0],
        ),
        (
            [None, -0.99999999, 1.0, None],
            [2e-08, -0.99999999, 1.0, -0.666666664],
        ),
        (
            [None, -1.0, 1.0, None],
            [0, -0.99999999, 1.0, -0.666666667],
        ),
        (
            [None, 0.0, 0.0, None],
            [0.0, 0.0, 0.0, 0.0],
        ),
        # Case 3: youngs_modulus and shear known (Inputs: [youngs_modulus, None, shear, None])
        (
            [1.0, None, 0.5, None],
            [1.0, 0.0, 0.5, 0.0],
        ),
        (
            [1.0, None, 1e-08, None],
            [1.0, 49999999, 1e-08, -1e-08],
        ),
        # Case 4: shear and lame known (Inputs: [None, None, shear, lame])
        (  # Error raised for lame even it is not changed
            [None, None, 0.0, 1e-08],
            [0.0, 0.5, 0.0, 1e-08],
        ),
        (
            [None, None, 1e-08, 0.0],
            [2e-08, 0.0, 1e-08, 0.0],
        ),
        # Case 5: youngs_modulus and lame known (Inputs: [youngs_modulus, None, None, lame])
        (
            [1.0, None, None, 0.0],
            [1.0, 0.0, 0.5, 0.0],
        ),
        (
            [1e-08, None, None, 0.0],
            [1e-08, 0.0, 5e-09, 0.0],
        ),
        # Case 6: nu and lame known (Inputs: [None, nu, None, lame])
        (
            [None, 1e-08, None, 1.0],
            [99999999.0, 1e-08, 49999999.0, 1.0],
        ),
        (
            [None, 1.0, None, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ),
        (
            [None, 0.5, None, 1.0],
            [0.0, 0.5, 0.0, 1.0],
        ),
    ],
)
def test_compute_elastic_properties(input_known, output_all):
    change_global_variables(input_known)
    print(
        "Youngs modulus: ",
        utils.youngs_modulus,
        "Nu: ",
        utils.nu,
        "Shear: ",
        utils.shear,
        "Lame: ",
        utils.lame,
    )
    compute_elastic_properties()
    global_results = result_global_variables()

    np.testing.assert_allclose(global_results, output_all)


# Test elastic properties for errors division by zero
@pytest.mark.parametrize(
    "input_error_zeros_elastic_properties",
    [
        # Case 1: youngs_modulus and nu known (Inputs: [E, nu, None, None])
        [0.0, -1.0, None, None],
        [2.0, 0.5, None, None],
        # [1.0, 1.0, None, None],
        # Case 2: nu and shear known (Inputs: [None, nu, shear, None])
        [None, 0.5, 1.0, None],
        # Case 3: youngs_modulus and shear known (Inputs: [youngs_modulus, None, shear, None])
        [3.0, None, 1.0, None],
        [1.0, None, 0.0, None],
        [0.0, None, 0.0, None],
        # Case 4: shear and lame known (Inputs: [None, None, shear, lame])
        [None, None, 1.0, -1.0],
        [None, None, -1.0, 1.0],
        [None, None, 0.0, 0.0],
        # Case 5: youngs_modulus and lame known (Inputs: [youngs_modulus, None, None, lame])
        [1.0, None, None, 1.0],
        [0.0, None, None, 1.0],
        [0.0, None, None, 0.0],
        [0.0, None, None, 1e-08],
        # Case 6: nu and lame known (Inputs: [None, nu, None, lame])
        [None, 0.0, None, 1.0],
    ],
)
def test_compute_elastic_properties_zero_error_raised(
    input_error_zeros_elastic_properties,
):
    change_global_variables(input_error_zeros_elastic_properties)
    with pytest.raises(ZeroDivisionError):
        compute_elastic_properties()


# Functions to test:
# - compute_elastic_properties()                                                        Done
# - bkd_log(x):                                     has to be different for every backend of the function, idk how to test this function
# - matrix_determinant_2D()                         see above testing with the different matrix                 Done
# - matrix_inverse_2D(a,b,c,d)                      what would happen, if the matrix is not invertible
# - matrix_determinant_3D(a,b,c,d,e,f,g,h,i,j)      same as matrix_determinant_2D() only with more entries
# - matrix_inverse_3D(a,b,c,d,e,f,g,h,i,j)
# - deformation_gradient_2D(x,y)
# - deformation_gradient_3D(x,y)
# - strain_energy_neo_hookean_2D(x,y)
# - strain_energy_neo_hookean_3D(x,y)
# - second_piola_stress_tensor_2D(x,y)
# - first_piola_stress_tensor_2D(x,y)
# - cauchy_stress_2d(x,y)
# - green_lagrange_strain_2D(x, y)
# - second_piola_stress_tensor_3D(x, y)
# - first_piola_stress_tensor_3D(x, y)
# - cauchy_stress_3D(x, y)
# - green_lagrange_strain_3D(x, y)

# If a programmer finds themselves in a situation where they have to test their own code, they need to be aware of this issue.
