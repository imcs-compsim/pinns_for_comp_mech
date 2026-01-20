
import numpy as np
import pytest

from compsim_pinns.hyperelasticity.hyperelasticity_utils import matrix_determinant_2D


# --------------------- Tests ---------------------

@pytest.mark.parametrize(
    "tensor_2d, determinant_2d",
    [
        (
            # tensor([[1.0, 2.0], [3.0, 4.0]]),
            [1.0, 4.0, 2.0, 3.0],
            -2.0,
        ),
        (
            # tensor([[5.0, 6.0], [7.0, 8.0]]),
            [5.0, 8.0, 6.0, 7.0],
            -2.0,
        ),
        (
            # tensor([[1.0, 1.5], [0.0, 1.5]]),
            [1.0, 1.5, 1.5, 0.0],
            1.5
        ),
    ],
)
def test_matrix_determinant_2D(tensor_2d, determinant_2d):
    result = matrix_determinant_2D(*tensor_2d)
    np_result = result

    np.testing.assert_almost_equal(np_result, determinant_2d)