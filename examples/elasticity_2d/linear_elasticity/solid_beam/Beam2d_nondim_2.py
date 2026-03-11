"""
2D Linear Elasticity PINN for Cantilever Beam Under Shear Load

This module implements a Physics-Informed Neural Network (PINN) to solve a 2D linear
elasticity problem for a cantilever beam subjected to distributed shear loading. The solution
is non-dimensionalized using characteristic length, stress, and displacement scales.

The beam is fixed at the left end and has traction-free boundary conditions on the top and
bottom surfaces, with applied shear on the right end. The PINN learns to predict displacement
and stress fields by satisfying equilibrium equations and constitutive relations in the domain,
along with Dirichlet and Neumann boundary conditions.

Reference solution:
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.504.4507&rep=rep1&type=pdf

Reference for PINNs formulation:
A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics

@author: tsahin
"""

import os

os.environ["DDE_BACKEND"] = "tensorflow.compat.v1"

import os

import deepxde as dde
import numpy as np
from deepxde.backend import tf
from pyevtk.hl import unstructuredGridToVTK

from compsim_pinns.elasticity import elasticity_utils
from compsim_pinns.elasticity.elasticity_utils import (
    problem_parameters,
    stress_plane_stress,
)
from compsim_pinns.geometry.custom_geometry import GmshGeometry2D
from compsim_pinns.geometry.geometry_utils import calculate_boundary_normals
from compsim_pinns.geometry.gmsh_models import Block_2D

gmsh_options = {"General.Terminal": 1, "Mesh.Algorithm": 6}
block_2d = Block_2D(
    coord_left_corner=[0, -0.5],
    coord_right_corner=[4, 0.5],
    mesh_size=0.05,
    gmsh_options=gmsh_options,
)

gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)

# The correct order for the normals --> 1 2 1 1
revert_curve_list = []
revert_normal_dir_list = [1, 2, 1, 1]
geom = GmshGeometry2D(
    gmsh_model,
    revert_curve_list=revert_curve_list,
    revert_normal_dir_list=revert_normal_dir_list,
)

l = block_2d.coord_right_corner[0] - block_2d.coord_left_corner[0]
h = block_2d.coord_right_corner[1] - block_2d.coord_left_corner[1]

# change global variables in elasticity_utils, they are used for getting the material properties for analytical model
e_1 = 2000
nu_1 = 0.3
elasticity_utils.lame = e_1 * nu_1 / ((1 + nu_1) * (1 - 2 * nu_1))
elasticity_utils.shear = e_1 / (2 * (1 + nu_1))
nu_analy, lame_analy, shear_analy, youngs_modulus_analy = (
    problem_parameters()
)  # with dimensions, will be used for analytical solution

# applied shear
shear_y = 0.01

# characteristic quantities which are used for non-dimensionalization
characteristic_nu = elasticity_utils.shear  # characteristic shear modulus
characteristic_disp = 1 / youngs_modulus_analy  # characteristic displacement
characteristic_length = 4  # characteristic length
characteristic_stress = characteristic_nu * characteristic_disp / characteristic_length

elasticity_utils.lame = (
    elasticity_utils.lame / characteristic_nu
)  # non-dimensionalized, used for PINNs
elasticity_utils.shear = (
    elasticity_utils.shear / characteristic_nu
)  # non-dimensionalized, used for PINNs

# non-dimensionalized shear
shear_y_norm = shear_y / characteristic_stress  # non-dimensionalized, used for PINNs
Inertia = 1 / 12 * h**3

elasticity_utils.geom = geom


def pde_stress(x, y):
    # governing equation
    """Compute the residual of `pde_stress` for the governing equation.

    Args:
        x: Input coordinates used to evaluate the function.
        y: Field values or model outputs associated with `x`.

    Returns:
        Any: Computed value returned by `pde_stress`.
    """
    sigma_xx_x = dde.grad.jacobian(y, x, i=2, j=0)
    sigma_yy_y = dde.grad.jacobian(y, x, i=3, j=1)
    sigma_xy_x = dde.grad.jacobian(y, x, i=4, j=0)
    sigma_xy_y = dde.grad.jacobian(y, x, i=4, j=1)

    momentum_x = sigma_xx_x + sigma_xy_y
    momentum_y = sigma_yy_y + sigma_xy_x

    # material law
    term_x, term_y, term_xy = lin_iso_elasticity_plane_stress(x, y)

    return [momentum_x, momentum_y, term_x, term_y, term_xy]


def lin_iso_elasticity_plane_stress(x, y):
    """Compute lin iso elasticity plane stress for this example setup.

    Args:
        x: Input coordinates used to evaluate the function.
        y: Field values or model outputs associated with `x`.

    Returns:
        Any: Computed value returned by `lin_iso_elasticity_plane_stress`.
    """
    sigma_xx, sigma_yy, sigma_xy = stress_plane_stress(x, y)

    term_x = sigma_xx - y[:, 2:3]
    term_y = sigma_yy - y[:, 3:4]
    term_xy = sigma_xy - y[:, 4:5]

    return term_x, term_y, term_xy


def zero_neumann_x(x, y, X):
    """Compute zero neumann x for this example setup.

    Args:
        x: Input coordinates used to evaluate the function.
        y: Field values or model outputs associated with `x`.
        X: Input coordinates used by this callback.

    Returns:
        Any: Computed value returned by `zero_neumann_x`.
    """
    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5]

    normals, cond = calculate_boundary_normals(X, geom)

    sigma_xx_n_x = sigma_xx[cond] * normals[:, 0:1]
    sigma_xy_n_y = sigma_xy[cond] * normals[:, 1:2]

    traction_x = sigma_xx_n_x + sigma_xy_n_y

    return traction_x


def zero_neumann_y(x, y, X):
    """Compute zero neumann y for this example setup.

    Args:
        x: Input coordinates used to evaluate the function.
        y: Field values or model outputs associated with `x`.
        X: Input coordinates used by this callback.

    Returns:
        Any: Computed value returned by `zero_neumann_y`.
    """
    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5]

    normals, cond = calculate_boundary_normals(X, geom)

    sigma_yx_n_x = sigma_xy[cond] * normals[:, 0:1]
    sigma_yy_n_y = sigma_yy[cond] * normals[:, 1:2]

    traction_y = sigma_yx_n_x + sigma_yy_n_y

    return traction_y


def neumann_x(x, y, X):
    """
    Represents the x component of the applied pressure
    """

    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5]

    normals, cond = calculate_boundary_normals(X, geom)

    return sigma_xx[cond]


def neumann_y(x, y, X):
    """
    Represents the y component of the applied pressure
    """

    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5]

    normals, cond = calculate_boundary_normals(X, geom)

    sigma_yx_n_x = sigma_xy[cond]

    y_loc = x[:, 1:2][cond]

    return sigma_yx_n_x + shear_y_norm / (2 * Inertia) * (y_loc - h / 2) * (
        y_loc + h / 2
    ) * (-1)  # *normals[:,0:1]


nu, lame, shear, youngs_modulus = problem_parameters()


def fun_u_x(x):
    """Compute fun u x for this example setup.

    Args:
        x: Input coordinates used to evaluate the function.

    Returns:
        Any: Computed value returned by `fun_u_x`.
    """
    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    u_x_analy = (
        shear_y_norm
        * y_loc
        / (6 * youngs_modulus * Inertia)
        * (2 + nu)
        * (y_loc**2 - h**2 / 4)
    )

    return u_x_analy


def fun_u_y(x):
    """Compute fun u y for this example setup.

    Args:
        x: Input coordinates used to evaluate the function.

    Returns:
        Any: Computed value returned by `fun_u_y`.
    """
    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    u_y_analy = -shear_y_norm / (6 * youngs_modulus * Inertia) * (3 * nu * y_loc**2 * l)

    return u_y_analy


def top_bottom(x, on_boundary):
    """Check whether a point satisfies the `top_bottom` boundary condition.

    Args:
        x: Input coordinates used to evaluate the function.
        on_boundary: Boundary indicator provided by the geometry callback.

    Returns:
        bool: Result of the `top_bottom` evaluation.
    """
    points_top = np.logical_and(np.isclose(x[1], h / 2), ~np.isclose(x[0], l))
    points_bottom = np.logical_and(np.isclose(x[1], -h / 2), ~np.isclose(x[0], l))

    return on_boundary and np.logical_or(points_top, points_bottom)


def left(x, on_boundary):
    """Check whether a point satisfies the `left` boundary condition.

    Args:
        x: Input coordinates used to evaluate the function.
        on_boundary: Boundary indicator provided by the geometry callback.

    Returns:
        bool: Result of the `left` evaluation.
    """
    return on_boundary and np.isclose(x[0], 0)


def right(x, on_boundary):
    """Check whether a point satisfies the `right` boundary condition.

    Args:
        x: Input coordinates used to evaluate the function.
        on_boundary: Boundary indicator provided by the geometry callback.

    Returns:
        bool: Result of the `right` evaluation.
    """
    return on_boundary and np.isclose(x[0], l)


bc1 = dde.DirichletBC(geom, fun_u_x, left, component=0)
bc2 = dde.DirichletBC(geom, fun_u_y, left, component=1)
bc3 = dde.OperatorBC(geom, zero_neumann_x, top_bottom)
bc4 = dde.OperatorBC(geom, zero_neumann_y, top_bottom)
bc5 = dde.OperatorBC(geom, neumann_x, right)
bc6 = dde.OperatorBC(geom, neumann_y, right)

n_dummy = 1
data = dde.data.PDE(
    geom,
    pde_stress,
    [bc1, bc2, bc5, bc6],
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=n_dummy,
    train_distribution="Sobol",
)


# non-dimensionalize the input using characteristic length
def input_transform(x):
    """Compute input transform for this example setup.

    Args:
        x: Input coordinates used to evaluate the function.

    Returns:
        Any: Computed value returned by `input_transform`.
    """
    return tf.concat(
        [x[:, 0:1] / characteristic_length, x[:, 1:2] / characteristic_length], axis=1
    )


def output_transform(x, y):
    """Compute output transform for this example setup.

    Args:
        x: Input coordinates used to evaluate the function.
        y: Field values or model outputs associated with `x`.

    Returns:
        Any: Computed value returned by `output_transform`.
    """
    u = y[:, 0:1]
    v = y[:, 1:2]
    sigma_xx = y[:, 2:3]
    sigma_yy = y[:, 3:4]
    sigma_xy = y[:, 4:5]
    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]

    return tf.concat(
        [
            u,
            v,
            sigma_xx,
            sigma_yy * (-h / 2 + y_loc) * (h / 2 + y_loc),
            sigma_xy * (-h / 2 + y_loc) * (h / 2 + y_loc),
        ],
        axis=1,
    )  # [ u*1e-2, v*1e-3]


# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 4 + [5]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_feature_transform(input_transform)
net.apply_output_transform(output_transform)
model = dde.Model(data, net)

model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=5000, display_every=200)

model.compile("L-BFGS")
model.train()

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

X, offset, cell_types, dol_triangles = geom.get_mesh()

output = model.predict(X)
u_pred, v_pred = (
    output[:, 0] * characteristic_disp / characteristic_length,
    output[:, 1] * characteristic_disp / characteristic_length,
)
sigma_xx, sigma_yy, sigma_xy = (
    output[:, 2] * characteristic_stress,
    output[:, 3] * characteristic_stress,
    output[:, 4] * characteristic_stress,
)

x = X[:, 0:1]
y = X[:, 1:2]
Inertia = 1 / 12 * h**3

u_x_analy = (
    shear_y
    * y
    / (6 * youngs_modulus_analy * Inertia)
    * ((6 * l - 3 * x) * x + (2 + nu_analy) * (y**2 - h**2 / 4))
)
u_y_analy = (
    -shear_y
    / (6 * youngs_modulus_analy * Inertia)
    * (
        3 * nu_analy * y**2 * (l - x)
        + (4 + 5 * nu_analy) * h**2 * x / 4
        + (3 * l - x) * x**2
    )
)

sigma_xx_analy = shear_y * (l - x) * y / Inertia
sigma_yy_analy = np.zeros(sigma_xx_analy.shape[0])
sigma_xy_analy = shear_y / (2 * Inertia) * (y - h / 2) * (y + h / 2)

combined_disp = tuple(
    np.vstack(
        (
            np.array(u_pred.tolist()),
            np.array(v_pred.tolist()),
            np.zeros(u_pred.shape[0]),
        )
    )
)
combined_stress = tuple(
    np.vstack(
        (
            np.array(sigma_xx.tolist()),
            np.array(sigma_yy.tolist()),
            np.array(sigma_xy.tolist()),
        )
    )
)
combined_disp_analy = tuple(
    np.vstack((u_x_analy.flatten(), u_y_analy.flatten(), np.zeros(u_x_analy.shape[0])))
)
combined_stress_analy = tuple(
    np.vstack(
        (sigma_xx_analy.flatten(), sigma_yy_analy.flatten(), sigma_xy_analy.flatten())
    )
)

error_x = abs(np.array(u_pred.tolist()) - u_x_analy.flatten())
error_y = abs(np.array(v_pred.tolist()) - u_y_analy.flatten())
combined_error = tuple(np.vstack((error_x, error_y, np.zeros(error_x.shape[0]))))


file_path = os.path.join(os.getcwd(), "Beam2D_nondim_newformulation")

x = X[:, 0].flatten()
y = X[:, 1].flatten()
z = np.zeros(y.shape)

unstructuredGridToVTK(
    file_path,
    x,
    y,
    z,
    dol_triangles.flatten(),
    offset,
    cell_types,
    pointData={
        "displacement": combined_disp,
        "disp_analy": combined_disp_analy,
        "stress": combined_stress,
        "stress_analy": combined_stress_analy,
        "disp_error": combined_error,
    },
)
