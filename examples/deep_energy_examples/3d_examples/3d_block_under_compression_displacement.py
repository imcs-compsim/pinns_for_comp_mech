"""
3D Block Under Compression with Displacement Boundary Conditions using Deep Energy Method.

This module implements a Physics-Informed Neural Network (PINN) example that solves
a 3D elastic block compression problem using the Deep Energy method. The example
demonstrates solving an elasticity problem by minimizing the potential energy functional
(internal energy minus external work) with displacement boundary conditions applied at
the top surface.

The problem setup includes:
- A 3D hexahedral block with configurable dimensions and mesh resolution
- Dirichlet boundary condition enforcing displacement at the top surface
- Essential boundary conditions (zero displacement) at the origin surfaces
- Stress and displacement predictions on the full domain mesh
- VTK output for visualization of displacement and stress fields

@author: tsahin
"""

import os

import deepxde as dde
import numpy as np
from deepxde import backend as bkd
from pyevtk.hl import unstructuredGridToVTK

from compsim_pinns.deep_energy.deep_pde import DeepEnergyPDE
from compsim_pinns.elasticity.elasticity_utils import (
    get_elastic_strain_3d,
    get_stress_tensor,
    problem_parameters,
)
from compsim_pinns.geometry.custom_geometry import (
    GmshGeometry3D,
    GmshGeometryElementDeepEnergy,
)
from compsim_pinns.geometry.gmsh_models import Block_3D_hex
from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule

length = 1
height = 1
width = 1
seed_l = 15
seed_h = 15
seed_w = 15
origin = [0, 0, 0]

# The applied pressure
pressure = -0.1

Block_3D_obj = Block_3D_hex(
    origin=origin,
    length=length,
    height=height,
    width=width,
    divisions=[seed_l, seed_h, seed_w],
)

gmsh_model = Block_3D_obj.generateGmshModel(visualize_mesh=False)

geom = GmshGeometry3D(gmsh_model)

quad_rule = GaussQuadratureRule(
    rule_name="gauss_legendre", dimension=3, ngp=1
)  # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()

quad_rule_boundary_integral = GaussQuadratureRule(
    rule_name="gauss_legendre", dimension=2, ngp=3
)  # gauss_legendre gauss_labotto
coord_quadrature_boundary, weight_quadrature_boundary = (
    quad_rule_boundary_integral.generate()
)


def boundary_inner(x):
    """
    Determine if points lie on the inner boundary at the top surface.

    Args:
        x (array-like): Coordinates of points, where x[1] represents the y-coordinate (height).

    Returns:
        array-like: True for points where the y-coordinate equals the height value,
            False otherwise. Uses np.isclose for floating-point comparison tolerance.
    """
    return np.isclose(x[1], height)


boundary_selection_map = [
    {"boundary_function": boundary_inner, "tag": "boundary_inner"}
]

geom = GmshGeometryElementDeepEnergy(
    gmsh_model,
    dimension=3,
    coord_quadrature=coord_quadrature,
    weight_quadrature=weight_quadrature,
    coord_quadrature_boundary=None,
    weight_quadrature_boundary=None,
    boundary_selection_map=None,
)

# # change global variables in elasticity_utils
# hyperelasticity_utils.youngs_modulus = 1.33
# hyperelasticity_utils.nu = 0.3
# nu,lame,shear,youngs_modulus = compute_elastic_properties()

# # change global variables in elasticity_utils
# elasticity_utils.lame = lame
# elasticity_utils.shear = shear

# The applied pressure
pressure = 0.1
nu, lame, shear, youngs_modulus = problem_parameters()
applied_disp_y = -pressure / youngs_modulus * (1 - nu**2) * 1


def potential_energy(
    X,
    inputs,
    outputs,
    beg_pde,
    beg_boundary,
    n_e,
    n_gp,
    n_e_boundary,
    n_gp_boundary,
    jacobian_t,
    global_element_weights_t,
    mapped_normal_boundary_t,
    jacobian_boundary_t,
    global_weights_boundary_t,
    boundary_selection_tag,
):
    """Calculate the potential energy of the system based on strain energy density.

    This function computes the total internal energy by integrating the Neo-Hookean
    strain energy density over the domain using numerical quadrature.

    Args:
        X (Tensor): The coordinate points in the domain.
        inputs (Tensor): Input features for the neural network (e.g., coordinates).
        outputs (Tensor): Output predictions from the neural network (e.g., displacements).
        beg_pde (int): Beginning index for PDE domain elements.
        beg_boundary (int): Beginning index for boundary elements.
        n_e (int): Number of elements in the domain.
        n_gp (int): Number of Gauss points per element.
        n_e_boundary (int): Number of boundary elements.
        n_gp_boundary (int): Number of Gauss points per boundary element.
        jacobian_t (Tensor): Jacobian determinants at domain integration points.
        global_element_weights_t (Tensor): Gaussian quadrature weights for domain integration.
        mapped_normal_boundary_t (Tensor): Mapped normal vectors at boundary Gauss points.
        jacobian_boundary_t (Tensor): Jacobian determinants at boundary integration points.
        global_weights_boundary_t (Tensor): Gaussian quadrature weights for boundary integration.
        boundary_selection_tag (Tensor): Tags identifying boundary regions.

    Returns:
        list: A list containing a single Tensor with the integrated internal energy
            over all domain elements.
    """

    eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_xz = get_elastic_strain_3d(
        inputs, outputs
    )
    sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz = get_stress_tensor(
        inputs, outputs
    )

    # get the internal energy
    internal_energy_density = 0.5 * (
        sigma_xx[beg_pde:beg_boundary] * eps_xx[beg_pde:beg_boundary]
        + sigma_yy[beg_pde:beg_boundary] * eps_yy[beg_pde:beg_boundary]
        + sigma_zz[beg_pde:beg_boundary] * eps_zz[beg_pde:beg_boundary]
        + 2 * sigma_xy[beg_pde:beg_boundary] * eps_xy[beg_pde:beg_boundary]
        + 2 * sigma_yz[beg_pde:beg_boundary] * eps_yz[beg_pde:beg_boundary]
        + 2 * sigma_xz[beg_pde:beg_boundary] * eps_xz[beg_pde:beg_boundary]
    )

    internal_energy = (
        global_element_weights_t[:, 0:1]
        * global_element_weights_t[:, 1:2]
        * global_element_weights_t[:, 1:2]
        * (internal_energy_density)
        * jacobian_t
    )

    # get the external energy
    # select the points where external force is applied
    # cond = boundary_selection_tag["on_top"]
    # n_e_boundary = int(cond.sum()/n_gp_boundary)
    # nx = mapped_normal_boundary_t[:,0:1][cond]
    # ny = mapped_normal_boundary_t[:,1:2][cond]

    # #sigma_xx_n_x = sigma_xx[beg_boundary:][cond]*nx
    # #sigma_xy_n_y = sigma_xy[beg_boundary:][cond]*ny

    # sigma_yx_n_x = sigma_xy[beg_boundary:][cond]*nx
    # sigma_yy_n_y = sigma_yy[beg_boundary:][cond]*ny

    # #t_x = sigma_xx_n_x + sigma_xy_n_y
    # t_y = sigma_yx_n_x + sigma_yy_n_y

    # u_x = outputs[:,0:1][beg_boundary:][cond]
    # u_y = outputs[:,1:2][beg_boundary:][cond]

    # external_force_density = -pressure*u_y
    # external_work = global_weights_boundary_t[cond]*(external_force_density)*jacobian_boundary_t[cond]

    # internal_energy_reshaped = bkd.reshape(internal_energy, (n_e, n_gp))
    # external_work_reshaped = bkd.reshape(external_work, (n_e_boundary, n_gp_boundary))

    # total_energy = bkd.reduce_sum(bkd.sum(internal_energy_reshaped, dim=1)) - bkd.reduce_sum(bkd.sum(external_work_reshaped, dim=1)) #+ bkd.reduce_sum(bkd.sum(internal_energy_reshaped, dim=1))

    return [internal_energy]


def points_at_top(x, on_boundary):
    """Check whether a point satisfies the `points_at_top` boundary condition.

    Args:
        x: Input coordinates used to evaluate the function.
        on_boundary: Boundary indicator provided by the geometry callback.

    Returns:
        bool: Result of the `points_at_top` evaluation.
    """
    points_top = np.isclose(x[1], height)

    return on_boundary and points_top


def points_at_bottom(x, on_boundary):
    """Check whether a point satisfies the `points_at_bottom` boundary condition.

    Args:
        x: Input coordinates used to evaluate the function.
        on_boundary: Boundary indicator provided by the geometry callback.

    Returns:
        bool: Result of the `points_at_bottom` evaluation.
    """
    points_bottom = np.isclose(x[1], 0)

    return on_boundary and points_bottom


bc_u_y = dde.DirichletBC(geom, lambda _: applied_disp_y, points_at_top, component=1)

n_dummy = 1
data = DeepEnergyPDE(
    geom,
    potential_energy,
    [bc_u_y],
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=None,
    train_distribution="Sobol",
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
    w = y[:, 2:3]

    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    z_loc = x[:, 2:3]

    # define surfaces
    bottom_surface = y_loc
    left_surface = x_loc
    front_surface = z_loc

    return bkd.concat(
        [
            u * (left_surface),  # displacement in x direction is 0 at x=0
            v * (bottom_surface),  # displacement in y direction is 0 at y=0
            w * (front_surface),  # displacement in z direction is 0 at z=0
        ],
        axis=1,
    )


# 3 inputs, 9 outputs for 3D
layer_size = [3] + [50] * 5 + [3]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)
loss_weights = [1, 1e2]

model = dde.Model(data, net)
model.compile("adam", lr=0.001, loss_weights=loss_weights)
losshistory, train_state = model.train(epochs=8000, display_every=200)

model.compile("L-BFGS", loss_weights=loss_weights)
losshistory, train_state = model.train(display_every=200)

X, offset, cell_types, elements = geom.get_mesh()

output = model.predict(X)
sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz = model.predict(
    X, operator=get_stress_tensor
)


# .tolist() is applied to remove datatype
u_pred, v_pred, w_pred = (
    output[:, 0].tolist(),
    output[:, 1].tolist(),
    output[:, 2].tolist(),
)  # displacements
sigma_xx_pred, sigma_yy_pred, sigma_zz_pred = (
    sigma_xx.flatten().tolist(),
    sigma_yy.flatten().tolist(),
    sigma_zz.flatten().tolist(),
)  # normal stresses
sigma_xy_pred, sigma_yz_pred, sigma_xz_pred = (
    sigma_xy.flatten().tolist(),
    sigma_yz.flatten().tolist(),
    sigma_xz.flatten().tolist(),
)  # shear stresses

combined_disp_pred = tuple(np.vstack((u_pred, v_pred, w_pred)))
combined_normal_stress_pred = tuple(
    np.vstack((sigma_xx_pred, sigma_yy_pred, sigma_zz_pred))
)
combined_shear_stress_pred = np.vstack((sigma_xy_pred, sigma_yz_pred, sigma_xz_pred))

x = X[:, 0].flatten()
y = X[:, 1].flatten()
z = X[:, 2].flatten()

file_path = os.path.join(os.getcwd(), "deep_energy_single_block_compression_3d")

unstructuredGridToVTK(
    file_path,
    x,
    y,
    z,
    elements.flatten(),
    offset,
    cell_types,
    pointData={
        "pred_displacement": combined_disp_pred,
        "pred_normal_stress": combined_normal_stress_pred,
        "pred_stress_xy": combined_shear_stress_pred[0],
        "pred_stress_yz": combined_shear_stress_pred[1],
        "pred_stress_xz": combined_shear_stress_pred[2],
    },
)
