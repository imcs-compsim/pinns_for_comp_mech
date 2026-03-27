"""
3D block torsion nonlinear displacement example using Deep Energy method.

This module demonstrates the application of Physics-Informed Neural Networks (PINNs)
with the Deep Energy approach to solve a 3D nonlinear torsion problem on a hexahedral block.
The model learns to predict displacement fields, Cauchy stresses, and Green-Lagrange strains
under applied torsional loading, using neo-Hookean hyperelasticity constitutive behavior.

The results are compared against reference FEM solutions and saved to VTK format for visualization.

@author: tsahin
"""

import os

import deepxde as dde
import numpy as np
import pyvista as pv
from deepxde import backend as bkd
from deepxde.optimizers.config import LBFGS_options

from compsim_pinns.deep_energy.deep_pde import DeepEnergyPDE
from compsim_pinns.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from compsim_pinns.geometry.gmsh_models import Block_3D_hex
from compsim_pinns.hyperelasticity import hyperelasticity_utils
from compsim_pinns.hyperelasticity.hyperelasticity_utils import (
    cauchy_stress_3D,
    compute_elastic_properties,
    green_lagrange_strain_3D,
    strain_energy_neo_hookean_3d,
)
from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule

length = 4
height = 1
width = 1
seed_l = 40
seed_h = 10
seed_w = 10
origin = [0, -0.5, -0.5]

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

domain_dimension = 3
quad_rule = GaussQuadratureRule(
    rule_name="gauss_legendre", dimension=domain_dimension, ngp=2
)  # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()

boundary_dimension = 2
quad_rule_boundary_integral = GaussQuadratureRule(
    rule_name="gauss_legendre", dimension=boundary_dimension, ngp=2
)  # gauss_legendre gauss_labotto
coord_quadrature_boundary, weight_quadrature_boundary = (
    quad_rule_boundary_integral.generate()
)


def on_back(x):
    """Determine if a point is located on the back face of the domain.

    This function checks if the x-coordinate of a given point is close to
    the specified length, which corresponds to the back boundary in 3D space.

    Args:
        x (array-like): Coordinates of a point, where x[0] is the x-coordinate.

    Returns:
        bool: True if the point is on the back face (x-coordinate ≈ length), False otherwise.
    """
    return np.isclose(x[0], length)


boundary_selection_map = [{"boundary_function": on_back, "tag": "on_back"}]

geom = GmshGeometryElementDeepEnergy(
    gmsh_model,
    dimension=domain_dimension,
    coord_quadrature=coord_quadrature,
    weight_quadrature=weight_quadrature,
    coord_quadrature_boundary=coord_quadrature_boundary,
    boundary_dim=boundary_dimension,
    weight_quadrature_boundary=weight_quadrature_boundary,
    boundary_selection_map=boundary_selection_map,
)

# export_normals_tangentials_to_vtk(geom, save_folder_path=str(Path(__file__).parent.parent.parent.parent), file_name="block_boundary_normals")# # change global variables in elasticity_utils
# hyperelasticity_utils.youngs_modulus = 1.33
# hyperelasticity_utils.nu = 0.3
# nu,lame,shear,youngs_modulus = compute_elastic_properties()

# # change global variables in elasticity_utils
# elasticity_utils.lame = lame
# elasticity_utils.shear = shear

# The applied pressure

pressure = 1
# hyperelasticity_utils.lame = 115.38461538461539
# hyperelasticity_utils.shear = 76.92307692307692
hyperelasticity_utils.youngs_modulus = 1.33
hyperelasticity_utils.nu = 0.33

nu, lame, shear, youngs_modulus = compute_elastic_properties()
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

    internal_energy_density = strain_energy_neo_hookean_3d(inputs, outputs)[
        beg_pde:beg_boundary
    ]

    internal_energy = (
        global_element_weights_t[:, 0:1]
        * global_element_weights_t[:, 1:2]
        * global_element_weights_t[:, 2:3]
        * (internal_energy_density)
        * jacobian_t
    )

    return [internal_energy]


def points_at_back(x, on_boundary):
    """Check whether a point satisfies the `points_at_back` boundary condition.

    Args:
        x: Input coordinates used to evaluate the function.
        on_boundary: Boundary indicator provided by the geometry callback.

    Returns:
        bool: Result of the `points_at_back` evaluation.
    """
    points_bottom = np.isclose(x[0], 0)

    return on_boundary and points_bottom


bc_u_y = dde.DirichletBC(geom, lambda _: 0, points_at_back, component=1)
bc_u_z = dde.DirichletBC(geom, lambda _: 0, points_at_back, component=2)

n_dummy = 1
data = DeepEnergyPDE(
    geom,
    potential_energy,
    [],
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=None,
    train_distribution="Sobol",
)


def output_transform(x, y):
    # displacement field (u, v, w)
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

    y0, z0 = 0.0, 0.0
    # theta = 2*np.pi / 3
    theta_deg = 150
    theta = np.radians(theta_deg)
    s = x_loc / length

    # rotation displacement at x = L
    v_l = y0 + (y_loc - y0) * np.cos(theta) - (z_loc - z0) * np.sin(theta) - y_loc
    w_l = z0 + (y_loc - y0) * np.sin(theta) + (z_loc - z0) * np.cos(theta) - z_loc

    # Simplified version for theta_deg = 180, and the center is y0, z0 = 0.0, 0.0
    # v_l = -2*y_loc
    # w_l = -2*z_loc

    u_out = s * (1 - s) * u  # no u_x prescribed, just fix at x=0
    v_out = s * v_l + s * (1 - s) * v  # smooth blend
    w_out = s * w_l + s * (1 - s) * w

    return bkd.concat([u_out, v_out, w_out], axis=1)


# 3 inputs, 9 outputs for 3D
layer_size = [3] + [50] * 5 + [3]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)
loss_weights = None

model = dde.Model(data, net)

# model.compile("adam", lr=0.001)
# losshistory, train_state = model.train(epochs=stabilization_model_epoch, display_every=100)

apply_load = True

model.compile("adam", lr=0.001)
# losshistory, train_state = model.train(epochs=3000, display_every=100)
# if you want to save the model, run the following
losshistory, train_state = model.train(epochs=3000, display_every=100)

# For pytorch
# LBFGS_options["iter_per_step"] = 1
# LBFGS_options["maxiter"] = 500

LBFGS_options["maxiter"] = 1000
model.compile("L-BFGS")
losshistory, train_state = model.train(display_every=100)
# losshistory, train_state = model.train(display_every=100, model_save_path=model_path)

# 'train' took 110.185835 s adam for 5000 iter, and I tried with 3000 so 'train' took 68.116314 s
# 'train' took 52.052350 s lbfgs, second is 'train' took 41.838312 s for 1018

# dde.optimizers.set_LBFGS_options(
#                                 maxiter=1000
#                                 )
# model.compile("L-BFGS")
# losshistory, train_state = model.train(display_every=200)

file_path = "/home/a11btasa/git_repos/nonlinear-reference-results/block_torsion/output-structure.pvd"
save_file_path = os.path.join(os.getcwd(), "deep_energy_3d_block_torsion_nonlinear")

# Convert the Path object to a string
reader = pv.get_reader(file_path)

reader.set_active_time_point(-1)
data = reader.read()[0]

X = data.points

output = model.predict(X)
(
    sigma_xx,
    sigma_yy,
    sigma_zz,
    sigma_xy,
    sigma_yx,
    sigma_xz,
    sigma_zx,
    sigma_yz,
    sigma_zy,
) = model.predict(X, operator=cauchy_stress_3D)
eps_xx, eps_yy, eps_zz, eps_xy, eps_xz, eps_yz = model.predict(
    X, operator=green_lagrange_strain_3D
)

cauchy_stress = np.column_stack(
    (sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz)
)
strain = np.column_stack((eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_xz))
displacement = np.column_stack((output[:, 0:1], output[:, 1:2], output[:, 2:3]))

data.point_data["pred_displacement"] = displacement
data.point_data["pred_cauchy_stress"] = cauchy_stress
data.point_data["pred_strain"] = strain

disp_fem = data.point_data["displacement"]
stress_fem = data.point_data["nodal_cauchy_stresses_xyz"]

error_disp = abs((disp_fem - displacement))
data.point_data["pointwise_displacement_error"] = error_disp
error_stress = abs((stress_fem - cauchy_stress))
data.point_data["pointwise_cauchystress_error"] = error_stress

data.save(f"{save_file_path}.vtu")

# X, offset, cell_types, elements = geom.get_mesh()

# output = model.predict(X)
# sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yx, sigma_xz, sigma_zx, sigma_yz, sigma_zy = model.predict(X, operator=cauchy_stress_3D)
# p_xx, p_yy, p_zz, p_xy, p_yx, p_xz, p_zx, p_yz, p_zy = model.predict(X, operator=first_piola_stress_tensor_3D)

# # .tolist() is applied to remove datatype
# u_pred, v_pred, w_pred = output[:,0].tolist(), output[:,1].tolist(), output[:,2].tolist() # displacements
# sigma_xx_pred, sigma_yy_pred, sigma_zz_pred = sigma_xx.flatten().tolist(), sigma_yy.flatten().tolist(), sigma_zz.flatten().tolist() # normal stresses
# sigma_xy_pred, sigma_yz_pred, sigma_xz_pred = sigma_xy.flatten().tolist(), sigma_yz.flatten().tolist(), sigma_xz.flatten().tolist() # shear stresses
# p_xx_pred, p_yy_pred, p_zz_pred = p_xx.flatten().tolist(), p_yy.flatten().tolist(), p_zz.flatten().tolist() # normal stresses
# p_xy_pred, p_yz_pred, p_xz_pred = p_xy.flatten().tolist(), p_yz.flatten().tolist(), p_xz.flatten().tolist() # shear stresses

# combined_disp_pred = tuple(np.vstack((u_pred, v_pred, w_pred)))
# combined_normal_stress_pred = tuple(np.vstack((sigma_xx_pred, sigma_yy_pred, sigma_zz_pred)))
# combined_shear_stress_pred = np.vstack((sigma_xy_pred, sigma_yz_pred, sigma_xz_pred))
# combined_normal_p_pred = tuple(np.vstack((p_xx_pred, p_yy_pred, p_zz_pred)))
# combined_shear_p_pred = np.vstack((p_xy_pred, p_yz_pred, p_xz_pred))

# x = X[:,0].flatten()
# y = X[:,1].flatten()
# z = X[:,2].flatten()

# file_path = os.path.join(os.getcwd(), "deep_energy_3d_block_torsion_nonlinear")

# unstructuredGridToVTK(file_path, x, y, z, elements.flatten(), offset,
#                         cell_types, pointData = { "pred_displacement" : combined_disp_pred,
#                                                 "pred_normal_stress" : combined_normal_stress_pred,
#                                                 "pred_normal_stress_p" : combined_normal_p_pred,
#                                                 "pred_stress_xy": combined_shear_stress_pred[0],
#                                                 "pred_stress_yz": combined_shear_stress_pred[1],
#                                                 "pred_stress_xz": combined_shear_stress_pred[2]})

# # von_mises
# # sqrt(0.5 * (
# #     (pred_normal_stress_X - pred_normal_stress_Y)^2 +
# #     (pred_normal_stress_Y - pred_normal_stress_Z)^2 +
# #     (pred_normal_stress_Z - pred_normal_stress_X)^2 +
# #     6 * (pred_stress_xy^2 + pred_stress_xz^2 + pred_stress_yz^2)
# # ))
