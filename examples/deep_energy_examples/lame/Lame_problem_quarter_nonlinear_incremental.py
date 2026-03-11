"""
Physics-informed neural network (PINN) solution for nonlinear incremental Lamé problem on quarter circle geometry.

This module solves a hyperelastic problem using the Deep Energy method with a quarter circle domain
containing a hole. It implements incremental loading to compute stress and displacement fields,
comparing predictions against analytical solutions for plane strain conditions.

Key features:
- Quadrilateral mesh generation using Gmsh
- Neo-Hookean hyperelastic material model
- Gaussian quadrature integration for strain energy
- Incremental pressure loading on inner boundary
- Comparison with analytical Lamé solution
- VTK export for visualization

@author: tsahin
"""

import os

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from deepxde import backend as bkd
from pyevtk.hl import unstructuredGridToVTK

from compsim_pinns.deep_energy.deep_pde import DeepEnergyPDE
from compsim_pinns.elasticity.elasticity_utils import (
    stress_plane_strain,
    stress_plane_stress,
)
from compsim_pinns.geometry.custom_geometry import (
    GmshGeometryElementDeepEnergy,
)
from compsim_pinns.geometry.geometry_utils import (
    polar_transformation_2d,
)
from compsim_pinns.geometry.gmsh_models import QuarterCirclewithHole
from compsim_pinns.hyperelasticity import hyperelasticity_utils
from compsim_pinns.hyperelasticity.hyperelasticity_utils import (
    cauchy_stress_2D,
    compute_elastic_properties,
    strain_energy_neo_hookean_2d,
)
from compsim_pinns.postprocess.custom_callbacks import EpochTracker
from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule

gmsh_options = {"General.Terminal": 1, "Mesh.Algorithm": 11}
quarter_circle_with_hole = QuarterCirclewithHole(
    center=[0, 0, 0],
    inner_radius=1,
    outer_radius=2,
    mesh_size=0.2,
    gmsh_options=gmsh_options,
)

gmsh_model = quarter_circle_with_hole.generateGmshModel()

quad_rule = GaussQuadratureRule(
    rule_name="gauss_legendre", dimension=2, ngp=2
)  # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()

quad_rule_boundary_integral = GaussQuadratureRule(
    rule_name="gauss_legendre", dimension=1, ngp=4
)  # gauss_legendre gauss_labotto
coord_quadrature_boundary, weight_quadrature_boundary = (
    quad_rule_boundary_integral.generate()
)

gmsh_model = quarter_circle_with_hole.generateGmshModel(visualize_mesh=False)

radius_inner = quarter_circle_with_hole.inner_radius
center_inner = [quarter_circle_with_hole.center[0], quarter_circle_with_hole.center[1]]
radius_outer = quarter_circle_with_hole.outer_radius
center_outer = [quarter_circle_with_hole.center[0], quarter_circle_with_hole.center[1]]


def boundary_inner(x):
    """Check whether a point satisfies the `boundary_inner` boundary condition.

    Args:
        x: Input coordinates used to evaluate the function.

    Returns:
        bool: Result of the `boundary_inner` evaluation.
    """
    return np.isclose(np.linalg.norm(x - center_inner, axis=-1), radius_inner)


boundary_selection_map = [
    {"boundary_function": boundary_inner, "tag": "boundary_inner"}
]

revert_curve_list = ["curve_2"]
revert_normal_dir_list = [2, 2, 1, 2]

geom = GmshGeometryElementDeepEnergy(
    gmsh_model,
    dimension=2,
    coord_quadrature=coord_quadrature,
    weight_quadrature=weight_quadrature,
    revert_curve_list=revert_curve_list,
    revert_normal_dir_list=revert_normal_dir_list,
    coord_quadrature_boundary=coord_quadrature_boundary,
    weight_quadrature_boundary=weight_quadrature_boundary,
    boundary_selection_map=boundary_selection_map,
)

# change global variables in elasticity_utils
hyperelasticity_utils.youngs_modulus = 200
hyperelasticity_utils.nu = 0.3
hyperelasticity_utils.stress_state = "plane_strain"
nu, lame, shear, youngs_modulus = compute_elastic_properties()

# The applied pressure
pressure_inlet = 10
epochs = 4000
steps = 20


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
    """Compute potential energy for this example setup.

    Args:
        X: Input coordinates used by this callback.
        inputs: Value for inputs.
        outputs: Value for outputs.
        beg_pde: Value for beg pde.
        beg_boundary: Value for beg boundary.
        n_e: Value for n e.
        n_gp: Value for n gp.
        n_e_boundary: Value for n e boundary.
        n_gp_boundary: Value for n gp boundary.
        jacobian_t: Value for jacobian t.
        global_element_weights_t: Value for global element weights t.
        mapped_normal_boundary_t: Value for mapped normal boundary t.
        jacobian_boundary_t: Value for jacobian boundary t.
        global_weights_boundary_t: Value for global weights boundary t.
        boundary_selection_tag: Value for boundary selection tag.

    Returns:
        Any: Computed value returned by `potential_energy`.
    """
    internal_energy_density = strain_energy_neo_hookean_2d(inputs, outputs)

    internal_energy = (
        global_element_weights_t[:, 0:1]
        * global_element_weights_t[:, 1:2]
        * (internal_energy_density[beg_pde:beg_boundary])
        * jacobian_t
    )
    ####################################################################################################################
    # get the external work
    # select the points where external force is applied
    cond = boundary_selection_tag["boundary_inner"]
    nx = mapped_normal_boundary_t[:, 0:1][cond]
    ny = mapped_normal_boundary_t[:, 1:2][cond]

    x_coord = inputs[:, 0:1][beg_boundary:][cond]
    y_coord = inputs[:, 1:2][beg_boundary:][cond]

    u_x = outputs[:, 0:1][beg_boundary:][cond]
    u_y = outputs[:, 1:2][beg_boundary:][cond]

    phi_x = u_x  # + x_coord
    phi_y = u_y  # + y_coord

    # incremental loading
    current_epoch = model.data.current_epoch
    chunk = current_epoch // ((epochs + 1) / steps)
    pressure = (chunk + 1) * pressure_inlet / steps
    # if current_epoch < (epochs/2):
    #     pressure = 2/epochs*current_epoch*pressure_inlet
    # else:
    #     pressure = pressure_inlet

    external_force_density = -pressure * nx * phi_x + -pressure * ny * phi_y
    external_work = (
        global_weights_boundary_t[cond]
        * (external_force_density)
        * jacobian_boundary_t[cond]
    )

    ####################################################################################################################
    # Reshape energy-work terms and sum over the gauss points
    # internal_energy_reshaped = bkd.sum(bkd.reshape(internal_energy, (n_e, n_gp)), dim=1)
    # external_work_reshaped = bkd.sum(bkd.reshape(external_work, (n_e_boundary_external, n_gp_boundary)), dim=1)
    # sum over the elements and get the overall loss
    # total_energy = bkd.reduce_sum(internal_energy_reshaped) #- bkd.reduce_sum(external_work_reshaped)

    return [internal_energy, -external_work]


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
    """Compute output transform for this example setup.

    Args:
        x: Input coordinates used to evaluate the function.
        y: Field values or model outputs associated with `x`.

    Returns:
        Any: Computed value returned by `output_transform`.
    """
    u = y[:, 0:1]
    v = y[:, 1:2]

    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]

    return bkd.concat([u * x_loc / youngs_modulus, v * y_loc / youngs_modulus], axis=1)


# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

epoch_tracker = EpochTracker(period=10)

model = dde.Model(data, net)
# if we want to save the model, we use "model_save_path=model_path" during training, if we want to load trained model, we use "model_restore_path=return_restore_path(model_path, num_epochs)"
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(
    epochs=epochs, callbacks=[epoch_tracker], display_every=100
)

model.compile("L-BFGS")
# model.train_step.optimizer_kwargs["options"]['maxiter']=2000
model.train(callbacks=[epoch_tracker], display_every=100)

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################


def compareModelPredictionAndAnalyticalSolution(model):
    """
    This function plots analytical solutions and the predictions.
    """

    r = np.linspace(radius_inner, radius_outer, 100)
    y = np.zeros(r.shape[0])

    # Rotation angle in radians
    theta = np.pi / 4  # 45 degrees

    # Rotation
    x_rot = r * np.cos(theta) - y * np.sin(theta)
    y_rot = r * np.sin(theta) + y * np.cos(theta)

    dr2 = radius_outer**2 - radius_inner**2

    sigma_rr_analytical = (
        radius_inner**2 * pressure_inlet / dr2 * (r**2 - radius_outer**2) / r**2
    )
    sigma_theta_analytical = (
        radius_inner**2 * pressure_inlet / dr2 * (r**2 + radius_outer**2) / r**2
    )
    # u_rad = radius_inner**2*pressure_inlet*r/(youngs_modulus*(radius_outer**2-radius_inner**2))*(1-nu+(radius_outer/r)**2*(1+nu))

    inv_dr2 = 1 / radius_inner**2 - 1 / radius_outer**2
    a = -pressure_inlet / inv_dr2
    c = -a / (2 * radius_outer**2)

    if hyperelasticity_utils.stress_state == "plane_strain":
        u_rad = (1 + nu) / youngs_modulus * (-a / r + 2 * (1 - 2 * nu) * c * r)
    elif hyperelasticity_utils.stress_state == "plane_stress":
        u_rad = (
            radius_inner**2
            * pressure_inlet
            * r
            / (youngs_modulus * (radius_outer**2 - radius_inner**2))
            * (1 - nu + (radius_outer / r) ** 2 * (1 + nu))
        )

    r_x = np.hstack((x_rot.reshape(-1, 1), y_rot.reshape(-1, 1)))
    disps = model.predict(r_x)
    u_pred, v_pred = disps[:, 0:1], disps[:, 1:2]
    u_rad_pred = np.sqrt(u_pred**2 + v_pred**2)
    if hyperelasticity_utils.stress_state == "plane_strain":
        sigma_xx, sigma_yy, sigma_xy, sigma_yx = model.predict(
            r_x, operator=cauchy_stress_2D
        )
    elif hyperelasticity_utils.stress_state == "plane_stress":
        sigma_xx, sigma_yy, sigma_xy = model.predict(r_x, operator=stress_plane_stress)
    sigma_rr, sigma_theta, sigma_rtheta = polar_transformation_2d(
        sigma_xx, sigma_yy, sigma_xy, r_x
    )

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(
        r / radius_inner,
        sigma_rr_analytical / radius_inner,
        label=r"Analytical $\sigma_{r}$",
    )
    axs[0].plot(
        r / radius_inner, sigma_rr / radius_inner, label=r"Predicted $\sigma_{r}$"
    )
    axs[0].plot(
        r / radius_inner,
        sigma_theta_analytical / radius_inner,
        label=r"Analytical $\sigma_{\theta}$",
    )
    axs[0].plot(
        r / radius_inner,
        sigma_theta / radius_inner,
        label=r"Predicted $\sigma_{\theta}$",
    )
    axs[0].set(ylabel="Normalized stress", xlabel="r/a")
    axs[1].plot(r / radius_inner, u_rad / radius_inner, label=r"Analytical $u_r$")
    axs[1].plot(r / radius_inner, u_rad_pred / radius_inner, label=r"Predicted $u_r$")
    axs[1].set(ylabel="Normalized radial displacement", xlabel="r/a")
    axs[0].legend()
    axs[0].grid()
    axs[1].legend()
    axs[1].grid()
    fig.tight_layout()

    plt.savefig("Lame_quarter_gmsh")
    plt.show()


X, offset, cell_types, dol_triangles = geom.get_mesh()

displacement = model.predict(X)
sigma_xx, sigma_yy, sigma_xy = model.predict(X, operator=stress_plane_strain)
sigma_rr, sigma_theta, sigma_rtheta = polar_transformation_2d(
    sigma_xx, sigma_yy, sigma_xy, X
)

combined_disp = tuple(
    np.vstack(
        (
            np.array(displacement[:, 0].tolist()),
            np.array(displacement[:, 1].tolist()),
            np.zeros(displacement[:, 0].shape[0]),
        )
    )
)
combined_stress = tuple(
    np.vstack(
        (
            np.array(sigma_xx.flatten().tolist()),
            np.array(sigma_yy.flatten().tolist()),
            np.array(sigma_xy.flatten().tolist()),
        )
    )
)
combined_stress_polar = tuple(
    np.vstack(
        (
            np.array(sigma_rr.tolist()),
            np.array(sigma_theta.tolist()),
            np.array(sigma_rtheta.tolist()),
        )
    )
)

file_path = os.path.join(os.getcwd(), "Lame_quarter_gmsh_nonlinear")

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
        "stress": combined_stress,
        "stress_polar": combined_stress_polar,
    },
)

compareModelPredictionAndAnalyticalSolution(model)
