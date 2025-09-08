import numpy as np
import matplotlib.pyplot as plt
import os
# os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde

from deepxde import backend as bkd
import torch
from pathlib import Path

import pyvista as pv

dde.config.set_default_float("float64") # use double precision (needed for L-BFGS)
seed = 17
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.set_default_device("cpu")
'''
@author: svoelkl

Torsion test for a 3D block, done with an incremental approach.
'''

from utils.geometry.custom_geometry import GmshGeometry3D
from utils.geometry.gmsh_models import Block_3D_hex
from utils.elasticity import elasticity_utils
from utils.elasticity.elasticity_utils import get_stress_tensor, get_elastic_strain_3d, problem_parameters
from utils.postprocess.elasticity_postprocessing import solutionFieldOnMeshToVtk3D

from utils.deep_energy.deep_pde import DeepEnergyPDE
from utils.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from utils.vpinns.quad_rule import GaussQuadratureRule

from utils.hyperelasticity import hyperelasticity_utils
from utils.hyperelasticity.hyperelasticity_utils import strain_energy_neo_hookean_3d, compute_elastic_properties, first_piola_stress_tensor_3D, cauchy_stress_3D

from utils.postprocess.save_normals_tangentials_to_vtk import export_normals_tangentials_to_vtk
from deepxde.optimizers.config import LBFGS_options


length = 4
height = 1
width = 1
seed_l = 40
seed_h = 10
seed_w = 10
origin = [0, -0.5, -0.5]

# The applied pressure 
pressure = -0.1

Block_3D_obj = Block_3D_hex(origin=origin, 
                            length=length,
                            height=height,
                            width=width,
                            divisions=[seed_l, seed_h, seed_w])

gmsh_model = Block_3D_obj.generateGmshModel(visualize_mesh=False)

domain_dimension = 3
quad_rule = GaussQuadratureRule(rule_name="gauss_legendre", dimension=domain_dimension, ngp=2) # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()

boundary_dimension = 2
quad_rule_boundary_integral = GaussQuadratureRule(rule_name="gauss_legendre", dimension=boundary_dimension, ngp=2) # gauss_legendre gauss_labotto
coord_quadrature_boundary, weight_quadrature_boundary = quad_rule_boundary_integral.generate()

def on_back(x):
    return np.isclose(x[0], length)

boundary_selection_map = [{"boundary_function" : on_back, "tag" : "on_back"}]

geom = GmshGeometryElementDeepEnergy(
                           gmsh_model,
                           dimension=domain_dimension, 
                           coord_quadrature=coord_quadrature, 
                           weight_quadrature= weight_quadrature, 
                           coord_quadrature_boundary=coord_quadrature_boundary,
                           boundary_dim=boundary_dimension,
                           weight_quadrature_boundary=weight_quadrature_boundary,
                           boundary_selection_map=boundary_selection_map)

# export_normals_tangentials_to_vtk(geom, save_folder_path=str(Path(__file__).parent.parent.parent.parent), file_name="block_boundary_normals")# # change global variables in elasticity_utils
# hyperelasticity_utils.e_modul = 1.33
# hyperelasticity_utils.nu = 0.3
# nu,lame,shear,e_modul = compute_elastic_properties()

# # change global variables in elasticity_utils
# elasticity_utils.lame = lame
# elasticity_utils.shear = shear

# The applied pressure

pressure = 1
# hyperelasticity_utils.lame = 115.38461538461539
# hyperelasticity_utils.shear = 76.92307692307692
hyperelasticity_utils.e_modul = 1.33
hyperelasticity_utils.nu = 0.33

nu,lame,shear,e_modul = compute_elastic_properties()
applied_disp_y = -pressure/e_modul*(1-nu**2)*1

def potential_energy(X, 
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
                     boundary_selection_tag):
    
    internal_energy_density = strain_energy_neo_hookean_3d(inputs, outputs)[beg_pde:beg_boundary]
    
    internal_energy = global_element_weights_t[:,0:1]*global_element_weights_t[:,1:2]*global_element_weights_t[:,2:3]*(internal_energy_density)*jacobian_t
    
    return [internal_energy]

def points_at_back(x, on_boundary):
    points_bottom = np.isclose(x[0],0)
    
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
    train_distribution = "Sobol"
)

def output_transform(x, y):
    # displacement field (u, v, w)
    u = y[:, 0:1]
    v = y[:, 1:2]
    w = y[:, 2:3]

    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    z_loc = x[:, 2:3]

    y0, z0 = 0.0, 0.0
    # theta = 2*np.pi / 3
    # theta_deg = 150
    theta = np.radians(theta_deg)
    s = x_loc / length
    # print(theta_deg)

    # rotation displacement at x = L
    v_l = (y0 + (y_loc - y0) * np.cos(theta) - (z_loc - z0) * np.sin(theta) - y_loc)
    w_l = (z0 + (y_loc - y0) * np.sin(theta) + (z_loc - z0) * np.cos(theta) - z_loc)
    
    # Simplified version for theta_deg = 180, and the center is y0, z0 = 0.0, 0.0
    # v_l = -2*y_loc
    # w_l = -2*z_loc 

    u_out = s * (1-s) * u  # no u_x prescribed, just fix at x=0
    v_out = s * v_l + s * (1 - s) * v  # smooth blend
    w_out = s * w_l + s * (1 - s) * w

    return bkd.concat([u_out, v_out, w_out], axis=1)



# 3 inputs, 9 outputs for 3D 
layer_size = [3] + [50] * 5 + [3]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)
loss_weights=None

model = dde.Model(data, net)   

# Model parameters 
steps = 10
torsion_angle = 150
model_path = str(Path(__file__).parent)
simulation_case = f"3d_block_torsion_nonlinear_displacement_incremental_compare"
adam_iterations = 2000
lbfgs_iterations = 3000
for i in range(steps):
    theta_deg = torsion_angle/steps*(i+1)
    model.compile("adam", lr=0.001)
    losshistory, train_state = model.train(iterations=adam_iterations, display_every=100)

    dde.optimizers.config.set_LBFGS_options(maxiter=lbfgs_iterations)
    model.compile("L-BFGS")
    losshistory, train_state = model.train(display_every=1000)

    # Compare with FEM reference
    fem_path = str(Path(__file__).parent.parent+f"/fem_reference/fem_reference_3d_block_torsion_angle_{theta_deg:03}.vtu")
    fem_reference = pv.read(fem_path)
    displacement_fem = fem_reference.point_data["displacement"]
    cauchy_stress_fem = fem_reference.point_data["nodal_cauchy_stresses_xyz"]
    
    # Save results
    points, _, cell_types, elements = geom.get_mesh()
    n_nodes_per_cell = elements.shape[1]
    n_cells = elements.shape[0]
    cells = np.hstack([np.insert(elem, 0, n_nodes_per_cell) for elem in elements])
    cells = np.array(cells, dtype=np.int64)
    cell_types = np.array(cell_types, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    output = model.predict(points)
    sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yx, sigma_xz, sigma_zx, sigma_yz, sigma_zy = model.predict(points, operator=cauchy_stress_3D)
    cauchy_stress = np.column_stack((sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz))
    displacement = np.column_stack((output[:,0:1], output[:,1:2], output[:,2:3]))
    grid.point_data['pred_displacement'] = displacement
    grid.point_data['pred_cauchy_stress'] = cauchy_stress


    file_path = os.path.join(model_path, f"{simulation_case}_{i}")
    grid.save(file_path)
    
    model.net.built = False
model.save(f"{model_path}/{simulation_case}")
dde.saveplot(
    losshistory, train_state, issave=True, isplot=False, output_dir=model_path, 
    loss_fname=f"{simulation_case}-{(steps)*(adam_iterations+lbfgs_iterations)}_loss.dat", 
    train_fname=f"{simulation_case}-{steps*(adam_iterations+lbfgs_iterations)}_train.dat", 
    test_fname=f"{simulation_case}-{steps*(adam_iterations+lbfgs_iterations)}_test.dat"
)


fig1, ax1 = plt.subplots(figsize=(10,8))
ax1.plot(losshistory.steps, [sum(l) for l in losshistory.loss_train], color="b", lw=2, label="Energy", marker="x")
ax1.set_xlabel("Iterations", size=17)
ax1.set_ylabel("MSE", size=17)
ax1.set_yscale("log")
ax1.tick_params(axis="both", labelsize=15)
ax1.legend(fontsize=17)
ax1.grid()
plt.tight_layout()
fig1.savefig(f"{model_path}/{simulation_case}-{steps*(adam_iterations+lbfgs_iterations)}_loss_plot.png", dpi=300)