import numpy as np
import matplotlib.pyplot as plt
import os
import deepxde as dde
import numpy as np
from deepxde import backend as bkd
from pathlib import Path
import pyvista as pv

dde.config.set_default_float("float64") # use double precision (needed for L-BFGS)

import torch
seed = 17
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
'''
@author: svoelkl

Torsion test for a 3D block, done with an incremental approach.
'''

from utils.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from utils.geometry.gmsh_models import SphereEighthHertzian
from utils.geometry.geometry_utils import polar_transformation_3d_spherical

from utils.elasticity import elasticity_utils
from utils.elasticity.elasticity_utils import get_stress_tensor, get_elastic_strain_3d, problem_parameters
from utils.postprocess.elasticity_postprocessing import solutionFieldOnMeshToVtk3D

from utils.deep_energy.deep_pde import DeepEnergyPDE
from utils.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from utils.vpinns.quad_rule import GaussQuadratureRule

from utils.hyperelasticity import hyperelasticity_utils
from utils.hyperelasticity.hyperelasticity_utils import strain_energy_neo_hookean_3d, compute_elastic_properties, first_piola_stress_tensor_3D, cauchy_stress_3D, green_lagrange_strain_3D
from utils.contact_mech.contact_utils import calculate_gap_in_normal_direction_deep_energy
from utils.contact_mech import contact_utils

from deepxde.optimizers.config import LBFGS_options

radius = 1
center = [0,0,0]

Block_3D_obj = SphereEighthHertzian(radius=radius, center=center)

gmsh_model = Block_3D_obj.generateGmshModel(visualize_mesh=False)

def on_boundary_circle_contact(x):
    return np.isclose(np.linalg.norm(x - center, axis=-1), radius)

def on_top(x):
    return np.isclose(x[1],0)

boundary_selection_map = [{"boundary_function" : on_boundary_circle_contact, "tag" : "on_boundary_circle_contact"},
                          {"boundary_function" : on_top, "tag" : "on_top"},]

domain_dimension = 3
quad_rule = GaussQuadratureRule(rule_name="gauss_legendre", dimension=domain_dimension, ngp=4, element_type="simplex") # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()

boundary_dimension = 2
quad_rule_boundary_integral = GaussQuadratureRule(rule_name="gauss_legendre", dimension=boundary_dimension, ngp=4, element_type="simplex") # gauss_legendre gauss_labotto
coord_quadrature_boundary, weight_quadrature_boundary = quad_rule_boundary_integral.generate()

geom = GmshGeometryElementDeepEnergy(
                           gmsh_model,
                           dimension=domain_dimension, 
                           coord_quadrature=coord_quadrature, 
                           weight_quadrature= weight_quadrature, 
                           coord_quadrature_boundary=coord_quadrature_boundary,
                           boundary_dim=boundary_dimension,
                           weight_quadrature_boundary=weight_quadrature_boundary,
                           boundary_selection_map=boundary_selection_map)

# change global variables in elasticity_utils
hyperelasticity_utils.e_modul = 50
hyperelasticity_utils.nu = 0.3
nu,lame,shear,e_modul = compute_elastic_properties()

# zero neumann BC functions need the geom variable to be 
elasticity_utils.geom = geom

projection_plane = {"y" : -1} # projection plane formula
contact_utils.projection_plane = projection_plane

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
    
    internal_energy_density = strain_energy_neo_hookean_3d(inputs, outputs)
    
    internal_energy = global_element_weights_t*(internal_energy_density[beg_pde:beg_boundary])*jacobian_t
    ####################################################################################################################
    # get the external work
    # select the points where external force is applied
    cond = boundary_selection_tag["on_top"]
    u_y = outputs[:,1:2][beg_boundary:][cond]
    
    external_force_density = -ext_traction*u_y
    external_work = global_weights_boundary_t[cond]*(external_force_density)*jacobian_boundary_t[cond]
    ####################################################################################################################
    # contact work
    cond = boundary_selection_tag["on_boundary_circle_contact"]
    
    gap_n = calculate_gap_in_normal_direction_deep_energy(inputs[beg_boundary:], outputs[beg_boundary:], X, mapped_normal_boundary_t, cond)
    #gap_y = inputs[:,1:2][beg_boundary:][cond] + outputs[:,1:2][beg_boundary:][cond] + radius
    #gap_n = tf.math.divide_no_nan(gap_y, tf.math.abs(mapped_normal_boundary_t[:,1:2][cond]))
    eta=3e4
    contact_force_density = 1/2*eta*bkd.relu(-gap_n)*bkd.relu(-gap_n)
    contact_work = global_weights_boundary_t[cond]*(contact_force_density)*jacobian_boundary_t[cond]
    
    return [internal_energy, -external_work, contact_work]

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
    u = y[:, 0:1]
    v = y[:, 1:2]
    w = y[:, 2:3]
    
    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    z_loc = x[:, 2:3]
    
    # define surfaces
    # top_surface = -y_loc
    x_0_surface = x_loc
    z_0_surface = z_loc
    
    return bkd.concat([u*(x_0_surface)/e_modul, #displacement in x direction is 0 at x=0
                      v/e_modul,
                      w*(z_0_surface)/e_modul, #displacement in z direction is 0 at z=0
                      ], axis=1)

# 3 inputs, 3 outputs for 3D 
layer_size = [3] + [50] * 5 + [3]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)
loss_weights=None

model = dde.Model(data, net)

# Model parameters 
steps = 1
max_ext_traction = 5
model_path = str(Path(__file__).parent)
simulation_case = f"3d_hertzian_spherical_contact_incremental_exponential_decay"
learning_rate_adam = 1E-3
learning_rate_total_decay = 1E-3
adam_iterations = 5000
exponential_decay = learning_rate_total_decay ** (1 / adam_iterations)
lbfgs_iterations = 0
relaxation = False

if relaxation:
    relaxation_epsilon = 1e0
    relaxation_adam_iterations = adam_iterations
    print(f"\nRelaxation step using a factor of {relaxation_epsilon} of the step width with {relaxation_adam_iterations} iterations.\n")
    ext_traction = relaxation_epsilon * max_ext_traction / steps
    model.compile("adam", lr=learning_rate_adam)
    losshistory, train_state = model.train(iterations=relaxation_adam_iterations, display_every=100)

# Incremental loop
for i in range(steps):
    ext_traction = max_ext_traction/steps*(i+1)
    print(f"\nTraining for a traction of {ext_traction}.\n")
       
    model.compile("adam", lr=learning_rate_adam, decay=("exponential", exponential_decay))
    losshistory, train_state = model.train(iterations=adam_iterations, display_every=100)

    # Save results
    points, _, cell_types, elements = geom.get_mesh()
    n_nodes_per_cell = elements.shape[1]
    n_cells = elements.shape[0]
    cells = np.hstack([np.insert(elem, 0, n_nodes_per_cell) for elem in elements])
    cells = np.array(cells, dtype=np.int64)
    cell_types = np.array(cell_types, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    output = model.predict(points)
    displacement_pred = np.column_stack((output[:,0:1], output[:,1:2], output[:,2:3]))
    sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yx, sigma_xz, sigma_zx, sigma_yz, sigma_zy = model.predict(points, operator=cauchy_stress_3D)
    cauchy_stress_pred = np.column_stack((sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz))
    grid.point_data['pred_displacement'] = displacement_pred
    grid.point_data['pred_cauchy_stress'] = cauchy_stress_pred
    file_path = os.path.join(model_path, f"{simulation_case}_{ext_traction:03}")
    grid.save(f"{file_path}.vtu")

model.save(f"{model_path}/{simulation_case}")
dde.saveplot(
    losshistory, train_state, issave=True, isplot=False, output_dir=model_path, 
    loss_fname=f"{simulation_case}-{relaxation_adam_iterations+steps*(adam_iterations+lbfgs_iterations)}_loss.dat", 
    train_fname=f"{simulation_case}-{relaxation_adam_iterations+steps*(adam_iterations+lbfgs_iterations)}_train.dat", 
    test_fname=f"{simulation_case}-{relaxation_adam_iterations+steps*(adam_iterations+lbfgs_iterations)}_test.dat"
)