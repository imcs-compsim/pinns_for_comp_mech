import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
from pyevtk.hl import unstructuredGridToVTK
from pathlib import Path
import matplotlib.tri as tri
import pandas as pd
import pyvista as pv

from compsim_pinns.elasticity import elasticity_utils

from compsim_pinns.geometry.custom_geometry import GmshGeometryElementDeepEnergy

from compsim_pinns.geometry.gmsh_models import QuarterTorus3D
from compsim_pinns.hyperelasticity import hyperelasticity_utils
from compsim_pinns.contact_mech.contact_utils import calculate_gap_in_normal_direction_deep_energy
from compsim_pinns.contact_mech import contact_utils

from compsim_pinns.hyperelasticity.hyperelasticity_utils import strain_energy_neo_hookean_3d, compute_elastic_properties, first_piola_stress_tensor_3D, cauchy_stress_3D, green_lagrange_strain_3D

from deepxde.optimizers.config import LBFGS_options

from deepxde import backend as bkd

from compsim_pinns.deep_energy.deep_pde import DeepEnergyPDE

from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule

tube_radius = 0.05
major_radius = 0.95
center = [0,0,0]

quarter_circle_with_hole = QuarterTorus3D(center=center, major_radius=major_radius, tube_radius=tube_radius, mesh_size=0.015)

gmsh_model = quarter_circle_with_hole.generateGmshModel(visualize_mesh=True)

def on_contact(x):
    return np.isclose((np.sqrt(x[0]**2+x[1]**2) - major_radius)**2 + x[2]**2, tube_radius**2)

def on_top(x):
    return np.isclose(x[1],0)

boundary_selection_map = [{"boundary_function" : on_contact, "tag" : "on_contact"},
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
                           boundary_dim=boundary_dimension,
                           coord_quadrature_boundary=coord_quadrature_boundary,
                           weight_quadrature_boundary=weight_quadrature_boundary,
                           boundary_selection_map=boundary_selection_map)
# change global variables in elasticity_utils
hyperelasticity_utils.e_modul = 10
hyperelasticity_utils.nu = 0.3
hyperelasticity_utils.stress_state = "plane_strain"
nu,lame,shear,e_modul = compute_elastic_properties()

applied_disp_y = -0.8

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
    # contact work
    cond = boundary_selection_tag["on_contact"]
    
    # gap_y = inputs[:,1:2][beg_boundary:][cond] + outputs[:,1:2][beg_boundary:][cond] + radius
    # gap_n = tf.math.divide_no_nan(gap_y, tf.math.abs(mapped_normal_boundary_t[:,1:2][cond]))
    gap_n = calculate_gap_in_normal_direction_deep_energy(inputs[beg_boundary:], outputs[beg_boundary:], X, mapped_normal_boundary_t, cond)
    eta=3e4
    contact_force_density = 1/2*eta*bkd.relu(-gap_n)*bkd.relu(-gap_n)
    contact_work = global_weights_boundary_t[cond]*(contact_force_density)*jacobian_boundary_t[cond]
    
    ####################################################################################################################
    # Reshape energy-work terms and sum over the gauss points  
    # internal_energy_reshaped = bkd.sum(bkd.reshape(internal_energy, (n_e, n_gp)), dim=1)
    # external_work_reshaped = bkd.sum(bkd.reshape(external_work, (n_e_boundary_external, n_gp_boundary)), dim=1)
    # contact_work_reshaped = bkd.sum(bkd.reshape(contact_work, (n_e_boundary_contact, n_gp_boundary)), dim=1)
    # sum over the elements and get the overall loss
    # total_energy = bkd.reduce_sum(internal_energy_reshaped) - bkd.reduce_sum(external_work_reshaped) + bkd.reduce_sum(contact_work_reshaped)
    
    return [internal_energy, contact_work]

n_dummy = 1
data = DeepEnergyPDE(
    geom,
    potential_energy,
    [],
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=None
)

def output_transform(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    w = y[:, 2:3]

    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    z_loc = x[:, 2:3]
    # tf.math.abs(x_loc)
    return bkd.concat([u*(y_loc*x_loc)/e_modul, 
                       v*(-y_loc)/e_modul+applied_disp_y,
                       w*(z_loc)/e_modul], axis=1)

# 3 inputs, 3 outputs for 3D 
layer_size = [3] + [50] * 5 + [3]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)
restore_model = False
model_path = str(Path(__file__).parent.parent.parent)+f"/pretrained_models/deep_energy_examples/3d_contact_ring_instability/3d_contact_ring_instability_quarter_torus"

if not restore_model:
    # model.compile("adam", lr=0.001)
    # losshistory, train_state = model.train(epochs=stabilization_model_epoch, display_every=100)
    
    model.compile("adam", lr=0.001)
    losshistory, train_state = model.train(epochs=5000, display_every=100)
    # if you want to save the model, run the following
    # losshistory, train_state = model.train(epochs=5000, display_every=100, model_save_path=model_path)
    
    # For pytorch
    # LBFGS_options["iter_per_step"] = 1
    # LBFGS_options["maxiter"] = 500
    
    LBFGS_options["maxiter"] = 1000
    model.compile("L-BFGS")
    losshistory, train_state = model.train(display_every=100)
    # losshistory, train_state = model.train(display_every=100, model_save_path=model_path)
else:
    n_epochs = 5471 
    model_restore_path = model_path + "-"+ str(n_epochs) + ".ckpt"
    
    model.compile("adam", lr=0.001)
    model.restore(save_path=model_restore_path)

# Get mesh data from your class
points, _, cell_types, elements = geom.get_mesh()

# 1. Flatten elements and prepend number of nodes per cell
n_nodes_per_cell = elements.shape[1]
n_cells = elements.shape[0]

cells = np.hstack([np.insert(elem, 0, n_nodes_per_cell) for elem in elements])

# 2. Make sure data types are correct
cells = np.array(cells, dtype=np.int64)
cell_types = np.array(cell_types, dtype=np.uint8)

# 3. Create the UnstructuredGrid
grid = pv.UnstructuredGrid(cells, cell_types, points)

# Predict the displacements
output = model.predict(points)

# Predict stress components
sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yx, sigma_xz, sigma_zx, sigma_yz, sigma_zy = model.predict(points, operator=cauchy_stress_3D)

cauchy_stress = np.column_stack((sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz))
displacement = np.column_stack((output[:,0:1], output[:,1:2], output[:,2:3]))


grid.point_data['pred_displacement'] = displacement
grid.point_data['pred_cauchy_stress'] = cauchy_stress

grid.save("deep_energy_3d_contact_ring_instability_quarter_torus.vtu")