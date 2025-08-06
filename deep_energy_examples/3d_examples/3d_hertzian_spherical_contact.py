import deepxde as dde
import numpy as np
from deepxde import backend as bkd
from pathlib import Path
import os
from pyevtk.hl import unstructuredGridToVTK
import pyvista as pv

'''
@author: tsahin

Simple compression test for a 3D block, results seem identical to 2D.
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

# The applied pressure 
ext_traction = 5

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
    
    ####################################################################################################################
    # Reshape energy-work terms and sum over the gauss points  
    # internal_energy_reshaped = bkd.sum(bkd.reshape(internal_energy, (n_e, n_gp)), dim=1)
    # external_work_reshaped = bkd.sum(bkd.reshape(external_work, (n_e_boundary_external, n_gp_boundary)), dim=1)
    # contact_work_reshaped = bkd.sum(bkd.reshape(contact_work, (n_e_boundary_contact, n_gp_boundary)), dim=1)
    # sum over the elements and get the overall loss
    # total_energy = bkd.reduce_sum(internal_energy_reshaped) - bkd.reduce_sum(external_work_reshaped) + bkd.reduce_sum(contact_work_reshaped)
    
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

restore_model = False
model_path = str(Path(__file__).parent.parent)+f"/trained_models/3d_herzian_spherical_contact/3d_herzian_spherical_contact_nonlinear"

if not restore_model:
    # model.compile("adam", lr=0.001)
    # losshistory, train_state = model.train(epochs=stabilization_model_epoch, display_every=100)
    
    # apply_load = True
    
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
    
    dde.saveplot(losshistory, train_state, issave=True, isplot=False)
else:
    n_epochs = 5471 
    model_restore_path = model_path + "-"+ str(n_epochs) + ".ckpt"
    
    model.compile("adam", lr=0.001)
    model.restore(save_path=model_restore_path)

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

# file_path = os.path.join(os.getcwd(), "deep_energy_3d_hertzian_spherical_nonlinear")

# unstructuredGridToVTK(file_path, x, y, z, elements.flatten(), offset, 
#                         cell_types, pointData = { "pred_displacement" : combined_disp_pred,
#                                                 "pred_normal_stress" : combined_normal_stress_pred,
#                                                 "pred_normal_stress_p" : combined_normal_p_pred,
#                                                 "pred_stress_xy": combined_shear_stress_pred[0],
#                                                 "pred_stress_yz": combined_shear_stress_pred[1],
#                                                 "pred_stress_xz": combined_shear_stress_pred[2]})


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

von_mises = np.sqrt(
    0.5 * (
        (sigma_xx - sigma_yy) ** 2 +
        (sigma_yy - sigma_zz) ** 2 +
        (sigma_zz - sigma_xx) ** 2 +
        6 * (sigma_xy ** 2 + sigma_yz ** 2 + sigma_xz ** 2)
    )
)

# Convert to spherical components
# sigma_rr, sigma_thetatheta, sigma_phiphi, sigma_rtheta, sigma_thetaphi, sigma_rphi = polar_transformation_3d_spherical(
#     sigma_xx.flatten(),
#     sigma_yy.flatten(),
#     sigma_zz.flatten(),
#     sigma_xy.flatten(),
#     sigma_yz.flatten(),
#     sigma_xz.flatten(),
#     points
# )
# cauchy_stress_polar = np.column_stack((sigma_rr.reshape(-1,1), sigma_thetatheta.reshape(-1,1), sigma_phiphi.reshape(-1,1), sigma_rtheta.reshape(-1,1), sigma_thetaphi.reshape(-1,1), sigma_rphi.reshape(-1,1)))

grid.point_data['pred_displacement'] = displacement
grid.point_data['pred_cauchy_stress'] = cauchy_stress
grid.point_data['von_mises'] = von_mises

# grid.point_data['pred_cauchy_stress_polar'] = cauchy_stress_polar

grid.save("deep_energy_3d_hertzian_spherical_nonlinear.vtu")

