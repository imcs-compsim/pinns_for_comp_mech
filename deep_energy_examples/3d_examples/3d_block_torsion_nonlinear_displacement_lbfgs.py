import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
from deepxde.backend import tf
from pyevtk.hl import unstructuredGridToVTK
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deepxde import backend as bkd

'''
@author: tsahin

Simple compression test for a 3D block, results seem identical to 2D.
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
    theta_deg = 150
    theta = np.radians(theta_deg)
    s = x_loc / length

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
# model.compile("adam", lr=0.001)
# losshistory, train_state = model.train(epochs=3000, display_every=200)

# dde.optimizers.set_LBFGS_options(
#                                 maxiter=2000
#                                 )
LBFGS_options["iter_per_step"] = 200
LBFGS_options["maxiter"] = 2000

model.compile("L-BFGS")
losshistory, train_state = model.train(display_every=200)

X, offset, cell_types, elements = geom.get_mesh()

output = model.predict(X)
sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yx, sigma_xz, sigma_zx, sigma_yz, sigma_zy = model.predict(X, operator=cauchy_stress_3D)
p_xx, p_yy, p_zz, p_xy, p_yx, p_xz, p_zx, p_yz, p_zy = model.predict(X, operator=first_piola_stress_tensor_3D)

# .tolist() is applied to remove datatype
u_pred, v_pred, w_pred = output[:,0].tolist(), output[:,1].tolist(), output[:,2].tolist() # displacements
sigma_xx_pred, sigma_yy_pred, sigma_zz_pred = sigma_xx.flatten().tolist(), sigma_yy.flatten().tolist(), sigma_zz.flatten().tolist() # normal stresses
sigma_xy_pred, sigma_yz_pred, sigma_xz_pred = sigma_xy.flatten().tolist(), sigma_yz.flatten().tolist(), sigma_xz.flatten().tolist() # shear stresses
p_xx_pred, p_yy_pred, p_zz_pred = p_xx.flatten().tolist(), p_yy.flatten().tolist(), p_zz.flatten().tolist() # normal stresses
p_xy_pred, p_yz_pred, p_xz_pred = p_xy.flatten().tolist(), p_yz.flatten().tolist(), p_xz.flatten().tolist() # shear stresses

combined_disp_pred = tuple(np.vstack((u_pred, v_pred, w_pred)))
combined_normal_stress_pred = tuple(np.vstack((sigma_xx_pred, sigma_yy_pred, sigma_zz_pred))) 
combined_shear_stress_pred = np.vstack((sigma_xy_pred, sigma_yz_pred, sigma_xz_pred))
combined_normal_p_pred = tuple(np.vstack((p_xx_pred, p_yy_pred, p_zz_pred))) 
combined_shear_p_pred = np.vstack((p_xy_pred, p_yz_pred, p_xz_pred))

x = X[:,0].flatten()
y = X[:,1].flatten()
z = X[:,2].flatten()

file_path = os.path.join(os.getcwd(), "deep_energy_3d_block_torsion_nonlinear")

unstructuredGridToVTK(file_path, x, y, z, elements.flatten(), offset, 
                        cell_types, pointData = { "pred_displacement" : combined_disp_pred,
                                                "pred_normal_stress" : combined_normal_stress_pred,
                                                "pred_normal_stress_p" : combined_normal_p_pred,
                                                "pred_stress_xy": combined_shear_stress_pred[0],
                                                "pred_stress_yz": combined_shear_stress_pred[1],
                                                "pred_stress_xz": combined_shear_stress_pred[2]})

# von_mises
# sqrt(0.5 * (
#     (pred_normal_stress_X - pred_normal_stress_Y)^2 +
#     (pred_normal_stress_Y - pred_normal_stress_Z)^2 +
#     (pred_normal_stress_Z - pred_normal_stress_X)^2 +
#     6 * (pred_stress_xy^2 + pred_stress_xz^2 + pred_stress_yz^2)
# ))








