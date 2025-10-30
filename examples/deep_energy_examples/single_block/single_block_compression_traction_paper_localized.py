import deepxde as dde
import numpy as np
import os
import sys
from pathlib import Path
from pyevtk.hl import unstructuredGridToVTK

from compsim_pinns.elasticity.elasticity_utils import problem_parameters, elastic_strain_2d, stress_plane_strain
from compsim_pinns.geometry.geometry_utils import calculate_boundary_normals, polar_transformation_2d
from compsim_pinns.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from compsim_pinns.geometry.gmsh_models import Block_2D
from compsim_pinns.elasticity import elasticity_utils

import gmsh
from deepxde import backend as bkd

from compsim_pinns.deep_energy.deep_pde import DeepEnergyPDE

from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule
from compsim_pinns.vpinns.quad_rule import get_test_function_properties


'''
Solves a hollow quarter cylinder under internal pressure (Lame problem)

Reference solution:
https://onlinelibrary.wiley.com/doi/epdf/10.1002/nme.6132

Reference for PINNs formulation:
A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics

@author: tsahin
'''


gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 11}
coord_left_corner=[0,0.]
coord_right_corner=[1,1]
l_beam = coord_right_corner[0] - coord_left_corner[0]
h_beam = coord_right_corner[1] - coord_left_corner[1]

block_2d = Block_2D(coord_left_corner=coord_left_corner, coord_right_corner=coord_right_corner, mesh_size=0.1, gmsh_options=gmsh_options)

quad_rule = GaussQuadratureRule(rule_name="gauss_legendre", dimension=2, ngp=4) # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()

quad_rule_boundary_integral = GaussQuadratureRule(rule_name="gauss_legendre", dimension=1, ngp=8) # gauss_legendre gauss_labotto
coord_quadrature_boundary, weight_quadrature_boundary = quad_rule_boundary_integral.generate()

gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)

def on_right(x):
    points_on_right = np.isclose(x[0],l_beam)
    y_cut = (x[1] >= 0.6) and ((x[1] <= 0.8))
    return points_on_right and y_cut

boundary_selection_map = [{"boundary_function" : on_right, "tag" : "on_right"}]

revert_curve_list = []
revert_normal_dir_list = [1,1,2,1]

geom = GmshGeometryElementDeepEnergy(
                           gmsh_model,
                           dimension=2, 
                           coord_quadrature=coord_quadrature, 
                           weight_quadrature= weight_quadrature, 
                           revert_curve_list=revert_curve_list, 
                           revert_normal_dir_list=revert_normal_dir_list,
                           coord_quadrature_boundary=coord_quadrature_boundary,
                           weight_quadrature_boundary=weight_quadrature_boundary,
                           boundary_selection_map=boundary_selection_map
                           )

# The applied pressure
pressure = 0.1
nu,lame,shear,e_modul = problem_parameters()
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
    
    eps_xx, eps_yy, eps_xy = elastic_strain_2d(inputs,outputs)
    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(inputs,outputs)
    
    # get the internal energy
    internal_energy_density = 1/2*(sigma_xx[beg_pde:beg_boundary]*eps_xx[beg_pde:beg_boundary] + 
                            sigma_yy[beg_pde:beg_boundary]*eps_yy[beg_pde:beg_boundary] + 
                          2*sigma_xy[beg_pde:beg_boundary]*eps_xy[beg_pde:beg_boundary])
    
    internal_energy = global_element_weights_t[:,0:1]*global_element_weights_t[:,1:2]*(internal_energy_density)*jacobian_t
    
    # get the external energy
    # select the points where external force is applied
    cond = boundary_selection_tag["on_right"]
    
    u_x = outputs[:,0:1][beg_boundary:][cond]
    #u_y = outputs[:,1:2][beg_boundary:][cond]
    
    external_force_density = pressure*u_x
    external_work = global_weights_boundary_t[cond]*(external_force_density)*jacobian_boundary_t[cond]
    
    # internal_energy_reshaped = bkd.reshape(internal_energy, (n_e, n_gp))
    # external_work_reshaped = bkd.reshape(external_work, (n_e_boundary, n_gp_boundary))
    
    # total_energy = bkd.reduce_sum(bkd.sum(internal_energy_reshaped, dim=1)) - bkd.reduce_sum(bkd.sum(external_work_reshaped, dim=1)) #+ bkd.reduce_sum(bkd.sum(internal_energy_reshaped, dim=1)) 
    
    return [internal_energy, -external_work]

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

    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    
    return bkd.concat([u*(x_loc),v*(x_loc)], axis=1)

# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)

model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=3000, display_every=100)

model.compile("L-BFGS")
model.train_step.optimizer_kwargs["options"]['maxiter']=2000
losshistory, train_state = model.train(display_every=200)

X, offset, cell_types, dol_triangles = geom.get_mesh()
nu,lame,shear,e_modul = problem_parameters()

# start_time_calc = time.time()
output = model.predict(X)
# end_time_calc = time.time()
# final_time = f'Prediction time: {(end_time_calc - start_time_calc):.3f} seconds'
# print(final_time)

u_x_pred, u_y_pred = output[:,0], output[:,1]
u_pred, v_pred = output[:,0], output[:,1]
sigma_xx, sigma_yy, sigma_xy = model.predict(X, operator=stress_plane_strain)

combined_disp = tuple(np.vstack((u_x_pred, u_y_pred, np.zeros(u_x_pred.shape[0]))))
combined_stress = tuple(np.vstack((sigma_xx.flatten(), sigma_yy.flatten(), sigma_xy.flatten())))


file_path = os.path.join(os.getcwd(), "deep_energy_single_block_compression_traction_paper_localized")

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)

#np.savetxt("Lame_inverse_large", X=np.hstack((X,output[:,0:2])))

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                      cell_types, pointData = { "displacement" : combined_disp,"stress" : combined_stress})