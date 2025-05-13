import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
from pyevtk.hl import unstructuredGridToVTK
from pathlib import Path

from utils.elasticity.elasticity_utils import stress_plane_strain, stress_plane_stress
from utils.geometry.geometry_utils import calculate_boundary_normals, polar_transformation_2d
from utils.geometry.custom_geometry import GmshGeometryElement
from utils.geometry.gmsh_models import QuarterCirclewithHole

from utils.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from utils.geometry.gmsh_models import Block_2D

from utils.hyperelasticity.hyperelasticity_utils import strain_energy_neo_hookean_2d, compute_elastic_properties, cauchy_stress_2D
from utils.hyperelasticity import hyperelasticity_utils
from utils.elasticity import elasticity_utils
from deepxde import backend as bkd

from utils.deep_energy.deep_pde import DeepEnergyPDE

from utils.vpinns.quad_rule import GaussQuadratureRule


from utils.postprocess.custom_callbacks import SaveModelVTU

'''
Solves a hollow quarter cylinder under internal pressure (Lame problem)

Reference solution:
https://onlinelibrary.wiley.com/doi/epdf/10.1002/nme.6132

@author: tsahin
'''


gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 11}
block_2d = Block_2D(coord_left_corner=[0,-1], coord_right_corner=[20,1], mesh_size=0.2, gmsh_options=gmsh_options)

gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)

quad_rule = GaussQuadratureRule(rule_name="gauss_legendre", dimension=2, ngp=2) # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()

quad_rule_boundary_integral = GaussQuadratureRule(rule_name="gauss_legendre", dimension=1, ngp=4) # gauss_legendre gauss_labotto
coord_quadrature_boundary, weight_quadrature_boundary = quad_rule_boundary_integral.generate()

l_beam = block_2d.coord_right_corner[0] -block_2d.coord_left_corner[0]
h_beam = block_2d.coord_right_corner[1] -block_2d.coord_left_corner[1]

def boundary_right(x):
    return np.isclose(x[0],l_beam)

boundary_selection_map = [{"boundary_function" : boundary_right, "tag" : "boundary_right"}]

revert_curve_list = []
revert_normal_dir_list = [1,2,1,1]

geom = GmshGeometryElementDeepEnergy(
                           gmsh_model,
                           dimension=2,
                           coord_quadrature=coord_quadrature,
                           weight_quadrature= weight_quadrature,
                           revert_curve_list=revert_curve_list,
                           revert_normal_dir_list=revert_normal_dir_list,
                           coord_quadrature_boundary=coord_quadrature_boundary,
                           weight_quadrature_boundary=weight_quadrature_boundary,
                           boundary_selection_map=boundary_selection_map)

# change global variables in elasticity_utils
hyperelasticity_utils.lame = 2.78
hyperelasticity_utils.shear = 4.17
hyperelasticity_utils.stress_state = "plane_strain"
nu,lame,shear,e_modul = compute_elastic_properties()

# The applied pressure
shear_load = 1e-2

apply_load = False

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

    internal_energy_density = strain_energy_neo_hookean_2d(inputs, outputs)

    internal_energy = global_element_weights_t[:,0:1]*global_element_weights_t[:,1:2]*(internal_energy_density[beg_pde:beg_boundary])*jacobian_t
    ####################################################################################################################
    # get the external work
    # select the points where external force is applied
    cond = boundary_selection_tag["boundary_right"]
    nx = mapped_normal_boundary_t[:,0:1][cond]
    ny = mapped_normal_boundary_t[:,1:2][cond]

    u_x = outputs[:,0:1][beg_boundary:][cond]
    u_y = outputs[:,1:2][beg_boundary:][cond]

    if not apply_load:
        shear_load_local = 0
    else:
        shear_load_local = shear_load

    external_force_density = -shear_load_local*u_y
    external_work = global_weights_boundary_t[cond]*(external_force_density)*jacobian_boundary_t[cond]

    ####################################################################################################################
    # Reshape energy-work terms and sum over the gauss points
    # internal_energy_reshaped = bkd.sum(bkd.reshape(internal_energy, (n_e, n_gp)), dim=1)
    # external_work_reshaped = bkd.sum(bkd.reshape(external_work, (n_e_boundary_external, n_gp_boundary)), dim=1)
    # sum over the elements and get the overall loss
    #total_energy = bkd.reduce_sum(internal_energy_reshaped) #- bkd.reduce_sum(external_work_reshaped)

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

    return bkd.concat([u*x_loc/e_modul,v*x_loc/e_modul], axis=1)

# two inputs x and y, output is ux and uy
layer_size = [2] + [64] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

file_path = os.path.join(os.getcwd(), "Beam_under_shear_load_nonlinear")
stabilization_model_epoch = 100
model_saver = SaveModelVTU(op=cauchy_stress_2D, period=1000, stabilization_epoch=stabilization_model_epoch, filename=file_path)

model = dde.Model(data, net)
# if we want to save the model, we use "model_save_path=model_path" during training, if we want to load trained model, we use "model_restore_path=return_restore_path(model_path, num_epochs)"

model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=stabilization_model_epoch, display_every=100)

apply_load = True

model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=12000, display_every=100)

# model.compile("L-BFGS")
# model.train()

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

X, offset, cell_types, dol_triangles = geom.get_mesh()

displacement = model.predict(X)
sigma_xx, sigma_yy, sigma_xy, sigma_yx = model.predict(X, operator=cauchy_stress_2D)
sigma_rr, sigma_theta, sigma_rtheta = polar_transformation_2d(sigma_xx, sigma_yy, sigma_xy, X)

combined_disp = tuple(np.vstack((np.array(displacement[:,0].tolist()),np.array(displacement[:,1].tolist()),np.zeros(displacement[:,0].shape[0]))))
combined_stress = tuple(np.vstack((np.array(sigma_xx.flatten().tolist()),np.array(sigma_yy.flatten().tolist()),np.array(sigma_xy.flatten().tolist()))))
combined_stress_polar = tuple(np.vstack((np.array(sigma_rr.tolist()),np.array(sigma_theta.tolist()),np.array(sigma_rtheta.tolist()))))

file_path = os.path.join(os.getcwd(), "Beam_under_shear_load_nonlinear")

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset,
                      cell_types, pointData = { "displacement" : combined_disp,"stress" : combined_stress, "stress_polar": combined_stress_polar})




