import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
from pyevtk.hl import unstructuredGridToVTK
from pathlib import Path

from compsim_pinns.elasticity.elasticity_utils import stress_plane_strain, stress_plane_stress
from compsim_pinns.geometry.geometry_utils import calculate_boundary_normals, polar_transformation_2d
from compsim_pinns.geometry.custom_geometry import GmshGeometryElement
from compsim_pinns.geometry.gmsh_models import QuarterCirclewithHole
import pyvista as pv

from compsim_pinns.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from compsim_pinns.geometry.gmsh_models import Block_2D

from compsim_pinns.hyperelasticity.hyperelasticity_utils import strain_energy_neo_hookean_2d, compute_elastic_properties, cauchy_stress_2D, green_lagrange_strain_2D, deformation_gradient_2D
from compsim_pinns.hyperelasticity import hyperelasticity_utils
from compsim_pinns.elasticity import elasticity_utils
from deepxde import backend as bkd
from deepxde.optimizers.config import LBFGS_options

from compsim_pinns.deep_energy.deep_pde import DeepEnergyPDE

from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule


from compsim_pinns.postprocess.custom_callbacks import SaveModelVTU
from compsim_pinns.hyperelasticity.hyperelasticity_utils import cauchy_stress_2D, first_piola_stress_tensor_2D

'''
Solves a hollow quarter cylinder under internal pressure (Lame problem)

Reference solution:
https://onlinelibrary.wiley.com/doi/epdf/10.1002/nme.6132

@author: tsahin
'''
# Sharp corners or geometric discontinuities (like re-entrant corners or sudden boundary changes) can cause stress singularities
# https://www.fidelisfea.com/post/stress-singularities-at-reentrant-corners-a-fundamental-problem-in-fea

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

    return bkd.concat([u*x_loc/l_beam,v*x_loc/l_beam], axis=1)

# two inputs x and y, output is ux and uy
layer_size = [2] + [64] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

file_path = os.path.join(os.getcwd(), "Beam_under_shear_load_nonlinear")
# stabilization_model_epoch = 100
# model_saver = SaveModelVTU(op=cauchy_stress_2D, period=1000, stabilization_epoch=stabilization_model_epoch, filename=file_path)

model = dde.Model(data, net)
# if we want to save the model, we use "model_save_path=model_path" during training, if we want to load trained model, we use "model_restore_path=return_restore_path(model_path, num_epochs)"

restore_model = True
model_path = str(Path(__file__).parent.parent.parent)+f"/pretrained_models/deep_energy_examples/beam/beam_nonlinear"

if not restore_model:
    # model.compile("adam", lr=0.001)
    # losshistory, train_state = model.train(epochs=stabilization_model_epoch, display_every=100)
    
    apply_load = True
    
    model.compile("adam", lr=0.001)
    losshistory, train_state = model.train(epochs=5000, display_every=100)
    # if you want to save the model, run the following
    # losshistory, train_state = model.train(epochs=5000, display_every=100, model_save_path=model_path)
    
    # For pytorch
    # LBFGS_options["iter_per_step"] = 1
    # LBFGS_options["maxiter"] = 500
    
    LBFGS_options["maxiter"] = 1500
    model.compile("L-BFGS")
    losshistory, train_state = model.train(display_every=100)
    # losshistory, train_state = model.train(display_every=100, model_save_path=model_path)
    
    dde.saveplot(losshistory, train_state, issave=True, isplot=False)
else:
    n_epochs = 5083 
    model_restore_path = model_path + "-"+ str(n_epochs) + ".ckpt"
    
    model.compile("adam", lr=0.001)
    model.restore(save_path=model_restore_path)

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

file_path =  str(Path(__file__).parent.parent)+f"/reference_results/beam/bending_beam_test-structure.pvd"
save_file_path = os.path.join(os.getcwd(), "Beam_under_shear_load_nonlinear")

# Convert the Path object to a string
reader = pv.get_reader(file_path)

reader.set_active_time_point(-1)
data = reader.read()[0]

# FEM geometry does not match with PINNs, it should be shifted 10 unit to the right
data.points[:, 0] += 10
X = data.points[:,0:2]

output = model.predict(X)
sigma_xx, sigma_yy, sigma_xy, _ = model.predict(X, operator=cauchy_stress_2D)
eps_xx, eps_yy, eps_xy = model.predict(X, operator=green_lagrange_strain_2D)
f_xx, f_yy, f_yx, f_xy = model.predict(X, operator=deformation_gradient_2D)

cauchy_stress = np.column_stack((sigma_xx, sigma_yy, sigma_xy))
strain = np.column_stack((eps_xx, eps_yy, eps_xy))
deformation_gradient = np.column_stack((f_xx, f_yy, f_yx, f_xy, np.zeros_like(f_xx), np.zeros_like(f_xx)))
# first_piola_stress = np.column_stack((p_xx, p_yy, p_xy))
displacement = np.column_stack((output[:,0:1], output[:,1:2], np.zeros_like(output[:,0:1])))

data.point_data['pred_displacement'] = displacement
data.point_data['pred_cauchy_stress'] = cauchy_stress
data.point_data['pred_strain'] = strain
data.point_data['deformation_gradient'] = deformation_gradient
# data.point_data['pred_first_piola_stress'] = first_piola_stress

disp_fem = data.point_data['displacement']
stress_fem = data.point_data['nodal_cauchy_stresses_xyz']

error_disp = abs((disp_fem - displacement[:,0:2]))
data.point_data['pointwise_displacement_error'] = error_disp
# select xx, yy, and xy component (1st, 2nd and 4th column)
columns = [0,1,3]
error_stress = abs((stress_fem[:,columns] - cauchy_stress))
data.point_data['pointwise_cauchystress_error'] = error_stress

data.save(f"{save_file_path}.vtu")

# X, offset, cell_types, dol_triangles = geom.get_mesh()

# displacement = model.predict(X)
# sigma_xx, sigma_yy, sigma_xy, sigma_yx = model.predict(X, operator=cauchy_stress_2D)

# combined_disp = tuple(np.vstack((np.array(displacement[:,0].tolist()),np.array(displacement[:,1].tolist()),np.zeros(displacement[:,0].shape[0]))))
# combined_stress = tuple(np.vstack((np.array(sigma_xx.flatten().tolist()),np.array(sigma_yy.flatten().tolist()),np.array(sigma_xy.flatten().tolist()))))

# file_path = os.path.join(os.getcwd(), "Beam_under_shear_load_nonlinear")

# x = X[:,0].flatten()
# y = X[:,1].flatten()
# z = np.zeros(y.shape)

# unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset,
#                       cell_types, pointData = { "displacement" : combined_disp,"stress" : combined_stress})




