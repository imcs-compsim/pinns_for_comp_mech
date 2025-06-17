import deepxde as dde
import numpy as np
from deepxde import backend as bkd
from pathlib import Path
import gmsh
'''
Quarter cylinder hertzian contact problem using a nonlinear complimentary problem (NCP) function
Enhanced by results from analytical solution
@author: svoelkl
based on the work of tsahin
'''

from utils.geometry.custom_geometry import GmshGeometry3D
from utils.geometry.gmsh_models import Geom_step_to_gmsh

from utils.elasticity import elasticity_utils
from utils.elasticity.elasticity_utils import pde_mixed_3d, problem_parameters
from utils.elasticity.elasticity_utils import apply_zero_neumann_x_mixed_formulation, apply_zero_neumann_y_mixed_formulation, apply_zero_neumann_z_mixed_formulation
from utils.postprocess.elasticity_postprocessing import solutionFieldOnMeshToVtk3D

from utils.contact_mech import contact_utils
from utils.contact_mech.contact_utils import zero_tangential_traction_component1_3d, zero_tangential_traction_component2_3d, zero_complementarity_function_based_fisher_burmeister_3d

path_to_step_file = str(Path(__file__).parent.parent.parent)+f"/step_files/hertzian_quarter_cylinder.stp"

curve_info = {"7":15, "9":15, 
              "14":8, "18":8, 
              "8":40, "6":25,
              "2":15, "16":15}
geom_obj = Geom_step_to_gmsh(path=path_to_step_file, curve_info=curve_info)
gmsh_model = geom_obj.generateGmshModel(visualize_mesh=False)

geom = GmshGeometry3D(gmsh_model)

projection_plane = {"y" : -1} # projection plane formula
pressure = -0.5
center = [0, 0, 0]
radius = 1
b_limit = -0.25

# # change global variables in elasticity_utils, they are used for getting the material properties for analytical model
lame = 115.38461538461539
shear = 76.92307692307692
elasticity_utils.lame = lame
elasticity_utils.shear = shear
nu,lame,shear,e_modul = problem_parameters() # with dimensions, will be used for analytical solution
# This will lead to e_modul=200 and nu=0.3

# assign local parameters from the current file in contact_utils and elasticity_utils
elasticity_utils.geom = geom
contact_utils.geom = geom
contact_utils.projection_plane = projection_plane
    
def boundary_not_contact(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x[:2] - center[:2], axis=-1), radius) and (x[0]<b_limit)

def boundary_contact(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x[:2] - center[:2], axis=-1), radius) and (x[0]>=b_limit)

def bottom_point(x, on_boundary):
    points_at_x_0 = np.isclose(x[0],0)
    points_on_the_radius = np.isclose(np.linalg.norm(x[:2] - center[:2], axis=-1), radius)

    return on_boundary and points_on_the_radius and points_at_x_0

# # Neumann BCs on non-contact zones of sphere
bc1 = dde.OperatorBC(geom, apply_zero_neumann_x_mixed_formulation, boundary_not_contact)
bc2 = dde.OperatorBC(geom, apply_zero_neumann_y_mixed_formulation, boundary_not_contact)
bc3 = dde.OperatorBC(geom, apply_zero_neumann_z_mixed_formulation, boundary_not_contact)

# # Contact BCs
# enforce tangential tractions to be zero
bc4 = dde.OperatorBC(geom, zero_tangential_traction_component1_3d, boundary_contact)
bc5 = dde.OperatorBC(geom, zero_tangential_traction_component2_3d, boundary_contact)
# KKT using fisher_burmeister
bc6 = dde.OperatorBC(geom, zero_complementarity_function_based_fisher_burmeister_3d, boundary_contact)

# additional data
p_max = 8.36
b = 0.07611333607551958
n_test = 50
z = np.linspace(0,3*b,n_test).reshape(-1,1)

s_z = -2*nu*p_max*(np.sqrt(1+z**2/b**2) - np.abs(z/b))
s_x = -p_max*((1+2*(z**2/b**2))/(np.sqrt(1+z**2/b**2)) - 2*np.abs(z/b))
s_y = -p_max/(np.sqrt(1+z**2/b**2))

y_coord = np.linspace(-1,-0.7642,n_test).reshape(-1,1)

ex_data_xyz_1 = np.hstack((np.zeros_like(y_coord), y_coord, -1*np.ones_like(y_coord)))
ex_data_xyz_2 = np.hstack((np.zeros_like(y_coord), y_coord, -0.5*np.ones_like(y_coord)))
ex_data_xyz_3 = np.hstack((np.zeros_like(y_coord), y_coord, np.zeros_like(y_coord)))

ex_data_xyz = np.vstack((ex_data_xyz_1,ex_data_xyz_2,ex_data_xyz_3))
s_x = np.vstack((s_x,s_x,s_x))
s_y = np.vstack((s_y,s_y,s_y))
s_z = np.vstack((s_z,s_z,s_z))

observe_sigma_xx = dde.PointSetBC(ex_data_xyz, s_x, component=3)
observe_sigma_yy = dde.PointSetBC(ex_data_xyz, s_y, component=4)
observe_sigma_zz = dde.PointSetBC(ex_data_xyz, s_z, component=5)

# bc7 = dde.DirichletBC(geom, lambda _: 0, bottom_point, component=1)

n_dummy = 1
data = dde.data.PDE(
    geom,
    pde_mixed_3d,
    [bc1, bc2, bc3, bc4, bc5, bc6, observe_sigma_xx, observe_sigma_yy, observe_sigma_zz],
    num_domain=1,
    num_boundary=n_dummy,
    num_test=None,
    train_distribution = "Sobol",
    anchors=ex_data_xyz
)

def output_transform(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    w = y[:, 2:3]
    
    sigma_xx =  y[:, 3:4]
    sigma_yy =  y[:, 4:5]
    sigma_zz =  y[:, 5:6]
    sigma_xy =  y[:, 6:7]
    sigma_yz =  y[:, 7:8]
    sigma_xz =  y[:, 8:9]
    
    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    z_loc = x[:, 2:3]
    
    # define surfaces
    top_surface = -y_loc
    front_surface = -z_loc
    back_surface = radius + z_loc
    right_surface = -x_loc
    
    # define the surfaces where shear forces will be applied.
    sigma_xy_surfaces = top_surface*right_surface
    sigma_yz_surfaces = top_surface*front_surface*back_surface
    sigma_xz_surfaces = front_surface*back_surface*right_surface
    
    return bkd.concat([u*(right_surface)/e_modul, #displacement in x direction is 0 at x=0
                      v/e_modul,
                      w*(back_surface)*(front_surface)/e_modul, #displacement in z direction is 0 at z=0
                      sigma_xx, 
                      pressure + sigma_yy*(top_surface),
                      sigma_zz,
                      sigma_xy*sigma_xy_surfaces,
                      sigma_yz*sigma_yz_surfaces,
                      sigma_xz*sigma_xz_surfaces
                      ], axis=1)

# 3 inputs, 9 outputs for 3D 
layer_size = [3] + [50] * 5 + [9]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)
# weights due to PDE
w_momentum_xx, w_momentum_yy, w_momentum_zz = 1e0, 1e0, 1e0
# weights due to stress-stress coupling
w_s_xx, w_s_yy, w_s_zz, w_s_xy, w_s_yz, w_s_xz = 1e0, 1e0, 1e0, 1e0, 1e0, 1e0
# weights due to Neumann BCs
w_zero_traction_x, w_zero_traction_y, w_zero_traction_z = 1e0, 1e0, 1e0
# weights due to Contact BCs
w_zero_tangential_traction_component1 = 1e0
w_zero_tangential_traction_component2 = 1e0
w_zero_fisher_burmeister = 5e2
# single dirichlet
# w_dirichlet = 1e0

loss_weights = [w_momentum_xx, w_momentum_yy, w_momentum_zz, 
                w_s_xx, w_s_yy, w_s_zz, w_s_xy, w_s_yz, w_s_xz,  
                w_zero_traction_x, w_zero_traction_y, w_zero_traction_z,
                w_zero_tangential_traction_component1, w_zero_tangential_traction_component2, w_zero_fisher_burmeister,
                1,1,1]

model_path = str(Path(__file__).parent.parent.parent)+f"/trained_models/hertzian/hertzian_3d_enhanced_analytical"
restore_model = False

if not restore_model:
    model.compile("adam", lr=0.001, loss_weights=loss_weights)
    losshistory, train_state = model.train(epochs=2000, display_every=100) 
    # losshistory, train_state = model.train(epochs=2000, display_every=200, model_save_path=model_path) # use if you want to save the model

    model.compile("L-BFGS", loss_weights=loss_weights)
    losshistory, train_state = model.train(display_every=200)
    # losshistory, train_state = model.train(display_every=200, model_save_path=model_path) # same as above
else:
    n_epochs = 12973 
    model_restore_path = model_path + "-"+ str(n_epochs) + ".ckpt"
    
    model.compile("adam", lr=0.001)
    model.restore(save_path=model_restore_path)
    
#########################################################################################################################################
#### POST-PROCESSING #####
#########################################################################################################################################
## gmsh.finalize()
gmsh.clear()

curve_info = {"7":20, "9":20, 
              "14":20, "18":20, 
              "8":40, "6":40, "10":40,
              "2":40, "16":40,
              "4":40, "12":40,
              "1":15, "5":15, "13":15, "15":15}
geom_obj = Geom_step_to_gmsh(path=path_to_step_file, curve_info=curve_info)
gmsh_model = geom_obj.generateGmshModel(visualize_mesh=False)

geom = GmshGeometry3D(gmsh_model)

solutionFieldOnMeshToVtk3D(geom, 
                           model, 
                           save_folder_path=str(Path(__file__).parent.parent.parent.parent), 
                           file_name="3D_hertzian_contact_quarter_cylinder_analytical", 
                           polar_transformation="cylindrical")