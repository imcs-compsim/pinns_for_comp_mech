import deepxde as dde
import numpy as np
from pathlib import Path
import gmsh

import numpy as np
from deepxde import backend as bkd

'''
@author: tsahin

Simple block under compression test for a 2D block for elastodynamics under linear load in time direction => p = -pressure*t for t [0,1])
In this example, z direction is considered as time direction. We generate the geometry using 3D mesh, but we consider the 3. dimension as time. 
Therefore, z direction must be the time direction. The followings must be taken into account:
- The z-direction must be the time direction. 
- The boundary normals must be consistent through the z-direction. The reason is that in Neumann BCs, we only take nx and ny terms assuming that nz is zero.   
'''

from utils.geometry.custom_geometry import GmshGeometry3D
from utils.geometry.gmsh_models import Block_3D_hex
from utils.elasticity import elasticity_utils
from utils.elasticity.elasticity_utils import pde_mixed_plane_strain_time_dependent, get_tractions_mixed_2d_time, problem_parameters
from utils.postprocess.elasticity_postprocessing import solutionFieldOnMeshToVtkSpaceTime
from utils.postprocess.save_normals_tangentials_to_vtk import export_normals_tangentials_to_vtk
from utils.geometry.geometry_utils import calculate_boundary_normals_3D

# This is for visualization
from utils.geometry.custom_geometry import GmshGeometry2D
from utils.geometry.gmsh_models import Block_2D

length = 1
height = 1
width = 1
seed_l = 10
seed_h = 10
seed_w = 10
origin = [0, 0, 0]

# The applied pressure 
pressure = 0.1
nu,lame,shear,e_modul = problem_parameters()

Block_3D_obj = Block_3D_hex(origin=origin, 
                            length=length,
                            height=height,
                            width=width,
                            divisions=[seed_l, seed_h, seed_w])

gmsh_model = Block_3D_obj.generateGmshModel(visualize_mesh=True)
geom = GmshGeometry3D(gmsh_model, target_surface_ids=[4])

# This allows for visualization of boundary normals in Paraview
export_normals_tangentials_to_vtk(geom, save_folder_path=str(Path(__file__).parent.parent.parent.parent), file_name="block_boundary_normals")

elasticity_utils.geom = geom
 
# Top surface
def boundary_top(x, on_boundary):
    return on_boundary and np.isclose(x[1],height)

# Front surface
def boundary_initial(x, on_boundary):
    time_dimension = x[2]
    return on_boundary and np.isclose(time_dimension,0)

# Neumann BC on top
def apply_pressure_y_top(x,y,X):
    Tx, Ty, Tn, Tt = get_tractions_mixed_2d_time(x, y, X)
    _, _, _, cond = calculate_boundary_normals_3D(X,geom)
    
    t_loc = x[:, 2:3][cond] # time only for boundary points
    
    return Ty + pressure*t_loc

# Initial BC for velocity component x
def apply_velocity_in_x(x,y,X):
    du_x_t = dde.grad.jacobian(y, x, i=0, j=2)# i=0 represents u_x, j=2 is time
    _, _, _, cond = calculate_boundary_normals_3D(X,geom)
    x_loc = x[:,0:1]

    return du_x_t[cond] - pressure/e_modul*nu*(1+nu)*x_loc[cond]

# Initial BC for velocity component y
def apply_velocity_in_y(x,y,X):
    du_y_t = dde.grad.jacobian(y, x, i=1, j=2) # i=1 represents u_y, j=2 is time
    _, _, _, cond = calculate_boundary_normals_3D(X,geom)
    y_loc = x[:,1:2]

    return du_y_t[cond] + pressure/e_modul*(1-nu**2)*y_loc[cond]

# Neumann BC
bc_pressure_y_top = dde.OperatorBC(geom, apply_pressure_y_top, boundary_top)
# Initial BCs for velocities
ic_velocity_in_x = dde.OperatorBC(geom, apply_velocity_in_x, boundary_initial)
ic_velocity_in_y = dde.OperatorBC(geom, apply_velocity_in_y, boundary_initial)    
# Initial BCs for displacements
# bc_u_x = dde.DirichletBC(geom, lambda _: 0, boundary_initial, component=0)
# bc_u_y = dde.DirichletBC(geom, lambda _: 0, boundary_initial, component=1)

bcs = [bc_pressure_y_top, ic_velocity_in_x, ic_velocity_in_y] 

n_dummy = 1
data = dde.data.PDE(
    geom,
    pde_mixed_plane_strain_time_dependent,
    bcs,
    num_domain=1,
    num_boundary=n_dummy,
    num_test=None,
    train_distribution = "Sobol"
)

def output_transform(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    sigma_xx =  y[:, 2:3]
    sigma_yy =  y[:, 3:4]
    sigma_xy =  y[:, 4:5]
    
    x_loc = x[:, 0:1] # coord x
    y_loc = x[:, 1:2] # coord y
    t_loc = x[:, 2:3] # time
    
    # define surfaces
    y_at_h = (height-y_loc)
    y_at_0 = (y_loc)
    x_at_l = (length-x_loc)
    x_at_0 = (x_loc)
    t_at_0 = (t_loc)
    t_at_w = (width-t_loc)
    
    # define the surfaces where shear forces will be applied.
    sigma_xy_surfaces = (y_at_h)*(y_at_0)*(x_at_l)*(x_at_0)
    
    return bkd.concat([u*(x_at_0)*t_at_0, # u_x is 0 at x=0 (Dirichlet BC) + u_x = 0 at t=0 (Initial BC) 
                      v*(y_at_0)*t_at_0, # u_y is 0 at y=0 (Dirichlet BC) + u_y = 0 at t=0 (Initial BC) 
                      sigma_xx*(x_at_l), 
                      sigma_yy, 
                      sigma_xy*sigma_xy_surfaces
                      ], axis=1)

# 3 inputs, 5 outputs for 3D 
layer_size = [3] + [50] * 5 + [5]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=1000, display_every=200)

model.compile("L-BFGS")
losshistory, train_state = model.train(display_every=200)

#########################################################################################################################################
#### POST-PROCESSING #####
#########################################################################################################################################
# 3D visualization
# def solutionFieldOnMeshToVtk3D(geom, 
#                                model, 
#                                file_name = "Test_time"):

    
#     X, offset, cell_types, elements = geom.get_mesh()

#     output = model.predict(X)

#     # .tolist() is applied to remove datatype
#     # .tolist() is applied to remove datatype
#     u_pred, v_pred = output[:,0].tolist(), output[:,1].tolist() # displacements
#     w_pred = np.zeros_like(output[:,0]).tolist()
#     sigma_xx_pred, sigma_yy_pred, sigma_xy_pred = output[:,2].tolist(), output[:,3].tolist(), output[:,4].tolist() # stresses


#     combined_disp_pred = tuple(np.vstack((u_pred, v_pred, w_pred)))
#     combined_stress_pred= tuple(np.vstack((sigma_xx_pred, sigma_yy_pred, sigma_xy_pred)))
    
#     x = X[:,0].flatten()
#     y = X[:,1].flatten()
#     z = X[:,2].flatten()

#     unstructuredGridToVTK(file_name, x, y, z, elements.flatten(), offset, 
#                         cell_types, pointData = { "pred_displacement" : combined_disp_pred,
#                                                     "pred_normal_stress" : combined_stress_pred}
#                                                     )
# solutionFieldOnMeshToVtk3D(geom, model) 
gmsh.clear()

gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
coord_left_corner=[0,0]
coord_right_corner=[1,1]
l_beam = coord_right_corner[0] - coord_left_corner[0]
h_beam = coord_right_corner[1] - coord_left_corner[1]

block_2d = Block_2D(coord_left_corner=coord_left_corner, coord_right_corner=coord_right_corner, mesh_size=0.05, gmsh_options=gmsh_options)

gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)

revert_curve_list = []
revert_normal_dir_list = [1,1,1,1]
geom = GmshGeometry2D(gmsh_model, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)

time_steps = 10
time_interval = [0, 1]

solutionFieldOnMeshToVtkSpaceTime(geom, 
                            model,
                            time_interval=time_interval,
                            time_steps=time_steps, 
                            save_folder_path=str(Path(__file__).parent.parent.parent.parent), 
                            file_name="2D_block_time_linear_load")








