import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
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

from compsim_pinns.geometry.custom_geometry import GmshGeometry3D
from compsim_pinns.geometry.gmsh_models import Block_3D_hex
from compsim_pinns.elasticity import elasticity_utils
from compsim_pinns.elasticity.elasticity_utils import pde_mixed_3d
from compsim_pinns.postprocess.elasticity_postprocessing import solutionFieldOnMeshToVtk3D
from compsim_pinns.geometry.geometry_utils import calculate_boundary_normals_3D

from compsim_pinns.contact_mech import contact_utils
from compsim_pinns.contact_mech.contact_utils import zero_tangential_traction_component1_3d, zero_tangential_traction_component2_3d, zero_complementarity_function_based_fischer_burmeister_3d

length = 1
height = 1
width = 1
seed_l = 10
seed_h = 10
seed_w = 10
origin = [0, 0, 0]

# The applied pressure 
pressure = -0.1

Block_3D_obj = Block_3D_hex(origin=origin, 
                            length=length,
                            height=height,
                            width=width,
                            divisions=[seed_l, seed_h, seed_w])

gmsh_model = Block_3D_obj.generateGmshModel(visualize_mesh=False)

geom = GmshGeometry3D(gmsh_model)

projection_plane = {"y" : 0} # y=10, plane formula
# assign local parameters from the current file in contact_utils and elasticity_utils
elasticity_utils.geom = geom
    
contact_utils.geom = geom
contact_utils.projection_plane = projection_plane

# define contact boundary points
def boundary_contact(x, on_boundary):
    bottom_points = np.isclose(x[1],0)
    z_edges = np.isclose(x[2],0) or np.isclose(x[2],width)
    x_edges = np.isclose(x[0],0) or np.isclose(x[0],length)
    target_points = bottom_points and (not z_edges) and (not x_edges) 
    return on_boundary and target_points

# enforce tangential tractions to be zero
bc1 = dde.OperatorBC(geom, zero_tangential_traction_component1_3d, boundary_contact)
bc2 = dde.OperatorBC(geom, zero_tangential_traction_component2_3d, boundary_contact)

# KKT using fischer_burmeister
bc3 = dde.OperatorBC(geom, zero_complementarity_function_based_fischer_burmeister_3d, boundary_contact)

n_dummy = 1
data = dde.data.PDE(
    geom,
    pde_mixed_3d,
    [bc1, bc2, bc3],
    num_domain=300,
    num_boundary=n_dummy,
    num_test=None,
    train_distribution = "Sobol"
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
    y_at_h = (height-y_loc)
    y_at_0 = (y_loc)
    x_at_l = (length-x_loc)
    x_at_0 = (x_loc)
    z_at_0 = (z_loc)
    z_at_w = (width-z_loc)
    
    # define the surfaces where shear forces will be applied.
    sigma_xy_surfaces = (y_at_h)*(x_at_l)*(x_at_0)
    sigma_yz_surfaces = (y_at_h)*(z_at_w)*(z_at_0)
    sigma_xz_surfaces = (x_at_l)*(z_at_w)*(z_at_0)*(x_at_0)
    
    return bkd.concat([u*(x_at_0), #displacement in x direction is 0 at x=0
                      v,
                      w*(z_at_0), #displacement in z direction is 0 at z=0
                      sigma_xx*(x_at_l), 
                      pressure + sigma_yy*(y_at_h),
                      sigma_zz*(z_at_w),
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
loss_weights=[1,1,1,1,1,1,1,1] #,1,1

def boundary_contact_callback(x):
    # Check if points are close to y=0 (bottom points)
    bottom_points = np.isclose(x[:, 1], 0)
    
    # Check if points are on the z=0 or z=width edges
    z_edges = np.logical_or(np.isclose(x[:, 2], 0), np.isclose(x[:, 2], width))
    
    # Check if points are on the x=0 or x=length edges
    x_edges = np.logical_or(np.isclose(x[:, 0], 0), np.isclose(x[:, 0], length))
    
    # Select points that are bottom points but not on z or x edges
    target_points = np.logical_and(bottom_points, np.logical_not(np.logical_or(z_edges, x_edges)))
    
    return target_points

callback_pts = geom.random_boundary_points(1)
callback_pts = callback_pts[boundary_contact_callback(callback_pts)]

def calculate_gap_in_normal_direction_3d_internal(x, y):
    # Calculate the boundary normals in 3D
    normals, tangentials_1, tangentials_2, cond = calculate_boundary_normals_3D(callback_pts, geom)
    # normals 
    nx = normals[:,0:1]
    ny = normals[:,1:2]
    nz = normals[:,2:3]
    
    # Calculate the gap in all three directions: x, y, and z
    x_x = x[:, 0:1] + y[:, 0:1] # X_x + u_x
    x_y = x[:, 1:2] + y[:, 1:2] # X_y + u_y
    x_z = x[:, 2:3] + y[:, 2:3] # X_z + u_z
    
    #projection planes
    if projection_plane.get("x") is not None:
        proj_x = projection_plane.get("x")
        proj_y = x[:, 1:2]
        proj_z = x[:, 2:3]
    elif projection_plane.get("y") is not None:
        proj_x = x[:, 0:1]
        proj_y = projection_plane.get("y")
        proj_z = x[:, 2:3]
    elif projection_plane.get("z") is not None:
        proj_x = x[:, 0:1]
        proj_y = x[:, 1:2]
        proj_z = projection_plane.get("z")
    
    # gaps in x, y, and z direction
    gap_x = x_x - proj_x
    gap_y = x_y - proj_y
    gap_z = x_z - proj_z

    # Calculate the gap in the normal direction
    gap_n = -(gap_x[cond]*nx + gap_y[cond]*ny + gap_z[cond]*nz)
    # gap_n = -(gap_y[cond]*ny)

    return gap_n

def get_tractions_mixed_3d_internal(x, y):  
    sigma_xx =  y[:, 3:4]
    sigma_yy =  y[:, 4:5]
    sigma_zz =  y[:, 5:6]
    sigma_xy =  y[:, 6:7]
    sigma_yz =  y[:, 7:8]
    sigma_xz =  y[:, 8:9]
    
    normals, tangentials_1, tangentials_2, cond = calculate_boundary_normals_3D(callback_pts,geom)
    # normals 
    nx = normals[:,0:1]
    ny = normals[:,1:2]
    nz = normals[:,2:3]

    # Calculate the traction components in Cartesian coordinates
    Tx = sigma_xx*nx + sigma_xy*ny + sigma_xz*nz
    Ty = sigma_xy*nx + sigma_yy*ny + sigma_yz*nz
    Tz = sigma_xz*nx + sigma_yz*ny + sigma_zz*nz
    
    Tn = Tx*nx + Ty*ny + Tz*nz

    return Tn

gap_value = dde.callbacks.OperatorPredictor(
    callback_pts, op=calculate_gap_in_normal_direction_3d_internal, period=1, filename="gap.txt"
)
pn_value = dde.callbacks.OperatorPredictor(
    callback_pts, op=get_tractions_mixed_3d_internal, period=1, filename="pn.txt"
)

model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=1000, display_every=200, callbacks=[gap_value,pn_value])

model.compile("L-BFGS")
losshistory, train_state = model.train(display_every=200, callbacks=[gap_value,pn_value])

solutionFieldOnMeshToVtk3D(geom, 
                           model, 
                           save_folder_path=str(Path(__file__).parent.parent.parent.parent), 
                           file_name="3D_block_compression_contact")







