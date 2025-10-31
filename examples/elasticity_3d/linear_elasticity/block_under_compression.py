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

from compsim_pinns.geometry.custom_geometry import GmshGeometry3D
from compsim_pinns.geometry.gmsh_models import Block_3D_hex
from compsim_pinns.elasticity import elasticity_utils
from compsim_pinns.elasticity.elasticity_utils import pde_mixed_3d, get_tractions_mixed_3d
from compsim_pinns.postprocess.elasticity_postprocessing import solutionFieldOnMeshToVtk3D
from compsim_pinns.geometry.geometry_utils import calculate_boundary_normals_3D

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

elasticity_utils.geom = geom

# This is for testing the Neumann BCs as soft constraints
soft_constraint = False 
bcs = []
if soft_constraint:     
    # Top surface
    def boundary_top(x, on_boundary):
        return on_boundary and np.isclose(x[1],height)
    # Neumann BC on top
    def apply_pressure_y_top(x,y,X):
        Tx, Ty, Tz, Tn, Tt_1, Tt_2 = get_tractions_mixed_3d(x, y, X)
        
        normals, tangentials_1, tangentials_2, cond= calculate_boundary_normals_3D(X,geom)

        return Ty - pressure*normals[:,1:2]


    bc_pressure_y_top = dde.OperatorBC(geom, apply_pressure_y_top, boundary_top)
    
    bcs = [bc_pressure_y_top]

n_dummy = 1
data = dde.data.PDE(
    geom,
    pde_mixed_3d,
    bcs,
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
    top_surface = (height-y_loc)
    bottom_surface = (y_loc)
    right_surface = (length-x_loc)
    left_surface = (x_loc)
    front_surface = (z_loc)
    back_surface = (width-z_loc)
    
    # define the surfaces where shear forces will be applied.
    sigma_xy_surfaces = (top_surface)*(right_surface)
    sigma_yz_surfaces = (top_surface)*(back_surface)
    sigma_xz_surfaces = (right_surface)*(back_surface)
    
    #
    if soft_constraint:
        sigma_yy_part = sigma_yy
    else:
        sigma_yy_part = pressure + sigma_yy*(top_surface) #sigma_yy
    
    return bkd.concat([u*(left_surface), #displacement in x direction is 0 at x=0
                      v*(bottom_surface), #displacement in y direction is 0 at y=0
                      w*(front_surface), #displacement in z direction is 0 at z=0
                      sigma_xx*(right_surface), 
                      sigma_yy_part, #pressure + sigma_yy*(top_surface), #sigma_yy
                      sigma_zz*(back_surface),
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
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(iterations=1000, display_every=200)

model.compile("L-BFGS")
losshistory, train_state = model.train(display_every=1000)

solutionFieldOnMeshToVtk3D(geom, 
                           model, 
                           save_folder_path=str(Path(__file__).parent.parent.parent), 
                           file_name="3D_block_compression")








