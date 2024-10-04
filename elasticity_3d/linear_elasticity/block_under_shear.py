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
Simple shear test for a 3D block, results seem logical.

BCs for a simple shear test: https://www.researchgate.net/publication/373713603_A_numerical-experimental_coupled_method_for_the_identification_of_model_parameters_from_-SPIF_test_using_a_finite_element_updating_method
'''

from utils.geometry.custom_geometry import GmshGeometry3D
from utils.geometry.gmsh_models import Block_3D_hex
from utils.elasticity import elasticity_utils
from utils.elasticity.elasticity_utils import pde_mixed_3d, get_tractions_mixed_3d
from utils.postprocess.elasticity_postprocessing import solutionFieldOnMeshToVtk3D
from utils.geometry.geometry_utils import calculate_boundary_normals

length = 1
height = 1
width = 1
seed_l = 10
seed_h = 10
seed_w = 10
origin = [0, 0, 0]

# The applied shear displacement
shear_disp = 0.05 # tau xy

Block_3D_obj = Block_3D_hex(origin=origin, 
                            length=length,
                            height=height,
                            width=width,
                            divisions=[seed_l, seed_h, seed_w])

gmsh_model = Block_3D_obj.generateGmshModel(visualize_mesh=True)

geom = GmshGeometry3D(gmsh_model)

elasticity_utils.geom = geom

def boundary_top(x, on_boundary):
    return on_boundary and np.isclose(x[1],height)

bc = dde.DirichletBC(geom, lambda _: shear_disp, boundary_top, component=0)

n_dummy = 1
data = dde.data.PDE(
    geom,
    pde_mixed_3d,
    [bc],
    num_domain=n_dummy,
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
    sigma_xy_surfaces = (right_surface)*(left_surface)
    sigma_yz_surfaces = (front_surface)*(back_surface)
    sigma_xz_surfaces = (right_surface)*(left_surface)*(front_surface)*(back_surface)
    
    return bkd.concat([u*(bottom_surface), #displacement in x direction is 0 at x=0
                      v*(bottom_surface)*(top_surface), #displacement in y direction is 0 at y=0 and y=3
                      w*(bottom_surface)*(top_surface), #displacement in z direction is 0 at z=0
                      sigma_xx*(right_surface)*(left_surface),
                      sigma_yy,
                      sigma_zz*(front_surface)*(back_surface),
                      sigma_xy*sigma_xy_surfaces,
                      sigma_yz*sigma_yz_surfaces,
                      sigma_xz*sigma_xz_surfaces
                      ], axis=1)

# 3 inputs, 9 outputs for 3D layer_size = [3] + [50] * 5 + [9]
layer_size = [3] + [50] * 5 + [9]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=1000, display_every=200)

model.compile("L-BFGS")
losshistory, train_state = model.train(display_every=200)

solutionFieldOnMeshToVtk3D(geom, 
                           model, 
                           save_folder_path=str(Path(__file__).parent.parent.parent), 
                           file_name="3D_block_shear")







