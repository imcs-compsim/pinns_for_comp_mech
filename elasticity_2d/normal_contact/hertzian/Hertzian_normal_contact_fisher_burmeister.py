import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from pathlib import Path
from deepxde.backend import tf
import matplotlib.tri as tri
from pyevtk.hl import unstructuredGridToVTK
import time
# add utils folder to the system path
path_utils = str(Path(__file__).parent.parent.absolute()) + "/utils"
sys.path.append(path_utils)

from custom_geometry import GmshGeometry2D
from gmsh_models import QuarterDisc
from elasticity_utils import problem_parameters, pde_mixed_plane_strain, stress_to_traction_2d
from geometry_utils import calculate_boundary_normals, polar_transformation_2d
import elasticity_utils

#dde.config.set_default_float("float64")

'''
@author: tsahin
'''
#dde.config.real.set_float64()

gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
radius = 1
center = [0,0]

Quarter_Disc = QuarterDisc(radius=radius, center=center, mesh_size=0.005, angle=265, refine_times=1, gmsh_options=gmsh_options)

gmsh_model, x_loc_partition, y_loc_partition = Quarter_Disc.generateGmshModel(visualize_mesh=False)

revert_curve_list = []
revert_normal_dir_list = [1,2,2,1]
geom = GmshGeometry2D(gmsh_model, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)

# # change global variables in elasticity_utils, they are used for getting the material properties for analytical model
lame = 115.38461538461539
shear = 76.92307692307692
elasticity_utils.lame = lame
elasticity_utils.shear = shear
nu,lame,shear,e_modul = problem_parameters() # with dimensions, will be used for analytical solution
# This will lead to e_modul=200 and nu=0.3

# The applied pressure 
ext_traction = -0.5

# zero neumann BC functions need the geom variable to be 
elasticity_utils.geom = geom

distance = 0

def calculate_gap_in_normal_direction(x,y,X):
    '''
    Calculates the gap in normal direction
    '''
    # calculate the gap in y direction    
    gap_y = x[:,1:2] + y[:,1:2] + radius + distance

    # calculate the boundary normals
    normals, cond = calculate_boundary_normals(X,geom)

    # Here is the idea to calculate gap_n:
    # gap_n/|n| = gap_y/|ny| --> since n is unit vector |n|=1
    gap_n = tf.math.divide_no_nan(gap_y[cond],tf.math.abs(normals[:,1:2]))
    
    return gap_n

def calculate_traction(x, y, X):
    '''
    Calculates traction vectors in both cartesian and radial coordinates by using Cauchy stress tensor
    '''

    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5] 
    
    normals, cond = calculate_boundary_normals(X,geom)

    Tx, Ty, Tn, Tt = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Tx, Ty, Tn, Tt

# Karush-Kuhn-Tucker conditions for frictionless contact
# gn>=0 (positive_normal_gap), Pn<=0 (negative_normal_traction), Tt=0 (zero_tangential_traction) and gn.Pn=0 (zero_complimentary)

def zero_fisher_burmeister(x,y,X):
    '''
    Enforces KKT conditions using Fisher-Burmeister equation
    '''
    # ref https://www.math.uwaterloo.ca/~ltuncel/publications/corr2007-17.pdf
    Tx, Ty, Pn, Tt = calculate_traction(x, y, X)
    gn = calculate_gap_in_normal_direction(x, y, X)
    
    a = gn
    b = -Pn
    
    return a + b - tf.sqrt(tf.maximum(a**2+b**2, 1e-9))

def zero_tangential_traction(x,y,X):
    '''
    Enforces tangential part of contact traction (Tt) to be zero.
    '''
    Tx, Ty, Pn, Tt = calculate_traction(x, y, X)

    return Tt

def zero_neumann_x(x,y,X):
    '''
    Enforces x component of zero Neumann BC to be zero.
    '''
    Tx, Ty, Pn, Tt = calculate_traction(x, y, X)

    return Tx

def zero_neumann_y(x,y,X):
    '''
    Enforces y component of zero Neumann BC to be zero.
    '''
    Tx, Ty, Pn, Tt = calculate_traction(x, y, X)

    return Ty

def boundary_circle_not_contact(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center, axis=-1), radius) and (x[0]<x_loc_partition)

def boundary_circle_contact(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center, axis=-1), radius) and (x[0]>=x_loc_partition)

# Neumann BC
bc_zero_traction_x = dde.OperatorBC(geom, zero_neumann_x, boundary_circle_not_contact)
bc_zero_traction_y = dde.OperatorBC(geom, zero_neumann_y, boundary_circle_not_contact)

# Contact BC
bc_zero_fisher_burmeister = dde.OperatorBC(geom, zero_fisher_burmeister, boundary_circle_contact)
bc_zero_tangential_traction = dde.OperatorBC(geom, zero_tangential_traction, boundary_circle_contact)

n_dummy = 1
data = dde.data.PDE(
    geom,
    pde_mixed_plane_strain,
    [bc_zero_traction_x,bc_zero_traction_y,bc_zero_tangential_traction,bc_zero_fisher_burmeister],
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=n_dummy,
    train_distribution = "Sobol"
)

def output_transform(x, y):
    '''
    Hard BCs:
        Dirichlet terms
            u(x=0)=0
        
        Neumann terms:
            sigma_yy(y=0) = ext_traction
            sigma_xy(x=0) = 0
            sigma_xy(y=0) = 0
    
    General formulation to enforce BC hardly:
        N'(x) = g(x) + l(x)*N(x)
    
        where N'(x) is network output before transformation, N(x) is network output after transformation, g(x) Non-homogenous part of the BC and 
            if x is on the boundary
                l(x) = 0 
            else
                l(x) < 0
    
    For instance sigma_yy(y=0) = -ext_traction
        N'(x) = N(x) = sigma_yy
        g(x) = ext_traction
        l(x) = -y
    so
        u' = g(x) + l(x)*N(x)
        sigma_yy = ext_traction + -y*sigma_yy
    '''
    u = y[:, 0:1]
    v = y[:, 1:2]
    sigma_xx = y[:, 2:3]
    sigma_yy = y[:, 3:4]
    sigma_xy = y[:, 4:5]
    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    
    #return tf.concat([u*(-x_loc), ext_dips + v*(-y_loc), sigma_xx, sigma_yy, sigma_xy*(x_loc)*(y_loc)], axis=1)
    return tf.concat([u*(-x_loc)/e_modul, v/e_modul, sigma_xx, ext_traction + sigma_yy*(-y_loc),sigma_xy*(x_loc)*(y_loc)], axis=1)

# 2 inputs: x and y, 5 outputs: ux, uy, sigma_xx, sigma_yy and sigma_xy
layer_size = [2] + [50] * 5 + [5]
activation = "tanh"
# more inside https://cs230.stanford.edu/section/4/#xavier-initialization
# The goal of Xavier Initialization is to initialize the weights such that the variance of the activations are the same across every layer. 
# This constant variance helps prevent the gradient from exploding or vanishing.
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

# weights due to PDE
w_pde_1,w_pde_2,w_pde_3,w_pde_4,w_pde_5 = 1e0,1e0,1e0,1e0,1e0
# weights due to Neumann BC
w_zero_traction_x, w_zero_traction_y = 1e0,1e0
# weights due to Contact BC
w_zero_tangential_traction = 1e0
w_zero_fisher_burmeister = 1e4

loss_weights = [w_pde_1,w_pde_2,w_pde_3,w_pde_4,w_pde_5,w_zero_traction_x,w_zero_traction_y,w_zero_tangential_traction,w_zero_fisher_burmeister]

model = dde.Model(data, net)
model.compile("adam", lr=0.001, loss_weights=loss_weights)
losshistory, train_state = model.train(epochs=2000, display_every=100) 

model.compile("L-BFGS-B", loss_weights=loss_weights)
losshistory, train_state = model.train(display_every=200)

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

fem_path = str(Path(__file__).parent)+"/Hertzian_fem/Hertzian_fem_fine_mesh.csv"
df = pd.read_csv(fem_path)
fem_results = df[["Points_0","Points_1","displacement_0","displacement_1","nodal_cauchy_stresses_xyz_0","nodal_cauchy_stresses_xyz_1","nodal_cauchy_stresses_xyz_3"]]
fem_results = fem_results.to_numpy()
node_coords_xy = fem_results[:,0:2]
displacement_fem = fem_results[:,2:4]
stress_fem = fem_results[:,4:7]

X = node_coords_xy
x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)
triangles = tri.Triangulation(x, y)

# predictions
start_time_calc = time.time()
output = model.predict(X)
end_time_calc = time.time()
final_time = f'Prediction time: {(end_time_calc - start_time_calc):.3f} seconds'
print(final_time)

u_pred, v_pred = output[:,0], output[:,1]
sigma_xx_pred, sigma_yy_pred, sigma_xy_pred = output[:,2:3], output[:,3:4], output[:,4:5]
sigma_rr_pred, sigma_theta_pred, sigma_rtheta_pred = polar_transformation_2d(sigma_xx_pred, sigma_yy_pred, sigma_xy_pred, X)

combined_disp_pred = tuple(np.vstack((np.array(u_pred.tolist()),np.array(v_pred.tolist()),np.zeros(u_pred.shape[0]))))
combined_stress_pred = tuple(np.vstack((np.array(sigma_xx_pred.flatten().tolist()),np.array(sigma_yy_pred.flatten().tolist()),np.array(sigma_xy_pred.flatten().tolist()))))
combined_stress_polar_pred = tuple(np.vstack((np.array(sigma_rr_pred.tolist()),np.array(sigma_theta_pred.tolist()),np.array(sigma_rtheta_pred.tolist()))))

# fem
u_fem, v_fem = displacement_fem[:,0], displacement_fem[:,1]
sigma_xx_fem, sigma_yy_fem, sigma_xy_fem = stress_fem[:,0:1], stress_fem[:,1:2], stress_fem[:,2:3]
sigma_rr_fem, sigma_theta_fem, sigma_rtheta_fem = polar_transformation_2d(sigma_xx_fem, sigma_yy_fem, sigma_xy_fem, X)

combined_disp_fem = tuple(np.vstack((np.array(u_fem.tolist()),np.array(v_fem.tolist()),np.zeros(u_fem.shape[0]))))
combined_stress_fem = tuple(np.vstack((np.array(sigma_xx_fem.flatten().tolist()),np.array(sigma_yy_fem.flatten().tolist()),np.array(sigma_xy_fem.flatten().tolist()))))
combined_stress_polar_fem = tuple(np.vstack((np.array(sigma_rr_fem.tolist()),np.array(sigma_theta_fem.tolist()),np.array(sigma_rtheta_fem.tolist()))))

# error
error_disp_x = abs(np.array(u_pred.tolist()) - u_fem.flatten())
error_disp_y =  abs(np.array(v_pred.tolist()) - v_fem.flatten())
combined_error_disp = tuple(np.vstack((error_disp_x, error_disp_y,np.zeros(error_disp_x.shape[0]))))

error_stress_x = abs(np.array(sigma_xx_pred.flatten().tolist()) - sigma_xx_fem.flatten())
error_stress_y =  abs(np.array(sigma_yy_pred.flatten().tolist()) - sigma_yy_fem.flatten())
error_stress_xy =  abs(np.array(sigma_xy_pred.flatten().tolist()) - sigma_xy_fem.flatten())
combined_error_stress = tuple(np.vstack((error_stress_x, error_stress_y, error_stress_xy)))

error_polar_stress_x = abs(np.array(sigma_rr_pred.flatten().tolist()) - sigma_rr_fem.flatten())
error_polar_stress_y =  abs(np.array(sigma_theta_pred.flatten().tolist()) - sigma_theta_fem.flatten())
error_polar_stress_xy =  abs(np.array(sigma_rtheta_pred.flatten().tolist()) - sigma_rtheta_fem.flatten())
combined_error_polar_stress = tuple(np.vstack((error_polar_stress_x, error_polar_stress_y, error_polar_stress_xy)))

file_path = os.path.join(os.getcwd(), "Hertzian_normal_contact_fisher_burmeister")

dol_triangles = triangles.triangles
offset = np.arange(3,dol_triangles.shape[0]*dol_triangles.shape[1]+1,dol_triangles.shape[1]).astype(dol_triangles.dtype)
cell_types = np.ones(dol_triangles.shape[0])*5

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                      cell_types, pointData = {"displacement_pred" : combined_disp_pred,
                                               "displacement_fem" : combined_disp_fem,
                                               "stress_pred" : combined_stress_pred, 
                                               "stress_fem" : combined_stress_fem, 
                                               "polar_stress_pred" : combined_stress_polar_pred, 
                                               "polar_stress_fem" : combined_stress_polar_fem, 
                                               "error_disp" : combined_error_disp, 
                                               "error_stress" : combined_error_stress, 
                                               "error_polar_stress" : combined_error_polar_stress
                                            })