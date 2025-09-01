import deepxde as dde
import numpy as np
import pandas as pd
import os
from pathlib import Path
from deepxde.backend import torch
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import seaborn as sns
from pyevtk.hl import unstructuredGridToVTK
import time

from utils.geometry.custom_geometry import GmshGeometry2D
from utils.geometry.gmsh_models import QuarterDisc
from utils.geometry.geometry_utils import polar_transformation_2d, calculate_boundary_normals

from utils.elasticity.elasticity_utils import problem_parameters, pde_mixed_plane_strain, calculate_traction_mixed_formulation
from utils.elasticity.elasticity_utils import zero_neumann_x_mixed_formulation, zero_neumann_y_mixed_formulation
from utils.contact_mech.contact_utils import zero_tangential_traction
from utils.elasticity import elasticity_utils
from utils.contact_mech import contact_utils

dde.config.set_default_float("float64")

'''
Solves Hertzian normal contact example inluding simulation data from 4C.

@author: tsahin
'''

gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
radius = 1
center = [0,0]

Quarter_Disc = QuarterDisc(radius=radius, center=center, mesh_size=0.008, angle=265, refine_times=100, gmsh_options=gmsh_options)

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
contact_utils.geom = geom

# how far above the block from ground
distance = 0

# assign local parameters from the current file in contact_utils and elasticity_utils
contact_utils.distance = distance

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
    # gap_n = tf.math.divide_no_nan(gap_y[cond],tf.math.abs(normals[:,1:2]))
    num = gap_y[cond]
    den = torch.abs(normals[:, 1:2])
    out = torch.zeros_like(num)
    mask = den != 0
    out[mask] = num[mask] / den[mask]
    gap_n = out

    return gap_n

def zero_fischer_burmeister(x,y,X):
    '''
    Enforces KKT conditions using Fischer-Burmeister equation
    '''
    # ref https://www.math.uwaterloo.ca/~ltuncel/publications/corr2007-17.pdf
    Tx, Ty, Pn, Tt = calculate_traction_mixed_formulation(x, y, X)
    gn = calculate_gap_in_normal_direction(x, y, X)
    
    a = gn
    b = -Pn
    
    return a + b - torch.sqrt(torch.maximum(a**2+b**2, torch.tensor(1e-9, dtype=a.dtype, device=a.device)))

def boundary_circle_not_contact(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center, axis=-1), radius) and (x[0]<x_loc_partition)

def boundary_circle_contact(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center, axis=-1), radius) and (x[0]>=x_loc_partition)

# Neumann BC
bc_zero_traction_x = dde.OperatorBC(geom, zero_neumann_x_mixed_formulation, boundary_circle_not_contact)
bc_zero_traction_y = dde.OperatorBC(geom, zero_neumann_y_mixed_formulation, boundary_circle_not_contact)

# Contact BC
bc_zero_fischer_burmeister = dde.OperatorBC(geom, zero_fischer_burmeister, boundary_circle_contact)
bc_zero_tangential_traction = dde.OperatorBC(geom, zero_tangential_traction, boundary_circle_contact)

bcs = [bc_zero_traction_x,bc_zero_traction_y,bc_zero_tangential_traction,bc_zero_fischer_burmeister]

add_external_data = True

if add_external_data:
    # load external data
    fem_path = str(Path(__file__).parent.parent.parent)+"/Hertzian_fem/Hertzian_fem_fine_mesh.csv"
    df = pd.read_csv(fem_path)
    fem_results = df[["Points_0","Points_1","displacement_0","displacement_1","nodal_cauchy_stresses_xyz_0","nodal_cauchy_stresses_xyz_1","nodal_cauchy_stresses_xyz_3"]]
    fem_results = fem_results.to_numpy()

    # shuffle fem_results so that we do not slice a specific part of mesh
    np.random.seed(12) # We will always use the same points #reproducibility
    np.random.shuffle(fem_results)

    # coordinates, diplacements and stresses in fem 
    node_coords_xy = fem_results[:,0:2]
    displacement_fem = fem_results[:,2:4]
    stress_fem = fem_results[:,4:7]

    # define condition to find boundary points 
    on_radius = np.isclose(np.linalg.norm(node_coords_xy - center, axis=-1), radius)
    on_right = np.isclose(node_coords_xy[:,0], center[0])
    on_top = np.isclose(node_coords_xy[:,1], center[1])
    on_boundary_ = np.logical_or(np.logical_or(on_radius,on_right),on_top)
    
    # we will take only 100 points from boundary and 100 points from domain
    n_boundary = 100
    n_domain = 100

    # 
    ex_data_xy = np.vstack((node_coords_xy[on_boundary_][:n_boundary],node_coords_xy[~on_boundary_][:n_domain]))
    ex_data_disp = np.vstack((displacement_fem[on_boundary_][:n_boundary],displacement_fem[~on_boundary_][:n_domain]))
    ex_data_stress = np.vstack((stress_fem[on_boundary_][:n_boundary],stress_fem[~on_boundary_][:n_domain]))

    # visualize points 
    # sns.set_theme()
    # fig, ax = plt.subplots(figsize=(10,8))
    
    # ax.scatter(node_coords_xy[on_boundary_][:n_boundary,0],node_coords_xy[on_boundary_][:n_boundary,1], label="boundary pts.")
    # ax.scatter(node_coords_xy[~on_boundary_][:n_domain,0],node_coords_xy[~on_boundary_][:n_domain,1], label="collocation pts.")
    # ax.set_xlabel(r"$x$", fontsize=24)
    # ax.set_ylabel(r"$y$", fontsize=24)
    # ax.tick_params(axis='both', which='major', labelsize=18)
    
    # ax.legend(fontsize=20)
    # plt.savefig("Hertzian_data_dist.png",dpi=200)
    # plt.show()

    # define boundary conditions for experimental data
    observe_u = dde.PointSetBC(ex_data_xy, ex_data_disp[:,0:1], component=0)
    observe_v = dde.PointSetBC(ex_data_xy, ex_data_disp[:,1:2], component=1)
    observe_sigma_xx = dde.PointSetBC(ex_data_xy, ex_data_stress[:,0:1], component=2)
    observe_sigma_yy = dde.PointSetBC(ex_data_xy, ex_data_stress[:,1:2], component=3)
    observe_sigma_xy = dde.PointSetBC(ex_data_xy, ex_data_stress[:,2:3], component=4)
    
    bcs_data = [observe_u, observe_v, observe_sigma_xx, observe_sigma_yy, observe_sigma_xy]
    
    bcs.extend(bcs_data)


n_dummy = 1
data = dde.data.PDE(
    geom,
    pde_mixed_plane_strain,
    bcs,
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=None,
    train_distribution = "Sobol",
    anchors=(ex_data_xy if add_external_data else None)
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
    return torch.cat([u*(-x_loc)/e_modul, v/e_modul, sigma_xx, ext_traction + sigma_yy*(-y_loc),sigma_xy*(x_loc)*(y_loc)], axis=1)

# 2 inputs: x and y, 5 outputs: ux, uy, sigma_xx, sigma_yy and sigma_xy
layer_size = [2] + [50] * 5 + [5]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

# weights due to PDE
w_pde_1,w_pde_2,w_pde_3,w_pde_4,w_pde_5 = 1e0,1e0,1e0,1e0,1e0
# weights due to Neumann BC
w_zero_traction_x, w_zero_traction_y = 1e0,1e0
# weights due to Contact BC
w_zero_tangential_traction = 1e0
w_zero_fischer_burmeister = 1e4
# weights due to external data
w_ext_u, w_ext_v, w_ext_sigma_xx, w_ext_sigma_yy, w_ext_sigma_xy = 1e4,1e4,1e-1,1e-1,1e-1

loss_weights = [w_pde_1,w_pde_2,w_pde_3,w_pde_4,w_pde_5,w_zero_traction_x,w_zero_traction_y,w_zero_tangential_traction,w_zero_fischer_burmeister]

if add_external_data:
    loss_weights_data = [w_ext_u, w_ext_v, w_ext_sigma_xx, w_ext_sigma_yy, w_ext_sigma_xy]
    loss_weights.extend(loss_weights_data)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, loss_weights=loss_weights)
losshistory, train_state = model.train(iterations=3000, display_every=100) 

model.compile("L-BFGS", loss_weights=loss_weights)
losshistory, train_state = model.train(display_every=1000)

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

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

file_path = os.path.join(os.getcwd(), "Hertzian_normal_contact_fischer_burmeister_data")

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

## Plot the normal traction on contact domain, analytical vs predicted
nu,lame,shear,e_modul = problem_parameters()
X, _, _, _ = geom.get_mesh()

output = model.predict(X)
sigma_xx_pred, sigma_yy_pred, sigma_xy_pred = output[:,2:3], output[:,3:4], output[:,4:5]
sigma_rr_pred, sigma_theta_pred, sigma_rtheta_pred = polar_transformation_2d(sigma_xx_pred, sigma_yy_pred, sigma_xy_pred, X)

x_contact_lim = 2*np.sqrt(2*radius**2*abs(ext_traction)*(1-nu**2)/(e_modul*np.pi))
x_contact_cond = np.logical_and(np.isclose(np.linalg.norm(X - center, axis=-1), radius), X[:,0]>-x_contact_lim)
node_coords_x_contact = X[x_contact_cond][:,0]
node_coords_x_contact = abs(node_coords_x_contact)

pc_analytical = 4*radius*abs(ext_traction)/(np.pi*x_contact_lim**2)*np.sqrt(x_contact_lim**2-node_coords_x_contact**2)
pc_predicted = -sigma_rr_pred[x_contact_cond]

sns.set_theme()
fig, ax = plt.subplots(figsize=(10,8))

ax.scatter(node_coords_x_contact,pc_analytical,label="true")
ax.scatter(node_coords_x_contact,pc_predicted,label="prediction")

ax.legend(fontsize=20)
ax.set_xlabel("|x|", fontsize=24)
ax.set_ylabel(r"$P_n$", fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.tight_layout()

plt.savefig("pressure_analy_pred.png", dpi=300)