import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from deepxde.backend import tf
from pyevtk.hl import unstructuredGridToVTK
# add utils folder to the system path
path_utils = str(Path(__file__).parent.parent.absolute()) + "/utils"
sys.path.append(path_utils)

from elasticity_utils import stress_plane_stress, momentum_2d_plane_stress, problem_parameters
from gmsh_models import Block_2D
import elasticity_utils

from custom_geometry import GmshGeometry2D
from elasticity_utils import problem_parameters, pde_mixed_plane_stress, stress_to_traction_2d
from geometry_utils import calculate_boundary_normals
import elasticity_utils

'''
Single patch-test for testing contact conditions. It is a simple block under compression. Check problem_figures/Contact_patch.png for details.

@author: tsahin
'''


gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
coord_left_corner=[-0,-0.]
coord_right_corner=[1,1]
l_beam = coord_right_corner[0] - coord_left_corner[0]
h_beam = coord_right_corner[1] - coord_left_corner[1]

block_2d = Block_2D(coord_left_corner=coord_left_corner, coord_right_corner=coord_right_corner, mesh_size=0.05, gmsh_options=gmsh_options)

gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)

revert_curve_list = []
revert_normal_dir_list = [1,1,1,1]
geom = GmshGeometry2D(gmsh_model, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)

# The applied pressure 
ext_traction = -0.1

# how far above the block from ground 
distance = 0

# complementarity parameter c
c_complementarity=0.001

def calculate_gap_in_normal_direction(x,y,X):
    '''
    Calculates the gap in normal direction
    '''
    # calculate the gap in y direction    
    gap_y = x[:,1:2] + y[:,1:2] + distance

    # calculate the boundary normals
    normals, cond = calculate_boundary_normals(X,geom)

    # Here is the idea to calculate gap_n:
    # gap_n/|n| = gap_y/|ny| --> since n is unit vector |n|=1
    gap_n = tf.math.divide_no_nan(gap_y[cond],tf.math.abs(normals[:,1:2]))
    
    return gap_n

def calculate_traction(x, y, X):
    '''
    Calculates x component of any traction vector using by Cauchy stress tensor
    '''

    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5] 
    
    normals, cond = calculate_boundary_normals(X,geom)

    Tx, Ty, Tn, Tt = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Tx, Ty, Tn, Tt

# Karush-Kuhn-Tucker conditions for frictionless contact
# gn>=0 (positive_normal_gap), Pn<=0 (negative_normal_traction), Tt=0 (zero_tangential_traction) and gn.Pn=0 (zero_complimentary)

def positive_normal_gap(x, y, X):
    '''
    Enforces normal gap (gn) to be positive.
    '''
    gn = calculate_gap_in_normal_direction(x, y, X)

    # If gn is negative, it will create contributions to overall loss. Aims is to get positive gap
    return (1.0-tf.math.sign(gn))*gn

def negative_normal_traction(x,y,X):
    '''
    Enforces normal part of contact traction (Pn) to be negative.
    '''
    Tx, Ty, Pn, Tt = calculate_traction(x, y, X)

    # If Pn is positive, it will create contributions to overall loss. Aims is to get negative normal traction
    return (1.0+tf.math.sign(Pn))*Pn

def zero_tangential_traction(x,y,X):
    '''
    Enforces tangential part of contact traction (Tt) to be zero.
    '''
    Tx, Ty, Pn, Tt = calculate_traction(x, y, X)

    return Tt

def zero_complimentary(x,y,X):
    '''
    Enforces complimentary term to be zero.
    '''
    Tx, Ty, Pn, Tt = calculate_traction(x, y, X)
    gn = calculate_gap_in_normal_direction(x, y, X)

    return gn*Pn

def zero_complementarity_function(x,y,X):
    # ref https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.2614
    Tx, Ty, Pn, Tt = calculate_traction(x, y, X)
    gn = calculate_gap_in_normal_direction(x, y, X)

    return Pn-tf.math.maximum(tf.constant(0, dtype=tf.float32), Pn-c_complementarity*gn)

def zero_fisher_burmeister(x,y,X):
    # ref https://www.math.uwaterloo.ca/~ltuncel/publications/corr2007-17.pdf
    Tx, Ty, Pn, Tt = calculate_traction(x, y, X)
    gn = calculate_gap_in_normal_direction(x, y, X)
    
    a = gn
    b = -Pn
    
    return a + b - tf.sqrt(tf.maximum(a**2+b**2, 1e-9))

def boundary_contact(x, on_boundary):
    return on_boundary and np.isclose(x[1],0)

method_list = ["KKT_inequality", "complementarity_popp", "fisher_burmeister"]
method_name = "KKT_inequality"
  
# Contact BCs
# we always enforce tangential traction
bc_zero_tangential_traction = dde.OperatorBC(geom, zero_tangential_traction, boundary_contact)

if method_name == "KKT_inequality":
    bc_positive_normal_gap = dde.OperatorBC(geom, positive_normal_gap, boundary_contact)
    bc_negative_normal_traction = dde.OperatorBC(geom, negative_normal_traction, boundary_contact)
    bc_zero_complimentary = dde.OperatorBC(geom, zero_complimentary, boundary_contact)
    bcs_ = [bc_positive_normal_gap,bc_negative_normal_traction,bc_zero_tangential_traction,bc_zero_complimentary]
    output_file_name = f"Patch_KKT_inequality"
elif method_name == "complementarity_popp":
    bc_zero_complementarity = dde.OperatorBC(geom, zero_complementarity_function, boundary_contact)
    bcs_ = [bc_zero_complementarity,bc_zero_tangential_traction]
    output_file_name = f"Patch_complementarity_function_c_{c_complementarity}"
elif method_name == "fisher_burmeister":
    bc_zero_complementarity = dde.OperatorBC(geom, zero_fisher_burmeister, boundary_contact)
    bcs_ = [bc_zero_complementarity,bc_zero_tangential_traction]
    output_file_name = "Patch_fisher_burmeister"
else:
    raise Exception("Method name does not exists!")

n_dummy = 1
data = dde.data.PDE(
    geom,
    pde_mixed_plane_stress,
    bcs_,# bc_gn, bc_Pn, bc_complimentary, bc_Tt
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=n_dummy,
    train_distribution = "Sobol"
)

def output_transform(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    sigma_xx = y[:, 2:3]
    sigma_yy = y[:, 3:4]
    sigma_xy = y[:, 4:5]
    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    
    #return tf.concat([u*(x_loc+l_beam/2),v*(y_loc+h_beam/2), pressure + sigma_xx*(l_beam/2-x_loc), sigma_yy*(h_beam/2-y_loc),sigma_xy*(l_beam/2-x_loc)*(x_loc+l_beam/2)*(h_beam/2-y_loc)*(y_loc+h_beam/2)], axis=1)
    return tf.concat([u*(x_loc),v, sigma_xx*(l_beam-x_loc), ext_traction + sigma_yy*(h_beam-y_loc),sigma_xy*(l_beam-x_loc)*(x_loc)*(h_beam-y_loc)], axis=1)

# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 5 + [5]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=1000, display_every=100)

model.compile("L-BFGS")
losshistory, train_state = model.train(display_every=200)

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################


X, offset, cell_types, dol_triangles = geom.get_mesh()
nu,lame,shear,e_modul = problem_parameters()

output = model.predict(X)
u_pred, v_pred = output[:,0], output[:,1]
sigma_xx, sigma_yy, sigma_xy = output[:,2:3], output[:,3:4], output[:,4:5]

theta = np.degrees(np.arctan2(X[:,1],X[:,0])).reshape(-1,1) # in degree
theta_radian = (theta-90)*np.pi/180
theta_radian = theta_radian
sigma_rr = ext_traction/2 + ext_traction/2*np.cos(2*theta_radian)
sigma_theta = ext_traction/2 - ext_traction/2*np.cos(2*theta_radian)
sigma_rtheta =  -ext_traction/2*np.sin(2*theta_radian)
sigma_combined_radial = np.hstack((sigma_rr,sigma_theta,sigma_rtheta))

k = (3-nu)/(1+nu)
r = np.sqrt(X[:,0:1]**2+X[:,1:2]**2)
u_r = ext_traction/(4*shear)*r*((k-1)/2+np.cos(2*theta_radian))
u_theta = -ext_traction/(4*shear)*r*np.sin(2*theta_radian)
u_x = u_r*np.cos(theta_radian) - u_theta*np.sin(theta_radian)
u_y = u_r*np.sin(theta_radian) + u_theta*np.cos(theta_radian)
u_x_temp = u_x
u_x = u_y
u_y = u_x_temp
u_x = u_x*-1
u_combined = np.hstack((u_x,u_y))


theta_radian = theta_radian.flatten()
A = [[np.cos(theta_radian)**2, np.sin(theta_radian)**2, 2*np.sin(theta_radian)*np.cos(theta_radian)],[np.sin(theta_radian)**2, np.cos(theta_radian)**2, -2*np.sin(theta_radian)*np.cos(theta_radian)],[-np.sin(theta_radian)*np.cos(theta_radian), np.sin(theta_radian)*np.cos(theta_radian), np.cos(theta_radian)**2-np.sin(theta_radian)**2]]
A = np.array(A)

sigma_analytical = np.zeros((len(sigma_rr),3))

for i in range(len(sigma_rr)):
    sigma_analytical[i:i+1,:] = np.matmul(np.linalg.inv(A[:,:,i]),sigma_combined_radial.T[:,i:i+1]).T
sigma_analytical_temp = sigma_analytical.copy()
sigma_analytical[:,0] = sigma_analytical_temp[:,1]
sigma_analytical[:,1] = sigma_analytical_temp[:,0]

error_x = abs(np.array(output[:,0].tolist()) - u_x.flatten())
error_y =  abs(np.array(output[:,1].tolist()) - u_y.flatten())
combined_error_disp = tuple(np.vstack((error_x, error_y,np.zeros(error_x.shape[0]))))

error_x = abs(np.array(output[:,2].tolist()) - sigma_analytical[:,0].flatten())
error_y =  abs(np.array(output[:,3].tolist()) - sigma_analytical[:,1].flatten())
error_z =  abs(np.array(output[:,4].tolist()) - sigma_analytical[:,2].flatten())
combined_error_stress = tuple(np.vstack((error_x, error_y,error_z)))

combined_disp = tuple(np.vstack((np.array(u_pred.tolist()),np.array(v_pred.tolist()),np.zeros(u_pred.shape[0]))))
combined_stress = tuple(np.vstack((np.array(sigma_xx.flatten().tolist()),np.array(sigma_yy.flatten().tolist()),np.array(sigma_xy.flatten().tolist()))))
combined_stress_analytical = tuple(np.vstack((np.array(sigma_analytical[:,0].flatten().tolist()),np.array(sigma_analytical[:,1].flatten().tolist()),np.array(sigma_analytical[:,2].flatten().tolist()))))
combined_disp_analytical = tuple(np.vstack((np.array(u_combined[:,0].flatten().tolist()),np.array(u_combined[:,1].flatten().tolist()),np.zeros(u_combined[:,1].shape[0]))))

file_path = os.path.join(os.getcwd(), output_file_name)

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)

#np.savetxt("Lame_inverse_large", X=np.hstack((X,output[:,0:2])))

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                      cell_types, pointData = { "displacement" : combined_disp,"stress" : combined_stress, "analy_stress" : combined_stress_analytical, "analy_disp" : combined_disp_analytical
                                               ,"error_disp":combined_error_disp, "error_stress":combined_error_stress})