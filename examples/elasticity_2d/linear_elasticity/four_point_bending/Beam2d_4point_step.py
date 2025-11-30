'''
@author: tsahin
'''
import os
os.environ["DDE_BACKEND"] = "tensorflow.compat.v1"

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from deepxde.backend import tf
import tensorflow as tf_original
from pyevtk.hl import unstructuredGridToVTK

from compsim_pinns.elasticity.elasticity_utils import stress_plane_stress, elastic_strain_2d
from compsim_pinns.geometry.geometry_utils import calculate_boundary_normals
from compsim_pinns.geometry.custom_geometry import GmshGeometry2D
from compsim_pinns.geometry.gmsh_models import Rectangle_4PointBendingCentered
from compsim_pinns.elasticity import elasticity_utils

'''
In this routine, 4 point bending test example is illustrated.

            P1                P2
***********----**************----*****************
*                                                *
*                                                *
*                                                *
***----*********************************----******
    P3                                   P4

pressure vectors are applied on P1, P2, P3 and P4. Also points on the middle are constrained in the horizontal direction using hard constraints

author: @tsahin 
'''
# parameters of the geometry
coord_left_corner = [-1.5, -0.15, 0]
coord_right_corner = [1.5, 0.15, 0]
x0 = coord_left_corner[0]
y0 = coord_left_corner[1]
x1 = coord_right_corner[0]
y1 = coord_right_corner[1]

# length and height
l = x1 - x0
h = y1 - y0

# gmsh options
gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
load_center = 0.4
load_deviation = 0.1/2
disp_center = 1.25
disp_deviation = 0.1/2
# the center points and the size of 4 points are given
region_size_dict={"r1":{"center":-disp_center, "deviation":disp_deviation}, "r2":{"center":disp_center, "deviation":disp_deviation}, "r3":{"center":-load_center, "deviation":load_deviation}, "r4":{"center":load_center, "deviation":load_deviation}}

# generate the block, gmsh model and geometry for further calculations
block_2d = Rectangle_4PointBendingCentered(coord_left_corner=coord_left_corner, coord_right_corner=coord_right_corner, region_size_dict=region_size_dict, mesh_size=0.025, refine_factor=None, gmsh_options=gmsh_options)

gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)
revert_curve_list = []
revert_normal_dir_list=[1,1,1,1,1,1,1,1,1,1,1,1]
geom = GmshGeometry2D(gmsh_model, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)

# change global variables in elasticity_utils
e_1 = 1.3
nu_1 = 0.0
elasticity_utils.lame = e_1*nu_1/((1+nu_1)*(1-2*nu_1))
elasticity_utils.shear = e_1/(2*(1+nu_1))

elasticity_utils.geom = geom

def heaviside_subtracted(x_center, x_loc, deviation):
    '''
    Step or heaviside function --> Same parameter as exp_func
    '''
    return np.heaviside(x_loc+(x_center+deviation),1)-np.heaviside(x_loc+(x_center-deviation),1)

def heaviside_subtracted_tf(x_center, x_loc,  deviation):
    '''
    Tensor version of heaviside_subtracted --> x_loc is a Tensor
    '''
    return tf_original.experimental.numpy.heaviside(x_loc+(x_center+deviation),1)-tf_original.experimental.numpy.heaviside(x_loc+(x_center-deviation),1)

# Test the functions and get heights of them so that the total area is same for all of them
x_loc_p = geom.random_boundary_points(1)
x_loc_upper = x_loc_p[np.isclose(x_loc_p[:,1],y1)][:,0]
x_loc_upper = np.unique(x_loc_upper)

# step function
y_step = (heaviside_subtracted(load_center, x_loc_upper, load_deviation) + heaviside_subtracted(-load_center, x_loc_upper, load_deviation))

# The height of the step function is chosen 1 -->  so the overall are 1*0.01*2=0.02
pressure_step = 4.4

plot_loads = False

if plot_loads:
    fig, axs = plt.subplots(3,figsize=(45,15))

    #axs[0].scatter(x_loc_upper,pressure_exp*y_exp)
    axs[1].scatter(x_loc_upper,pressure_step*y_step)
    #axs[2].scatter(x_loc_upper,pressure_bsp*y_bsp)

    plt.setp(axs, xticks=np.arange(-1,1,0.1))
    plt.show()

assign_func = {"step" : heaviside_subtracted_tf}
assign_pressure = {"step" : pressure_step}
assign_deviation_top = {"step" : load_deviation}

assign_func_type = "step"

def neumann_y_top(x, y, X):
    '''
    Represents the y component of the applied pressure
    '''

    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5] 

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_yy_n_y = sigma_yy[cond]
    
    x_loc = x[:,0:1][cond] # Tensor
    
    assigned_func = assign_func[assign_func_type]

    return sigma_yy_n_y + assign_pressure[assign_func_type]*(assigned_func(load_center, x_loc, assign_deviation_top[assign_func_type]) + assigned_func(-load_center, x_loc, assign_deviation_top[assign_func_type]))    

assign_deviation_bottom = {"step" : disp_deviation}

def neumann_y_bottom(x, y, X):
    '''
    Represents the y component of the applied pressure
    '''

    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5] 

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_yy_n_y = sigma_yy[cond]
    
    x_loc = x[:,0:1][cond] # Tensor
    
    assigned_func = assign_func[assign_func_type]

    return sigma_yy_n_y + assign_pressure[assign_func_type]*(assigned_func(disp_center, x_loc, assign_deviation_bottom[assign_func_type]) + assigned_func(-disp_center, x_loc, assign_deviation_bottom[assign_func_type]))  

def pde_stress(x,y):

    # governing equation
    sigma_xx_x = dde.grad.jacobian(y, x, i=2, j=0)
    sigma_yy_y = dde.grad.jacobian(y, x, i=3, j=1)
    sigma_xy_x = dde.grad.jacobian(y, x, i=4, j=0)
    sigma_xy_y = dde.grad.jacobian(y, x, i=4, j=1)

    momentum_x = sigma_xx_x + sigma_xy_y
    momentum_y = sigma_yy_y + sigma_xy_x
    
    # material law
    term_x, term_y, term_xy = lin_iso_elasticity_plane_stress(x,y)

    return [momentum_x, momentum_y, term_x, term_y, term_xy]

def lin_iso_elasticity_plane_stress(x,y):
    
    sigma_xx, sigma_yy, sigma_xy = stress_plane_stress(x,y)
    
    term_x = sigma_xx - y[:, 2:3]
    term_y = sigma_yy - y[:, 3:4]
    term_xy = sigma_xy - y[:, 4:5]
    
    return term_x, term_y, term_xy

def zero_neumann_x(x,y,X):
    
    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5] 

    normals, cond = calculate_boundary_normals(X, geom)

    sigma_xx_n_x = sigma_xx[cond]*normals[:,0:1]
    sigma_xy_n_y = sigma_xy[cond]*normals[:,1:2]
    
    traction_x = sigma_xx_n_x + sigma_xy_n_y

    return traction_x

def zero_neumann_y(x,y,X):
    
    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5] 

    normals, cond = calculate_boundary_normals(X, geom)

    sigma_yx_n_x = sigma_xy[cond]*normals[:,0:1]
    sigma_yy_n_y = sigma_yy[cond]*normals[:,1:2]
    
    traction_y = sigma_yx_n_x + sigma_yy_n_y

    return traction_y

def top_region(x, on_boundary):
    '''Find the points on where the load vectors are applied'''
    top_mask = np.isclose(x[1],coord_right_corner[1])
    
    return on_boundary and top_mask

def bottom_region(x, on_boundary):
    '''Find the points on where the load vectors are applied'''
    bottom_mask = np.isclose(x[1],coord_left_corner[1])
    
    return on_boundary and bottom_mask

def top_bottom(x, on_boundary):
    top_mask = np.isclose(x[1],coord_right_corner[1])
    bottom_mask = np.isclose(x[1],coord_left_corner[1])
    
    return on_boundary and np.logical_or(bottom_mask,top_mask)

# def left_right(x, on_boundary):
#     '''Find the points on where no force or boundary conditions are applied (zero Neumann)'''
#     right = np.isclose(x[0],coord_right_corner[0])
#     left = np.isclose(x[0],coord_left_corner[0])
    
#     return on_boundary and np.logical_or(right, left)

def strain_xx(x,y,X):
    return dde.grad.jacobian(y, x, i=0, j=0)

# # add FEM data
# ext_data = np.loadtxt(Path(__file__).parent.absolute()/"train_data/4PointBendingTest_2DFEM_data.txt")
# ext_data = ext_data[:100,:]
# ex_data_xy = ext_data[:,0:2]
# observe_u  = dde.PointSetBC(ex_data_xy, ext_data[:,2:3], component=0)
# observe_v  = dde.PointSetBC(ex_data_xy, ext_data[:,3:4], component=1)


# Experimental Data from Johannes 
xChannel2 = np.load(Path(__file__).parent.parent.parent.absolute()/"train_data/xChannel2.npy") 
measured_strains_2 = np.load(Path(__file__).parent.parent.parent.absolute()/"train_data/strainsInitialChannel2.npy")
#xChannel1 = np.load(Path(__file__).parent.parent.parent.absolute()/"train_data/xChannel1.npy") 
#measured_strains_1 = np.load(Path(__file__).parent.parent.parent.absolute()/"train_data/strainsChannel1Luna2Initial.npy")

# Filter out nans 
xChannel2 = xChannel2[np.isfinite(measured_strains_2)]
measured_strains_2 = measured_strains_2[np.isfinite(measured_strains_2)]

# rescale the strains
scale_factor = 1.3/27600000*33/(4.4*0.1)
measured_strains_2 = measured_strains_2/1e6/scale_factor
#measured_strains_1 = measured_strains_1/1e6/scale_factor

xchannel2_strain_combined = np.hstack((xChannel2.reshape(-1,1),0.12*np.ones(xChannel2.shape).reshape(-1,1),measured_strains_2.reshape(-1,1)))
#xchannel1_strain_combined = np.hstack((xChannel1.reshape(-1,1),-0.12*np.ones(xChannel1.shape).reshape(-1,1),measured_strains_1.reshape(-1,1)))
np.random.seed(12)
np.random.shuffle(xchannel2_strain_combined)
#np.random.shuffle(xchannel1_strain_combined)

# plt.scatter(xchannel2_strain_combined[:100,0],xchannel2_strain_combined[:100,2])
# plt.scatter(xchannel1_strain_combined[:100,0],xchannel1_strain_combined[:100,2])
# plt.show()

ex_data_xy_strain = xchannel2_strain_combined[:200,0:2]#np.vstack((xchannel2_strain_combined[:100,0:2],xchannel1_strain_combined[:100,0:2]))
measured_strains_sliced = xchannel2_strain_combined[:200,2:3]#np.vstack((xchannel2_strain_combined[:100,2:3], xchannel1_strain_combined[:100,2:3]))

observe_eps_xx = dde.PointSetOperatorBC(ex_data_xy_strain, measured_strains_sliced, strain_xx)

#ex_data_xy = np.vstack((ex_data_xy, ex_data_xy_strain))

bc1 = dde.OperatorBC(geom, neumann_y_top, top_region)
bc2 = dde.OperatorBC(geom, neumann_y_bottom, bottom_region)
bc3 = dde.OperatorBC(geom, zero_neumann_x, top_bottom)

n_dummy = 1
data = dde.data.PDE(
    geom,
    pde_stress,
    [bc1, bc2, bc3, observe_eps_xx], #, observe_u, observe_v
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=n_dummy,
    train_distribution = "Sobol",
    anchors=ex_data_xy_strain
)

def output_transform(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    sigma_xx = y[:, 2:3]
    sigma_yy = y[:, 3:4]
    sigma_xy = y[:, 4:5]
    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    
    return tf.concat([ u*x_loc,v,sigma_xx*(x_loc-l/2)*(x_loc+l/2),sigma_yy,sigma_xy*(x_loc-l/2)*(x_loc+l/2)], axis=1)

# two inputs x and y, five outputs ux, uy, sigma_xx, sigma_yy, sigma_xy
layer_size = [2] + [50] * 4 + [5]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform) # enforce the zero Neumann BCs
model = dde.Model(data, net)

loss_weights = []

model.compile("adam", lr=0.001) 
losshistory, train_state = model.train(epochs=5000, display_every=200)

model.compile("L-BFGS")
model.train()

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

X, offset, cell_types, dol_triangles = geom.get_mesh()

output = model.predict(X)
u_pred, v_pred = output[:,0], output[:,1]
sigma_xx, sigma_yy, sigma_xy = output[:,2],output[:,3],output[:,4]
eps_xx, eps_yy, eps_xy = model.predict(X, operator=elastic_strain_2d)
eps_xx, eps_yy, eps_xy = eps_xx, eps_yy, eps_xy

combined_disp = tuple(np.vstack((np.array(u_pred.tolist()),np.array(v_pred.tolist()),np.zeros(u_pred.shape[0]))))
combined_stress = tuple(np.vstack((np.array(sigma_xx.tolist()),np.array(sigma_yy.tolist()),np.array(sigma_xy.tolist()))))
combined_strain = tuple(np.vstack((np.array(eps_xx.flatten().tolist()),np.array(eps_yy.flatten().tolist()),np.array(eps_xy.flatten().tolist()))))


file_path = os.path.join(os.getcwd(), "4PointBendingMixedFormulation")

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                    cell_types, pointData = { "displacement" : combined_disp,
                    "stress" : combined_stress, "strain": combined_strain}) # , 


