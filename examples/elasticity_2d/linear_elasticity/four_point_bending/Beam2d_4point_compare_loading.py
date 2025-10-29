import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
from deepxde.backend import tf
from deepxde import config
import tensorflow as tf_original
from pyevtk.hl import unstructuredGridToVTK

from scipy.interpolate import BSpline

from compsim_pinns.elasticity.elasticity_utils import stress_plane_stress, momentum_2d_plane_stress, zero_neumman_plane_stress_x, zero_neumman_plane_stress_y, elastic_strain_2d
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
    D1                                   D2

where a pressure vector is applied on P1 and P2, while it is constrained in x and y direction at regions D1 and D2 

Also one can apply different loadings e.g. exponential, step or B-Splines

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
block_2d = Rectangle_4PointBendingCentered(coord_left_corner=coord_left_corner, coord_right_corner=coord_right_corner, region_size_dict=region_size_dict, mesh_size=0.08, refine_factor=30, gmsh_options=gmsh_options)

gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)
revert_curve_list = []
revert_normal_dir_list=[1,1,1,1,1,1,1,1,1,1,1,1]
geom = GmshGeometry2D(gmsh_model, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)

# change global variables in elasticity_utils
e_1 = 1.333
nu_1 = 0.33
elasticity_utils.lame = e_1*nu_1/((1+nu_1)*(1-2*nu_1))
elasticity_utils.shear = e_1/(2*(1+nu_1))
# zero neumann BC functions need the geom variable to be defined
elasticity_utils.geom = geom

# functions for upper load
def exp_func(x_center, x_loc, factor):
    '''
    Gaussian-like function
    
    Parameters
    ----------
    x_center : float
        center location of applied load
    x_loc : numpy array
        x locations where the force is applied
    factor : float
        decides the width of the exponential function
    '''
    return (np.exp(-factor*(x_loc+x_center)**2))

def heaviside_subtracted(x_center, x_loc, deviation):
    '''
    Step or heaviside function --> Same parameter as exp_func
    '''
    return np.heaviside(x_loc+(x_center+deviation),1)-np.heaviside(x_loc+(x_center-deviation),1)

def exp_func_tf(x_center, x_loc, factor):
    '''
    Tensor version of exp_func --> x_loc is a Tensor
    '''
    return (tf.exp(-factor*(x_loc+x_center)**2))

def heaviside_subtracted_tf(x_center, x_loc,  deviation):
    '''
    Tensor version of heaviside_subtracted --> x_loc is a Tensor
    '''
    return tf_original.experimental.numpy.heaviside(x_loc+(x_center+deviation),1)-tf_original.experimental.numpy.heaviside(x_loc+(x_center-deviation),1)

def b_spline(x_center, x_loc,  deviation):
    '''
    Generates B-splines --> Same parameter as exp_func
    '''
    b = BSpline.basis_element([x_center-deviation,x_center-deviation,x_center,x_center+deviation,x_center+deviation])
    return b(x_loc)

# Test the functions and get heights of them so that the total area is same for all of them
x_loc_p = geom.random_boundary_points(1)
x_loc_upper = x_loc_p[np.isclose(x_loc_p[:,1],0.15)][:,0]
x_loc_upper = np.unique(x_loc_upper)

# exponential function
factor = 5000
y_exp = (exp_func(load_center, x_loc_upper, factor) + exp_func(-load_center, x_loc_upper, factor))

# step function
y_step = (heaviside_subtracted(load_center, x_loc_upper, load_deviation) + heaviside_subtracted(-load_center, x_loc_upper, load_deviation))

# B-spline function
y_bsp = (b_spline(load_center, x_loc_upper, load_deviation) + b_spline(-load_center, x_loc_upper, load_deviation))
# tf.convert_to_tensor(y_bsp, dtype=config.real(tf))

# The height of the step function is chosen 1 -->  so the overall are 1*0.01*2=0.02
pressure_step = 1
pressure_exp = np.trapz(y_step,x_loc_upper)/np.trapz(y_exp,x_loc_upper) # max height of exponential func.
pressure_bsp = np.trapz(y_step,x_loc_upper)/np.trapz(y_bsp,x_loc_upper) # max height of Bspline.

print(f"The expected area: 0.02, step func. area: {pressure_step*np.trapz(y_step,x_loc_upper)}")
print(f"The expected area: 0.02, exp. func. area: {pressure_exp*np.trapz(y_exp,x_loc_upper)}")
print(f"The expected area: 0.02, B-spline area  : {pressure_bsp*np.trapz(y_bsp,x_loc_upper)}")

plot_loads = True

if plot_loads:
    fig, axs = plt.subplots(3,figsize=(45,15))

    axs[0].scatter(x_loc_upper,pressure_exp*y_exp)
    axs[1].scatter(x_loc_upper,y_step)
    axs[2].scatter(x_loc_upper,pressure_bsp*y_bsp)

    plt.setp(axs, xticks=np.arange(-1.5,1.5,0.1))
    plt.show()

assign_func = {"exp":exp_func_tf, "step" : heaviside_subtracted_tf, "bspline":b_spline}
assign_pressure = {"exp":pressure_exp, "step" : 1, "bspline":pressure_bsp}
assign_deviation = {"exp":factor, "step" : load_deviation, "bspline":load_deviation}

assign_func_type = "bspline"

def neumann_y(x, y, X):
    '''
    Represents the y component of the applied pressure
    '''

    sigma_xx, sigma_yy, sigma_xy = stress_plane_stress(x,y)

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_yy_n_y = sigma_yy[cond]
    
    x_loc = x[:,0:1][cond] # Tensor
    
    assigned_func = assign_func[assign_func_type]
    
    if assign_func_type == "bspline": # B-spline is a bit special since it is not the tensorflow function, X instead of x is used (X and x are identical but X is np.array and x in tf.Tensor)
        x_loc = X[:,0:1][cond]
        y_np = assign_pressure[assign_func_type]*(assigned_func(load_center, x_loc, assign_deviation[assign_func_type]) + assigned_func(-load_center, x_loc, assign_deviation[assign_func_type])) 
        y_tf = tf.convert_to_tensor(y_np, dtype=config.real(tf))
        return sigma_yy_n_y + y_tf
    else:
        return sigma_yy_n_y + assign_pressure[assign_func_type]*(assigned_func(load_center, x_loc, assign_deviation[assign_func_type]) + assigned_func(-load_center, x_loc, assign_deviation[assign_func_type]))    

def dirichlet_region(x, on_boundary):
    '''Find the points on where the BCs are applied'''
    region_1 = np.logical_and(x[0] >= (region_size_dict["r1"]["center"]-region_size_dict["r1"]["deviation"]), x[0] <= (region_size_dict["r1"]["center"] + region_size_dict["r1"]["deviation"]))
    region_2 = np.logical_and(x[0] >= (region_size_dict["r2"]["center"]-region_size_dict["r2"]["deviation"]), x[0] <= (region_size_dict["r2"]["center"] + region_size_dict["r2"]["deviation"]))
    bottom_mask = np.isclose(x[1],coord_left_corner[1])
    
    return on_boundary and np.logical_or(region_1, region_2) and bottom_mask

def force_region(x, on_boundary):
    '''Find the points on where the load vectors are applied'''
    top_mask = np.isclose(x[1],coord_right_corner[1])
    
    return on_boundary and top_mask

def free_regions(x, on_boundary):
    '''Find the points on where no force or boundary conditions are applied (zero Neumann)'''
    bottom_points = dirichlet_region(x, on_boundary)
    top_points = np.isclose(x[1],coord_right_corner[1])
    
    return on_boundary and ~bottom_points and ~top_points #and ~corner_points


# ext_data = np.loadtxt("/home/a11btasa/git_repos/pinnswithdxde/elasticity_2d/fourpoint_train_anchors.txt")
# ext_data = ext_data[:100,:]
# ex_data_xy = ext_data[:,0:2]
# observe_u  = dde.PointSetBC(ex_data_xy, ext_data[:,2:3], component=0)
# observe_v  = dde.PointSetBC(ex_data_xy, ext_data[:,3:4], component=1)

bc1 = dde.DirichletBC(geom, lambda _: 0.0, dirichlet_region, component=0)
bc2 = dde.DirichletBC(geom, lambda _: 0.0, dirichlet_region, component=1)
bc3 = dde.OperatorBC(geom, zero_neumman_plane_stress_x, free_regions)
bc4 = dde.OperatorBC(geom, zero_neumman_plane_stress_y, free_regions)
bc5 = dde.OperatorBC(geom, zero_neumman_plane_stress_x, force_region)
bc6 = dde.OperatorBC(geom, neumann_y, force_region)

n_dummy = 1
data = dde.data.PDE(
    geom,
    momentum_2d_plane_stress,
    [bc1, bc2, bc3, bc4, bc5, bc6], #, observe_u, observe_v
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=100,
    train_distribution = "Sobol",
    #anchors=ex_data_xy
)

def output_transform(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    return tf.concat([ u*1e-2, v*1e-3], axis=1) #[ u*1e-2, v*1e-3]

def feature_transform(x):
    return tf.concat([x[:, 0:1] / l, x[:, 1:2] /l], axis=1)

# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 3 + [2]
#layer_size = [2] + [60,60] + [60,60] + [60,60] + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
# net.apply_output_transform(output_transform)
# net.apply_feature_transform(feature_transform)
loss_weights=[1,1,1e0,1e0,1,1,1,1] #,1,1
model = dde.Model(data, net)


model.compile("adam", lr=0.001, loss_weights=loss_weights)
losshistory, train_state = model.train(epochs=5000, display_every=200)

model.compile("L-BFGS",loss_weights=loss_weights)
model.train()

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

X, offset, cell_types, dol_triangles = geom.get_mesh()

displacement = model.predict(X)
sigma_xx, sigma_yy, sigma_xy = model.predict(X, operator=stress_plane_stress)
eps_xx, eps_yy, eps_xy = model.predict(X, operator=elastic_strain_2d)


combined_disp = tuple(np.vstack((np.array(displacement[:,0].tolist()),np.array(displacement[:,1].tolist()),np.zeros(displacement[:,0].shape[0]))))
combined_stress = tuple(np.vstack((np.array(sigma_xx.flatten().tolist()),np.array(sigma_yy.flatten().tolist()),np.array(sigma_xy.flatten().tolist()))))
combined_strain = tuple(np.vstack((np.array(eps_xx.flatten().tolist()),np.array(eps_yy.flatten().tolist()),np.array(eps_xy.flatten().tolist()))))


file_path = os.path.join(os.getcwd(), f"Beam2D_finalize_continuous_{assign_func_type}")

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                    cell_types, pointData = { "displacement" : combined_disp,
                    "stress" : combined_stress, "strain": combined_strain})


