import deepxde as dde
import numpy as np
import os
from deepxde.backend import tf
from pyevtk.hl import unstructuredGridToVTK
from deepxde import backend as bkd
import pandas as pd
from pathlib import Path
from matplotlib import tri
import pyvista as pv

from utils.elasticity.elasticity_utils import problem_parameters, first_piola_stress_tensor, momentum_2d_firstpiola, problem_parameters, zero_neumman_first_piola_x, zero_neumman_first_piola_y, cauchy_stress
from utils.geometry.geometry_utils import calculate_boundary_normals
from utils.geometry.custom_geometry import GmshGeometry2D
from utils.geometry.gmsh_models import Block_2D
from utils.elasticity import elasticity_utils


'''
The correct order for the normals --> 1 2 1 1

Reference solution:
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.504.4507&rep=rep1&type=pdf

@author: tsahin
'''

height = 1
width = 5
applied_displacement = -0.1
elasticity_utils.model_complexity = "nonlinear"
model_complexity = elasticity_utils.model_complexity

gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
block_2d = Block_2D(coord_left_corner=[-width/2,-height/2], coord_right_corner=[width/2,height/2], mesh_size=0.095, gmsh_options=gmsh_options) #0.095

gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)

revert_curve_list = []
revert_normal_dir_list = [1,2,1,1]
geom = GmshGeometry2D(gmsh_model, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)

l = block_2d.coord_right_corner[0] -block_2d.coord_left_corner[0] #5
h = block_2d.coord_right_corner[1] -block_2d.coord_left_corner[1]

# change global variables in elasticity_utils
e_1 = 10
nu_1 = 0.3
# elasticity_utils.lame = e_1*nu_1/((1+nu_1)*(1-2*nu_1))
# elasticity_utils.shear = e_1/(2*(1+nu_1))
elasticity_utils.lame = 75/13
elasticity_utils.shear = 50/13
# zero neumann BC functions need the geom variable to be 
elasticity_utils.geom = geom

nu, lame, shear, e_modul = problem_parameters()

def top_bottom(x, on_boundary):
    not_included_points = np.logical_or(np.isclose(x[0],width/2), np.isclose(x[0],-width/2))
    points_top = np.logical_and(np.isclose(x[1],height/2),~not_included_points)
    points_bottom = np.logical_and(np.isclose(x[1],-height/2),~not_included_points)
    
    return on_boundary and np.logical_or(points_top, points_bottom)

def top_bottom_right(x, on_boundary):
    not_included_points = np.logical_or(np.isclose(x[0],width/2), np.isclose(x[0],-width/2))
    points_top = np.logical_and(np.isclose(x[1],height/2),~not_included_points)
    points_bottom = np.logical_and(np.isclose(x[1],-height/2),~not_included_points)
    points_right = np.isclose(x[0],width/2)
    
    return on_boundary and np.logical_or(np.logical_or(points_top, points_bottom), points_right)

def left(x, on_boundary):
    return on_boundary and np.isclose(x[0],-width/2)

def right(x, on_boundary):
    return on_boundary and np.isclose(x[0],width/2)


bc1 = dde.OperatorBC(geom, zero_neumman_first_piola_x, top_bottom_right)
bc2 = dde.OperatorBC(geom, zero_neumman_first_piola_y, top_bottom)
bc3 = dde.DirichletBC(geom, lambda _: applied_displacement, right, component=1)
bc4 = dde.DirichletBC(geom, lambda _: 0, left, component=0)
bc5 = dde.DirichletBC(geom, lambda _: 0, left, component=1)


n_dummy = 1
data = dde.data.PDE(
    geom,
    momentum_2d_firstpiola,
    [bc1, bc2, bc3, bc4, bc5],
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=None,
    train_distribution = "Sobol",
)

def output_transform(x, y):
    u = y[:, 0:1]       #x-displacement
    v = y[:, 1:2]       #y-displacement
    x_loc = x[:, 0:1] 
    print(x_loc)                  
    y_loc = x[:, 1:2]
    left_side = (width/2+x_loc)
    right_side = (width/2-x_loc)
    return bkd.concat([(u)*width, (v)/e_modul], axis=1)
    #return bkd.concat([(u*left_side/(e_modul**2*width)), (v*left_side/(e_modul))], axis=1)
    # return bkd.concat([(u*left_side)/e_modul, (v*right_side*left_side+left_side*-1.5/width)/e_modul], axis=1)                                  #Hard enforcement of DBC on the right

# in case hard Dirichlet is desired (no scaling!! so it must be tested)
# def output_transform(x, y):
#     x_loc = x[:,0:1]
#     y_loc = x[:,1:2]
#     u_x_analy = y[:,0:1]*shear_y*y_loc/(6*e_modul*Inertia)*((6*l-3*x_loc)*x_loc + (2+nu)*(y_loc**2-h**2/4))
#     u_y_analy = -y[:,1:2]*shear_y/(6*e_modul*Inertia)*(3*nu*y_loc**2*(l-x_loc) + (4+5*nu)*h**2*x_loc/4 + (3*l-x_loc)*x_loc**2)
#     return tf.concat([ u_x_analy, u_y_analy], axis=1)

# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 3 + [2]
activation = "swish"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
# net.apply_output_transform(output_transform)
loss_weights=[1,1,1,1,1]

model = dde.Model(data, net)
model.compile("adam", lr=0.001, loss_weights=None)
losshistory, train_state = model.train(epochs=2500, display_every=200)

model.compile("L-BFGS",loss_weights=None)
model.train()

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

# # analytical solution for graph over line

# def f(x, applied_displacement, e_modul, height):
#     """
#     Compute f(x) based on given parameters.

#     Parameters:
#         x (float): Value at which to evaluate the function.
#         applied_displacement (float): Applied displacement (-1.5, -1.0, -0.5, -0.1).
#         e_modul (float): Elastic modulus.
#         height (float): Height divided by 12.

#     Returns:
#         float: Computed value of the function.
#     """
#     nu, lame, shear, e_modul = problem_parameters()

#     # F based on applied_displacement
#     if applied_displacement == -1.5:
#         F = -0.03
#     elif applied_displacement == -1.0:
#         F = -0.02
#     elif applied_displacement == -0.5:
#         F = -0.01
#     elif applied_displacement == -0.1:
#         F = -0.002
#     else:
#         raise ValueError("Invalid applied_displacement value.")
    
#     return (F * (2.5 + x)**3) / (3 * e_modul * (height / 12))

# # Generate x values from -width/2 to +width/2
# x_values = np.linspace(-width/2, width/2, 6)

# # Compute analytical_displacement_y for each x
# analytical_displacement_y = np.array([f(x, applied_displacement, e_modul, height) for x in x_values])

# # Update PyVista points and data
# data.points = np.column_stack((x_values, np.zeros_like(x_values), np.zeros_like(x_values)))  # 3D points
# # Extract points (or define them manually if not available)
# points = np.column_stack((x_values, np.zeros_like(x_values), np.zeros_like(x_values)))  # 3D points

# # Convert to PyVista
# pv_data = pv.PolyData(points)

# # Add analytical_displacement_y to the PyVista object
# pv_data.point_data['analytical_displacement_y'] = analytical_displacement_y

# # Save to a file
# filename = f"Beam2D_{model_complexity}_u_{applied_displacement:.1f}_actfunc_{activation}_analy.vtp"
# pv_data.save(filename)

file_path =  f"/home_student/kappen/Comparison_FE_to_PINN_in_paraview/ba-kappen-reference-results-main/bending_beam/{model_complexity}/bending_beam_{model_complexity}_E=10.0_disp={applied_displacement:.1f}/bending_beam_{model_complexity}_E=10.0_disp={applied_displacement:.1f}-structure.pvd"

# Convert the Path object to a string
reader = pv.get_reader(file_path)

reader.set_active_time_point(-1)
data = reader.read()[0]

X = data.points

# print("Shape of X:", X.shape)
# print("Shape of X[:, 0:2]:", X[:, 0:2].shape)

displacement = model.predict(X[:,0:2])
T_xx, T_yy, T_xy, T_yx = model.predict(X[:,0:2], operator=cauchy_stress)
P_xx, P_yy, P_xy, P_yx = model.predict(X[:,0:2], operator=first_piola_stress_tensor)

first_piola = np.column_stack((P_xx, P_yy, P_xy))
cauchy = np.column_stack((T_xx, T_yy, T_xy))

displacement_extended = np.hstack((displacement, np.zeros_like(displacement[:,0:1])))

data.point_data['pred_first_piola'] = first_piola
data.point_data['pred_displacement'] = displacement_extended
data.point_data['pred_stress'] = cauchy

disp_fem = data.point_data['displacement']
stress_fem = data.point_data['nodal_cauchy_stresses_xyz']

error_disp = abs((disp_fem - displacement))
data.point_data['pointwise_displacement_error'] = error_disp
# select xx, yy, and xy component (1st, 2nd and 4th column)
columns = [0,1,3]
error_stress = abs((stress_fem[:, columns] - cauchy))
data.point_data['pointwise_cauchystress_error'] = error_stress
#data.point_data['pointwise_cauchystress_error'].column_names

data.save(f"Beam2D_{model_complexity}u{applied_displacement:.1f}_actfunc_{activation}.vtu")

print("NOTE THAT 'Warp By Vector' DOES NOT WORK HERE AS THE Z-DIMENSION VALUES ARE ILL-DEFINED.")
print("USE CALCULATION WITH 'displacement_X*iHat + displacement_Y*jHat + 0*kHat' AND THEN APPLY 'Warp By Vector'.")

exit()
# displacement_fem = data.point_data['displacement']
# stress_fem = data.point_data['nodal_cauchy_stresses_xyz']cauchy


# fem_path = str(Path(__file__).parent.parent.parent.parent.parent)+"/Comparison_FE_to_PINN_in_paraview/2D Beam/Set_2_correct_visualization/fem_spreadsheet_2d_beam.csv"
# df = pd.read_csv(fem_path)
# fem_results = df[["Points_0","Points_1","displacement_0","displacement_1","nodal_cauchy_stresses_xyz_0","nodal_cauchy_stresses_xyz_1","nodal_cauchy_stresses_xyz_3"]]
# fem_results = fem_results.to_numpy()

# displacement_fem = fem_results[:,2:4]
# stress_fem = fem_results[:,4:7]

# X, offset, cell_types, dol_triangles = geom.get_mesh()
triangles = tri.Triangulation(x, y)

triangle_coordinate_x = x[triangles.triangles]
triangle_coordinate_y = y[triangles.triangles]

# np.isclose(np.linalg.norm(x - center_outer, axis=-1), radius_outer)
calculate_norm = np.sqrt(triangle_coordinate_x**2+triangle_coordinate_y**2)
mask = np.isclose(calculate_norm,1)
dol_triangles = triangles.triangles[~mask.all(axis=1)]

# fem
u_fem, v_fem = displacement_fem[:,0], displacement_fem[:,1]
sigma_xx_fem, sigma_yy_fem, sigma_xy_fem = stress_fem[:,0:1], stress_fem[:,1:2], stress_fem[:,2:3]

combined_disp_fem = tuple(np.vstack((np.array(u_fem.tolist()),np.array(v_fem.tolist()),np.zeros(u_fem.shape[0]))))
combined_stress_fem = tuple(np.vstack((np.array(sigma_xx_fem.flatten().tolist()),np.array(sigma_yy_fem.flatten().tolist()),np.array(sigma_xy_fem.flatten().tolist()))))

displacement = model.predict(X)
u_pred, v_pred = displacement[:,0], displacement[:,1]
T_xx, T_yy, T_xy, T_yx = model.predict(X, operator=cauchy_stress)                                   # Original output sigma_xx, sigma_yy, sigma_xy

p_xx, p_yy, p_xy, p_yx = model.predict(X, operator=first_piola_stress_tensor) 

combined_disp = tuple(np.vstack((np.array(displacement[:,0].tolist()),np.array(displacement[:,1].tolist()),np.zeros(displacement[:,0].shape[0]))))
combined_stress = tuple(np.vstack((np.array(T_xx.flatten().tolist()),np.array(T_yy.flatten().tolist()),np.array(T_xy.flatten().tolist()))))     # Original output sigma_xx, sigma_yy, sigma_xy
combined_stress_p = tuple(np.vstack((np.array(p_xx.flatten().tolist()),np.array(p_yy.flatten().tolist()),np.array(p_xy.flatten().tolist()))))

# error
error_disp_x = abs(np.array(u_pred.tolist()) - u_fem.flatten())
error_disp_y =  abs(np.array(v_pred.tolist()) - v_fem.flatten())
combined_error_disp = tuple(np.vstack((error_disp_x, error_disp_y,np.zeros(error_disp_x.shape[0]))))

error_stress_x = abs(np.array(T_xx.flatten().tolist()) - sigma_xx_fem.flatten())
error_stress_y =  abs(np.array(T_yy.flatten().tolist()) - sigma_yy_fem.flatten())
error_stress_xy =  abs(np.array(T_xy.flatten().tolist()) - sigma_xy_fem.flatten())
combined_error_stress = tuple(np.vstack((error_stress_x, error_stress_y, error_stress_xy)))


#dol_triangles = dol_triangles
offset = np.arange(3,dol_triangles.shape[0]*dol_triangles.shape[1]+1,dol_triangles.shape[1]).astype(dol_triangles.dtype)
cell_types = np.ones(dol_triangles.shape[0])*5

#file_path = os.path.join(os.getcwd(), "Beam2D_gmsh_nicht_linear_applied_displacment")
file_path = os.path.join(os.getcwd(),f"Beam2D_gmsh_nicht_linear_displacement_{applied_displacement:.2f}_activationfunc_{activation}")


x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, cell_types, pointData = { "displacement" : combined_disp, "stress" : combined_stress, "1st piola stress": combined_stress_p, 
                                                                                                    "stress_fem": combined_stress_fem, "1st piola stress": combined_stress_p, "error_disp_x": error_disp_x, "error_disp_y": error_disp_y, 
                                                                                                    "error_stress_x": error_stress_x, "error_stress_y": error_stress_y, "error_stress_xy": error_stress_xy})
