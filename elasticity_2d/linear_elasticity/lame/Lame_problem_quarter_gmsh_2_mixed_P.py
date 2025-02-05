import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
from pyevtk.hl import unstructuredGridToVTK
from pathlib import Path
from deepxde import backend as bkd
import pandas as pd
from matplotlib import tri
import pyvista as pv

from utils.elasticity.elasticity_utils import problem_parameters, first_piola_stress_tensor, piola_stress_to_traction_2d, cauchy_stress_mixed_P, momentum_mixed_P, problem_parameters, zero_neumann_x_mixed_P_formulation, zero_neumann_y_mixed_P_formulation, cauchy_stress, compute_relative_l2_error
from utils.geometry.geometry_utils import calculate_boundary_normals, polar_transformation_2d
from utils.geometry.custom_geometry import GmshGeometryElement
from utils.geometry.gmsh_models import QuarterCirclewithHole
from utils.elasticity import elasticity_utils

'''
Solves a hollow quarter cylinder under internal pressure (Lame problem)

Reference solution:
https://onlinelibrary.wiley.com/doi/epdf/10.1002/nme.6132

@author: tsahin
'''

elasticity_utils.model_complexity = "nonlinear"     #with "linear" --> linear strain definition, everyhing else i.e. "hueicii" nonlinear
elasticity_utils.model_type = "plane_stress"        #with "plane_strain" --> plane strain, everyhing else i.e. "hueicii" plane stress
model_type = elasticity_utils.model_type 
model_complexity = elasticity_utils.model_complexity

gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
quarter_circle_with_hole = QuarterCirclewithHole(center=[0,0,0], inner_radius=1, outer_radius=2, mesh_size=0.1, gmsh_options=gmsh_options)

gmsh_model = quarter_circle_with_hole.generateGmshModel(visualize_mesh=True)

revert_curve_list = ["curve_2"]
revert_normal_dir_list = [2,2,1,2]
geom = GmshGeometryElement(gmsh_model, dimension=2, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)

radius_inner = quarter_circle_with_hole.inner_radius
center_inner = [quarter_circle_with_hole.center[0],quarter_circle_with_hole.center[1]]
radius_outer = quarter_circle_with_hole.outer_radius
center_outer = [quarter_circle_with_hole.center[0],quarter_circle_with_hole.center[1]]

# change global variables in elasticity_utils
elasticity_utils.geom = geom
# change global variables in elasticity_utils
elasticity_utils.lame = 150/13     #75/26 -->E-Modul=5    75/13 -->E-Modul=10    150/13 -->E-Modul=20       1500/13 -->E-Modul=200    1153.846  -->E-Modul= 2000   1575000/13   --> E-Modul=210000
elasticity_utils.shear = 100/13    #25/13                  50/13                   100/13                     1000/13                    769.23                       1050000/13
nu,lame,shear,e_modul = problem_parameters()

# The applied pressure 
pressure_inlet = 1 #0.9101  #1/0.9325*0.8325

def pressure_inner_x(x, y, X):
    
    p_xx, p_yy, p_xy, p_yx = first_piola_stress_tensor(x,y)
    
    normals, cond = calculate_boundary_normals(X, geom)
    Tx, _, _, _ = piola_stress_to_traction_2d(p_xx, p_yy, p_xy, p_yx, normals, cond)

    return Tx + pressure_inlet*normals[:,0:1]

def pressure_inner_y(x, y, X):

    p_xx, p_yy, p_xy, p_yx = first_piola_stress_tensor(x,y)
    
    normals, cond = calculate_boundary_normals(X,geom)
    _, Ty, _, _ = piola_stress_to_traction_2d(p_xx, p_yy, p_xy, p_yx, normals, cond)

    return Ty + pressure_inlet*normals[:,1:2]

def boundary_outer(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center_outer, axis=-1), radius_outer)

def boundary_inner(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center_inner, axis=-1), radius_inner) #and ~np.logical_and(np.isclose(x[0],1),np.isclose(x[1],0)) and ~np.logical_and(np.isclose(x[0],0),np.isclose(x[1],1))

def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0],0)

def boundary_bottom(x, on_boundary):
    return on_boundary and np.isclose(x[1],0)

bc1 = dde.OperatorBC(geom, pressure_inner_x, boundary_inner)
bc2 = dde.OperatorBC(geom, pressure_inner_y, boundary_inner)
bc3 = dde.DirichletBC(geom, lambda _: 0.0, boundary_left, component=0)
bc4 = dde.DirichletBC(geom, lambda _: 0.0, boundary_bottom, component=1)
bc5 = dde.OperatorBC(geom, zero_neumann_x_mixed_P_formulation, boundary_outer)
bc6 = dde.OperatorBC(geom, zero_neumann_y_mixed_P_formulation, boundary_outer)

n_dummy = 1
data = dde.data.PDE(
    geom,
    momentum_mixed_P,
    [bc1, bc2, bc3, bc4, bc5, bc6],       #DBC are now hard constraints
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=None,
    train_distribution = "Sobol"
)

def output_transform(x, y):
    '''
    Hard BCs:
        Dirichlet terms
            u(x=0)=0
        
        Neumann terms:
            sigma_xx(x=l) = 0
            sigma_yy(y=h) = ext_traction
            sigma_xy(x=l) = 0, sigma_xy(x=0) = 0 and sigma_xy(y=h) = 0

        where h:=h_beam and l:=l_beam.
    
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
    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    P_xx, P_yy, P_xy, P_yx = y[:, 2:3], y[:, 3:4], y[:, 4:5], y[:, 5:6]
    
    return bkd.concat([u*(x_loc)/e_modul,v*(y_loc)/e_modul, P_xx, P_yy, P_xy, P_yx], axis=1)

# two inputs x and y, output is ux, uy, Pxx, Pyy, Pxy, Pyx
layer_size = [2] + [50] * 3 + [6]
activation = "swish"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

#model_path = str(Path(__file__).parent.parent.parent)+"/trained_models/lame/lame"
#n_epochs = 3106 # trained model has 3106 iterations
#model_restore_path = model_path + "-"+ str(n_epochs) + ".ckpt"

model = dde.Model(data, net)
# if we want to save the model, we use "model_save_path=model_path" during training, if we want to load trained model, we use "model_restore_path=return_restore_path(model_path, num_epochs)"
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=4000, display_every=200) #, model_restore_path=None)

model.compile("L-BFGS")
model.train(model_save_path=None, display_every=200)

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

file_path2 =  f"/home_student/kappen/Comparison_FE_to_PINN_in_paraview/ba-kappen-reference-results-main/lame_quarter_circle/{model_complexity}/lame_quarter_circle_{model_complexity}_{model_type}_E=20_pres=1.0-structure.pvd"

# Convert the Path object to a string
reader = pv.get_reader(file_path2)

reader.set_active_time_point(-1)
data = reader.read()[0]

X = data.points

# disp_linear_fem_polar = np.sqrt(disp_linear_fem[:,0:1]**2 + disp_linear_fem[:,1:2]**2)
# data.point_data['disp_linear_fem_polar'] = disp_linear_fem_polar

# T_rr_fe_linear, T_theta_fe_linear, T_rtheta_fe_linear = polar_transformation_2d(stress_linear_fem[:, 0:1], stress_linear_fem[:, 1:2], stress_linear_fem[:, 3:4], X)  
# data.point_data['FEM_linear_Cauchy_stress_polar'] = np.column_stack((T_rr_fe_linear, T_theta_fe_linear, T_rtheta_fe_linear))
# r_mesh = r_mesh.reshape((-1,1))

# New part

# analytical solution

inner_radius = 1
outer_radius = 2

nu, lame, shear, e_modul = problem_parameters()
    
r = np.sqrt(X[:,0:1]**2 + X[:,1:2]**2)
disp_pred_based_on_fe_mesh = model.predict(X[:, 0:2])       # is the same as: predictions = model.predict(X[:, 0:2])
# theta = np.linspace(0, np.pi/2, 100)
# r_mesh, theta_mesh = np.meshgrid(r, theta)
# r_mesh = r_mesh.reshape(-1)
# theta_mesh = theta_mesh.reshape((-1,1))
# y = np.zeros(r.shape[0])

dr2 = (outer_radius**2 - inner_radius**2)

# Analytical stress and displacement
T_rr_analytical = inner_radius**2 * pressure_inlet / dr2 * (r**2 - outer_radius**2) / r**2
T_theta_analytical = inner_radius**2 * pressure_inlet / dr2 * (r**2 + outer_radius**2) / r**2
u_rad_analytical = inner_radius**2 * pressure_inlet * r / (e_modul * (outer_radius**2 - inner_radius**2)) * (1 - nu + (outer_radius / r)**2 * (1 + nu))

# Predict the outputs using the model
predictions = model.predict(X[:, 0:2])  # Input the spatial points (x, y)

#calculate arc length for each point
arc_length = r
data.point_data['arc_length'] = arc_length

# Extract the Cauchy stresses (P_xx, P_yy, P_xy, P_yx) from the last four columns
P_xx, P_yy, P_xy, P_yx = predictions[:, 2], predictions[:, 3], predictions[:, 4], predictions[:, 5]

# Optionally, calculate Cauchy stress using the given operator
T_xx, T_yy, T_xy, T_yx = model.predict(X[:, 0:2], operator=cauchy_stress_mixed_P)

displacement = predictions[:, :2]

first_piola = np.column_stack((P_xx, P_yy, P_xy, P_yx))
cauchy = np.column_stack((T_xx, T_yy, T_xy))

displacement_extended = np.hstack((displacement, np.zeros_like(displacement[:,0:1])))

data.point_data['pred_first_piola'] = first_piola
data.point_data['pred_displacement'] = displacement_extended
data.point_data['pred_stress'] = cauchy

disp_fem = data.point_data['displacement']

error_disp = abs((disp_fem - displacement[:, 0:2]))

data.point_data['pointwise_displacement_error'] = error_disp

disp_fem_polar = np.sqrt(disp_fem[:,0:1]**2 + disp_fem[:,1:2]**2)
disp_pred_polar = np.sqrt(displacement[:,0:1]**2 + displacement[:,1:2]**2)

data.point_data['disp_fem_nonlinear_polar'] = disp_fem_polar
data.point_data['disp_pred_polar'] = disp_pred_polar

T_rr, T_theta, T_rtheta = polar_transformation_2d(T_xx, T_yy, T_xy, X)   
data.point_data['PINN_Cauchy_stress_polar'] = np.column_stack((T_rr, T_theta, T_rtheta))


# select xx, yy, and xy component (1st, 2nd and 4th column)
stress_fem = data.point_data['nodal_cauchy_stresses_xyz']
columns = [0,1,3]

T_rr_fe, T_theta_fe, T_rtheta_fe = polar_transformation_2d(stress_fem[:, 0:1], stress_fem[:, 1:2], stress_fem[:, 3:4], X)  
data.point_data['FEM_Cauchy_stress_polar'] = np.column_stack((T_rr_fe, T_theta_fe, T_rtheta_fe))

error_stress = abs((stress_fem[:, columns] - cauchy))


error_cauchy_stress_polar_rr = abs(T_rr - T_rr_fe)
error_cauchy_stress_polar_theta = abs(T_theta_fe - T_theta)
error_cauchy_stress_polar_rtheta = abs(T_rtheta_fe - T_rtheta)

data.point_data['PINN_Cauchy_stress_polar_error'] = np.column_stack((error_cauchy_stress_polar_rr, error_cauchy_stress_polar_theta, error_cauchy_stress_polar_rtheta))

# relative L2-error cartesian PINN(nonlinear) relative to nonlinear FE solution

relative_l2_error_stress_xx = compute_relative_l2_error(stress_fem[:, columns], cauchy, 0)
relative_l2_error_stress_yy = compute_relative_l2_error(stress_fem[:, columns], cauchy, 1)
relative_l2_error_stress_xy = compute_relative_l2_error(stress_fem[:, columns], cauchy, 2)
relative_l2_error_disp_x = compute_relative_l2_error(disp_fem, displacement, 0)
relative_l2_error_disp_y = compute_relative_l2_error(disp_fem, displacement, 1)

print(f"Errors for cartesian comparison PINN({model_complexity}) to FE({model_complexity})")
print(f"relative_l2_error_disp_x:{relative_l2_error_disp_x}")
print(f"relative_l2_error_disp_y:{relative_l2_error_disp_y}")
print(f"relative_l2_error_stress_xx:{relative_l2_error_stress_xx}")
print(f"relative_l2_error_stress_yy:{relative_l2_error_stress_yy}")
print(f"relative_l2_error_stress_xy:{relative_l2_error_stress_xy}")


if elasticity_utils.model_complexity == "linear":
    # relative L2-error polar PINN(linear) relative to analytical solution
    print("Errors for polar comparison PINN(linear) to analytical")
    
    relative_l2_error_disp_polar = compute_relative_l2_error(u_rad_analytical, disp_pred_polar, 0)
    relative_l2_error_stress_polar_rr_pred = compute_relative_l2_error(T_rr_analytical, T_rr, 0)
    relative_l2_error_stress_polar_theta_pred = compute_relative_l2_error(T_theta_analytical, T_theta, 0)

    print(f"relative_l2_error_disp_polar:{relative_l2_error_disp_polar}")
    print(f"relative_l2_error_stress_polar_rr_pred:{relative_l2_error_stress_polar_rr_pred}")
    print(f"relative_l2_error_stress_polar_theta_pred:{relative_l2_error_stress_polar_theta_pred}")
    
    # print("Errors for polar comparison FE(linear) to analytical")
    
    # relative_l2_error_stress_polar_theta_fe = compute_relative_l2_error(T_theta_analytical, T_theta_fe, 0)

    # print(f"relative_l2_error_disp_polar_fe:{relative_l2_error_disp_polar_fe}")
    # print(f"relative_l2_error_stress_polar_rr_fe:{relative_l2_error_stress_polar_rr_fe}")
    # print(f"relative_l2_error_stress_polar_theta_fe:{relative_l2_error_stress_polar_theta_fe}")
    

elif elasticity_utils.model_complexity == "nonlinear":
    # relative L2-error polar PINN(nonlinear) relative to FE(nonlinear)
    print("Errors for polar comparison PINN(nonlinear) to FE(nonlinear)")
    
    relative_l2_error_disp_polar_nonlinear = compute_relative_l2_error(disp_fem_polar, disp_pred_polar, 0)
    relative_l2_error_stress_polar_rr_nonlinear = compute_relative_l2_error(T_rr_fe, T_rr, 0)
    relative_l2_error_stress_polar_theta_nonlinear = compute_relative_l2_error(T_theta_fe, T_theta, 0)

    print(f"relative_l2_error_disp_polar_nonlinear:{relative_l2_error_disp_polar_nonlinear}")
    print(f"relative_l2_error_stress_polar_rr_nonlinear:{relative_l2_error_stress_polar_rr_nonlinear}")
    print(f"relative_l2_error_stress_polar_theta_nonlinear:{relative_l2_error_stress_polar_theta_nonlinear}")
else: 
    print("Cannot find right L2-Error formula")


# # relative L2-error polar FE(linear) relative to analytical solution

# relative_l2_error_disp_polar_linear_fem = compute_relative_l2_error(u_rad_analytical, disp_linear_fem_polar, 0)
# relative_l2_error_stress_polar_rr_linear_fem = compute_relative_l2_error(T_rr_analytical, T_rr_fe_linear, 0)
# relative_l2_error_stress_polar_theta_linear_fem = compute_relative_l2_error(T_theta_analytical, T_theta_fe_linear, 0)

# print(f"relative_l2_error_disp_x:{relative_l2_error_disp_polar_linear_fem}")
# print(f"relative_l2_error_disp_y:{relative_l2_error_stress_polar_rr_linear_fem}")
# print(f"relative_l2_error_disp_y:{relative_l2_error_stress_polar_theta_linear_fem}")

data.point_data['pointwise_cauchystress_error'] = error_stress
#data.point_data['pointwise_cauchystress_error'].column_names

data.save(f"Lame2D_mixed_P_{model_complexity}_{activation}_{model_type}.vtu")

print("NOTE THAT 'Warp By Vector' DOES NOT WORK HERE AS THE Z-DIMENSION VALUES ARE ILL-DEFINED.")
print("USE CALCULATION WITH 'displacement_X*iHat + displacement_Y*jHat + 0*kHat' AND THEN APPLY 'Warp By Vector'.")

exit()

def compareModelPredictionAndAnalyticalSolution(model):
    '''
    This function plots analytical solutions and the predictions, 
    '''
        
    nu, lame, shear, e_modul = problem_parameters()
    
    r = np.linspace(radius_inner, radius_outer, 100)
    y = np.zeros(r.shape[0])

    dr2 = (radius_outer**2 - radius_inner**2)

    # Analytical stress and displacement
    sigma_rr_analytical = radius_inner**2 * pressure_inlet / dr2 * (r**2 - radius_outer**2) / r**2
    sigma_theta_analytical = radius_inner**2 * pressure_inlet / dr2 * (r**2 + radius_outer**2) / r**2
    u_rad = radius_inner**2 * pressure_inlet * r / (e_modul * (radius_outer**2 - radius_inner**2)) * (1 - nu + (radius_outer / r)**2 * (1 + nu))

    # Model predictions
    r_x = np.hstack((r.reshape(-1, 1), y.reshape(-1, 1)))
    disps = model.predict(r_x)
    u_pred, v_pred = disps[:, 0:1], disps[:, 1:2]
    u_rad_pred = np.sqrt(u_pred**2 + v_pred**2)
    T_xx, T_yy, T_xy, T_yx = model.predict(r_x, operator=cauchy_stress)
    sigma_rr, sigma_theta, sigma_rtheta = polar_transformation_2d(T_xx, T_yy, T_xy, r_x)
    
    # FE solution
    data_fe = pd.read_excel("/home_student/kappen/Downloads/output_num_example_new_pressure_definition.ods", engine="odf")
    data_pinn = pd.read_excel("/home_student/kappen/Downloads/PINN_lame_solution_coordinates_over_paraview_set_5_mesh.ods", engine="odf")
    
    # print(data_fe.head())  # Print the first few rows to check the data structure
    # print("Number of columns:", data_fe.shape[1])  # Check the number of columns
    
    u_rad_pred_fe = data_fe.iloc[:, 8]      #displacement_magnitude in direction r from FE
    u_rad_pred_pinn = data_pinn.iloc[:, 25]  #displacement_magnitude in direct r from PINNs (derived from paraview)
    
    
    # Create figure and subplots for stress, displacement, and strain
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Plot stress comparison
    axs[0].plot(r / radius_inner, sigma_rr_analytical / radius_inner, label=r"Analytical $\sigma_{r}$")
    axs[0].plot(r / radius_inner, sigma_rr / radius_inner, label=r"Predicted $\sigma_{r}$")
    axs[0].plot(r / radius_inner, sigma_theta_analytical / radius_inner, label=r"Analytical $\sigma_{\theta}$")
    axs[0].plot(r / radius_inner, sigma_theta / radius_inner, label=r"Predicted $\sigma_{\theta}$")
    axs[0].set(ylabel="Normalized stress", xlabel="r/a")
    axs[0].legend()
    axs[0].grid()

    # Plot displacement comparison
    axs[1].plot(r / radius_inner, u_rad / radius_inner, label=r"Analytical $u_r$")
    axs[1].plot(r / radius_inner, u_rad_pred / radius_inner, label=r"Predicted $u_r$")
    axs[1].set(ylabel="Normalized radial displacement", xlabel="r/a")
    axs[1].legend()
    axs[1].grid()

    r_2 = np.linspace(radius_inner, radius_outer, 1001)
    y = np.zeros(r_2.shape[0])

    axs[2].plot(r_2 / radius_inner, u_rad_pred_fe / radius_inner, label=r"Predicted $u_r$ FE")
    axs[2].plot(r_2 / radius_inner, u_rad_pred_pinn / radius_inner, label=r"Predicted $u_r$ PINN")
    axs[2].set(ylabel="Normalized radial displacement_plot over line", xlabel="r/a")
    axs[2].legend()
    axs[2].grid()

    fig.tight_layout()
    plt.savefig("Lame_quarter_gmsh_nicht_linear_with_strain_convex_mesh")
    plt.show()

fem_path = str(Path(__file__).parent.parent.parent.parent.parent)+"/Comparison_FE_to_PINN_in_paraview/Lame/Set_5_PINN_with_FEM_mesh/small_grid_fem_spreadsheet.csv"
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

triangle_coordinate_x = x[triangles.triangles]
triangle_coordinate_y = y[triangles.triangles]

# np.isclose(np.linalg.norm(x - center_outer, axis=-1), radius_outer)
calculate_norm = np.sqrt(triangle_coordinate_x**2+triangle_coordinate_y**2)
mask = np.isclose(calculate_norm,1)
dol_triangles = triangles.triangles[~mask.all(axis=1)]

# Calculate centroid radius of each triangle
triangle_centroids_x = x[triangles.triangles].mean(axis=1)
triangle_centroids_y = y[triangles.triangles].mean(axis=1)
centroid_radii = np.sqrt(triangle_centroids_x**2 + triangle_centroids_y**2)

# Define the inner and outer radii of your quarter-disc
inner_radius, outer_radius = 1.0, 2.0  # Adjust based on your domain's geometry

# Create a mask for triangles outside the radial bounds or in unwanted regions
mask = (centroid_radii < inner_radius) | (centroid_radii > outer_radius) | (triangle_centroids_x < 0) | (triangle_centroids_y < 0)
triangles.set_mask(mask)

# fem
u_fem, v_fem = displacement_fem[:,0], displacement_fem[:,1]
sigma_xx_fem, sigma_yy_fem, sigma_xy_fem = stress_fem[:,0:1], stress_fem[:,1:2], stress_fem[:,2:3]
sigma_rr_fem, sigma_theta_fem, sigma_rtheta_fem = polar_transformation_2d(sigma_xx_fem, sigma_yy_fem, sigma_xy_fem, X)

combined_disp_fem = tuple(np.vstack((np.array(u_fem.tolist()),np.array(v_fem.tolist()),np.zeros(u_fem.shape[0]))))
combined_stress_fem = tuple(np.vstack((np.array(sigma_xx_fem.flatten().tolist()),np.array(sigma_yy_fem.flatten().tolist()),np.array(sigma_xy_fem.flatten().tolist()))))
combined_stress_polar_fem = tuple(np.vstack((np.array(sigma_rr_fem.tolist()),np.array(sigma_theta_fem.tolist()),np.array(sigma_rtheta_fem.tolist()))))

displacement = model.predict(X)
u_pred, v_pred = displacement[:,0], displacement[:,1]
T_xx, T_yy, T_xy, T_yx = model.predict(X, operator=cauchy_stress)                                   # Original output sigma_xx, sigma_yy, sigma_xy
sigma_rr, sigma_theta, sigma_rtheta = polar_transformation_2d(T_xx, T_yy, T_xy, X)                  # sigma_rr, sigma_theta, sigma_rtheta left unchanged

p_xx, p_yy, p_xy, p_yx = model.predict(X, operator=first_piola_stress_tensor) 

combined_disp = tuple(np.vstack((np.array(displacement[:,0].tolist()),np.array(displacement[:,1].tolist()),np.zeros(displacement[:,0].shape[0]))))
combined_stress = tuple(np.vstack((np.array(T_xx.flatten().tolist()),np.array(T_yy.flatten().tolist()),np.array(T_xy.flatten().tolist()))))     # Original output sigma_xx, sigma_yy, sigma_xy
combined_stress_polar = tuple(np.vstack((np.array(sigma_rr.tolist()),np.array(sigma_theta.tolist()),np.array(sigma_rtheta.tolist()))))
combined_stress_p = tuple(np.vstack((np.array(p_xx.flatten().tolist()),np.array(p_yy.flatten().tolist()),np.array(p_xy.flatten().tolist()))))

# error
error_disp_x = abs(np.array(u_pred.tolist()) - u_fem.flatten())
error_disp_y =  abs(np.array(v_pred.tolist()) - v_fem.flatten())
combined_error_disp = tuple(np.vstack((error_disp_x, error_disp_y,np.zeros(error_disp_x.shape[0]))))

error_stress_x = abs(np.array(T_xx.flatten().tolist()) - sigma_xx_fem.flatten())
error_stress_y =  abs(np.array(T_yy.flatten().tolist()) - sigma_yy_fem.flatten())
error_stress_xy =  abs(np.array(T_xy.flatten().tolist()) - sigma_xy_fem.flatten())
combined_error_stress = tuple(np.vstack((error_stress_x, error_stress_y, error_stress_xy)))

error_polar_stress_x = abs(np.array(sigma_rr.flatten().tolist()) - sigma_rr_fem.flatten())
error_polar_stress_y =  abs(np.array(sigma_theta.flatten().tolist()) - sigma_theta_fem.flatten())
error_polar_stress_xy =  abs(np.array(sigma_rtheta.flatten().tolist()) - sigma_rtheta_fem.flatten())
combined_error_polar_stress = tuple(np.vstack((error_polar_stress_x, error_polar_stress_y, error_polar_stress_xy)))


file_path = os.path.join(os.getcwd(), "Lame_quarter_gmsh_nicht_linear_convex_mesh_P_1")

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)

#dol_triangles = dol_triangles
offset = np.arange(3,dol_triangles.shape[0]*dol_triangles.shape[1]+1,dol_triangles.shape[1]).astype(dol_triangles.dtype)
cell_types = np.ones(dol_triangles.shape[0])*5

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                    cell_types, pointData = { "displacement" : combined_disp, "displacement_fem" : combined_disp_fem, "stress" : combined_stress, "stress_polar": combined_stress_polar, "stress_fem": combined_stress_fem, "stress_polar_fem": combined_stress_polar_fem, "1st piola stress": combined_stress_p, "error_disp_x": error_disp_x, "error_disp_y": error_disp_y, "combined_error_disp": combined_error_disp, 
                                             "error_stress_x": error_stress_x, "error_stress_y": error_stress_y, "error_stress_xy": error_stress_xy, "combined_error_stress": combined_error_stress, 
                                             "error_polar_stress_x": error_polar_stress_x, "error_polar_stress_y": error_polar_stress_y, "error_polar_stress_xy": error_polar_stress_xy, "combined_error_polar_stress": combined_error_polar_stress})

compareModelPredictionAndAnalyticalSolution(model)