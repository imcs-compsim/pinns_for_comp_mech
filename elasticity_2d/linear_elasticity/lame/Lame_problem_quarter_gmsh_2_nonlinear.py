import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
from pyevtk.hl import unstructuredGridToVTK
from pathlib import Path
from deepxde import backend as bkd
import pandas as pd

from utils.elasticity.elasticity_utils import first_piola_stress_tensor, momentum_2d_firstpiola, problem_parameters, piola_stress_to_traction_2d, zero_neumman_first_piola_x, zero_neumman_first_piola_y, cauchy_stress, green_lagrange_strain_tensor_2, green_lagrange_strain_tensor_1
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


gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
quarter_circle_with_hole = QuarterCirclewithHole(center=[0,0,0], inner_radius=1, outer_radius=2, mesh_size=0.1, gmsh_options=gmsh_options)

gmsh_model = quarter_circle_with_hole.generateGmshModel(visualize_mesh=False)

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
elasticity_utils.lame = 150/3     #75/26 -->E-Modul=5    75/13 -->E-Modul=10    150/3 -->E-Modul=20       1500/13 -->E-Modul=200    1153.846  -->E-Modul= 2000   1575000/13   --> E-Modul=210000
elasticity_utils.shear = 100/3    #25/13                  50/3                   100/3                     1000/3                    769.23                       1050000/13
nu,lame,shear,e_modul = problem_parameters()

# The applied pressure 
pressure_inlet = 1

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
#bc3 = dde.DirichletBC(geom, lambda _: 0.0, boundary_left, component=0)
#bc4 = dde.DirichletBC(geom, lambda _: 0.0, boundary_bottom, component=1)
bc5 = dde.OperatorBC(geom, zero_neumman_first_piola_x, boundary_outer)
bc6 = dde.OperatorBC(geom, zero_neumman_first_piola_y, boundary_outer)

n_dummy = 1
data = dde.data.PDE(
    geom,
    momentum_2d_firstpiola,
    [bc1, bc2, bc5, bc6],       #DBC are now hard constraints
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
    
    return bkd.concat([u*(x_loc)/e_modul,v*(y_loc)/e_modul], axis=1)

# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model_path = str(Path(__file__).parent.parent.parent)+"/trained_models/lame/lame"
n_epochs = 3106 # trained model has 3106 iterations
model_restore_path = model_path + "-"+ str(n_epochs) + ".ckpt"

model = dde.Model(data, net)
# if we want to save the model, we use "model_save_path=model_path" during training, if we want to load trained model, we use "model_restore_path=return_restore_path(model_path, num_epochs)"
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=4000, display_every=200, model_restore_path=None)

model.compile("L-BFGS")
model.train(model_save_path=model_path, display_every=200)

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

# def compareModelPredictionAndAnalyticalSolution(model):
#     '''
#     This function plots analytical solutions and the predictions. 
#     '''

#     nu,lame,shear,e_modul = problem_parameters()
    
#     r = np.linspace(radius_inner, radius_outer,100)
#     y = np.zeros(r.shape[0])

#     dr2 = (radius_outer**2 - radius_inner**2)

#     sigma_rr_analytical = radius_inner**2*pressure_inlet/dr2*(r**2-radius_outer**2)/r**2
#     sigma_theta_analytical = radius_inner**2*pressure_inlet/dr2*(r**2+radius_outer**2)/r**2
#     u_rad = radius_inner**2*pressure_inlet*r/(e_modul*(radius_outer**2-radius_inner**2))*(1-nu+(radius_outer/r)**2*(1+nu))

#     r_x = np.hstack((r.reshape(-1,1),y.reshape(-1,1)))
#     disps = model.predict(r_x)
#     u_pred, v_pred = disps[:,0:1], disps[:,1:2]
#     u_rad_pred = np.sqrt(u_pred**2+v_pred**2)
#     T_xx, T_yy, T_xy, T_yx  = model.predict(r_x, operator=cauchy_stress)                          # Original output sigma_xx, sigma_yy, sigma_xy
#     sigma_rr, sigma_theta, sigma_rtheta = polar_transformation_2d(T_xx, T_yy, T_xy, r_x)          # sigma_rr, sigma_theta, sigma_rtheta left unchanged



#     fig, axs = plt.subplots(1,2,figsize=(12,5))

#     axs[0].plot(r/radius_inner, sigma_rr_analytical/radius_inner, label = r"Analytical $\sigma_{r}$")
#     axs[0].plot(r/radius_inner, sigma_rr/radius_inner, label = r"Predicted $\sigma_{r}$")
#     axs[0].plot(r/radius_inner, sigma_theta_analytical/radius_inner, label = r"Analytical $\sigma_{\theta}$")
#     axs[0].plot(r/radius_inner, sigma_theta/radius_inner, label = r"Predicted $\sigma_{\theta}$")
#     axs[0].set(ylabel="Normalized stress", xlabel = "r/a")
#     axs[1].plot(r/radius_inner, u_rad/radius_inner, label = r"Analytical $u_r$")
#     axs[1].plot(r/radius_inner, u_rad_pred/radius_inner, label = r"Predicted $u_r$")
#     axs[1].set(ylabel="Normalized radial displacement", xlabel = "r/a")
#     axs[0].legend()
#     axs[0].grid()
#     axs[1].legend()
#     axs[1].grid()
#     fig.tight_layout()

#     plt.savefig("Lame_quarter_gmsh_nicht_linear")
#     plt.show()

def compareModelPredictionAndAnalyticalSolution(model):
    '''
    This function plots analytical solutions and the predictions, 
    as well as a comparison between type 1 and type 2 strains in the formulation see elasticity_utils.py.
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
    data_fe = pd.read_excel("/home_student/kappen/Downloads/FE_lame_solution_coordinates_over_paraview_new.ods", engine="odf")
    data_pinn = pd.read_excel("/home_student/kappen/Downloads/PINN_lame_solution_coordinates_over_paraview_new.ods", engine="odf")
    
    print(data_fe.head())  # Print the first few rows to check the data structure
    print("Number of columns:", data_fe.shape[1])  # Check the number of columns
    
    u_rad_pred_fe = data_fe.iloc[:, 8]      #displacement_magnitude in direction r from FE
    u_rad_pred_pinn = data_pinn.iloc[:, 9]  #displacement_magnitude in direct r from PINNs (derived from paraview)
    
    
    # Create figure and subplots for stress, displacement, and strain
    fig, axs = plt.subplots(1, 5, figsize=(18, 5))

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

    # Polarization of green_lagrange_tensor      
    e2_xx, e2_yy, e2_xy, e2_yx = model.predict(r_x, operator=green_lagrange_strain_tensor_2)
    e1_xx, e1_yy, e1_xy, e1_yx = model.predict(r_x, operator=green_lagrange_strain_tensor_1)
    
    e2_rr, e2_theta, e2_rtheta = polar_transformation_2d(e2_xx, e2_yy, e2_xy, r_x)
    e1_rr, e1_theta, e1_rtheta = polar_transformation_2d(e1_xx, e1_yy, e1_xy, r_x)

    # print("e1_rr:", e1_rr)
    # print("e2_rr:", e2_rr)

    # print("e1_theta:", e1_rr)
    # print("e2_theta:", e2_theta)

    # Plot strain comparison
    scale_factor_rr = 1
    # Plot Strain_rr with scaling
    axs[2].plot(r / radius_inner, e2_rr * scale_factor_rr / radius_inner, label=r"Type 2 Strain_rr")
    axs[2].plot(r / radius_inner, e1_rr * scale_factor_rr / radius_inner, label=r"Type 1 Strain_rr")
    axs[2].set(ylabel=f"Strain_rr (x{scale_factor_rr:.0e})", xlabel="r/a")
    axs[2].set_yscale('linear')
    axs[2].legend()
    axs[2].grid()

    scale_factor_theta = 1
    # Plot Strain_theta with scaling
    axs[3].plot(r / radius_inner, e2_theta * scale_factor_theta / radius_inner, label=r"Type 2 Strain_theta")
    axs[3].plot(r / radius_inner, e1_theta * scale_factor_theta / radius_inner, label=r"Type 1 Strain_theta")
    axs[3].set(ylabel=f"Strain_theta (x{scale_factor_theta:.0e})", xlabel="r/a")
    axs[3].set_yscale('linear')
    axs[3].legend()
    axs[3].grid()

    r_2 = np.linspace(radius_inner, radius_outer, 1001)
    y = np.zeros(r_2.shape[0])

    axs[4].plot(r_2 / radius_inner, u_rad_pred_fe / radius_inner, label=r"Predicted $u_r$ FE")
    axs[4].plot(r_2 / radius_inner, u_rad_pred_pinn / radius_inner, label=r"Predicted $u_r$ PINN")
    axs[4].set(ylabel="Normalized radial displacement_plot over line", xlabel="r/a")
    axs[4].legend()
    axs[4].grid()


    # # Plot strain comparison with y-axis scaling adjustment
    # axs[2].plot(r / radius_inner, e2_rr / radius_inner, label="Type 2 Strain_rr")
    # axs[2].plot(r / radius_inner, e1_rr / radius_inner, label="Type 1 Strain_rr")
    # axs[2].plot(r / radius_inner, e2_theta / radius_inner, label="Type 2 Strain_theta")
    # axs[2].plot(r / radius_inner, e1_theta / radius_inner, label="Type 1 Strain_theta")
    # axs[2].set(ylabel="Strain", xlabel="r/a")
    # axs[2].set_yscale('linear')  # Change to 'log' if strain values differ greatly in magnitude
    # axs[2].legend()
    # axs[2].grid()

    fig.tight_layout()
    plt.savefig("Lame_quarter_gmsh_nicht_linear_with_strain")
    plt.show()


X, offset, cell_types, dol_triangles = geom.get_mesh()

displacement = model.predict(X)
T_xx, T_yy, T_xy, T_yx = model.predict(X, operator=cauchy_stress)                                   # Original output sigma_xx, sigma_yy, sigma_xy
sigma_rr, sigma_theta, sigma_rtheta = polar_transformation_2d(T_xx, T_yy, T_xy, X)                  # sigma_rr, sigma_theta, sigma_rtheta left unchanged

e2_xx, e2_yy, e2_xy, e2_yx = model.predict(X, operator=green_lagrange_strain_tensor_2)
e1_xx, e1_yy, e1_xy, e1_yx = model.predict(X, operator=green_lagrange_strain_tensor_1)
e2_rr, e2_theta, e2_rtheta = polar_transformation_2d(e2_xx, e2_yy, e2_xy, X)
e1_rr, e1_theta, e1_rtheta = polar_transformation_2d(e1_xx, e1_yy, e1_xy, X)

combined_disp = tuple(np.vstack((np.array(displacement[:,0].tolist()),np.array(displacement[:,1].tolist()),np.zeros(displacement[:,0].shape[0]))))
combined_stress = tuple(np.vstack((np.array(T_xx.flatten().tolist()),np.array(T_yy.flatten().tolist()),np.array(T_xy.flatten().tolist()))))     # Original output sigma_xx, sigma_yy, sigma_xy
combined_stress_polar = tuple(np.vstack((np.array(sigma_rr.tolist()),np.array(sigma_theta.tolist()),np.array(sigma_rtheta.tolist()))))
# type_2_strain_polar = tuple(np.vstack((np.array(e2_rr.tolist()),np.array(e2_theta.tolist()),np.array(e2_rtheta.tolist()))))
# type_1_strain_polar = tuple(np.vstack((np.array(e1_rr.tolist()),np.array(e1_theta.tolist()),np.array(e1_rtheta.tolist()))))
type_2_strain = tuple(np.vstack((np.array(e2_xx.flatten().tolist()),np.array(e2_yy.flatten().tolist()),np.array(e2_xy.flatten().tolist()))))
type_1_strain = tuple(np.vstack((np.array(e1_xx.flatten().tolist()),np.array(e1_yy.flatten().tolist()),np.array(e1_xy.flatten().tolist()))))


file_path = os.path.join(os.getcwd(), "Lame_quarter_gmsh_nicht_linear")

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)

# unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
#                       cell_types, pointData = { "displacement" : combined_disp,"stress" : combined_stress, "stress_polar": combined_stress_polar, "type_2_strain_polar": type_2_strain_polar, "type_1_strain_polar": type_1_strain_polar})


unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                      cell_types, pointData = { "displacement" : combined_disp,"stress" : combined_stress, "stress_polar": combined_stress_polar, "type_2_strain": type_2_strain, "type_1_strain": type_1_strain})

compareModelPredictionAndAnalyticalSolution(model)





