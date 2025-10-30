# import helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#import time
from pathlib import Path
from matplotlib import ticker
import matplotlib.lines as mlines

# set plot options
mpl.rcParams['mathtext.fontset'] = 'stix'
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))

# import deepxde libraries
from deepxde import backend as bkd
import deepxde as dde
from deepxde import config


# import computational mechanics libraries
from compsim_pinns.geometry.custom_geometry import GmshGeometry3D
from compsim_pinns.geometry.gmsh_models import Geom_step_to_gmsh
from compsim_pinns.elasticity import elasticity_utils
from compsim_pinns.elasticity.elasticity_utils import pde_mixed_3d_time, get_tractions_mixed_3d_spacetime, problem_parameters
from compsim_pinns.postprocess.elasticity_postprocessing import solutionFieldOnMeshToVtkSpaceTime
from compsim_pinns.postprocess.save_normals_tangentials_to_vtk import export_normals_tangentials_to_vtk
from compsim_pinns.geometry.geometry_utils import polar_transformation_3d_spherical, calculate_boundary_normals_3D
from compsim_pinns.geometry.geometry_time import ModifiedGeometryXTime

'''
@author: tsahin

3D lame problem with time.  

D. Schillinger, J. A. Evans, F. Frischmann, R. R. Hiemstra, M.-C. Hsu, T. J. Hughes, A
collocated c0 finite element method: Reduced quadrature perspective, cost comparison
with standard finite elements, and explicit structural dynamics, International Journal
for Numerical Methods in Engineering 102 (3-4) (2015) 576–631.
'''

path_to_step_file = str(Path(__file__).parent.parent.parent)+f"/step_files/lame_3d.stp"

curve_info = {"4":14, "8":14, "10":14,
              "6":18, "7":18, "2":18,}
Lame_geom_obj = Geom_step_to_gmsh(path=path_to_step_file, curve_info=curve_info)

gmsh_model = Lame_geom_obj.generateGmshModel(visualize_mesh=False)

# Target surface IDs for each set. 
# This means if any surface is neighbor of any of following surfaces, then repeated nodes will take normals+tangenitals of the target surface 
target_surface_ids = [2, 4]

geom = GmshGeometry3D(gmsh_model, target_surface_ids=target_surface_ids)
# This allows for visualization of boundary normals in Paraview
export_normals_tangentials_to_vtk(geom, save_folder_path=str(Path(__file__).parent.parent.parent.parent), file_name="lame_boundary_normals")

radius_inner = 1
radius_outer = 2
center = [0,0,0]

# The applied pressure 
pressure = 1
t_min = 0
t_max = 1

# change global variables in elasticity_utils
elasticity_utils.lame = 1153.846
elasticity_utils.shear = 769.23
nu,lame,shear,e_modul = problem_parameters()

#spaceDomain = dde.geometry.Rectangle([x_min, y_min], [x_max, y_max])
timeDomain = dde.geometry.TimeDomain(t_min, t_max)
spaceTimeDomain = ModifiedGeometryXTime(geom, timeDomain)

elasticity_utils.geom = geom
elasticity_utils.spacetime_domain = spaceTimeDomain

# Front surface
def boundary_initial(x, on_boundary):
    time_dimension = x[3]
    return np.isclose(time_dimension,0)

def boundary_outer(x, on_boundary):
    space_coords = x[:3]
    return on_boundary and np.isclose(np.linalg.norm(space_coords - center, axis=-1), radius_outer)

def boundary_inner(x, on_boundary):
    space_coords = x[:3]
    return on_boundary and np.isclose(np.linalg.norm(space_coords - center, axis=-1), radius_inner) #and cond_on_edge

def apply_presure(x,y,X):
    Tx, Ty, Tz, Pn, Tt_1, Tt_2 = get_tractions_mixed_3d_spacetime(x, y, X)
    _, _, _, cond = calculate_boundary_normals_3D(X,spaceTimeDomain)
    t_loc = x[:, 3:4][cond] # time only for boundary points
    
    return Pn + pressure*(bkd.sin(2*np.pi*t_loc)) 

def zero_neumann_in_normal(x,y,X):
    Tx, Ty, Tz, Pn, Tt_1, Tt_2 = get_tractions_mixed_3d_spacetime(x, y, X)
    
    return Pn 

def zero_neumann_in_tangential1(x,y,X):
    Tx, Ty, Tz, Pn, Tt_1, Tt_2 = get_tractions_mixed_3d_spacetime(x, y, X)
    
    return Tt_1

def zero_neumann_in_tangential2(x,y,X):
    Tx, Ty, Tz, Pn, Tt_1, Tt_2 = get_tractions_mixed_3d_spacetime(x, y, X)
    
    return Tt_2

# Initial BC for velocity component x
def apply_velocity_in_x(x,y,X):
    du_x_t = dde.grad.jacobian(y, x, i=0, j=3)# i=0 represents u_x, j=2 is time
    return du_x_t

# Initial BC for velocity component y
def apply_velocity_in_y(x,y,X):
    du_y_t = dde.grad.jacobian(y, x, i=1, j=3) # i=1 represents u_y, j=2 is time
    return du_y_t

# Initial BC for velocity component y
def apply_velocity_in_z(x,y,X):
    du_z_t = dde.grad.jacobian(y, x, i=2, j=3) # i=1 represents u_y, j=2 is time
    return du_z_t

# zero Neumann at R=2 (outer boundary)
bc1 = dde.OperatorBC(spaceTimeDomain, zero_neumann_in_normal, boundary_outer)
bc2 = dde.OperatorBC(spaceTimeDomain, zero_neumann_in_tangential1, boundary_outer)
bc3 = dde.OperatorBC(spaceTimeDomain, zero_neumann_in_tangential2, boundary_outer)

# nonzero and zero Neumann at R=1 (inner boundary)
bc4 = dde.OperatorBC(spaceTimeDomain, apply_presure, boundary_inner)
bc5 = dde.OperatorBC(spaceTimeDomain, zero_neumann_in_tangential1, boundary_inner)
bc6 = dde.OperatorBC(spaceTimeDomain, zero_neumann_in_tangential2, boundary_inner)

# Initial BCs for velocities
ic_velocity_in_x = dde.icbc.OperatorBC(spaceTimeDomain, apply_velocity_in_x, boundary_initial)
ic_velocity_in_y = dde.icbc.OperatorBC(spaceTimeDomain, apply_velocity_in_y, boundary_initial) 
ic_velocity_in_z = dde.icbc.OperatorBC(spaceTimeDomain, apply_velocity_in_z, boundary_initial)   

bcs = [bc1, bc2, bc3, bc4, bc5, bc6, ic_velocity_in_x, ic_velocity_in_y, ic_velocity_in_z]

data = dde.data.TimePDE(
    spaceTimeDomain,
    pde_mixed_3d_time,
    bcs,
    num_domain=geom.random_points(1).shape[0],#spaceDomain.random_points(1).shape[0]
    num_boundary=geom.random_boundary_points(1).shape[0],#spaceDomain.random_boundary_points(1).shape[0]
    num_initial=100,
    num_test=None
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
    t_loc = x[:, 3:4]
    
    # define surfaces
    x_0_surface = x_loc
    y_0_surface = y_loc
    z_0_surface = z_loc
    
    # define the surfaces where shear forces will be applied.
    sigma_xy_surfaces = (x_0_surface)*(y_0_surface)
    sigma_yz_surfaces = (y_0_surface)*(z_0_surface)
    sigma_xz_surfaces = (x_0_surface)*(z_0_surface)
    
    
    return bkd.concat([u*(x_0_surface)*t_loc/e_modul, #displacement in x direction is 0 at x=0
                      v*(y_0_surface)*t_loc/e_modul, #displacement in y direction is 0 at y=0
                      w*(z_0_surface)*t_loc/e_modul, #displacement in z direction is 0 at z=0
                      sigma_xx, 
                      sigma_yy,
                      sigma_zz,
                      sigma_xy*sigma_xy_surfaces,
                      sigma_yz*sigma_yz_surfaces,
                      sigma_xz*sigma_xz_surfaces
                      ], axis=1)

# 3 inputs, 9 outputs for 3D 
layer_size = [4] + [50] * 5 + [9]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)

restore_model = True
model_path = str(Path(__file__).parent.parent.parent)+f"/trained_models/lame_time/lame_3d_time"

if not restore_model:
    model.compile("adam", lr=0.001)
    losshistory, train_state = model.train(epochs=2000, display_every=100) 
    # losshistory, train_state = model.train(epochs=2000, display_every=200, model_save_path=model_path) # use if you want to save the model

    model.compile("L-BFGS")
    losshistory, train_state = model.train(display_every=100) 
    # losshistory, train_state = model.train(display_every=200, model_save_path=model_path) # same as above
else:
    n_epochs = 16194 
    model_restore_path = model_path + "-"+ str(n_epochs) + ".ckpt"
    
    model.compile("adam", lr=0.001)
    model.restore(save_path=model_restore_path)

solutionFieldOnMeshToVtkSpaceTime(geom, 
                           model, 
                           save_folder_path=str(Path(__file__).parent.parent.parent.parent), 
                           file_name="lame_3d_time", 
                           polar_transformation="spherical")

def compareModelPredictionAndAnalyticalSolution(model):
    '''
    This function plots analytical solutions and the predictions. 
    '''
    
    # Number of points
    num_points = 100

    # Radii between 1 and 2
    r = np.linspace(radius_inner, radius_outer, num_points)

    # Constants for 45 degrees in spherical coordinates
    cos_phi = np.cos(np.pi / 4)
    sin_phi = np.sin(np.pi / 4)
    cos_theta = np.cos(np.pi / 4)
    sin_theta = np.sin(np.pi / 4)

    # Compute x, y, z coordinates using vectorized operations
    x = r * sin_theta * cos_phi
    y = r * sin_theta * sin_phi
    z = r * cos_theta
    
    nu,lame,shear,e_modul = problem_parameters()
    
    time_element = [1/8, 1/4, 1/2, 3/4, 7/8]
    
    # Define a colormap for time steps
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_element)))

    fig, axs = plt.subplots(1,4,figsize=(20,5))
    
    for i, current_time in enumerate(time_element):    
    # Combine spatial coordinates (X) with the current time into spacetime input
        r_input = np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))
        X_spacetime = np.hstack((r_input, np.full([len(r_input), 1], current_time, dtype=config.real(np))))
        output = model.predict(X_spacetime)
        
        p = pressure*np.sin(2*np.pi*current_time)
        r_a = radius_outer
        r_i = radius_inner
        E = e_modul
        nu = nu

        sigma_rr_a = -p/((r_a/r_i)**3-1)*((r_a/r)**3 - 1)
        sigma_thetatheta_a = p/((r_a/r_i)**3-1)*(1/2*(r_a/r)**3 + 1)
        sigma_phiphi_a = sigma_thetatheta_a
        u_rad_a = r/E*((1-nu)*sigma_thetatheta_a - nu*sigma_rr_a)

        
        
        u_pred, v_pred, w_pred = output[:,0:1], output[:,1:2], output[:,2:3]
        u_rad_p = np.sign(p)*np.sqrt(u_pred**2 + v_pred**2 + w_pred**2)
        sigma_xx_pred, sigma_yy_pred, sigma_zz_pred = output[:,3:4], output[:,4:5], output[:,5:6]
        sigma_xy_pred, sigma_yz_pred, sigma_xz_pred = output[:,6:7], output[:,7:8], output[:,8:9]
        
        sigma_rr_p, sigma_thetatheta_p, sigma_phiphi_p, sigma_rtheta_p, sigma_thetaphi_p, sigma_rphi_p = polar_transformation_3d_spherical(sigma_xx_pred.flatten(), 
                                                                                                                sigma_yy_pred.flatten(), 
                                                                                                                sigma_zz_pred.flatten(), 
                                                                                                                sigma_xy_pred.flatten(), 
                                                                                                                sigma_yz_pred.flatten(), 
                                                                                                                sigma_xz_pred.flatten(), 
                                                                                                                r_input)
        
        rel_err_l2_disp = np.linalg.norm(u_rad_a.flatten() - u_rad_p.flatten()) / np.linalg.norm(u_rad_a)
        print(f"Relative L2 error for displacement at t={current_time}: ", rel_err_l2_disp)
        
        sigma = np.vstack((sigma_rr_a.reshape(-1,1),sigma_thetatheta_a.reshape(-1,1), sigma_phiphi_a.reshape(-1,1)))
        sigma_pred = np.vstack((sigma_rr_p.reshape(-1,1),sigma_thetatheta_p.reshape(-1,1), sigma_phiphi_p.reshape(-1,1)))
        rel_err_l2_stress = np.linalg.norm(sigma.flatten() - sigma_pred.flatten()) / np.linalg.norm(sigma)
        print(f"Relative L2 error for stress at t={current_time}: ", rel_err_l2_stress)
        
        # Set the color for the current iteration
        color = colors[i]
        
        axs[0].plot(r, sigma_rr_a, label = f"t = {current_time}",color=color)
        axs[0].plot(r, sigma_rr_p/radius_inner, color=color, marker='o', markersize=5, markevery=5)
        axs[0].set_xlabel(r"$r$", fontsize=17)
        axs[0].set_ylabel(r"$\sigma_{rr}$", fontsize=17)
        axs[0].tick_params(axis='both', labelsize=12)
        axs[0].legend(fontsize=13)
        axs[0].grid()
        
        axs[1].plot(r, sigma_thetatheta_a, label = f"t = {current_time}",color=color)
        axs[1].plot(r, sigma_thetatheta_p,color=color, marker='o', markersize=5, markevery=5)
        axs[1].set_xlabel(r"$r$", fontsize=17)
        axs[1].set_ylabel(r"$\sigma_{\theta\theta}$", fontsize=17)
        axs[1].tick_params(axis='both', labelsize=12)
        axs[1].legend(fontsize=13)
        axs[1].grid()
        
        axs[2].plot(r, sigma_phiphi_a, label = f"t = {current_time}",color=color)
        axs[2].plot(r, sigma_phiphi_p,color=color, marker='o', markersize=5, markevery=5)
        axs[2].set_xlabel(r"$r$", fontsize=17)
        axs[2].set_ylabel(r"$\sigma_{\phi\phi}$", fontsize=17)
        axs[2].tick_params(axis='both', labelsize=12)
        axs[2].legend(fontsize=13)
        axs[2].yaxis.set_major_formatter(formatter)
        axs[2].grid()
        
        axs[3].plot(r, u_rad_a, label = f"t = {current_time}", color=color)
        axs[3].plot(r, u_rad_p, color=color, marker='o', markersize=5, markevery=5)
        axs[3].set_xlabel(r"$r$", fontsize=17)
        axs[3].set_ylabel(r"$u_r$", fontsize=17)
        axs[3].tick_params(axis='both', labelsize=12)
        axs[3].legend(fontsize=13)
        axs[3].yaxis.set_major_formatter(formatter)
        axs[3].grid()
        
    
    # Custom legend: one for analytical solutions (lines) and one for predictions (marked lines)
    analytical_legend = mlines.Line2D([], [], color='black', label='Analytical Solution')
    predicted_legend = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=5, label='Prediction')
    
    # Add the custom legend below all plots
    fig.legend(handles=[analytical_legend, predicted_legend], loc='lower center', ncol=2, fontsize=14, bbox_to_anchor=(0.5, -0.05))

    
    fig.tight_layout()
    plt.savefig("lame_3d_time_result.png", dpi=300, bbox_inches="tight")
    # plt.show()

compareModelPredictionAndAnalyticalSolution(model)










