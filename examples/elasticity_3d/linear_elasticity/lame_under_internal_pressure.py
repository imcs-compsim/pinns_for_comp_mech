# import helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#import time
from pathlib import Path
from matplotlib import ticker

# set plot options
mpl.rcParams['mathtext.fontset'] = 'stix'
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))

# import deepxde libraries
from deepxde import backend as bkd
import deepxde as dde

# import computational mechanics libraries
from utils.geometry.custom_geometry import GmshGeometry3D
from utils.geometry.gmsh_models import Geom_step_to_gmsh
from utils.elasticity import elasticity_utils
from utils.elasticity.elasticity_utils import pde_mixed_3d, get_tractions_mixed_3d, problem_parameters
from utils.postprocess.elasticity_postprocessing import solutionFieldOnMeshToVtk3D
from utils.geometry.geometry_utils import polar_transformation_3d_spherical

'''
@author: tsahin

3D lame problem: 

D. Schillinger, J. A. Evans, F. Frischmann, R. R. Hiemstra, M.-C. Hsu, T. J. Hughes, A
collocated c0 finite element method: Reduced quadrature perspective, cost comparison
with standard finite elements, and explicit structural dynamics, International Journal
for Numerical Methods in Engineering 102 (3-4) (2015) 576–631.
'''

path_to_step_file = str(Path(__file__).parent.parent)+f"/step_files/lame_3d.stp"

curve_info = {"4":14, "8":14, "10":14,
              "6":18, "7":18, "2":18,}
Lame_geom_obj = Geom_step_to_gmsh(path=path_to_step_file, curve_info=curve_info)

gmsh_model = Lame_geom_obj.generateGmshModel(visualize_mesh=False)

# Target surface IDs for each set. 
# This means if any surface is neighbor of any of following surfaces, then repeated nodes will take normals+tangenitals of the target surface 
target_surface_ids = [2, 4]

geom = GmshGeometry3D(gmsh_model, target_surface_ids=target_surface_ids)

radius_inner = 1
radius_outer = 2
center = [0,0,0]

# The applied pressure 
pressure = 1

# change global variables in elasticity_utils
elasticity_utils.geom = geom
# change global variables in elasticity_utils
elasticity_utils.lame = 1153.846
elasticity_utils.shear = 769.23
nu,lame,shear,e_modul = problem_parameters()

def boundary_outer(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center, axis=-1), radius_outer)

def boundary_inner(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center, axis=-1), radius_inner) #and cond_on_edge

def apply_presure(x,y,X):
    Tx, Ty, Tz, Pn, Tt_1, Tt_2 = get_tractions_mixed_3d(x, y, X)
    
    return Pn + pressure 

def zero_neumann_in_normal(x,y,X):
    Tx, Ty, Tz, Pn, Tt_1, Tt_2 = get_tractions_mixed_3d(x, y, X)
    
    return Pn 

def zero_neumann_in_tangential1(x,y,X):
    Tx, Ty, Tz, Pn, Tt_1, Tt_2 = get_tractions_mixed_3d(x, y, X)
    
    return Tt_1

def zero_neumann_in_tangential2(x,y,X):
    Tx, Ty, Tz, Pn, Tt_1, Tt_2 = get_tractions_mixed_3d(x, y, X)
    
    return Tt_2

# zero Neumann at R=2 (outer boundary)
bc1 = dde.OperatorBC(geom, zero_neumann_in_normal, boundary_outer)
bc2 = dde.OperatorBC(geom, zero_neumann_in_tangential1, boundary_outer)
bc3 = dde.OperatorBC(geom, zero_neumann_in_tangential2, boundary_outer)

# nonzero and zero Neumann at R=1 (inner boundary)
bc4 = dde.OperatorBC(geom, apply_presure, boundary_inner)
bc5 = dde.OperatorBC(geom, zero_neumann_in_tangential1, boundary_inner)
bc6 = dde.OperatorBC(geom, zero_neumann_in_tangential2, boundary_inner)

bcs = [bc1, bc2, bc3, bc4, bc5, bc6]

n_dummy = 1
data = dde.data.PDE(
    geom,
    pde_mixed_3d,
    bcs,
    num_domain=300,
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
    x_0_surface = x_loc
    y_0_surface = y_loc
    z_0_surface = z_loc
    
    # define the surfaces where shear forces will be applied.
    sigma_xy_surfaces = (x_0_surface)*(y_0_surface)
    sigma_yz_surfaces = (y_0_surface)*(z_0_surface)
    sigma_xz_surfaces = (x_0_surface)*(z_0_surface)
    
    
    return bkd.concat([u*(x_0_surface)/e_modul, #displacement in x direction is 0 at x=0
                      v*(y_0_surface)/e_modul, #displacement in y direction is 0 at y=0
                      w*(z_0_surface)/e_modul, #displacement in z direction is 0 at z=0
                      sigma_xx, 
                      sigma_yy,
                      sigma_zz,
                      sigma_xy*sigma_xy_surfaces,
                      sigma_yz*sigma_yz_surfaces,
                      sigma_xz*sigma_xz_surfaces
                      ], axis=1)

# 3 inputs, 9 outputs for 3D 
layer_size = [3] + [50] * 5 + [9]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)

restore_model = False
model_path = str(Path(__file__).parent.parent)+f"/trained_models/lame/lame_3d"

if not restore_model:
    model.compile("adam", lr=0.001)
    losshistory, train_state = model.train(iterations=2000, display_every=100) 
    # losshistory, train_state = model.train(iterations=2000, display_every=200, model_save_path=model_path) # use if you want to save the model

    model.compile("L-BFGS")
    losshistory, train_state = model.train(display_every=1000)
    # losshistory, train_state = model.train(display_every=200, model_save_path=model_path) # same as above
else:
    n_iterations = 11398 
    model_restore_path = model_path + "-"+ str(n_iterations) + ".ckpt"
    
    model.compile("adam", lr=0.001)
    model.restore(save_path=model_restore_path)

solutionFieldOnMeshToVtk3D(geom, 
                           model, 
                           save_folder_path=str(Path(__file__).parent.parent.parent), 
                           file_name="lame_3d_new", 
                           polar_transformation="spherical")

def compareModelPredictionAndAnalyticalSolution(model):
    '''
    This function plots analytical solutions and the predictions. 
    '''
    nu,lame,shear,e_modul = problem_parameters()
    
    r = np.linspace(radius_inner, radius_outer,100)
    y = np.zeros(r.shape[0])
    z = np.zeros(r.shape[0])
    
    p = pressure
    r_a = radius_outer
    r_i = radius_inner
    E = e_modul
    nu = nu

    sigma_rr_a = -p/((r_a/r_i)**3-1)*((r_a/r)**3 - 1)
    sigma_thetatheta_a = p/((r_a/r_i)**3-1)*(1/2*(r_a/r)**3 + 1)
    sigma_phiphi_a = sigma_thetatheta_a
    u_rad_a = r/E*((1-nu)*sigma_thetatheta_a - nu*sigma_rr_a)

    r_input = np.hstack((r.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))
    output = model.predict(r_input)
    u_pred, v_pred, w_pred = output[:,0:1], output[:,1:2], output[:,2:3]
    u_rad_p = np.sqrt(u_pred**2 + v_pred**2 + w_pred**2)
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
    print("Relative L2 error for displacement: ", rel_err_l2_disp)
    
    sigma = np.vstack((sigma_rr_a.reshape(-1,1),sigma_thetatheta_a.reshape(-1,1), sigma_phiphi_a.reshape(-1,1)))
    sigma_pred = np.vstack((sigma_rr_p.reshape(-1,1),sigma_thetatheta_p.reshape(-1,1), sigma_phiphi_p.reshape(-1,1)))
    rel_err_l2_stress = np.linalg.norm(sigma.flatten() - sigma_pred.flatten()) / np.linalg.norm(sigma)
    print("Relative L2 error for stress: ", rel_err_l2_stress)
    
    fig, axs = plt.subplots(1,3,figsize=(15,5))

    axs[0].plot(r, sigma_rr_a, label = r"Analytical $\sigma_{rr}$",color="tab:blue")
    axs[0].plot(r, sigma_rr_p/radius_inner, label = r"Predicted $\sigma_{rr}$", color="tab:blue", marker='o', markersize=5, markevery=5)
    axs[0].plot(r, sigma_thetatheta_a, label = r"Analytical $\sigma_{\theta\theta}$",color="tab:orange")
    axs[0].plot(r, sigma_thetatheta_p,label = r"Predicted $\sigma_{\theta\theta}$",color="tab:orange", marker='o', markersize=5, markevery=5)
    axs[0].set_xlabel(r"$r$", fontsize=17)
    axs[0].set_ylabel(r"$\sigma$", fontsize=17)
    axs[0].tick_params(axis='both', labelsize=12)
    axs[0].legend(fontsize=13)
    axs[0].grid()
    
    axs[1].plot(r, sigma_phiphi_a, label = r"Analytical $\sigma_{\phi\phi}$",color="tab:orange")
    axs[1].plot(r, sigma_phiphi_p, label = r"Predicted $\sigma_{\phi\phi}$",color="tab:orange", marker='o', markersize=5, markevery=5)
    axs[1].set_xlabel(r"$r$", fontsize=17)
    axs[1].set_ylabel(r"$\sigma$", fontsize=17)
    axs[1].tick_params(axis='both', labelsize=12)
    axs[1].legend(fontsize=13)
    axs[1].yaxis.set_major_formatter(formatter)
    axs[1].grid()
    
    axs[2].plot(r, u_rad_a, label = r"Analytical $u_r$",color="tab:orange")
    axs[2].plot(r, u_rad_p, label = r"Predicted $u_r$",color="tab:orange", marker='o', markersize=5, markevery=5)
    axs[2].set_xlabel(r"$r$", fontsize=17)
    axs[2].set_ylabel(r"$u$", fontsize=17)
    axs[2].tick_params(axis='both', labelsize=12)
    axs[2].legend(fontsize=13)
    axs[2].yaxis.set_major_formatter(formatter)
    axs[2].grid()
    
    
    fig.tight_layout()

    plt.savefig("Lame_3d.png", dpi=300)
    plt.show()

compareModelPredictionAndAnalyticalSolution(model)

# def calculate_loss():
#     losses = np.hstack(
#             (
#                 np.array(losshistory.steps)[:, None],
#                 np.array(losshistory.loss_train),
#             )
#         )
#     steps = losses[:,0]
#     pde_loss = losses[:,1:9].sum(axis=1)
#     neumann_loss = losses[:,10:16].sum(axis=1)
    
#     return steps, pde_loss, neumann_loss

# def compareModelPredictionAndAnalyticalSolution(model):
#     '''
#     This function plots analytical solutions and the predictions. 
#     '''
#     nu,lame,shear,e_modul = problem_parameters()
    
#     r = np.linspace(radius_inner, radius_outer,100)
#     y = np.zeros(r.shape[0])
#     z = np.zeros(r.shape[0])
    
#     p = pressure
#     r_a = radius_outer
#     r_i = radius_inner
#     E = e_modul
#     nu = nu

#     sigma_rr_a = -p/((r_a/r_i)**3-1)*((r_a/r)**3 - 1)
#     sigma_thetatheta_a = p/((r_a/r_i)**3-1)*(1/2*(r_a/r)**3 + 1)
#     sigma_phiphi_a = sigma_thetatheta_a
#     u_rad_a = r/E*((1-nu)*sigma_thetatheta_a - nu*sigma_rr_a)

#     r_input = np.hstack((r.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))
#     output = model.predict(r_input)
#     u_pred, v_pred, w_pred = output[:,0:1], output[:,1:2], output[:,2:3]
#     u_rad_p = np.sqrt(u_pred**2 + v_pred**2 + w_pred**2)
#     sigma_xx_pred, sigma_yy_pred, sigma_zz_pred = output[:,3:4], output[:,4:5], output[:,5:6]
#     sigma_xy_pred, sigma_yz_pred, sigma_xz_pred = output[:,6:7], output[:,7:8], output[:,8:9]
    
#     sigma_rr_p, sigma_thetatheta_p, sigma_phiphi_p, sigma_rtheta_p, sigma_thetaphi_p, sigma_rphi_p = polar_transformation_3d(sigma_xx_pred.flatten(), 
#                                                                                                              sigma_yy_pred.flatten(), 
#                                                                                                              sigma_zz_pred.flatten(), 
#                                                                                                              sigma_xy_pred.flatten(), 
#                                                                                                              sigma_yz_pred.flatten(), 
#                                                                                                              sigma_xz_pred.flatten(), 
#                                                                                                              r_input)
    
#     rel_err_l2_disp = np.linalg.norm(u_rad_a.flatten() - u_rad_p.flatten()) / np.linalg.norm(u_rad_a)
#     print("Relative L2 error for displacement: ", rel_err_l2_disp)
    
#     sigma = np.vstack((sigma_rr_a.reshape(-1,1),sigma_thetatheta_a.reshape(-1,1), sigma_phiphi_a.reshape(-1,1)))
#     sigma_pred = np.vstack((sigma_rr_p.reshape(-1,1),sigma_thetatheta_p.reshape(-1,1), sigma_phiphi_p.reshape(-1,1)))
#     rel_err_l2_stress = np.linalg.norm(sigma.flatten() - sigma_pred.flatten()) / np.linalg.norm(sigma)
#     print("Relative L2 error for stress: ", rel_err_l2_stress)

#     steps, pde_loss, neumann_loss = calculate_loss()
    
#     fig, axs = plt.subplots(1,3,figsize=(15,5))

#     axs[0].plot(r, sigma_rr_a, label = r"Analytical $\sigma_{rr}/R_i$",color="tab:blue")
#     axs[0].plot(r, sigma_rr_p/radius_inner, label = r"Predicted $\sigma_{rr}/R_i$", color="tab:blue", marker='o', markersize=5, markevery=5)
#     # axs[0].scatter(r/radius_inner, sigma_rr/radius_inner, label = r"Predicted $\sigma_{rr}/R_i$", s=10, c="tab:blue", marker='o', edgecolors="tab:orange")
#     axs[0].plot(r, sigma_thetatheta_a, label = r"Analytical $\sigma_{\theta\theta}/R_i$",color="tab:orange")
#     axs[0].plot(r, sigma_thetatheta_p,label = r"Predicted $\sigma_{\theta\theta}/R_i$",color="tab:orange", marker='o', markersize=5, markevery=5)
#     axs[0].set_xlabel(r"$r \ /R_i$", fontsize=17)
#     axs[0].set_ylabel(r"$\sigma \ /R_i$", fontsize=17)
#     axs[0].tick_params(axis='both', labelsize=12)
#     axs[0].legend(fontsize=13)
#     axs[0].grid()
    
#     axs[1].plot(r, u_rad_a, label = r"Analytical $u_r/R_i$",color="tab:orange")
#     axs[1].plot(r, u_rad_p, label = r"Predicted $u_r/R_i$",color="tab:orange", marker='o', markersize=5, markevery=5)
#     axs[1].set_xlabel(r"$r \ /R_i$", fontsize=17)
#     axs[1].set_ylabel(r"$u \ /R_i$", fontsize=17)
#     axs[1].tick_params(axis='both', labelsize=12)
#     axs[1].legend(fontsize=13)
#     axs[1].yaxis.set_major_formatter(formatter)
#     axs[1].grid()
    
#     axs[2].plot(steps, pde_loss/5, color='b', lw=2, label="PDE")
#     axs[2].plot(steps, neumann_loss/4, color='r', lw=2,label="NBC")
#     axs[2].vlines(x=2000,ymin=0, ymax=1, linestyles='--', colors="k")
#     axs[2].annotate(r"ADAM $\ \Leftarrow$ ", xy=[610,0.5], size=13)
#     axs[2].annotate(r"$\Rightarrow \ $ L-BGFS", xy=[2150,0.5], size=13)
#     axs[2].tick_params(axis="both", labelsize=12)
#     axs[2].set_xlabel("Iterations", size=17)
#     axs[2].set_ylabel("MSE", size=17)
#     axs[2].set_yscale('log')
#     axs[2].legend(fontsize=13)
#     axs[2].grid()
    
    
#     fig.tight_layout()

#     plt.savefig("Lame_3d.png", dpi=300)
#     plt.show()










