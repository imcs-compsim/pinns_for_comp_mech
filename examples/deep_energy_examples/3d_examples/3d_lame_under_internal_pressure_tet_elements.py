# import helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#import time
from pathlib import Path
from matplotlib import ticker
import os

# set plot options
mpl.rcParams['mathtext.fontset'] = 'stix'
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))

# import deepxde libraries
from deepxde import backend as bkd
import deepxde as dde

# import computational mechanics libraries
from compsim_pinns.geometry.custom_geometry import GmshGeometry3D
from compsim_pinns.geometry.gmsh_models import Geom_step_to_gmsh
from compsim_pinns.elasticity import elasticity_utils
from compsim_pinns.elasticity.elasticity_utils import pde_mixed_3d, get_tractions_mixed_3d, problem_parameters
from compsim_pinns.postprocess.elasticity_postprocessing import solutionFieldOnMeshToVtk3D
from compsim_pinns.geometry.geometry_utils import polar_transformation_3d_spherical
from compsim_pinns.elasticity.elasticity_utils import get_stress_tensor, get_elastic_strain_3d, problem_parameters

from compsim_pinns.deep_energy.deep_pde import DeepEnergyPDE
from compsim_pinns.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule

from pyevtk.hl import unstructuredGridToVTK

'''
@author: tsahin

3D lame problem: 

D. Schillinger, J. A. Evans, F. Frischmann, R. R. Hiemstra, M.-C. Hsu, T. J. Hughes, A
collocated c0 finite element method: Reduced quadrature perspective, cost comparison
with standard finite elements, and explicit structural dynamics, International Journal
for Numerical Methods in Engineering 102 (3-4) (2015) 576–631.
'''


path_to_step_file = str(Path(__file__).parent.parent.parent)+f"/elasticity_3d/step_files/lame_3d.stp"

curve_info = {"4":20, "8":20, "10":20,
              "6":18, "7":18, "2":18,}
Lame_geom_obj = Geom_step_to_gmsh(path=path_to_step_file, curve_info=curve_info)

gmsh_model = Lame_geom_obj.generateGmshModel(visualize_mesh=False)

# Target surface IDs for each set. 
# This means if any surface is neighbor of any of following surfaces, then repeated nodes will take normals+tangenitals of the target surface 
target_surface_ids = [2, 4]

domain_dimension = 3
quad_rule = GaussQuadratureRule(rule_name="gauss_legendre", element_type="simplex", dimension=domain_dimension, ngp=4) # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()

boundary_dimension = 2
quad_rule_boundary_integral = GaussQuadratureRule(rule_name="gauss_legendre", element_type="simplex", dimension=boundary_dimension, ngp=3) # gauss_legendre gauss_labotto
coord_quadrature_boundary, weight_quadrature_boundary = quad_rule_boundary_integral.generate()

radius_inner = 1
radius_outer = 2
center = [0,0,0]

def boundary_inner(x):
    return np.isclose(np.linalg.norm(x - center, axis=-1), radius_inner) #and cond_on_edge

boundary_selection_map = [{"boundary_function" : boundary_inner, "tag" : "boundary_inner"}]

geom = GmshGeometryElementDeepEnergy(
                           gmsh_model,
                           dimension=3, 
                           coord_quadrature=coord_quadrature, 
                           weight_quadrature= weight_quadrature, 
                           boundary_dim=boundary_dimension,
                           coord_quadrature_boundary=coord_quadrature_boundary,
                           weight_quadrature_boundary=weight_quadrature_boundary,
                           target_surface_ids=target_surface_ids,
                           boundary_selection_map=boundary_selection_map)

# The applied pressure 
pressure = 1

# change global variables in elasticity_utils
elasticity_utils.geom = geom
# change global variables in elasticity_utils
elasticity_utils.lame = 1153.846
elasticity_utils.shear = 769.23
nu,lame,shear,e_modul = problem_parameters()

def potential_energy(X, 
                     inputs, 
                     outputs, 
                     beg_pde, 
                     beg_boundary, 
                     n_e, 
                     n_gp, 
                     n_e_boundary, 
                     n_gp_boundary, 
                     jacobian_t, 
                     global_element_weights_t, 
                     mapped_normal_boundary_t, 
                     jacobian_boundary_t, 
                     global_weights_boundary_t,
                     boundary_selection_tag):
    
    eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_xz = get_elastic_strain_3d(inputs,outputs)
    sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz = get_stress_tensor(inputs,outputs)
    
    # get the internal energy
    internal_energy_density = 0.5 * (
                                    sigma_xx[beg_pde:beg_boundary] * eps_xx[beg_pde:beg_boundary] +
                                    sigma_yy[beg_pde:beg_boundary] * eps_yy[beg_pde:beg_boundary] +
                                    sigma_zz[beg_pde:beg_boundary] * eps_zz[beg_pde:beg_boundary] +
                                    2 * sigma_xy[beg_pde:beg_boundary] * eps_xy[beg_pde:beg_boundary] +
                                    2 * sigma_yz[beg_pde:beg_boundary] * eps_yz[beg_pde:beg_boundary] +
                                    2 * sigma_xz[beg_pde:beg_boundary] * eps_xz[beg_pde:beg_boundary])
    
    internal_energy = global_element_weights_t*(internal_energy_density)*jacobian_t
    
    # get the external energy
    # select the points where external force is applied
    cond = boundary_selection_tag["boundary_inner"]
    #n_e_boundary = int(cond.sum()/n_gp_boundary)
    nx = mapped_normal_boundary_t[:,0:1][cond]
    ny = mapped_normal_boundary_t[:,1:2][cond]
    nz = mapped_normal_boundary_t[:,2:3][cond]

    # #sigma_xx_n_x = sigma_xx[beg_boundary:][cond]*nx
    # #sigma_xy_n_y = sigma_xy[beg_boundary:][cond]*ny

    # sigma_yx_n_x = sigma_xy[beg_boundary:][cond]*nx
    # sigma_yy_n_y = sigma_yy[beg_boundary:][cond]*ny
    
    # #t_x = sigma_xx_n_x + sigma_xy_n_y
    # t_y = sigma_yx_n_x + sigma_yy_n_y
    
    u_x = outputs[:,0:1][beg_boundary:][cond]
    u_y = outputs[:,1:2][beg_boundary:][cond]
    u_z = outputs[:,2:3][beg_boundary:][cond]
    
    external_force_density = -pressure*(nx*u_x + ny*u_y + nz*u_z)
    
    external_work = global_weights_boundary_t[cond]*(external_force_density)*jacobian_boundary_t[cond]
    
    # internal_energy_reshaped = bkd.reshape(internal_energy, (n_e, n_gp))
    # external_work_reshaped = bkd.reshape(external_work, (n_e_boundary, n_gp_boundary))
    
    # total_energy = bkd.reduce_sum(bkd.sum(internal_energy_reshaped, dim=1)) - bkd.reduce_sum(bkd.sum(external_work_reshaped, dim=1)) #+ bkd.reduce_sum(bkd.sum(internal_energy_reshaped, dim=1)) 
    
    return [internal_energy, -external_work]

n_dummy = 1
data = DeepEnergyPDE(
    geom,
    potential_energy,
    [],
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=None,
    train_distribution = "Sobol"
)

def output_transform(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    w = y[:, 2:3]
    
    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    z_loc = x[:, 2:3]
    
    # define surfaces
    x_0_surface = x_loc
    y_0_surface = y_loc
    z_0_surface = z_loc
    
    return bkd.concat([u*(x_0_surface)/e_modul, #displacement in x direction is 0 at x=0
                      v*(y_0_surface)/e_modul, #displacement in y direction is 0 at y=0
                      w*(z_0_surface)/e_modul #displacement in z direction is 0 at z=0
                      ], axis=1)

# 3 inputs, 9 outputs for 3D 
layer_size = [3] + [50] * 5 + [3]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)


model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=10000, display_every=100) 

model.compile("L-BFGS")
losshistory, train_state = model.train(display_every=200)

X, offset, cell_types, elements = geom.get_mesh()

output = model.predict(X)
sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz = model.predict(X, operator=get_stress_tensor)


# .tolist() is applied to remove datatype
u_pred, v_pred, w_pred = output[:,0].tolist(), output[:,1].tolist(), output[:,2].tolist() # displacements
sigma_xx_pred, sigma_yy_pred, sigma_zz_pred = sigma_xx.flatten().tolist(), sigma_yy.flatten().tolist(), sigma_zz.flatten().tolist() # normal stresses
sigma_xy_pred, sigma_yz_pred, sigma_xz_pred = sigma_xy.flatten().tolist(), sigma_yz.flatten().tolist(), sigma_xz.flatten().tolist() # shear stresses

sigma_rr, sigma_thetatheta, sigma_phiphi, sigma_rtheta, sigma_thetaphi, sigma_rphi = polar_transformation_3d_spherical(np.array(sigma_xx_pred), 
                                                                                                                np.array(sigma_yy_pred), 
                                                                                                                np.array(sigma_zz_pred), 
                                                                                                                np.array(sigma_xy_pred), 
                                                                                                                np.array(sigma_yz_pred), 
                                                                                                                np.array(sigma_xz_pred), 
                                                                                                                X)

combined_disp_pred = tuple(np.vstack((u_pred, v_pred, w_pred)))
combined_normal_stress_pred = tuple(np.vstack((sigma_xx_pred, sigma_yy_pred, sigma_zz_pred))) 
combined_shear_stress_pred = np.vstack((sigma_xy_pred, sigma_yz_pred, sigma_xz_pred))

combined_normal_stress_pred_polar = tuple(np.vstack((sigma_rr.tolist(), sigma_thetatheta.tolist(), sigma_phiphi.tolist())))
combined_shear_stress_pred_polar = tuple(np.vstack((sigma_rtheta.tolist(), sigma_thetaphi.tolist(), sigma_rphi.tolist())))  

x = X[:,0].flatten()
y = X[:,1].flatten()
z = X[:,2].flatten()

file_path = os.path.join(os.getcwd(), "deep_energy_3d_lame")

unstructuredGridToVTK(file_path, x, y, z, elements.flatten(), offset, 
                            cell_types, pointData = { "pred_displacement" : combined_disp_pred,
                                                    "pred_normal_stress" : combined_normal_stress_pred,
                                                    "pred_stress_xy": combined_shear_stress_pred[0],
                                                    "pred_stress_yz": combined_shear_stress_pred[1],
                                                    "pred_stress_xz": combined_shear_stress_pred[2],
                                                    "pred_normal_stress_polar" : combined_normal_stress_pred_polar,
                                                    "pred_shear_stress_polar" : combined_shear_stress_pred_polar})

def compareModelPredictionAndAnalyticalSolution(model):
    '''
    This function plots analytical solutions and the predictions. 
    '''    
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

# compareModelPredictionAndAnalyticalSolution(model)








