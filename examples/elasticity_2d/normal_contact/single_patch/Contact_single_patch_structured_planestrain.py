import deepxde as dde
import numpy as np
import os
import deepxde.backend as bkd
from pyevtk.hl import unstructuredGridToVTK
import time
from pathlib import Path

from compsim_pinns.geometry.gmsh_models import Block_2D
from compsim_pinns.geometry.custom_geometry import GmshGeometry2D

from compsim_pinns.elasticity.elasticity_utils import problem_parameters, pde_mixed_plane_strain
from compsim_pinns.contact_mech.contact_utils import zero_tangential_traction, positive_normal_gap_sign, negative_normal_traction_sign, zero_complimentary
from compsim_pinns.contact_mech.contact_utils import positive_normal_gap_adopted_sigmoid, negative_normal_traction_adopted_sigmoid
from compsim_pinns.contact_mech.contact_utils import zero_complementarity_function_based_popp, zero_complementarity_function_based_fischer_burmeister
from compsim_pinns.elasticity import elasticity_utils
from compsim_pinns.contact_mech import contact_utils

dde.config.set_default_float("float64")

'''
Single patch-test for testing contact conditions. It is a simple block under compression. Check problem_figures/Contact_patch.png for details.

In this script, four different methods are described to enforce the Karush-Kuhn-Tucker inequalities (gn>=0, Pn<=0 and gn*Pn=0):
 - Inequalities with sign function 
   - ref: https://arxiv.org/pdf/2003.02751.pdf
 - Inequalities using an adopted sigmoid function
   - ref: https://arxiv.org/abs/2203.09789
 - A complementarity function: Pn-max(0, Pn-c*gn) used often in computational contact mechanics
   - ref: https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.2614
 - A complementarity function called Fischer-Burmeister
   - ref: https://www.math.uwaterloo.ca/~ltuncel/publications/corr2007-17.pdf

@author: tsahin

@author: tsahin
'''

# Define GMSH and geometry parameters
gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
coord_left_corner=[-0,-0.]
coord_right_corner=[1,1]
l_beam = coord_right_corner[0] - coord_left_corner[0]
h_beam = coord_right_corner[1] - coord_left_corner[1]
# create a block
block_2d = Block_2D(coord_left_corner=coord_left_corner, coord_right_corner=coord_right_corner, mesh_size=0.05, gmsh_options=gmsh_options)
# generate gmsh model
gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)

revert_curve_list = []
revert_normal_dir_list = [1,1,1,1]
geom = GmshGeometry2D(gmsh_model, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)

# The applied pressure 
ext_traction = -0.1

# how far above the block from ground
distance = 0

# complementarity parameter c
c_complementarity = 0.001

# delta parameters for adopted sigmoid function (experimental parameters)
delta_gap = 10
delta_pressure = 100

# assign local parameters from the current file in contact_utils and elasticity_utils
contact_utils.distance = distance
contact_utils.c_complementarity=c_complementarity
contact_utils.delta_gap = delta_gap
contact_utils.delta_pressure = delta_pressure
elasticity_utils.geom = geom
contact_utils.geom = geom

# define contact boundary points
def boundary_contact(x, on_boundary):
    return on_boundary and np.isclose(x[1],0)

method_list = ["KKT_inequality_sign", "KKT_inequality_sigmoid", "complementarity_popp", "fischer_burmeister"]
method_name = "KKT_inequality_sigmoid"

# Karush-Kuhn-Tucker conditions for frictionless contact
# gn>=0 (positive_normal_gap), Pn<=0 (negative_normal_traction), Tt=0 (zero_tangential_traction) and gn.Pn=0 (zero_complimentary)

bc_zero_tangential_traction = dde.OperatorBC(geom, zero_tangential_traction, boundary_contact)

if method_name == "KKT_inequality_sign":
    bc_positive_normal_gap = dde.OperatorBC(geom, positive_normal_gap_sign, boundary_contact)
    bc_negative_normal_traction = dde.OperatorBC(geom, negative_normal_traction_sign, boundary_contact)
    bc_zero_complimentary = dde.OperatorBC(geom, zero_complimentary, boundary_contact)
    bcs_ = [bc_positive_normal_gap,bc_negative_normal_traction,bc_zero_tangential_traction,bc_zero_complimentary]
    output_file_name = f"Patch_sign"
elif method_name == "KKT_inequality_sigmoid":
    bc_positive_normal_gap = dde.OperatorBC(geom, positive_normal_gap_adopted_sigmoid, boundary_contact)
    bc_negative_normal_traction = dde.OperatorBC(geom, negative_normal_traction_adopted_sigmoid, boundary_contact)
    bc_zero_complimentary = dde.OperatorBC(geom, zero_complimentary, boundary_contact)
    bcs_ = [bc_positive_normal_gap,bc_negative_normal_traction,bc_zero_tangential_traction,bc_zero_complimentary]
    output_file_name = f"Patch_adopted_sigmoid"
elif method_name == "complementarity_popp":
    bc_zero_complementarity = dde.OperatorBC(geom, zero_complementarity_function_based_popp, boundary_contact)
    bcs_ = [bc_zero_complementarity,bc_zero_tangential_traction]
    output_file_name = f"Patch_complementarity_function_c_{c_complementarity}"
elif method_name == "fischer_burmeister":
    bc_zero_complementarity = dde.OperatorBC(geom, zero_complementarity_function_based_fischer_burmeister, boundary_contact)
    bcs_ = [bc_zero_complementarity,bc_zero_tangential_traction]
    output_file_name = "Patch_fischer_burmeister"
else:
    raise Exception("Method name does not exist!")

n_dummy = 1
data = dde.data.PDE(
    geom,
    pde_mixed_plane_strain,
    bcs_,
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=n_dummy,
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
    sigma_xx = y[:, 2:3]
    sigma_yy = y[:, 3:4]
    sigma_xy = y[:, 4:5]
    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    
    return bkd.concat([u*(x_loc),v, sigma_xx*(l_beam-x_loc), ext_traction + sigma_yy*(h_beam-y_loc),sigma_xy*(l_beam-x_loc)*(x_loc)*(h_beam-y_loc)], axis=1)

# two inputs x and y, 5 outputs are ux, uy, sigma_xx, sigma_yy and sigma_xy
layer_size = [2] + [50] * 5 + [5]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)

restore_model = False
# store the model
model_path = str(Path(__file__).parent.parent.parent.parent)+f"/pretrained_models/elasticity_2d/patch/{method_name}/{method_name}"

# number epochs required for restoring model
n_epoch_dict = {"KKT_inequality_sign": 2265, "KKT_inequality_sigmoid": 2272, "fischer_burmeister": 2394}

if not restore_model:
    model.compile("adam", lr=0.001)
    losshistory, train_state = model.train(epochs=2000, display_every=100, model_save_path=model_path)

    model.compile("L-BFGS")
    losshistory, train_state = model.train(display_every=200, model_save_path=model_path)
else:
    n_epochs = n_epoch_dict[method_name] # trained model has 3106 iterations
    model_restore_path = model_path + "-"+ str(n_epochs) + ".ckpt"
    
    model.compile("adam", lr=0.001)
    model.restore(save_path=model_restore_path)

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

# create a block
block_2d = Block_2D(coord_left_corner=coord_left_corner, coord_right_corner=coord_right_corner, mesh_size=0.01, gmsh_options=gmsh_options)
# generate gmsh model
gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)

revert_curve_list = []
revert_normal_dir_list = [1,1,1,1]
geom = GmshGeometry2D(gmsh_model, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)

X, offset, cell_types, dol_triangles = geom.get_mesh()
nu,lame,shear,e_modul = problem_parameters()

start_time_calc = time.time()
output = model.predict(X)
end_time_calc = time.time()
final_time = f'Prediction time: {(end_time_calc - start_time_calc):.3f} seconds'
print(final_time)

u_x_pred, u_y_pred = output[:,0], output[:,1]
sigma_xx, sigma_yy, sigma_xy = output[:,2], output[:,3], output[:,4]

u_x_analytical = -ext_traction/e_modul*nu*(1+nu)*X[:,0]
u_y_analytical = ext_traction/e_modul*(1-nu**2)*X[:,1]
s_xx_analytical = np.zeros(X.shape[0])
s_yy_analytical = ext_traction*np.ones(X.shape[0])
s_xy_analytical = np.zeros(X.shape[0])

error_u_x = abs(u_x_pred - u_x_analytical)
error_u_y = abs(u_y_pred - u_y_analytical)
combined_error_disp = tuple(np.vstack((error_u_x, error_u_y, np.zeros(error_u_x.shape[0]))))

error_s_xx = abs(sigma_xx - s_xx_analytical)
error_s_yy = abs(sigma_yy - s_yy_analytical)
error_s_xy = abs(sigma_xy - s_xy_analytical)
combined_error_stress = tuple(np.vstack((error_s_xx, error_s_yy, error_s_xy)))

combined_disp = tuple(np.vstack((u_x_pred, u_y_pred, np.zeros(u_x_pred.shape[0]))))
combined_stress = tuple(np.vstack((sigma_xx, sigma_yy, sigma_xy)))
combined_disp_analytical = tuple(np.vstack((u_x_analytical, u_y_analytical, np.zeros(u_x_analytical.shape[0]))))
combined_stress_analytical = tuple(np.vstack((s_xx_analytical, s_yy_analytical, s_xy_analytical)))


file_path = os.path.join(os.getcwd(), output_file_name)

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)

#np.savetxt("Lame_inverse_large", X=np.hstack((X,output[:,0:2])))

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                      cell_types, pointData = { "displacement" : combined_disp,"stress" : combined_stress, "analy_stress" : combined_stress_analytical, "analy_disp" : combined_disp_analytical
                                               ,"error_disp":combined_error_disp, "error_stress":combined_error_stress})

# Calculate l2-error
u_combined_pred = np.asarray(combined_disp).T
s_combined_pred = np.asarray(combined_stress).T
u_combined_analytical = np.asarray(combined_disp_analytical).T
s_combined_analytical = np.asarray(combined_stress_analytical).T

rel_err_l2_disp = np.linalg.norm(u_combined_pred - u_combined_analytical) / np.linalg.norm(u_combined_analytical)
print("Relative L2 error for disp: ", rel_err_l2_disp)
rel_err_l2_stress = np.linalg.norm(s_combined_pred - s_combined_analytical) / np.linalg.norm(s_combined_analytical)
print("Relative L2 error for stress: ", rel_err_l2_stress)

# adopted sign
# Relative L2 error for disp:  0.003821093510421621
# Relative L2 error for stress:  0.001537474850331165

# sigmoid
# Relative L2 error for disp:  0.0009028400465172404
# Relative L2 error for stress:  0.0009412639930073679

# fb
# Relative L2 error for disp:  0.0002480401972722835
# Relative L2 error for stress:  0.00030554812293575304


