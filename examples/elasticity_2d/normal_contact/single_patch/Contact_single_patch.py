import deepxde as dde
import numpy as np
import os
import deepxde.backend as bkd
from pyevtk.hl import unstructuredGridToVTK

from utils.geometry.gmsh_models import Block_2D
from utils.geometry.custom_geometry import GmshGeometry2D

from utils.elasticity.elasticity_utils import problem_parameters, pde_mixed_plane_stress
from utils.contact_mech.contact_utils import zero_tangential_traction, positive_normal_gap_sign, negative_normal_traction_sign, zero_complimentary
from utils.contact_mech.contact_utils import positive_normal_gap_adopted_sigmoid, negative_normal_traction_adopted_sigmoid
from utils.contact_mech.contact_utils import zero_complementarity_function_based_popp, zero_complementarity_function_based_fischer_burmeister
from utils.elasticity import elasticity_utils
from utils.contact_mech import contact_utils

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
method_name = "fischer_burmeister"

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
    pde_mixed_plane_stress,
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