### Eighth sphere hertzian contact problem using the Deep Energy Method (DEM)
### @author: svoelkl, dwolff, apopp
### based on the work of tsahin
# Import required libraries
import deepxde as dde
dde.config.set_default_float("float64") # use double precision (needed for L-BFGS)
import numpy as np
import pyvista as pv
from pathlib import Path
from deepxde import backend as bkd
import time
import torch 
torch.set_default_device("cpu")

# Import custom modules
from utils.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from utils.geometry.gmsh_models import SphereEighthHertzian
from utils.elasticity import elasticity_utils
from utils.deep_energy.deep_pde import DeepEnergyPDE
from utils.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from utils.vpinns.quad_rule import GaussQuadratureRule
from utils.hyperelasticity import hyperelasticity_utils
from utils.hyperelasticity.hyperelasticity_utils import strain_energy_neo_hookean_3d, compute_elastic_properties, cauchy_stress_3D
from utils.contact_mech.contact_utils import calculate_gap_in_normal_direction_deep_energy
from utils.contact_mech import contact_utils

## Set custom Flag to either restore the model from pretrained
## or simulate yourself
restore_pretrained_model = False

## Create geometry
# Dimensions of disk
radius = 1
center = [0,0,0]
# Create the eighth sphere using gmsh
start_time_meshing = time.time()
Block_3D_obj = SphereEighthHertzian(radius=radius, center=center)
gmsh_model = Block_3D_obj.generateGmshModel(visualize_mesh=False)
end_time_meshing = time.time()

# Define boundary selection map
def on_boundary_circle_contact(x):
    return np.isclose(np.linalg.norm(x - center, axis=-1), radius)
def on_top(x):
    return np.isclose(x[1],0)
boundary_selection_map = [{"boundary_function" : on_boundary_circle_contact, "tag" : "on_boundary_circle_contact"},
                          {"boundary_function" : on_top, "tag" : "on_top"},]
# Define quadrature rule for interior
domain_dimension = 3
quad_rule = GaussQuadratureRule(rule_name="gauss_legendre", dimension=domain_dimension, ngp=4, element_type="simplex") # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()
# Define quadrature rule for boundary
boundary_dimension = domain_dimension - 1
quad_rule_boundary_integral = GaussQuadratureRule(rule_name="gauss_legendre", dimension=boundary_dimension, ngp=4, element_type="simplex") # gauss_legendre gauss_labotto
coord_quadrature_boundary, weight_quadrature_boundary = quad_rule_boundary_integral.generate()
# Create geom object
geom = GmshGeometryElementDeepEnergy(
                           gmsh_model,
                           dimension=domain_dimension, 
                           coord_quadrature=coord_quadrature, 
                           weight_quadrature= weight_quadrature,
                           coord_quadrature_boundary=coord_quadrature_boundary,
                           boundary_dim=boundary_dimension,
                           weight_quadrature_boundary=weight_quadrature_boundary,
                           boundary_selection_map=boundary_selection_map)
# Define geometric parameters
projection_plane = {"y" : -1} # projection plane formula

## Adjust global definitions
# Material parameters
hyperelasticity_utils.e_modul = 50
hyperelasticity_utils.nu = 0.3
nu,lame,shear,e_modul = compute_elastic_properties()

# Communicate parameters to dependencies
elasticity_utils.geom = geom
contact_utils.projection_plane = projection_plane

## Define BCs
# Applied pressure 
ext_traction = 5

## Define energy potentials (internal energy, external work and contact work)
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
    
    ## Internal energy
    # Get internal energy density
    internal_energy_density = strain_energy_neo_hookean_3d(inputs, outputs)
    # Compute internal energy
    internal_energy = global_element_weights_t*(internal_energy_density[beg_pde:beg_boundary])*jacobian_t

    ## External work
    # Select the points where external force is applied
    cond = boundary_selection_tag["on_top"]
    u_y = outputs[:,1:2][beg_boundary:][cond]
    # Get external work density
    external_force_density = -ext_traction*u_y
    # Compute external work
    external_work = global_weights_boundary_t[cond]*(external_force_density)*jacobian_boundary_t[cond]

    ## Contact work
    # Select the points on the boundary
    cond = boundary_selection_tag["on_boundary_circle_contact"]
    # Compute boundary gap
    gap_n = calculate_gap_in_normal_direction_deep_energy(inputs[beg_boundary:], outputs[beg_boundary:], X, mapped_normal_boundary_t, cond)
    # Get contact force density
    eta=3e4
    contact_force_density = 1/2*eta*bkd.relu(-gap_n)*bkd.relu(-gap_n)
    # Compute contact work
    contact_work = global_weights_boundary_t[cond]*(contact_force_density)*jacobian_boundary_t[cond]

    return [internal_energy, -external_work, contact_work]

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
    '''
    Enforce the following conditions in a hard way
            ux(x=0)=0
            uz(z=0)=0
    '''
    u = y[:, 0:1]
    v = y[:, 1:2]
    w = y[:, 2:3]

    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    z_loc = x[:, 2:3]
    
    # Define surfaces
    x_0_surface = x_loc
    z_0_surface = z_loc
    
    return bkd.concat([u*(x_0_surface)/e_modul, #displacement in x direction is 0 at x=0
                      v/e_modul,
                      w*(z_0_surface)/e_modul, #displacement in z direction is 0 at z=0
                      ], axis=1)

## Define the neural network
layer_size = [3] + [50] * 5 + [3]  # 3 inputs: x, y, z, 5 hidden layers with 50 neurons each, 3 outputs: ux, uy,uz
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

## Train the model or use a pre-trained model
model = dde.Model(data, net)
model_path = str(Path(__file__).parent)
simulation_case = f"eighth_sphere"
adam_iterations = 5000

if not restore_pretrained_model:
    start_time_train = time.time()

    model.compile("adam", lr=0.001) # No adjustment of loss weights
    end_time_adam_compile = time.time()
    losshistory, train_state = model.train(iterations=adam_iterations, display_every=100)
    end_time_adam_train = time.time()

    dde.optimizers.config.set_LBFGS_options(maxiter=1000) # stop L-BFGS after 1000 iterations as it starts to oscillate otherwise
    model.compile("L-BFGS") # No adjustment of loss weights
    end_time_LBFGS_compile = time.time()
    losshistory, train_state = model.train(display_every=1000, model_save_path=f"{model_path}/{simulation_case}")

    end_time_train = time.time()
    time_train = f"Total compilation and training time: {(end_time_train - start_time_train):.3f} seconds"
    print(time_train)

    # Retrieve the total number of iterations at the end of training
    n_iterations = train_state.step

    # Save results
    dde.saveplot(
        losshistory, train_state, issave=True, isplot=False, output_dir=model_path, 
        loss_fname=f"{simulation_case}-{n_iterations}_loss.dat", 
        train_fname=f"{simulation_case}-{n_iterations}_train.dat", 
        test_fname=f"{simulation_case}-{n_iterations}_test.dat"
    )

else:
    n_iterations = 6000
    model_restore_path = f"{model_path}/pretrained/{simulation_case}-{n_iterations}.pt"
    model_loss_path = f"{model_path}/pretrained/{simulation_case}-{n_iterations}_loss.dat"
    
    model.compile("L-BFGS")
    model.restore(save_path=model_restore_path)

## Save simulated data to vtk
# points, _, cell_types, elements = geom.get_mesh()
# n_nodes_per_cell = elements.shape[1]
# n_cells = elements.shape[0]
# cells = np.hstack([np.insert(elem, 0, n_nodes_per_cell) for elem in elements])
# cells = np.array(cells, dtype=np.int64)
# cell_types = np.array(cell_types, dtype=np.uint8)
# pinn_results = pv.UnstructuredGrid(cells, cell_types, points)
# output = model.predict(points)

# sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yx, sigma_xz, sigma_zx, sigma_yz, sigma_zy = model.predict(points, operator=cauchy_stress_3D)
# cauchy_stress = np.column_stack((sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz))
# displacement = np.column_stack((output[:,0:1], output[:,1:2], output[:,2:3]))
# pinn_results.point_data['pred_displacement'] = displacement
# pinn_results.point_data['pred_cauchy_stress'] = cauchy_stress
# pinn_results.save(str(Path(__file__).parent.parent.parent.parent.parent)+f"/Hertzian_contact_3d_eighth_sphere.vtu", binary=True)

## Create a comparison with FEM results
# Load the FEM results
fem_results = pv.read(str(Path(__file__).parent.parent)+f"/fem_reference/eighth_sphere_nonlinear_fem_reference.vtu")
prediction_points = fem_results.points
start_time_predict = time.time()
prediction_displacement = model.predict(prediction_points)
prediction_sigma_xx, prediction_sigma_yy, prediction_sigma_zz, prediction_sigma_xy, _, prediction_sigma_xz, _, prediction_sigma_yz, _ = model.predict(prediction_points, operator=cauchy_stress_3D)
prediction_stresses = np.hstack((prediction_sigma_xx,prediction_sigma_yy,prediction_sigma_zz,prediction_sigma_xy,prediction_sigma_yz,prediction_sigma_xz))
end_time_predict = time.time()

# Compute differences
fem_displacements = fem_results.point_data["displacement"]
error_displacement = abs(prediction_displacement - fem_displacements)
fem_stresses = fem_results.point_data["nodal_cauchy_stresses_xyz"]
error_stresses = abs(prediction_stresses - fem_stresses)

# Save and return them in vtu file
fem_results.point_data["displacement_prediction"] = prediction_displacement
fem_results.point_data["error_displacement"] = error_displacement
fem_results.point_data["stresses_prediction"] = prediction_stresses
fem_results.point_data["error_stresses"] = error_stresses
fem_results.save(str(Path(__file__).parent.parent.parent.parent.parent)+f"/Hertzian_contact_eighth_sphere_nonlinear_predictions.vtu", binary=True)

# Output l2-error into console and file
rel_err_l2_displacement = np.linalg.norm(prediction_displacement - fem_displacements) / np.linalg.norm(fem_displacements)
rel_err_l2_stresses = np.linalg.norm(prediction_stresses - fem_stresses) / np.linalg.norm(fem_stresses)
print("Relative L2 error for displacement: ", rel_err_l2_displacement)
print("Relative L2 error for stresses:     ", rel_err_l2_stresses)
with open(f"{model_path}/{simulation_case}-{n_iterations}_L2_error_norm.txt", "w") as text_file:
    print(f"Relative L2 error for displacement: {rel_err_l2_displacement:.8e}",   file=text_file)
    print(f"Relative L2 error for stresses:     {rel_err_l2_stresses:.8e}"    ,   file=text_file)

# Print times to output file
if not restore_pretrained_model:
    with open(f"{model_path}/{simulation_case}-{n_iterations}_times.txt", "w") as text_file:
        print(f"Compilation and training times in [s]", file=text_file)
        print(f"Meshing took:        {(end_time_meshing - start_time_meshing):6.3f}", file=text_file)
        print(f"Adam compilation:    {(end_time_adam_compile - start_time_train):6.3f}", file=text_file)
        print(f"Adam training:       {(end_time_adam_train - end_time_adam_compile):6.3f}", file=text_file)
        print(f"L-BFGS compilation:  {(end_time_LBFGS_compile - end_time_adam_train):6.3f}", file=text_file)
        print(f"L-BFGS training:     {(end_time_train - end_time_LBFGS_compile):6.3f}", file=text_file)
        print(f"Total:               {(end_time_train - start_time_train):6.3f}", file=text_file)
        print(f"Prediction:          {(end_time_predict - start_time_predict):6.3f}", file=text_file)