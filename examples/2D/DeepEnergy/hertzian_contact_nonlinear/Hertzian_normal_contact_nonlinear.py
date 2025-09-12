### Quarter disc hertzian contact problem using the Deep Energy Method (DEM)
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

# Import custom modules
from utils.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from utils.geometry.gmsh_models import QuarterDisc
from utils.geometry.geometry_utils import polar_transformation_2d
from utils.elasticity import elasticity_utils
from utils.hyperelasticity.hyperelasticity_utils import strain_energy_neo_hookean_2d, compute_elastic_properties, cauchy_stress_2D, first_piola_stress_tensor_2D
from utils.hyperelasticity import hyperelasticity_utils
from utils.contact_mech.contact_utils import calculate_gap_in_normal_direction_deep_energy
from utils.contact_mech import contact_utils
from utils.vpinns.quad_rule import GaussQuadratureRule
from utils.deep_energy.deep_pde import DeepEnergyPDE

## Set custom Flag to either restore the model from pretrained
## or simulate yourself
restore_pretrained_model = True

## Create geometry
# Dimensions of disk
radius = 1
center = [0,0]
# Create the quarter disk using gmsh
gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 11}
start_time_meshing = time.time()
Quarter_Disc = QuarterDisc(radius=radius, center=center, mesh_size=0.04, angle=225, refine_times=10, gmsh_options=gmsh_options)
gmsh_model, x_loc_partition, y_loc_partition = Quarter_Disc.generateGmshModel(visualize_mesh=False)
end_time_meshing = time.time()
# Modifications to define a proper outer normal
revert_curve_list = []
revert_normal_dir_list = [1,2,2,1]
# Define boundary selection map
def on_boundary_circle_contact(x):
    return np.isclose(np.linalg.norm(x - center, axis=-1), radius) and (x[0]>=x_loc_partition)
def on_top(x):
    return np.isclose(x[1],0)
def points_at_top(x):
    cond_points_top = np.isclose(x, 0)
    return cond_points_top
boundary_selection_map = [{"boundary_function" : on_boundary_circle_contact, "tag" : "on_boundary_circle_contact"},
                          {"boundary_function" : on_top, "tag" : "on_top"},]
# Define quadrature rule for interior
quad_rule = GaussQuadratureRule(rule_name="gauss_legendre", dimension=2, ngp=2) # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()
# Define quadrature rule for boundary
quad_rule_boundary_integral = GaussQuadratureRule(rule_name="gauss_legendre", dimension=1, ngp=6) # gauss_legendre gauss_labotto
coord_quadrature_boundary, weight_quadrature_boundary = quad_rule_boundary_integral.generate()
# Create geom object
geom = GmshGeometryElementDeepEnergy(
                           gmsh_model,
                           dimension=2, 
                           coord_quadrature=coord_quadrature, 
                           weight_quadrature= weight_quadrature, 
                           revert_curve_list=revert_curve_list, 
                           revert_normal_dir_list=revert_normal_dir_list,
                           coord_quadrature_boundary=coord_quadrature_boundary,
                           weight_quadrature_boundary=weight_quadrature_boundary,
                           boundary_selection_map=boundary_selection_map)
# Define geometric parameters
projection_plane = {"y" : -1} # projection plane formula

## Adjust global definitions
# Material parameters
hyperelasticity_utils.e_modul = 50
hyperelasticity_utils.nu = 0.3
hyperelasticity_utils.stress_state = "plane_strain"
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
    internal_energy_density = strain_energy_neo_hookean_2d(inputs, outputs)
    # Compute internal energy
    internal_energy = global_element_weights_t[:,0:1]*global_element_weights_t[:,1:2]*(internal_energy_density[beg_pde:beg_boundary])*jacobian_t

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
    num_test=None
)

def output_transform(x, y):
    '''
    Enforce the following conditions in a hard way
            u(x=0)=0
    '''
    u = y[:, 0:1]
    v = y[:, 1:2]

    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    
    return bkd.concat([u*(-x_loc)/e_modul, v/e_modul], axis=1)

## Define the neural network
layer_size = [2] + [50] * 5 + [2] # 2 inputs: x, y, 5 hidden layers with 50 neurons each, 2 outputs: ux, uy
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

## Train the model or use a pre-trained model
model = dde.Model(data, net)
model_path = str(Path(__file__).parent)
simulation_case = f"herztian_contact_nonlinear"
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
    # If you use a machine that doesnt have a GPU or the GPU does not support float64 (e.g., in MacOS) use this line instead
    # model.restore(save_path=model_restore_path, device="cpu")

# Output results to VTU
file_path = str(Path(__file__).parent.parent.parent.parent.parent)+f"/Hertzian_normal_contact_nonlinear.vtk"

solution_points = data.train_x
pinn_results = pv.PolyData(np.hstack((solution_points, np.zeros((solution_points.shape[0], 1)))))
solution_output = model.predict(solution_points)
pinn_results["displacement"] = np.hstack((solution_output, np.zeros((solution_output.shape[0], 1))))
pinn_results.save(file_path)

## Create a comparison with FEM results
# Load the FEM results
fem_results = pv.read(str(Path(__file__).parent.parent)+f"/fem_reference_nonlinear/nonlinear_fem_reference.vtu")
prediction_points = fem_results.points
start_time_predict = time.time()
prediction_displacement = model.predict(prediction_points[:,0:2])
prediction_sigma_xx, prediction_sigma_yy, prediction_sigma_xy, _ = model.predict(prediction_points[:,0:2], operator=cauchy_stress_2D)
prediction_stresses = np.hstack((prediction_sigma_xx,prediction_sigma_yy,np.zeros_like(prediction_sigma_xx),prediction_sigma_xy,np.zeros_like(prediction_sigma_xx),np.zeros_like(prediction_sigma_xx)))
end_time_predict = time.time()

# Compute differences
fem_displacements = fem_results.point_data["displacement"]
error_displacement = abs(prediction_displacement - fem_displacements)
fem_stresses = fem_results.point_data["nodal_cauchy_stresses_xyz"]
error_stresses = abs(prediction_stresses - fem_stresses)

# Save and return them in vtu file
fem_results.point_data["displacement"] = np.hstack((fem_displacements, np.zeros((prediction_displacement.shape[0], 1)))) # add displacement in z to warp properly
fem_results.point_data["displacement_prediction"] = np.hstack((prediction_displacement, np.zeros((prediction_displacement.shape[0], 1)))) # add displacement in z to warp properly
fem_results.point_data["error_displacement"] = np.hstack((error_displacement, np.zeros((error_displacement.shape[0], 1))))
fem_results.point_data["stresses_prediction"] = prediction_stresses
fem_results.point_data["error_stresses"] = error_stresses
fem_results.save(str(Path(__file__).parent.parent.parent.parent.parent)+f"/Hertzian_normal_contact_nonlinear_predictions.vtu", binary=True)

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
