import sys
import numpy as np
import os
from pyevtk.hl import unstructuredGridToVTK

from deepxde.callbacks import Callback
from deepxde import utils
from deepxde.backend import backend_name, tf, torch, paddle

from compsim_pinns.hyperelasticity.hyperelasticity_utils import deformation_gradient_3D_t
from compsim_pinns.contact_mech.contact_utils import calculate_gap_in_normal_direction_deep_energy

class EpochTracker(Callback):
    """Tracks and provides access to the current epoch number during training."""

    def __init__(self, period=10):
        super().__init__()
        self.period = period
        #self.epochs_since_last_tracker = 0

    def on_epoch_begin(self):
        # self.epochs_since_last_tracker += 1
        # if self.epochs_since_last_tracker < self.period:
        #     return
        # self.epochs_since_last_tracker = 0
        self.model.data.current_epoch = self.model.train_state.epoch
        # print(self.model.data.current_epoch)
        # You can use this variable wherever needed
        # Or call a method here with self.current_epoch as argument
        
    def on_train_end(self):
        # self.epochs_since_last_tracker += 1
        # if self.epochs_since_last_tracker < self.period:
        #     return
        # self.epochs_since_last_tracker = 0
        self.model.data.current_epoch = None
        # print(self.model.data.current_epoch)
        # You can use this variable wherever needed
        # Or call a method here with self.current_epoch as argument
    
    def get_epoch(self):
        return self.current_epoch

class SaveModelVTU(Callback):
    """Generates operator values for the input samples.

    Args:
        x: The input data.
        op: The operator with inputs (x, y).
        period (int): Interval (number of epochs) between checking values.
        filename (string): Output the values to the file `filename`.
            The file is kept open to allow instances to be re-used.
            If ``None``, output to the screen.
        precision (int): The precision of variables to display.
    """

    def __init__(self, op=None, period=100, stabilization_epoch=None, filename=None):
        super().__init__()
        self.op = op
        self.period = period
        self.filename = filename
        self.stabilization_epoch = stabilization_epoch
        
        self.value = None
        self.epochs_since_last = 0
        
    def on_epoch_end(self):
        self.epochs_since_last += 1
        if self.epochs_since_last >= self.period:
            self.epochs_since_last = 0
            
            model = self.model
            if self.stabilization_epoch:
                current_iteration = self.model.train_state.epoch - self.stabilization_epoch
            else:
                current_iteration = self.model.train_state.epoch 
            
            X, offset, cell_types, dol_triangles = model.data.geom.get_mesh()
            
            displacement = model.predict(X)
            if X.shape[1] == 2:
                combined_disp = tuple(np.vstack((np.array(displacement[:,0].tolist()),np.array(displacement[:,1].tolist()),np.zeros(displacement[:,0].shape[0]))))
            if X.shape[1] == 3:
                combined_disp = tuple(np.vstack((np.array(displacement[:,0].tolist()),np.array(displacement[:,1].tolist()),np.array(displacement[:,2].tolist()))))
            
            pointData = { "displacement" : combined_disp}
            if self.op:
                sigma_xx, sigma_yy, sigma_xy, sigma_yx = model.predict(X, operator=self.op)
                combined_stress = tuple(np.vstack((np.array(sigma_xx.flatten().tolist()),np.array(sigma_yy.flatten().tolist()),np.array(sigma_xy.flatten().tolist()))))
                pointData["stress" ] = combined_stress
                
            x = X[:,0].flatten()
            y = X[:,1].flatten()
            z = np.zeros(y.shape) if X.shape[1] == 2 else X[:,2].flatten()

            unstructuredGridToVTK(self.filename +"_"+ str(current_iteration), x, y, z, dol_triangles.flatten(), offset,
                                cell_types, pointData = pointData)     

class LossPlateauStopping(Callback):
    """
    Stop training if the training loss does not change significantly
    within the last `patience` iterations.
    """

    def __init__(self, min_delta=1e-3, patience=1000, monitor="loss_train", start_from_iteration=0):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.last_loss = np.inf
        self.wait = 0
        self.stopped_iteration = 0
        self.start_from_iteration = start_from_iteration

    def on_train_begin(self):
        self.wait = 0
        self.stopped_iteration = 0

    def on_epoch_end(self):
        if self.model.train_state.iteration < self.start_from_iteration:
            return
        current = self.get_monitor_value()
        if abs((self.last_loss - current)/current) < self.min_delta:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_iteration = self.model.train_state.iteration
                self.model.stop_training = True
        else:
            self.wait = 0
        self.last_loss = current

    def on_train_end(self):
        if self.stopped_iteration > 0:
            print(f"Early stopping at iteration {self.stopped_iteration} as relative change of {self.monitor} was below {self.min_delta:.1E} for at least {self.patience} iterations.")

    def get_monitor_value(self):
        if self.monitor == "loss_train":
            result = sum(self.model.train_state.loss_train)
        elif self.monitor == "loss_test":
            result = sum(self.model.train_state.loss_test)
        else:
            raise ValueError("The specified monitor function is incorrect.")

        return result

class WeightsBiasPlateauStopping(Callback):
    """
    Stop training if the network weights and bias do not change significantly
    within the last `patience` iterations.
    """

    def __init__(self, min_delta=1e-2, patience=1000, norm_choice="fro", start_from_iteration=0):
        super().__init__()
        self.norm_choice = norm_choice
        self.patience = patience
        self.min_delta = min_delta
        self.norm_list = []
        self.wait = 0
        self.stopped_iteration = 0
        self.start_from_iteration = start_from_iteration
        if norm_choice == "fro":
            self.order = (2,"fro")
        elif norm_choice == "l1":
            self.order = (1,1)
        elif norm_choice == "l2":
            self.order = (2,2)
        elif norm_choice == "linf":
            self.order = (float("inf"),float("inf"))
        else:
            raise ValueError("The specified norm is not implemented or correct.")

    def on_train_begin(self):
        self.wait = 0
        self.stopped_iteration = 0

    def on_epoch_end(self):
        if self.model.train_state.iteration < self.start_from_iteration:
            return
        self.norm_list.append(self.get_norm_value())
        if len(self.norm_list) > self.patience:
            self.norm_list.pop(0)
        if abs((max(self.norm_list) - min(self.norm_list)) * len(self.norm_list)/ sum(self.norm_list)) < self.min_delta:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_iteration = self.model.train_state.iteration
                self.model.stop_training = True
        else:
            self.wait = 0

    def on_train_end(self):
        if self.stopped_iteration > 0:
            print(f"Early stopping at iteration {self.stopped_iteration} as relative change of {self.norm_choice} norm was below {self.min_delta:.1E} for at least {self.patience} iterations.")

    def get_norm_value(self):
        device = next(self.model.net.parameters()).device
        result = torch.tensor(0.0, device=device)
        for name, param in self.model.net.named_parameters():
            if "bias" in name:
                result += torch.linalg.norm(param, ord=self.order[0])
            elif "weight" in name:
                result += torch.linalg.norm(param, ord=self.order[1])
        return float(result)
    
class ResetLagrangeParameters(Callback):
    """
    Reset the lagrange parameters at the beginning of a new iteration.
    This is necessary, as the lagrange parameters are only initialiazed as zero with the geometry creation.
    """

    def __init__(self):
        super().__init__()

    def on_train_begin(self):
        if not self.model.data.geom.lagrange_method:
            raise ValueError("You are requesting a reset of lagrange parameters even though you are not using a lagrange method.")
        self.model.data.geom.lagrange_parameter = np.zeros_like(self.model.data.geom.lagrange_parameter)
        print(f"Resetting lagrange parameters.")

class UpdateLagrangeParameters(Callback):
    """
    Update the lagrange parameters at the end of a training.
    This is necessary for a Uzawa type of Augmented Lagrangian method.
    """

    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = float(epsilon)

    def on_train_end(self):
        # find tensor location
        device = next(self.model.net.parameters()).device
        dtype = next(self.model.net.parameters()).dtype
        # collect data from neural network
        X = self.model.data.train_x
        cond = self.model.data.geom.boundary_selection_tag["on_boundary_circle_contact"]
        inputs = torch.as_tensor(X, device=device, dtype=dtype).requires_grad_(True)
        self.model.net.eval()
        outputs = self.model.net(inputs)
        mapped_normal_boundary_t = torch.as_tensor(self.model.data.geom.mapped_normal_boundary, device=device, dtype=dtype)
        bcs_start = np.cumsum([0] + self.model.data.num_bcs)
        bcs_start = list(map(int, bcs_start))
        pde_start = bcs_start[-1]
        beg_boundary =  pde_start + self.model.data.geom.mapped_coordinates.shape[0]
        # compute current normals and current gap
        deformations_grad = deformation_gradient_3D_t(inputs, outputs)
        F = deformations_grad[beg_boundary:]
        J = torch.det(F).view(-1, 1, 1)
        FinvT = torch.linalg.inv(F).transpose(1, 2)
        current_normals = torch.nn.functional.normalize(torch.einsum("bij,bj->bi", FinvT, mapped_normal_boundary_t), dim=1)
        gap_n = calculate_gap_in_normal_direction_deep_energy(inputs[beg_boundary:], outputs[beg_boundary:], X, current_normals, cond)
        # compute the new lagrange parameters
        lagrange_parameter = torch.as_tensor(self.model.data.geom.lagrange_parameter, device=device, dtype=dtype)
        lagrange_parameter[cond] = torch.relu(lagrange_parameter[cond] - self.epsilon * gap_n)
        self.model.data.geom.lagrange_parameter = lagrange_parameter.detach().cpu().numpy()
        print(f"Updating lagrange parameters.")