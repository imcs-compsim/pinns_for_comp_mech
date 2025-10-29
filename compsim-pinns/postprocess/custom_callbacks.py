import sys
import numpy as np
import os
from pyevtk.hl import unstructuredGridToVTK

from deepxde.callbacks import Callback
from deepxde import utils
from deepxde.backend import backend_name, tf, torch, paddle

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
            

      


