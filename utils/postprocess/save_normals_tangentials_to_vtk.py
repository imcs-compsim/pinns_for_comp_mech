import warnings
import os
import numpy as np
from pyevtk.hl import unstructuredGridToVTK

def export_normals_tangentials_to_vtk(geom, 
                        save_folder_path = None,
                        file_name = "default_boundary_normals_tangentials"):
    '''
    Stores 
    '''
    X, offset, cell_types, elements = geom.get_mesh()
    
    global_normals = np.zeros_like(X) 
    global_tangentials_1 = np.zeros_like(X) 
    global_tangentials_2 = np.zeros_like(X) 
    # boundary points
    cond = geom.on_boundary(X)
    boundary_points = X[cond]

    # get boundary normals
    boundary_normals = geom.boundary_normal(boundary_points)
    # get boundary tangentials
    boundary_tangentials_1 = geom.boundary_tangential_1(boundary_points)
    boundary_tangentials_2 = geom.boundary_tangential_2(boundary_points)
    
    pos = 0
    for correct_loc in cond:
        if correct_loc:
            global_normals[pos] = boundary_normals[pos]
            global_tangentials_1[pos] = boundary_tangentials_1[pos]
            global_tangentials_2[pos] = boundary_tangentials_2[pos]
        pos += 1
    
    n_x = global_normals[:,0].tolist()
    n_y = global_normals[:,1].tolist()
    n_z = global_normals[:,2].tolist()
    
    # tangentials in epsilon direction
    t1x = global_tangentials_1[:,0]
    t1y = global_tangentials_1[:,1]
    t1z = global_tangentials_1[:,2]
    
    # tangentials in eta direction
    t2x = global_tangentials_2[:,0]
    t2y = global_tangentials_2[:,1]
    t2z = global_tangentials_2[:,2]
    
    combined_normals = tuple(np.vstack((n_x, n_y, n_z)))
    combined_tangentials_1 = tuple(np.vstack((t1x, t1y, t1z)))
    combined_ntangentials_2 = tuple(np.vstack((t2x, t2y, t2z)))
    
    if (save_folder_path is None) or (not os.path.isdir(save_folder_path)):
        file_path = os.path.join(os.getcwd(), file_name)
        warnings.warn(f"The folder to save does not exist or not given! It should be an existing full path! Example is stored in {file_path}")
    else: 
        file_path = save_folder_path + "/" + file_name
    

    x = X[:,0].flatten()
    y = X[:,1].flatten()
    z = X[:,2].flatten()

    unstructuredGridToVTK(file_path, x, y, z, elements.flatten(), offset, 
                            cell_types, pointData = {"normal" : combined_normals,
                                                     "tangential_1" : combined_tangentials_1,
                                                     "tangential_2" : combined_ntangentials_2,})