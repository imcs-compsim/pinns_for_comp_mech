import numpy as np
import os
import warnings

import triangle as tr

from geometry_utils import polar_transformation_2d
from deepxde.geometry.csg import CSGUnion, CSGDifference, CSGIntersection

def meshGeometry(geom, n_boundary=100, holes=None, max_mesh_area=None, boundary_distribution="pseudo"):
    '''
    Meshes the geometry using boundary points
    
    Parameters
    ----------
    geom : Class object
        contains the geometry object
    n_boundary: int
        represents the number of points on boundary
    holes: np.array
        center location of the holes
    max_mesh_area: float
        max area of elements
    boundary_distribution
        the distribution of points on boundary, available options: pseudo, uniform, Sobol, LHS, Halton, Hammersley

    Returns
    -------
    vertices: numpy array
        The location of vertices
    triangles: numpy array
        Mesh as triangle
    '''

    if isinstance(geom,(CSGUnion, CSGDifference, CSGIntersection)) and holes:

        if boundary_distribution == "uniform":
            boundary_points_1 = geom.geom1.uniform_boundary_points(n_boundary)
            connectivity_1 = tr.convex_hull(boundary_points_1)

            boundary_points_2 = geom.geom2.uniform_boundary_points(n_boundary)
            connectivity_2 = tr.convex_hull(boundary_points_2)
        else:
            boundary_points_1 = geom.geom1.random_boundary_points(n_boundary, random=boundary_distribution)
            connectivity_1 = tr.convex_hull(boundary_points_1)

            boundary_points_2 = geom.geom2.random_boundary_points(n_boundary, random=boundary_distribution)
            connectivity_2 = tr.convex_hull(boundary_points_2)

        pts = np.vstack([boundary_points_1, boundary_points_2])
        seg = np.vstack([connectivity_1, connectivity_2 + connectivity_1.shape[0]])

        domain = dict(vertices=pts, segments=seg, holes=holes)
        
        if max_mesh_area is None:
            domain_size_1 = (geom.geom1.bbox[1]-geom.geom1.bbox[0]).min()
            domain_size_2 = (geom.geom2.bbox[1]-geom.geom2.bbox[0]).min()
            max_mesh_area = min(domain_size_1,domain_size_2)/100

        max_mesh_area_str = f"qpa{max_mesh_area}"
   
    else:
        if boundary_distribution == "uniform":
            boundary_points = geom.uniform_boundary_points(n_boundary)
        else:
            boundary_points = geom.random_boundary_points(n_boundary, random=boundary_distribution)
        
        domain = dict(vertices=boundary_points)

        if max_mesh_area is None:
            max_mesh_area = (geom.bbox[1]-geom.bbox[0]).min()/100  
        
        max_mesh_area_str = f"qa{max_mesh_area}"

    mesh = tr.triangulate(domain, max_mesh_area_str)

    return mesh['vertices'], mesh['triangles']

def postProcess(model, X, triangles, output_name="displacement", operator=None, operator_name="stress", polar_transf = False, file_path=None):
    '''
    Generates the vtu file to visualize results.

    Parameters
    ----------
    model : Class object
        Trained model
    X: numpy array
        The location of vertices
    triangles: numpy array
        Mesh as triangle
    output_name: str
        Name of the output quantity
    operator: function
        The operator function which will be used to predict function quantity 
    operator_name: str
        Name of the function output quantity
    polar_transf: booelan
        Activates if polar transformation will be done
    file_path: str
        The full file path to store the results
    '''
    output = model.predict(X)

    # check if the ouput 1d or 2d/3d or larger
    if output.shape[1] == 1:
        pointData = {output_name : output.flatten()}
    elif output.shape[1] == 2:
        output = tuple(np.vstack((np.array(output[:,0].tolist()),np.array(output[:,1].tolist()),np.zeros(output[:,0].shape[0]))))
        pointData = {output_name : output}
    elif output.shape[1] == 3:
        output = tuple(np.vstack((np.array(output[:,0].tolist()),np.array(output[:,1].tolist()),np.array(output[:,2].tolist()))))
        pointData = {output_name : output}
    else:
        raise ValueError("Output dimension can not be larger than 3. If you have time as dimension, change the source code!")
    
    if operator is not None:
        operator_output = model.predict(X, operator=operator)
        if len(operator_output) == 1:
            pointData[operator_name] = operator_output.flatten()
        elif len(operator_output) == 2:
            operator_output = tuple(np.vstack((np.array(operator_output[0].flatten()),np.array(operator_output[1].flatten()),np.zeros(operator_output[0].shape[0]))))
            pointData[operator_name] = operator_output
        elif len(operator_output) == 3:
            operator_output = tuple(np.vstack((np.array(operator_output[0].flatten()),np.array(operator_output[1].flatten()),np.array(operator_output[2].flatten()))))
            pointData[operator_name] = operator_output
        else:
            warnings.warn("The dimension of the output of operator dimension can not be larger than 3!. Operator results will not be stored!")

    if (polar_transf) and (operator is not None):
        if len(operator_output) == 3:
            operator_output_transformed = polar_transformation_2d(operator_output[0], operator_output[1], operator_output[2], X)
            pointData[operator_name+"_polar"] = operator_output_transformed
        else:
            warnings.warn(f"Transformation for {len(operator_output)}D not implemented")

    if file_path is None:
        file_path = os.path.join(os.getcwd(),"default_result_name")

    solutionFieldOnMeshToVtk(X, triangles, pointData, file_path)

def solutionFieldOnMeshToVtk(X, triangles, pointData, file_path):
    '''
    Creates the vtu file.

    Parameters
    ----------
    X: numpy array
        The location of vertices
    triangles: numpy array
        Mesh as triangle
    point_data: numpy array
        Visualized the result as a point cloud 
    file_path: str
        The full file path to store the results
    '''
    
    from pyevtk.hl import unstructuredGridToVTK
    
    x = X[:,0].flatten()
    y = X[:,1].flatten()
    z = np.zeros(y.shape)
    
    dol_triangles = triangles
    offset = np.arange(3,dol_triangles.shape[0]*dol_triangles.shape[1]+1,dol_triangles.shape[1]).astype(dol_triangles.dtype)
    cell_types = np.ones(dol_triangles.shape[0])*5
    
    unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                          cell_types, pointData=pointData)