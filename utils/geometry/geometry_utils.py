import deepxde as dde
import numpy as np
from deepxde import utils


def calculate_boundary_normals(X,geom):
    '''
    Calculates the boundary normals and the masked array to select the points on boundary which is quite useful to avoid unnecessary multiplications.

    Parameters
    ----------
    X : numpy array
        contains the coordinates of input points
    geom : object
        contains the geometry object of the problem

    Returns 
    -------
    normals: Tensor
        contains the normal vector as a tensor
    cond: numpy boolean array
        contains a masked array to select points on boundary 
    '''
    # boundary points
    cond = geom.on_boundary(X)
    boundary_points = X[cond]

    # convert them the normal function to tensor
    boundary_normal_tensor = utils.return_tensor(geom.boundary_normal)

    # normals
    normals = boundary_normal_tensor(boundary_points)

    return normals, cond


def polar_transformation_2d(sigma_xx, sigma_yy, sigma_xy, X):
    '''
    Makes tranformation from cartesian to polar coordinates for stress tensor in 2D.

    Parameters
    ----------
    X : numpy array
        contains the coordinates of input points
    sigma_xx, sigma_yy, sigma_xy: numpy array
        stress components in cartesian coordinates

    Returns 
    -------
    sigma_rr, sigma_theta, sigma_rtheta: numpy array
       stress components in polar coordinates
    '''
    
    theta = np.degrees(np.arctan2(X[:,1],X[:,0])).reshape(-1,1) # in degree
    theta_radian = theta*np.pi/180

    sigma_rr = ((sigma_xx + sigma_yy)/2 + (sigma_xx - sigma_yy)*np.cos(2*theta_radian)/2 + sigma_xy*np.sin(2*theta_radian)).flatten()
    sigma_theta = ((sigma_xx + sigma_yy)/2 - (sigma_xx - sigma_yy)*np.cos(2*theta_radian)/2 - sigma_xy*np.sin(2*theta_radian)).flatten()
    sigma_rtheta = (-(sigma_xx - sigma_yy)*np.sin(2*theta_radian)/2 + sigma_xy*np.cos(2*theta_radian)).flatten()
    
    return sigma_rr, sigma_theta, sigma_rtheta