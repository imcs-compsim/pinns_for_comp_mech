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

def calculate_boundary_normals_3D(X,geom):
    '''
    Calculates the boundary normals in 3D and the masked array to select the points on boundary which is quite useful to avoid unnecessary multiplications.
    This method is different compared to calculate_boundary_normals since we need to get also tangential components of the normal vector as well.

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
    tangentials_1: Tensor
        contains the first tangential vector as a tensor
    tangentials_2: Tensor
        contains the second tangential vector as a tensor
    cond: numpy boolean array
        contains a masked array to select points on boundary 
    '''
    # boundary points
    cond = geom.on_boundary(X)
    boundary_points = X[cond]

    # convert them the normal function and tangential functions to tensor functions
    boundary_normal_tensor = utils.return_tensor(geom.boundary_normal)
    boundary_tangential_1_tensor = utils.return_tensor(geom.boundary_tangential_1)
    boundary_tangential_2_tensor = utils.return_tensor(geom.boundary_tangential_2)
    
    # normals
    normals = boundary_normal_tensor(boundary_points)
    # tangentials
    tangentials_1 = boundary_tangential_1_tensor(boundary_points)
    tangentials_2 = boundary_tangential_2_tensor(boundary_points)

    return normals, tangentials_1, tangentials_2, cond


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

def polar_transformation_3d_spherical(sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz, X):
    '''
    Makes transformation from Cartesian to spherical coordinates for stress tensor in 3D.
    https://www.brown.edu/Departments/Engineering/Courses/En221/Notes/Polar_Coords/Polar_Coords.htm
    
    Parameters
    ----------
    X : numpy array
        contains the coordinates of input points in Cartesian coordinates (x, y, z)
    sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz : numpy arrays
        stress components in Cartesian coordinates

    Returns 
    -------
    sigma_rr, sigma_thetatheta, sigma_phiphi, sigma_rtheta, sigma_thetaphi, sigma_rphi : numpy arrays
        stress components in spherical coordinates
    '''
    
    # Convert Cartesian coordinates to spherical coordinates (r, theta, phi)
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    # find the point at origin
    cond = np.isclose(r,0)
        
    theta = np.arccos(z / r)  # Polar angle (theta)
    phi = np.arctan2(y, x)    # Azimuthal angle (phi)
    phi[cond] = 0  # set angle zero, since arctan2 might generate NaN
    
    # Precompute trigonometric functions
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    # sin_2theta = np.sin(2 * theta)
    # cos_2theta = np.cos(2 * theta)
    # sin_2phi = np.sin(2 * phi)
    # cos_2phi = np.cos(2 * phi)
    ##########################
    # R matrix (rotation)
    r_11 = sin_theta*cos_phi
    r_12 = sin_theta*sin_phi
    r_13 = cos_theta
    r_21 = cos_theta*cos_phi
    r_22 = cos_theta*sin_phi
    r_23 = -sin_theta
    r_31 = -sin_phi
    r_32 = cos_phi 
    r_33 = 0
    ###########################
    # S matrix (Stress)
    s_11 = sigma_xx
    s_12 = sigma_xy
    s_13 = sigma_xz 
    s_22 = sigma_yy
    s_23 = sigma_yz
    s_33 = sigma_zz
    ###########################
    # S' = R x S x R^T (polar stresses)
    ## First: R x S
    rs_11 = r_11*s_11 + r_12*s_12 + r_13*s_13
    rs_12 = r_11*s_12 + r_12*s_22 + r_13*s_23 
    rs_13 = r_11*s_13 + r_12*s_23 + r_13*s_33  
    
    rs_21 = r_21*s_11 + r_22*s_12 + r_23*s_13
    rs_22 = r_21*s_12 + r_22*s_22 + r_23*s_23 
    rs_23 = r_21*s_13 + r_22*s_23 + r_23*s_33  
    
    rs_31 = r_31*s_11 + r_32*s_12 + r_33*s_13
    rs_32 = r_31*s_12 + r_32*s_22 + r_33*s_23 
    rs_33 = r_31*s_13 + r_32*s_23 + r_33*s_33  
    
    ## Now RS x R'T
    sp_11 = rs_11 * r_11 + rs_12 * r_12 + rs_13 * r_13
    sp_12 = rs_11 * r_21 + rs_12 * r_22 + rs_13 * r_23
    sp_13 = rs_11 * r_31 + rs_12 * r_32 + rs_13 * r_33

    sp_21 = rs_21 * r_11 + rs_22 * r_12 + rs_23 * r_13
    sp_22 = rs_21 * r_21 + rs_22 * r_22 + rs_23 * r_23
    sp_23 = rs_21 * r_31 + rs_22 * r_32 + rs_23 * r_33

    sp_31 = rs_31 * r_11 + rs_32 * r_12 + rs_33 * r_13
    sp_32 = rs_31 * r_21 + rs_32 * r_22 + rs_33 * r_23
    sp_33 = rs_31 * r_31 + rs_32 * r_32 + rs_33 * r_33
    
    sigma_rr = sp_11
    sigma_rtheta = sp_12
    sigma_rphi = sp_13
    sigma_thetatheta = sp_22
    sigma_thetaphi = sp_23
    sigma_phiphi = sp_33
           
    # # Radial stress component (sigma_rr)
    # sigma_rr = (
    #     sigma_xx * sin_theta**2 * cos_phi**2 +
    #     sigma_yy * sin_theta**2 * sin_phi**2 +
    #     sigma_zz * cos_theta**2 +
    #     2 * sigma_xy * sin_theta**2 * cos_phi * sin_phi +
    #     2 * sigma_xz * sin_theta * cos_theta * cos_phi +
    #     2 * sigma_yz * sin_theta * cos_theta * sin_phi
    # )
    
    # # Polar stress component (sigma_theta)
    # sigma_theta = (
    #     sigma_xx * cos_theta**2 * cos_phi**2 +
    #     sigma_yy * cos_theta**2 * sin_phi**2 +
    #     sigma_zz * sin_theta**2 -
    #     2 * sigma_xz * sin_theta * cos_theta * cos_phi -
    #     2 * sigma_yz * sin_theta * cos_theta * sin_phi +
    #     2 * sigma_xy * cos_theta**2 * cos_phi * sin_phi
    # )
    
    # # Azimuthal stress component (sigma_phi)
    # sigma_phi = (
    #     sigma_xx * sin_phi**2 +
    #     sigma_yy * cos_phi**2 -
    #     2 * sigma_xy * cos_phi * sin_phi
    # )
    
    # # Shear stress component (sigma_rtheta)
    # sigma_rtheta = (
    #     (sigma_zz - sigma_xx) * sin_theta * cos_theta * cos_phi +
    #     (sigma_zz - sigma_yy) * sin_theta * cos_theta * sin_phi +
    #     sigma_xz * cos_2theta * cos_phi +
    #     sigma_yz * cos_2theta * sin_phi -
    #     sigma_xy * sin_2theta * cos_phi * sin_phi
    # )
    
    # # Shear stress component (sigma_rphi)
    # sigma_rphi = (
    #     (sigma_yy - sigma_xx) * sin_theta * cos_theta * cos_2phi +
    #     sigma_xy * cos_2theta * cos_2phi
    # )
    
    # # Shear stress component (sigma_thetaphi)
    # sigma_thetaphi = (
    #     (sigma_yz - sigma_xz) * sin_2theta * cos_phi * sin_phi
    # )
    
    return sigma_rr, sigma_thetatheta, sigma_phiphi, sigma_rtheta, sigma_thetaphi, sigma_rphi

def polar_transformation_3d_cylindrical(sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz, X):
    '''
    Makes transformation from Cartesian to cylindrical coordinates for stress tensor in 3D.
    https://www.brown.edu/Departments/Engineering/Courses/En221/Notes/Polar_Coords/Polar_Coords.htm
    
    Parameters
    ----------
    X : numpy array
        contains the coordinates of input points in Cartesian coordinates (x, y, z)
    sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz : numpy arrays
        stress components in Cartesian coordinates

    Returns 
    -------
    sigma_rr, sigma_thetatheta, sigma_zz, sigma_rtheta, sigma_thetaz, sigma_rz : numpy arrays
        stress components in cylindrical coordinates
    '''
    
    # Convert Cartesian coordinates to cylindrical coordinates (r, theta, phi)
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    r = np.sqrt(x**2 + y**2)
    # find the point at origin
    cond = np.isclose(r,0)
        
    theta = np.arctan2(y, x) # Rotation angle theta
    theta[cond] = 0 # set angle zero at origin, since arctan2 might generate NaN: https://stackoverflow.com/questions/47909048/what-will-be-atan2-output-for-both-x-and-y-as-0
    
    # Precompute trigonometric functions
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    ##########################
    # R matrix (rotation)
    r_11 = cos_theta
    r_12 = sin_theta
    r_13 = 0
    r_21 = -sin_theta
    r_22 = cos_theta
    r_23 = 0
    r_31 = 0
    r_32 = 0 
    r_33 = 1
    ###########################
    # S matrix (stress)
    s_11 = sigma_xx
    s_12 = sigma_xy
    s_13 = sigma_xz 
    s_22 = sigma_yy
    s_23 = sigma_yz
    s_33 = sigma_zz
    ###########################
    # S' = R x S x R^T (polar stresses)
    ## First: R x S
    rs_11 = r_11*s_11 + r_12*s_12 + r_13*s_13
    rs_12 = r_11*s_12 + r_12*s_22 + r_13*s_23 
    rs_13 = r_11*s_13 + r_12*s_23 + r_13*s_33  
    
    rs_21 = r_21*s_11 + r_22*s_12 + r_23*s_13
    rs_22 = r_21*s_12 + r_22*s_22 + r_23*s_23 
    rs_23 = r_21*s_13 + r_22*s_23 + r_23*s_33  
    
    rs_31 = r_31*s_11 + r_32*s_12 + r_33*s_13
    rs_32 = r_31*s_12 + r_32*s_22 + r_33*s_23 
    rs_33 = r_31*s_13 + r_32*s_23 + r_33*s_33  
    
    ## Now RS x R'T
    sp_11 = rs_11 * r_11 + rs_12 * r_12 + rs_13 * r_13
    sp_12 = rs_11 * r_21 + rs_12 * r_22 + rs_13 * r_23
    sp_13 = rs_11 * r_31 + rs_12 * r_32 + rs_13 * r_33

    sp_21 = rs_21 * r_11 + rs_22 * r_12 + rs_23 * r_13
    sp_22 = rs_21 * r_21 + rs_22 * r_22 + rs_23 * r_23
    sp_23 = rs_21 * r_31 + rs_22 * r_32 + rs_23 * r_33

    sp_31 = rs_31 * r_11 + rs_32 * r_12 + rs_33 * r_13
    sp_32 = rs_31 * r_21 + rs_32 * r_22 + rs_33 * r_23
    sp_33 = rs_31 * r_31 + rs_32 * r_32 + rs_33 * r_33
    
    sigma_rr = sp_11
    sigma_rtheta = sp_12
    sigma_rz = sp_13
    sigma_thetatheta = sp_22
    sigma_thetaz = sp_23
    sigma_zz = sp_33
    
    return sigma_rr, sigma_thetatheta, sigma_zz, sigma_rtheta, sigma_thetaz, sigma_rz