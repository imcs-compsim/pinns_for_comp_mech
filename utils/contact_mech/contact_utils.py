from deepxde.backend import tf, torch
from utils.geometry.geometry_utils import calculate_boundary_normals, calculate_boundary_normals_3D
from utils.elasticity.elasticity_utils import calculate_traction_mixed_formulation, get_tractions_mixed_3d
from deepxde import backend as bkd
from deepxde.backend import backend_name
# Global contact parameters
geom = None 
distance = None
c_complementarity = None
delta_gap = None
delta_pressure = None
projection_plane = None

backend_options = {"pytorch" : torch,
                   "tensorflow.compat.v1" : tf,
                   "tensorflow" : tf}

def adopted_sigmoid(delta,x):
    '''
    Adopted sigmoid function ref: https://arxiv.org/abs/2203.09789
    
    Parameters
    ----------
    x : tensor
        the input arguments (coordinates x and y)
    delta: float
        a scalar whihch scales 
    Returns
    -------
    gap_n: tensor
        gap in normal direction
    '''
    return 1/(1+tf.exp(-delta*x))

def calculate_gap_in_normal_direction(x,y,X):
    '''
    Calculates the gap in normal direction. 
    
    Parameters
    ----------
    x : tensor
        the input arguments (coordinates x and y)
    y: tensor
        the network output (predicted displacement in x and y direction)
    X: np.array
        the input arguments as an array (coordinates x and y)

    Returns
    -------
    gap_n: tensor
        gap in normal direction
    '''
    # calculate the gap in y direction    
    gap_y = x[:,1:2] + y[:,1:2] + distance

    # calculate the boundary normals
    normals, cond = calculate_boundary_normals(X,geom)

    # Here is the idea to calculate gap_n:
    # gap_n/|n| = gap_y/|ny| --> since n is unit vector |n|=1
    gap_n = tf.math.divide_no_nan(gap_y[cond],tf.math.abs(normals[:,1:2]))
    
    return gap_n

def positive_normal_gap_sign(x, y, X):
    '''
    Enforces normal gap (gn) to be positive using the sign function.
    KKT condition: gn>=0
    
    Parameters
    ----------
    x : tensor
        the input arguments (coordinates x and y)
    y: tensor
        the network output (predicted displacement in x and y direction)
    X: np.array
        the input arguments as an array (coordinates x and y)

    Returns
    -------
    (1.0-tf.math.sign(gn))*gn: tensor
        non-zero tensor if gn is negative, otherwise returns zero-tensor
    '''
    gn = calculate_gap_in_normal_direction(x, y, X)

    # If gn is negative, it will create contributions to overall loss. Aims is to get positive gap
    return (1.0-tf.math.sign(gn))*gn

def negative_normal_traction_sign(x,y,X):
    '''
    Enforces normal part of contact traction (Pn) to be negative using the sign function.
    KKT condition: Pn<=0
    
    Parameters
    ----------
    x : tensor
        the input arguments (coordinates x and y)
    y: tensor
        the network output (predicted displacement in x and y direction)
    X: np.array
        the input arguments as an array (coordinates x and y)

    Returns
    -------
    (1.0+tf.math.sign(Pn))*Pn: tensor
        non-zero tensor if Pn is positive, otherwise returns zero-tensor
    '''
    Tx, Ty, Pn, Tt = calculate_traction_mixed_formulation(x, y, X)

    # If Pn is positive, it will create contributions to overall loss. Aims is to get negative normal traction
    return (1.0+tf.math.sign(Pn))*Pn

def positive_normal_gap_adopted_sigmoid(x, y, X):
    '''
    Enforces normal gap (gn) to be positive using an adopted sigmoid function based on https://arxiv.org/abs/2203.09789.
    KKT condition: gn>=0
    
    Parameters
    ----------
    x : tensor
        the input arguments (coordinates x and y)
    y: tensor
        the network output (predicted displacement in x and y direction)
    X: np.array
        the input arguments as an array (coordinates x and y)

    Returns
    -------
    adopted_sigmoid(delta_gap,-gn)*gn: tensor
        non-zero tensor if gn is negative, otherwise returns zero-tensor (not a sharp function)
    '''
    gn = calculate_gap_in_normal_direction(x, y, X)

    # If gn is negative, it will create contributions to overall loss. Aims is to get positive gap
    return adopted_sigmoid(delta_gap,-gn)*gn

def negative_normal_traction_adopted_sigmoid(x,y,X):
    '''
    Enforces normal part of contact traction (Pn) to be negative using an adopted sigmoid function based on https://arxiv.org/abs/2203.09789.
    KKT condition: Pn<=0
    
    Parameters
    ----------
    x : tensor
        the input arguments (coordinates x and y)
    y: tensor
        the network output (predicted displacement in x and y direction)
    X: np.array
        the input arguments as an array (coordinates x and y)

    Returns
    -------
    adopted_sigmoid(delta_pressure,Pn)*Pn: tensor
        non-zero tensor if Pn is positive, otherwise returns zero-tensor
    '''
    Tx, Ty, Pn, Tt = calculate_traction_mixed_formulation(x, y, X)

    # If Pn is positive, it will create contributions to overall loss. Aims is to get negative normal traction
    return adopted_sigmoid(delta_pressure,Pn)*Pn

def zero_complimentary(x,y,X):
    '''
    Enforces complimentary term to be zero.
    KKT condition: gn*Pn=0
    
    Parameters
    ----------
    x : tensor
        the input arguments (coordinates x and y)
    y: tensor
        the network output (predicted displacement in x and y direction)
    X: np.array
        the input arguments as an array (coordinates x and y)
        
    Returns
    -------
    gn*Pn: tensor
        matrix multiplication between normal gap (gn) and normal pressure (Pn)
    '''
    
    Tx, Ty, Pn, Tt = calculate_traction_mixed_formulation(x, y, X)
    gn = calculate_gap_in_normal_direction(x, y, X)

    return gn*Pn

def zero_tangential_traction(x,y,X):
    '''
    Enforces tangential component of contact traction (Tt) to be zero.
    
    Parameters
    ----------
    x : tensor
        the input arguments (coordinates x and y)
    y: tensor
        the network output (predicted displacement in x and y direction)
    X: np.array
        the input arguments as an array (coordinates x and y)
        
    Returns
    -------
    Tt: tensor
        tangential component of contact traction (Tt)
    '''
    
    Tx, Ty, Pn, Tt = calculate_traction_mixed_formulation(x, y, X)

    return Tt


def zero_complementarity_function_based_popp(x,y,X):
    '''
    Enforces KKT conditions using a complementarity function based on https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.2614.
    This function is mathematically equal to combination of the following functions:
        - positive_normal_gap_sign 
        - negative_normal_traction_sign
        - zero_complimentary
        
    Parameters
    ----------
    x : tensor
        the input arguments (coordinates x and y)
    y: tensor
        the network output (predicted displacement in x and y direction)
    X: np.array
        the input arguments as an array (coordinates x and y)
        
    Returns
    -------
    Pn-tf.math.maximum(tf.constant(0, dtype=tf.float32), Pn-c_complementarity*gn): tensor
        -
    '''
    
    Tx, Ty, Pn, Tt = calculate_traction_mixed_formulation(x, y, X)
    gn = calculate_gap_in_normal_direction(x, y, X)

    return Pn-tf.math.maximum(tf.constant(0, dtype=tf.float32), Pn-c_complementarity*gn)

def zero_complementarity_function_based_fisher_burmeister(x,y,X):
    '''
    Enforces KKT conditions using a complementarity function called Fisher-Burmeister based on ref https://www.math.uwaterloo.ca/~ltuncel/publications/corr2007-17.pdf.
    This function is mathematically equal to combination of the following functions:
        - positive_normal_gap_sign
        - negative_normal_traction_sign
        - zero_complimentary
    
    Parameters
    ----------
    x : tensor
        the input arguments (coordinates x and y)
    y: tensor
        the network output (predicted displacement in x and y direction)
    X: np.array
        the input arguments as an array (coordinates x and y)
        
    Returns
    -------
    Pn-tf.math.maximum(tf.constant(0, dtype=tf.float32), Pn-c_complementarity*gn): tensor
        -
    '''

    Tx, Ty, Pn, Tt = calculate_traction_mixed_formulation(x, y, X)
    gn = calculate_gap_in_normal_direction(x, y, X)
    
    a = gn
    b = -Pn
    
    return a + b - tf.sqrt(tf.maximum(a**2+b**2, 1e-9))

def calculate_gap_in_normal_direction_3d(x, y, X):
    '''
    Calculates the gap in normal direction in 3D.
    
    Parameters
    ----------
    x : tensor
        The input arguments (coordinates x, y, and z).
    y : tensor
        The network output (predicted displacement in x, y, and z directions).
    X : np.array
        The input arguments as an array (coordinates x, y, and z).

    Returns
    -------
    gap_n : tensor
        Gap in normal direction in 3D.
    '''
    # Calculate the boundary normals in 3D
    normals, tangentials_1, tangentials_2, cond = calculate_boundary_normals_3D(X, geom)
    # normals 
    nx = normals[:,0:1]
    ny = normals[:,1:2]
    nz = normals[:,2:3]
    
    # Calculate the gap in all three directions: x, y, and z
    # First get the current locations using reference locations and displacements: X_i + u_i
    x_x = x[:, 0:1] + y[:, 0:1] # X_x + u_x
    x_y = x[:, 1:2] + y[:, 1:2] # X_y + u_y
    x_z = x[:, 2:3] + y[:, 2:3] # X_z + u_z
    
    # Get the projectection of the current location on to the desired projection plane
    if projection_plane.get("x") is not None:
        proj_x = projection_plane.get("x")
        proj_y = x_y#x[:, 1:2]
        proj_z = x_z#[:, 2:3]
    elif projection_plane.get("y") is not None:
        proj_x = x_x#[:, 0:1]
        proj_y = projection_plane.get("y")
        proj_z = x_z#x[:, 2:3]
    elif projection_plane.get("z") is not None:
        proj_x = x_x#x[:, 0:1]
        proj_y = x_y#x[:, 1:2]
        proj_z = projection_plane.get("z")
    
    # Calculate gaps in x, y, and z direction
    gap_x = x_x - proj_x
    gap_y = x_y - proj_y
    gap_z = x_z - proj_z

    # Calculate the gap in the normal direction
    gap_n = -(gap_x[cond]*nx + gap_y[cond]*ny + gap_z[cond]*nz)
    #gap_n = -(gap_y[cond]*ny)

    return gap_n

def calculate_gap_in_normal_direction_deep_energy(x, y, X, mapped_normal_boundary_t, cond):
    '''
    Calculates the gap in normal direction in 3D.
    
    Parameters
    ----------
    x : tensor
        The input arguments (coordinates x, y, and z).
    y : tensor
        The network output (predicted displacement in x, y, and z directions).
    X : np.array
        The input arguments as an array (coordinates x, y, and z).

    Returns
    -------
    gap_n : tensor
        Gap in normal direction in 3D.
    '''
    # normals
    nx = mapped_normal_boundary_t[:,0:1]
    ny = mapped_normal_boundary_t[:,1:2]
    if X.shape[1] == 3: 
        nz = mapped_normal_boundary_t[:,2:3]    
    # Calculate the gap in all three directions: x, y, and z
    # First get the current locations using reference locations and displacements: X_i + u_i
    x_x = x[:, 0:1] + y[:, 0:1] # X_x + u_x
    x_y = x[:, 1:2] + y[:, 1:2] # X_y + u_y
    if X.shape[1] == 3: 
        x_z = x[:, 2:3] + y[:, 2:3] # X_z + u_z
    
    # Get the projectection of the current location on to the desired projection plane
    if projection_plane.get("x") is not None:
        proj_x = projection_plane.get("x")
        proj_y = x_y#x[:, 1:2]
        if X.shape[1] == 3: 
            proj_z = x_z#[:, 2:3]
    elif projection_plane.get("y") is not None:
        proj_x = x_x#[:, 0:1]
        proj_y = projection_plane.get("y")
        if X.shape[1] == 3: 
            proj_z = x_z#x[:, 2:3]
    elif projection_plane.get("z") is not None:
        proj_x = x_x#x[:, 0:1]
        proj_y = x_y#x[:, 1:2]
        if X.shape[1] == 3: 
            proj_z = projection_plane.get("z")
    
    # Calculate gaps in x, y, and z direction
    gap_x = x_x - proj_x
    gap_y = x_y - proj_y
    if X.shape[1] == 3: 
        gap_z = x_z - proj_z

    # Calculate the gap in the normal direction
    #gap_n = -(gap_x[cond]*nx[cond] + gap_y[cond]*ny[cond])
    gap_n = gap_y[cond]
    if X.shape[1] == 3:
        gap_n = gap_n - gap_z[cond]*nz
    #gap_n = -(gap_y[cond]*ny)

    return gap_n

def zero_complementarity_function_based_fisher_burmeister_3d(x,y,X):
    '''
    Enforces KKT conditions using a complementarity function called Fisher-Burmeister based on ref https://www.math.uwaterloo.ca/~ltuncel/publications/corr2007-17.pdf.
    This function is mathematically equal to combination of the following functions:
        - positive_normal_gap_sign
        - negative_normal_traction_sign
        - zero_complimentary
    
    Parameters
    ----------
    x : tensor
        the input arguments (coordinates x and y)
    y: tensor
        the network output (predicted displacement in x and y direction)
    X: np.array
        the input arguments as an array (coordinates x and y)
        
    Returns
    -------
    Pn-tf.math.maximum(tf.constant(0, dtype=tf.float32), Pn-c_complementarity*gn): tensor
        -
    '''

    Tx, Ty, Tz, Pn, Tt_1, Tt_2  = get_tractions_mixed_3d(x, y, X)
    gn = calculate_gap_in_normal_direction_3d(x, y, X)
    
    a = gn
    b = -Pn
    
    current_backend = backend_options[backend_name]
    tol = 1e-9
    if backend_name == "pytorch":
        tol = torch.tensor(tol)

    return a + b - current_backend.sqrt(current_backend.maximum(a**2 + b**2, tol))

def zero_tangential_traction_component1_3d(x,y,X):
    '''
    Enforces 1. tangential component of contact traction (Tt) to be zero in 3D.
    
    Parameters
    ----------
    x : tensor
        the input arguments (coordinates x and y)
    y: tensor
        the network output (predicted displacement in x and y direction)
    X: np.array
        the input arguments as an array (coordinates x and y)
        
    Returns
    -------
    Tt: tensor
        tangential component of contact traction (Tt)
    '''
    
    Tx, Ty, Tz, Pn, Tt_1, Tt_2  = get_tractions_mixed_3d(x, y, X)

    return Tt_1

def zero_tangential_traction_component2_3d(x,y,X):
    '''
    Enforces 1. tangential component of contact traction (Tt) to be zero in 3D.
    
    Parameters
    ----------
    x : tensor
        the input arguments (coordinates x and y)
    y: tensor
        the network output (predicted displacement in x and y direction)
    X: np.array
        the input arguments as an array (coordinates x and y)
        
    Returns
    -------
    Tt: tensor
        tangential component of contact traction (Tt)
    '''
    
    Tx, Ty, Tz, Pn, Tt_1, Tt_2  = get_tractions_mixed_3d(x, y, X)

    return Tt_2