from deepxde.backend import tf, torch
from utils.geometry.geometry_utils import calculate_boundary_normals
from utils.elasticity.elasticity_utils import calculate_traction_mixed_formulation
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


def zero_complimentarity_function_based_fischer_burmeister(x,y,X):
    '''
    Enforces KKT conditions using a complimentarity function called Fischer-Burmeister based on ref https://www.math.uwaterloo.ca/~ltuncel/publications/corr2007-17.pdf.
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