import deepxde as dde
import deepxde.backend as bkd

from utils.linalg.linalg_utils import identity_like, transpose


def displacement_gradient(disp, coords):
    r"""Compute a (batch of) displacement gradient tensor(s) from a (batch of) 
    displacement field(s) and a (batch of) coordinate(s):

    .. math::

            (\nabla \mathbf{u})_{i,j} = \frac{\partial u_i}{\partial x_j}
    
    Parameters
    ----------
    disp: Tensor
        The (batch of) displacement field(s).
    coords: Tensor
        The (batch of) coordinate(s).
    
    Returns
    -------
    _disp_grad: Tensor
        The (batch of) displacement gradient tensor(s).
    """
    assert bkd.ndim(disp) == 2, \
        "displacement_gradient() requires the displacement field to be a batch of rank 1 tensors (vectors)."
    assert bkd.ndim(coords) == 2, \
        "displacement_gradient() requires the coordinates to be a batch of rank 1 tensors (vectors)."
    _grad_list = []
    for _i in range(bkd.shape(disp)[-1]):
        _grad_i = dde.grad.jacobian(disp, coords, i=_i, j=None)
        _grad_list.append(_grad_i)
    # consider the computed gradients as column vectors of the displacement 
    # gradient matrix and stack them accordingly
    _disp_grad = bkd.stack(_grad_list, axis=1)
    return _disp_grad


def deformation_gradient(disp_grad):
    r"""Compute a (batch of) deformation gradient tensor(s) from a (batch of) 
    displacement gradient tensor(s).

    .. math::

            \mathbf{F} = \nabla \mathbf{u} + \mathbf{I}
    
    Parameters
    ----------
    def_grad: Tensor
        The (batch of) deformation gradient(s).
    
    Returns
    -------
    _deformation_gradient: Tensor
        The (batch of) deformation gradient tensor(s).
    """
    _def_grad = disp_grad + identity_like(disp_grad)
    return _def_grad


def right_cauchy_green(def_grad):
    r"""Compute a (batch of) right Cauchy-Green tensor(s) from a (batch of) 
    deformation gradient(s).

    .. math::

            \mathbf{C} = \mathbf{F}^T \mathbf{F}
    
    Parameters
    ----------
    def_grad: Tensor
        The (batch of) deformation gradient(s).
    
    Returns
    -------
    _right_cauchy_green: Tensor
        The (batch of) right Cauchy-Green tensor(s).
    """
    _rcg = bkd.matmul(transpose(def_grad), def_grad)
    return _rcg


def left_cauchy_green(def_grad):
    """Compute a (batch of) left Cauchy-Green tensor(s) from a (batch of) 
    deformation gradient(s).

    .. math::

            \mathbf{b} = \mathbf{F} \mathbf{F}^T
    
    Parameters
    ----------
    def_grad: Tensor
        The (batch of) deformation gradient(s).
    
    Returns
    -------
    _left_cauchy_green: Tensor
        The (batch of) left Cauchy-Green tensor(s).
    """
    _lcg = bkd.matmul(def_grad, transpose(def_grad))
    return _lcg


class Strain:
    """Class for strain computations."""

    def __init__(self):
        """Initialize the class."""
        self._evaluation_method = None

    def __call__(self, *args, **kwds):
        """General interface for evaluating the strain tensor.
        
        Passes all arguments to member :py:attr:`_evaluation_method` which 
        implements the actual computation of the strain tensor.
        """
        return self._evaluation_method(*args, **kwds)

    def select(self, method):
        """Select the method for evaluating the strain tensor.
        
        Parameters
        ----------
        method: Callable
            The method to evaluate the strain tensor.
        """
        self._evaluation_method = method

    @staticmethod
    def GreenLagrange(def_grad, linearize=False):
        """Compute a (batch of) Green-Lagrange strain tensor(s) from a 
        (batch of) deformation gradient(s).
        
        Parameters
        ----------
        def_grad: Tensor
            The (batch of) deformation gradient tensor(s).
        linearize: bool
            A flag indicating whether to use the linearized definition of the 
            Green-Lagrange strain tensor.
        
        Returns
        -------
        _strain: Tensor
            The (batch of) strain tensor(s).
        """
        if linearize:
            _def_grad_T = transpose(def_grad)
            _strain = 0.5 * (def_grad + _def_grad_T - 2.0 * identity_like(def_grad))
        else:
            _rcg = right_cauchy_green(def_grad)
            _strain = 0.5 * (_rcg - identity_like(_rcg))
        return _strain
    

# define the global variables which will be used in the equations
strain = Strain()