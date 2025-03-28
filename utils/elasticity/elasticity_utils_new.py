from abc import ABC, abstractmethod

import deepxde as dde
import deepxde.backend as bkd

from utils.linalg.linalg_utils import \
    identity_like as _identity_like, \
    trace as _trace, \
    transpose as _transpose


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
    _def_grad = disp_grad + _identity_like(disp_grad)
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
    _rcg = bkd.matmul(_transpose(def_grad), def_grad)
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
    _lcg = bkd.matmul(def_grad, _transpose(def_grad))
    return _lcg


class Strain:
    r"""Class for strain computations.
    
    The class provides a general interface for evaluating the strain tensor.
    The convention for this class is that the strain tensor is a function of 
    the deformation gradient tensor, i.e.,

    .. math::

            \mathbf{E} = \mathbf{E}(\mathbf{F})

    Attributes
    ----------
    _measure: Callable
        The method for evaluating the strain tensor.
    """

    def __init__(self, measure=None):
        """Initialize the class."""
        self._measure = measure

    def __call__(self, def_grad, **kwds):
        """General interface for evaluating the strain tensor.
        
        Passes all arguments to member :py:attr:`_measure` which 
        implements the actual computation of the strain tensor.
        """
        return self._measure(def_grad, **kwds)

    def select(self, method):
        """Select the method for evaluating the strain tensor.
        
        Parameters
        ----------
        method: Callable
            The method to evaluate the strain tensor.
        """
        self._measure = method

    @staticmethod
    def GreenLagrange(def_grad, linearized_gl=False):
        """Compute a (batch of) Green-Lagrange strain tensor(s) from a 
        (batch of) deformation gradient(s).
        
        Parameters
        ----------
        def_grad: Tensor
            The (batch of) deformation gradient tensor(s).
        linearized_gl: bool
            A flag indicating whether to use the linearized definition of the 
            Green-Lagrange strain tensor.
        
        Returns
        -------
        _strain: Tensor
            The (batch of) strain tensor(s).
        """
        if linearized_gl:
            _def_grad_T = _transpose(def_grad)
            _strain = 0.5 * (def_grad + _def_grad_T - 2.0 * _identity_like(def_grad))
        else:
            _rcg = right_cauchy_green(def_grad)
            _strain = 0.5 * (_rcg - _identity_like(_rcg))
        return _strain
    

# define the global variables which will be used in the equations
strain = Strain()


class Stress:
    """Class for stress computations."""

    def __init__(self):
        """Initialize the class."""
        self._material_law = None

    def __call__(self, def_grad, **kwds):
        """General interface for evaluating the stress tensor provided a 
        deformation gradient.
        
        Passes all arguments to member :py:attr:`_material_law` which 
        implements the actual computation of the stress tensor.
        """
        return self._material_law(def_grad, **kwds)

    def select(self, law):
        """Select the law for evaluating the stress tensor.
        
        Parameters
        ----------
        method: StressLawInterface
            The method to evaluate the stress tensor.
        """
        self._material_law = law


# define the global variables which will be used in the equations
stress = Stress()


class StressLawInterface(ABC):

    def __init__(self):
        """Initialize the class.
        
        Here you can provide the material parameters if necessary.
        """
        pass

    def __call__(self, def_grad, **kwds):
        """Public interface for computing the 2nd Piola Kirchhoff stress tensor 
        from the deformation gradient.
        
        Parameters
        ----------
        def_grad: Tensor
            The (batch of) deformation gradient tensor(s).
        
        Returns
        -------
        _stress: Tensor
            The (batch of) stress tensor(s).
        """
        return self._2pk(def_grad, **kwds)

    @abstractmethod
    def _2pk(self, def_grad, **kwds):
        """Compute the 2nd Piola Kirchhoff stress tensor from the deformation 
        gradient.

        This method has to be implemented by the derived classes.
        
        Parameters
        ----------
        def_grad: Tensor
            The (batch of) deformation gradient tensor(s).
        
        Returns
        -------
        _stress: Tensor
            The (batch of) stress tensor(s).
        """
        pass

class StVenantKirchhoff(StressLawInterface):

    def __init__(self, lamb, mu):
        """Initialize the class."""
        self._lamb = lamb
        self._mu = mu

    def _2pk(self, def_grad, **kwds):
        """Compute the 2nd Piola Kirchhoff stress tensor for a St. Venant Kirchhoff 
        material from the deformation gradient.
        """
        # compute the strain from the deformation gradient
        _strain = strain(def_grad, **kwds)
        # compute the stress from the strain
        # HINT: Since the trace operation is batched but the identity tensor 
        # is NOT batched for efficiency reasons, we have to perform an outer 
        # vector-matrix product here to broadcast the result of the 
        # multiplication to the correct dimensions
        _stress = self._lamb * bkd.lib.einsum('i,jk->ijk', _trace(_strain), _identity_like(_strain)) 
        _stress += 2.0 * self._mu * _strain
        return _stress
    