import deepxde.backend as bkd


def determinant(tensor):
    """Calculate the determinant of a given tensor.
    
    Parameters
    ----------
    tensor: Tensor
        The (batch of) tensors for which a determinant is supposed to be computed.

    Returns
    -------
    The (batch of) tensor determinant(s).
    """
    assert bkd.ndim(tensor) == 3, "determinant() requires a batch of rank 2 tensors."
    return bkd.lib.linalg.det(tensor)

def identity(dim):
    """Generate an identity tensor of a given dimension.

    Parameters
    ----------
    dim: int
        The dimension of the identity tensor.

    Returns
    -------
    An identity tensor of the given dimension.
    """
    return bkd.lib.eye(dim)

def identity_like(tensor):
    """ Generate an identity tensor with the same shape as the input tensor.

    Parameters
    ----------
    tensor: Tensor
        A (batch of) tensor(s) to determine the shape of the identity tensor.

    Returns
    -------
    An identity tensor with the same shape as each tensor in the input batch.
    """
    assert bkd.ndim(tensor) == 3, "identity_like() requires a batch of rank 2 tensors."
    return identity(bkd.shape(tensor)[-1])

def inverse(tensor):
    """Calculates the inverse of a given tensor.

    Parameters
    ----------
    tensor: Tensor
        The (batch of) tensor(s) to be inverted.

    Returns
    -------
    The (batch of) inverted tensor(s).
    """
    assert bkd.ndim(tensor) == 3, "inverse() requires a batch of rank 2 tensors."
    return bkd.lib.linalg.inv(tensor)

def transpose(tensor):
    """Calculates the transpose of a given tensor.

    Parameters
    ----------
    tensor: Tensor
        The (batch of) tensor(s) to be transposed.

    Returns
    -------
    The (batch of) transposed tensor(s).
    """
    assert bkd.ndim(tensor) == 3, "transpose() requires a batch of rank 2 tensors."
    return bkd.transpose(tensor, axes=[0, 2, 1])