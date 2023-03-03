import typing as tp
import finitediffx as fdx
import jax.numpy as jnp
from .padding import generate_forward_padding, generate_central_padding, generate_backward_padding


def fd_central_init(dx: tp.Iterable[float], axis: int=0, accuracy: int=2, derivative: int=1) -> tp.Tuple:
    
    
    # get offsets and kernel
    offsets = fdx._src.utils._generate_central_offsets(derivative=derivative, accuracy=accuracy)
    kernel = fdx._src.utils.generate_finitediff_coeffs(offsets, derivative=derivative)
    
    # add kernel dims
    kernel = _add_kernel_dims(kernel, len(dx))
    
    # add dx
    kernel = kernel / dx[axis]
    
    # get padding
    padding = generate_central_padding(derivative=derivative, accuracy=accuracy)
    
    return kernel, padding


def fd_forward_init(dx: tp.Iterable[float], axis: int=0, accuracy: int=2, derivative: int=1) -> tp.Tuple:
    
    
    # get offsets and kernel
    offsets = fdx._src.utils._generate_forward_offsets(derivative=derivative, accuracy=accuracy)
    kernel = fdx._src.utils.generate_finitediff_coeffs(offsets, derivative=derivative)
    
    # add kernel dims
    kernel = _add_kernel_dims(kernel, len(dx))
    
    # add dx
    kernel = kernel / dx[axis]
    
    # get padding
    padding = generate_forward_padding(derivative=derivative, accuracy=accuracy)
    
    return kernel, padding


def fd_backward_init(dx: tp.Iterable[float], axis: int=0, accuracy: int=2, derivative: int=1) -> tp.Tuple:
    
    
    # get offsets and kernel
    offsets = fdx._src.utils._generate_backward_offsets(derivative=derivative, accuracy=accuracy)
    kernel = fdx._src.utils.generate_finitediff_coeffs(offsets, derivative=derivative)
    
    # add kernel dims
    kernel = _add_kernel_dims(kernel, len(dx))
    
    # add dx
    kernel = kernel / dx[axis]
    
    # get padding
    padding = generate_backward_padding(derivative=derivative, accuracy=accuracy)
    
    return kernel, padding


def _add_kernel_dims(kernel, ndim):
    
    if ndim > 1:
        for _ in range(ndim - 1):
            kernel = rearrange(kernel, "... -> ... 1")
        
        # move kernel to correct axis
        kernel = np.moveaxis(kernel, -1, axis)
        
    return kernel
    