import typing as tp
from jaxtyping import Array
import jax.numpy as jnp
import jax.scipy as jsp


def fd_kernel_init(dx: tp.Iterable[int], coeffs: tp.Tuple[int], axis: int=0) -> Array:
    
    # add kernel dims
    kernel = _add_kernel_dims(kernel, len(dx))
    
    # add dx
    kernel = kernel / dx[axis]
    
    # flip (convolutions are the opposite)
    kernel = jnp.flip(kernel, axis=tuple(range(kernel.ndim)))
    
    return kernel
    

def fd_convolution(
    x: Array, 
    kernel: Array, 
    pad: tp.Optional[tp.Tuple[int]]=None, 
    mode: str="edge"
) -> Array:
    
    if pad is not None:
        x = jnp.pad(x, pad_width=pad, mode=mode)
    
    return jsp.signal.convolve(x, kernel, mode="valid")