import typing as tp
from jaxtyping import Array
import jax.numpy as jnp
import jax.scipy as jsp


def fd_convolve(
    x: Array, 
    kernel: Array, 
    pad: tp.Optional[Array]=None, mode: str="edge") -> Array:
    
    if pad is not None:
        x = jnp.pad(x, pad_width=pad, mode=mode)
    
    kernel = jnp.flip(kernel, axis=tuple(range(kernel.ndim)))
    
    return jsp.signal.convolve(x, kernel, mode="valid")