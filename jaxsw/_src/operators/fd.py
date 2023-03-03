from jaxtyping import Array
import typing as tp
from .functional.fd import (
    fd_forward_init, fd_central_init, fd_backward_init
)
from .functional.conv import fd_convolve
import equinox as eqx
from finitediffx._src.utils import _check_and_return


class Derivative(eqx.Module):
    kernel: Array = eqx.static_field()
    padding: Array = eqx.static_field()
    axis: int = eqx.static_field()
    derivative: int = eqx.static_field()
    accuracy: int = eqx.static_field()
    dx: tp.Tuple[int] = eqx.static_field()
    mode: str = eqx.static_field()
    method: str = eqx.static_field()
    
    
    def __init__(
        self, 
        dx: tp.Tuple[int],
        axis: int,
        method: str="forward",
        derivative: int=1,
        accuracy: int=8,
        mode: str="edge"):
        
        self.axis = axis
        self.derivative = derivative
        self.accuracy = accuracy #_check_and_return(accuracy)
        self.dx = dx
        self.mode = mode
        self.method = method
        
        if method == "forward":
            f = fd_forward_init
        elif method == "central":
            f = fd_central_init
        elif method == "backward":
            f = fd_backward_init
        else:
            raise ValueError(f"Unrecognized FD method: {method}")
        
        self.kernel, self.padding = f(
            dx=dx, 
            axis=axis, 
            accuracy=accuracy, 
            derivative=derivative
        )
        
    
    def __call__(self, x):
        return fd_convolve(x, self.kernel, self.padding, mode="edge")
    