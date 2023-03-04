from jaxtyping import Array
import typing as tp
# from .functional.conv import fd_convolve
import equinox as eqx
from finitediffx._src.utils import _check_and_return
from .functional.fd import (
    generate_central_offsets, 
    generate_backward_offsets, 
    generate_forward_offsets,
    generate_finitediff_coeffs,
    finite_difference
)
from .functional.padding import generate_forward_padding, generate_central_padding, generate_backward_padding


class Derivative(eqx.Module):
    derivative: int = eqx.static_field()
    accuracy: int = eqx.static_field()
    step_size: int = eqx.static_field()
    mode: str = eqx.static_field()
    method: str = eqx.static_field()
    coeffs: tp.Sequence[int] = eqx.static_field()
    offsets: tp.Sequence[int] = eqx.static_field()
    padding: tp.Optional[tp.Sequence[int]] = eqx.static_field()
    axis: int = eqx.static_field()

    def __init__(
        self,
        step_size: int,
        axis: int=0,
        derivative: int=1,
        accuracy: int=8, 
        method: str="central",
        mode: str="edge",
        ):
        if method == "central":
            offsets = generate_central_offsets(derivative=derivative, accuracy=accuracy)
            coeffs = generate_finitediff_coeffs(offsets=offsets, derivative=derivative)
            padding = generate_central_padding(derivative=derivative, accuracy=accuracy)
            
        elif method == "forward":
            
            offsets = generate_forward_offsets(derivative=derivative, accuracy=accuracy)
            coeffs = generate_finitediff_coeffs(offsets=offsets, derivative=derivative)
            padding = generate_forward_padding(derivative=derivative, accuracy=accuracy)
            
        elif method == "backward":
            
            offsets = generate_backward_offsets(derivative=derivative, accuracy=accuracy)
            coeffs = generate_finitediff_coeffs(offsets=offsets, derivative=derivative)
            padding = generate_backward_padding(derivative=derivative, accuracy=accuracy)
            
        elif method == "central_mixed":
            raise NotImplementedError
        else:
            raise ValueError(f"Unrecognized method: {method}")
    
        self.derivative = derivative
        self.accuracy = accuracy
        self.coeffs = coeffs
        self.offsets = offsets
        self.padding = padding
        self.method = method
        self.step_size = step_size
        self.axis = axis
        self.mode = mode

    def __call__(self, x: Array) -> Array:

        return finite_difference(
            x, 
            axis=self.axis,
            derivative=self.derivative,
            step_size=self.step_size,
            coeffs=self.coeffs,
            offsets=self.offsets,
            padding=self.padding,
            mode=self.mode
            )
        


# class Derivative(eqx.Module):
#     kernel: Array = eqx.static_field()
#     padding: Array = eqx.static_field()
#     axis: int = eqx.static_field()
#     derivative: int = eqx.static_field()
#     accuracy: int = eqx.static_field()
#     dx: tp.Tuple[int] = eqx.static_field()
#     mode: str = eqx.static_field()
#     method: str = eqx.static_field()
    
    
#     def __init__(
#         self, 
#         dx: tp.Tuple[int],
#         axis: int,
#         method: str="forward",
#         derivative: int=1,
#         accuracy: int=8,
#         mode: str="edge"):
        
#         self.axis = axis
#         self.derivative = derivative
#         self.accuracy = accuracy #_check_and_return(accuracy)
#         self.dx = dx
#         self.mode = mode
#         self.method = method
        
#         if method == "forward":
#             f = fd_forward_init
#         elif method == "central":
#             f = fd_central_init
#         elif method == "backward":
#             f = fd_backward_init
#         else:
#             raise ValueError(f"Unrecognized FD method: {method}")
        
#         self.kernel, self.padding = f(
#             dx=dx, 
#             axis=axis, 
#             accuracy=accuracy, 
#             derivative=derivative
#         )
        
    
#     def __call__(self, x):
#         return fd_convolve(x, self.kernel, self.padding, mode="edge")
    