# from .functional.conv import fd_convolve
import typing as tp
import equinox as eqx
import finitediffx as fdx
from jaxtyping import Array

from jaxsw._src.fields.base import Field


class Difference(eqx.Module):
    axis: int = eqx.static_field()
    accuracy: int = eqx.static_field()
    derivative: int = eqx.static_field()
    method: str = eqx.static_field()

    def __init__(
        self,
        axis: int = 0,
        accuracy: int = 1,
        derivative: int = 1,
        method: str = "central",
    ):
        self.axis = axis
        self.accuracy = accuracy
        self.derivative = derivative
        self.method = method

    def __call__(
        self, u: Field, step_size: tp.Optional[Array] = None, axis: int = None
    ) -> Field:
        axis = self.axis if axis is None else axis
        step_size = u.domain.dx[self.axis] if step_size is None else step_size

        out = fdx.difference(
            u.values,
            axis=axis,
            accuracy=self.accuracy,
            step_size=step_size,
            derivative=self.derivative,
            method=self.method,
        )
        assert out.shape == u.values.shape

        return eqx.tree_at(lambda x: x.values, u, out)


# from finitediffx._src.utils import _check_and_return
# from .functional.fd import (
#     generate_central_diff,
#     generate_backward_diff,
#     generate_forward_diff,
#     finite_difference,
# )
# from .functional.padding import (
#     generate_forward_padding,
#     generate_central_padding,
#     generate_backward_padding,
# )
# from .functional.conv import fd_convolution, fd_kernel_init


# class Derivative(eqx.Module):
#     derivative: int = eqx.static_field()
#     accuracy: int = eqx.static_field()
#     step_size: int = eqx.static_field()
#     mode: str = eqx.static_field()
#     method: str = eqx.static_field()
#     coeffs: tp.Sequence[int] = eqx.static_field()
#     offsets: tp.Sequence[int] = eqx.static_field()
#     padding: tp.Optional[tp.Sequence[int]] = eqx.static_field()
#     axis: int = eqx.static_field()

#     def __init__(
#         self,
#         step_size: int,
#         axis: int = 0,
#         derivative: int = 1,
#         accuracy: int = 8,
#         method: str = "central",
#         mode: str = "edge",
#     ):
#         if method == "central":
#             offsets, coeffs, padding = generate_central_diff(derivative, accuracy)

#         elif method == "forward":
#             offsets, coeffs, padding = generate_forward_diff(derivative, accuracy)

#         elif method == "backward":
#             offsets, coeffs, padding = generate_backward_diff(derivative, accuracy)

#         elif method == "central_mixed":
#             raise NotImplementedError
#         else:
#             raise ValueError(f"Unrecognized method: {method}")

#         self.derivative = derivative
#         self.accuracy = accuracy
#         self.coeffs = coeffs
#         self.offsets = offsets
#         self.padding = padding
#         self.method = method
#         self.step_size = step_size
#         self.axis = axis
#         self.mode = mode

#     def __call__(self, x: Array) -> Array:
#         return finite_difference(
#             x,
#             axis=self.axis,
#             derivative=self.derivative,
#             step_size=self.step_size,
#             coeffs=self.coeffs,
#             offsets=self.offsets,
#             padding=self.padding,
#             mode=self.mode,
#         )


# class ConvDerivative(eqx.Module):
#     derivative: int = eqx.static_field()
#     accuracy: int = eqx.static_field()
#     step_size: tp.Iterable[int] = eqx.static_field()
#     mode: str = eqx.static_field()
#     method: str = eqx.static_field()
#     kernel: Array = eqx.static_field()
#     padding: tp.Optional[tp.Sequence[int]] = eqx.static_field()
#     axis: int = eqx.static_field()

#     def __init__(
#         self,
#         step_size: tp.Tuple[int],
#         axis: int = 0,
#         derivative: int = 1,
#         accuracy: int = 8,
#         method: str = "central",
#         mode: str = "edge",
#     ):
#         if method == "central":
#             _, coeffs, padding = generate_central_diff(derivative, accuracy)

#         elif method == "forward":
#             _, coeffs, padding = generate_forward_diff(derivative, accuracy)

#         elif method == "backward":
#             _, coeffs, padding = generate_backward_diff(derivative, accuracy)

#         elif method == "central_mixed":
#             raise NotImplementedError
#         else:
#             raise ValueError(f"Unrecognized method: {method}")

#         self.derivative = derivative
#         self.accuracy = accuracy
#         self.padding = padding
#         self.method = method
#         self.kernel = fd_kernel_init(dims=step_size, coeffs=coeffs, axis=axis)
#         self.axis = axis
#         self.mode = mode
#         self.step_size = step_size

#     def __call__(self, x: Array) -> Array:
#         return fd_convolution(x, kernel=self.kernel, pad=self.padding, mode=self.mode)
