import typing as tp
import jax
from jaxtyping import Array
import jax.numpy as jnp
import kernex as kex
import einops
import jax.numpy as jnp
import jax.scipy as jsp
from jaxsw._src.domain.utils import create_meshgrid_coordinates
from jaxsw._src.domain.base import Domain
from jaxtyping import Array
from jaxsw._src.fields.base import Field
import functools as ft
from jaxtyping import Float, Array
import equinox as eqx


class CartesianGrid:
    """
    Linear Multivariate Cartesian Grid interpolation in arbitrary dimensions. Based
    on ``map_coordinates``.

    Notes:
        Translated directly from https://github.com/JohannesBuchner/regulargrid/ to jax.
    """

    values: Array
    """
    Values to interpolate.
    """
    limits: tp.Iterable[tp.Iterable[float]]
    """
    Limits along each dimension of ``values``.
    """

    def __init__(
        self,
        xmin: tp.Iterable[tp.Iterable[float]],
        xmax: tp.Iterable[tp.Iterable[float]],
        values: Array,
        mode: str = "constant",
        cval: float = jnp.nan,
    ):
        """
        Initializer.

        Args:
            limits: collection of pairs specifying limits of input variables along
                each dimension of ``values``
            values: values to interpolate. These must be defined on a regular grid.
            mode: how to handle out of bounds arguments; see docs for ``map_coordinates``
            cval: constant fill value; see docs for ``map_coordinates``
        """
        super().__init__()
        self.values = values
        self.xmin = xmin
        self.xmax = xmax
        self.mode = mode
        self.cval = cval

    def __call__(self, *coords) -> Array:
        """
        Perform interpolation.

        Args:
            coords: point at which to interpolate. These will be broadcasted if
                they are not the same shape.

        Returns:
            Interpolated values, with extrapolation handled according to ``mode``.
        """
        # transform coords into pixel values
        coords = jnp.broadcast_arrays(*coords)

        # coords = jnp.asarray(coords)
        coords = [
            (c - lo) * (n - 1) / (hi - lo)
            for lo, hi, c, n in zip(self.xmin, self.xmax, coords, self.values.shape)
        ]

        return jax.scipy.ndimage.map_coordinates(
            self.values, coords, mode=self.mode, cval=self.cval, order=1
        )


def field_domain_transform(
    u: Field,
    domain: Domain,
    mode: str = "reflect",
    cval: float = 1.0,
    method: str = "cartesian",
):
    # initialize Cartesian Grid
    grid = CartesianGrid(
        xmin=u.domain.xmin,
        xmax=u.domain.xmax,
        values=u[:],
        mode=mode,
        cval=cval,
    )

    # interpolate points
    u_values = grid(domain.coords)

    # reshape to match grid values
    u_values = jnp.reshape(u_values, newshape=domain.coords.shape[0])

    return Field(values=u_values, domain=domain)


def interp(u: Array, axis: int = 0, method: str = "linear", **kwargs) -> Array:
    if method in ["linear", "geometric", "arithmetic", "harmonic"]:
        # return interp_linear_constant(u=u, axis=axis)
        return _interp(u=u, axis=axis, method=method, **kwargs)
    elif method == "upwind":
        return NotImplementedError()
    else:
        msg = f"Unrecognized method: {method}"
        msg += "\nMust be: 'linear', 'geometric', 'upwind'"
        raise ValueError(msg)


def interp_center(u: Array, method: str = "linear", **kwargs) -> Array:
    return interp(u=u, method=method, center=True, **kwargs)


def get_interp_fn(method: str = "linear") -> tp.Callable:
    if method == "linear":
        return lambda x: jnp.mean(x)
    elif method == "geometric":
        return lambda x: jnp.exp(jnp.mean(jnp.log(x)))
    elif method == "arithmetic":
        raise NotImplementedError()
    elif method == "upwind":
        raise NotImplementedError()
    elif method == "harmonic":
        raise NotImplementedError()
    else:
        msg = f"Unrecognized method: {method}"
        msg += "\nMust be: 'linear', 'geometric"
        raise ValueError(msg)


def get_kernel_axis(axis: int, ndim: int) -> tp.Tuple:
    assert axis >= 0
    assert ndim >= 1
    return tuple([2 if idim == axis else 1 for idim in range(ndim)])


def get_kernel_center(ndim: int) -> tp.Tuple:
    assert ndim >= 1
    return (2,) * ndim


def _interp(
    u: Array, axis: int, method: str = "linear", center: bool = False, **kwargs
) -> Array:
    padding = kwargs.get("padding", "valid")

    if not center:
        kernel_size = get_kernel_axis(axis=axis, ndim=u.ndim)
    else:
        kernel_size = get_kernel_center(u.ndim)

    interp_fn = get_interp_fn(method=method)

    @kex.kmap(kernel_size=kernel_size, padding=padding)
    def kernel_fn(u):
        return interp_fn(u)

    return kernel_fn(u)
