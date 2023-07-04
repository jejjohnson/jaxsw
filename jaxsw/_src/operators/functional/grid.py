import typing as tp
import jax.numpy as jnp
import kernex as kex
from jaxtyping import Array


def x_average_1D(u: Array, padding: tp.Optional[tp.Tuple] = "valid") -> Array:
    """Returns the two-point average at the centres between grid points.

    Grid:
        + -- ⋅ -- +
        u -- u̅ -- u
        + -- ⋅ -- +

    Args:
        u (Array): the field [Nx,]

    Returns:
        ubar (Array): the field averaged [Nx-1,]

    """

    @kex.kmap(kernel_size=(2,), padding=padding)
    def kernel_fn(u):
        return jnp.mean(u)

    return kernel_fn(u)


def x_average_2D(u: Array, padding: tp.Optional[tp.Tuple] = "valid") -> Array:
    """Returns the two-point average at the centres between grid points.

    Grid:
        u -- u̅ -- u
        |         |
        ⋅         ⋅
        |         |
        u -- u̅ -- u

    Args:
        u (Array): the field [Nx,Ny]

    Returns:
        ubar (Array): the field averaged [Nx-1, Ny]

    """

    @kex.kmap(kernel_size=(2, 1), padding=padding)
    def kernel_fn(u):
        return jnp.mean(u)

    return kernel_fn(u)


def y_average_2D(u: Array, padding: tp.Optional[tp.Tuple] = "valid") -> Array:
    """Returns the two-point average at the centres between grid points.

    Grid:
        u -- ⋅ -- u
        |         |
        u̅         u̅
        |         |
        u -- ⋅ -- u

    Args:
        u (Array): the field [Nx,Ny]

    Returns:
        ubar (Array): the field averaged [Nx, Ny-1]

    """

    @kex.kmap(kernel_size=(1, 2), padding=padding)
    def kernel_fn(u):
        return jnp.mean(u)

    return kernel_fn(u)


def center_average_2D(u: Array, padding: tp.Optional[tp.Tuple] = "valid") -> Array:
    """Returns the four-point average at the centres between grid points.

    Grid:
        u -- ⋅ -- u
        |         |
        ⋅    u̅    ⋅
        |         |
        u -- ⋅ -- u

    Args:
        u (Array): the field [Nx,Ny]

    Returns:
        ubar (Array): the field averaged [Nx-1, Ny-1]

    """

    @kex.kmap(kernel_size=(2, 2), padding=padding)
    def kernel_fn(u):
        return jnp.mean(u)

    return kernel_fn(u)


def u_at_v(u: Array) -> Array:
    return center_average_2D(u)[1:-1, :]


def v_at_u(v: Array) -> Array:
    return center_average_2D(v)[:, 1:-1]


def node_to_edge(u: Array) -> Array:
    pass


def edge_to_node(u: Array) -> Array:
    pass


def node_to_face(u: Array) -> Array:
    pass


def face_to_node(u: Array) -> Array:
    pass


def edge_to_face(u: Array) -> Array:
    pass


def face_to_edge(u: Array) -> Array:
    pass
