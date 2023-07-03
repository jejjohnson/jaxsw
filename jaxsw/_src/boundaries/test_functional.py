# import pytest
from jaxsw._src.boundaries.functional import (
    apply_periodic_x,
    apply_periodic_y,
    # apply_periodic_pad_ND,
    apply_periodic_pad_1D,
    apply_periodic_pad_2D,
    # apply_neumann_pad_ND,
    apply_neumann_pad_1D,
    apply_neumann_pad_2D,
    apply_neumann_x,
    apply_neumann_y,
)
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

# import einops


KEY = jrandom.PRNGKey(123)
U_1D = jrandom.normal(key=KEY, shape=(10,))
U_2D = jrandom.normal(key=KEY, shape=(10, 20))
# U_3D = jrandom.normal(key=KEY, shape=(10, 20, 15))


f_periodic_1D = lambda u: apply_periodic_x(u)
f_periodic_2D = lambda u: apply_periodic_y(apply_periodic_x(u))
# f_periodic_3D = lambda u: apply_periodic_z(apply_periodic_y(apply_periodic_x(u)))

f_neumann_1D = lambda u: apply_neumann_x(u)
f_neumann_2D = lambda u: apply_neumann_y(apply_neumann_x(u))


# f_neumann_1D = lambda u: apply_neumann_x(u)
# f_neumann_2D = lambda u: apply_neumann_y(apply_neumann_x(u))
# f_neumann_3D = lambda u: apply_neumann_z(apply_neumann_y(apply_neumann_x(u)))


def test_periodic_1D():
    # pad parameters
    pad_width = (1, 1)
    mode = "empty"

    # manually create periodic BCs
    u_periodic = jnp.pad(U_1D, pad_width=pad_width, mode=mode)
    u_periodic = f_periodic_1D(u_periodic)

    # use function
    u_periodic__ = apply_periodic_pad_1D(U_1D)

    np.testing.assert_array_almost_equal(u_periodic, u_periodic__)


def test_periodic_2D():
    # pad parameters
    pad_width = ((1, 1), (1, 1))
    mode = "empty"

    # manually create periodic BCs
    u_periodic = jnp.pad(U_2D, pad_width=pad_width, mode=mode)
    u_periodic = f_periodic_2D(u_periodic)

    # use function
    u_periodic__ = apply_periodic_pad_2D(U_2D)

    np.testing.assert_array_almost_equal(u_periodic, u_periodic__)


# @pytest.mark.parameterize(
#     "U",
#     "pad_width",
#     "f",
#     [
#         (U_1D, ((1, 1)), f_periodic_1D),
#         (U_2D, ((1, 1), (1, 1)), f_periodic_2D),
#     ],
# )
# def test_periodic_ND(U, pad_width, f):
#     mode = "empty"

#     # manually create periodic BCs
#     u_periodic = jnp.pad(U, pad_width=pad_width, mode=mode)
#     u_periodic = f(u_periodic)

#     # use function
#     u_periodic_ = apply_periodic_pad_ND(U, pad_width=pad_width)

#     np.testing.assert_array_almost_equal(u_periodic, u_periodic_)


def test_neumann_1D():
    # pad parameters
    pad_width = (1, 1)
    mode = "empty"

    # manually create periodic BCs
    u_neumann = jnp.pad(U_1D, pad_width=pad_width, mode=mode)
    u_neumann = f_neumann_1D(u_neumann)

    # use function
    u_neumann_ = apply_neumann_pad_1D(U_1D)

    np.testing.assert_array_almost_equal(u_neumann, u_neumann_)


def test_neumann_2D():
    # pad parameters
    pad_width = ((1, 1), (1, 1))
    mode = "empty"

    # manually create periodic BCs
    u_neumann = jnp.pad(U_2D, pad_width=pad_width, mode=mode)
    u_neumann = f_neumann_2D(u_neumann)

    # use function
    u_neumann_ = apply_neumann_pad_2D(U_2D)

    np.testing.assert_array_almost_equal(u_neumann, u_neumann_)


# def test_neumann_ND():
#     # define field
#     u = jnp.arange(1, 11)
#     u = einops.repeat(u, "Nx -> Nx Ny", Ny=15)

#     # pad parameters
#     pad_width = ((1, 1), (1, 1))
#     mode = "empty"

#     # manually create periodic BCs
#     u_neumann = jnp.pad(u, pad_width=pad_width, mode=mode)
#     u_neumann = u_neumann.at[0].set(u_neumann[1])
#     u_neumann = u_neumann.at[-1].set(u_neumann[-2])
#     u_neumann = u_neumann.at[:, 0].set(u_neumann[:, 1])
#     u_neumann = u_neumann.at[:, -1].set(u_neumann[:, -2])

#     # use function
#     u_neumann_ = apply_neumann_pad_ND(u, pad_width=pad_width)

#     np.testing.assert_array_almost_equal(u_neumann, u_neumann_)
