import jax.numpy as jnp
import numpy as np
from jaxsw._src.operators.functional.grid import x_average_1D


Q_1D_EDGE = jnp.arange(1, 11)
U_1D_FACE_EXTERIOR = jnp.arange(0.5, 11.5)
U_1D_FACE_INTERIOR = jnp.arange(1.5, 10.5)


def test_x_average_1D_no_padding():
    assert len(Q_1D_EDGE) == len(U_1D_FACE_EXTERIOR) - 1

    u_on_q = x_average_1D(U_1D_FACE_EXTERIOR, padding="valid")

    np.testing.assert_array_equal(u_on_q, Q_1D_EDGE)


def test_x_average_1D_padding():
    assert len(Q_1D_EDGE) == len(U_1D_FACE_INTERIOR) + 1

    u_on_q = x_average_1D(U_1D_FACE_INTERIOR)

    np.testing.assert_array_equal(u_on_q, Q_1D_EDGE[1:-1])
