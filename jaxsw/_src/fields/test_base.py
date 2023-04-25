import jax.numpy as jnp
import numpy as np
import pytest

from ..domain.base import Domain
from .base import Field
from .utils import DiscretizationError, check_discretization


@pytest.fixture
def domain_1d_params():
    xmin = (0.0,)
    xmax = (2.0,)
    nx = (21,)
    return xmin, xmax, nx


@pytest.fixture
def domain_1d(domain_1d_params):
    domain = Domain.from_numpoints(
        xmin=domain_1d_params[0], xmax=domain_1d_params[1], N=domain_1d_params[2]
    )
    return domain


@pytest.fixture
def field_1d(domain_1d):
    domain = domain_1d

    u = jnp.ones_like(domain.grid, dtype=jnp.float64)

    u = u.at[int(0.5 / domain.dx[0]) : int(1 / domain.dx[0] + 1)].set(2.0)

    u = Field(u, domain)

    return u


def test_passes(domain_1d):
    domain1 = domain_1d
    xmin = 0.0
    xmax = 2.0
    nx = 21
    domain2 = Domain.from_numpoints(xmin=(xmin,), xmax=(xmax,), N=(nx,))
    try:
        check_discretization(domain1, domain2)
    except DiscretizationError as exc:
        assert False, f"domain1,domain 2 raised an exception {exc}"


def test_fails():
    # alternative way
    xmin = 0.0
    xmax = 2.0
    nx = 21
    domain1 = Domain.from_numpoints(xmin=(xmin,), xmax=(xmax,), N=(nx,))

    xmin = 0.0
    xmax = 1.0
    nx = 21
    domain2 = Domain.from_numpoints(xmin=(xmin,), xmax=(xmax,), N=(nx,))
    with pytest.raises(DiscretizationError):
        check_discretization(domain1, domain2)


def test_field_operations_1d(domain_1d):
    domain = domain_1d
    """Initial condition from grid"""
    u = jnp.ones_like(domain.grid, dtype=jnp.float64)

    u = u.at[int(0.5 / domain.dx[0]) : int(1 / domain.dx[0] + 1)].set(2.0)

    u1 = Field(u, domain)
    u2 = Field(u, domain)

    np.testing.assert_array_equal(u1.values + u2.values, (u1 + u2).values)
    np.testing.assert_array_equal(u1.values * u2.values, (u1 * u2).values)
    np.testing.assert_array_equal(u1.values - u2.values, (u1 - u2).values)
    np.testing.assert_array_equal(u1.values**2, (u1**2).values)
    np.testing.assert_array_equal(-u1.values, (-u1).values)
    np.testing.assert_array_equal(u1.values / u2.values, (u1 / u2).values)
    np.testing.assert_array_equal(1 / u1.values, (u1.inverse()).values)


def test_field_replace_values_1d(field_1d):
    """Initial condition from grid"""

    u = field_1d

    u_replace = jnp.ones_like(u.values)

    u_new = u.replace_values(u_replace)

    np.testing.assert_array_equal(u_new.values, u_replace)
