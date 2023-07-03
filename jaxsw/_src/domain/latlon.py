import typing as tp

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from jaxsw._src.domain.base import Domain
from jaxsw._src.utils.constants import R_EARTH
from jaxsw._src.utils.coriolis import beta_plane, coriolis_param


class LatLonDomain(Domain):
    f0: float = eqx.static_field()
    beta: float = eqx.static_field()

    def __init__(self, lat: Array, lon: Array):
        # get latlon delts
        dx, dy = lat_lon_deltas(lon=lon, lat=lat)

        # get mean lat (for parameters)
        lat0 = jnp.mean(lat)

        xmin, ymin = lon.min(), lat.min()
        xmax, ymax = lon.max(), lat.max()

        self.xmin = (xmin, ymin)
        self.xmax = (xmax, ymax)
        self.f0 = coriolis_param(lat0)
        self.beta = beta_plane(lat0)
        self.dx = (dx, dy)

    @property
    def coords(self) -> tp.List:
        def make_coords(xmin, xmax, delta):
            return jnp.arange(xmin, xmax, delta)

        return list(map(make_coords, self.xmin, self.xmax, self.dx))

    @property
    def dx_mean(self):
        return jnp.mean(jnp.asarray(self.dx))


class LatLonMeanDomain(LatLonDomain):
    def __init__(self, lat: Array, lon: Array):
        # get latlon delts
        dx, dy = lat_lon_deltas(lon=lon, lat=lat)

        # get mean lat (for parameters)
        lat0 = jnp.mean(lat)

        dx, dy = jnp.mean(dx), jnp.mean(dy)

        xmin, ymin = 0.0, 0.0
        xmax, ymax = dx * lon.size, dy * lat.size

        self.xmin = (xmin, ymin)
        self.xmax = (xmax, ymax)
        self.f0 = coriolis_param(lat0)
        self.beta = beta_plane(lat0)
        self.dx = (dx, dy)


def lat_lon_deltas(
    lon: Array, lat: Array, radius: float = R_EARTH
) -> tp.Tuple[Array, Array]:
    """Calculates the dx,dy for lon/lat coordinates. Uses
    the spherical Earth projected onto a plane approx.

    Eqn:
        d = R √ [Δϕ² + cos(ϕₘ)Δλ]

        Δϕ - change in latitude
        Δλ - change in longitude
        ϕₘ - mean latitude
        R - radius of the Earth

    Args:
        lon (Array): the longitude coordinates [degrees]
        lat (Array): the latitude coordinates [degrees]

    Returns:
        dx (Array): the change in x [m]
        dy (Array): the change in y [m]

    Resources:
        https://en.wikipedia.org/wiki/Geographical_distance#Spherical_Earth_projected_to_a_plane

    """

    assert lon.ndim == lat.ndim
    assert lon.ndim > 0 and lon.ndim < 3

    if lon.ndim < 2:
        lon, lat = jnp.meshgrid(lon, lat, indexing="ij")

    lon = jnp.deg2rad(lon)
    lat = jnp.deg2rad(lat)

    lat_mean = jnp.mean(lat)

    dlon_dx, dlon_dy = jnp.gradient(lon)
    dlat_dx, dlat_dy = jnp.gradient(lat)

    dx = radius * jnp.hypot(dlat_dx, dlon_dx * jnp.cos(lat_mean))
    dy = radius * jnp.hypot(dlat_dy, dlon_dy * jnp.cos(lat_mean))

    return dx, dy
