import typing as tp
from jaxsw._src.domain.base import Domain
import jax.numpy as jnp
from jaxtyping import Array
import equinox as eqx
from jaxsw._src.utils.coriolis import beta_plane, coriolis_param
from jaxsw._src.utils.constants import R_EARTH

class LatLonMeanDomain(Domain):
    f0: float = eqx.static_field()
    beta: float = eqx.static_field()
    
    def __init__(self, lat: Array, lon: Array):
        
        # get latlon delts
        dx, dy = lat_lon_deltas(lon=lon, lat=lat)
        
        # get mean lat (for parameters)
        lat0 = jnp.mean(lat)
        
        dx, dy = jnp.mean(dx), jnp.mean(dy)
        
        xmin, ymin = 0.0, 0.0
        xmax, ymax = dx * lat.size, dy * lon.size
        
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
        

def lat_lon_deltas(lon: Array, lat: Array) -> tp.Tuple[Array, Array]:
    
    lon_grid, lat_grid = jnp.meshgrid(lon, lat, indexing="ij")
    

    lon_grid = jnp.deg2rad(lon_grid)
    lat_grid = jnp.deg2rad(lat_grid)

    dlon_dx, dlon_dy = jnp.gradient(lon_grid)
    dlat_dx, dlat_dy = jnp.gradient(lat_grid)


    dx = R_EARTH * jnp.hypot(dlon_dx * jnp.cos(lat_grid), dlat_dx)
    dy = R_EARTH * jnp.hypot(dlon_dy * jnp.cos(lat_grid), dlat_dy)
    
    
    return dx, dy