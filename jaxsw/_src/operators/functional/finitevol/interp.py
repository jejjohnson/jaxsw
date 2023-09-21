import einops
import jax.numpy as jnp
import jax.scipy as jsp
from jaxsw._src.domain.utils import create_meshgrid_coordinates
from jaxsw._src.domain.base import Domain
from jaxtyping import Array
from jaxsw._src.fields.base import Field


def domain_transformation(u: Field, v: Field, **kwargs) -> Field:
    assert u.domain.ndim == v.domain.ndim

    if u.domain.ndim == 1:
        return Field(
            values=_domain_transformation_1D(u.values, u.domain, v.domain, **kwargs),
            domain=v.domain,
        )
    elif u.domain.ndim == 2:
        return Field(
            values=_domain_transformation_2D(u.values, u.domain, v.domain, **kwargs),
            domain=v.domain,
        )
    elif u.domain.ndim == 3:
        return Field(
            values=_domain_transformation_3D(u.values, u.domain, v.domain, **kwargs),
            domain=v.domain,
        )
    elif u.domain.ndim == 4:
        return Field(
            values=_domain_transformation_4D(u.values, u.domain, v.domain, **kwargs),
            domain=v.domain,
        )
    else:
        raise NotImplementedError(f"Only implemented for ndim 1, 2, 3, 4")


def _domain_transformation_1D(
    values: Array, domain: Domain, new_domain: Domain, **kwargs
) -> Array:
    # check domains
    assert len(domain.xmin) == len(new_domain.xmin) == 1

    # pad array
    values = jnp.pad(values, pad_width=((1, 1)), mode="reflect")

    # get offsets
    offset = [
        d2_xmin - d1_xmin for d1_xmin, d2_xmin in zip(domain.xmin, new_domain.xmin)
    ]

    # create meshgrid
    meshgrid = jnp.meshgrid(*[jnp.arange(size) for size in values.shape], indexing="ij")
    # create indices
    indices = jnp.concatenate([jnp.expand_dims(x, axis=-1) for x in meshgrid], axis=-1)
    # rearrange indices
    indices = einops.rearrange(indices, "Nx c -> c (Nx)", c=1)

    # add offsets
    indices += jnp.asarray(offset)[..., jnp.newaxis]

    # get indices for new domain
    values_new = jsp.ndimage.map_coordinates(
        values,
        indices,
        order=kwargs.pop("order", 1),
        mode=kwargs.pop("mode", "reflect"),
        **kwargs,
    )

    # rearrange to fit domain
    values_new = einops.rearrange(values_new, "(Nx) -> Nx", Nx=domain.size[0] + 2)
    return values_new[1:-1]


def _domain_transformation_2D(
    values: Array, domain: Domain, new_domain: Domain, **kwargs
) -> Array:
    # check domains
    assert len(domain.xmin) == len(new_domain.xmin) == 2

    # pad array
    values = jnp.pad(values, pad_width=((1, 1), (1, 1)), mode="reflect")

    # get offsets
    offset = [
        d2_xmin - d1_xmin for d1_xmin, d2_xmin in zip(domain.xmin, new_domain.xmin)
    ]

    # create meshgrid
    meshgrid = jnp.meshgrid(*[jnp.arange(size) for size in values.shape], indexing="ij")
    # create indices
    indices = jnp.concatenate([jnp.expand_dims(x, axis=-1) for x in meshgrid], axis=-1)
    # rearrange indices
    indices = einops.rearrange(indices, "Nx Ny c -> c (Nx Ny)", c=2)

    # add offsets
    indices += jnp.asarray(offset)[..., jnp.newaxis]

    # get indices for new domain
    values_new = jsp.ndimage.map_coordinates(
        values,
        indices,
        order=kwargs.pop("order", 1),
        mode=kwargs.pop("mode", "reflect"),
        **kwargs,
    )

    # rearrange to fit domain
    values_new = einops.rearrange(
        values_new, "(Nx Ny) -> Nx Ny", Nx=domain.size[0] + 2, Ny=domain.size[1] + 2
    )
    return values_new[1:-1, 1:-1]


def _domain_transformation_3D(
    values: Array, domain: Domain, new_domain: Domain, **kwargs
) -> Array:
    # check domains
    assert len(domain.xmin) == len(new_domain.xmin) == 3

    # pad array
    values = jnp.pad(values, pad_width=((1, 1), (1, 1), (1, 1)), mode="reflect")

    # get offsets
    offset = [
        d2_xmin - d1_xmin for d1_xmin, d2_xmin in zip(domain.xmin, new_domain.xmin)
    ]

    # create meshgrid
    meshgrid = jnp.meshgrid(*[jnp.arange(size) for size in values.shape], indexing="ij")
    # create indices
    indices = jnp.concatenate([jnp.expand_dims(x, axis=-1) for x in meshgrid], axis=-1)
    # rearrange indices
    indices = einops.rearrange(indices, "Nx Ny Nz c -> c (Nx Ny Nz)", c=3)

    # add offsets
    indices += jnp.asarray(offset)[..., jnp.newaxis]

    # get indices for new domain
    values_new = jsp.ndimage.map_coordinates(
        values,
        indices,
        order=kwargs.pop("order", 1),
        mode=kwargs.pop("mode", "reflect"),
        **kwargs,
    )

    # rearrange to fit domain
    values_new = einops.rearrange(
        values_new,
        "(Nx Ny Nz) -> Nx Ny Nz",
        Nx=domain.size[0] + 2,
        Ny=domain.size[1] + 2,
        Nz=domain.size[2] + 2,
    )

    return values_new[1:-1, 1:-1, 1:-1]


def _domain_transformation_4D(
    values: Array, domain: Domain, new_domain: Domain, **kwargs
) -> Array:
    # check domains
    assert len(domain.xmin) == len(new_domain.xmin) == 4

    # pad array
    values = jnp.pad(values, pad_width=((1, 1), (1, 1), (1, 1), (1, 1)), mode="reflect")

    # get offsets
    offset = [
        d2_xmin - d1_xmin for d1_xmin, d2_xmin in zip(domain.xmin, new_domain.xmin)
    ]

    # create meshgrid
    meshgrid = jnp.meshgrid(*[jnp.arange(size) for size in values.shape], indexing="ij")
    # create indices
    indices = jnp.concatenate([jnp.expand_dims(x, axis=-1) for x in meshgrid], axis=-1)
    # rearrange indices
    indices = einops.rearrange(indices, "Nx Ny Nz Nt c -> c (Nx Ny Nz Nt)", c=4)

    # add offsets
    indices += jnp.asarray(offset)[..., jnp.newaxis]

    # get indices for new domain
    values_new = jsp.ndimage.map_coordinates(
        values,
        indices,
        order=kwargs.pop("order", 1),
        mode=kwargs.pop("mode", "reflect"),
        **kwargs,
    )

    # rearrange to fit domain
    values_new = einops.rearrange(
        values_new,
        "(Nx Ny Nz Nt) -> Nx Ny Nz Nt",
        Nx=domain.size[0] + 2,
        Ny=domain.size[1] + 2,
        Nz=domain.size[2] + 2,
        Nt=domain.size[3] + 2,
    )

    return values_new[1:-1, 1:-1, 1:-1, 1:-1]
