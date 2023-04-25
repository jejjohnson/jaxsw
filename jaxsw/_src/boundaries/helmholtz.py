from jaxtyping import Array


def enforce_boundaries_helmholtz(u: Array, u_bc: Array, beta: Array) -> Array:
    u = u.at[0, :].set(-beta * u_bc[0, :])
    u = u.at[-1, :].set(-beta * u_bc[-1, :])
    u = u.at[:, 0].set(-beta * u_bc[:, 0])
    u = u.at[:, -1].set(-beta * u_bc[:, -1])
    return u
