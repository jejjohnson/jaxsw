
def enforce_boundaries_helmholtz(u, u_bc, beta):
    u = u.at[0,:].set(-beta * u_bc[0,:])
    u = u.at[-1,:].set(-beta * u_bc[-1,:])
    u = u.at[:,0].set(-beta * u_bc[:,0])
    u = u.at[:,-1].set(-beta * u_bc[:,-1])
    return u