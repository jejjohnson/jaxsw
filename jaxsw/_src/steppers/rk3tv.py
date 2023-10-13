


def rk3_tvp(x, dt: float, f) -> x:
    
    
    
    # 1st order time derivative (Euler)
    dx0 = f(x)
    
    # do time step
    x += dt * dx0
    
    # ===============
    # 2nd order?
    # ===============
    dx1 = f(x)
    
    # do time step
    x += (dt/4.0) * (dx1 - 3.0*dx0)
    
    
    # ===============
    # 3rd order?
    # ===============
    dx2 = f(x)
    
    # do time step
    x +=  (dt/12.0) * (8.0 * dx2 - dx1 - dx0)
    
    return x