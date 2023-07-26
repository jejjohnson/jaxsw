# Summary

In this section, we look at how we can solve the Shallow water equations using elements from this package.


---

## Linear Shallow Water Equations - [Example](./sw_linear)

$$
\begin{aligned}
\frac{\partial h}{\partial t} &+ H
\left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} \right) = 0 \\
\frac{\partial u}{\partial t} &- fv =
- g \frac{\partial h}{\partial x}
- \kappa u \\
\frac{\partial v}{\partial t} &+ fu =
- g \frac{\partial h}{\partial y}
- \kappa v
\end{aligned}
$$ (eq:sw_linear)


---

## Non-Linear Shallow Water Equations - [Example](./sw_nonlinear)



Taking the equation from [wikipedia](https://en.wikipedia.org/wiki/Shallow_water_equations#Non-conservative_form).


$$
\begin{aligned}
\frac{\partial h}{\partial t} &+ 
\frac{\partial}{\partial x}\left((H+h)u\right) +
\frac{\partial}{\partial y}\left((H+h)v\right)= 0 \\
\frac{\partial u}{\partial t} &+ u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} - fv =
-g\frac{\partial h}{\partial x} -ku + \nu \left( \frac{\partial^2 u}{\partial x^2} + 
\frac{\partial^2 u}{\partial y^2} \right)\\
\frac{\partial v}{\partial t} &+ u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} + fu =
-g\frac{\partial h}{\partial y} -kv + 
\nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)\\
\end{aligned}
$$ (eq:sw)


| Symbol | Variable | Unit | 
|:---------:|:------|:----:|
| $u$ | Zonal Velocity |  $m/s^2$ |
| $v$ | Meridial Velocity |   $m/s^2$ |
| $H$ |Mean Height |   $m$ |
| $h$ |Height Deviation |$m$ |
|$b$ | Topographical height 


**Velocities**. The $u,v$ represent the zonal and meridional velocities in the x,y directions respectively.

**Heights** ($H,h,b$). 
The $H$ represents the mean hight of the horizontal pressure surface. 
The $h$ represents the height deviation of the horizontal pressure surface from its mean height.
$b$ represents the topographical height from a reference $D$.

$$
\begin{aligned}
\eta(x,y,t) &= H(x,y) + h(x,y,t) \\
H(x,y) &= D + b(x,y)
\end{aligned}
$$

**Constants** ($f,k,\nu$). $g$ is the acceleration due to gravity, $k$ is the viscous drag coefficient, and $\nu$ is the kinematic viscosity.