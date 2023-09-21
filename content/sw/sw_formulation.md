---
title: Shallow Water Model Formulation
subject: Differentiable Ocean Models
# subtitle: How can I estimate the state AND the parameters?
short_title: Formulation
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - CNRS
      - MEOM
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: jax, shallow water model, differentiable
abbreviations:
    SW: Shallow Water
    QG: Quasi-Geostrophic
    PDE: Partial Differential Equation
    RHS: Right Hand Side
    PV: Potential Vorticity
    SF: Stream Function
    N-S: Navier-Stokes
---


## Shallow Water


```{figure} https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSjtdkiXdNKeWhSEVqnOImCqztCJKzySLLZNA&usqp=CAU
:name: fig-sw
:alt: Shallow Water Equations PDE.
:align: center

The shallow water equations schematic as per [{cite}`10.1017/9781107588431`].
```




$$
h(x,y,t) = \eta(x,y,t) - \eta_B(x,y,t)
$$ (eq:sw_surface)

where $h$ is the total fluid thickness, $\eta$ is the height of the upper free surface, and $\eta_B$ is the height of the lower surface or the bottom topography.
**Note**: when the lower surface is completely flat, then $\eta_B$ will be equivalent to zero so $h=\eta$ exactly. 
However, when we do have bottom topography or a free form layer beneath, $\eta_2$, we need to include the $\eta_B$


We can also define the layer thickness to be a function of the surface displacement and mean height as

$$
h(x,y,t) = \eta(x,y,t) + H(x,y)
$$ (eq:sw_layer_thickness)

where $H$ is the mean thickness.
Note, all methods are related to one another via this formula

$$
\eta = h + \eta_B = H + \eta_T
$$

where $\eta_T$ is the deviation from the top of the total mean thickness.


---
## NonLinear Form

For the SW equations, we define a PDE that captures spatiotemporal relationships between the total thickness height, $h$, and well as the velocity components $(u,v)$.

$$
\begin{aligned}
\text{Height}: &&
h & =\boldsymbol{h}(\vec{\mathbf{x}},t) && && 
\vec{\mathbf{x}}\in\Omega\sub\mathbb{R}^{D_s} , &&
t\in\mathcal{T}\sub\mathbb{R}^{+} \\
\text{Zonal Velocity}: &&
u & =\boldsymbol{u}(\vec{\mathbf{x}},t) && && 
\vec{\mathbf{x}}\in\Omega\sub\mathbb{R}^{D_s} , &&
t\in\mathcal{T}\sub\mathbb{R}^{+} \\
\text{Meridonal Velocity}: &&
v & =\boldsymbol{v}(\vec{\mathbf{x}},t) && && 
\vec{\mathbf{x}}\in\Omega\sub\mathbb{R}^{D_s},  &&
t\in\mathcal{T}\sub\mathbb{R}^{+} \\
\end{aligned}
$$ (eq:qg_fields)

We can write out the momentum constraints for the velocity vector-field, $\vec{\boldsymbol{u}}$, and the height field, $h$, explicitly. 
For a single-layer fluid, including the Coriolis term, and the inviscid SW equations as

$$
\begin{aligned}
\text{Momentum}: && &&
D_t\vec{\boldsymbol{u}} + f\times \vec{\boldsymbol{u}} &= 
- g \boldsymbol{\nabla}(h + \eta_B) \\
\text{Mass}: && &&
\partial_t h + 
\boldsymbol{\nabla}\cdot(h\vec{\boldsymbol{u}}) &= 
0 
\end{aligned}
$$ (eq:sw_momentum_mass)

where $h\vec{\boldsymbol{u}}$ is the horizontal velocity, $f$ is the planetary vorticty (equation [](#eq:planetary_vorticity)) and $g$ is the gravitational constant.
We can expand these equations to be:

$$
\begin{aligned}
\text{Height}: && &&
\partial_t h + 
\partial_x (hu) + \partial_y (hv) &= 0\\
\text{Zonal Velocity}: && &&
\partial_t u + 
u \partial_x u + v\partial_y u - fv &=
- g \partial_x (h + \eta_B) + F_x + B_x + M_x + \xi_x \\
\text{Meridonal Velocity}: && &&
\partial_t v + 
u \partial_x v + v\partial_y v + fu &=
- g \partial_y (h + \eta_B) + F_y + B_y + M_y + \xi_y
\end{aligned}
$$  (eq:sw_nonlinear)



We have a lot of extra terms in the above equation as was proposed in [{cite}`10.1016/j.ocemod.2018.09.006`]. 
These take into account all of the external and/or internal forces and dissipations.
We outline each of them explicitly below.


### Forcing Terms


There is a forcing vector which operates on the $(x,y)$ plane and changes wrt time, i.e. $\vec{\boldsymbol{f}}:\Omega\times\mathcal{T} \rightarrow \mathbb{R}^{2}$.
Each component of the vector operates on an individual velocity component as shown in [](#eq:sw_nonlinear). 
In general, it is defined as

$$
\vec{\boldsymbol{f}} = 
[F_x, F_y]^\top = 
\left[
    \boldsymbol{F}_x(\vec{\boldsymbol{x}},t), 
    \boldsymbol{F}_y(\vec{\boldsymbol{x}},t)
\right]^\top
$$

This forcing operates on the top of the model, for example the wind or atmospheric component.

### Bottom Friction Term

$$
\vec{\boldsymbol{b}} = 
[B_x, B_y]^\top = 
\left[
    \boldsymbol{B}_x(\vec{\boldsymbol{u}},h), 
    \boldsymbol{B}_y(\vec{\boldsymbol{u}},h)
\right]^\top
$$

### Lateral Mixing of Momentum

$$
\vec{\boldsymbol{m}} = 
[M_x, M_y]^\top = 
\left[
    \boldsymbol{M}_x(\vec{\boldsymbol{u}},h), 
    \boldsymbol{M}_y(\vec{\boldsymbol{u}},h)
\right]^\top
$$

### Negative Viscosity Backscatter

$$
\vec{\boldsymbol{\Xi}} = [\xi_x, \xi_y]^\top
$$

---

## Vector Invariant Formulation

We can rewrite the formulation for the momentum parts, $(u,v)$, in equation [](#eq:sw_nonlinear) in the vectorized notation

$$
\partial_t  \vec{\boldsymbol{u}} +
\vec{\boldsymbol{u}} \cdot \boldsymbol{\nabla} \vec{\boldsymbol{u}}
+ f\times \vec{\boldsymbol{u}} =
- g \boldsymbol{\nabla}h
$$ (eq:sw_nonlinear_vector)

We will use the vector identity of equation [](#eq:vector_identity) and plug this into equation [](#eq:sw_nonlinear_vector) to arrive at the vector form.

$$
\begin{aligned}
\text{Height}: && &&
\partial_t h &=  
-\partial_x (hu) - \partial_y (hv) = 0\\
\text{Zonal Velocity}: && &&
\partial_t u &= 
qhv - \partial_x p + F_x + M_x + B_x + \xi_x \\
\text{Meridonal Velocity}: && &&
\partial_t v
&= - qhu - \partial_y p + F_y + M_y + B_y + \xi_y
\end{aligned}
$$  (eq:sw_nonlinear_vorticity)

where $q$ is the potential vorticity (equation [](#eq:potential_vorticity)) and $p$ is the work given by the Bernoulli potential (equation [](#eq:bernoulli_potential)).

:::{note} Proof
<!-- :class: dropdown -->

We will walk through the entire derivation.
Let's rewrite all of the terms

$$
\partial_t  \vec{\boldsymbol{u}} +
\vec{\boldsymbol{u}} \cdot \boldsymbol{\nabla} \vec{\boldsymbol{u}}
+ f\times \vec{\boldsymbol{u}} =
- g \boldsymbol{\nabla}h
$$ (eq:sw_nonlinear_vector)

We will use the vector identity in equation [](#eq:vector_identity) and plug this into equation [](#eq:sw_nonlinear_vector)

$$
\partial_t  \vec{\boldsymbol{u}} +
(\boldsymbol{\nabla}\times \vec{\boldsymbol{u}})\times
\vec{\boldsymbol{u}} +
\frac{1}{2}\boldsymbol{\nabla}(\vec{\boldsymbol{u}} \cdot \vec{\boldsymbol{u}})
+ f\times \vec{\boldsymbol{u}} =
- g \boldsymbol{\nabla}h
$$


We will rearrange the equation to isolate the curl portions

$$
\partial_t  \vec{\boldsymbol{u}} +
(\boldsymbol{\nabla}\times \vec{\boldsymbol{u}})\times
\vec{\boldsymbol{u}} +
f\times \vec{\boldsymbol{u}} =
- g \boldsymbol{\nabla}h -
\frac{1}{2}\boldsymbol{\nabla}(\vec{\boldsymbol{u}} \cdot \vec{\boldsymbol{u}})
$$

Now, we will collapse all like terms

$$
\partial_t  \vec{\boldsymbol{u}} +
(\boldsymbol{\nabla}\times \vec{\boldsymbol{u}} + f)\times \vec{\boldsymbol{u}} =
- g \boldsymbol{\nabla} \left(h +
\frac{1}{2}\vec{\boldsymbol{u}} \cdot \vec{\boldsymbol{u}}\right)
$$

We will use the definition of relative vorticity, $\zeta$ (equation [](eq:relative_vorticity)) and plug this into our equation

$$
\partial_t  \vec{\boldsymbol{u}} +
(\zeta + f)\times \vec{\boldsymbol{u}} =
- g \boldsymbol{\nabla} \left(h +
\frac{1}{2}\vec{\boldsymbol{u}} \cdot \vec{\boldsymbol{u}}\right)
$$

Now, we will use the definition of potential vorticity in the context of the height (equation [](#eq:potential_vorticity)) and plug this into our function.
With a sleight of hand, we will introduce a constant $\frac{h}{h}$ to make sure it works

$$
\partial_t  \vec{\boldsymbol{u}} +
(qh)\times \vec{\boldsymbol{u}} =
- g \boldsymbol{\nabla} \left(h +
\frac{1}{2}\vec{\boldsymbol{u}} \cdot \vec{\boldsymbol{u}}\right)
$$

Now, we can expand the curl term to be

$$
\partial_t  \vec{\boldsymbol{u}} +
qh\vec{\boldsymbol{u}} =
- g \boldsymbol{\nabla} \left(h +
\frac{1}{2}\vec{\boldsymbol{u}} \cdot \vec{\boldsymbol{u}}\right)
$$

So we can rewrite the full formulas as

$$
\begin{aligned}
\text{Height}: && &&
\partial_t h &=  
-\partial_x (hu) - \partial_y (hv) = 0\\
\text{Zonal Velocity}: && &&
\partial_t u &= 
qhv
- \partial_x \left(g(h + \eta_B) + \frac{1}{2}(u^2 + v^2) \right) \\
\text{Meridonal Velocity}: && &&
\partial_t v
&= - qhu
- \partial_y\left(g(h + \eta_B) + \frac{1}{2}(u^2 + v^2)\right)
\end{aligned}
$$  (eq:sw_nonlinear_vorticity)

where $q$ is the potential vorticity and $g(h + \eta_B) + \frac{1}{2}(u^2 + v^2)$ is the Bernoulli potential (equation [](#eq:bernoulli_potential)).

:::


---

## Linearized Shallow Water Equations

We can remove the advection terms from equation [](#eq:sw_nonlinear) by *linearizing* the PDE.

$$
\begin{aligned}
\partial_t h &+ H
\left(\partial_x u + \partial_y v \right) = 0 \\
\partial_t u &- fv =
- g \partial_x h
- \kappa u \\
\partial_t v &+ fu =
- g \partial_y h
- \kappa v
\end{aligned}
$$ (eq:sw_linear)

This is appropriate for when the Rossby number is small. 
We also assume that the wave height is much smaller than the actual mean height. 

### Example Applications

#### Data Assimilation

This has been used in data assimilation schemes [{cite}`10.1029/2021ms002613`] to jointly assimilate observations of SSH whereby the a QG model was used for the balanced motions and a linear SW mode was used for the internal tides.

---
## Wind Forcing


We can construct a very typical wind forcing system that is typical within the Northern Hemisphere at mid-latitudes.
It will resemble the tradewinds in the southern part of the domain and the westerlies in the northern part.

We first define a domain with a height, $H$ and a basin length, $(L_x,L_y)$.


$$
\begin{aligned}
F_y &= 0 \\
F_x &= \frac{F_0}{\rho_0 H}
\left[ 
    \cos \left( 2\pi \left(\frac{2}{L_y} - \frac{1}{2} \right)\right) +
    2\sin \left(2\pi \left( \frac{y}{L_y} - \frac{1}{2} \right) \right)
\right]
\end{aligned}
$$ (eq:sw_wind_stress)

where $F_0=0.12$Pa and $\rho_0=1e3$ kgm$^{-3}$.


---
## Boundary Conditions


**Kinematic Boundaries**.
These boundaries prevent and flow from going through the boundaries.
These are typically on the East-West extent as the flow moves from left/right to right/left

$$
\begin{aligned}
u(x=0)=u(x=L_x)&=0 \\
v(x=0)=v(x=L_y)&=0
\end{aligned}
$$

**Gradient Boundaries**.
We can also say that there is no gradient across the boundaries for the height, $h$.
So we can define this as

$$
\begin{aligned}
\partial_x h(x=0)=\partial_x h(x=L_x)&=0 \\
\partial_y h(x=0)=\partial_y h(x=L_y)&=0
\end{aligned}
$$

**No-Slip Boundaries**.
These refer to the idea that the tangential velocity should vanish.
We typically prescribe these for the North-South extent.

$$
\begin{aligned}
u(y=0)=u(y=L_y)&=0 \\
v(y=0)=v(y=L_x)&=0
\end{aligned}
$$

**Free-Slip Boundaries**.
These refer to the idea that the tangential velocity gradients should be zero.
We often refer to these as *friction-less* boundaries.
We also typically prescribe these on the North-South extent.

$$
\begin{aligned}
\partial_y u(y=0)=\partial_y u(y=L_y)&=0 \\
\partial_x v(x=0)=\partial_x v(x=L_x)&=0
\end{aligned}
$$



---
## HelpFul Tools

### Material Derivative

This is an operator that models the movement of a fluid parcel within a Eulerian framework.

$$
D_t = \partial_t + 
\vec{\boldsymbol{u}} \cdot \boldsymbol{\nabla} =
\partial_t + u\partial_x + v\partial_y
$$ (eq:material_derivative)


### Planetary Vorticity

$$
f(y) = 2\Omega\sin(\theta) + \frac{1}{R}2\Omega\cos(\theta) y
$$ (eq:planetary_vorticity)

where $f_0=2\Omega\sin(\theta_0)$ is the Coriolis force and $\beta=\frac{1}{R}2\Omega\cos(\theta_0)$ is the approximate $\beta$-plane forcing.
For more information, see this [wiki](https://en.wikipedia.org/wiki/Beta_plane) article for a better overview or this [video](https://www.youtube.com/watch?app=desktop&v=Ddj1CQdwOHY) for a more in-depth overview.


### Relative Vorticity

$$
\zeta = 
\partial_x v - \partial_y u = 
\boldsymbol{\nabla}\times\vec{\boldsymbol{u}}
$$ (eq:relative_vorticity)


### Potential Vorticity

$$
q = \frac{\zeta + f}{h}
$$ (eq:potential_vorticity)

where $\zeta$ is the relative vorticity given by [](#eq:relative_vorticity), $f$ is the planetary vorticity given by [](#eq:planetary_vorticity).
A good overview between the relationshop between each of the vorticity equations can be found in this [youtube video](https://www.youtube.com/watch?v=6hmJ_3Es8xI).

### Kinetic Energy

$$
\mathcal{KE} = \frac{1}{2}\left( u^2 + v^2\right)
$$ (eq:kinetic_energy)

### Bernoulli Potential

$$
\begin{aligned}
\text{Bernoulli Potential}: && &&
p &= \frac{1}{2}(u^2 + v^2) + g (h + \eta_B) = \mathcal{KE} + g(h + \eta_B)
\end{aligned}
$$ (eq:bernoulli_potential)

### Vector Identity

$$
\vec{\boldsymbol{u}} \cdot \boldsymbol{\nabla}\vec{\boldsymbol{u}} =
(\boldsymbol{\nabla}\times \vec{\boldsymbol{u}})\times
\vec{\boldsymbol{u}} +
\frac{1}{2}\boldsymbol{\nabla}(\vec{\boldsymbol{u}} \cdot \vec{\boldsymbol{u}})
$$ (eq:vector_identity)