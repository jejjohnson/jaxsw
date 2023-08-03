---
title: Quasi-Geostrophic Equations
subject: QuasiGeostrophic Equations
# subtitle: How can I estimate the state AND the parameters?
short_title: Summary
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - CNRS
      - MEOM
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
  - name: Takaya Uchida
    affiliations:
      - FSU
    orcid: https://orcid.org/0000-0002-8654-6009
    email: tuchida@fsu.edu
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

In this section, we look at how we can solve the QG equations using elements from this package.


---
## Generalized QG Equations

**Note**: This formational is largely based on [{cite}`10.3934/dcdss.2022058`].
See this paper for more information including a derivation about the energetics.

We can define two fields as via the potential vorticity (PV) and stream function (SF).

$$
\begin{aligned}
\text{Stream Function}: &&
\psi & =\boldsymbol{\psi}(\vec{\mathbf{x}},t) && && 
\vec{\mathbf{x}}\in\Omega\sub\mathbb{R}^{D_s} , &&
t\in\mathcal{T}\sub\mathbb{R}^{+} \\
\text{Potential Vorticity}: &&
q& =\boldsymbol{q}(\vec{\mathbf{x}},t) && && 
\vec{\mathbf{x}}\in\Omega\sub\mathbb{R}^{D_s},  &&
t\in\mathcal{T}\sub\mathbb{R}^{+} \\
\end{aligned}
$$ (eq:qg_fields)

Let's assume we have a stack of said fields, i.e. $q_k,\psi_k$.
So we have a state defined as:

$$
\begin{aligned}
\text{Stream Function}: &&
\boldsymbol{\psi} = [\psi_1, \psi_2, \ldots, \psi_K]^\top \\
\text{Potential Vorticity}: &&
\boldsymbol{q} = [q_1, q_2, \ldots, q_K]^\top \\
\end{aligned}
$$ (eq:qg_state)

We can write the QG PDE to describe the spatiotemporal relationship between the PV and the SF which is defined as:

$$
\begin{aligned}
\partial_t q_k + \vec{\boldsymbol{u}}_k\cdot\boldsymbol{\nabla}q_k = F_k + D_k
\end{aligned}
$$ (eq:qg_general)

where at each layer $k$, we have $\vec{\boldsymbol{u}}_k$ is the velocity vector, $q_k$ is the potential vorticity, $F_k$ are the forcing term(s), and $D_k$ are dissipation terms.
The forcing term can be made up of any external forces we deem necessary for the PDE.
For example, we could have wind stress that affects the top layer or bottom friction that affects the bottom layer.
The dissipation term represents all of the diffusion terms.
For example, we could have some lateral friction or hyper-viscosity terms.

The advection term in equation [](#eq:qg_general) includes the velocity vector, $\vec{\boldsymbol{u}}_k$, which is defined in terms of the stream function, $\psi$. 
This is given by:

$$
\vec{\boldsymbol{u}} = [u, v]^\top = \left[ -\partial_y \psi, \partial_x \psi\right]^\top
$$ (eq:qg_general_vel)

The PV is the culmination of the potential energy contained within the dynamical forces, the thermal/stretching forces and the planetary forces.

$$
\text{Potential Vorticity Forces} = \text{Dynamical} + \text{Thermal} + \text{Planetary} + \ldots
$$

Concretely, we can define this the PV and its relationship to the SF and other forces as:

$$
\begin{aligned}
q_k = \boldsymbol{\Delta}\psi_k + (\mathbf{M}\psi)_k + f_k
\end{aligned}
$$ (eq:qg_general_pv)

where at each layer, $k$, we have the dynamical (relative) vorticity, $\boldsymbol{\Delta}\psi_k$, the thermal/stretching vorticity, $(\mathbf{M}\psi)_k$, and the planetary vorticity, $f_k$. 


---
## Idealized QG Model

We can use the above formulation to describe an idealized QG model.
This idealized model will be fairly abstract with parameters that represent real things but do not actually correspond to anything tangible.
However, these types of flows are *very* well-studied and are very useful for having a controlled setting which removes all uncertainties.
We will use the notation used in [{cite}`10.48550/arxiv.2204.03911,10.48550/arxiv.2304.05029 `] with some explanations taken from [{cite}`10.48550/arxiv.2209.15616`].

The domain is simplified to be periodic and defined on a $2\pi$ domain, i.e. $Lx = L_y = 2\pi$.
We will also assume a periodic domain.
This domain is topologically equivalent to a two-dimensional torus.

We will define a final form for the idealized QG model of the vorticity-stream functions on this idealized, periodic domain as

$$
\partial_t \omega + \det\boldsymbol{J}(\psi,\omega) =
 - \mu\omega +
 \nu\boldsymbol{\nabla}^2\omega - 
\beta\partial_x\psi +
F
$$ (eq:qg_idealized)

where $\omega$ is the vorticity and $\psi$ is the stream function.
The $\beta\psi_x =\beta v$-term is geophysical system approximation that captures the effect of the differential rotation.
This force is experienced within the Earth system in a tangent plane approximation, i.e. $\beta = \partial_y f$.
This $\beta$-term is important and it allows the flow to manifest different turbulent regimes. 
For example, Rossby waves are common within the planetary systems and can appear when $\beta=0$. 
The determinant Jacobian term encapsulates the advection term seen in [](#eq:qg_general). 
It is defined as:

$$
\begin{aligned}
\text{Determinant Jacobian}: && &&
\det \boldsymbol{J}(\psi,\omega) &= 
\partial_x\psi\partial_y \omega - 
\partial_y\psi\partial_x \omega
\end{aligned}
$$ (eq:eq_detj)

We see from [](#eq:qg_detj) that it is directly related to the advection term seen in [](#eq:qg_general) but written in a different way.

:::{note} Determinant Jacobian
:class: dropdown

Like most things in physics, there are often many ways to express the same expression.
Ultimately, they are all advection expressions.
See the example on [wikipedia](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant#Example_1) for more details.
The detemrinant Jacobian term [](#eq:qg_detj) can be written in many ways.
Let's rewrite it as:

$$
\text{Advection} = \det \boldsymbol{J}(\psi,\omega)
$$

We can expand the full expression which gives us

$$
\text{Advection}  = 
\partial_x\psi\partial_y \omega - 
\partial_y\psi\partial_x \omega 
$$

We can plug int the velocity components of the stream function definition [](#eq:qg_general_vel) into the above equation

$$
\text{Advection} = v\partial_y \omega - u\partial_x \omega
$$

The partial derivative operator is commutable so we can take out the operator of both terms

$$
\text{Advection} = \partial_y (v\omega) - \partial_x (u\omega)
$$

Alternatively, we can write the velocity and partial derivative operators in the vector format

$$
\text{Advection} = \vec{\boldsymbol{u}} \cdot \boldsymbol{\nabla} \omega
$$

and we see that we arrive at formulation in [](#eq:qg_idealized).
I personally prefer this way of writing it as it is more general. 
Furthermore, it exposes the many ways we can express this term like the determinant Jacobian or simple partial derivatives.

**Note**: It is important to take note of the many ways to express this as it can be useful for numerical schemes. 
For example, an upwind scheme might benefit from advection term where the velocity components are multiplied with the partial derivatives. 
Alternatively the determinant Jacobian on an Arakawa C-grid is a well known formulation for dealing with this.

:::

The forcing term is typically chosen to be a constant wind forcing

$$
\boldsymbol{F}_\omega(\vec{\mathbf{x}}) = k_f
\left[ \cos (k_f x) + \cos (k_f y)\right]
$$ (eq:qg_idealized_wind)

:::{seealso} Relation to Navier-Stokes Equations
:class: dropdown

> This derivation and explanation was largely taken from [{cite}`10.48550/arxiv.2304.05029`] which has one of the best high-level explanation of the derivation without too much mathematical detail.

Taking a velocity field, $\vec{\boldsymbol{u}} = [u,v]^\top$,
we can write the non-dimensional form of the Navier-Stokes (N-S) equations as

$$
\partial_t \vec{\boldsymbol{u}} + 
\vec{\boldsymbol{u}} \cdot \boldsymbol{\nabla}\vec{\boldsymbol{u}} +
f (k \times \vec{\boldsymbol{u}}) =
-\mu \vec{\boldsymbol{u}} +
\frac{1}{Re}\boldsymbol{\nabla}^2\vec{\boldsymbol{u}} +
\boldsymbol{F}_{\vec{\boldsymbol{u}}}(\vec{\mathbf{x}})
$$ (eq:ns_idealized_nondim)

The, $f$, is the Coriolis parameter which is the local rotation rate of the Earth and/or other planetary atmospheric forcings.
The example in []() showcases an example with beta-plane forcing.
$\boldsymbol{k}$ is the unit-vector normal to the $(x,y)$-plane. 
The $\mu$ is the linear drag coefficient which represents the bottom friction.
The $Re$ is the Reynolds number measuring the strength of the non-linear advection term, relative to the viscous term. 
In otherwords, the relationship give by:

$$
\vec{\boldsymbol{u}} \cdot \boldsymbol{\nabla}\vec{\boldsymbol{u}}\propto
\frac{1}{Re}\boldsymbol{\nabla}^2\vec{\boldsymbol{u}}
$$

This Reynolds number is indirectly proportional to the viscosity and proportional to the absolute velocity [{cite}`10.48550/arxiv.2209.15616`]

[{cite}`10.48550/arxiv.2304.05029`] choose a Reynolds number of $Re=2,500$.
The forcing is given by

$$
\begin{aligned}
\text{Forcing}: && 
\boldsymbol{F}_{\vec{\boldsymbol{u}}}(\vec{\mathbf{x}}) &=
[-\sin(k_f y), \sin(k_f x)]^\top
\end{aligned}
$$

which is a sinusoidal time-invariant forcing field that continuously drives the flow. In [{cite}`10.1016/j.physd.2022.133568`], the wavenumber $k_f=4$, was chosen.

The velocity field in equation [](#eq:ns_idealized_nondim) is required to satisfy the mass conservation principal given by the continuity equation

$$
\begin{aligned}
\text{Continuity Equation}: &&
\boldsymbol{\nabla}\cdot\vec{\boldsymbol{u}} &= \partial_yu + \partial_xv = 0
\end{aligned}
$$

One can satisfy this by defining a stream function, $\psi(\vec{\mathbf{x}})$, which is a scalar field that is defined as

$$
\begin{aligned}
u = - \psi_y, && && v =\psi_x
\end{aligned}
$$ (eq:qg_idealized_vel)

The stream function and the continuity equations can be expressed as an evolution equation of a single scalar field, the vorticity.
This scalar field is defined as the two-dimensional curl of the velocity field

$$
\omega = \boldsymbol{\nabla}\times \vec{\boldsymbol{u}}=\partial_x v - \partial_y u = \boldsymbol{\nabla}^2\psi
$$

This equation captures the local rotation of a fluid parcel.
The final result is the incompressible 2D Navier-Stokes equations in the **scalar vorticity stream function form**.
In other words, the Quasi-Geostrophic equations.

$$
\partial_t \omega + \det\boldsymbol{J}(\psi,\omega) - \beta v=
 - \mu\omega +
 \frac{1}{Re}\boldsymbol{\nabla}^2\omega +
F
$$ (eq:qg_ns)

**Note**:
There are some small differences between this equation and .
The first is the coefficient in front of the diffusion term, $\boldsymbol{\nabla}^2\omega$. Here, we have the Reynolds number, $1/Re$ instead of the viscosity term, $\nu$, as shown in [](#eq:qg_idealized).
In addition, we have the $\beta$ term. 
In this formulation, it is $\beta v$ whereas in [](#eq:qg_idealized) it is expressed as $\beta \partial_x$.
However, these are equivalent because the first component of the stream function velocities [](#eq:qg_idealized_vel) is defined as $v=\partial_x\psi$.
So we can plug this into the equation above.

**NOTE**: I am not sure about the sign issue of the $\beta$-term in [](#eq:qg_ns).
I think it is a mistake and that it should be positive which would match the equation in [](#eq:qg_idealized) along with various other formulations [{cite}`10.48550/arxiv.2204.03911`]

:::


### Parameter Configurations


Below are some experimental parameters found in [{cite}`10.48550/arxiv.2204.03911`] which showcase 3 different flow regimes based on the parameter scheme.

```{list-table} Table with idealized configuration
:header-rows: 1
:name: tb:qg_idealized

* - Name
  - Symbol
  - Units
  - Decay Flow
  - Forced Flow
  - $\beta$-Plane Flow
* - Resolution
  - $N_x\times N_y$ 
  - 
  - $2,048\times 2,048$
  - $2,048\times 2,048$
  - $2,048\times 2,048$
* - Domain
  - $L_x\times L_y$
  - km
  - $10e3 \times 10e3$
  - $10e3 \times 10e3$
  - $10e3 \times 10e3$
* - Time Step
  - $\Delta_t$
  - s
  - $120$
  - $120$
  - $120$
* - Linear Drag Coefficient
  - $\mu$
  - m$^{-1}$
  - $0$ 
  - $1.25e-8$
  - $1.25e-8$
* - Viscosity
  - $\nu$
  - m$^2$s$^{-1}$
  - $67.0$ 
  - $22.0$
  - $22.0$
* - Beta-Term
  - $\beta$
  - m$^{-1}$s$^{-1}$
  - $0.0$ 
  - $0.0$
  - $1.14e-11$
* - Reynolds Number
  - $Re$
  - 
  - $32e3$
  - $22e4$
  - $34e4$
```



---
## QG Equations

We are going to be using the formulation that is described in the Q-GCM model. The manual can be found [here](). We write the multi-layer QG equations in terms of the vorticity term, $q$, and the stream function term, $\psi$. We consider the stream function and the potential vorticity to be $N_Z$ stacked  isopycnal layers.


$$
\partial_t q_k + (u_kq_k)_x + (v_kq_k)_y = F_k + D_k
$$ (eq:qg_form_adv)

where the $F_k$ and $D_k$ are forcing terms for each layer, $k$.  The vorticity term is defined as

$$
q = 
\frac{1}{f_0} \boldsymbol{\nabla}_H^2\psi -
f_0\mathbf{A}\psi + \beta(y-y_0)+ \tilde{\mathbf{D}}
$$ (eq:qg_vorticity)

where $\tilde{D}$ is the dynamic topography and $\beta$ is the $\beta$-plane approximation. The term that links each of the layers together, $\mathbf{A}$, is a tri-diagonal matrix that can be written as

$$
\mathbf{A} =
\begin{bmatrix}
\frac{1}{H_1 g_1'} & \frac{-1}{H_1 g_2'} & \ldots & \ldots & \ldots  \\
\frac{-1}{H_2 g_1'} & \frac{1}{H_1}\left(\frac{1}{g_1'} + \frac{1}{g_2'} \right) & \frac{-1}{H_2 g_2'} & \ldots & \ldots  \\
\ldots & \ldots & \ldots & \ldots & \ldots \\
\ldots & \ldots & \frac{-1}{H_{n-1} g_{n-2}'} & \frac{1}{H_{n-1}}\left(\frac{1}{g_{n-2}'} + \frac{1}{g_{n-1}'} \right) & \frac{-1}{H_{n-1} g_{n-2}'}  \\
\ldots & \ldots& \ldots & \frac{-1}{H_n g_{n-1}'} & \frac{1}{H_n g_{n-1}'}   \\
\end{bmatrix}
$$ (eq:qg_A)


---
### Jacobian Form

We can also write this using the deterimant Jacobian formulation

$$
\partial_t q + \det\boldsymbol{J}\left(q, \psi \right) = 
BE + \frac{A_2}{f_0}\boldsymbol{\nabla}_H^4\psi -
\frac{A_4}{f_0}\boldsymbol{\nabla}_H^6\psi
$$ (eq:qg_form_detjac)

where the determinant Jacobian is defined as:

$$
\det\boldsymbol{J}\left( q, \psi \right) = 
\partial_x q \partial_y\psi - 
\partial_y q \partial_x\psi
$$ (eq:qg_detjac)

---
### Example Forcing + Diffusion Terms

The formulation used [in]() included the following terms

$$
\begin{aligned}
\text{Forcing}: && 
\boldsymbol{F} &= 
BE\\
\text{Diffusion}: && 
\boldsymbol{D} &= 
\frac{A_2}{f_0}\boldsymbol{\nabla}_H^4\psi -
\frac{A_4}{f_0}\boldsymbol{\nabla}_H^6\psi
\end{aligned}
$$


where B is a matrix containing the coefficients of the forcing terms and $e$ is the entrainment vector.

In the paper [[Thiry et al., 2023](https://doi.org/10.22541/essoar.167397445.54992823/v1)], they use the following method

$$
\begin{aligned}
\text{Hyperviscosity}: && 
\boldsymbol{D_1} &= 
-a_4\boldsymbol{\nabla}_H^6\psi\\
\text{Wind Forcing}: && 
\boldsymbol{F} &= 
\frac{\tau_0}{\rho_0H_1}\left[\partial_x\tau_y - \partial_y\tau_x, 0\cdots,0\right]\\
\text{Bottom Drag}: && 
\boldsymbol{D_2} &= 
\frac{\delta_{ek}}{2H_{N_Z}}
\left[0,\cdots,0,\Delta\psi_N\right]
\end{aligned}
$$


---
### Sea Surface Height

My of my applications are related to Sea Surface Height (SSH) and the QG model is a decent estimation. For example, the formulation that was used here was taken from the Q-GCM which is a fully-fledged ocean model which works for small-medium scale problems over specific regions.

SSH is linked to the QG equations via the stream function which we can write this as:

$$
\psi = \frac{g}{f_0}\eta
$$ (eq:qg_ssh_streamfunction)

This adds some additional interpretation how the vorticity term can be interpreted when dealing with the SSH over the globe.

$$
q_l = 
\underbrace{\boldsymbol{\nabla}_H \psi_l}_{\text{Dynamical}} +
\underbrace{(\mathbf{A}\psi)_k}_{\text{Thermal}} +
\underbrace{f_k}_{\text{Planetary}}
$$ (eq:qg_ssh_vorticity)

We also have . See {cite}`BFNQG` for more information about this term.


---
## Navier-Stokes Relation

This paragraph was largely borrowed from the following paper {cite}`FNOUNET`

We introduce vorticity, $\omega$, as the curl of the flow vorticity, i.e. $\omega = \nabla \times \boldsymbol{v}$.
This results in the incompressible 2D Navier-Stokes equations in the **scalar vorticity stream function form**.

$$
\begin{aligned}
\frac{\partial \omega}{\partial t} + 
\frac{\partial \psi}{\partial y}\frac{\partial \omega}{\partial x} +
\frac{\partial \psi}{\partial x}\frac{\partial \omega}{\partial y}&=
\frac{1}{\text{Re}}
\left(
    \frac{\partial^2\omega}{\partial\omega^2} + 
    \frac{\partial^2\omega}{\partial\omega^2}
\right), \hspace{10mm}
\left(\frac{\partial^2\psi}{\partial x^2} + \frac{\partial^2\psi}{\partial y^2}\frac{}{}\right) = -\omega
\end{aligned}
$$

where the *streamfunction* is defined as $v_x = \frac{\partial\psi}{\partial y}$ and $u_y = - \frac{\partial\psi}{\partial x}$.



## Sea Surface Height Edition

$$
\begin{aligned}
\partial_t q &= - \det\boldsymbol{J}(\psi,q) - \beta\partial_x\psi \\
\psi &= \frac{g}{f_0}\eta \\
q &= \nabla^2 \psi - \frac{f_0^2}{c_1^2}\psi \\
\psi &= \frac{f_0}{g}\eta \\
u &= -\partial_y\psi \\
v &= \partial_x\psi \\
f &= 2\Omega\sin\theta_0 + \frac{1}{R}2\Omega\cos\theta_0 y \\
f_0 &= \mathcal{E}[f] \\
L_R &= \frac{c_1}{f_0}
\end{aligned}
$$ (eq:qg_parts)


## Examples

### Idealized QG (**TODO**)

We have a jupyter notebook showscasing how we can reconstruct the fields displayed in the above section using the idealized QG equation [](#eq:qg_idealized).
This is a simple 1 Layer QG model on a periodic domain.
We will use the experimental configuration found in [](#tb:qg_idealized) which will showcase the different flow regimes depending upon the parameters, i.e. decay flow, forced flow and $\beta$-plane flow.

### *Realistic* Idealized QG (**TODO**)

We have a jupyter notebook showcasing how we can reconstruct the fields displayed in the above section for the *realistic* idealized QG equation.
This is another simple 1 Layer QG model but with parameters that mean something in real space.


### Free-Run QG (**TODO**)

This is a simple 1.5 layer QG and the mapping problem for SSH. 
This comes from {cite}`10.1175/jtech-d-20-0104.1` paper where they showcase the relationship between SSH and the QG model.

### Stacked QG (**TODO**)

This is an example that implements an example for the stacked QG model.
The underlying mathematics is based on the [Q-GCM](http://www.q-gcm.org/) numerical model.
However, the configurations are based upon the papers [{cite}`10.48550/arxiv.2204.13914,10.22541/essoar.167397445.54992823/v1`].

