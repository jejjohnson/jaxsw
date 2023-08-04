---
title: Shallow Water Equations
subject: Jax Approximate Ocean Models
# subtitle: How can I estimate the state AND the parameters?
short_title: Overview
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - CNRS
      - MEOM
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: data-assimilation, open-science
abbreviations:
    GP: Gaussian Process
---



---
## Examples

We have a few examples of how one can use the Shallow water equations to generate some simulations under different parameter regimes.


---
### Linear Shallow Water Model

[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](./sw_linear_jet_api1)

In this example, we look at the linearized shallow water model given by:

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

We demonstrate how we can use this for generating a simulation with idealistic Kelvin waves that move around the boundary and an idealistic jet across the East-West extent.


---
### NonLinear Shallow Water Model

[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](./sw_nonlinear_jet_api1)

In this example, we look at the nonlinear shallow water model given by:

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


We demonstrate how we can use this for generating a simulation with an idealistic jet across the East-West extent.


---
#### Vorticity Formulation

[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](./sw_nonlinear_jet_api1)

We can also write the shallow water equation as

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

which is the *vector invariant formulation* that uses vorticity.
We also showcase how we can use this formulation for the same parameter regime but with slightly different end results due to numerical computation differences.