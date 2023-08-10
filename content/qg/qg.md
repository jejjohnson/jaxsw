---
title: Quasi-Geostrophic Equations
subject: QuasiGeostrophic Equations
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

We demonstrate a few case studies of QG models under different parameter regimes that have appeared within the literature.

**Note**:See [this page](./qg_formulation) for more information about the formulations.



---
## Examples

We have various examples which showcase different simulations of various types of the QG model under different parameter configurations.

---
### Idealized QG (**TODO**)

We can define the completely idealized 1 layer QG model given as

$$
\begin{aligned}
\partial_t \omega &+ \det\boldsymbol{J}(\psi,\omega) =
 - \mu\omega +
 \nu\boldsymbol{\nabla}^2\omega - 
\beta\partial_x\psi +
F \\
\omega &= \boldsymbol{\nabla}\psi
\end{aligned}
$$ (eq:qg_idealized)

This is a very idealistic case with a periodic domain.
We showcase how we can have simulations similar to [{cite}`10.48550/arxiv.2204.03911`] using the above formulation.
We demonstrated the different flow regimes which depend upon the parameters, i.e. decay flow, forced flow and $\beta$-plane flow.

---
### *Realistic* Idealized QG (**TODO**)

We can define a simple 1 layer QG model 

$$
\begin{aligned}
\partial_t q &+ \vec{\boldsymbol{u}}\cdot q = 
\frac{1}{\rho H}\boldsymbol{\nabla}_H\times\vec{\boldsymbol{\tau}} -
\kappa\boldsymbol{\nabla}_H\psi +
a_4\boldsymbol{\Delta}_H^2 -
\beta\partial_x\psi \\
q &= \boldsymbol{\nabla}\psi
\end{aligned}
$$ (eq:qg_toy)

which starts to encapsulate real parameter regimes we can see in nature.
Inspired by this [course](https://rhwhite.github.io/numeric_2022/notebooks/lab9/01-lab9.html), we showcase how we can have "real-like" simulations using the above formulation.


---
### SSH Free-Run QG (**TODO**)

[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](./ssh_free_run)


The relationship between SSH and the streamfunction is given by the following relationship.


$$
\psi = \frac{g}{f_0}\eta
$$

So we have a way to approximately model the trajectory of SSH using the QG model.
Inspired by [{cite}`10.1175/jtech-d-20-0104.1`], we demonstrate a simple free-run simulation using the configurations.


---
### Idealized 2-Layer QG (**TODO**)

This is a simpler realization of the stacked QG model. 
In [{cite}`10.1002/qj.3891`], they were exploring the effectiveness of a data assimilation method (4DVar) when applied to observation data. 
They used a simple 2-Layer QG model with the stream function $\psi_k$ and the potential vorticity, $q_k$, as shown in equation [](#eq:qg_stacked). 
However, they have a slight different linking term with no extra forcing or dissipation terms.


$$
\begin{aligned}
q_1 &= \nabla^2\psi_1 - F_1(\psi_1 - \psi_2) + \beta y \\
q_2 &= \nabla^2\psi_2 - F_2(\psi_2 - \psi_1) + \beta y + R_s
\end{aligned}
$$

We showcase how we can recreate a free-run simulation from this configuration.


---
### Idealized Stacked QG (**TODO**)

[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](./multilayer_qg)

The culmination of the above examples is the stacked QG model given by:


$$
\begin{aligned}
\partial_t \vec{\boldsymbol{q}} &+
\vec{\boldsymbol{u}}\cdot\boldsymbol{\nabla}
\vec{\boldsymbol{q}} = 
\mathbf{BF} + \mathbf{D} \\
\vec{\boldsymbol{q}} &=
\left(\boldsymbol{\nabla}^2 - f_0^2\mathbf{M}\right)\psi
+ \beta y
\end{aligned}
$$ (eq:qg_stacked)


This is an example that implements an example for the stacked QG model.
The underlying mathematics is based on the [Q-GCM](http://www.q-gcm.org/) numerical model.
However, the configurations are based upon the papers [{cite}`10.48550/arxiv.2204.13914,10.22541/essoar.167397445.54992823/v1`].