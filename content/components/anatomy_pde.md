---
title: Anatomy of Partial Differential Equations
subject: Jax Approximate Ocean Models
# subtitle: How can I estimate the state AND the parameters?
short_title: Anatomy of PDEs
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
    SW: Shallow Water
    QG: Quasi-Geostrophic
    PDE: Partial Differential Equation
    RHS: Right Hand Side
---

## Coordinates and Domain

$$
\begin{aligned}
\vec{\mathbf{x}} \in \Omega \sub \mathbb{R}^{D_s}
\end{aligned}
$$ (eq:coords_spatial)

where $\vec{\mathbf{x}}$ is a vector of coordinate values, i.e. $\vec{\mathbf{x}}=[x_1, x_2, \ldots]^\top$.

$$
\begin{aligned}
t \in \mathcal{T} \sub \mathbb{R}^+
\end{aligned}
$$ (eq:coords_temporal)

where $t$ is a scalar value along the real number line.

```python
domain: Domain = Domain(xmin, xmax, dx)
```

---
## Field

$$
\boldsymbol{u}(\vec{\mathbf{x}}, t): \mathbb{R}^{D_s}\times\mathbb{R}^+ \rightarrow \mathbb{R}
$$ (eq:field)

```python
u: Field = Field(values, domain)
```

---
## State

A state is a collection of different fields, i.e. $\vec{\boldsymbol{u}}=[u_1, u_2, \ldots]^\top$

$$
\vec{\boldsymbol{u}}(\vec{\mathbf{x}}, t): \mathbb{R}^{D_s}\times\mathbb{R}^+ \rightarrow \mathbb{R}^{D_u}
$$ (eq:state)


For example, in the SW we have the height and the velocity vecto,
whereas in the QG equations, we have the potential vorticity and the stream function.


```python
state: State = State(u, v, h)
```

---
## Partial Differential Equations

$$
\begin{aligned}
\partial_t \vec{\boldsymbol{u}}(\vec{\mathbf{x}}, t) &=
\boldsymbol{F}
\left[\vec{\boldsymbol{u}}; \boldsymbol{\theta}\right]
(\vec{\mathbf{x}}, t) && &&
\boldsymbol{F}: \mathbb{R}^{D_u} \times \mathbb{R}^{D_\theta}
\rightarrow \mathbb{R}^{D_u}
\end{aligned}
$$ (eq:pde)

```python
F: Callable = equation_of_motion(t, state, params)
```


---
## Initial Conditions Function


$$
\begin{aligned}
\vec{\boldsymbol{u}}_0 :=
\vec{\boldsymbol{u}}(\vec{\mathbf{x}}, 0) &=
\boldsymbol{F}_{IC}
\left[\vec{\boldsymbol{u}}; \boldsymbol{\theta}\right]
(\vec{\mathbf{x}})
 && &&
\boldsymbol{F}_{IC}: \mathbb{R}^{D_u}
\rightarrow \mathbb{R}^{D_u} && &&
\vec{\mathbf{x}} \in \Omega \sub \mathbb{R}^{D_s}
\end{aligned}
$$ (eq:ics)

```python
F_ic: Callable = FIC(domain, params)
```


---
## Boundary Conditions Function


$$
\begin{aligned}
\vec{\boldsymbol{u}}_b :=
\vec{\boldsymbol{u}}(\vec{\mathbf{x}}, t) &= 
\boldsymbol{F}_{BC}
\left[\vec{\boldsymbol{u}}; \boldsymbol{\theta}\right]
(\vec{\mathbf{x}})  && &&
\boldsymbol{F}_{BC}: \mathbb{R}^{D_u}
\rightarrow \mathbb{R}^{D_u} && &&
\vec{\mathbf{x}} \in \partial\Omega \sub \mathbb{R}^{D_s} && &&
t \in \mathcal{T} \sub \mathbb{R}^+
\end{aligned}
$$ (eq:bcs)

```python
F_bc: Callable = FBC(t, state, params)
```

---
## Time Stepper


$$
\vec{\boldsymbol{u}}(\vec{\mathbf{x}}, t) =
\vec{\boldsymbol{u}}(\vec{\mathbf{x}}, 0) +
\int_0^t
\boldsymbol{F}
\left[\vec{\boldsymbol{u}}; \boldsymbol{\theta}\right]
(\vec{\mathbf{x}}, \tau) d\tau
$$ (eq:theorem_calculus)


```python
state: State = TimeStepper(F, params, state, t0, t1, dt)
```

---
## Summary Table

| Symbol | Name |
|:------:|:----:|
| $\mathbb{R}^{D_s}$ | Spatial Coordinate Space |
| $\mathbb{R}^+$ | Temporal Coordinate Space |
| $\vec{\mathbf{x}}$ | Spatial Coordinates |
| $t$ | Temporal Coordinate |
| $\Omega$ | Spatial Domain |
| $\partial\Omega$ | Boundary of Spatial Domain |
| $\mathcal{T}$ | Temporal Domain |
| $u$ | scalar field |
| $\boldsymbol{u}$ | vector field |
| $\vec{\boldsymbol{u}}$ | state |
| $\boldsymbol{F}$ | equation of motion operator (RHS) |
| $\boldsymbol{F}_{IC}$ | initial condition function |
| $\boldsymbol{F}_{BC}$ | boundary condition functions |
| $\boldsymbol{\theta}$ | parameters (PDE, spatiotemporal discretization) |
| $\int,\text{TimeStepper}$ | Time stepper|