---
title: Home Page
subject: Jax Approximate Ocean Models
# subtitle: How can I estimate the state AND the parameters?
short_title: Start Here
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
    SWM: Shallow Water Model
    QG: Quasi-Geostrophic
    PDE: Partial Differential Equation
    RHS: Right Hand Side
---

> A simple package for doing simple differential ocean model approximations in `JAX`.


---
## Motivation



:::{seealso} Other Packages
:class: dropdown

[**PhiFlow**](https://tum-pbs.github.io/PhiFlow/).
This package is an excellent example of how to interface machine learning and. 
In fact, it is very close to what we want to have in the future.
This is a single instantiation of that as their scope is much wider than ours. 
In addition, we do not offer anything with neural networks but we still try to maintain compatibility.

[**pyqg**](https://github.com/pyqg/pyqg) / [**pyqg-jax**](https://github.com/karlotness/pyqg-jax).
This package almost works exclusively with the Quasi-Geostrophic equations.
They also only use the spectral decomposition methodologies.
We were motivated by community behind this package however we wanted something that was more general that what they offered.

[**jaxdf**](https://github.com/ucl-bug/jaxdf).
This package was inspirational for defining spatial discretizations.

[**jax-cfd**](https://github.com/google/jax-cfd). 
This package has been a great inspiration thinking about fluids simulations and machine learning.
However, we felt that it was very complex and did not cater to people being able to tinker with it at different levels of granularity.
The internals were optimized but very difficult to real which is a barrier for people with less coding experience.

**Various QG and SW Model Codes**.
There are many examples of simplified being implemented in Python.
There are some examples for QG models are [1 Layer QG](https://github.com/redouanelg/qgsw-DI/tree/master), [Spectral QG](https://github.com/hrkz/torchqg), and a [Stacked QG](https://github.com/louity/qgm_pytorch) which are all implemented in PyTorch.
However, each of them are blueprints.
Actually, I want even more instances of models being implemented. 
But we wish for some of them to be integrated into the platform so that we can continue to grow as a community and take the latest and the greatest improvements.

:::






---
## Installation

#### pip

We can directly install it via pip from the

```bash
pip install "git+https://github.com/jejjohnson/jaxsw.git"
```

#### Cloning

We can also clone the git repository

```bash
git clone https://github.com/jejjohnson/jaxsw.git
cd jaxsw
```

**poetry**

The easiest way to get started is to simply use the poetry package which installs all necessary dev packages as well

```bash
poetry install
```

**pip**

We can also install via `pip` as well

```bash
pip install .
```

**Conda**

We also have a conda environment with all of the equivalent dependencies.

```bash
conda env create -f environments/jax_linux_cpu.yaml
conda activate jaxsw
```


---

## Tutorials




#### [**Anatomy of a PDE**](./components/anatomy_pde)

This goes through a full example looking at the components of the `jaxsw` framework.
We explicitly describe the components of the PDE which will be important for this package.
This includes:
1. Domain
2. Params
2. Initial Condition Operators
3. Boundary Operators
4. Spatial Operators
5. RHS Operators
6. Time Steppers

To accomplish this, I will showcase how we can use many other libraries as the backbone to do many canonical PDEs, e.g. [FiniteDiffX](), , [jaxdf](), and [kernex]() for spatial discretizations and [Diffrax]() for timestepping schemes.
In addition, I will do my best to use some of the best elements of `JAX` to really take advantage of some of the native elements, e.g. `vmap`, `scan`.

---

#### **3 APIs for PDEs (TODO)**

In this 3 part tutorial, we describe three APIs for implementing PDEs:

* **Low-Level/Researcher**: We showcase the functional API which offers a high level of granularity and control
* **Mid-Level/Engineer**: We showcase the operator API which offers a medium level of granularity and control.
* **High-Level/Casual**: We showcase the prebuilt models where we need minimum intervention to get started with PDEs.

---

#### **Spatial Operators (TODO)**

This tutorial goes through how to do some spatial operations using this library.
We will look at how we can define a simple geometry and then choose various ways to do operations such as finite difference, e.g. slicing, convolutions, or spectral methods.
In addition, we will look at how we can do some simple procedures to get staggered grids which can greatly improve the accuracy of methods.

---

#### [**Grid Operators**](./components/grid_operations.ipynb)

This tutorial goes through how we handle grid operations.
This is very useful for implementing staggered grids for different fields.
We showcase how we can use the grid operators to move between the domains along the staggered grid.

---

#### **Boundary Operators (TODO)**

This tutorial goes through how we handle boundary operations.
These are some of the most important components of PDEs as they essentially govern a huge portion of the dynamics and stability within the system.
We show case how we can make some custom operators using some of the staple methods like periodic, Dirichlet and Neumann.

---

#### **TimeSteppers (TODO)**

This tutorial goes through how to do time stepping in with JAX.
I'll show how this can be accomplished from scratch and through the native `JAX` package.
We also look at the `diffrax` which allows us to remove a lot of the complexity.


---
## Examples

> This are more in-depth tutorials about we can use this package for various canonical PDEs that are necessary for understanding and constructing simple differentiable ocean models.

#### [**Lorenz ODEs**](./lorenz/lorenz)

I look at the canonical Lorenz ODEs.
In particular, we look at the Lorenz-63, the Lorenz-96 and the two level Lorenz-96 ODEs.

---

#### [**12 Steps to Navier-Stokes**](./12_steps/12_steps)

In this tutorial, I revisit step-by-step through the original [12 Steps to Navier-Stokes]() that was created by Lorena Barber.
This includes going through elements of typical PDEs such as advection, diffusion and elliptical equations.

---

#### [**Quasi-Geostrophic Equations**](./qg/qg)

In this tutorial, we look at the quasi-geostrophic (QG) equations and demonstrate how we can use elements of this package

---

#### [**Shallow Water Equations**](sw/sw)

In this tutorial, we look at the shallow water (SW) equations and demonstrate how we can use elements of this package.

---

#### **Learning (TODO)**

In these set of tutorials, we will look at how one can use these differentiable models to do some learning. 
We will look at parameter estimation, state estimation and the joint bi-level optimization scheme.
Some applications will include hybrid models for parameterizations and inverse problems for interpolation schemes.
