# Home
> A simple package for doing simple differential ocean model approximations in `JAX`.


---

## Quick Start


### **3 APIs for PDEs (TODO)**

In this 3 part tutorial, we describe three hierarchies for describing PDEs:

* **Low-Level**: Fully explicit where we make everything explicit
* **Mid-Level**: Semi-explicit where we specify some parts but don't specify others.
* **High-Level**: Fully implicit where we hide many of the mechanics underneath syntactic sugar.


### **PDE Elements (TODO)**

This goes through a full example looking at the components of the `jaxsw` framework.
We explicitly describe the components of the PDE which will be important for this package.
This includes:
1. State
2. Params
3. Geometry
4. Spatial Discretization
5. Equation of motion
6. Time Stepping

To accomplish this, I will showcase how we can use many other libraries as the backbone to do many canonical PDEs, e.g. [FiniteDiffX](), , [jaxdf](), and [kernex]() for spatial discretizations and [Diffrax]() for timestepping schemes.
In addition, I will do my best to use some of the best elements of `JAX` to really take advantage of some of the native elements, e.g. `vmap`, `scan`.

### **Spatial Discretization (TODO)**

This tutorial goes through how to do some spatial discretizations using this library.
We will look at how we can define a simple geometry and then choose various ways to do operations such as finite difference, e.g. slicing, convolutions, or spectral methods.
In addition, we will look at how we can do some simple procedures to get staggered grids which can greatly improve the accuracy of methods.


### **TimeSteppers (TODO)**

This tutorial goes through how to do time stepping in with JAX.
I'll show how this can be accomplished from scratch and through the native `JAX` package.
We also look at the `diffrax` which allows us to remove a lot of the complexity.


---
## Tutorials

This are more in-depth tutorials about we can use this package for various canonical PDEs that are necessary for understanding and constructing simple differentiable ocean models.

#### [**Lorenz ODEs**](./lorenz/overview.md)

I look at the canonical Lorenz ODEs.
In particular, we look at the Lorenz-63, the Lorenz-96 and the two level Lorenz-96 ODEs.

---

#### [**12 Steps to Navier-Stokes**](12_steps/overview.md)

In this tutorial, I revisit step-by-step through the original [12 Steps to Navier-Stokes]() that was created by Lorena Barber.
This includes going through elements of typical PDEs such as advection, diffusion and elliptical equations.

---

#### [**Quasi-Geostrophic Equations**](qg/overview.md)

In this tutorial, we look at the quasi-geostrophic (QG) equations and demonstrate how we can use elements of this package

---

#### **Shallow Water Equations (TODO)**

In this tutorial, we look at the shallow water (SW) equations and demonstrate how we can use elements of this package.

---

#### **Learning (TODO)**

In these set of tutorials, we will look at how one can use these differentiable models to do some learning. 
We will look at parameter estimation, state estimation and the joint bi-level optimization scheme.
Some applications will include hybrid models for parameterizations and inverse problems for interpolation schemes.
