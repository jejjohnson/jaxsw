---
title: Package Design
subject: Jax Approximate Ocean Models
# subtitle: How can I estimate the state AND the parameters?
short_title: Design
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



```{figure} https://keras-dev.s3.amazonaws.com/tutorials-img/model-training-spectrum.png
:name: myFigure
:alt: Random image of the beach or ocean!
:align: center

Image from Francois Chollet - [Colab Tutorial](https://colab.research.google.com/github/dlmacedo/starter-academic/blob/master/content/courses/deeplearning/notebooks/tensorflow/TensorFlow_2_0_Keras_Crash_Course.ipynb)
```

## Beginner / Causal

> These users use the prebuilt models and generally do not need to interact with the underlying components underneath the hood.

* Prebuild Dynamical Model
* Initial Condition Functions
* Boundary Condition Functions

Some example applications would include:

* Exploration / Teaching
* Dynamical Model priors for data assimilation applications


---
## Intermediate / Engineer

> These users might want to build a simple model using known physics but don't want to get too deep into
> They can use the *Operator* API.

* Custom Params / Domains
* Custom Fields / State
* Spatial Discretizations
* Spatial Operations
* Grid Operations
* Process Operations
* ODE Solver / Step

Some example use cases might be:

* generating simulations of a specific process
* adding a custom forcing or diffusion term to an existing model
* exploring process parameterizations with neural networks


---
## Expert / Researcher

> These users can use the *Functional* API

* Custom Finite Differences - with FiniteDiffX, Kernex
* Custom Grid Operations - with kernex
* Custom Processes - with equinox
* Custom TimeSteppers - with Diffrax.