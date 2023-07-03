# Summary

In these sets of notebooks, we look at the canonical Lorenz systems.
We look at the Lorenz-63, the Lorenz-96 and the two level Lorenz-96 ODEs.


## Lorenz 63




[**Demo**](./demo_lorenz63.ipynb)


$$
\begin{aligned}
\frac{dx}{dt} &= \sigma (y - x) \\
\frac{dy}{dt} &= x (\rho - z) - y \\
\frac{dz}{dt} &= xy - \beta z
\end{aligned}
$$ (eq:lorenz63)


## Lorenz 96

$$
\frac{dx}{dt} = (x_{i+1} - x_{i-2})x_{i-1}-x_i+F
$$ (eq:lorenz96)


[**Demo**](./demo_lorenz96.ipynb)


## Lorenz 96 (2 Level)

$$
\begin{aligned}
\frac{dx}{dt} &= (x_{i+1} - x_{i-2})x_{i-1}-x_i + F - \frac{h c}{b} \sum_{j}y_j \\
\frac{dy}{dt} &= -b c (y_{j+2} - y_{j-1})y_{j+1}- c y_j  - \frac{h c}{b} x_i 
\end{aligned}
$$ (eq:lorenz96t)


[**Demo**](./demo_lorenz96t.ipynb)





## Lorenz 96 - Two Level (**TODO**)