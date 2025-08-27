<div align="center">
<img width="539" height="250" alt="image" src="https://github.com/user-attachments/assets/24174fef-9a4d-4165-b575-8ed4db9e336a" />

  [Maziar Raissi](https://icqmb.ucr.edu/maziar-raissi) and [George Em Karniadakis](https://engineering.brown.edu/people/george-e-karniadakis)
</div>

---

# Introduction to PINN

**Physics-Informed Neural Networks (PINNs)** are a numerical method that combines **deep learning** with **physical equations**. Its core idea is:

* Use a **neural network** to approximate the solution function \$u(x,t,...)\$ of a PDE/ODE.
* During training, minimize not only the data error but also include the **residuals of the physical equations** (PDE, boundary conditions, initial conditions) as part of the loss function, forcing the neural network to obey physical laws.

In this way, PINNs leverage the advantages of data-driven methods while incorporating physical constraints, enabling more reliable solutions when data are scarce or noisy.

---

## 1. Basic Idea of PINN

Let the PDE be:

![equation](https://latex.codecogs.com/png.latex?%5Cmathcal%7BN%7D%5Bu%5D\(x%2Ct\)%3D0%2C%20%5Cquad%20\(x%2Ct\)%5Cin%20%5COmega)

where \$\mathcal{N}\$ is a differential operator.

* Neural Network: Approximate the solution with \$u\_\theta(x,t)\$ (parameters \$\theta\$).
* Residual definition:

![equation](https://latex.codecogs.com/png.latex?r_%5Ctheta\(x%2Ct\)%20%3D%20%5Cmathcal%7BN%7D%5Bu_%5Ctheta%5D\(x%2Ct\))

* Loss function:

![equation](https://latex.codecogs.com/png.latex?%5Cmathcal%7BL%7D\(%5Ctheta\)%20%3D%20%5Cfrac%7B1%7D%7BN_f%7D%20%5Csum%20%7C%20r_%5Ctheta\(x_f%2Ct_f\)%20%7C%5E2%20%2B%20%5Cfrac%7B1%7D%7BN_b%7D%20%5Csum%20%7C%20u_%5Ctheta\(x_b%2Ct_b\)-g_b%20%7C%5E2%20%2B%20%5Cfrac%7B1%7D%7BN_0%7D%20%5Csum%20%7C%20u_%5Ctheta\(x_0%2C0\)-g_0%20%7C%5E2%20%2B%20%5Cfrac%7B1%7D%7BN_d%7D%20%5Csum%20%7C%20u_%5Ctheta\(x_d%2Ct_d\)-u%5E%7Bobs%7D%20%7C%5E2)

Training is optimizing the neural network parameters \$\theta\$ so that the solution satisfies both physical laws and data constraints as much as possible.

---

## 2. Typical Applications of PINN

1. **Forward Problem**

   * Given PDE and parameters, predict the solution \$u(x,t)\$.
   * Example: Solve the heat conduction equation, Navier–Stokes equation.

2. **Inverse Problem**

   * Given partial observation data, PINN can simultaneously learn the solution function and unknown parameters (or coefficients).
   * Example: Infer thermal conductivity from temperature observations, infer viscosity coefficient from flow field data.

3. **Data Assimilation**

   * With sparse or noisy data, PINN can use PDE physical laws for completion and denoising.

---

## 3. Advantages of PINN

* **Physical consistency**: Ensures the solution satisfies PDEs instead of just fitting data.
* **Low data requirement**: Can be trained with only a small number of data points.
* **High generality**: Applicable to ODEs, PDEs (including high-dimensional and complex PDEs).
* **Naturally handles inverse problems**: No need to modify numerical formats, directly infer via the loss function.

---

## 4. Challenges of PINN

* **Difficult training**: Loss function is non-convex, gradient propagation unstable for high-order PDEs.
* **Slow convergence**: Requires long training time; higher computational cost compared to traditional numerical methods (e.g., FEM/FDM).
* **Multiscale problems are difficult**: Performance unstable in high-frequency solutions or strongly nonlinear PDEs.
* **Parameter sensitivity**: Sampling size, network depth, activation functions all affect results.

---

## 5. Comparison Between PINN and Traditional Numerical Methods

| Feature      | PINN                                                                                      | FEM/FDM                                               |
| ------------ | ----------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| Dimension    | Can handle high-dimensional problems (partially overcoming the “curse of dimensionality”) | Difficult to extend to high dimensions                |
| Data         | Can integrate experimental/observational data                                             | Purely physics-based                                  |
| Efficiency   | Training is costly, inference is fast                                                     | Usually faster and more stable                        |
| Applications | Sparse data, inverse problems, complex geometry                                           | Engineering simulations, industrial-scale PDE solving |

---

## Summary

**PINN is a neural network method that integrates physical laws with deep learning, suitable for both forward and inverse problems of PDEs/ODEs. It is particularly advantageous when data are scarce, but training is difficult, and the method is still in the research and exploration stage.**

---

Links to the Original Papers

\[Part I on arXiv (2017)] – Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations

\[Part II on arXiv (2017)] – Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations

\[Journal Publication (2019)] – Physics-informed neural networks: A deep learning framework… in Journal of Computational Physics

---


