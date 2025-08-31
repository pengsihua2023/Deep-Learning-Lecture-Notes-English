## Deep Backward Stochastic Differential Equation (Deep BSDE) Method

The Deep Backward Stochastic Differential Equation (Deep BSDE) Method is a deep learning–based numerical method for solving high-dimensional (even hundreds of dimensions) nonlinear parabolic partial differential equations (PDEs), particularly in scenarios where traditional grid-based methods (such as finite differences) fail due to the curse of dimensionality. It was proposed by Jiequn Han, Arnulf Jentzen, and Weinan E in 2017. The method transforms the PDE into a backward stochastic differential equation (BSDE) via the Feynman-Kac theorem, then uses neural networks to approximate the solution and gradient of the BSDE, and trains the model by minimizing the terminal condition loss. Deep BSDE is mesh-free, making it suitable for applications in financial pricing (e.g., high-dimensional Black-Scholes), quantum mechanics, and control problems, but it has high computational cost and its convergence depends on time discretization and network architecture.

Compared with the Deep Galerkin Method (DGM) or Deep Ritz Method (DRM), Deep BSDE is more suitable for time-dependent parabolic PDEs and naturally handles randomness, but requires simulating stochastic paths, which may introduce variance.

### Mathematical Description

Consider a general semilinear parabolic PDE:

$$
\partial_t u(t,x) + \mu(t,x) \cdot \nabla_x u(t,x) + \tfrac{1}{2} \text{Tr}\big(\sigma(t,x)\sigma(t,x)^* \text{Hess}_x u(t,x)\big)  +  f(t,x,u(t,x),[\sigma(t,x)]^* \nabla_x u(t,x)) = 0,
$$

for \$t \in \[0,T], ; x \in \mathbb{R}^d\$, with terminal condition \$u(T,x) = g(x)\$.
Here, \$\mu : \[0,T] \times \mathbb{R}^d \to \mathbb{R}^d\$ is the drift term,
\$\sigma : \[0,T] \times \mathbb{R}^d \to \mathbb{R}^{d \times d}\$ is the diffusion matrix,
\$f\$ is the nonlinear term, \$\text{Tr}(\cdot)\$ denotes the trace operator, \$A^\*\$ denotes transpose, and \$\text{Hess}\_x u\$ is the Hessian matrix.

By the Feynman-Kac theorem, this PDE can be represented as a forward–backward stochastic differential equation (FBSDE) system:

* **Forward SDE (path process):**

$$
X_t = x + \int_0^t \mu(s,X_s) ds + \int_0^t \sigma(s,X_s) dW_s,
$$

where \$W\_s\$ is a \$d\$-dimensional Wiener process.

* **Backward SDE (value process):**

$$
Y_t = g(X_T) + \int_t^T f(s,X_s,Y_s,Z_s) ds - \int_t^T Z_s^* dW_s,
$$

where \$Y\_t = u(t,X\_t), \quad Z\_t = \[\sigma(t,X\_t)]^\* \nabla\_x u(t,X\_t)\$ (gradient process).

---

Deep BSDE approximates the BSDE via time discretization (Euler scheme):
The time interval $\[0,T]\$ is divided into \$N\$ steps with step size \$\Delta t = T/N\$,
Brownian increments are simulated as \$\Delta W\_n \sim \mathcal{N}(0,\Delta t I\_d)\$.
Then, neural networks are used to parameterize the initial estimate \$Y\_0^\theta\$ (a scalar) and the gradient process \$Z\_n^\theta(t\_n,X\_n)\$ (a neural network at each time step).

The problem becomes minimizing the empirical loss:

$$
J(\theta) = \mathbb{E}\Big[ \big| Y_T^\theta - g(X_T) \big|^2 \Big],
$$

where \$Y\_T^\theta\$ is computed via backward iteration:

$$
Y_{n+1}^\theta = Y_n^\theta - f(t_n, X_n, Y_n^\theta, Z_n^\theta)\Delta t + (Z_n^\theta)^* \Delta W_n,
$$

starting from \$Y\_0^\theta\$.
The expectation is approximated by Monte Carlo sampling (batch simulation of paths), and \$\theta\$ is trained using optimizers such as Adam. This is equivalent to solving a stochastic control problem, where \$Z\$ is the control variable.

---

### Code Implementation

Below is a simple 1D Deep BSDE example implemented in PyTorch, used to solve a nonlinear PDE:

$$
\partial_t u + \tfrac{1}{2} \partial_{xx} u + u^3 = 0,
$$

for \$t \in \[0,1], ; x \in \mathbb{R}\$, with terminal condition

$$
u(1,x) = \cos\left(\tfrac{\pi x}{2}\right).
$$

This is a simplified version of the Allen-Cahn equation variant, with the true solution approximating a smooth function. The code includes neural network definition, path simulation, and training loop.



