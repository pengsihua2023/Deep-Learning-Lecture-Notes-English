## Discontinuous Galerkin (DG) Based Neural Networks

Discontinuous Galerkin (DG) Based Neural Networks is a novel hybrid method that combines the Discontinuous Galerkin (DG) method (a numerical technique for solving partial differential equations (PDEs) that allows solutions to be discontinuous across element boundaries) with neural networks, for efficient PDE solving, especially in high-dimensional, nonlinear, or discontinuous problems. It parameterizes trial functions with neural networks, or enriches DG basis functions, to capture complex dynamics while preserving DG’s locality and flexibility. This method excels at handling complex geometries, perturbations, or steady-state problems, and has been applied to Poisson equations, heat equations, and hyperbolic balance laws.

Typical variants include:

* **DGNet**: Inspired by Interior Penalty DG (IPDG), it uses piecewise neural networks as the trial space and piecewise polynomials as the test space, improving accuracy and training efficiency.
* **Local Randomized Neural Networks with DG (LRNN-DG)**: Combines randomized NNs in subdomains coupled with DG, improving efficiency.
* **DG-PINNs**: Uses Physics-Informed Neural Networks (PINNs) to enrich DG basis functions, achieving approximately well-balanced properties for balance laws.

---

### Mathematical Description

Consider a general PDE, e.g., the Poisson equation:

$$
-\Delta u = f \quad \text{in } \Omega \subset \mathbb{R}^d,
$$

with Dirichlet boundary condition

$$
u = g \quad \text{on } \partial \Omega.
$$

In the classical DG method, the domain \$\Omega\_h\$ is discretized into a mesh \$\Omega\_h = \bigcup\_{i=1}^N E\_i\$ (elements \$E\_i\$). The trial and test spaces are piecewise polynomials \$V\_h^k = { v: v|\_{E\_i} \in \mathcal{P}\_k(E\_i) }\$, allowing discontinuities at boundaries. The weak formulation is: find \$u\_h \in V\_h^k\$, such that \$\forall v\_h \in V\_h^k\$:

<img width="724" height="70" alt="image" src="https://github.com/user-attachments/assets/634cae4e-29b9-48a8-9b6c-bb9d8c1f6441" />  

where \${ \cdot }\$ denotes average and jump operators, and \$\alpha\$ is the penalty parameter (IPDG).

In DG Based Neural Networks (e.g., DGNet), the trial space is replaced with a piecewise neural network space

$$
\mathcal{N}_{\Omega_h} = \{ u_\theta: u_\theta|_{E_i} \in \mathcal{N}_{l,\text{nn}}(\theta_i) \},
$$

where \$\mathcal{N}\_{l,\text{nn}}(\theta\_i)\$ is a shallow NN (layers \$L \leq 2\$, hidden units \$r\$) on element \$E\_i\$, with independent parameters \$\theta\_i\$:

$$
u_\theta(x) = \sum_{i=1}^N u^i_{\text{NN}}(x; \theta_i) \chi_{E_i}(x),
$$

where \$u^i\_{\text{NN}}\$ is the NN, and \$\chi\_{E\_i}\$ is the indicator function.

The test space remains as piecewise polynomials \$V\_h^k\$. The weak formulation is similar, but the solution is trained by minimizing a residual loss:

<img width="754" height="74" alt="image" src="https://github.com/user-attachments/assets/96c73b51-a207-4110-abfe-3296c8ff0526" />  

where integrals are approximated by Monte Carlo sampling or quadrature points, and gradients are computed via automatic differentiation. Training minimizes \$J(\theta)\$ to optimize \$\theta\$.

For time-dependent PDEs (e.g., parabolic equations \$u\_t - \Delta u = f\$), this can be extended to space–time DG, or solved with time-stepping methods.

In DG-PINNs, DG basis functions are enriched with PINN-based corrections: standard DG basis \$\phi\_j\$ plus NN-approximated stabilization terms \$\psi\_\theta\$, to achieve well-balanced properties (exact capture of steady states).

---

### Code Implementation

Below is a simple PyTorch implementation of a 1D DGNet example for solving the Poisson equation:

$$
-u''(x) = \pi^2 \sin(\pi x), \quad x \in [0,1],
$$

with boundary conditions

$$
u(0) = u(1) = 0.
$$

The true solution is

$$
u(x) = \sin(\pi x).
$$

We divide the domain into 4 elements, each with an independent shallow NN (single hidden layer), using the IPDG weak formulation as the loss. Test functions are piecewise linear polynomials (\$k=1\$), and integration is approximated via Monte Carlo sampling.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Local NN: one shallow NN per element
class LocalNN(nn.Module):
    def __init__(self, hidden_size=20):
        super(LocalNN, self).__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))

# DGNet: piecewise NN
class DGNet(nn.Module):
    def __init__(self, num_elements=4, hidden_size=20):
        super(DGNet, self).__init__()
        self.num_elements = num_elements
        self.element_size = 1.0 / num_elements
        self.locals = nn.ModuleList([LocalNN(hidden_size) for _ in range(num_elements)])
    
    def forward(self, x):
        u = torch.zeros_like(x)
        for i in range(self.num_elements):
            mask = (x >= i * self.element_size) & (x < (i+1) * self.element_size)
            local_x = x[mask] - i * self.element_size  # map to [0, h]
            u[mask] = self.locals[i](local_x)
        return u

# Compute first derivative
def grad_u(model, x):
    x.requires_grad_(True)
    u = model(x)
    du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return du

# DG loss: IPDG weak form approximation (Monte Carlo sampling)
def dg_loss(model, num_int_samples=50, num_bd_samples=10, alpha=10.0, lambda_bd=10.0):
    # Interior sampling
    x_int = torch.rand((num_int_samples, 1))
    f = (np.pi ** 2) * torch.sin(np.pi * x_int)
    
    # Simplified test function (random residual sampling)
    du = grad_u(model, x_int)
    loss_int = torch.mean(du ** 2 / 2 - f * model(x_int))  # simplified variational form (symmetric IPDG)
    
    # Boundary penalty (element jumps + domain boundaries)
    loss_jump = 0.0
    for i in range(1, model.num_elements):
        x_e = torch.tensor([[i * model.element_size]])
        u_left = model(x_e - 1e-5)
        u_right = model(x_e + 1e-5)
        du_left = grad_u(model, x_e - 1e-5)
        du_right = grad_u(model, x_e + 1e-5)
        jump_u = u_left - u_right
        avg_du = (du_left + du_right) / 2
        loss_jump += alpha * jump_u ** 2 - avg_du * jump_u  # IPDG penalty
    
    # Domain boundary conditions
    x0 = torch.tensor([[0.0]])
    x1 = torch.tensor([[1.0]])
    loss_bd = model(x0)**2 + model(x1)**2
    
    return loss_int + loss_jump / model.num_elements + lambda_bd * loss_bd

# Training
model = DGNet()
optimizer = optim.Adam(model.parameters(), lr=0.005)
losses = []
for epoch in range(5000):
    loss = dg_loss(model)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Test
x_test = torch.linspace(0, 1, 100).unsqueeze(1)
u_pred = model(x_test).detach().numpy()
u_true = np.sin(np.pi * x_test.numpy())
plt.plot(x_test.numpy(), u_pred, label='Prediction')
plt.plot(x_test.numpy(), u_true, label='True')
plt.legend()
plt.show()
```


### Code Explanation

1. **LocalNN**: Independent shallow NN (tanh activation) per element.
2. **DGNet**: Combines local NNs and evaluates piecewise.
3. **dg\_loss**: Approximates IPDG weak form loss, including interior residuals, boundary jump penalties ($\[u]\$ and \${\nabla u}\$), and boundary conditions. Derivatives are computed via automatic differentiation; integrals via Monte Carlo sampling.
4. **train**: Minimizes the loss to optimize all local NN parameters. After training, predictions approximate the true solution.

This is a simplified demo (1D, sampling-based test space); in practice (e.g., DGNet), polynomial test functions with quadrature rules can be used, or extensions to higher dimensions/time-dependent PDEs. For a full implementation, see DG-PINNs GitHub notebooks (training PINN priors and enriching DG bases with PyTorch).



