# Deep Galerkin Method (DGM)
## 📖 Introduction
The Deep Galerkin Method (DGM) is a deep learning algorithm for solving partial differential equations (PDEs), especially in high dimensions. It was proposed by Justin Sirignano and Konstantinos Spiliopoulos in 2017, inspired by the classical Galerkin method, but replaces traditional finite-dimensional basis functions with deep neural networks to approximate PDE solutions. DGM trains the network by minimizing the integral form of the PDE residual, avoiding mesh discretization. It is mesh-free and suitable for nonlinear, high-dimensional, and complex-domain PDEs, such as the Black-Scholes equation in finance or fluid dynamics problems.

Similar to Physics-Informed Neural Networks (PINNs), DGM also uses neural networks to represent solutions, but DGM emphasizes Galerkin orthogonality conditions, approximating the loss function through random sampling of the integration domain, which enables efficient handling of high-dimensional problems. It has been extended to various PDEs, such as the Fokker–Planck equation, the Stokes equation, and mean-field games.


## 📖 Mathematical Description

Consider a general PDE problem: in the domain \$\Omega \subset \mathbb{R}^d\$, satisfy:

$$
\mathcal{L}u(x) = f(x), \quad x \in \Omega,
$$

with boundary condition:

$$
\mathcal{B}u(x) = g(x), \quad x \in \partial \Omega,
$$

where \$\mathcal{L}\$ is a differential operator (possibly nonlinear), \$\mathcal{B}\$ is a boundary operator, and \$u(x)\$ is the unknown function to solve.

DGM uses a parameterized neural network \$u\_\theta(x)\$ (with \$\theta\$ as network parameters) to approximate \$u(x)\$. Similar to the Galerkin method, it solves by minimizing the integral form of the residual, but the integral is approximated by Monte Carlo sampling. Specifically, the loss function is defined as:

<img width="505" height="76" alt="image" src="https://github.com/user-attachments/assets/a531160e-2861-44cc-b17b-2178e7238781" />  

where:

* \$x\_i \sim \mathcal{U}(\Omega)\$ are uniformly sampled points from the interior domain \$\Omega\$ (\$N\_\Omega\$ samples).
* \$y\_j \sim \mathcal{U}(\partial \Omega)\$ are uniformly sampled points from the boundary \$\partial \Omega\$ (\$N\_{\partial \Omega}\$ samples).
* \$\lambda > 0\$ is a weighting hyperparameter balancing interior residual and boundary conditions.
* The loss uses the \$L^2\$ norm (or other norms), with \$\mathcal{L}u\_\theta\$ and \$\mathcal{B}u\_\theta\$ computed via automatic differentiation.

The training process uses stochastic gradient descent (SGD) or Adam optimizer to minimize \$J(\theta)\$. Since sampling is random, each batch of data is different, helping to avoid local minima. Convergence of DGM has been proven under certain conditions, such as for linear elliptic PDEs. For more complex PDEs (e.g., physical problems), time-stepping extensions (DGMT) can be introduced.

---

## 📖 Code Implementation

Below is a simple PyTorch example of DGM, used to solve the Poisson equation:

$$
-u''(x) = \pi^2 \sin(\pi x), \quad x \in [0,1],
$$

with boundary conditions:

$$
u(0) = u(1) = 0.
$$

The true solution is:

$$
u(x) = \sin(\pi x).
$$

The code includes neural network definition, loss computation, and training loop.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Neural network to approximate u(x)
class DGMNet(nn.Module):
    def __init__(self, hidden_size=50, num_layers=3):
        super(DGMNet, self).__init__()
        self.fc_in = nn.Linear(1, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.fc_out = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.tanh(self.fc_in(x))
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return self.fc_out(x)

# PDE residual: -u''(x) - pi^2 sin(pi x)
def pde_residual(model, x):
    x.requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    f = (np.pi ** 2) * torch.sin(np.pi * x)
    return -u_xx - f  # residual

# Boundary loss: u(0)=0, u(1)=0
def boundary_loss(model):
    x0 = torch.tensor([[0.0]])
    x1 = torch.tensor([[1.0]])
    return model(x0)**2 + model(x1)**2

# Training
def train(model, optimizer, num_epochs=5000, num_samples=100, lambda_bd=1.0):
    losses = []
    for epoch in range(num_epochs):
        # Interior sampling
        x_int = torch.rand((num_samples, 1))
        res = pde_residual(model, x_int)
        loss_int = torch.mean(res ** 2)
        loss_bd = boundary_loss(model)
        loss = loss_int + lambda_bd * loss_bd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    return losses

# Main program
model = DGMNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
losses = train(model)

# Test
x_test = torch.linspace(0, 1, 100).unsqueeze(1)
u_pred = model(x_test).detach().numpy()
u_true = np.sin(np.pi * x_test.numpy())
plt.plot(x_test.numpy(), u_pred, label='Prediction')
plt.plot(x_test.numpy(), u_true, label='True')
plt.legend()
plt.show()
```



## 📖 Code Explanation

1. **DGMNet**: A simple fully connected network with tanh activations, input 1D (\$x\$), output 1D ( \$u(x)\$ ).
2. **pde\_residual**: Uses automatic differentiation to compute the second derivative and calculate the PDE residual.
3. **boundary\_loss**: Directly enforces boundary conditions (hard constraint, could also be soft).
4. **train**: Each epoch randomly samples interior points, computes the loss (interior residual + boundary loss), and optimizes the network. After training, the loss decreases, and the predicted curve approaches the true solution.

This implementation is the simplest demonstration; in practice, it can be extended to high-dimensional PDEs or adaptive sampling.

