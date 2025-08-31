## Deep Ritz Method (DRM)

The Deep Ritz Method (DRM) is a deep learning–based numerical method for solving variational problems, especially those arising from the variational formulation of partial differential equations (PDEs). It was proposed by Weinan E and Bing Yu in 2017, inspired by the classical Ritz method, which approximates PDE solutions by minimizing an energy functional. DRM uses deep neural networks as trial functions, and minimizes the variational energy via Monte Carlo sampling of the integrals, making it a mesh-free solver. Compared with Physics-Informed Neural Networks (PINNs), DRM focuses more on the variational weak form, making it suitable for elliptic PDEs (such as the Poisson equation), high-dimensional problems, and nonlinear PDEs. It has natural adaptivity and nonlinear capacity, but may require additional penalty terms for boundary conditions.

The advantage of DRM lies in handling high-dimensional problems (such as option pricing in finance or quantum mechanics) and it has been extended to fractional PDEs, linear elasticity, and more. Its limitations include reliance on the energy functional (not all PDEs have variational forms) and numerical integration errors during training.

---

### Mathematical Description

Consider a typical elliptic PDE, for example the Poisson equation: in the domain \$\Omega \subset \mathbb{R}^d\$,

$$
-\Delta u(x) = f(x), \quad x \in \Omega,
$$

with Dirichlet boundary condition:

$$
u(x) = g(x), \quad x \in \partial \Omega.
$$

Its variational form is realized by minimizing the energy functional \$I\[u]\$ over the Sobolev space \$H^1\_g(\Omega)\$ (the function space satisfying boundary conditions):

$$
I[u] = \int_\Omega \left( \frac{1}{2} |\nabla u(x)|^2 - f(x)u(x) \right) dx,
$$

where the minimizer \$u\$ is the weak solution of the PDE (by Ritz theorem).

In DRM, a parameterized neural network \$u\_\theta(x)\$ (with \$\theta\$ as network parameters) approximates \$u(x)\$. Since direct computation of the integral is difficult, Monte Carlo sampling is used to approximate the loss function:

<img width="625" height="84" alt="image" src="https://github.com/user-attachments/assets/ab7591a7-798b-43aa-9aff-bb6a6574b2ea" />  

where:

* \$x\_i \sim \mathcal{U}(\Omega)\$ are uniformly sampled points from the interior domain (\$N\_\Omega\$ samples).
* \$y\_j \sim \mathcal{U}(\partial \Omega)\$ are uniformly sampled points from the boundary (\$N\_{\partial \Omega}\$ samples).
* \$\lambda > 0\$ is a penalty weight to softly enforce boundary conditions (hard constraints, such as modifying the network architecture, can also be used).
* The gradient \$\nabla u\_\theta\$ is computed via automatic differentiation.

Training minimizes \$J(\theta)\$ using stochastic gradient descent (SGD). For more general variational problems, the functional may be

$$
I[u] = \int_\Omega F(x, u, \nabla u) dx,
$$

and the loss is constructed similarly. Convergence of DRM has been analyzed under certain conditions, such as for linear PDEs with two-layer networks.

---

### Code Implementation

Below is a simple PyTorch implementation of a 1D DRM example for solving the Poisson equation:

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

The code includes neural network definition, variational loss computation, and training loop.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Neural network approximation of u(x)
class DRMNet(nn.Module):
    def __init__(self, hidden_size=50, num_layers=3):
        super(DRMNet, self).__init__()
        self.fc_in = nn.Linear(1, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.fc_out = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.tanh(self.fc_in(x))
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return self.fc_out(x)

# Variational energy loss: (1/2) ∫ (u')^2 dx - ∫ f u dx
def energy_loss(model, x_int, f_int):
    x_int.requires_grad_(True)
    u = model(x_int)
    u_x = torch.autograd.grad(u, x_int, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    term1 = 0.5 * (u_x ** 2)  # (1/2) |∇u|^2
    term2 = f_int * u  # f u
    return torch.mean(term1 - term2)

# Boundary penalty: u(0)^2 + u(1)^2
def boundary_penalty(model):
    x0 = torch.tensor([[0.0]])
    x1 = torch.tensor([[1.0]])
    return model(x0)**2 + model(x1)**2

# Training
def train(model, optimizer, num_epochs=5000, num_int_samples=100, num_bd_samples=10, lambda_bd=10.0):
    losses = []
    for epoch in range(num_epochs):
        # Interior sampling
        x_int = torch.rand((num_int_samples, 1))
        f_int = (np.pi ** 2) * torch.sin(np.pi * x_int)
        loss_energy = energy_loss(model, x_int, f_int)
        # Boundary sampling (simple 1D boundary)
        loss_bd = boundary_penalty(model)
        loss = loss_energy + lambda_bd * loss_bd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    return losses

# Main program
model = DRMNet()
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



### Code Explanation

1. **DRMNet**: A simple fully connected network with tanh activations, input 1D (\$x\$), output 1D (\$u(x)\$).
2. **energy\_loss**: Computes the Monte Carlo approximation of the variational energy using automatic differentiation for gradients.
3. **boundary\_penalty**: Soft penalty to enforce Dirichlet boundary conditions (for complex boundaries, boundary points can be randomly sampled).
4. **train**: Each epoch randomly samples interior points, computes total loss (energy + boundary penalty), and optimizes the network. After training, the loss decreases and the predicted curve approaches the true solution.

This is the simplest demonstration; in practice, it can be extended to high-dimensional PDEs (with higher-dimensional inputs) or adaptive sampling. PyTorch environment is required to run the code.


