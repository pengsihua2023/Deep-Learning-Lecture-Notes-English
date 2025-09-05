# PINN: Physics-Informed Neural Networks


## 📖 Introduction

Physics-Informed Neural Networks (PINNs) are a neural network framework that combines deep learning with physical laws to solve partial differential equations (PDEs) or simulate physical systems. PINNs embed physical equations (such as governing equations, initial conditions, and boundary conditions) into the loss function of the neural network, and approximate the solution of PDEs by optimizing network parameters. Compared with traditional numerical methods (such as finite difference or finite element), PINNs do not require discretization of the grid and are suitable for high-dimensional or complex geometry problems.

## 📖 Principles

1. **Core Idea**:

* PINNs use a neural network \$u(x,t;\theta)\$ (with parameters \$\theta\$) to approximate the solution \$u(x,t)\$ of a PDE.
* By including PDE residuals, initial conditions, and boundary conditions in the loss function, the network output is constrained to satisfy physical laws.
* The optimization goal is to minimize the loss function so that the network output approaches the true solution.

2. **PDE Form**: Consider a general PDE:

$$
\mathcal{N}[u(x,t)] = f(x,t), \quad (x,t) \in \Omega
$$

with initial and boundary conditions:

$$
u(x,0) = u_{0}(x), \quad u(x,t) \in \partial \Omega = g(x,t)
$$

Where:

* \$\mathcal{N}\$: the PDE operator (e.g. \$\frac{\partial u}{\partial t} + \frac{\partial^2 u}{\partial x^2} = 0\$).
* \$\Omega\$: the domain.
* \$u\_0, g\$: initial and boundary conditions.

3. **Loss Function**:
   The loss function of PINNs consists of three parts:

* **PDE Residual Loss:**

$$
L_{\text{PDE}} = \frac{1}{N_f} \sum_{i=1}^{N_f} \left| \mathcal{N}[u(x_i, t_i; \theta)] - f(x_i, t_i) \right|^2
$$

Calculated at sampled points \${x\_i, t\_i}\$.

* **Initial Condition Loss:**

$$
L_{\text{init}} = \frac{1}{N_i} \sum_{i=1}^{N_i} \left| u(x_i, 0; \theta) - u_0(x_i) \right|^2
$$

* **Boundary Condition Loss:**

$$
L_{\text{bc}} = \frac{1}{N_b} \sum_{i=1}^{N_b} \left| u(x_i, t_i; \theta) - g(x_i, t_i) \right|^2
$$

* **Total Loss:**

$$
L = \lambda_1 L_{\text{PDE}} + \lambda_2 L_{\text{init}} + \lambda_3 L_{\text{bc}}
$$

where \$\lambda\_1, \lambda\_2, \lambda\_3\$ are weights to balance the parts.

4. **Automatic Differentiation**:

* PINNs use neural network automatic differentiation (autograd) to compute derivatives in PDEs (such as \$\frac{\partial u}{\partial t}, \frac{\partial^2 u}{\partial x^2}\$) without explicit discretization.
* The network input is space-time coordinates \$(x,t)\$, and the output is the solution \$u(x,t)\$.

5. **Advantages**:

   * No need for mesh discretization, suitable for high-dimensional or complex geometries.
   * Flexible and easy to incorporate physical constraints.
   * Can combine small amounts of observational data with physical laws.

6. **Disadvantages**:

* Training may be slow, optimization of complex PDEs can be unstable.
* Loss weights \$\lambda\_i\$ require tuning.
* Accuracy for complex nonlinear PDEs may not match traditional numerical methods.

7. **Application Scenarios**:

   * Solving PDEs (e.g. heat conduction, wave equations, fluid dynamics).
   * Data-driven physical modeling (e.g. with experimental data).
   * Inverse problems (e.g. parameter estimation).


## 📖 PyTorch Usage

The following is a minimal PyTorch code example showing how to use PINNs to solve the 1D Burgers equation (a nonlinear PDE), along with explanations.

##### **Problem Description**

**Burgers Equation:**

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}, 
\quad x \in [-1,1], t \in [0,1]
$$

**Initial Condition:**

$$
u(x,0) = -\sin(\pi x)
$$

**Boundary Condition:**

$$
u(-1,t) = u(1,t) = 0
$$

where \$\nu = \frac{0.01}{\pi}\$ is the viscosity coefficient.

---

### 📖 Code Example

```python
import torch
import torch.nn as nn
import numpy as np

# 1. Define neural network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),  # Input (x, t)
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)  # Output u(x, t)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# 2. Define loss function
def compute_loss(model, x_f, t_f, x_i, t_i, u_i, x_b, t_b, u_b, nu=0.01/np.pi):
    x_f, t_f = x_f.requires_grad_(True), t_f.requires_grad_(True)  # derivatives required
    u = model(x_f, t_f)
    
    # PDE residual
    u_t = torch.autograd.grad(u, t_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    pde_residual = u_t + u * u_x - nu * u_xx
    loss_pde = torch.mean(pde_residual ** 2)
    
    # Initial condition
    u_init = model(x_i, t_i)
    loss_init = torch.mean((u_init - u_i) ** 2)
    
    # Boundary condition
    u_bc = model(x_b, t_b)
    loss_bc = torch.mean((u_bc - u_b) ** 2)
    
    # Total loss
    return loss_pde + loss_init + loss_bc

# 3. Prepare data
N_f, N_i, N_b = 2000, 100, 100
x_f = torch.rand(N_f, 1) * 2 - 1  # x in [-1, 1]
t_f = torch.rand(N_f, 1)  # t in [0, 1]
x_i = torch.rand(N_i, 1) * 2 - 1  # initial condition x
t_i = torch.zeros(N_i, 1)  # t = 0
u_i = -torch.sin(np.pi * x_i)  # u(x, 0) = -sin(πx)
x_b = torch.cat([torch.ones(N_b//2, 1), -torch.ones(N_b//2, 1)])  # x = ±1
t_b = torch.rand(N_b, 1)  # t in [0, 1]
u_b = torch.zeros(N_b, 1)  # u(±1, t) = 0

# 4. Train model
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1000):
    optimizer.zero_grad()
    loss = compute_loss(model, x_f, t_f, x_i, t_i, u_i, x_b, t_b, u_b)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 5. Test prediction
x_test = torch.linspace(-1, 1, 100).reshape(-1, 1)
t_test = torch.ones(100, 1) * 0.5  # t = 0.5
u_pred = model(x_test, t_test)
print("Prediction shape:", u_pred.shape)  # Output: torch.Size([100, 1])

```

### 📖 Code Explanation

* **Model**:

  * Define a simple fully connected neural network with input \$(x,t)\$ and output \$u(x,t)\$.
  * Use \$\tanh\$ activation for smooth PDE solutions.

* **Loss Function**:

  * `compute_loss` computes three parts:

    * **PDE residual**: compute \$u\_t, u\_x, u\_{xx}\$ with `torch.autograd.grad` to form Burgers equation residual.
    * **Initial condition**: enforce

$$
u(x,0) \approx -\sin(\pi x).
$$

```
* **Boundary condition**: enforce  
```

$$
u(-1,t) = u(1,t) = 0.
$$

* The total loss is the weighted sum of all parts (weights set to 1 here, can be tuned).

* **Data**:

  * Randomly sample PDE points \$(x\_f,t\_f)\$, initial points \$(x\_i,t\_i)\$, and boundary points \$(x\_b,t\_b)\$.
  * Initial condition \$u\_i = -\sin(\pi x)\$, boundary condition \$u\_b = 0\$.

* **Training**:

  * Use Adam optimizer (well-suited for PINNs, SGD/Adagrad also possible).
  * Train 1000 iterations, print losses.

* **Testing**:

  * Predict the solution at \$t=0.5, x \in \[-1,1]\$, verify output shape.

---

#### **Notes**

1. **Hyperparameters**:

   * Number of sampled points \$(N\_f, N\_i, N\_b)\$: more points improve accuracy but increase computation cost.
   * Network structure: depth/width depends on PDE complexity.
   * Loss weights: may need tuning for complex PDEs.

2. **Automatic Differentiation**:

   * Use `requires_grad_(True)` to compute higher-order derivatives.
   * Ensure correct tensor shapes `[N,1]`.

3. **Optimizer**:

   * Adam is better than SGD/Adagrad for PINNs due to adaptivity.
   * L-BFGS can be tried for faster convergence.

4. **Applications**:

   * Replace Burgers equation with other PDEs (heat, wave).
   * Add observational data for inverse problems.
   * Use GPU for acceleration: `model.cuda(), x_f.cuda(), ...`.

5. **Visualization**:

   * Use matplotlib to plot predicted \$u(x,t)\$.
```python
import matplotlib.pyplot as plt
plt.plot(x_test, u_pred.detach().numpy(), label='Predicted u(x, 0.5)')
plt.xlabel('x'); plt.ylabel('u'); plt.legend(); plt.show()
```

---

## 📖 Summary

PINNs are a powerful method for solving PDEs or modeling physical systems by embedding physical equations into neural network loss functions. PyTorch implementation is straightforward, using autograd for PDE residuals and optimization with initial and boundary conditions. The example demonstrates basic application to Burgers equation.

---

# Multi-Dimensional PDEs, Inverse Problems, and Visualization

## 📖 More Detailed PINN Code Examples

Based on the previous introduction and 1D Burgers equation, here are more detailed examples covering:

* **Multi-dimensional PDEs**: e.g. 2D heat conduction equation (variant of 2D Laplace equation).
* **Inverse problems**: estimating unknown PDE parameters (e.g. diffusion coefficient) with observational data.
* **Visualization**: use Matplotlib for heatmaps and animations.

These examples are based on PyTorch, with full code including data generation, training loop, loss logging, and visualization. Assumes you have basic PyTorch and Matplotlib environment.

### 1. **Multi-Dimensional PDE Example: 2D Heat Conduction**

2D heat conduction equation (simplified Laplace):

$$
\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0, 
\quad (x,y) \in [0,1] \times [0,1]
$$

**Boundary Conditions:**

* \$u(0,y) = 0, \quad u(1,y) = 0\$
* \$u(x,0) = 0, \quad u(x,1) = \sin(\pi x)\$

This is a steady-state problem (no time dimension), solved by minimizing residuals with PINN.

### 📖 Code Example

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Define neural network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),  # Input (x, y)
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)  # Output u(x, y)
        )

    def forward(self, x, y):
        inputs = torch.cat([x, y], dim=1)
        return self.net(inputs)

# 2. Define loss function
def compute_loss(model, x_f, y_f, x_b, y_b, u_b):
    x_f, y_f = x_f.requires_grad_(True), y_f.requires_grad_(True)  # derivatives required
    u = model(x_f, y_f)
    
    # PDE residual: Laplace equation u_xx + u_yy = 0
    u_x = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_f, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    pde_residual = u_xx + u_yy
    loss_pde = torch.mean(pde_residual ** 2)
    
    # Boundary condition
    u_bc = model(x_b, y_b)
    loss_bc = torch.mean((u_bc - u_b) ** 2)
    
    # Total loss
    return loss_pde + 10 * loss_bc  # Increase boundary weight to strengthen constraint

# 3. Prepare data
N_f = 10000  # PDE interior sampling points
N_b = 400    # Boundary sampling points (100 points per side)

# PDE interior points
x_f = torch.rand(N_f, 1)
y_f = torch.rand(N_f, 1)

# Boundary points
x_b_left = torch.zeros(N_b//4, 1)  # x=0
y_b_left = torch.rand(N_b//4, 1)
u_b_left = torch.zeros(N_b//4, 1)  # u=0

x_b_right = torch.ones(N_b//4, 1)  # x=1
y_b_right = torch.rand(N_b//4, 1)
u_b_right = torch.zeros(N_b//4, 1)  # u=0

x_b_bottom = torch.rand(N_b//4, 1)  # y=0
y_b_bottom = torch.zeros(N_b//4, 1)
u_b_bottom = torch.zeros(N_b//4, 1)  # u=0

x_b_top = torch.rand(N_b//4, 1)  # y=1
y_b_top = torch.ones(N_b//4, 1)
u_b_top = torch.sin(np.pi * x_b_top)  # u=sin(πx)

x_b = torch.cat([x_b_left, x_b_right, x_b_bottom, x_b_top], dim=0)
y_b = torch.cat([y_b_left, y_b_right, y_b_bottom, y_b_top], dim=0)
u_b = torch.cat([u_b_left, u_b_right, u_b_bottom, u_b_top], dim=0)

# 4. Train model
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
losses = []  # record loss

for epoch in range(5000):
    optimizer.zero_grad()
    loss = compute_loss(model, x_f, y_f, x_b, y_b, u_b)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 5. Visualization
# Plot loss curve
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# Plot predicted solution heatmap
nx, ny = 100, 100
x_grid = torch.linspace(0, 1, nx).reshape(-1, 1)
y_grid = torch.linspace(0, 1, ny).reshape(-1, 1)
X, Y = torch.meshgrid(x_grid.squeeze(), y_grid.squeeze(), indexing='ij')
X_flat, Y_flat = X.reshape(-1, 1), Y.reshape(-1, 1)
u_pred = model(X_flat, Y_flat).detach().numpy().reshape(nx, ny)

plt.subplot(1, 2, 2)
plt.imshow(u_pred, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
plt.colorbar(label='u(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Predicted Solution')
plt.tight_layout()
plt.show()

```

## 📖 Code Explanation

* **Network**: 3 fully connected layers, Tanh activations for smoothness.
* **Loss**: PDE residual (second derivatives) and boundary condition loss, boundary weighted by 10.
* **Data**: Randomly sample interior and boundary points uniformly.
* **Training**: 5000 iterations, record loss curve.
* **Visualization**: Loss curve + 2D heatmap with `imshow`.

---

# Inverse Problem Example: Parameter Estimation

In Burgers equation, assume viscosity coefficient \$\nu\$ is unknown. Use observational data to estimate \$\nu\$. Add observation loss and treat \$\nu\$ as learnable.

## 📖 Code Example

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. Define neural network (add learnable parameter nu)
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),  # Input (x, t)
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)  # Output u(x, t)
        )
        self.nu = nn.Parameter(torch.tensor(0.1))  # Initial guess for nu, learnable

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# 2. Define loss function (add observation loss)
def compute_loss(model, x_f, t_f, x_i, t_i, u_i, x_b, t_b, u_b, x_o, t_o, u_o):
    x_f, t_f = x_f.requires_grad_(True), t_f.requires_grad_(True)
    u = model(x_f, t_f)
    
    # PDE residual
    u_t = torch.autograd.grad(u, t_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    pde_residual = u_t + u * u_x - model.nu * u_xx
    loss_pde = torch.mean(pde_residual ** 2)
    
    # Initial condition
    u_init = model(x_i, t_i)
    loss_init = torch.mean((u_init - u_i) ** 2)
    
    # Boundary condition
    u_bc = model(x_b, t_b)
    loss_bc = torch.mean((u_bc - u_b) ** 2)
    
    # Observation data loss
    u_obs = model(x_o, t_o)
    loss_obs = torch.mean((u_obs - u_o) ** 2)
    
    # Total loss
    return loss_pde + loss_init + loss_bc + loss_obs

# 3. Prepare data (add observation data)
N_f, N_i, N_b, N_o = 2000, 100, 100, 50  # Add observation points
true_nu = 0.01 / np.pi  # True nu

# PDE, initial, and boundary data (same as previous 1D example)
x_f = torch.rand(N_f, 1) * 2 - 1
t_f = torch.rand(N_f, 1)
x_i = torch.rand(N_i, 1) * 2 - 1
t_i = torch.zeros(N_i, 1)
u_i = -torch.sin(np.pi * x_i)
x_b = torch.cat([torch.ones(N_b//2, 1), -torch.ones(N_b//2, 1)])
t_b = torch.rand(N_b, 1)
u_b = torch.zeros(N_b, 1)

# Observation data (simulation: true solution + noise)
x_o = torch.rand(N_o, 1) * 2 - 1
t_o = torch.rand(N_o, 1)
# Simulated observation u_o (assume analytical or numerical solution exists; here simplified using sin + noise)
u_o = -torch.sin(np.pi * x_o) * torch.exp(-true_nu * np.pi**2 * t_o) + 0.01 * torch.randn(N_o, 1)

# 4. Train model
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
losses = []
nu_history = []  # Record nu estimation

for epoch in range(2000):
    optimizer.zero_grad()
    loss = compute_loss(model, x_f, t_f, x_i, t_i, u_i, x_b, t_b, u_b, x_o, t_o, u_o)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    nu_history.append(model.nu.item())
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Estimated nu: {model.nu.item():.6f}")

# 5. Visualization
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 3, 2)
plt.plot(nu_history)
plt.axhline(true_nu, color='r', linestyle='--', label='True nu')
plt.xlabel('Epoch')
plt.ylabel('nu')
plt.title('nu Estimation')
plt.legend()

# Animation visualization: u(x, t) changes over time
fig, ax = plt.subplots()
x_test = torch.linspace(-1, 1, 100).reshape(-1, 1)
def animate(t_frame):
    ax.clear()
    t_test = torch.ones_like(x_test) * (t_frame / 50.0)  # t from 0 to 1
    u_pred = model(x_test, t_test).detach().numpy()
    ax.plot(x_test.numpy(), u_pred, label=f't={t_frame/50:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.set_ylim(-1.5, 1.5)
    ax.legend()
ani = FuncAnimation(fig, animate, frames=50, interval=100)
plt.close(fig)  # Prevent static display
ani.save('burgers_animation.gif', writer='imagemagick')  # Save as GIF (imagemagick required)
plt.show()  # Or directly show animation

```

## 📖 Code Explanation

* **Network**: Add `self.nu` as learnable parameter (initial 0.1).
* **Loss**: Add observation loss `loss_obs` using simulated data (true solution + noise).
* **Training**: Record `nu_history`, estimate \$\nu\$.
* **Visualization**: Loss curve, \$\nu\$ estimation curve + animation of \$u(x,t)\$ (saved as GIF).


## 📖 Notes and Extensions

* **Multi-dimensional PDEs**: For 3D or higher, adjust input layer (e.g. `nn.Linear(3, ...)`) and ensure uniform sampling.
* **Inverse problems**: Observational data may come from experiments/simulations; for complex parameters (e.g. functional), use additional networks.
* **Visualization**: Use `FuncAnimation` for dynamic PDEs, heatmaps for steady-state.
* **Optimization**: If convergence is slow, try L-BFGS or more sample points.
* **Running**: Code requires PyTorch, NumPy, Matplotlib. Animations require imagemagick for GIF output.

