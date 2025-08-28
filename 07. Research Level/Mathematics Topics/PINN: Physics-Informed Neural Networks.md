

## PINN: Physics-Informed Neural Networks

### Principles and Usage

#### **Introduction**

Physics-Informed Neural Networks (PINNs) are a neural network framework that combines deep learning and physical laws, used for solving Partial Differential Equations (PDEs) or simulating physical systems. PINNs embed physical equations (such as governing equations, initial conditions, and boundary conditions) into the neural network loss function and approximate PDE solutions by optimizing network parameters. Compared with traditional numerical methods (such as finite difference, finite element), PINNs do not require mesh discretization, making them suitable for high-dimensional or complex geometrical problems.

#### **Principles**

1. **Core Idea**:

* PINNs use a neural network \$u(x,t;\theta)\$ (with parameters \$\theta\$) to approximate the solution of PDE \$u(x,t)\$.
* By incorporating PDE residuals, initial conditions, and boundary conditions into the loss function, the network output is constrained to satisfy physical laws.
* The optimization objective is to minimize the loss function so that the network output approximates the true solution.

2. **PDE Form**: Consider a general PDE:

$$
\mathcal{N}[u(x,t)] = f(x,t), \quad (x,t) \in \Omega
$$

with initial and boundary conditions:

$$
u(x,0) = u_{0}(x), \quad u(x,t) \in \partial \Omega = g(x,t)
$$

where:

* \$\mathcal{N}\$: PDE operator (e.g., \$\frac{\partial u}{\partial t} + \frac{\partial^2 u}{\partial x^2} = 0\$).
* \$\Omega\$: Domain.
* \$u\_0, g\$: Initial and boundary conditions.

3. **Loss Function**:
   PINNs’ loss function consists of three parts:

* **PDE Residual Loss:**

$$
L_{\text{PDE}} = \frac{1}{N_f} \sum_{i=1}^{N_f} \left| \mathcal{N}[u(x_i, t_i; \theta)] - f(x_i, t_i) \right|^2
$$

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

where \$\lambda\_1, \lambda\_2, \lambda\_3\$ are weights to balance the components.

4. **Automatic Differentiation**:

* PINNs use neural network automatic differentiation (autograd) to compute PDE derivatives (e.g., \$\frac{\partial u}{\partial t}, \frac{\partial^2 u}{\partial x^2}\$), without explicit discretization.
* Network input is spatial and temporal coordinates \$(x,t)\$, output is solution \$u(x,t)\$.

5. **Advantages**:

   * No mesh discretization, suitable for high dimensions or complex geometry.
   * Flexible and easy to incorporate physical constraints.
   * Can combine small amounts of observational data with physics.

6. **Disadvantages**:

   * Training may be slow, optimization may be unstable for complex PDEs.
   * Loss weights \$\lambda\_i\$ need tuning.
   * Accuracy for complex nonlinear PDEs may be lower than traditional numerical methods.

7. **Applications**:

   * Solving PDEs (e.g., heat conduction, wave equation, fluid dynamics).
   * Data-driven physical modeling (e.g., incorporating experimental data).
   * Inverse problems (e.g., parameter estimation).

---

#### **PyTorch Usage**

Below is a simple PyTorch code example showing how to solve the 1D Burgers equation with PINNs.

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

where \$\nu = \frac{0.01}{\pi}\$ is viscosity.

---

##### **Code Example**

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
    x_f, t_f = x_f.requires_grad_(True), t_f.requires_grad_(True)  # Need derivatives
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
x_i = torch.rand(N_i, 1) * 2 - 1  # Initial condition x
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


## 1. **Multi-dimensional PDE Example: 2D Heat Conduction (Laplace Equation Variant)**

2D heat conduction equation (simplified Laplace equation):

$$
\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0, 
\quad (x,y) \in [0,1] \times [0,1]
$$

**Boundary conditions:**

* \$u(0,y) = 0, \quad u(1,y) = 0\$
* \$u(x,0) = 0, \quad u(x,1) = \sin(\pi x)\$

This is a steady-state problem (no time dimension). PINN solves it by minimizing residuals.

---

### **Code Example**

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
    x_f, y_f = x_f.requires_grad_(True), y_f.requires_grad_(True)  # Require derivatives
    u = model(x_f, y_f)
    
    # PDE residual: Laplace equation u_xx + u_yy = 0
    u_x = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_f, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    pde_residual = u_xx + u_yy
    loss_pde = torch.mean(pde_residual ** 2)
    
    # Boundary conditions
    u_bc = model(x_b, y_b)
    loss_bc = torch.mean((u_bc - u_b) ** 2)
    
    # Total loss (weight boundary constraints)
    return loss_pde + 10 * loss_bc

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
losses = []

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


##### **Code Explanation**

* **Network**: A 3-layer fully connected network using Tanh activation to ensure smoothness.
* **Loss**: PDE residuals (second-order derivatives) and boundary condition loss, with boundary weight set to 10 to strengthen constraints.
* **Data**: Randomly sample interior and boundary points to ensure uniform distribution.
* **Training**: 5000 iterations with loss curve recording.
* **Visualization**: Loss curve + 2D heatmap of the predicted solution, using `imshow` to display the temperature distribution.

---

要不要我也把 **1D Burgers 示例的代码说明** 翻译成英文，和这个一起整理？


## 2. **Inverse Problem Example: Parameter Estimation**

In the Burgers equation, assume viscosity coefficient \$\nu\$ is unknown. Use observation data to estimate \$\nu\$ by adding observation loss and treating \$\nu\$ as a learnable parameter.

---

### **Code Example**

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Define neural network (with learnable parameter nu)
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
        self.nu = nn.Parameter(torch.tensor(0.1))  # Initial guess for nu (learnable)

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

# 3. Prepare data (with observation data)
N_f, N_i, N_b, N_o = 2000, 100, 100, 50  # Include observation points
true_nu = 0.01 / np.pi  # True viscosity

# PDE, initial, boundary data (same as 1D Burgers example)
x_f = torch.rand(N_f, 1) * 2 - 1
t_f = torch.rand(N_f, 1)
x_i = torch.rand(N_i, 1) * 2 - 1
t_i = torch.zeros(N_i, 1)
u_i = -torch.sin(np.pi * x_i)
x_b = torch.cat([torch.ones(N_b//2, 1), -torch.ones(N_b//2, 1)])
t_b = torch.rand(N_b, 1)
u_b = torch.zeros(N_b, 1)

# Observation data (simulated with true solution + noise)
x_o = torch.rand(N_o, 1) * 2 - 1
t_o = torch.rand(N_o, 1)
u_o = -torch.sin(np.pi * x_o) * torch.exp(-true_nu * np.pi**2 * t_o) + 0.01 * torch.randn(N_o, 1)

# 4. Train model
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
losses = []
nu_history = []

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

# Animation: u(x,t) evolution
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
ani.save('burgers_animation.gif', writer='imagemagick')  # Save as GIF (requires imagemagick)
plt.show()
```


##### **Code Explanation**

* **Network**: Add `self.nu` as a learnable parameter (initialized as 0.1).
* **Loss**: Add observation loss `loss_obs`, using simulated observation data (true solution + noise).
* **Training**: Record `nu_history` to estimate nu.
* **Visualization**: Loss curve, nu estimation curve, and animation of \$u(x, t)\$ (saved as GIF).

#### 3. **Notes and Extensions**

* **Multi-dimensional PDEs**: For 3D or higher dimensions, change the input layer to `nn.Linear(3, ...)` or more, and ensure sampling points are evenly distributed.
* **Inverse problems**: Observation data can come from experiments or simulations; for more complex parameters (e.g., functional parameters), additional networks can be used.
* **Visualization**: Use `FuncAnimation` for dynamic PDEs; heatmaps are more suitable for steady-state problems.
* **Optimization**: If convergence is slow, try L-BFGS optimizer or increase the number of sampling points.
* **Execution**: Requires PyTorch, NumPy, and Matplotlib. Animation requires imagemagick for GIF output.







