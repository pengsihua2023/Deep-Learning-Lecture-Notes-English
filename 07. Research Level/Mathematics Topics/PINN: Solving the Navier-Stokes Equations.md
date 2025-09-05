# PINN: Solving Navier-Stokes Equations

Physics-Informed Neural Networks (PINNs) can be used to solve the **Navier-Stokes equations**, which are a set of nonlinear partial differential equations describing fluid motion, widely applied in fluid dynamics, weather forecasting, aerospace, and more. PINNs embed the residuals of the Navier-Stokes equations, initial conditions, and boundary conditions into the neural network’s loss function, and use automatic differentiation to compute derivatives to approximate the solution. Below is a detailed introduction, including principles, implementation, and a simple PyTorch code example for solving the 2D incompressible Navier-Stokes equations with visualization.


## 📖 1. Introduction to Navier-Stokes Equations*

The Navier–Stokes equations describe the fluid velocity field \$\mathbf{u} = (u, v)\$ and the pressure field \$p\$. For 2D incompressible fluids, the equations take the form:

### Momentum equations:

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} 
= -\frac{1}{\rho} \frac{\partial p}{\partial x} + \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
$$

$$
\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} 
= -\frac{1}{\rho} \frac{\partial p}{\partial y} + \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)
$$

### Continuity equation (incompressibility condition):

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

Where:

* \$u, v\$: velocity components (in \$x, y\$ directions).
* \$p\$: pressure.
* \$\rho\$: fluid density (often set to 1).
* \$\nu\$: kinematic viscosity.
* Domain: \$(x,y) \in \Omega, ; t \in \[0, T]\$.

### Initial and boundary conditions:

* **Initial condition**:

$$
u(x,y,0) = u_0(x,y), \quad v(x,y,0) = v_0(x,y)
$$

* **Boundary conditions**: For example, Dirichlet conditions

$$
u = g_u, \quad v = g_v
$$

or Neumann conditions.



## 📖 2. Principles of PINNs for Navier-Stokes

PINNs approximate the velocity fields \$u(x,y,t), v(x,y,t)\$ and pressure field \$p(x,y,t)\$ via neural networks, solving as follows:

1. **Neural network**: Define a network with input \$(x,y,t)\$ and output \$(u,v,p)\$.

2. **Loss function**:

   * **PDE residuals**: Compute momentum and continuity equation residuals.
   * **Initial conditions**: Ensure velocity fields satisfy \$t=0\$.
   * **Boundary conditions**: Ensure specified boundary constraints (e.g., no-slip walls).
   * **Total loss**: \$L = L\_{\text{PDE}} + L\_{\text{init}} + L\_{\text{bc}}\$.

3. **Automatic differentiation**: Use PyTorch `torch.autograd` to compute derivatives (e.g., \$\frac{\partial u}{\partial t}, ; \frac{\partial^2 u}{\partial x^2}\$).

4. **Optimization**: Minimize the loss function using an optimizer (e.g., Adam).

---

## 📖 3. Simple Code Example: 2D Navier-Stokes Equations

Below is a simplified PyTorch example solving the 2D incompressible Navier–Stokes equations in a rectangular domain $\[0,1] \times \[0,1]\$, over $\[0,1]\$ in time. We assume:

* **Initial conditions**: \$u(x,y,0) = \sin(\pi x) \cos(\pi y), \quad v(x,y,0) = -\cos(\pi x) \sin(\pi y).\$
* **Boundary conditions**: No-slip walls, \$u=v=0\$ at boundaries.
* **Parameters**: \$\rho = 1, ; \nu = 0.01.\$
* **Problem**: Steady or transient flow, with incompressibility enforced by the continuity equation.

#### **Code**

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Define PINN network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 50),  # Input (x, y, t)
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 3)  # Output (u, v, p)
        )

    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        return self.net(inputs)

# 2. Define loss function
def compute_loss(model, x_f, y_f, t_f, x_i, y_i, t_i, u_i, v_i, x_b, y_b, t_b, u_b, v_b, rho=1.0, nu=0.01):
    x_f, y_f, t_f = x_f.requires_grad_(True), y_f.requires_grad_(True), t_f.requires_grad_(True)
    uvp = model(x_f, y_f, t_f)
    u, v, p = uvp[:, 0:1], uvp[:, 1:2], uvp[:, 2:3]

    # PDE residuals
    u_t = torch.autograd.grad(u, t_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_f, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    
    v_t = torch.autograd.grad(v, t_f, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x_f, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y_f, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x_f, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y_f, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    
    p_x = torch.autograd.grad(p, x_f, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y_f, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    
    # Navier-Stokes momentum equations
    pde_u = u_t + u * u_x + v * u_y + (1/rho) * p_x - nu * (u_xx + u_yy)
    pde_v = v_t + u * v_x + v * v_y + (1/rho) * p_y - nu * (v_xx + v_yy)
    # Continuity equation
    pde_cont = u_x + v_y
    
    loss_pde = torch.mean(pde_u**2 + pde_v**2 + pde_cont**2)
    
    # Initial conditions
    uvp_i = model(x_i, y_i, t_i)
    u_init, v_init = uvp_i[:, 0:1], uvp_i[:, 1:2]
    loss_init = torch.mean((u_init - u_i)**2 + (v_init - v_i)**2)
    
    # Boundary conditions
    uvp_b = model(x_b, y_b, t_b)
    u_bc, v_bc = uvp_b[:, 0:1], uvp_b[:, 1:2]
    loss_bc = torch.mean((u_bc - u_b)**2 + (v_bc - v_b)**2)
    
    # Total loss
    return loss_pde + 10 * loss_init + 10 * loss_bc

# 3. Prepare data
N_f = 10000  # PDE interior points
N_i = 200    # Initial points
N_b = 400    # Boundary points

# PDE interior points
x_f = torch.rand(N_f, 1)
y_f = torch.rand(N_f, 1)
t_f = torch.rand(N_f, 1)

# Initial conditions
x_i = torch.rand(N_i, 1)
y_i = torch.rand(N_i, 1)
t_i = torch.zeros(N_i, 1)
u_i = torch.sin(np.pi * x_i) * torch.cos(np.pi * y_i)
v_i = -torch.cos(np.pi * x_i) * torch.sin(np.pi * y_i)

# Boundary conditions (no-slip)
x_b_left = torch.zeros(N_b//4, 1)  # x=0
y_b_left = torch.rand(N_b//4, 1)
t_b_left = torch.rand(N_b//4, 1)
x_b_right = torch.ones(N_b//4, 1)  # x=1
y_b_right = torch.rand(N_b//4, 1)
t_b_right = torch.rand(N_b//4, 1)
x_b_bottom = torch.rand(N_b//4, 1)  # y=0
y_b_bottom = torch.zeros(N_b//4, 1)
t_b_bottom = torch.rand(N_b//4, 1)
x_b_top = torch.rand(N_b//4, 1)  # y=1
y_b_top = torch.ones(N_b//4, 1)
t_b_top = torch.rand(N_b//4, 1)

x_b = torch.cat([x_b_left, x_b_right, x_b_bottom, x_b_top], dim=0)
y_b = torch.cat([y_b_left, y_b_right, y_b_bottom, y_b_top], dim=0)
t_b = torch.cat([t_b_left, t_b_right, t_b_bottom, t_b_top], dim=0)
u_b = torch.zeros(N_b, 1)
v_b = torch.zeros(N_b, 1)

# 4. Train model
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
losses = []

for epoch in range(10000):
    optimizer.zero_grad()
    loss = compute_loss(model, x_f, y_f, t_f, x_i, y_i, t_i, u_i, v_i, x_b, y_b, t_b, u_b, v_b)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 5. Visualization
# Loss curve
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# Velocity field animation (t=0 to t=1)
nx, ny = 50, 50
x_grid = torch.linspace(0, 1, nx).reshape(-1, 1)
y_grid = torch.linspace(0, 1, ny).reshape(-1, 1)
X, Y = torch.meshgrid(x_grid.squeeze(), y_grid.squeeze(), indexing='ij')
X_flat, Y_flat = X.reshape(-1, 1), Y.reshape(-1, 1)

fig, ax = plt.subplots()
def animate(t_frame):
    ax.clear()
    t_test = torch.ones_like(X_flat) * (t_frame / 50.0)
    uvp_pred = model(X_flat, Y_flat, t_test)
    u_pred = uvp_pred[:, 0].detach().numpy().reshape(nx, ny)
    ax.imshow(u_pred, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'u(x, y, t={t_frame/50:.2f})')
ani = FuncAnimation(fig, animate, frames=50, interval=100)
plt.close(fig)
ani.save('navier_stokes_animation.gif', writer='imagemagick')  # requires imagemagick
plt.show()

```


## 📖 Code Explanation

* **Network:**

  * Input: \$(x, y, t)\$, Output: \$(u, v, p)\$.
  * 3 fully connected layers, 50 neurons, Tanh activation.

* **Loss function:**

  * PDE residuals: momentum equations (\$x\$ and \$y\$ directions) + continuity equation.
  * Initial conditions: \$u(x, y, 0) = \sin(\pi x)\cos(\pi y), ; v(x, y, 0) = - \cos(\pi x)\sin(\pi y)\$.
  * Boundary conditions: No-slip, \$u=v=0\$ at the boundary.
  * Loss weighting: initial and boundary losses weighted by \$10\$ to strengthen constraints.

* **Data:**

  * Randomly sampled interior points (10000), initial points (200), boundary points (400).
  * Initial condition simulates a vortex flow field, boundaries are no-slip.

* **Training:**

  * Adam optimizer, 10000 iterations, loss tracked.

* **Visualization:**

  * Loss curve: shows convergence.
  * Animation: heatmap of \$u(x, y, t)\$ evolving in time, saved as GIF.


## 📖 Notes

1. **Computational complexity**:

   * Navier-Stokes is highly nonlinear, training may be slow. GPU is recommended (`model.cuda(), x_f.cuda(), ...`).
   * Try L-BFGS optimizer for faster convergence.

2. **Sampling points**:

   * Increasing \$N\_f, N\_i, N\_b\$ improves accuracy but increases cost.
   * Ensure boundary points are uniformly distributed.

3. **Initial conditions**:

   * Here analytical forms are used; in practice, experimental or numerical data can be used.


**4. Visualization:**

* Requires `matplotlib` and `imagemagick` (for GIF saving).
* Can add heatmaps of \$v\$ and \$p\$, or velocity vector fields (`plt.quiver`).

**5. Extensions:**

* **Inverse problems**: Treat \$\nu\$ or \$p\$ as learnable parameters (`nn.Parameter`) with observational data.
* **Complex geometries**: Define more complex boundaries (e.g., circular obstacles).
* **3D problems**: Extend input \$(x, y, z, t)\$, output \$(u, v, w, p)\$.

---

# Extension: Inverse Problems
To estimate \$\nu\$, modify the `PINN` class:

```python
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 3)
        )
        self.nu = nn.Parameter(torch.tensor(0.1))  # learnable nu

    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        return self.net(inputs)
```

In `compute_loss`, use `model.nu` and add observation data loss (similar to earlier inverse examples).

---

## 📖 Conclusion

PINNs can effectively solve Navier-Stokes equations. The code embeds residuals of the momentum and continuity equations, combined with initial and boundary conditions, to approximate velocity and pressure fields. The example demonstrates a basic 2D implementation with visualization.


