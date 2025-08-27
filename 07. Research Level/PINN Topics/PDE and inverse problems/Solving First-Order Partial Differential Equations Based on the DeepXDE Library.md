Here is a complete example using **DeepXDE** (based on PyTorch) to solve a first-order partial differential equation: the linear advection equation

$$
u_t + c\,u_x = 0,\quad (x,t)\in(0,1)\times(0,1],\quad
u(x,0)=\sin(2\pi x),\ c=1.
$$

This problem has the analytical solution \$u(x,t)=\sin(2\pi(x-ct))\$. For the over-determined hyperbolic equation with \$c>0\$, it is only necessary to impose the Dirichlet boundary condition at the inflow boundary \$x=0\$:

$$
u(0,t)=\sin(-2\pi t).
$$

---

### Code (runnable directly)

```python
# pip install deepxde torch matplotlib
import numpy as np
import deepxde as dde

# Use PyTorch backend
dde.backend.set_default_backend("pytorch")
from deepxde.backend import torch

# Constants
c = 1.0
pi2 = 2 * np.pi

# PDE: u_t + c u_x = 0
def pde(x, u):
    du_x = dde.grad.jacobian(u, x, i=0)     # ∂u/∂x, x at dim 0
    du_t = dde.grad.jacobian(u, x, i=1)     # ∂u/∂t, t at dim 1
    return du_t + c * du_x

# Analytical solution (for IC, BC, and error evaluation)
def true_solution(x):
    # x: (N,2) -> [:,0]=space, [:,1]=time
    return np.sin(pi2 * (x[:, 0:1] - c * x[:, 1:2]))

# Initial condition: t=0
def ic_func(x):
    return np.sin(pi2 * x[:, 0:1])  # u(x,0)

# Inflow boundary x=0
def inflow_bc(x):
    # u(0,t) = sin(2π(0 - c t)) = sin(-2π t)
    return np.sin(-pi2 * x[:, 1:2])

# Select boundary: x=0
def on_x0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0)

# Geometry and time domain
geom = dde.geometry.Interval(0.0, 1.0)
timedomain = dde.geometry.TimeDomain(0.0, 1.0)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Condition definitions
ic = dde.icbc.IC(geomtime, ic_func, lambda x, on_b: on_b and np.isclose(x[1], 0.0))
bc_in = dde.icbc.DirichletBC(geomtime, inflow_bc, on_x0)

# Build dataset (number of sample points adjustable as needed)
data = dde.data.TimePDE(
    geomtime,
    pde,
    [ic, bc_in],
    num_domain=8000,
    num_boundary=2000,
    num_initial=2000,
    train_distribution="pseudo",
)

# Network
net = dde.maps.FNN([2] + [64] * 4 + [1], "tanh", "Glorot uniform")

# Model
model = dde.Model(data, net)

# Training (first Adam, then L-BFGS)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(epochs=15000, display_every=1000)
model.compile("L-BFGS")
losshistory, train_state = model.train()

# Error evaluation
X_test = geomtime.random_points(2000)
u_pred = model.predict(X_test)
u_true = true_solution(X_test)
rel_l2 = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
print("Relative L2 error:", rel_l2)

# Visualization: plot u(x,t) at several time instants
import matplotlib.pyplot as plt
xs = np.linspace(0, 1, 200)
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    XT = np.stack([xs, np.full_like(xs, t)], axis=1)
    u_p = model.predict(XT).squeeze()
    u_t = true_solution(XT).squeeze()
    plt.plot(xs, u_p, label=f"PINN t={t:.2f}")
    plt.plot(xs, u_t, "--", label=f"True t={t:.2f}")
plt.xlabel("x"); plt.ylabel("u(x,t)")
plt.title("Advection: u_t + u_x = 0")
plt.legend(ncol=2, fontsize=8)
plt.show()
```

---

### Notes and Key Points

* **Problem type**: first-order hyperbolic PDE (linear advection). PINNs are very suitable for such initial-inflow boundary value problems.
* **Boundary setting**: for \$c>0\$, impose only a Dirichlet condition at the **inflow** boundary \$x=0\$; no condition is needed at the outflow end.
* **Loss composition**: DeepXDE combines the MSE of PDE residual, initial condition, and boundary condition.
* **Network and sampling**: hidden layer width/number and sample sizes (`num_domain/num_boundary/num_initial`) significantly affect convergence. The above values work on a typical laptop; if the error is large, increase samples or training epochs.
* **Validation**: the analytical solution is provided to compute the relative \$L^2\$ error and comparison curves.

---

要不要我也帮你做一个 **纯英文版本（不带中文原文）**，方便直接发到英文社区？

