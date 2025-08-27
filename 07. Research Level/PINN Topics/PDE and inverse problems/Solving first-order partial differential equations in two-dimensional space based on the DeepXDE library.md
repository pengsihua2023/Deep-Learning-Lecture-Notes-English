## First-order PDE in 2D Space Based on DeepXDE Library

2D linear transport equation:

$$
u_t + a u_x + b u_y = 0, \quad (x,y,t)\in(0,1)\times(0,1)\times(0,1],
$$

Initial condition:

$$
u(x,y,0) = \sin(2\pi x)\sin(2\pi y),
$$

Analytical solution:

$$
u(x,y,t) = \sin\big(2\pi(x-a t)\big)\sin\big(2\pi(y-b t)\big).
$$

Here \$(a,b)\$ is the velocity vector.

---

### Code Example (DeepXDE + PyTorch)

```python
# pip install deepxde torch matplotlib
import numpy as np
import deepxde as dde

# Use PyTorch backend
dde.backend.set_default_backend("pytorch")

# Constants
a, b = 1.0, 0.5
pi2 = 2 * np.pi

# PDE: u_t + a u_x + b u_y = 0
def pde(x, u):
    du_x = dde.grad.jacobian(u, x, i=0)   # ∂u/∂x
    du_y = dde.grad.jacobian(u, x, i=1)   # ∂u/∂y
    du_t = dde.grad.jacobian(u, x, i=2)   # ∂u/∂t
    return du_t + a * du_x + b * du_y

# True solution
def true_solution(x):
    return np.sin(pi2*(x[:,0:1] - a*x[:,2:3])) * np.sin(pi2*(x[:,1:2] - b*x[:,2:3]))

# Initial condition t=0
def ic_func(x):
    return np.sin(pi2*x[:,0:1]) * np.sin(pi2*x[:,1:2])

def on_initial(x, on_boundary):
    return on_boundary and np.isclose(x[2], 0.0)

# Inflow boundary condition: Dirichlet condition needs to be imposed on the inflow faces along the velocity direction
def inflow_bc(x):
    return true_solution(x)

def on_x0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0)

def on_y0(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0.0)

# Geometry and time domain
geom = dde.geometry.Rectangle([0,0],[1,1])    # space (x,y)
timedomain = dde.geometry.TimeDomain(0,1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Condition definitions
ic = dde.icbc.IC(geomtime, ic_func, on_initial)
bc_x0 = dde.icbc.DirichletBC(geomtime, inflow_bc, on_x0)
bc_y0 = dde.icbc.DirichletBC(geomtime, inflow_bc, on_y0)

# Data
data = dde.data.TimePDE(
    geomtime,
    pde,
    [ic, bc_x0, bc_y0],
    num_domain=10000,
    num_boundary=2000,
    num_initial=2000,
    train_distribution="pseudo",
)

# Network
net = dde.maps.FNN([3] + [64]*4 + [1], "tanh", "Glorot uniform")

# Model
model = dde.Model(data, net)

# Training
model.compile("adam", lr=1e-3)
model.train(epochs=15000, display_every=1000)
model.compile("L-BFGS")
model.train()

# Error
X_test = geomtime.random_points(2000)
u_pred = model.predict(X_test)
u_true = true_solution(X_test)
rel_l2 = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
print("Relative L2 error:", rel_l2)
```

---

### Notes

* **Input dimension**: the input here is \$(x,y,t)\$, so the network input is 3D.
* **Boundary conditions**:

  * If \$a>0\$, then an inflow condition is required at \$x=0\$; if \$b>0\$, then an inflow condition is required at \$y=0\$.
  * If \$a<0\$ or \$b<0\$, the corresponding inflow boundary should be switched to \$x=1\$ or \$y=1\$.
* **Network structure**: a four-layer fully connected network with width 64 is generally sufficient. Can be adjusted according to accuracy requirements.
* **Sampling numbers**: `num_domain` / `num_boundary` / `num_initial` will affect training stability and accuracy.

---


