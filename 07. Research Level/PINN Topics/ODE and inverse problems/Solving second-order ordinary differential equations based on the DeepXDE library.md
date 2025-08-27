
## A Example of a Second-Order ODE: Simple Harmonic Oscillator

$$
u''(x) + u(x) = 0,\quad x\in(0, 2\pi),\qquad
u(0)=1, u'(0)=0.
$$

Analytical solution: \$u^\*(x)=\cos x\$ .

```python
# pip install deepxde tensorflow matplotlib  (if not installed)

import numpy as np
import deepxde as dde
from deepxde.backend import tf  # use TF backend
import matplotlib.pyplot as plt

# 1) Interval [0, 2π]
L = 2 * np.pi
geom = dde.geometry.Interval(0.0, L)

# 2) PDE residual: u''(x) + u(x) = 0
def pde(x, y):
    dy_x   = dde.grad.jacobian(y, x, i=0, j=0)   # u'(x)
    d2y_xx = dde.grad.hessian(y, x, i=0, j=0)    # u''(x)
    return d2y_xx + y                             # u'' + u = 0

# 3) Apply boundary condition only at the left endpoint x=0
def on_left(x, on_boundary):
    # DeepXDE inputs x as [N, d], here d=1
    return on_boundary and tf.less_equal(x[:, 0:1], 1e-8)

# Dirichlet: u(0) = 1
bc_u0 = dde.DirichletBC(geom, lambda x: 1.0, on_left)

# Neumann: u'(0) = 0 (equivalent to normal derivative for the interval)
bc_du0 = dde.NeumannBC(geom, lambda x: 0.0, on_left)

# 4) (Optional) Analytical solution for evaluation
def exact(x):
    return np.cos(x)

# 5) Build data (domain/boundary sampling)
data = dde.data.PDE(
    geom,
    pde,
    [bc_u0, bc_du0],
    num_domain=200,        # collocation points in domain
    num_boundary=20,       # boundary sample points (sampled at both ends, but only effective at on_left)
    solution=exact,
    num_test=1000,
)

# 6) Network
net = dde.maps.FNN(
    layer_sizes=[1, 64, 64, 64, 1],
    activation="tanh",
    initializer="Glorot normal",
)

model = dde.Model(data, net)

# 7) Training: Adam -> L-BFGS
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(epochs=6000, display_every=1000)

model.compile("L-BFGS")
losshistory, train_state = model.train()

# 8) Prediction and evaluation
X = np.linspace(0, L, 400)[:, None]
u_pred  = model.predict(X)
u_exact = exact(X)
l2_rel = np.linalg.norm(u_pred - u_exact, 2) / np.linalg.norm(u_exact, 2)
print("Relative L2 error:", l2_rel)

# Visualization
plt.figure(figsize=(7,4))
plt.plot(X, u_exact, label="Exact: cos(x)", linestyle="--")
plt.plot(X, u_pred,  label="PINN prediction")
plt.xlabel("x"); plt.ylabel("u(x)")
plt.title("2nd-order ODE: u'' + u = 0 with u(0)=1, u'(0)=0")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.show()
```

### Notes

* Here both conditions \$u(0)=1\$ and \$u'(0)=0\$ are placed at the left endpoint \$x=0\$ (equivalent to an initial value problem).
* `on_left` is used to impose `DirichletBC` and `NeumannBC` only at the left endpoint; the right endpoint is not affected by these constraints.
* The training objective is still to satisfy **PDE residual** and **boundary (initial) conditions** simultaneously; no true solution labels are needed.
* If you want to change it into a boundary value problem (for example \$u(0)=1,u(L)=\cos L\$), replace the second condition with another `DirichletBC` that only acts at the right endpoint: write an `on_right` function to check \$x\approx L\$.

---


