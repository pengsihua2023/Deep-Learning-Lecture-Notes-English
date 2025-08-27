
Here is a **simple ODE (ordinary differential equation) solving code example based on the DeepXDE library**.
We take the classical first-order ODE

$$
\frac{dy}{dx} = -y, \quad y(0) = 1
$$

Its analytical solution is \$y(x) = e^{-x}\$.

---

```python
import deepxde as dde
import numpy as np


# Define the differential equation dy/dx + y = 0
def ode(x, y):
    dy_dx = dde.grad.jacobian(y, x, i=0, j=0)
    return dy_dx + y


# Define the initial condition y(0) = 1
ic = dde.IC(
    geom=dde.geometry.Interval(0, 5),  # Define interval [0,5]
    func=lambda x: 1,                  # Initial value y(0)=1
    on_boundary=lambda x, _: np.isclose(x[0], 0),
)

# Define geometry region
geom = dde.geometry.Interval(0, 5)

# Build dataset
data = dde.data.PDE(geom, ode, ic, num_domain=50, num_boundary=2)

# Build neural network
net = dde.nn.FNN([1] + [50] * 3 + [1], "tanh", "Glorot normal")

# PINN model
model = dde.Model(data, net)

# Training
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(epochs=5000)

# Test prediction
X = np.linspace(0, 5, 100)[:, None]
y_pred = model.predict(X)

# True solution
y_true = np.exp(-X)

# Plotting
import matplotlib.pyplot as plt

plt.plot(X, y_true, "k-", label="True solution")
plt.plot(X, y_pred, "r--", label="DeepXDE prediction")
plt.legend()
plt.show()
```

---

### Code explanation:

1. **ode function** defines the differential equation \$\frac{dy}{dx} + y = 0\$.
2. **IC** defines the initial condition \$y(0) = 1\$.
3. Training data is constructed using `dde.data.PDE`.
4. Neural network `FNN` is used for approximate solution.
5. After training, predictions are made and compared with the analytical solution \$e^{-x}\$.

In this way, you can use **DeepXDE** to obtain a neural network approximation of the ODE solution.

---

要不要我把这个英文版帮你导出为一个 `.md` 文件？
