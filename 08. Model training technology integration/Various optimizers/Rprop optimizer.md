# Rprop (Resilient Backpropagation) Optimizer

## 1. Definition

**Rprop (Resilient Backpropagation)** is an optimization algorithm based on the **sign of gradients**, originally designed for training feedforward neural networks.
Its core ideas are:

* **Only use the sign of the gradient (positive/negative), not its magnitude**, to determine the update direction.
* Each parameter has its own independent update step size (learning rate), which is **adaptively adjusted** depending on whether the gradient direction is consistent across steps.
* This avoids problems of vanishing gradients (too small) or exploding gradients (too large).

Unlike SGD or Adam, Rprop does not directly scale updates by gradient values. Instead, it maintains an **independent step size Δ** and adjusts it dynamically according to the gradient sign.

---

## 2. Mathematical Formulation

Let:

* Parameters: \$\theta\_{t}\$
* Gradient: \$g\_t = \frac{\partial L}{\partial \theta\_t}\$
* Update step size: \$\Delta\_t\$
* Initial step size: \$\Delta\_0\$ (e.g. 0.1)
* Increase factor: \$\eta^+ > 1\$ (e.g. 1.2)
* Decrease factor: \$\eta^- < 1\$ (e.g. 0.5)
* Step size bounds: $\[\Delta\_{\min}, \Delta\_{\max}]\$

Rules:

1. **Compare the sign of current gradient with the previous gradient**:

   * If \$\text{sign}(g\_t) = \text{sign}(g\_{t-1})\$: directions are consistent, increase step size:

$$
\Delta_t = \min(\Delta_{t-1} \cdot \eta^+, \Delta_{\max})
$$

* If \$\text{sign}(g\_t) \neq \text{sign}(g\_{t-1})\$: means it overshot the minimum, decrease step size and undo the last update:

$$
\Delta_t = \max(\Delta_{t-1} \cdot \eta^-, \Delta_{\min})
$$

```
 and set $g_t = 0$ (to avoid oscillations).  
```

* Otherwise (if gradient is 0), keep step size unchanged.

2. **Parameter update**:

$$
\theta_{t+1} = \theta_t - \text{sign}(g_t) \cdot \Delta_t
$$

---

## 3. Minimal Code Example

A minimal Rprop implementation using **NumPy**:

```python
import numpy as np

# Target function f(x) = x^2
def grad(theta):
    return 2 * theta

theta = np.array([5.0])   # initial parameter
delta = np.array([0.1])   # initial step size
prev_grad = np.array([0.0])

eta_plus, eta_minus = 1.2, 0.5
delta_min, delta_max = 1e-6, 50

for t in range(50):
    g = grad(theta)

    # Same sign -> increase step size
    if g * prev_grad > 0:
        delta = np.minimum(delta * eta_plus, delta_max)
    # Opposite sign -> decrease step size, undo update
    elif g * prev_grad < 0:
        delta = np.maximum(delta * eta_minus, delta_min)
        g = 0  # avoid oscillation

    # Parameter update (using only the sign)
    theta -= np.sign(g) * delta
    prev_grad = g

    print(f"Iteration {t+1}: theta = {theta[0]:.6f}")

print("Optimized parameter:", theta)
```

After running, parameter \$\theta\$ will gradually converge to 0 (the minimum of the function).

---

## PyTorch Rprop Example

```python
import torch
import torch.nn as nn
import torch.optim as optim

# A simple linear model y = wx + b
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Construct data y = 2x + 3
x = torch.randn(100, 1)
y = 2 * x + 3 + 0.1 * torch.randn(100, 1)  # add some noise

# Define model, loss, and Rprop optimizer
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.Rprop(model.parameters(), lr=0.01)

# Training
for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/200], Loss: {loss.item():.4f}")

# Print learned parameters
w, b = model.linear.weight.item(), model.linear.bias.item()
print(f"Learned parameters: w = {w:.2f}, b = {b:.2f}")
```

---

### Explanation

1. **Model**: `SimpleModel` defines a one-layer linear regression model.
2. **Data**: Training target is `y = 2x + 3` with added noise.
3. **Optimizer**: Uses `torch.optim.Rprop`, which only depends on gradient signs, so the meaning of learning rate differs from SGD.
4. **Training loop**: Standard forward pass → compute loss → backpropagation → parameter update.
5. **Result**: Learned parameters `w` will be close to 2, and `b` close to 3.


