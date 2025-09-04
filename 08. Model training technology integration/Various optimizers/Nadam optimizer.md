# Nadam Optimizer

## 📖 1. Definition

**Nadam (Nesterov-accelerated Adaptive Moment Estimation)** is a commonly used deep learning optimization algorithm that combines the ideas of **Adam** and **Nesterov momentum**.

* Adam: dynamically adjusts the learning rate using the first moment estimate (momentum) and the second moment estimate (exponentially weighted average of squared gradients).
* Nesterov momentum: an improved momentum method that looks ahead one step to accelerate convergence and reduce oscillations.

Nadam introduces Nesterov momentum into the Adam update rule, thus leveraging look-ahead gradient correction while retaining adaptive learning rates.



## 📖 2. Mathematical Formulation

Let:

* Parameters: \$\theta\_t\$
* Loss gradient: \$g\_t = \nabla\_\theta f\_t(\theta\_{t-1})\$
* First moment (momentum): \$m\_t\$
* Second moment: \$v\_t\$
* Learning rate: \$\alpha\$
* Decay coefficients: \$\beta\_1, \beta\_2 \in \[0,1)\$
* Numerical stability term: \$\epsilon\$

Steps:

1. Gradient calculation:

$$
g_t = \nabla_\theta f_t(\theta_{t-1})
$$

2. First and second moment updates:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

3. Bias correction:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

4. **Nesterov correction**:

$$
\tilde{m}_t = \beta_1 \hat{m}_t + \frac{(1-\beta_1)}{1-\beta_1^t} g_t
$$

5. Parameter update:

$$
\theta_t = \theta_{t-1} - \alpha \cdot \frac{\tilde{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

---

## 3. Minimal Code Example

A **NumPy** implementation of a minimal Nadam update (only core logic):

```python
import numpy as np

# Initialize parameters
theta = np.array([1.0])       # initial parameter
m, v = 0, 0                   # first moment & second moment
beta1, beta2 = 0.9, 0.999
alpha, eps = 0.001, 1e-8

# A simple objective function f(x) = x^2
def grad(theta):
    return 2 * theta

# Iterative updates
for t in range(1, 101):
    g = grad(theta)

    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * (g ** 2)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    m_nesterov = beta1 * m_hat + (1 - beta1) / (1 - beta1 ** t) * g

    theta -= alpha * m_nesterov / (np.sqrt(v_hat) + eps)

print("Optimized parameter:", theta)
```

This code drives parameter \$\theta\$ closer to 0 (the minimum of \$f(x)=x^2\$).



## 📖 PyTorch Nadam Example

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

# Generate data y = 2x + 3
x = torch.randn(100, 1)
y = 2 * x + 3 + 0.1 * torch.randn(100, 1)  # add some noise

# Define model, loss function, and Nadam optimizer
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.NAdam(model.parameters(), lr=0.01)

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



## 📖 Code Explanation

1. **Model**: a simple linear model `y = wx + b`.
2. **Data**: input `x` is randomly generated, output `y` follows the true function `2x + 3` with small noise added.
3. **Optimizer**: uses `torch.optim.NAdam`, with `lr=0.01`.
4. **Training loop**: forward pass → compute loss → backward pass → parameter update.
5. **Result**: after training, parameters `w, b` will be very close to `2` and `3`.



