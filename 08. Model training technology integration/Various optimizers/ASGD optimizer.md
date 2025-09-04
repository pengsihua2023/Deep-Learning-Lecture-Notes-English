
# ASGD (Averaged Stochastic Gradient Descent) Optimizer

## 📖 1. Definition

**ASGD (Averaged Stochastic Gradient Descent)** is an improved version of **SGD**, with the core idea:

* Standard SGD often has high variance in parameter updates during training, which may cause instability in convergence.
* ASGD introduces **parameter averaging (Polyak-Ruppert averaging)** on top of SGD:

  * In addition to the normal parameter updates, it maintains an **averaged parameter vector**.
  * The final model weights are taken as the averaged parameters, which reduces noise and improves generalization ability.

Its typical applications include convex optimization tasks (e.g., linear models, logistic regression), where it converges faster and achieves better solutions.



## 📖 2. Mathematical Formulation

Let:

* Parameters: \$\theta\_t\$
* Learning rate: \$\eta\_t\$
* Gradient: \$g\_t = \nabla\_\theta f\_t(\theta\_t)\$
* Averaged parameters: \$\bar{\theta}\_t\$
* Decay factor: \$\lambda\$

### Update steps:

1. **SGD update**:

$$
\theta_{t+1} = \theta_t - \eta_t \cdot g_t
$$

2. **Averaged parameters update** (starting from some iteration \$t\_0\$):

$$
\bar{\theta}_ {t+1} = \frac{1}{t - t_0 + 1} \sum_{k=t_0}^{t} \theta_k
$$

The final model parameters are:

$$
\theta^* = \bar{\theta}_T
$$

This smooths out the oscillations of SGD and improves convergence stability.



## 📖 3. Minimal Code Example

Using **PyTorch** to implement a minimal ASGD training on a linear model:

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
y = 2 * x + 3 + 0.1 * torch.randn(100, 1)

# Define model, loss function, and ASGD optimizer
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.ASGD(model.parameters(), lr=0.05)

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



### 📖 Explanation

1. `torch.optim.ASGD` internally implements parameter averaging.
2. `lr=0.05` is the learning rate, adjustable depending on the task.
3. The final parameters are typically more stable than standard SGD and closer to the true values (here close to `w=2, b=3`).



