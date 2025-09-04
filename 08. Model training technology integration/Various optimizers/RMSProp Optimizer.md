# RMSProp Optimizer

RMSProp (Root Mean Square Propagation) is a commonly used **adaptive learning rate optimizer**, mainly for training neural networks. It is an improvement of the Adagrad optimizer, proposed by Geoffrey Hinton in 2012.

## 📖 Background

* **Problem of Adagrad**: Adagrad continuously accumulates the sum of squared gradients, causing the learning rate to monotonically decrease over time, eventually becoming too small and affecting training.
* **Improvement of RMSProp**: RMSProp uses an **Exponential Moving Average (EMA)** to record the historical squared gradients instead of simple accumulation, so that the learning rate does not shrink indefinitely.

## 📖 Algorithm Principle

For each parameter \$\theta\$, at the \$t\$-th update:

1. Compute the gradient \$g\_t\$.
2. Update the moving average of the squared gradients:

<img width="290" height="45" alt="image" src="https://github.com/user-attachments/assets/88440188-10d0-4f98-bb9e-fede2630bfba" />

where \$\rho\$ is usually 0.9.
3\. Update the parameters with the adjusted learning rate:

\$\theta\_{t+1} = \theta\_t - \frac{\eta}{\sqrt{E\[g^2]\_t + \epsilon}} \cdot g\_t\$

* \$\eta\$: global learning rate (usually around 0.001).
* \$\epsilon\$: a constant to prevent division by zero (e.g., \$1e-8\$).



## 📖 Features

* Advantages

- Can automatically adjust the learning rate of different parameters, avoiding overly fast decay of the learning rate.
- Performs well on non-stationary targets (e.g., RNN training).
- More stable than Adagrad, with better convergence.

* Disadvantages

- Still requires manual selection of the initial learning rate.
- In some tasks, the convergence speed may not be as good as Adam.

## 📖 Application Scenarios

* Training RNN, LSTM, and other models in deep learning.
* One of the optimizer choices for image, speech, and other tasks.

---

## 📖 Example of Using RMSProp in PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple feedforward neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model, loss function, and optimizer
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9, eps=1e-08)

# Generate some dummy data
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# Training loop
for epoch in range(10):
    optimizer.zero_grad()           # Clear gradients
    outputs = model(X)              # Forward propagation
    loss = criterion(outputs, y)    # Compute loss
    loss.backward()                 # Backpropagation
    optimizer.step()                # Update parameters

    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")
```

## 📖 Key Notes

* `optim.RMSprop()` is the built-in RMSProp optimizer in PyTorch.
* Important parameters:

  * `lr=0.01`: learning rate (default is usually 0.001).
  * `alpha=0.9`: corresponds to \$\rho\$ in the algorithm, controlling the decay rate of historical squared gradients.
  * `eps=1e-08`: a small constant to prevent denominator from being 0.


