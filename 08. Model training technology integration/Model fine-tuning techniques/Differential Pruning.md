# Diff Pruning

## 📖 1. Definition

**Diff Pruning** is a **Parameter-Efficient Fine-Tuning (PEFT)** method, first proposed for efficient adaptation of large-scale pre-trained language models (Guo et al., 2021).

Core idea:

* Do not directly modify the pre-trained model parameters \$\theta\$, but instead introduce a **diff parameter** \$\Delta \theta\$ for each parameter.
* Only learn these diff parameters for downstream tasks, rather than the whole model.
* Through **sparsity constraints (such as L1 regularization or gating mechanisms)**, introduce differential updates only where necessary, achieving **efficiency + interpretability**.

👉 In simple terms:
The model parameter update changes from:

$$
\theta' = \theta + \Delta \theta
$$

to:

* **Most positions**: \$\Delta \theta = 0\$ (i.e., frozen, no update).
* **Few positions**: \$\Delta \theta \neq 0\$ (only update necessary parameters).



## 📖 2. Mathematical Description

### 2.1 Parameter Representation

Let:

* Pre-trained parameters: \$\theta \in \mathbb{R}^d\$
* Diff parameters: \$\Delta \theta \in \mathbb{R}^d\$
* Downstream model parameters:

$$
\theta' = \theta + \Delta \theta
$$

### 2.2 Loss Function

The training objective is to minimize the downstream task loss while making the diff parameters sparse:

$$
\mathcal{L}(\Delta \theta) = \mathcal{L}_{task}(f(x; \theta + \Delta \theta)) + \lambda \|\Delta \theta\|_1
$$

* \$\mathcal{L}\_{task}\$: downstream task loss (e.g., classification cross-entropy).
* \$|\Delta \theta|\_1\$: L1 regularization, encouraging sparsity.
* \$\lambda\$: regularization coefficient.

### 2.3 Pruning Mechanism

* During training, many \$\Delta \theta\$ values converge close to 0.
* Finally, these positions can be pruned, keeping only a small number of nonzero parameters.



## 📖 3. Simple Code Example (PyTorch)

Below is a PyTorch implementation of a **linear layer with diff pruning**:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple linear classification model
class BaseModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BaseModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        # Freeze pre-trained parameters
        for p in self.fc.parameters():
            p.requires_grad = False
        # Diff parameters (trainable)
        self.delta = nn.Parameter(torch.zeros_like(self.fc.weight))

    def forward(self, x):
        # Original parameters + diff parameters
        return (self.fc.weight + self.delta) @ x.T

# Simulated data
X = torch.randn(10, 5)   # batch=10, input_dim=5
y = torch.randint(0, 2, (10,))  # binary classification labels

# Initialize model
model = BaseModel(input_dim=5, output_dim=2)

# Loss function + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([model.delta], lr=0.01)

# Training
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs.T, y)
    # L1 regularization constraint on diff parameters
    loss = loss + 0.01 * torch.norm(model.delta, p=1)
    loss.backward()
    optimizer.step()

print("Trained Δθ (sparse updates):")
print(model.delta)
```



## 📖 4. Summary

* **Definition**: Diff Pruning fine-tunes pre-trained models by learning **sparse diff parameters**.
* **Mathematical formula**:

$$
\theta' = \theta + \Delta \theta, \quad 
\mathcal{L} = \mathcal{L}_{task} + \lambda \|\Delta \theta\|_1
$$

* **Features**:

  * Saves memory and computation (only updates a few parameters).
  * Retains the generalization ability of the pre-trained model.
  * Produces sparse, interpretable differential updates after pruning.
* **Code**: Freeze the original parameters, train only \$\Delta \theta\$, and add L1 regularization to enforce sparsity.


