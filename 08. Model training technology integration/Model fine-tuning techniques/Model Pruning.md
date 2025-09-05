

# Model Pruning

## 📖 1. Definition

**Model Pruning** is a **parameter-efficient fine-tuning method** that reduces model size and computation by **removing redundant or unimportant parameters/connections** in neural networks, while trying to maintain performance.

* Pruning is usually performed on **pre-trained models**, followed by **fine-tuning** the remaining parameters.
* Types of pruning:

  * **Unstructured pruning**: directly sets some weights to zero without changing the tensor shape.
  * **Structured pruning**: removes entire channels, convolution kernels, or attention heads.
* Fine-tuning stage:

  * Keeps the non-zero parameters and continues training on the downstream task to recover performance.



## 📖 2. Mathematical Formulation

Let the original model parameters be:

$$
W = \{ w_1, w_2, \dots, w_n \}
$$

Define a **mask vector** \$M = { m\_1, m\_2, \dots, m\_n }\$, where:

$$
m_i \in \{0, 1\}, \quad W' = W \odot M
$$

Where:

* \$m\_i = 0\$ means the parameter is pruned;
* \$m\_i = 1\$ means the parameter is kept;
* \$\odot\$ denotes element-wise multiplication.

The training objective becomes:

$$
\mathcal{L}(W', M) = - \sum_{(x, y)} \log p(y \mid x; W \odot M)
$$

During fine-tuning, \$M\$ is fixed, and only the remaining parameters \$W'\$ are optimized.

---

## 📖 3. Simple Code Demonstration (PyTorch)

The following code shows a minimal example of **unstructured pruning + fine-tuning** with PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

# A simple fully connected network
class SimpleNet(nn.Module):
    def __init__(self, input_size=100, hidden_size=50, num_classes=2):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Initialize model
model = SimpleNet()
print("Number of parameters (before pruning):", sum(p.numel() for p in model.parameters()))

# Perform unstructured pruning on the first layer weights (prune 50% of smallest weights)
prune.l1_unstructured(model.fc1, name="weight", amount=0.5)

# View pruning mask
print("Pruning mask:")
print(model.fc1.weight_mask)

# Run a forward pass with the pruned model
x = torch.randn(4, 100)
logits = model(x)
print("Output logits:", logits)

# Fine-tune after pruning
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
labels = torch.tensor([0, 1, 0, 1])
loss = F.cross_entropy(logits, labels)
loss.backward()
optimizer.step()
```



## 📖 Summary

* **Definition**: Model Pruning compresses models by removing unimportant weights/channels, then fine-tunes the remaining parameters.
* **Mathematical Formulation**: Applies a mask \$M\$ to sparsify parameters \$W\$, optimizing \$\mathcal{L}(W \odot M)\$.
* **Code**: PyTorch’s `torch.nn.utils.prune` provides convenient tools for weight pruning, followed by fine-tuning.



