# LoHA Fine-tuning Method (Low-rank Hadamard Product Approximation)

## 📖 1. Definition

**LoHA** is a Parameter-Efficient Fine-Tuning (PEFT) method. It is similar to **LoRA**, but introduces a **Hadamard element-wise product** in the low-rank decomposition, which enhances representational capacity while maintaining low-rank updates.

* **LoRA**: For a weight matrix \$W \in \mathbb{R}^{d \times k}\$, it applies low-rank decomposition

$$
\Delta W = B A, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, \; r \ll \min(d,k)
$$

* **LoHA**: Based on low-rank decomposition, introduces a Hadamard product (element-wise multiplication) to enhance parameterization ability:

$$
\Delta W = (B A) \odot (D C)
$$

  Where:

  * \$B, A\$ are the first set of low-rank decomposition parameters;
  * \$D, C\$ are the second set of low-rank decomposition parameters;
  * \$\odot\$ denotes element-wise multiplication (Hadamard product).

Thus, compared with LoRA, LoHA can represent more complex variations with a similar number of parameters.



## 📖 2. Mathematical Formulation

Let the original weight be \$W\$. During LoHA training, \$W\$ is frozen, and only \$\Delta W\$ is trained:

1. **Effective Weight**:

$$
W^{\text{eff}} = W + \Delta W
$$

2. **LoHA Low-rank Update**:

$$
\Delta W = (B A) \odot (D C)
$$

Where:

* \$B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}\$
* \$D \in \mathbb{R}^{d \times r}, C \in \mathbb{R}^{r \times k}\$
* \$\odot\$ is the element-wise product.

During training, only \$(A, B, C, D)\$ are updated, while the original weight \$W\$ remains frozen.

---

## 📖 3. Minimal Code Example

A minimal **PyTorch** implementation of a LoHA linear layer:

```python
import torch
import torch.nn as nn

class LoHALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        # Frozen original weight
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        
        # LoHA parameters (two sets of low-rank decompositions)
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.C = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.D = nn.Parameter(torch.randn(out_features, rank) * 0.01)

    def forward(self, x):
        # LoHA low-rank update
        delta_W = (self.B @ self.A) * (self.D @ self.C)  # Hadamard product
        W_eff = self.weight + delta_W
        return x @ W_eff.T  # Linear layer computation

# ===== Test =====
x = torch.randn(2, 10)   # Input
layer = LoHALinear(10, 5, rank=4)
out = layer(x)
print("Output shape:", out.shape)
```

Execution result: `Output shape: torch.Size([2, 5])`, indicating that the LoHA linear layer works correctly.



📖 Summary:

* **LoRA**: Low-rank additive update \$\Delta W = BA\$.
* **LoHA**: Low-rank + Hadamard update \$\Delta W = (BA) \odot (DC)\$, providing stronger representational power.



