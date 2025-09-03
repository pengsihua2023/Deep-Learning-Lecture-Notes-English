

# LoKr Fine-tuning Method (Low-rank Kronecker Product Adaptation)

## 1. Definition

**LoKr** is a Parameter-Efficient Fine-Tuning (PEFT) method. Similar to **LoRA** and **LoHA**, the core idea is to **freeze the original weights** of pre-trained large models and only learn a low-rank update matrix \$\Delta W\$.

The differences:

* **LoRA**: \$\Delta W = BA\$ , low-rank approximation.
* **LoHA**: \$\Delta W = (BA) \odot (DC)\$ , enhanced by Hadamard product.
* **LoKr**: \$\Delta W\$ uses the **Kronecker product (⊗)**, i.e.,

$$
\Delta W = A \otimes B
$$

where \$A, B\$ are smaller matrices.

This allows LoKr to represent a large matrix update with fewer parameters, saving memory and computation for larger models.



## 2. Mathematical Formulation

Let the original weight matrix be \$W \in \mathbb{R}^{d \times k}\$, which is frozen. The LoKr low-rank update is:

1. **Effective Weight**:

$$
W^{\text{eff}} = W + \Delta W
$$

2. **Kronecker Decomposition Form**:

$$
\Delta W = A \otimes B
$$

Where:

* \$A \in \mathbb{R}^{m \times n}\$, \$B \in \mathbb{R}^{p \times q}\$
* The Kronecker product result \$\Delta W \in \mathbb{R}^{(m p) \times (n q)}\$
* By choosing suitable \$m,n,p,q\$, it can approximate the original dimensions \$(d \times k)\$.

During training, only \$A, B\$ are updated, and their parameter count is much smaller than the full \$d \times k\$.



## 3. Minimal Code Example

A minimal **PyTorch** implementation of a LoKr linear layer:

```python
import torch
import torch.nn as nn

class LoKrLinear(nn.Module):
    def __init__(self, in_features, out_features, m=2, n=2):
        super().__init__()
        # Frozen original weight
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)

        # LoKr parameters (two small matrices)
        self.A = nn.Parameter(torch.randn(m, n) * 0.01)
        # Derive shape of the other Kronecker matrix
        p, q = out_features // m, in_features // n
        self.B = nn.Parameter(torch.randn(p, q) * 0.01)

    def forward(self, x):
        # Kronecker product to generate ΔW
        delta_W = torch.kron(self.A, self.B)  # Kronecker product
        W_eff = self.weight + delta_W
        return x @ W_eff.T

# ===== Test =====
x = torch.randn(2, 8)   # Input [batch, in_features]
layer = LoKrLinear(8, 4, m=2, n=2)  # out=4, in=8
out = layer(x)
print("Output shape:", out.shape)
```

Output:

```
Output shape: torch.Size([2, 4])
```

This shows the LoKr layer works correctly.



## Summary

* **LoKr** uses the Kronecker product \$(A \otimes B)\$ to construct low-rank updates.
* This reduces parameter count from \$O(dk)\$ to \$O(mn + pq)\$, while still approximating large matrix updates.
* Suitable for parameter-efficient fine-tuning in large models (e.g., LLMs).



## 🔹 Comparison: LoRA / LoHA / LoKr Fine-tuning Methods

| Method   | Update Formula                   | Extra Parameter Scale                               | Expressive Power                                         | Features                                                                                                 |
| -------- | -------------------------------- | --------------------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **LoRA** | \$\Delta W = BA\$                | \$O(d r + r k)\$                                    | Medium (low-rank linear approx)                          | Classic PEFT, small parameter count, effective and simple                                                |
| **LoHA** | \$\Delta W = (B A) \odot (D C)\$ | \$O(2 (d r + r k))\$                                | Stronger (Hadamard product adds expressivity)            | Maintains low-rank while enhancing non-linear modeling, good for complex tasks                           |
| **LoKr** | \$\Delta W = A \otimes B\$       | \$O(m n + p q)\$ (much smaller than \$d \times k\$) | Strong (Kronecker can represent large matrix structures) | Extremely few parameters but can represent large matrices, ideal for memory optimization in large models |



## Final Takeaways

* **LoRA**: The most basic low-rank approximation, simple and widely applied to LLM fine-tuning.
* **LoHA**: Extends LoRA with Hadamard product, stronger expressivity, suitable for more complex tasks.
* **LoKr**: Uses Kronecker product to approximate large matrix updates with very few parameters, ideal for ultra-large models with memory constraints.



