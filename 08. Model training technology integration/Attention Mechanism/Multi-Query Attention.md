# Multi-Query Attention (MQA)

It is a variant of Multi-Head Attention (MHA), designed to improve efficiency.

## 1. Definition

In standard **Multi-Head Attention (MHA)**, each attention head has its own independent \$Q, K, V\$ projection matrices, which makes computation and storage expensive.

The core improvement of **Multi-Query Attention (MQA)** is:

* Each attention head **has its own independent Query projection**;
* But **shares the same Key and Value projections**.

Thus:

* The cost and storage for computing \$K, V\$ are reduced from \$O(h \cdot d)\$ to \$O(d)\$;
* Attention heads can still capture different features through independent queries \$Q\_i\$;
* Especially suitable for saving memory and reducing latency during large model inference (used in PaLM, LLaMA, etc.).
<div align="center">
<img width="660" height="260" alt="image" src="https://github.com/user-attachments/assets/27daf7ab-99a9-4cf7-853c-682ac6baf531" />
</div>

## 2. Mathematical Description

Let:

* Input sequence \$X \in \mathbb{R}^{n \times d\_{\text{model}}}\$
* Number of heads \$h\$, with per-head dimension \$d\_k = d\_{\text{model}} / h\$

1. **Projection**

$$
Q_i = X W_i^Q \quad (i=1,\dots,h), \quad K = X W^K, \quad V = X W^V
$$

where:

* \$W\_i^Q \in \mathbb{R}^{d\_{\text{model}} \times d\_k}\$ (independent for each head)
* \$W^K, W^V \in \mathbb{R}^{d\_{\text{model}} \times d\_k}\$ (shared across all heads)

2. **Attention Computation**
   For each head:

$$
\text{head}_i = \text{softmax}\left(\frac{Q_i K^\top}{\sqrt{d_k}}\right)V
$$

3. **Concatenation Output**

$$
\text{MQA}(X) = \text{Concat}(\text{head}_1,\dots,\text{head}_h) W^O
$$

where \$W^O \in \mathbb{R}^{hd\_k \times d\_{\text{model}}}\$.

## 3. Minimal PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Independent Q for each head
        self.W_q = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        # Shared K and V
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_k)
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Shared K, V
        K = self.W_k(x)  # (batch, seq_len, d_k)
        V = self.W_v(x)  # (batch, seq_len, d_k)

        heads = []
        for Wq in self.W_q:
            Q = Wq(x)  # (batch, seq_len, d_k)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
            attn = F.softmax(scores, dim=-1)
            heads.append(torch.matmul(attn, V))  # (batch, seq_len, d_k)

        # Concatenate all heads
        out = torch.cat(heads, dim=-1)  # (batch, seq_len, d_model)
        return self.W_o(out)

# Test
x = torch.rand(2, 5, 16)  # batch=2, seq_len=5, d_model=16
mqa = MultiQueryAttention(d_model=16, num_heads=4)
y = mqa(x)
print(y.shape)  # (2, 5, 16)
```

👉 summary:

* **MHA**: each head has independent \$Q, K, V\$
* **MQA**: each head has independent \$Q\$, but shares \$K, V\$, saving memory and computation

