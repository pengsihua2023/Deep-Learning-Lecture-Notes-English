# Multi-Head Latent Attention (MHLA)

## 1. Definition

**Multi-Head Latent Attention** is an attention mechanism proposed in some efficient Transformer variants (such as **Perceiver, Perceiver IO, Transformer-XL improvements**, etc.). Its core idea is:

* Instead of performing fully-connected self-attention on all tokens in the input sequence (complexity \$O(n^2)\$ ), it introduces a set of **latent variables (latent array)** as an intermediate “bottleneck.”
* The input sequence is projected via attention into a fixed-length latent representation; then the latent representation is used for subsequent cross/self attention processing.
* The multi-head structure allows the latent space to learn features in parallel from different subspaces.

This is like introducing a set of “memory slots” or “query points” to compress the information of the input sequence while preserving expressive power.

## 2. Mathematical Description

Let:

* Input sequence: \$X \in \mathbb{R}^{n \times d\_{\text{in}}}\$
* Latent vectors (trainable parameters, fixed length): \$L \in \mathbb{R}^{m \times d\_{\text{model}}}\$, where \$m \ll n\$

### Step 1: Cross-Attention (Latent Queries, Input Keys/Values)

For each head \$i\$:

$$
Q_i = L W_i^Q, \quad K_i = X W_i^K, \quad V_i = X W_i^V
$$

where \$W\_i^Q \in \mathbb{R}^{d\_{\text{model}} \times d\_k}\$, and similarly for the others.

Attention:

$$
\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right)V_i
$$

### Step 2: Multi-Head Concatenation

$$
H = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

Finally, the updated latent representation is obtained:

$$
L' = H
$$

In the Perceiver architecture, there is also an internal **Latent Self-Attention**, where \$Q,K,V\$ all come from \$L\$, thus repeatedly updating the latents.

---

## 3. Minimal Code Implementation (PyTorch)

Here is a minimal **Multi-Head Latent Attention**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, n_latents, d_input):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.latents = nn.Parameter(torch.randn(n_latents, d_model))  # Trainable latents

        # Projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_input, d_model)
        self.W_v = nn.Linear(d_input, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: (batch, seq_len, d_input)
        """
        batch_size = x.size(0)
        L = self.latents.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, n_latents, d_model)

        Q = self.W_q(L).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.W_o(out)  # (batch, n_latents, d_model)

# Test
x = torch.rand(2, 100, 32)  # batch=2, sequence length=100, input dimension=32
mhla = MultiHeadLatentAttention(d_model=64, num_heads=8, n_latents=16, d_input=32)
y = mhla(x)
print(y.shape)  # (2, 16, 64)
```

Here:

* The input sequence length is 100, with dimension 32;
* 16 latents are used as the intermediate representation;
* The output is batch × number of latents × d\_model.


