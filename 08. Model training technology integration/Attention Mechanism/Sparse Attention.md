# Sparse Attention

It is an optimization of standard fully-connected attention, mainly aimed at **reducing computational complexity**.

## 1. Definition

In standard **self-attention**, each token interacts with all \$n\$ tokens, so both computation and memory cost are \$O(n^2)\$.

The core idea of **Sparse Attention** is:

* Restrict each token to interact with only a subset of tokens, instead of all;
* By designing specific sparse patterns (such as local window, stride, block-sparse, causal mask, etc.), reduce complexity while maintaining expressive power.

Typical applications:

* Transformer-XL, Longformer, BigBird, Sparse Transformer all use sparse attention to handle long sequences.
<div align="center">
<img width="850" height="514" alt="image" src="https://github.com/user-attachments/assets/c2613226-db99-4b71-a008-4abe76462a6a" />
</div>

## 2. Mathematical Description

Standard attention:

$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M \right)V
$$

where \$M\$ is the mask matrix. If it is fully-connected attention, \$M=0\$.

In sparse attention, a **sparse mask matrix** \$M \in {-\infty, 0}^{n \times n}\$ is defined:

* \$M\_{ij} = 0\$: position \$i\$ can attend to position \$j\$;
* \$M\_{ij} = -\infty\$: attention is prohibited.

Thus, sparse attention simply adds a mask before softmax:

$$
\alpha_{ij} = \frac{\exp\left(\frac{Q_i K_j^\top}{\sqrt{d_k}} + M_{ij}\right)}{\sum_{j'} \exp\left(\frac{Q_i K_{j'}^\top}{\sqrt{d_k}} + M_{ij'}\right)}
$$

Output:

$$
O_i = \sum_{j} \alpha_{ij} V_j
$$

## 3. Minimal Code Implementation (PyTorch)

Below is a minimal implementation of **local window sparse attention**:

```python
import torch
import torch.nn.functional as F

def sparse_attention(Q, K, V, window_size=4):
    """
    Simplified sparse attention (local window)
    Q, K, V: (batch, n, d)
    """
    batch, n, d = Q.shape
    scale = 1.0 / (d ** 0.5)

    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (batch, n, n)

    # Build sparse mask (each token only attends to previous and next window_size tokens)
    mask = torch.full((n, n), float("-inf"))
    for i in range(n):
        start, end = max(0, i - window_size), min(n, i + window_size + 1)
        mask[i, start:end] = 0.0
    scores = scores + mask  # (batch, n, n)

    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)  # (batch, n, d)
    return out

# Test
batch, n, d = 2, 10, 16
Q = torch.randn(batch, n, d)
K = torch.randn(batch, n, d)
V = torch.randn(batch, n, d)

out = sparse_attention(Q, K, V, window_size=2)
print(out.shape)  # (2, 10, 16)
```

## 4. Summary

* **Fully-connected attention**: complexity \$O(n^2)\$
* **Sparse attention**: by restricting interactions via sparse masks, complexity can be reduced to \$O(n \log n)\$ or even \$O(n)\$. Common patterns:

  * **Local window** (e.g., Longformer)
  * **Stride** (attend every \$k\$-th token)
  * **Block-sparse** (e.g., BigBird)
  * **Random sparse**


