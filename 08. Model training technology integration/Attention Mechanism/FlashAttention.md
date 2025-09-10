# FlashAttention

It is not a new attention mechanism (unlike MHA/MQA/LHA), but an **efficient attention computation method** that solves the computation and memory bottleneck of Transformer on long sequences.

## 1. Definition

**FlashAttention** is a **memory-optimized, I/O-aware attention algorithm**. The core idea is:

* Standard attention first computes the full \$QK^T\$ matrix (size \$n \times n\$), then applies softmax, then multiplies by \$V\$, which requires **\$O(n^2)\$ memory**.
* FlashAttention **does not explicitly store the entire attention matrix**, but instead:

  * Splits the input sequence into blocks (tiling);
  * Computes softmax incrementally within each block (ensuring numerical stability);
  * Normalizes while computing, multiplies with \$V\$ on the fly, and writes back the result directly without saving the large intermediate matrix.

Thus:

* Time complexity remains \$O(n^2)\$, but **memory complexity is reduced from \$O(n^2)\$ to \$O(n)\$**;
* It enables GPUs to handle much longer sequences (thousands to tens of thousands of tokens).
<div align="center">
  <img width="420" height="463" alt="image" src="https://github.com/user-attachments/assets/df33e347-ff65-47a4-827d-9e6a432bfed9" />
</div>

## 2. Mathematical Description

Standard attention:

$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

Improvements of FlashAttention:

1. Do not directly form the entire matrix \$S = QK^\top / \sqrt{d\_k}\$, but split it into blocks:

$$
S_{i,j} = \frac{Q_i K_j^\top}{\sqrt{d_k}}
$$

Iteratively compute by blocks of \$i\$ and \$j\$.

2. Update softmax block by block:

$$
\text{softmax}(S_{i,:}) = \frac{\exp(S_{i,j} - m_i)}{\sum_j \exp(S_{i,j} - m_i)}
$$

where \$m\_i\$ is the maximum value of the row (for numerical stability).

3. While computing softmax, immediately multiply by \$V\$ and accumulate:

$$
O_i = \sum_j \text{softmax}(S_{i,j}) V_j
$$

Therefore, **the matrix \$S\$ never needs to be stored**, only normalization factors and the current block output are kept.

## 3. Minimal Code Implementation

Here is the official PyTorch FlashAttention API usage example. Since PyTorch 2.0, `torch.nn.functional.scaled_dot_product_attention` already supports **FlashAttention kernel**, and will automatically call the optimized implementation on GPU.

### 1. Basic Usage

```python
import torch
import torch.nn.functional as F

# Simulate input
batch, n, d, h = 2, 128, 64, 8   # batch=2, sequence length=128, d_model=64, 8 heads
d_k = d // h

Q = torch.randn(batch, h, n, d_k, device="cuda")
K = torch.randn(batch, h, n, d_k, device="cuda")
V = torch.randn(batch, h, n, d_k, device="cuda")

# Call PyTorch FlashAttention API
out = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False)

print(out.shape)  # (batch, h, n, d_k)
```

### 2. Parameter Description

* `Q, K, V`: input tensors, shape `(batch, heads, seq_len, d_k)`
* `attn_mask`: optional, supports padding mask or causal mask
* `dropout_p`: attention dropout probability
* `is_causal`: if `True`, enables **autoregressive causal mask** (only looks at past tokens)
* Return value: same shape as `Q`, i.e. `(batch, heads, seq_len, d_k)`

### 3. Combining with Multi-Head Attention

You can directly replace the original attention implementation inside `nn.Module`:

```python
class FlashMHA(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        # Linear transformations
        Q = self.W_q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # FlashAttention
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.W_o(out)

# Test
x = torch.randn(2, 128, 64, device="cuda")
mha = FlashMHA(64, 8).cuda()
y = mha(x)
print(y.shape)  # (2, 128, 64)
```

✅ Summary:

* Simply using **`scaled_dot_product_attention`** will automatically invoke **FlashAttention** on CUDA (if conditions are met, e.g., sequence length is large enough).
* For users, the change is almost “zero-modification”—just replace the line `softmax(QK^T)V`.


