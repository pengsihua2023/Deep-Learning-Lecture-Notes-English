

# IA³ Fine-tuning

## 1. Definition

**IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)** is a **Parameter-Efficient Fine-Tuning (PEFT)** method proposed by Liu et al. (2022).

Its core idea:

* Do not update the original pre-trained model parameters \$\theta\$.
* Introduce **trainable scalar vectors** in the **Transformer attention and feed-forward layers** to **scale (inhibit/amplify)** activations.
* This requires training only a small number of parameters, yet efficiently adapts to downstream tasks.

👉 Simply put: IA³ adds **per-channel scaling factors** to the attention and value projections of each layer, like knobs that adjust signal strength.

---

## 2. Mathematical Description

### 2.1 Attention in Transformer

Standard **Scaled Dot-Product Attention**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

Where:

* \$Q = XW\_Q\$
* \$K = XW\_K\$
* \$V = XW\_V\$

### 2.2 IA³ Modified Attention

IA³ inserts **per-channel scaling vectors** \$l\_k, l\_v, l\_{ff}\$ into the attention and feed-forward layers:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q (K \odot l_k)^T}{\sqrt{d_k}}\right)(V \odot l_v)
$$

$$
\text{FFN}(X) = (X W_1 \odot l_{ff}) W_2
$$

Where:

* \$\odot\$ denotes element-wise multiplication (with broadcasting).
* \$l\_k, l\_v, l\_{ff}\$ are **trainable parameter vectors**, with dimensions matching \$K\$, \$V\$, and the FFN hidden layer, respectively.

### 2.3 Loss Function

Only these scaling parameters are trained:

$$
\mathcal{L} = \mathcal{L}_{task}(f(x; \theta, l_k, l_v, l_{ff}))
$$

$\theta\$ is fixed (frozen), and only \$l\_k, l\_v, l\_{ff}\$ are updated.

---

## 3. Simple Code Example (PyTorch)

Here’s a **simplified IA³ attention layer** in PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class IA3Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(IA3Attention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        # Original projection layers (frozen)
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)

        for p in self.parameters():
            p.requires_grad = False

        # IA³ trainable scaling parameters
        self.l_k = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.l_v = nn.Parameter(torch.ones(1, 1, embed_dim))

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x) * self.l_k  # Scale Keys
        V = self.W_V(x) * self.l_v  # Scale Values

        # Multi-head split
        Q = Q.view(x.size(0), -1, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(x.size(0), -1, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(x.size(0), -1, self.num_heads, self.head_dim).transpose(1,2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2,-1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1,2).reshape(x.size(0), -1, self.embed_dim)
        return out

# Test
x = torch.rand(2, 5, 16)  # batch=2, seq_len=5, embed_dim=16
attn = IA3Attention(embed_dim=16, num_heads=2)
out = attn(x)
print("Output shape:", out.shape)  # (2, 5, 16)
```

In real Transformer layers, similar scaling parameters \$l\_{ff}\$ are also added in the FFN (feed-forward network).

---

## 4. Summary

* **Definition**: IA³ is a parameter-efficient fine-tuning method that introduces scaling vectors into attention and feed-forward layers to adjust activations.
* **Formulas**:

$$
\text{Attn}(Q,K,V) = \text{softmax}\Big(\frac{Q (K \odot l_k)^T}{\sqrt{d_k}}\Big)(V \odot l_v)
$$

$$
\text{FFN}(X) = (X W_1 \odot l_{ff}) W_2
$$

* **Features**:

  * Trains only a few parameters (scaling vectors).
  * Does not modify pre-trained weights, with minimal memory/storage overhead.
  * Performance close to full fine-tuning, especially suitable for large model adaptation.
* **Code**: Only requires adding trainable scaling vectors in attention and FFN.

---

# 📊 Comparison: LoRA vs Diff Pruning vs IA³

| Method                                                                      | Definition                                                                                             | Formula                                                                                                                                              | Trainable Parameters                                             | Advantages                                                                         | Disadvantages                                                 | Typical Applications                                 |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------- |
| **LoRA**<br>(Low-Rank Adaptation)                                           | Decomposes weight updates into low-rank matrices, training only those                                  | \$\theta' = \theta + BA\$, where \$B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}\$, \$r \ll \min(d,k)\$                               | Low-rank matrix params \$\mathcal{O}(r(d+k))\$                   | Efficient, high parameter savings, plug-and-play, supports merging into base model | Must choose suitable rank, too low may harm performance       | LLM downstream adaptation (e.g., GPT, BERT)          |
| **Diff Pruning**                                                            | Adds diff updates \$\Delta \theta\$ with L1 regularization for sparsity, updating only necessary parts | \$\theta' = \theta + \Delta \theta\$, \$\mathcal{L} = \mathcal{L}\_{task} + \lambda \|\Delta \theta\|\_1\$                                           | Same dimension as original params, but only sparse part retained | Flexible, automatically selects “important params”, interpretable                  | Requires sparsity constraints, may cause training instability | Small models, model compression                      |
| **IA³**<br>(Infused Adapter by Inhibiting and Amplifying Inner Activations) | Introduces trainable scaling vectors in attention and FFN to amplify/inhibit activations               | \$\text{Attn}(Q,K,V)=\text{softmax}\Big(\frac{Q (K \odot l\_k)^T}{\sqrt{d\_k}}\Big)(V \odot l\_v)\$<br>\$\text{FFN}(X)=(X W\_1 \odot l\_{ff}) W\_2\$ | Only 2–3 vectors per layer, far fewer than weight matrices       | Extremely lightweight, minimal storage, simple to implement                        | Limited to scaling, less expressive                           | Large model quick fine-tuning, low-resource settings |

---

## 🔑 Final Takeaways

* **LoRA**: Best for **large model deployment**, saves many parameters via low-rank updates, most widely used.
* **Diff Pruning**: Best when **sparsity & interpretability** are needed, automatically identifies critical parameters.
* **IA³**: Most lightweight, only scaling vectors, ideal for **memory-constrained** or **fast transfer learning** scenarios.



