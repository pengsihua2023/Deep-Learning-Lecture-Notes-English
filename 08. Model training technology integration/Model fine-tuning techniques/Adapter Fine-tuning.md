# Adapter Fine-tuning

## 📖 1. Definition

**Adapter fine-tuning** is an efficient parameter tuning method, commonly used for large-scale pre-trained models (such as BERT, GPT, etc.).
Core idea:

* **Freeze the parameters of the original pre-trained model** to avoid large-scale updates;
* **Insert a small bottleneck structure (Adapter module) into each Transformer layer**, and only train these Adapter parameters;
* Adapter is generally: **Down-projection → Activation → Up-projection**, that is:

  * Project the input dimension \$d\$ down to a small bottleneck dimension \$r\$;
  * Pass through a nonlinear activation (e.g., ReLU / GELU);
  * Project back to the original dimension \$d\$, with a residual connection.

This greatly reduces the number of parameters that need to be trained, while maintaining model performance.

## 📖 2. Mathematical Formula

Let:

* Transformer hidden state vector: \$h \in \mathbb{R}^d\$
* Adapter down-projection matrix: \$W\_{down} \in \mathbb{R}^{r \times d}\$
* Adapter up-projection matrix: \$W\_{up} \in \mathbb{R}^{d \times r}\$
* Activation function: \$\sigma(\cdot)\$

**Adapter forward computation**:

$$
h' = h + W_{up} \, \sigma(W_{down} h)
$$

Where:

* \$W\_{down}\$: compresses the \$d\$-dimensional vector into lower dimension \$r\$;
* \$\sigma\$: nonlinear mapping (e.g., ReLU);
* \$W\_{up}\$: maps back to \$d\$-dimension;
* Finally, residual connection adds back \$h\$.

During training **only \$W\_{down}, W\_{up}\$ are updated**, while the rest of the model parameters remain frozen.

## 📖 3. Minimal Code Example

Write a minimal Adapter layer in **PyTorch** and insert it into a model:

```python
import torch
import torch.nn as nn

# ===== Adapter Module =====
class Adapter(nn.Module):
    def __init__(self, d_model, r=16):
        super().__init__()
        self.down = nn.Linear(d_model, r, bias=False)
        self.act = nn.ReLU()
        self.up = nn.Linear(r, d_model, bias=False)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))  # Residual connection

# ===== Using Adapter in Transformer Layer =====
class ToyTransformerLayer(nn.Module):
    def __init__(self, d_model=128, adapter_r=16):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=4)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        self.adapter = Adapter(d_model, r=adapter_r)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out
        x = self.ffn(x)
        x = self.adapter(x)  # Insert Adapter
        return x

# ===== Simple Test =====
x = torch.randn(10, 32, 128)  # [seq_len, batch_size, hidden_dim]
layer = ToyTransformerLayer()
out = layer(x)
print("Output shape:", out.shape)
```

## 📖 Explanation

1. **Adapter**: Two fully connected layers form a down–up bottleneck.
2. **Residual connection**: Ensures the original model structure is not broken.
3. **Training**: In practice, freeze all pre-trained parameters and only train the Adapter layer parameters.


