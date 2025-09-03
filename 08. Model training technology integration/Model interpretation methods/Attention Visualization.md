
# Attention Visualization

## 1. Definition

**Attention Visualization** is a **model interpretability method**. By visualizing the **attention weights** in deep learning models, it helps us understand which parts of the input the model “focuses on” during prediction.

* In **NLP (Natural Language Processing)**: shows the attention relationships between words. For example, in machine translation, it reveals which context words are relied upon when translating a specific word.
* In **CV (Computer Vision)**: shows the attention distribution over image regions. For example, a Vision Transformer might focus mainly on a cat’s face when recognizing the category “cat”.

---

## 2. Mathematical Description

The formula for calculating attention weights comes from **Scaled Dot-Product Attention**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where:

* \$Q\$ = Queries
* \$K\$ = Keys
* \$V\$ = Values
* \$\frac{QK^T}{\sqrt{d\_k}}\$ = similarity matrix (relevance scores)
* \$\text{softmax}(\cdot)\$ = converts scores into attention weights, ranging in $\[0,1]\$ and summing to 1

👉 **Core of visualization**: display the attention weight matrix

$$
A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$

as a **heatmap / arrow plot / overlay**, directly illustrating the model’s focus patterns.

---

## 3. Code Examples

### 3.1 Attention Visualization in NLP (Heatmap)

```python
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

# Define Multi-Head Attention
mha = nn.MultiheadAttention(embed_dim=16, num_heads=2, batch_first=True)

# Input (batch=1, seq_len=5, d_model=16)
x = torch.rand(1, 5, 16)

# Self-attention (Q=K=V=x)
out, attn_weights = mha(x, x, x)

print("Attention weights shape:", attn_weights.shape)  # [1, num_heads, seq_len, seq_len]

# Visualize the attention of the first head
sns.heatmap(attn_weights[0, 0].detach().numpy(), cmap="viridis", annot=True)
plt.xlabel("Key positions")
plt.ylabel("Query positions")
plt.title("Attention Heatmap (Head 1)")
plt.show()
```

👉 Running this produces a **heatmap**, where the horizontal axis represents **Keys (words being attended to)**, the vertical axis represents **Queries (current words)**, and the color intensity represents the magnitude of the attention weights.

---

### 3.2 Attention Visualization in CV (Overlay)

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Simulated attention weights (assume the image is split into 4x4 patches)
attn_weights = torch.rand(1, 1, 16, 16)  # [batch, head, patch_num, patch_num]
attn_map = attn_weights[0, 0].mean(0).reshape(4,4).detach().numpy()

# Original image (randomly simulate a 64x64 image)
img = np.random.rand(64,64)

# Visualization
plt.imshow(img, cmap="gray")
plt.imshow(attn_map, cmap="jet", alpha=0.5, extent=(0,64,64,0))  # Overlay attention
plt.title("Attention Overlay on Image")
plt.colorbar()
plt.show()
```

👉 Running this produces an image with an **attention heatmap overlay**, showing which regions the model mainly focuses on during image classification.

---

## 4. Summary

* **Definition of Attention Visualization**: explain where the model “looks” during prediction by displaying attention weights.
* **Mathematical Description**: based on

$$
A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$

* **Code Examples**:

  * NLP → use a heatmap to show attention relationships between words.
  * CV → overlay a heatmap on the image to show the model’s focus regions.


