
# SparseAdam Optimizer

## 1. Definition

**SparseAdam** is a variant of **Adam** specifically designed for scenarios with **sparse gradients**, with a typical application being the **embedding layer** (e.g., training word embeddings in NLP).

Standard Adam maintains first- and second-moment estimates (`m, v`) for all parameters, even if some parameters do not receive updates in a given iteration. This leads to unnecessary overhead. SparseAdam, however:

* **Only updates the parameters and moment estimates corresponding to non-zero gradients**, avoiding meaningless updates;
* For parameters that are not updated, their moment estimates remain unchanged (they do not decay to zero).

Therefore, in large-scale sparse gradient scenarios (such as vocabularies with millions of tokens), SparseAdam can significantly improve efficiency.

---

## 2. Mathematical Formulation

Let:

* Parameters: \$\theta\_t\$
* Gradients: \$g\_t\$ (sparse, i.e., most elements are 0)
* First moment (momentum): \$m\_t\$
* Second moment: \$v\_t\$
* Learning rate: \$\alpha\$
* Decay factors: \$\beta\_1, \beta\_2 \in \[0,1)\$
* Numerical stability term: \$\epsilon\$

### Update steps:

1. Gradient computation (sparse):

$$
g_t = \nabla_\theta f_t(\theta_{t-1}) \quad (\text{only non-zero elements})
$$

2. First and second moment updates (only non-zero indices):

$$
m_t[i] = \beta_1 m_{t-1}[i] + (1-\beta_1) g_t[i]
$$

$$
v_t[i] = \beta_2 v_{t-1}[i] + (1-\beta_2) g_t[i]^2
$$

3. Bias correction:

$$
\hat{m}_t[i] = \frac{m_t[i]}{1-\beta_1^t}, \quad 
\hat{v}_t[i] = \frac{v_t[i]}{1-\beta_2^t}
$$

4. Parameter update:

$$
\theta_t[i] = \theta_{t-1}[i] - \alpha \cdot \frac{\hat{m}_t[i]}{\sqrt{\hat{v}_t[i]} + \epsilon}
$$

Here \$i\$ indicates that only the parameters corresponding to **non-zero gradients** are updated.

---

## 3. Minimal Code Example

Using **PyTorch’s SparseAdam** on an embedding layer:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simulate an embedding layer (vocab size=1000, dimension=50)
embedding = nn.Embedding(1000, 50, sparse=True)

# Randomly generate some word IDs (batch_size=4, seq_len=5)
input_ids = torch.randint(0, 1000, (4, 5))

# Forward pass (obtain embedding vectors)
embeddings = embedding(input_ids)

# Define a simple objective (minimize the L2 norm of embeddings)
loss = embeddings.pow(2).sum()

# Optimizer: SparseAdam
optimizer = optim.SparseAdam(embedding.parameters(), lr=0.01)

# Backpropagation & update
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Updated embedding vectors (partial):")
print(embedding.weight[input_ids[0]])
```

---

### Explanation

1. **Embedding layer** is set with `sparse=True`, so its gradient is sparse.
2. **SparseAdam** only updates embeddings corresponding to tokens that appear in the current batch.
3. Embeddings for tokens not seen in this batch remain unchanged, making the training more efficient.

---

## Adam vs SparseAdam Comparison

| Feature             | **Adam**                                                                                       | **SparseAdam**                                                                                                      |
| ------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Use case**        | General-purpose optimizer, suitable for all parameters (dense gradients).                      | Specifically designed for **sparse gradients** (e.g., `nn.Embedding(sparse=True)`).                                 |
| **Update method**   | Updates **all parameters** with first- and second-moment estimates, even if the gradient is 0. | Only updates **parameters with nonzero gradients** and their momentum terms; untouched parameters remain unchanged. |
| **Memory overhead** | Must maintain momentum states for all parameters.                                              | Maintains momentum states only for nonzero indices, making it more memory- and compute-efficient.                   |
| **Convergence**     | May be inefficient and slow in sparse scenarios.                                               | Faster in sparse scenarios, significantly reducing useless updates.                                                 |
| **Typical use**     | CNNs, RNNs, Transformers, and other general tasks.                                             | Word embeddings, large-scale NLP vocabularies (hundreds of thousands or even millions of tokens).                   |
| **PyTorch usage**   | `optim.Adam(model.parameters(), lr=...)`                                                       | `optim.SparseAdam(embedding.parameters(), lr=...)` — supported only for sparse parameters.                          |

---

👉 **Summary**:

* If model parameters are **dense** (e.g., convolution layers, fully connected layers), use **Adam**.
* If model parameters are **sparse** (especially embedding layers), use **SparseAdam** for higher efficiency.




