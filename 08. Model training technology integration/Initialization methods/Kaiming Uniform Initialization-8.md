
# Kaiming Uniform Initialization
## 📖 Definition

**Kaiming Uniform Initialization** is the **uniform distribution version** of He Initialization, used for ReLU and its variants (such as ReLU, LeakyReLU).
Its goal is to **keep the output variance of each layer stable during forward propagation**, avoiding gradient vanishing or explosion.

Kaiming Initialization was proposed by Kaiming He et al. in the 2015 paper *Delving Deep into Rectifiers*.



## 📖 Mathematical Description

Suppose the input dimension of a layer is \$n\_\text{in}\$ (i.e., \$fan\_\text{in}\$, the number of input neurons), then the weights \$W\$ are drawn from the interval:

$$
W \sim U\left(-\text{bound}, \text{bound}\right)
$$

where:

$$
\text{bound} = \sqrt{\frac{6}{n_\text{in} \cdot (1 + a^2)}}
$$

* \$a\$ is the **negative slope** of ReLU (for standard ReLU, \$a = 0\$; for Leaky ReLU, \$a\$ is the leakage coefficient).
* When \$a = 0\$ (standard ReLU), the formula simplifies to:

$$
W \sim U\left(-\sqrt{\frac{6}{n_\text{in}}},  \sqrt{\frac{6}{n_\text{in}}}\right)
$$

This range is larger than Xavier Uniform, because ReLU discards half of the inputs (negative values become 0), so the variance of the weights needs to be larger.

## 📖 Simplest Code Example

### PyTorch Implementation

```python
import torch
import torch.nn as nn

# Define a linear layer (input=3, output=2)
linear = nn.Linear(3, 2)

# Apply Kaiming Uniform initialization (corresponds to He Uniform)
nn.init.kaiming_uniform_(linear.weight, a=0, nonlinearity='relu')
nn.init.zeros_(linear.bias)

print("Kaiming Uniform weights:\n", linear.weight)
print("Bias:\n", linear.bias)
```

### NumPy Implementation

```python
import numpy as np

n_in, n_out = 3, 2
a = 0  # Negative slope of ReLU

bound = np.sqrt(6 / (n_in * (1 + a**2)))
weights = np.random.uniform(-bound, bound, size=(n_out, n_in))

print("Kaiming Uniform initialized weights:\n", weights)
```


