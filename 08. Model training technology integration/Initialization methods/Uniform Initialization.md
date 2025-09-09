
# Uniform Initialization
## 📖 Definition

**Uniform Initialization** is a parameter initialization method in deep learning. It initializes neural network weights with values randomly sampled from a uniform distribution interval $\[a, b]\$.
The core idea is to reasonably choose the interval range to avoid gradient vanishing or gradient explosion at the beginning of training.

## 📖 Mathematical Description

If weights \$W\$ are drawn from a uniform distribution:

$$
W \sim U(-a, a)
$$

then each weight satisfies:

$$
P(W = x) = \frac{1}{2a}, \quad x \in [-a, a]
$$

Common definitions of the interval \$a\$ (depending on initialization strategy) include:

1. **Simple Uniform Distribution**:
   Manually specify a constant range, for example:

$$
W \sim U(-0.05, 0.05)
$$

2. **Xavier/Glorot Uniform Initialization** (commonly used for Sigmoid/Tanh activations):

$$
W \sim U\Big(-\sqrt{\frac{6}{n_\text{in}+n_\text{out}}},  \sqrt{\frac{6}{n_\text{in}+n_\text{out}}}\Big)
$$

where \$n\_\text{in}\$ is the input dimension and \$n\_\text{out}\$ is the output dimension.

3. **He/Kaiming Uniform Initialization** (commonly used for ReLU activations):

$$
W \sim U\Big(-\sqrt{\frac{6}{n_\text{in}}},  \sqrt{\frac{6}{n_\text{in}}}\Big)
$$

## 📖 Simplest Code Example

### PyTorch Example

```python
import torch
import torch.nn as nn

# Define a simple linear layer
linear = nn.Linear(3, 2)

# Apply Uniform initialization [-0.1, 0.1]
nn.init.uniform_(linear.weight, a=-0.1, b=0.1)
nn.init.zeros_(linear.bias)

print("Weights:", linear.weight)
print("Bias:", linear.bias)
```

### NumPy Example

```python
import numpy as np

# Input dimension = 3, Output dimension = 2
n_in, n_out = 3, 2

# Xavier Uniform range
limit = np.sqrt(6 / (n_in + n_out))

# Sample from uniform distribution
weights = np.random.uniform(-limit, limit, size=(n_out, n_in))

print("Initialized weights:\n", weights)
```



