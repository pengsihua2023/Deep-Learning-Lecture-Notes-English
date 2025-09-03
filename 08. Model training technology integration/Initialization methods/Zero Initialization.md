
# Zero Initialization 
## Definition

**Zero Initialization** means initializing all **weights and biases of a neural network to 0**.

It is the most naive initialization method, but it is almost never used in deep learning (at least not for weights), because it leads to the **neuron symmetry problem**:

* Different neurons produce exactly the same output at the start of training;
* During backpropagation, their gradients are also identical;
* This prevents them from learning different features, causing training to fail.

However, **bias initialized to 0** is a common practice, since biases do not cause symmetry issues.

## Mathematical Description

Suppose a layer’s parameters are the weight matrix \$W\$ and bias vector \$b\$, then zero initialization is defined as:

$$
W_{ij} = 0, \quad b_i = 0
$$

For any input \$x\$, the forward propagation result is:

$$
y = f(Wx + b) = f(0) = f(\mathbf{0})
$$

If \$f\$ is ReLU / Sigmoid / Tanh, then all neurons output the same value, and learning cannot proceed.

## Simplest Code Example

### PyTorch Example

```python
import torch
import torch.nn as nn

# Define a linear layer (input=3, output=2)
linear = nn.Linear(3, 2)

# Apply Zero Initialization
nn.init.zeros_(linear.weight)
nn.init.zeros_(linear.bias)

print("Weights:\n", linear.weight)
print("Bias:\n", linear.bias)
```

### NumPy Example

```python
import numpy as np

# Input dimension=3, Output dimension=2
weights = np.zeros((2, 3))
bias = np.zeros(2)

# Input data
x = np.random.randn(4, 3)

# Forward propagation
y = x.dot(weights.T) + bias
print("Output:\n", y)
```

## Summary

* **Zero-initialized weights**: Not advisable (causes all neurons to learn the same thing).
* **Zero-initialized biases**: Common and reasonable.
* Modern networks usually adopt initialization methods such as Xavier, He, or LSUV to address gradient vanishing/explosion problems.


