
# LSUV Initialization (Layer-Sequential Unit-Variance Initialization)

## Definition

**LSUV (Layer-Sequential Unit-Variance Initialization)** is a neural network weight initialization method proposed by Mishkin and Matas in 2015 (paper *All you need is a good init*).

Its core idea is:

1. First initialize weights using an existing method (usually **Orthogonal Initialization**);
2. Then perform forward propagation layer by layer (Layer-Sequential) with a mini-batch of data;
3. Adjust the variance of each layer’s output so that it approaches **unit variance (variance = 1)**;
4. If necessary, further adjust the mean to be close to 0.

This ensures that at the start of training, the output distribution of all layers is relatively stable, avoiding gradient vanishing or explosion.

---

## Mathematical Description

Suppose the output of a layer is:

$$
y = W x + b
$$

where \$x\$ is the input, and \$W\$ is the weight matrix.

Steps of LSUV:

1. **Initial Weights**
   Use orthogonal initialization:

$$
W_0 = \text{orthogonal}(shape)
$$

2. **Forward Propagation**
   Compute outputs with a mini-batch of data:

$$
y = f(Wx + b)
$$

3. **Adjust Variance**
   Compute the current output variance:

$$
\sigma^2 = \text{Var}(y)
$$

Update the weights:

$$
W \leftarrow \frac{W}{\sqrt{\sigma^2}}
$$

4. **Repeat** until the output variance is close to 1 (tolerance \$\epsilon\$, e.g., \$|\sigma^2 - 1| < 0.1\$).

---

## Simplest Code Example

### PyTorch Pseudo Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# A simple two-layer network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

def lsuv_init(layer, data, tol=0.1, max_iter=10):
    # Orthogonal initialization
    nn.init.orthogonal_(layer.weight)
    
    for _ in range(max_iter):
        # Forward propagation
        with torch.no_grad():
            output = layer(data)
        
        var = output.var().item()
        if abs(var - 1.0) < tol:
            break
        
        # Adjust weights
        layer.weight.data /= torch.sqrt(torch.tensor(var))

# Demonstration with random data
net = SimpleNet()
x = torch.randn(16, 100)  # batch_size=16

# Apply LSUV to the first layer
lsuv_init(net.fc1, x)
print("After fc1 weight initialization, output variance:", net.fc1(x).var().item())
```

---

### NumPy Simplified Version (Single Layer Demonstration)

```python
import numpy as np

def lsuv_init(weights, x, tol=0.1, max_iter=10):
    # Orthogonal initialization
    u, _, v = np.linalg.svd(weights, full_matrices=False)
    weights = u if u.shape == weights.shape else v
    
    for _ in range(max_iter):
        y = x.dot(weights.T)
        var = np.var(y)
        if abs(var - 1.0) < tol:
            break
        weights /= np.sqrt(var)
    return weights

# Input data (batch=16, dim=100)
x = np.random.randn(16, 100)
weights = np.random.randn(50, 100)  # output dimension = 50

weights_lsuv = lsuv_init(weights, x)
print("Output variance after LSUV:", np.var(x.dot(weights_lsuv.T)))
```

---

✅ Summary:

* **LSUV Initialization** = Orthogonal initialization + sequential adjustment of variance to 1;
* Mathematically, this ensures \$\text{Var}(y) \approx 1\$ by scaling the weights;
* Suitable for deep networks, significantly improving convergence speed.



