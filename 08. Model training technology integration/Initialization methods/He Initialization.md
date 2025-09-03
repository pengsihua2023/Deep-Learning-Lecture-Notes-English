
# He Initialization (also called Kaiming Initialization)

## Definition

**He Initialization** is a parameter initialization method specifically designed for **ReLU and its variants** (such as ReLU, LeakyReLU, ELU, etc.).
Its goals are:

* To keep the output variance stable across different layers during forward propagation;
* To reduce vanishing gradient or exploding gradient problems.

He initialization was proposed by Kaiming He et al. in 2015 (*Delving Deep into Rectifiers*).

---

## Mathematical Description

Suppose the input dimension of a layer is \$n\_\text{in}\$ (i.e., the number of input neurons), then He initialization has two versions: **Uniform** and **Normal**.

1. **He Normal (Normal distribution version)**

$$
W \sim \mathcal{N}\left(0, \frac{2}{n_\text{in}}\right)
$$

2. **He Uniform (Uniform distribution version)**

$$
W \sim U\left(-\sqrt{\frac{6}{n_\text{in}}},  \sqrt{\frac{6}{n_\text{in}}}\right)
$$

Where:

* \$\mathcal{N}(0, \sigma^2)\$ denotes a normal distribution with mean 0 and variance \$\sigma^2\$;
* \$U(-a, a)\$ denotes a uniform distribution over the interval $\[-a, a]\$.

> Core idea:
> Since ReLU cuts half of the inputs to 0, the **weight variance needs to be scaled up**, using \$\frac{2}{n\_\text{in}}\$ instead of \$\frac{1}{n\_\text{in}}\$ as in Xavier initialization.

---

## Simplest Code Examples

### PyTorch Example

```python
import torch
import torch.nn as nn

# Define a linear layer (input=3, output=2)
linear = nn.Linear(3, 2)

# He Uniform initialization
nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
nn.init.zeros_(linear.bias)

print("He Uniform weights:\n", linear.weight)
print("Bias:\n", linear.bias)

# He Normal initialization (if you want to use normal distribution)
linear2 = nn.Linear(3, 2)
nn.init.kaiming_normal_(linear2.weight, nonlinearity='relu')
print("He Normal weights:\n", linear2.weight)
```

### NumPy Example

```python
import numpy as np

n_in, n_out = 3, 2

# He Uniform
limit = np.sqrt(6 / n_in)
weights_uniform = np.random.uniform(-limit, limit, size=(n_out, n_in))

# He Normal
std = np.sqrt(2 / n_in)
weights_normal = np.random.normal(0, std, size=(n_out, n_in))

print("He Uniform initialization:\n", weights_uniform)
print("He Normal initialization:\n", weights_normal)
```


