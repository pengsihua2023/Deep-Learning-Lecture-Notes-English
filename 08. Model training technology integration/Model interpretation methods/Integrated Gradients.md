
# Integrated Gradients (IG) Model Interpretation Method

## 📖 1. Definition

**Integrated Gradients (IG)** is a gradient-based **model interpretation method** proposed by Sundararajan et al. in 2017.
Its core idea is:

> Gradually vary the input features from a **reference input (baseline)** (e.g., an all-zero vector, mean input, etc.) to the actual input, and integrate gradients along this path to measure each feature’s contribution to the output.

This solves two problems of **ordinary gradient methods**:

* Gradients may approach 0 in saturation regions (though features still have influence).
* Single-point gradients may be unstable.



## 📖 2. Mathematical Description

Let:

* Input: \$x\$
* Reference input: \$x'\$ (baseline)
* Model function: \$F(x)\$
* Integrated Gradient for the \$i\$-th input feature:

<img width="432" height="70" alt="image" src="https://github.com/user-attachments/assets/71740f5a-5633-46c0-9a23-d3f5bc9ce906" />


Explanation:

* Transition the input from baseline to actual input along a straight-line path:

$$
x(\alpha) = x' + \alpha(x - x')
$$

* Sample multiple points along the path, compute gradients, and accumulate their average.
* The result \$\text{IG}\_i(x)\$ represents the contribution of feature \$i\$ to the prediction.

---

## 📖 3. Simple Code Example (PyTorch + Captum)

We use the `captum` library to implement **Integrated Gradients**.

```python
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

# 1. Define a simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SimpleNet()

# 2. Construct input
inputs = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
baseline = torch.zeros_like(inputs)  # baseline=0

# 3. Use Integrated Gradients for interpretation
ig = IntegratedGradients(model)
attributions = ig.attribute(inputs, baselines=baseline, n_steps=50)

print("Inputs:", inputs)
print("Attributions (feature contributions):", attributions)
print("Sum of attributions:", attributions.sum().item())
print("Prediction difference:", (model(inputs) - model(baseline)).item())
```

Sample output:

```
Inputs: tensor([[1., 2., 3.]], grad_fn=<...>)
Attributions: tensor([[0.15, 0.42, 0.38]], grad_fn=<...>)
Sum of attributions: 0.95
Prediction difference: 0.95
```

We can see:

* Each feature has a contribution value (positive/negative).
* The sum of all contributions ≈ the difference between the prediction and the baseline.

---

## 📖 4. Summary

* **Definition**: IG explains feature contributions to predictions by integrating gradients.

* **Formula**:

  $$
  IG_i(x) = (x_i - x'_i) \times \int_0^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha
  $$

* **Properties**:

  * Satisfies **completeness** (sum of contributions = prediction difference).
  * Avoids the issue of gradients being 0 in saturation regions.

* **Implementation**: `captum.attr.IntegratedGradients`


