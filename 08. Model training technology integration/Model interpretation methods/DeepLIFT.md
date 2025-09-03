
# DeepLIFT Model Interpretation Method

## 1. Definition

**DeepLIFT (Deep Learning Important FeaTures)** is a **model interpretation method**. It compares the differences in activations of a neural network under a **reference input** and an **actual input**, and assigns contribution values of each input feature to the output.

Unlike gradient-based methods, DeepLIFT uses **difference propagation rules (rescale rules & reveal-cancel rules)**, which avoids gradient vanishing or explosion, and provides more stable explanations of model predictions.

👉 Intuitive understanding:

* Reference point (baseline) = the model’s output on some “neutral input” (e.g., all-zero vector, mean input).
* DeepLIFT explanation result = the **decomposed contribution of changes** in prediction caused by the actual input relative to the reference input.

---

## 2. Mathematical Description

### 2.1 Basic Idea

Let:

* Input features: \$x\$
* Reference input: \$x'\$
* Model output: \$f(x)\$, reference output: \$f(x')\$
* Differences:

  $$
  \Delta x = x - x', \quad \Delta y = f(x) - f(x')
  $$

DeepLIFT assigns influence to each input feature by computing the **contribution score \$C\_{\Delta x\_i \to \Delta y}\$**:

$$
\sum_i C_{\Delta x_i \to \Delta y} = \Delta y
$$

### 2.2 Propagation Rules

During layer-by-layer propagation in the neural network, DeepLIFT defines several rules:

* **Rescale Rule**: When the input and output are monotonic, contributions are distributed proportionally:

  $$
  C_{\Delta x_i \to \Delta y} = \frac{\Delta y}{\sum_j \Delta x_j} \cdot \Delta x_i
  $$

* **RevealCancel Rule**: Used to capture nonlinear interactions among inputs by comparing the independent contributions of positive and negative parts.

---

## 3. Simple Code Example

We use `captum` (PyTorch’s interpretability library) to demonstrate how to apply DeepLIFT.

```python
import torch
import torch.nn as nn
from captum.attr import DeepLift

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
baseline = torch.zeros_like(inputs)  # Reference input baseline=0

# 3. Use DeepLIFT for interpretation
deeplift = DeepLift(model)
attributions = deeplift.attribute(inputs, baselines=baseline)

print("Inputs:", inputs)
print("Attributions (feature contributions):", attributions)
```

Sample output:

```
Inputs: tensor([[1., 2., 3.]], grad_fn=<...>)
Attributions: tensor([[ 0.12,  0.45,  0.33]], grad_fn=<...>)
```

Here, `Attributions` are the contribution values assigned by DeepLIFT to each feature. Their sum equals the difference between the prediction and the baseline.

---

## 4. Summary

* **Definition**: DeepLIFT assigns prediction contributions by comparing input with reference input.

* **Mathematical formula**:

  $$
  f(x) - f(x') = \sum_i C_{\Delta x_i \to \Delta y}
  $$

* **Advantage**: Solves the failure of gradient methods in saturation regions.

* **Code**: Can be directly applied via `captum`’s `DeepLift`.

