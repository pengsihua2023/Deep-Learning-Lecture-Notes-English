
# Saliency Maps Model Interpretation Method

## 📖 1. Definition

**Saliency Map** is a **gradient-based model interpretation method**, originally used for image classification.
It measures which input features are most important to the model’s prediction by computing the **gradient magnitude of the output prediction with respect to the input pixels**.

* If a pixel’s gradient is larger, it means it has more influence on the prediction result;
* Saliency maps are usually visualized as heatmaps, highlighting the most critical regions.

This method is suitable for **neural network image models**, and can also be extended to text and tabular tasks.



## 📖 2. Mathematical Description

Let:

* Input image vector \$x \in \mathbb{R}^d\$
* Prediction score (logit) for target class \$S\_c(x)\$

The saliency map is defined as:

$$
M(x) = \left| \frac{\partial S_c(x)}{\partial x} \right|
$$

Where:

* \$\frac{\partial S\_c(x)}{\partial x}\$: gradient of the target class score with respect to each pixel;
* Absolute value (or square) indicates the importance of each pixel.

Finally, the saliency map \$M(x)\$ can be visualized as a heatmap to show the regions the model focuses on.



## 📖 3. Minimal Code Example

Generate a saliency map on MNIST using **PyTorch**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ===== Simple CNN model =====
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.fc1 = nn.Linear(26*26*16, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

# ===== Load data & model =====
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
x, y = dataset[0]   # take the first image
x = x.unsqueeze(0)  # [1,1,28,28]

model = SimpleCNN()
model.eval()

# ===== Forward pass =====
x.requires_grad_()  # allow gradient computation w.r.t. input
output = model(x)
pred_class = output.argmax(dim=1).item()

# ===== Compute gradients =====
score = output[0, pred_class]  # target class score
score.backward()  # backpropagation
saliency = x.grad.data.abs().squeeze().numpy()  # saliency map

# ===== Visualization =====
plt.subplot(1,2,1)
plt.title(f"Original Image (label={y})")
plt.imshow(x.detach().squeeze(), cmap="gray")

plt.subplot(1,2,2)
plt.title("Saliency Map")
plt.imshow(saliency, cmap="hot")
plt.show()
```



## 📖 Results Explanation

* **Left image**: Original MNIST digit (e.g., “7”).
* **Right image**: Saliency map, showing which pixels contribute most to predicting “7” (usually the digit’s edges).



## 📖 Summary

* **Definition**: Saliency Maps explain model predictions using input gradients.
* **Formula**: \$M(x) = \left| \frac{\partial S\_c(x)}{\partial x} \right|\$.
* **Code**: Just a few PyTorch gradient operations are needed to generate saliency maps.


## 📖 Saliency Maps / LIME / SHAP Comparison

| Feature              | **Saliency Maps**                                                             | **LIME**                                                                  | **SHAP**                                                                                            |
| -------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Principle**        | Compute gradients of prediction score w\.r.t. inputs                          | Perturb input neighborhood and fit an interpretable model (linear/tree)   | Use game theory Shapley values to measure marginal contribution of each feature                     |
| **Scope**            | **Local explanation**: highlights pixels/features important for one input     | **Local explanation**: shows features that drive prediction for one input | **Local + global explanation**: per-feature Shapley values can be aggregated into global importance |
| **Model dependency** | Requires gradients → only works for differentiable models (e.g., neural nets) | **Model-agnostic**, needs only prediction API                             | **Model-agnostic**, with optimized versions (TreeSHAP, DeepSHAP)                                    |
| **Output form**      | Heatmap (pixel-level), highlights critical regions                            | Feature weights list (positive/negative influence)                        | Feature contribution values (additive, fair distribution of prediction)                             |
| **Stability**        | Unstable (sensitive to noise, gradient vanishing)                             | Unstable (different samples may yield different explanations)             | Stable (Shapley values are unique, fairness guaranteed)                                             |
| **Cost**             | Low (just one backpropagation)                                                | Medium (sampling + linear model fitting)                                  | High (exact Shapley is exponential; approximations exist)                                           |
| **Use cases**        | Computer vision (image classification, medical imaging)                       | General (NLP, tabular, images)                                            | High-risk tasks (finance, healthcare) requiring reliable explanations                               |


## 📖 Final Summary

* **Saliency Maps**: Gradient-based, good for image models, fast but unstable.
* **LIME**: Sampling + interpretable model, intuitive and flexible but potentially unstable.
* **SHAP**: Strongest theoretical basis, stable and reliable but computationally more expensive.



