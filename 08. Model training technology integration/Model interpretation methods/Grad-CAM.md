
# Grad-CAM Model Interpretation Method (Gradient-weighted Class Activation Mapping)

## 1. Definition

**Grad-CAM** is a method for **visualizing the basis of neural network predictions**, commonly used for explaining CNN image classification models.

* By calculating the **effect of gradients of the target class on convolutional feature maps**, we obtain the importance weights of each channel;
* These weights are then used to weight the feature maps, producing a **class activation heatmap**;
* This heatmap shows which regions of the input image the model focuses on most.

Compared with **Saliency Maps**, Grad-CAM produces smoother and more human-interpretable results.

---

## 2. Mathematical Description

Let:

* Input image \$x\$,
* Feature map of the last convolutional layer \$A^k \in \mathbb{R}^{H \times W}\$, where \$k\$ is the channel index,
* Prediction score (logit) for class \$c\$: \$y^c\$.

**Steps of Grad-CAM**:

1. **Compute weights**: For class \$c\$, compute the gradient of \$y^c\$ with respect to the feature map \$A^k\$, and take the spatial average:

$$
\alpha_k^c = \frac{1}{H \times W} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}
$$

2. **Generate heatmap**:

$$
L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)
$$

3. **Upsample** to the input image size and overlay it on the image.

---

## 3. Minimal Code Example

Implementing Grad-CAM on ResNet18 using **PyTorch**:

```python
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ===== Pretrained model =====
model = models.resnet18(pretrained=True)
model.eval()

# ===== Input image processing =====
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
img = Image.open("cat.jpg")  # Input an image
x = preprocess(img).unsqueeze(0)

# ===== Hook: capture feature maps and gradients of the last conv layer =====
features = []
grads = []

def forward_hook(module, input, output):
    features.append(output)

def backward_hook(module, grad_in, grad_out):
    grads.append(grad_out[0])

layer = model.layer4[1].conv2  # Last convolutional layer in ResNet18
layer.register_forward_hook(forward_hook)
layer.register_backward_hook(backward_hook)

# ===== Forward prediction =====
output = model(x)
pred_class = output.argmax(dim=1).item()

# ===== Backpropagation to get gradients =====
model.zero_grad()
score = output[0, pred_class]
score.backward()

# ===== Compute Grad-CAM =====
feature_map = features[0].squeeze(0)      # [C,H,W]
grad = grads[0].squeeze(0)                # [C,H,W]
weights = grad.mean(dim=(1,2))            # [C]

cam = torch.zeros(feature_map.shape[1:], dtype=torch.float32)
for i, w in enumerate(weights):
    cam += w * feature_map[i]

cam = F.relu(cam)
cam = cam / cam.max()  # Normalize to [0,1]

# ===== Visualization =====
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img)

plt.subplot(1,2,2)
plt.title("Grad-CAM")
plt.imshow(img)
plt.imshow(cam.detach().numpy(), cmap='jet', alpha=0.5)  # Overlay heatmap
plt.show()
```

---

## Results Explanation

* **Left image**: the original input image (e.g., a cat).
* **Right image**: the Grad-CAM heatmap, where red regions indicate the parts the model focuses on most (e.g., the cat’s face).

---

## Summary

* **Grad-CAM definition**: a gradient-based class activation visualization method to explain CNN predictions.
* **Formula**: \$\alpha\_k^c = \frac{1}{H \times W}\sum\_{i,j} \frac{\partial y^c}{\partial A^k\_{ij}},; L^c = ReLU(\sum\_k \alpha\_k^c A^k)\$.
* **Code**: requires only hooking feature maps and gradients, then weighting to generate the heatmap.



