## Adadelta Optimizer

### Principle and Usage of the Adadelta Optimizer

#### 📖 **Principle**

Adadelta (Adaptive Delta) is an adaptive learning rate optimization algorithm designed to address the issue in Adagrad where the learning rate monotonically decreases towards zero over time. It is an improved version of Adagrad, combining the ideas of momentum and adaptive learning rate, and is suitable for optimizing deep learning models. The core principles of Adadelta are as follows:

1. **Core Idea**:

   * Instead of directly accumulating all historical squared gradients (as in Adagrad), Adadelta uses a **sliding window** to compute the **Exponential Moving Average (EMA)** of squared gradients, limiting the influence of historical information.
   * It introduces two accumulators:

* EMA of squared gradients \$(E\[g^2])\$: Used to adaptively scale the learning rate.

* EMA of squared parameter updates \$(E\[\Delta x^2])\$: Used to normalize the update step size, simulating the effect of momentum.

  * By combining these, Adadelta achieves adaptive updates without the need to manually set a global learning rate.

2. **Update Formulas**:

   * EMA of squared gradients:

<img width="270" height="45" alt="image" src="https://github.com/user-attachments/assets/1a3b1fd4-27ce-4eda-977d-cbc729c83301" />

where \$g\_t\$ is the current gradient, and \$\rho\$ is the decay rate (typically close to 1, e.g., 0.9).

* Parameter update step:

$$
\Delta x_t = - \frac{\sqrt{E[\Delta x^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t
$$

where \$\epsilon\$ is a small constant (to prevent division by zero), and \$E\[\Delta x^2]\$ is the EMA of squared updates.

* **Parameter Update**:

$$
x_{t+1} = x_t + \Delta x_t
$$

* **EMA of squared updates**:

<img width="323" height="40" alt="image" src="https://github.com/user-attachments/assets/0fa804b2-af48-407a-9463-09167cc55b6a" />

3. **Advantages**:

   * **No need to set a learning rate**: Step size is adaptively adjusted via EMA.
   * **Robust to sparse gradients**: Suitable for non-stationary objectives in deep learning.
   * **Stable convergence**: Avoids Adagrad’s rapid learning rate decay issue.

4. **Disadvantages**:

   * Slightly higher computational complexity than SGD.
   * Hyperparameters \$\rho\$ and \$\epsilon\$ must be carefully chosen (default values usually work well).

#### 📖 **Use Cases**

* Suitable for optimizing deep learning models (such as CNNs and RNNs), especially in tasks with sparse data or highly variable gradients.
* Commonly used in image classification, text classification, and similar tasks, particularly when reducing the need for manual learning rate tuning.


#### **PyTorch Usage**

PyTorch provides a built-in `torch.optim.Adadelta` optimizer, which is very simple to use. Below is a minimal code example showing how to train a simple CNN with Adadelta for an image classification task.

##### **Code Example**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# 1. Define model (using pretrained ResNet18 for simplicity)
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # Assume 10 classes

# 2. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), rho=0.9, eps=1e-6)  # Adadelta optimizer

# 3. Prepare data (example data)
inputs = torch.rand(16, 3, 224, 224)  # Batch of images (batch, channels, height, width)
labels = torch.randint(0, 10, (16,))   # Random labels

# 4. Training step
model.train()
optimizer.zero_grad()  # Clear gradients
outputs = model(inputs)  # Forward pass
loss = criterion(outputs, labels)  # Compute loss
loss.backward()  # Backward pass
optimizer.step()  # Update parameters

print(f"Loss: {loss.item()}")
```

##### 📖 **Code Explanation**

* **Model**: Uses pretrained ResNet18, replacing the last layer for a 10-class classification task.
* **Optimizer**: `optim.Adadelta` initialization, parameters include:

  * `model.parameters()`: Learnable parameters of the model.
  * `rho`: EMA decay rate, default 0.9.
  * `eps`: Small constant to prevent division by zero, default 1e-6.
  * `lr`: Initial learning rate (default 1.0, usually no adjustment needed).
* **Training**: Standard forward pass, loss calculation, backward pass, and parameter update process.
* **Data**: Randomly generated images and labels as examples; in practice, replace with real datasets (e.g., CIFAR-10).



#### 📖 **Notes**

1. **Hyperparameters**:

   * `rho`: Controls influence of past gradients, typically 0.9–0.95.
   * `eps`: Prevents division by zero, typically set between 1e-6 and 1e-8.
   * Adadelta is insensitive to the initial learning rate `lr`, default 1.0 is generally fine.
2. **Data Preprocessing**:

   * Images should be normalized (e.g., mean \[0.485, 0.456, 0.406], std \[0.229, 0.224, 0.225]).
   * Use `torchvision.transforms` for preprocessing.
3. **Computation Device**:

   * If using GPU, move model and data to GPU: `model.cuda()`, `inputs.cuda()`, `labels.cuda()`.
4. **Practical Application**:

   * Replace example data with real datasets (e.g., `torchvision.datasets.CIFAR10`).
   * Add data loaders (`DataLoader`) and multiple training epochs.



#### 📖 **Summary**

Adadelta is an efficient adaptive optimizer that dynamically adjusts the learning rate using EMA, making it suitable for deep learning tasks. PyTorch’s `optim.Adadelta` is simple and easy to use, requiring only model parameters and a few hyperparameters to get started.

