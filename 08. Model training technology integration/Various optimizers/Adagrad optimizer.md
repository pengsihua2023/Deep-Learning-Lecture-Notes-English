
## Adagrad Optimizer (Adaptive Gradient Algorithm)

### Principle and Usage

#### 📖 **Principle**

Adagrad (Adaptive Gradient Algorithm) is an adaptive learning rate optimization algorithm specifically designed for handling sparse data and convex optimization problems. It adjusts the learning rate of each parameter based on the accumulation of historical gradients, so that frequently updated parameters have smaller learning rates, while sparsely updated parameters maintain larger learning rates. The core principles of Adagrad are as follows:

1. **Core Idea**:

   * Adagrad maintains a cumulative sum of squared historical gradients for each parameter, which is used to adaptively scale the learning rate.
   * Parameters with larger gradients (frequent updates) gradually have smaller learning rates, while parameters with smaller gradients (sparse updates) maintain larger learning rates, thus accelerating the convergence of sparse features.

2. **Update Formulas**:

* **Accumulation of squared gradients**:

$$
G_t = G_{t-1} + g_t^2
$$

where \$g\_t\$ is the current gradient, and \$G\_t\$ is the cumulative sum of squared historical gradients.

* **Parameter Update**:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \cdot g_t
$$

* **Where**:

  * \$\theta\_t\$: Current parameter.
  * \$\eta\$: Initial learning rate (typically 0.01).
  * \$\epsilon\$: A small constant (to prevent division by zero, typically \$1e^{-8}\$).
  * \$\sqrt{G\_t} + \epsilon\$: Adaptive scaling factor, ensuring the learning rate decreases with accumulated gradients.

3. **Characteristics**:

   * **Advantages**:

     * Adaptive learning rate, less manual tuning required.
     * Especially suitable for sparse data (e.g., word embeddings in NLP).
   * **Disadvantages**:

     * The accumulation of squared gradients increases monotonically, which may cause the learning rate to decay too quickly to nearly zero, halting learning.
     * Convergence on non-convex problems (such as deep neural networks) may be slow.
   * **Comparison with Adadelta**:

     * Adagrad accumulates all historical squared gradients, which may cause the learning rate to decay prematurely.
     * Adadelta uses Exponential Moving Average (EMA) to limit the effect of historical gradients, improving Adagrad’s decay issue.
   * **Comparison with SGD**:

     * SGD uses a fixed learning rate, requiring manual adjustment.
     * Adagrad adaptively adjusts the learning rate, making it suitable for sparse data, but optimization may stop if the learning rate decays too quickly.

4. **Use Cases**:

   * Suitable for sparse data tasks, such as NLP (word embedding training) and recommendation systems.
   * For non-convex problems in deep learning, it may need to be combined with improved algorithms such as Adadelta or RMSProp.

---

#### 📖 **PyTorch Usage**

PyTorch provides a built-in `torch.optim.Adagrad` optimizer, which is very simple to use. Below is a minimal code example showing how to train a simple CNN with Adagrad for an image classification task.

##### **Code Example**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# 1. Define model (using pretrained ResNet18)
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # Assume 10 classes

# 2. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters(), lr=0.01, eps=1e-8)  # Adagrad optimizer

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

#### 📖 **Code Explanation**

* **Model**: Uses pretrained ResNet18, replacing the last layer for a 10-class classification task.
* **Optimizer**: `optim.Adagrad` initialization, parameters include:

  * `model.parameters()`: Learnable parameters of the model.
  * `lr`: Initial learning rate, set to 0.01 (commonly between 0.001\~0.1).
  * `eps`: Small constant to prevent division by zero, set to 1e-8 (default value).
* **Training**: Standard forward pass, loss calculation, backward pass, and parameter update process.
* **Data**: Randomly generated images and labels as examples; in practice, replace with real datasets (e.g., CIFAR-10).



#### 📖 **Notes**

1. **Hyperparameters**:

   * `lr`: Initial learning rate, usually set to 0.01, should be adjusted based on the task.
   * `eps`: Prevents division by zero, commonly set to 1e-8, has minor impact.
   * Optional parameter `weight_decay` (default 0) for L2 regularization.
2. **Data Preprocessing**:

   * Images should be normalized (e.g., mean \[0.485, 0.456, 0.406], std \[0.229, 0.224, 0.225]).
   * Use `torchvision.transforms` for preprocessing.
3. **Computation Device**:

   * If using GPU, move model and data to GPU: `model.cuda()`, `inputs.cuda()`, `labels.cuda()`.
4. **Practical Application**:

   * Replace example data with real datasets (e.g., `torchvision.datasets.CIFAR10`).
   * Add data loaders (`DataLoader`) and multiple training epochs.
5. **Limitations**:

   * If convergence is too slow or halts (due to rapid learning rate decay), consider Adadelta or RMSProp.



#### **Summary**

Adagrad is an adaptive learning rate optimizer that dynamically adjusts the learning rate through the accumulation of squared gradients, making it especially suitable for sparse data tasks. PyTorch’s `optim.Adagrad` implementation is simple and only requires specifying the learning rate and a few additional parameters. Compared to SGD, Adagrad reduces the need for learning rate tuning; compared to Adadelta, Adagrad is simpler but may stop optimizing due to overly rapid learning rate decay.

