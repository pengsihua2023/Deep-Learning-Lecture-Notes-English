## L1 Norm Regularization

### 📖 What is L1 norm regularization?  
L1 norm regularization (also known as Lasso regularization) is a commonly used regularization technique in machine learning and deep learning to prevent model overfitting and promote parameter sparsity. Unlike L2 norm regularization (Ridge regularization), L1 regularization adds the L1 norm of the model parameters (i.e., the sum of the absolute values of the weights) as a penalty term to the loss function, which tends to push some weights toward zero, thereby generating a sparse model.

### 📖 Core principle
**Modification of loss function:** L1 regularization adds an L1 norm penalty term to the original loss function:

$$
Loss_{\text{regularized}} = Loss_{\text{original}} + \lambda \sum_i |w_i|
$$

- **Where**:

* $Loss_{\text{original}}$ is the original loss (such as mean squared error or cross-entropy).
* $w_i$ are model parameters (e.g., weights).
* $|w_i|$ is the absolute value of the weight, $\sum |w_i|$ is the L1 norm.
* $\lambda$ is the hyperparameter controlling the strength of regularization.

- **Effect:**

* L1 regularization tends to set unimportant weights to zero, generating a sparse model (i.e., feature selection effect).
* Sparsity helps reduce model complexity, improve interpretability, and reduce overfitting risk.
* Compared with L2 regularization (which makes weights smaller but nonzero), L1 regularization is more suitable for scenarios requiring sparse solutions.

- **Differences with L2 regularization:**

* L2 regularization $\left(\sum w_i^2\right)$ makes weights tend toward small values, keeping all weights nonzero.
* L1 regularization $\left(\sum |w_i|\right)$, due to its non-differentiability (at zero), can push some weights exactly to zero.
* L1 regularization is more effective in high-dimensional data (e.g., feature selection), while L2 regularization is more suitable for smooth weight distributions.

- **Applications**

* **Feature selection:** In machine learning (such as linear regression, logistic regression), L1 regularization can automatically select important features.
* **Neural networks:** In deep learning, L1 regularization can be used to sparsify networks (such as convolutional or fully connected layers).
* **Model compression:** Sparse models are easier to prune or quantize, suitable for resource-constrained devices.
---

### 📖 Python code example
Below is an example of implementing L1 norm regularization using PyTorch, based on the MNIST handwritten digit classification task. L1 regularization usually needs to be manually added to the loss function, since PyTorch optimizers (such as SGD, Adam) support `weight_decay` (corresponding to L2 regularization) but do not directly support L1 regularization.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Step 1: Define a simple fully connected neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input: 28x28 pixels
        self.fc2 = nn.Linear(128, 10)       # Output: 10 classes
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten input
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 2: Load MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Step 3: Initialize model, loss function and optimizer
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
l1_lambda = 1e-4  # L1 regularization coefficient

# Step 4: Training function (manually adding L1 regularization)
def train(epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # Clear gradients
        output = model(data)
        # Compute original loss
        loss = criterion(output, target)
        # Add L1 regularization term
        l1_norm = sum(torch.abs(p).sum() for p in model.parameters())
        loss = loss + l1_lambda * l1_norm
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch}, Average Loss: {total_loss / len(train_loader):.4f}')

# Step 5: Testing function
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# Step 6: Run training and testing
epochs = 5
for epoch in range(1, epochs + 1):
    train(epoch)
    test()
```

### 📖 Explanation of the code

* **Model definition:**
  SimpleNet is a simple fully connected neural network, with MNIST’s 28x28 pixel images as input, and 10-class classification as output.

* **Dataset:**
  MNIST dataset is loaded using torchvision, batch size is 64, preprocessing only includes conversion to tensor.

* **L1 regularization implementation:**
  PyTorch optimizers do not support direct L1 regularization (unlike L2 with `weight_decay`), so L1 norm is computed manually.
  $`sum(torch.abs(p).sum() for p in model.parameters())`$ computes the L1 norm (sum of absolute values) of all parameters.
  The L1 norm is multiplied by coefficient `l1_lambda` (e.g., 1e-4) and added to the loss function.

* **Training and testing:**
  During training, the loss includes both the original cross-entropy loss and the L1 penalty.
  Each epoch prints the average loss, testing computes classification accuracy.
  L1 regularization pushes some weights toward zero, sparsity can be verified by checking `model.parameters()`.

* **Example output:**

  ```
  Epoch 1, Average Loss: 0.9123
  Test Accuracy: 91.50%
  Epoch 2, Average Loss: 0.4321
  Test Accuracy: 93.80%
  ...
  Epoch 5, Average Loss: 0.3124
  Test Accuracy: 95.20%
  ```

  Actual values vary due to random initialization.

---

**Check sparsity**
To verify the sparsity effect of L1 regularization, check model weights after training:

```python
# Check weight sparsity
def check_sparsity(model):
    total_params = 0
    zero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
    sparsity = 100. * zero_params / total_params
    print(f'Model sparsity: {sparsity:.2f}% (ratio of zero weights)')

# Call after training
check_sparsity(model)
```

L1 regularization increases the proportion of zero weights, especially when `l1_lambda` is large.

---

### 📖 **Comparison between L1 and L2 regularization**

* **L1 regularization:**

  * Tends to produce sparse weights (some weights = 0).
  * Suitable for feature selection or compression scenarios.
  * Gradient is discontinuous (non-differentiable at zero), optimization may be more complex.

* **L2 regularization:**

  * Makes weights small but nonzero, keeps them smooth.
  * Suitable for scenarios requiring stable gradient updates.
  * PyTorch supports it directly via `weight_decay`.


### 📖 **Combining L1 and L2 (Elastic Net)**

Both L1 and L2 regularization can be used simultaneously to form Elastic Net regularization:

```python
l1_lambda = 1e-4
l2_lambda = 1e-4
loss = criterion(output, target)
l1_norm = sum(torch.abs(p).sum() for p in model.parameters())
l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm
```



### 📖 **Practical application scenarios**

* **Feature selection:** In traditional machine learning (e.g., logistic regression), L1 regularization is used to select important features.
* **Neural network compression:** L1 regularization can generate sparse networks, making pruning or deployment on resource-limited devices easier.
* **High-dimensional data:** When there are many input features, L1 regularization effectively reduces irrelevant weights.
* **Hyperparameter tuning:** `l1_lambda` needs to be tuned via cross-validation or Bayesian optimization, typical range is from 1e-5 to 1e-3.


