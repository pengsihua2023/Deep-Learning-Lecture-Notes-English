# L2 Norm Regularization
### 📖 What is L2 Norm Regularization?

L2 norm regularization (also known as Weight Decay) is a commonly used regularization technique in machine learning and deep learning to prevent model overfitting. Its core idea is to add a penalty term to the loss function to constrain the size of model parameters (weights), thereby improving the generalization ability of the model.

### 📖 Core Principle
- **Loss function modification**: Add an L2 norm penalty term to the original loss function (such as mean squared error or cross-entropy):

$$
\text{Loss}_ {\text{regularized}} = \text{Loss}_{\text{original}} + \lambda \sum_i \|w_i\|_2^2
$$

Where:

* $\text{Loss}_{\text{original}}$ is the original loss (such as classification error).  
* $w_i$ are model parameters (e.g., weights).  
* $\|w_i\|_2^2$ is the L2 norm of the weights (i.e., the sum of squares of weights).  
* $\lambda$ is the hyperparameter controlling the strength of the regularization penalty.  

### 📖 **Effect**:  
  - L2 regularization tends to make weights smaller (but not exactly zero), making the model smoother and reducing overfitting to training data.  
  - It encourages the model to learn smaller weights, lowering model complexity and preventing overfitting to noise.  

- **Difference from L1 regularization**:  

* L1 regularization (Lasso) uses $\sum |w_i|$, which tends to produce sparse weights (some weights are exactly 0).  
* L2 regularization (Ridge) uses $\sum w_i^2$, which tends to spread out the weight values, keeping all weights small.  

### 📖 Application Scenarios
- In deep neural networks (such as CNN, RNN, Transformer), L2 regularization is commonly used to prevent overfitting.  
- Combined with other regularization methods such as Dropout and Batch Normalization.  
- Hyperparameter $\lambda$ is usually tuned via cross-validation or Bayesian optimization (as described previously).  

---

### 📖 Python Code Example

Below is an example of implementing L2 norm regularization using PyTorch, showing how to apply weight decay during neural network training. The example is based on a simple fully connected network trained on the MNIST handwritten digit classification task.

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

# Step 3: Initialize model, loss function and optimizer (with L2 regularization)
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
weight_decay = 1e-4  # L2 regularization coefficient (i.e. \lambda)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=weight_decay)

# Step 4: Training function
def train(epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # Clear gradients
        output = model(data)
        loss = criterion(output, target)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters (includes L2 regularization)
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
````



### 📖 Code Explanation

1. **Model definition**:

   * `SimpleNet` is a simple fully connected neural network for MNIST classification (28x28 pixel input, 10-class output).
   * Contains two linear layers and ReLU activation.

2. **Dataset**:

   * MNIST dataset is loaded with PyTorch `torchvision`, batch size = 64.
   * Preprocessing only includes converting to tensor (`ToTensor`).

3. **L2 regularization**:

* In the optimizer, L2 regularization is achieved via `weight_decay=1e-4`.
* PyTorch’s `weight_decay` parameter automatically adds an L2 penalty in the optimizer update step, equivalent to adding the following term to the loss function:

$$
\lambda \sum w_i^2
$$

* Mathematically, the SGD weight update becomes:

$$
w \leftarrow w - \eta \left( \frac{\partial \text{Loss}}{\partial w} + \lambda w \right)
$$

Where \$\eta\$ is the learning rate, and \$\lambda\$ is `weight_decay`.

4. **Training and testing**:

   * Train for 5 epochs, printing average loss each time.
   * During testing, classification accuracy is computed to evaluate performance.

5. **Example output**:

   ```
   Epoch 1, Average Loss: 0.8923
   Test Accuracy: 92.15%
   Epoch 2, Average Loss: 0.4125
   Test Accuracy: 94.30%
   ...
   Epoch 5, Average Loss: 0.2987
   Test Accuracy: 95.80%
   ```

   Actual results may vary slightly due to random initialization.

---

### 📖 Practical Application Scenarios

* **Deep learning**: L2 regularization is often used in CNNs, RNNs, Transformers, etc., to prevent overfitting.
* **Hyperparameter selection**: $\lambda$ (e.g., 1e-4) is usually tuned via cross-validation or Bayesian optimization, with a typical range of 1e-5 to 1e-2.
* **Combined with other regularization**: L2 regularization is often combined with Dropout, BatchNorm, etc., for better results.
* **Limitations of weight decay**: For certain tasks (e.g., sparse models), L1 regularization or Elastic Net (L1+L2) may be more suitable.



