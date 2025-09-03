

## Adaptive Learning Rate (Adam Optimizer)

### What is Adaptive Learning Rate (Adam Optimizer)?

The Adam optimizer (Adaptive Moment Estimation) is a widely used optimization algorithm in deep learning. It combines the advantages of Momentum and RMSProp, adaptively adjusting the learning rate to accelerate gradient descent convergence. It is particularly suitable for handling sparse gradients or noisy optimization problems.

#### Core Principle

Adam dynamically adjusts the learning rate of each parameter by tracking the exponential moving averages of the first moment (mean) and the second moment (variance) of gradients. The main steps are:

1. **Compute Gradient**: Compute the gradient of the loss function with respect to the parameters \$g\_t\$.

2. **Update First Moment (Momentum)**:
   $m\_t = \beta\_1 m\_{t-1} + (1 - \beta\_1) g\_t\$ , similar to Momentum.

3. **Update Second Moment (Variance)**:
   $v\_t = \beta\_2 v\_{t-1} + (1 - \beta\_2) g\_t^2\$ , similar to RMSProp.

4. **Bias Correction**: Apply bias correction to \$m\_t\$ and \$v\_t\$ to ensure unbiased estimates at the beginning.

5. **Parameter Update**: Update parameters using adaptive learning rates:

$$
\theta_{t+1} = \theta_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

Where:

* \$\eta\$ : Initial learning rate (commonly 0.001).
* \$\beta\_1, \beta\_2\$ : Momentum parameters (commonly 0.9 and 0.999).
* \$\epsilon\$ : Small constant to prevent division by zero (commonly \$1\text{e-}8\$).

---

#### Advantages

* **Adaptivity**: Automatically adjusts learning rates based on gradient history, no need for manual tuning.
* **Efficiency**: Suitable for large-scale datasets and complex models (e.g., deep neural networks).
* **Stability**: Performs well with sparse gradients or noisy optimization problems.

#### Limitations

* May not converge to the optimal solution in some tasks compared to SGD + Momentum.
* Hyperparameters (such as \$\beta\_1, \beta\_2\$) still need careful selection.

---

### Python Code Example

Below is a simple PyTorch example using the Adam optimizer to train a fully connected neural network for the MNIST handwritten digit classification task. The code focuses on Adam implementation and remains concise.

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

# Step 3: Initialize model, loss function, and Adam optimizer
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

# Step 4: Training function
def train(epoch):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()  # Clear gradients
        output = model(data)
        loss = criterion(output, target)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters using Adam
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

---

### Code Explanation

1. **Model Definition**:

   * `SimpleNet` is a simple fully connected neural network, input is a 28x28 MNIST image, output is 10 classes.

2. **Dataset**:

   * Uses `torchvision` to load MNIST dataset, batch size is 64, preprocessing only includes tensor conversion.

3. **Adam Optimizer**:

   * Initialized with `optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)`.
   * `lr=0.001`: Initial learning rate, Adam’s default usually works well.
   * `betas=(0.9, 0.999)`: Decay rates for first and second moments, standard values.
   * `eps=1e-8`: Small constant to prevent division by zero.

4. **Training & Testing**:

   * During training, Adam adaptively adjusts the learning rate based on gradients and updates parameters.
   * Prints average loss per epoch, computes classification accuracy during testing.

5. **Output Example**:

   ```
   Epoch 1, Average Loss: 0.3256
   Test Accuracy: 94.50%
   Epoch 2, Average Loss: 0.1423
   Test Accuracy: 96.20%
   ...
   Epoch 5, Average Loss: 0.0854
   Test Accuracy: 97.80%
   ```

   Actual values may vary due to random initialization.

---

### Adam vs. SGD

To demonstrate Adam’s adaptivity, compare with SGD implementation (without momentum):

```python
# Using SGD optimizer (without momentum)
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Replace Adam
```

* **Adam**: Adaptively adjusts learning rates, usually converges faster, less sensitive to initial learning rate.
* **SGD**: Fixed learning rate, may require manual adjustment or combination with schedulers (e.g., StepLR).

---

### Practical Applications

* **Deep Learning**: Adam is the default optimizer for CNNs, RNNs, Transformers, etc., due to fast convergence and stability.
* **Sparse Data**: Performs well on sparse gradients (e.g., NLP tasks).
* **Hyperparameter Tuning**: Although Adam is less sensitive to learning rate, tuning `lr`, `betas`, etc. (e.g., via Bayesian optimization) can further improve results.

#### Notes

* **Learning Rate**: Default `lr=0.001` usually works well, but may require fine-tuning (e.g., \$1\text{e-}4\$ to \$1\text{e-}2\$).
* **Convergence**: On some tasks, Adam may converge to suboptimal solutions; alternatives include SGD + Momentum or AdamW (improved Adam with L2 regularization).
* **Memory Usage**: Adam requires storing first and second moments, using slightly more memory than SGD.



