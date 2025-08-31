

## Adaptive Learning Rate (RMSprop)

### What is Adaptive Learning Rate (RMSprop, Optimizer)?

RMSprop (Root Mean Square Propagation) is a commonly used adaptive learning rate optimization algorithm in deep learning. It aims to accelerate the convergence of gradient descent by adaptively adjusting the learning rate. It is particularly suitable for handling non-stationary objective functions (such as loss functions in neural networks) by dynamically adjusting each parameter’s learning rate based on the exponential moving average of the squared gradients. RMSprop is one of the predecessors of the Adam optimizer, simple yet efficient.

#### Core Principle

RMSprop scales the learning rate by maintaining the exponential moving average of squared gradients. The detailed steps are:

1. **Compute Gradient**: Compute the gradient of the loss function with respect to the parameters \$g\_t\$.

2. **Update Mean of Squared Gradients**:

<img width="200" height="35" alt="image" src="https://github.com/user-attachments/assets/28de9382-ff6b-4837-9e47-6cd6199a3e2b" />  

* \$\rho\$ is the decay rate (typically 0.9), which controls the weight of historical gradients.

3. **Parameter Update**:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t} + \epsilon} g_t
$$

* \$\eta\$: Initial learning rate (commonly 0.001).
* \$\epsilon\$: A small constant (commonly \$1\text{e-}8\$) to prevent division by zero.
* \$\sqrt{E\[g^2]\_t}\$: The root mean square (RMS) of the squared gradients, used to adaptively scale the learning rate.

---

#### Advantages

* **Adaptivity**: Dynamically adjusts the learning rate based on gradient magnitudes, suitable for sparse or noisy gradients.
* **Simple and Efficient**: Faster convergence than SGD and easy to implement.
* **Stability**: Smooths gradient fluctuations through exponential moving average, reducing oscillations.

#### Limitations

* **Lack of Momentum**: Unlike Adam, RMSprop does not use first-order momentum, which may result in slower convergence in some tasks.
* **Hyperparameter Sensitivity**: Initial learning rate and decay rate need careful tuning.

#### Comparison with Adam

* **RMSprop**: Uses only the second-order moment (mean of squared gradients) to scale the learning rate.
* **Adam**: Combines first-order moment (momentum) and second-order moment, typically converging faster (see Adam optimizer section).

---

### Python Code Example

Below is a minimal PyTorch example demonstrating how to use RMSprop for the MNIST handwritten digit classification task. The code is kept simple, focusing on the RMSprop implementation.

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
        self.fc = nn.Linear(28 * 28, 10)  # Input: 28x28 pixels, Output: 10 classes
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten input
        x = self.fc(x)
        return x

# Step 2: Load MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Step 3: Initialize model, loss function, and RMSprop optimizer
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-8)

# Step 4: Training function
def train(epoch):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}')

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

# Step 6: Training loop
epochs = 5
for epoch in range(1, epochs + 1):
    train(epoch)
    test()
```

---

### Code Explanation

1. **Model Definition**:

   * `SimpleNet` is a minimal fully connected neural network, input is a 28x28 MNIST image, output is 10 classification categories.

2. **Dataset**:

   * MNIST dataset is loaded with `torchvision`, batch size is 64, and preprocessing only includes tensor conversion.

3. **RMSprop Optimizer**:

   * Initialized as `optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-8)`.

     * `lr=0.001`: Initial learning rate, common default for RMSprop.
     * `alpha=0.9`: Decay rate (corresponding to \$\rho\$ in the formula), controls smoothing of squared gradient averages.
     * `eps=1e-8`: Small constant to avoid division by zero.
   * RMSprop adaptively adjusts learning rates based on squared gradient averages.

4. **Training & Testing**:

   * During training, RMSprop updates parameters and prints average loss.
   * During testing, classification accuracy is computed.

5. **Output Example**:

   ```
   Epoch 1, Loss: 0.4567, Test Accuracy: 92.30%
   Epoch 2, Loss: 0.2987, Test Accuracy: 93.50%
   Epoch 3, Loss: 0.2678, Test Accuracy: 94.10%
   Epoch 4, Loss: 0.2456, Test Accuracy: 94.40%
   Epoch 5, Loss: 0.2321, Test Accuracy: 94.60%
   ```

   Actual results may vary due to random initialization.

---

### Key Points

* **Adaptivity**: RMSprop dynamically adjusts learning rates based on squared gradients, suitable for non-stationary loss functions.
* **Hyperparameters**:

  * `lr=0.001`: Default value usually effective.
  * `alpha=0.9`: Controls historical gradient weighting, commonly between 0.9–0.99.
* **Comparison with Adam**: RMSprop only uses the second-order moment, while Adam also incorporates the first-order moment, leading to faster convergence but slightly higher memory cost.

---

### Practical Applications

* **Deep Learning**: Suitable for CNNs, RNNs, especially in tasks with sparse or noisy gradients.
* **Alternative to Adam**: Simpler computation, suitable for resource-constrained scenarios.
* **Combination with Other Techniques**: Can be combined with Dropout, BatchNorm, or learning rate schedulers (e.g., ReduceLROnPlateau).

#### Notes

* **Learning Rate Tuning**: `lr` should be adjusted depending on the task, typically ranging from 1e-4 to 1e-2.
* **Decay Rate**: If `alpha` is too high, the optimizer may be insensitive to new gradients; if too low, training may become unstable.
* **Convergence**: In some tasks, RMSprop may be less stable than Adam, in which case Adam or AdamW can be considered.


