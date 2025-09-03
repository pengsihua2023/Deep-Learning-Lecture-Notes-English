# StepLR

StepLR is a **Learning Rate Scheduler** commonly used in deep learning training. Its function is to proportionally decay the learning rate at fixed interval periods. This method is often seen in the PyTorch framework.

## Definition

The basic idea of `StepLR` is:
During training, after every fixed number of epochs, the learning rate is multiplied by a decay factor γ (gamma), thereby gradually reducing the learning rate and helping the model converge better.

## Mathematical Description

Set:

* Initial learning rate: \$\eta\_0\$
* Decay factor: \$\gamma \in (0, 1)\$
* Step size: \$s\$ (how many epochs between each decay)
* Current epoch: \$t\$

Then, the learning rate at epoch \$t\$ is:

$$
\eta_t = \eta_0 \cdot \gamma^{\left\lfloor \frac{t}{s} \right\rfloor}
$$

Where:

* \$\left\lfloor \cdot \right\rfloor\$ denotes the floor function.
* When \$t < s\$, the learning rate remains \$\eta\_0\$;
* When \$s \leq t < 2s\$, the learning rate becomes \$\eta\_0 \cdot \gamma\$;
* When \$2s \leq t < 3s\$, the learning rate becomes \$\eta\_0 \cdot \gamma^2\$, and so on.

## Example

If:

* \$\eta\_0 = 0.1\$
* \$\gamma = 0.5\$
* \$s = 10\$

Then:

* epoch 0–9: \$\eta\_t = 0.1\$
* epoch 10–19: \$\eta\_t = 0.05\$
* epoch 20–29: \$\eta\_t = 0.025\$
* epoch 30–39: \$\eta\_t = 0.0125\$, and so on.

---

### Python Code Example

Below is a minimal PyTorch example showing how to use **StepLR** (step decay) scheduler to dynamically adjust the learning rate in an MNIST digit classification task. The code is concise and uses the Adam optimizer.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

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

# Step 3: Initialize model, loss function, optimizer, and scheduler
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=2, gamma=0.1)  # Every 2 epochs, learning rate × 0.1

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
    print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

# Step 5: Test function
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
    scheduler.step()  # Update learning rate
```

---

### Code Explanation

1. **Model Definition**:

   * `SimpleNet` is a minimal fully connected neural network, input is a 28x28 MNIST image, output is 10 classes.

2. **Dataset**:

   * MNIST dataset is loaded using `torchvision`. Batch size is 64, preprocessing includes only tensor conversion.

3. **Learning Rate Scheduler**:

   * Using `StepLR` scheduler: `step_size=2` (adjust every 2 epochs), `gamma=0.1` (multiply learning rate by 0.1).
   * Initial learning rate `lr=0.001`, reduced to 0.0001 at epoch 3, then to 0.00001 at epoch 5.

4. **Training & Testing**:

   * During training, loss and current learning rate (`optimizer.param_groups[0]["lr"]`) are printed.
   * At the end of each epoch, call `scheduler.step()` to update learning rate.
   * Testing computes classification accuracy.

5. **Output Example**:

   ```
   Epoch 1, Loss: 0.4321, LR: 0.001000
   Test Accuracy: 92.50%
   Epoch 2, Loss: 0.2876, LR: 0.001000
   Test Accuracy: 93.80%
   Epoch 3, Loss: 0.2345, LR: 0.000100
   Test Accuracy: 94.20%
   Epoch 4, Loss: 0.2234, LR: 0.000100
   Test Accuracy: 94.30%
   Epoch 5, Loss: 0.2109, LR: 0.000010
   Test Accuracy: 94.50%
   ```

   Actual values may vary due to random initialization.

---

### Key Points

* **Scheduler Call**: `scheduler.step()` is usually called at the end of each epoch to update the learning rate.
* **StepLR**: Simple and effective, reduces learning rate every fixed number of epochs, suitable for quick experiments.
* **Learning Rate Change**: Output shows learning rate decreases from 0.001 to 0.00001, with loss stabilizing over time.

---

### Practical Applications

* **Deep Learning**: Widely used in CNNs, RNNs, Transformers (e.g., BERT).
* **Complex Models**: Combined with Adam optimizer (as mentioned earlier), schedulers can improve convergence stability.
* **With Other Regularization**: Can be combined with Dropout, BatchNorm, L2 regularization, etc.

#### Notes

* **Scheduling Strategy Choice**: StepLR is simple, while Cosine Annealing or ReduceLROnPlateau offer more flexibility.
* **Step Size and Decay Rate**: `step_size` and `gamma` need to be tuned based on the task.
* **Call Order**: `scheduler.step()` is usually called after `optimizer.step()`.



