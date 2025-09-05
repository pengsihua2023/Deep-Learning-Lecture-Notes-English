
# Adaptive Learning Rate (Adam Optimizer)

## 📖 What is Adaptive Learning Rate (Adam Optimizer)?

The Adam optimizer (Adaptive Moment Estimation) is a widely used optimization algorithm in deep learning. It combines the advantages of Momentum and RMSProp, accelerating the convergence of gradient descent by adaptively adjusting the learning rate. It is particularly well-suited for handling sparse gradients or noisy optimization problems.

## 📖 Core Principle

Adam dynamically adjusts the learning rate of each parameter by tracking the exponential moving averages of the first moment (mean) and second moment (variance) of the gradients. The main steps are:

1. **Compute gradient**: Compute the gradient of the loss function with respect to the parameters \$g\_t\$.

2. **Update first moment (momentum)**:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

Similar to Momentum.

3. **Update second moment (variance)**:

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

Similar to RMSProp.

4. **Bias correction**: Apply bias correction to \$m\_t\$ and \$v\_t\$ to ensure unbiased estimates in the early stages.

5. **Parameter update**: Update parameters using the adaptive learning rate:

$$
\theta_{t+1} = \theta_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

* **Where**:

  * \$\eta\$: Initial learning rate (typically 0.001).
  * \$\beta\_1, \beta\_2\$: Momentum parameters (typically 0.9 and 0.999).
  * \$\epsilon\$: Small constant to prevent division by zero (typically \$1e^{-8}\$).

### 📖 Advantages

* **Adaptivity**: Automatically adjusts learning rates based on gradient history, reducing manual tuning.
* **Efficiency**: Suitable for large-scale datasets and complex models (e.g., deep neural networks).
* **Stability**: Performs well with sparse gradients or noisy optimization problems.

## 📖 Limitations

* In some tasks, it may not converge to the optimal solution as well as SGD + Momentum.
* Hyperparameters (such as \$\beta\_1, \beta\_2\$) still need to be chosen carefully.

---

## 📖 Python Code Example

Below is a simple PyTorch example that uses the Adam optimizer to train a fully connected neural network for the MNIST handwritten digit classification problem. The code focuses on the Adam implementation and is kept simple.

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



## 📖 Code Explanation

1. **Model definition**:

   * `SimpleNet` is a simple fully connected neural network. The input is a 28x28 MNIST image, and the output is 10 classes.

2. **Dataset**:

   * The MNIST dataset is loaded using `torchvision`, with batch size 64. Preprocessing only converts data into tensors.

3. **Adam Optimizer**:

   * Initialized as `optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)`.
   * `lr=0.001`: Initial learning rate, Adam’s default value usually works well.
   * `betas=(0.9, 0.999)`: Decay rates for the first and second moments, standard values.
   * `eps=1e-8`: Small constant to prevent division by zero.

4. **Training & Testing**:

   * During training, Adam adjusts the learning rate adaptively based on gradients to update model parameters.
   * Each epoch prints average loss, while testing computes classification accuracy.

5. **Example Output**:

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



## 📖 Adam vs. SGD

To demonstrate Adam’s adaptivity, here’s a comparison with SGD (without momentum):

```python
# Using SGD optimizer (without momentum)
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Replace Adam
```

* **Adam**: Adaptively adjusts learning rates, usually converges faster, less sensitive to initial learning rate.
* **SGD**: Fixed learning rate, may require manual adjustment or use of learning rate schedulers (e.g., StepLR).



## 📖 Practical Applications

* **Deep Learning**: Adam is the default optimizer for CNNs, RNNs, Transformers, etc., due to its fast convergence and stability.
* **Sparse Data**: Adam performs well with sparse gradients (e.g., NLP tasks).
* **Hyperparameter Tuning**: While Adam is not very sensitive to learning rate, parameters like `lr` and `betas` can still be tuned (e.g., via Bayesian optimization).

## 📖 Notes

* **Learning Rate**: The default `lr=0.001` usually works well, but may need fine-tuning (e.g., between 1e-4 and 1e-2) for specific tasks.
* **Convergence**: Adam may converge to suboptimal solutions in some tasks. Alternatives include SGD+Momentum or AdamW (improved Adam with L2 regularization).
* **Memory Overhead**: Adam requires storing both first and second moments, leading to slightly higher memory usage than SGD.



