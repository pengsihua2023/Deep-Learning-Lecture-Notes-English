
## Adam Variant (AdamW)

### What is Adam Variant (AdamW)?

AdamW (Adaptive Moment Estimation with Weight Decay) is a variant of the Adam optimizer that improves how Adam handles regularization, especially L2 regularization or weight decay. AdamW decouples weight decay from the adaptive learning rate, addressing the suboptimal behavior of the original Adam when applying weight decay. This makes it converge faster and generalize better in many tasks.

#### Core Principle

The Adam optimizer combines first-order momentum (mean of gradients) and second-order momentum (mean of squared gradients) to adjust the learning rate (see Adam section above). Original Adam incorporates weight decay directly into the gradient update, which is equivalent to adding an L2 regularization term to the loss function:

$$
Loss_{Adam} = Loss_{original} + \frac{\lambda}{2} \sum w_i^2
$$

However, this approach interferes with Adam’s adaptive learning rate mechanism (based on squared gradients), leading to suboptimal regularization.
AdamW decouples weight decay by directly subtracting the decay term from the parameter update step, instead of including it in the gradient calculation:

1. **Compute gradient**: Compute gradient \$g\_t\$ from the original loss function.

2. **Update first moment**:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

3. **Update second moment**:

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

4. **Bias correction**: Apply correction to \$m\_t\$ and \$v\_t\$ to remove initialization bias.

5. **Parameter update (AdamW differs here)**:

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{m_t}{\sqrt{v_t} + \epsilon} + \lambda \theta_t \right)
$$

* Where:

  * \$\eta\$: Initial learning rate (typically 0.001).
  * \$\beta\_1, \beta\_2\$: Momentum parameters (typically 0.9 and 0.999).
  * \$\epsilon\$: Small constant to prevent division by zero (typically \$1e^{-8}\$).
  * \$\lambda\$: Weight decay coefficient (set via *weight\_decay*).

AdamW directly applies decay \$\lambda \theta\_t\$ to parameters rather than incorporating it into the gradients, better balancing optimization and regularization.

#### Advantages

* **Better regularization**: Decoupled weight decay improves generalization, superior to L2 regularization in original Adam.
* **Faster convergence**: In many tasks (e.g., Transformers, CNNs), AdamW is more stable than Adam.
* **Widespread adoption**: AdamW is the default optimizer in modern deep learning (e.g., BERT, GPT).

#### Limitations

* **Hyperparameter sensitivity**: The weight decay coefficient \$\lambda\$ requires tuning.
* **Memory requirements**: Same as Adam, needs to store first and second moment estimates.

---

### Python Code Example

Below is a minimal PyTorch example demonstrating the use of AdamW for training on the MNIST handwritten digit classification task. The code is kept simple and focuses on the AdamW implementation.

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

# Step 3: Initialize model, loss function, and AdamW optimizer
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

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


### Code Explanation

1. **Model definition**:

   * `SimpleNet` is a minimal fully connected neural network, taking a 28x28 MNIST image as input and outputting 10 classes.

2. **Dataset**:

   * Loaded using `torchvision`, with batch size 64. Preprocessing only converts images to tensors.

3. **AdamW optimizer**:

   * Initialized as `optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)`.

     * `lr=0.001`: Initial learning rate, commonly used with AdamW.
     * `betas=(0.9, 0.999)`: First- and second-moment parameters, same as Adam.
     * `eps=1e-8`: Prevents division by zero.
     * `weight_decay=0.01`: Weight decay coefficient, controls regularization strength (more effective than Adam’s L2).

4. **Training & Testing**:

   * Training uses AdamW to update parameters, printing average loss.
   * Testing computes classification accuracy.

5. **Example Output**:

   ```
   Epoch 1, Loss: 0.4321, Test Accuracy: 92.50%
   Epoch 2, Loss: 0.2876, Test Accuracy: 93.80%
   Epoch 3, Loss: 0.2564, Test Accuracy: 94.20%
   Epoch 4, Loss: 0.2345, Test Accuracy: 94.50%
   Epoch 5, Loss: 0.2213, Test Accuracy: 94.70%
   ```

   Actual results may vary due to random initialization.



### Key Points

* **Decoupled weight decay**: AdamW applies decay directly on parameters (`λθ`), rather than embedding it in gradients, achieving better results than Adam’s L2 regularization.
* **Hyperparameters**:

  * `lr=0.001`: Default value usually works well.
  * `weight_decay=0.01`: Common value, tuneable depending on the task (range 1e-4 to 1e-1).
* **Compared to Adam**: AdamW generally provides better generalization, especially in Transformer models.



### Practical Applications

* **Transformer Models**: AdamW is the standard optimizer for BERT, GPT, and similar architectures due to its superior regularization.
* **Deep Learning**: Suitable for CNNs, RNNs, and other tasks, particularly when strong regularization is needed.
* **Combination with Other Techniques**: Can be used alongside Dropout, BatchNorm, ReduceLROnPlateau, etc.

#### Notes

* **Weight decay tuning**: `weight_decay` should be tuned using cross-validation or Bayesian optimization.
* **Learning rate**: AdamW is sensitive to learning rate and can be combined with learning rate schedulers (e.g., ReduceLROnPlateau).
* **Memory requirements**: Same as Adam, requires storing momentum estimates.

