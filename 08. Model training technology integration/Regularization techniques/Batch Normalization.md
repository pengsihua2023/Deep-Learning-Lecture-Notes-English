
## Batch Normalization of Input
### 📖 What is Batch Normalization?

Batch Normalization (BN) is a widely used regularization technique in deep learning. By normalizing the inputs at each layer, BN accelerates training and improves model stability. BN normalizes activations within each mini-batch so that their mean is 0 and variance is 1, then applies a linear transformation with learnable scaling and shifting parameters.

### 📖 Core Principle

For the input (activation) $x$ of each layer, BN performs the following steps:

**1. Compute batch statistics:**

* Batch mean: $\mu_B = \frac{1}{m} \sum_{i=1}^m x_i$

* Batch variance: $\sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$

* where $m$ is the batch size.

**2. Normalization:**

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

* $\epsilon$ is a small constant to prevent division by zero.

**3. Scaling and shifting:**

$$
y_i = \gamma \hat{x}_i + \beta
$$

* $\gamma$ and $\beta$ are learnable parameters that control scaling and shifting.

- **Training phase**: Use the statistics (mean and variance) of the current batch for normalization.  
- **Testing phase**: Use global mean and variance accumulated during training (usually via exponential moving average).  

### 📖 Advantages
- **Accelerates training**: Reduces Internal Covariate Shift, stabilizes training, and allows higher learning rates.  
- **Regularization effect**: Like Dropout, BN introduces randomness from batches, reducing overfitting.  
- **Less initialization dependency**: Less sensitive to parameter initialization, simplifies hyperparameter tuning.  

### 📖 Limitations
- **Batch size dependency**: Small batches may cause unstable statistics; proper batch size is needed.  
- **Inference overhead**: Requires maintaining global statistics during inference, adding slight computation.  
- **Not suitable for certain tasks**: e.g., online learning or very small batch sizes.  

---

### 📖 Python Code Example

Below is a simple PyTorch example implementing Batch Normalization on the MNIST handwritten digit classification task. The code adds BN layers to a fully connected neural network, combined with Adam optimizer and early stopping (as mentioned earlier).


```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Step 1: Define a neural network with BatchNorm

class BatchNormNet(nn.Module):
def **init**(self):
super(BatchNormNet, self).**init**()
self.fc1 = nn.Linear(28 \* 28, 128)  # Input: 28x28 pixels
self.bn1 = nn.BatchNorm1d(128)      # BatchNorm layer
self.fc2 = nn.Linear(128, 10)       # Output: 10 classes
self.relu = nn.ReLU()


def forward(self, x):
    x = x.view(-1, 28 * 28)  # Flatten input
    x = self.fc1(x)
    x = self.bn1(x)          # Apply BatchNorm
    x = self.relu(x)
    x = self.fc2(x)
    return x


# Step 2: Load MNIST dataset and split into train/validation

transform = transforms.ToTensor()
train\_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test\_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
train\_size = int(0.8 \* len(train\_dataset))
val\_size = len(train\_dataset) - train\_size
train\_subset, val\_subset = torch.utils.data.random\_split(train\_dataset, \[train\_size, val\_size])
train\_loader = DataLoader(train\_subset, batch\_size=64, shuffle=True)
val\_loader = DataLoader(val\_subset, batch\_size=64)
test\_loader = DataLoader(test\_dataset, batch\_size=64)

# Step 3: Initialize model, loss function, and optimizer

model = BatchNormNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Early stopping class (reuse previous logic)

class EarlyStopping:
def **init**(self, patience=3, delta=0):
self.patience = patience
self.delta = delta
self.best\_loss = float('inf')
self.counter = 0
self.best\_model\_state = None
self.early\_stop = False


def __call__(self, val_loss, model):
    if val_loss < self.best_loss - self.delta:
        self.best_loss = val_loss
        self.best_model_state = model.state_dict()
        self.counter = 0
    else:
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True


# Step 5: Training and validation functions

def train(epoch):
model.train()  # Enable BatchNorm (training mode)
total\_loss = 0
for data, target in train\_loader:
optimizer.zero\_grad()
output = model(data)
loss = criterion(output, target)
loss.backward()
optimizer.step()
total\_loss += loss.item()
return total\_loss / len(train\_loader)

def validate():
model.eval()  # Disable BatchNorm (evaluation mode, use global statistics)
val\_loss = 0
with torch.no\_grad():
for data, target in val\_loader:
output = model(data)
val\_loss += criterion(output, target).item()
return val\_loss / len(val\_loader)

# Step 6: Testing function

def test():
model.eval()
correct = 0
total = 0
with torch.no\_grad():
for data, target in test\_loader:
output = model(data)
\_, predicted = torch.max(output.data, 1)
total += target.size(0)
correct += (predicted == target).sum().item()
return 100. \* correct / total

# Step 7: Training loop with early stopping

early\_stopping = EarlyStopping(patience=3, delta=0.001)
epochs = 20
for epoch in range(1, epochs + 1):
train\_loss = train(epoch)
val\_loss = validate()
print(f'Epoch {epoch}, Train Loss: {train\_loss:.4f}, Val Loss: {val\_loss:.4f}')


early_stopping(val_loss, model)
if early_stopping.early_stop:
    print("Early stopping triggered!")
    break


# Restore best model

if early\_stopping.best\_model\_state:
model.load\_state\_dict(early\_stopping.best\_model\_state)
print("Restored best model from early stopping.")

# Step 8: Test best model

test\_accuracy = test()
print(f'Test Accuracy: {test\_accuracy:.2f}%')

```



### 📖 Code Explanation

1. **Model definition**:  
   - `BatchNormNet` is a fully connected neural network with MNIST 28x28 pixel input and 10-class output.  
   - `nn.BatchNorm1d(128)` is added after the first fully connected layer (`fc1`) to normalize 128-dimensional activations.  

2. **Dataset**:  
   - MNIST dataset is loaded using `torchvision`, split into 80% train and 20% validation.  
   - Batch size = 64, preprocessing only converts data to tensor.  

3. **BatchNorm Layer**:  
   - `nn.BatchNorm1d(128)`: Normalizes 128-dimensional features (1D for fully connected layers; use `nn.BatchNorm2d` for conv layers).  
   - Training: Uses batch statistics (mean and variance).  
   - Testing: Uses global statistics accumulated during training (controlled by `model.eval()`).  

4. **Training and validation**:  
   - Training mode (`model.train()`): BatchNorm uses batch statistics.  
   - Validation/Testing mode (`model.eval()`): Uses global statistics.  
   - Early stopping (`EarlyStopping` class) monitors validation loss and saves the best model.  

5. **Example output**:  


Epoch 1, Train Loss: 0.2987, Val Loss: 0.1654  
Epoch 2, Train Loss: 0.1234, Val Loss: 0.1321  
Epoch 3, Train Loss: 0.0987, Val Loss: 0.1105  
Epoch 4, Train Loss: 0.0765, Val Loss: 0.1123  
Epoch 5, Train Loss: 0.0654, Val Loss: 0.1132  
Early stopping triggered!  
Restored best model from early stopping.  
Test Accuracy: 97.20%  


Actual results may vary due to random initialization.    



### 📖 Key Points
- **Placement of BatchNorm**: Usually placed after linear/conv layers and before activation functions.  
- **Training vs Testing behavior**:  
- Training: Compute batch mean and variance, update global statistics.  
- Testing: Use global statistics, disable batch statistics.  
- **Batch size**: Very small batches (<16) may cause unstable statistics; recommended batch size ≥ 32.  
- **Learnable parameters**: `nn.BatchNorm1d` automatically maintains `gamma` and `beta`, which are learned through the optimizer.  



### 📖 Practical Application Scenarios
- **Deep learning**: Widely used in CNNs (e.g., ResNet), Transformers (e.g., BERT), significantly improves training speed and stability.  
- **Combined with other regularization**: Can be used together with Dropout, L1/L2 regularization (as mentioned earlier).  
- **Large model training**: Reduces internal covariate shift when training with high learning rates or complex models.  

### 📖 Notes
- **Batch size**: Very small batch size may degrade BN performance.  
- **Alternatives**: LayerNorm, GroupNorm are better suited for small-batch or sequential tasks.  
- **Initialization**: BN is less sensitive to parameter initialization but proper setup is still required.  


