## Two methods for dealing with class imbalance: weighted loss function and oversampling
### 📖 What is Class Imbalance and How to Address It?
**Class imbalance** refers to an uneven distribution of classes in a dataset for classification tasks, where some classes have significantly more samples than others (e.g., 90% belong to class A, 10% to class B). This can cause models to favor the majority class, neglecting the minority class, and reducing performance. Common methods to handle class imbalance in deep learning include:
1. **Resampling**:
   - **Oversampling**: Increase minority class samples (e.g., by duplication or generating new samples like SMOTE).
   - **Undersampling**: Reduce majority class samples (e.g., by random removal).
2. **Weighted Loss Function**:
   - Assign higher loss weights to minority class samples to make the model focus more on them.
3. **Data Augmentation**:
   - Generate augmented data for the minority class (e.g., image rotations or flips).
4. **Generative Models (e.g., GANs)**:
   - Use generative adversarial networks to create new minority class samples.
5. **Adjust Classification Threshold**:
   - Adjust decision boundaries during prediction (e.g., lower the threshold for the majority class).

The following focuses on two of the most common and easy-to-implement methods: **weighted loss function** and **oversampling**, with simple code examples.


### 📖 Method 1: Weighted Loss Function
#### Principle
In the loss function (e.g., cross-entropy loss), assign different weights to classes, giving higher weights to minority class samples to encourage the model to focus on them. Weights are typically calculated as the inverse of class frequencies.

#### Code Example
Below is a PyTorch-based example showing how to use a weighted cross-entropy loss to handle class imbalance in the MNIST dataset (artificially imbalanced).

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# 1. Define the Model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 2. Create an Imbalanced Dataset (assume class 0 is 90%, others 10%)
transform = transforms.ToTensor()
train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)

# Get indices for each class
indices = []
for i in range(10):
    idx = [j for j, (_, label) in enumerate(train_dataset) if label == i]
    if i == 0:  # Class 0 (majority) keeps 90%
        indices.extend(np.random.choice(idx, int(0.9 * len(idx)), replace=False))
    else:  # Other classes (minority) keep 10%
        indices.extend(np.random.choice(idx, int(0.1 * len(idx)), replace=False))

imbalanced_dataset = Subset(train_dataset, indices)
train_loader = DataLoader(imbalanced_dataset, batch_size=64, shuffle=True)

# 3. Calculate Class Weights (inverse of class frequencies)
class_counts = np.bincount([train_dataset.targets[i] for i in indices])
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_counts)  # Normalize
class_weights = torch.FloatTensor(class_weights).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 4. Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(weight=class_weights)  # Weighted loss

# 5. Training Loop
for epoch in range(2):  # 2 epochs as an example
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Avg Loss: {total_loss / len(train_loader):.6f}")

# 6. Testing
test_loader = DataLoader(datasets.MNIST('.', train=False, transform=transform), batch_size=64)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
print(f"Test Accuracy: {correct / total * 100:.2f}%")
```

#### Code Explanation
1. **Create Imbalance**: Sample the dataset so class 0 constitutes 90% and other classes 10%, simulating an imbalanced dataset.
2. **Class Weights**: Compute weights as the inverse of class frequencies, passed to `nn.CrossEntropyLoss`’s `weight` parameter.
3. **Training**: The weighted loss function prioritizes minority classes (1-9).
4. **Evaluation**: Test model performance on the full test set.

---

### 📖 Method 2: Oversampling
#### Principle
Increase the frequency of minority class samples in training by repeatedly sampling them, balancing the dataset. This can be achieved using `WeightedRandomSampler` to sample based on weights.

#### Code Example
A PyTorch-based oversampling example, also using an artificially imbalanced MNIST dataset.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np

# 1. Define the Model (same as Method 1)
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 2. Create an Imbalanced Dataset (same as Method 1)
transform = transforms.ToTensor()
train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
indices = []
for i in range(10):
    idx = [j for j, (_, label) in enumerate(train_dataset) if label == i]
    if i == 0:
        indices.extend(np.random.choice(idx, int(0.9 * len(idx)), replace=False))
    else:
        indices.extend(np.random.choice(idx, int(0.1 * len(idx)), replace=False))
imbalanced_dataset = Subset(train_dataset, indices)

# 3. Oversampling: Calculate Sampling Weights
targets = [train_dataset.targets[i].item() for i in indices]
class_counts = np.bincount(targets)
class_weights = 1.0 / class_counts
samples_weights = [class_weights[t] for t in targets]
sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
train_loader = DataLoader(imbalanced_dataset, batch_size=64, sampler=sampler)

# 4. Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()  # No weighted loss, balanced by oversampling

# 5. Training Loop
for epoch in range(2):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Avg Loss: {total_loss / len(train_loader):.6f}")

# 6. Testing
test_loader = DataLoader(datasets.MNIST('.', train=False, transform=transform), batch_size=64)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
print(f"Test Accuracy: {correct / total * 100:.2f}%")
```

#### Code Explanation
1. **Create Imbalance**: Same as Method 1, class 0 is 90%.
2. **Oversampling**: Use `WeightedRandomSampler` with weights as inverse class frequencies to sample minority classes more frequently.
3. **Training**: The sampler balances the data distribution, so no changes to the loss function are needed.
4. **Evaluation**: Validate model performance on the test set.

---

### Key Points
1. **Weighted Loss**:
   - Simple, no dataset modification needed, adjusts weights directly in the loss function.
   - Suitable for most classification tasks but requires manual weight calculation.
2. **Oversampling**:
   - Balances data via sampling, useful for smaller datasets.
   - May lead to overfitting on minority classes, so regularization is advised.
3. **Integration with Other Techniques**:
   - Can combine with **Curriculum Learning** (train on balanced easy samples first), **AMP** (to speed up training), or **Optuna/Ray Tune** (to optimize hyperparameters).
   - The example can incorporate `torch.cuda.amp` or Optuna for learning rate tuning.

---

### Practical Effects
- **Weighted Loss**: Improves minority class (e.g., classes 1-9) prediction accuracy with minimal impact on the majority class.
- **Oversampling**: Balances training but may increase training time due to repeated minority samples.
- **Applicability**: Both methods significantly improve minority class performance when imbalance ratios are high (e.g., 10:1), typically boosting minority class accuracy by 5-20%.
