## Automatic Mixed Precision (AMP)
"Automatic Mixed Precision (AMP)" is a technique in deep learning that **automatically uses different numerical precisions (FP16 and FP32) for computation**, aiming to **accelerate training and reduce memory usage while maintaining model accuracy**.

---

### 1. Background
Common floating-point precisions in deep learning include:
* **FP32 (Single Precision Floating-Point)**: The standard format for training, offering a wide numerical range and good stability, but with slower computation and higher memory usage.
* **FP16 (Half Precision Floating-Point)**: Lower precision, faster computation, and less memory usage, but prone to overflow, underflow, or rounding errors.
Switching all computations to FP16 may cause training to fail due to these issues. AMP addresses this by **intelligently deciding which operations use FP16 and which retain FP32**.

---

### 2. Core Concept of AMP
AMP works by:
* Using **FP16 in operations suitable for it** (e.g., matrix multiplications, convolutions) → speeds up computation and saves memory.
* Retaining **FP32 in operations requiring high precision** (e.g., loss calculation, gradient accumulation, softmax, batch norm) → ensures numerical stability.
* Applying **dynamic loss scaling** to prevent underflow issues in FP16.
This achieves both speed and stability.

---

### 3. Implementation in Mainstream Frameworks
* **PyTorch**:
  Provides `torch.cuda.amp` with `autocast` and `GradScaler` for AMP.
  ```python
  scaler = torch.cuda.amp.GradScaler()
  for data, target in loader:
      optimizer.zero_grad()
      with torch.cuda.amp.autocast():
          output = model(data)
          loss = loss_fn(output, target)
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
  ```
* **TensorFlow**:
  Supports AMP via the `mixed_float16` policy.
* **NVIDIA Apex** (early implementation):
  Provided `amp.initialize` for simplified FP16 training, but PyTorch’s `torch.cuda.amp` is now the standard.

---

### 4. Advantages of AMP
- Faster training (especially on GPU Tensor Cores).
- Lower memory usage, enabling larger batch sizes or models.
- Maintains nearly identical convergence accuracy.

---

### Code Example
A minimal **PyTorch + AMP** example, demonstrating automatic mixed precision with a **fully connected network trained on MNIST**.
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. Data Loading
transform = transforms.Compose([transforms.ToTensor()])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)

# 2. Simple Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 3. AMP-Related Objects
scaler = torch.cuda.amp.GradScaler()  # Automatically scales to prevent overflow

# 4. Training Loop
for epoch in range(1, 3):  # Run 2 epochs for demonstration
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # Automatic mixed precision under autocast
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, target)
        # Scale gradients during backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if batch_idx % 200 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")
```

---

### 🔑 Key Points:
1. **`torch.cuda.amp.autocast()`**
   * Automatically selects FP16 or FP32 during forward propagation.
   * For example, convolutions and matrix multiplications use FP16 for speed and memory savings, while loss calculations remain in FP32 for stability.
2. **`torch.cuda.amp.GradScaler()`**
   * Automatically scales the loss during backpropagation to prevent gradient underflow in FP16.
3. The rest of the training process is nearly identical to standard training, requiring minimal code changes to enable AMP.

---
