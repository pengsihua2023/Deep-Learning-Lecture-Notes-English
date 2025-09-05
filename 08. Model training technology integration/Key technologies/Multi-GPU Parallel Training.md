# Multi-GPU Parallel Training (Distributed Data Parallel, DDP)
## 📖 What is Distributed Data Parallel (DDP)?
**Distributed Data Parallel (DDP)** is a technique in PyTorch for distributed training, accelerating deep learning model training by processing data in parallel across multiple GPUs or machines. It is an implementation of data parallelism, suitable for large-scale models and datasets, effectively utilizing multi-GPU or distributed environments.
## 📖 Core Features:
- **Applicable Scenarios**: Suitable for large-scale deep learning tasks requiring accelerated training, such as image classification and language model training.
- **Principle**:
  - Splits the dataset into multiple subsets, with each GPU (or process) handling a portion of the data.
  - Each GPU holds a complete copy of the model, computing gradients independently.
  - Uses **AllReduce** operations to synchronize gradients across all GPUs after backpropagation, updating model parameters.
  - Employs the Ring-AllReduce algorithm to optimize communication efficiency and reduce synchronization overhead.
- **Advantages**:
  - Training speed scales near-linearly with the number of GPUs.
  - Balanced memory usage, supporting large model training.
  - Minimal code changes from single-GPU setups, easy to implement.
- **Disadvantages**:
  - Requires multi-GPU or distributed environment support.
  - Communication overhead can be a bottleneck, especially with slow networks.

## 📖 Principles of DDP
1. **Data Sharding**:
   - The dataset is divided into multiple subsets, with each GPU (process) handling a subset (mini-batch).
2. **Model Replication**:
   - Each GPU holds an identical model copy with consistent initial parameters.
3. **Parallel Computation**:
   - Each GPU independently performs forward and backward propagation, computing local gradients.
4. **Gradient Synchronization**:
   - Uses AllReduce operations (based on NCCL or MPI) to average gradients across all GPUs, ensuring consistent parameter updates.
5. **Parameter Updates**:
   - Each GPU updates model parameters using the averaged gradients, keeping models synchronized.
6. **Distributed Initialization**:
   - Initializes the communication backend (e.g., NCCL) using `torch.distributed.init_process_group` to ensure inter-process communication.
---
## 📖 Simple Code Example: DDP Training with PyTorch
Below is a simple example demonstrating how to use DDP in PyTorch to train a simple neural network on multiple GPUs using the MNIST dataset. The code assumes a single-machine multi-GPU environment.
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
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
# 2. Training Function
def train(rank, world_size):
    # Initialize distributed environment
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    
    # Set device
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Data loading
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    sampler = DistributedSampler(train_dataset)  # Distributed data sampling
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    
    # Model and optimizer
    model = SimpleNet().to(device)
    model = DDP(model, device_ids=[rank])  # Wrap as DDP model
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(2):  # 2 epochs as an example
        model.train()
        sampler.set_epoch(epoch)  # Ensure data shuffling per epoch
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Rank {rank}, Epoch {epoch + 1}, Avg Loss: {total_loss / len(train_loader):.6f}")
    
    # Cleanup
    dist.destroy_process_group()
# 3. Main Function
def main():
    world_size = torch.cuda.device_count()  # Number of GPUs
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
if __name__ == "__main__":
    # Set environment variables (single-machine multi-GPU)
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    main()
```

## 📖 Running the Code
Run in the command line:
```bash
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS your_script.py
```
Where `NUM_GPUS` is the number of available GPUs (e.g., 2). Alternatively, run the script directly (code includes `mp.spawn`).
---
### Code Explanation
1. **Model Definition**:
   - Defines a simple fully connected network for MNIST classification (input 28x28, output 10 classes).
2. **Distributed Initialization**:
   - Uses `dist.init_process_group` to initialize the distributed environment (`backend="nccl"` for GPUs).
   - `MASTER_ADDR` and `MASTER_PORT` set the communication address (localhost for single-machine).
3. **Data Loading**:
   - Uses `DistributedSampler` to automatically shard the dataset, ensuring each GPU processes a different subset.
   - `set_epoch` ensures data is shuffled randomly per epoch.
4. **DDP Model**:
   - `DDP(model, device_ids=[rank])` wraps the model for distributed parallelism, automatically synchronizing gradients.
   - Each GPU runs independent forward/backward propagation, with DDP handling gradient AllReduce.
5. **Training and Cleanup**:
   - Standard training loop computes loss and updates parameters.
   - Destroys the process group after training to release resources.

## 📖 Key Points
1. **Distributed Sampling**:
   - `DistributedSampler` ensures each GPU processes a unique data subset, avoiding duplication.
2. **Gradient Synchronization**:
   - DDP automatically synchronizes gradients after backpropagation, maintaining model consistency.
3. **Extensibility**:
   - Can combine with **AMP** (refer to previous examples, add `torch.cuda.amp` for acceleration), **Curriculum Learning** (gradually introduce complex data), or **Optuna/Ray Tune** (optimize hyperparameters).
   - The example can include `StandardScaler` for data preprocessing (refer to previous examples).
4. **Hardware Requirements**:
   - Requires multiple GPUs or a distributed cluster; NCCL backend optimizes GPU communication.

## 📖 Practical Effects
- **Training Speed**: Training time decreases near-linearly with more GPUs (e.g., 2 GPUs yield nearly 2x speedup).
- **Memory Allocation**: Each GPU processes data shards independently, balancing memory usage.
- **Accuracy Preservation**: Matches single-GPU training accuracy due to gradient synchronization ensuring model consistency.
- **Applicability**: Ideal for large-scale models (e.g., ResNet, Transformers), with significant impact in multi-GPU or cluster setups.
