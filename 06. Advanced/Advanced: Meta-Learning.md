## Advanced: Meta-Learning
Meta-Learning, also known as "Learning to Learn," is a branch of machine learning aimed at enabling models to quickly adapt to new tasks. By training on multiple related tasks, the model acquires general learning strategies, allowing it to perform well on new tasks, especially with limited data.
<div align="center">
<img width="380" height="240" alt="image" src="https://github.com/user-attachments/assets/60a738f6-bea6-4738-bf3d-b2d5719c20e3" />
</div>
<div align="center">
(This picture was obtained from the Internet.)
</div>
### Core Concepts
- **Objective**: Instead of optimizing for a specific task, meta-learning focuses on learning how to learn efficiently. For example, it automatically adjusts model parameters, hyperparameters, or learning rules.
- **Key Feature**: Meta-learning typically involves training on a "task set," where each task includes a support set (for rapid learning) and a query set (for performance evaluation).

### Main Methods of Meta-Learning
1. **Optimization-Based Methods**: Such as MAML (Model-Agnostic Meta-Learning), which optimizes initial model parameters to adapt to new tasks with few gradient updates.

2. **Metric-Based Methods**: Such as Prototypical Networks, which perform classification by learning similarities between data points.
3. **Model-Based Methods**: Utilize memory mechanisms or recurrent networks to store learning experiences.

### Application Scenarios
- **Few-Shot Learning**: Completing classification or regression tasks with only a few samples.
- **Transfer Learning**: Rapid adaptation to new environments, such as robot control or personalized recommendations.
- **Hyperparameter Optimization**: Automatically tuning hyperparameters like learning rate.

### Advantages and Challenges
- **Advantages**: Enables rapid adaptation in data-scarce scenarios, mimicking human learning capabilities.
- **Challenges**: High computational cost, limited task generalization, and integration with large-scale pre-trained models requires further exploration.

Below is an example implementation of Few-Shot classification using Python, PyTorch, and Prototypical Networks. Prototypical Networks is a metric-based meta-learning method suitable for Few-Shot classification tasks. This example uses a simple 2D point classification task to demonstrate how meta-learning achieves Few-Shot classification.

```python

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.functional import pairwise_distance

# Define a simple neural network as a feature extractor
class ProtoNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super(ProtoNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Generate simple 2D classification task data
def generate_task(n_way=5, k_shot=5, k_query=15):
    # n_way: number of classes, k_shot: number of support samples per class, k_query: number of query samples per class
    centers = np.random.uniform(-5.0, 5.0, (n_way, 2))
    data = []
    labels = []
    
    for i in range(n_way):
        # Support set
        support_data = np.random.normal(centers[i], 0.5, (k_shot, 2))
        support_labels = np.full(k_shot, i)
        # Query set
        query_data = np.random.normal(centers[i], 0.5, (k_query, 2))
        query_labels = np.full(k_query, i)
        
        data.append(np.vstack([support_data, query_data]))
        labels.append(np.hstack([support_labels, query_labels]))
    
    data = np.vstack(data)
    labels = np.hstack(labels)
    
    return (torch.FloatTensor(data[:n_way*k_shot]).reshape(n_way, k_shot, 2),
            torch.LongTensor(labels[:n_way*k_shot]).reshape(n_way, k_shot),
            torch.FloatTensor(data[n_way*k_shot:]).reshape(n_way, k_query, 2),
            torch.LongTensor(labels[n_way*k_shot:]).reshape(n_way, k_query))

# Prototypical Networks loss function
def proto_loss(embeddings, labels, n_way, k_shot):
    # Compute prototype (mean) for each class
    prototypes = embeddings.view(n_way, k_shot, -1).mean(dim=1)
    query_embeddings = embeddings[k_shot*n_way:]  # Query set embeddings
    
    # Compute Euclidean distance from query samples to prototypes
    distances = pairwise_distance(
        query_embeddings.unsqueeze(1),  # [n_query, 1, hidden_dim]
        prototypes.unsqueeze(0)         # [1, n_way, hidden_dim]
    )
    
    # Compute classification loss
    log_p_y = -distances
    target = labels[k_shot*n_way:].reshape(-1)
    loss = nn.CrossEntropyLoss()(log_p_y, target)
    
    # Compute accuracy
    pred = torch.argmin(distances, dim=1)
    acc = (pred == target).float().mean()
    
    return loss, acc

# Train Prototypical Networks
def train_protonet(model, n_tasks=1000, n_way=5, k_shot=5, k_query=15, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for task_idx in range(n_tasks):
        model.train()
        optimizer.zero_grad()
        
        # Generate task data
        support_x, support_y, query_x, query_y = generate_task(n_way, k_shot, k_query)
        
        # Flatten data for model input
        support_x = support_x.view(-1, 2)
        query_x = query_x.view(-1, 2)
        all_x = torch.cat([support_x, query_x], dim=0)
        all_y = torch.cat([support_y.view(-1), query_y.view(-1)], dim=0)
        
        # Forward pass
        embeddings = model(all_x)
        
        # Compute loss and accuracy
        loss, acc = proto_loss(embeddings, all_y, n_way, k_shot)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (task_idx + 1) % 100 == 0:
            print(f"Task {task_idx + 1}, Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")
    
    return model

# Test model performance on a new task
def test_protonet(model, n_way=5, k_shot=5, k_query=15):
    model.eval()
    support_x, support_y, query_x, query_y = generate_task(n_way, k_shot, k_query)
    
    support_x = support_x.view(-1, 2)
    query_x = query_x.view(-1, 2)
    all_x = torch.cat([support_x, query_x], dim=0)
    all_y = torch.cat([support_y.view(-1), query_y.view(-1)], dim=0)
    
    with torch.no_grad():
        embeddings = model(all_x)
        _, acc = proto_loss(embeddings, all_y, n_way, k_shot)
    
    return acc.item()

# Main program
if __name__ == "__main__":
    # Initialize model
    model = ProtoNet(input_dim=2, hidden_dim=64)
    
    # Train model
    print("Training Prototypical Networks...")
    model = train_protonet(model, n_tasks=1000, n_way=5, k_shot=5, k_query=15)
    
    # Test model
    test_acc = test_protonet(model, n_way=5, k_shot=5)
    print(f"Test Accuracy on new task: {test_acc:.4f}")
```


### Code Description
1. **Task**: The code implements a simple 2D point classification task, where each task consists of `n_way` classes, with each class having `k_shot` support samples and `k_query` query samples. Data points are generated around random centers to simulate classification tasks.
2. **Model**: `ProtoNet` is a three-layer fully connected network that maps inputs to an embedding space for prototype and distance calculations.
3. **Prototypical Networks Algorithm**:
   - **Prototype Computation**: Computes the mean of the embeddings of the support set to obtain a prototype for each class.
   - **Distance Computation**: Uses Euclidean distance to calculate the distance from query samples to prototypes, with the nearest prototype determining the classification.
   - **Loss Function**: Cross-entropy loss based on distances, optimizing the embedding space.
4. **Training and Testing**:
   - During training, the model optimizes the embedding space across multiple tasks to improve prototype-based classification accuracy.
   - During testing, the model evaluates classification accuracy on new tasks.

### Execution Requirements
- Install PyTorch and NumPy: `pip install torch numpy`
- Hardware: Compatible with CPU or GPU (code not specifically optimized for GPU).
- Runtime: Training 1,000 tasks may take a few minutes, depending on the hardware.

### Sample Output
After running, the program will output something like:
```
Training Prototypical Networks...
Task 100, Loss: 0.8234, Accuracy: 0.7600
Task 200, Loss: 0.5123, Accuracy: 0.8400
...
Test Accuracy on new task: 0.8933
```
This indicates the model’s classification accuracy on a new task.

### Extensions
- **Dataset**: This example uses synthetic data but can be adapted for real datasets (e.g., Omniglot or miniImageNet).
- **Hyperparameters**: Adjust `n_way`, `k_shot`, `k_query`, or network architecture to suit different scenarios.
- **Visualization**: Add matplotlib code to visualize the embedding space or classification results.
