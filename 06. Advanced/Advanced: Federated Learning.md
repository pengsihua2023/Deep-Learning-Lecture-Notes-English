# Advanced: Federated Learning

Federated Learning is a distributed machine learning method designed to allow multiple devices or clients (such as mobile phones, computers, or servers) to collaboratively train a shared machine learning model without sharing raw data. It distributes the model training process across clients, and only aggregates model updates (such as gradients or parameters) on a central server, thereby protecting data privacy.
<div align="center">
<img width="600" height="314" alt="image" src="https://github.com/user-attachments/assets/93f48676-6f4c-4b71-affa-08d500cce16d" />
</div>

<div align="center">
(This picture was obtained from the Internet.)
</div>


## 📖 Core Concepts

* **Local Training**: Each client trains the model on its local dataset and generates model updates (e.g., weights or gradients).
* **Model Aggregation**: The central server collects updates from clients (without raw data) and updates the global model using weighted averaging or other methods.
* **Privacy Protection**: Raw data always remains on the client side, reducing the risk of data leakage.
* **Communication Efficiency**: Model updates need to be transmitted between clients and the server, so communication costs must be optimized.

## 📖 Main Types of Federated Learning

1. **Horizontal Federated Learning**:

   * Clients have data with the same feature space but different samples (e.g., user behavior data on different mobile phones).
   * Common in scenarios such as mobile devices and IoT.
2. **Vertical Federated Learning**:

   * Clients have data with the same samples but different features (e.g., a bank and a hospital with data on the same user).
   * Requires encryption techniques (such as secure multi-party computation) to align samples and train collaboratively.
3. **Federated Transfer Learning**:

   * Combines transfer learning to handle scenarios where both client data and feature spaces differ.

## 📖 Workflow (Example: Horizontal Federated Learning)

1. The central server initializes the global model and distributes it to clients.
2. Each client trains the model on its local data and computes updates (e.g., gradients).
3. Clients upload updates to the server (without uploading raw data).
4. The server aggregates updates (e.g., weighted averaging) to update the global model.
5. Steps 1–4 are repeated until the model converges.

## 📖 Application Scenarios

* **Mobile Devices**: e.g., predictive keyboards on smartphones (Google Gboard), training on user devices to protect input privacy.
* **Healthcare**: Collaboration between hospitals to train disease prediction models while keeping data local.
* **Finance**: Banks jointly build risk control models while protecting customer privacy.
* **IoT**: Smart devices (e.g., cameras) collaboratively optimize models.


## 📖 Mathematical Description of Federated Learning

### 1. Problem Definition

In federated learning, suppose there are \$N\$ clients (participants), each client \$i\$ holds its own local dataset:

<img width="280" height="45" alt="image" src="https://github.com/user-attachments/assets/edf86e1d-b949-46ed-bd03-196d1b51bf5a" />

The total amount of data is:

\$n = \sum\_{i=1}^N n\_i\$

We aim to learn a shared global model parameter \$\mathbf{w} \in \mathbb{R}^d\$ **without centralizing data**, such that the expected loss across all distributed data is minimized.

### 2. Global Objective Function

Define the local empirical risk function of each client:

$$
F_i(\mathbf{w}) = \frac{1}{n_i}\sum_{j=1}^{n_i}\ell\big(f(x_{i,j};\mathbf{w}),\, y_{i,j}\big)
$$

The global objective function is the weighted sum of all client losses:

$$
F(\mathbf{w}) = \sum_{i=1}^N \frac{n_i}{n} F_i(\mathbf{w})
$$

Thus, the optimization problem of federated learning is:

$$
\mathbf{w}^* = \arg\min_{\mathbf{w}} F(\mathbf{w})
$$

### 3. Optimization Process (FedAvg Algorithm)

#### 3.1 Initialization

The server initializes the global model parameters:

\$\mathbf{w}^0 \in \mathbb{R}^d\$

#### 3.2 Client Update

At round \$t\$, each selected client \$i\$ starts from \$\mathbf{w}^t\$ and performs \$E\$ steps of local stochastic gradient descent (SGD):

$$
\mathbf{w}_{i}^{t+1} \leftarrow \mathbf{w}_i^t - \eta \nabla \tilde{F}_i(\mathbf{w}_i^t;\,\xi_i^t)
$$

Where:

* \$\eta\$ is the learning rate
* \$\xi\_i^t\$ is a mini-batch sampled from \$\mathcal{D}\_i\$
* \$\nabla \tilde{F}\_i\$ represents the gradient approximation based on this mini-batch

#### 3.3 Parameter Aggregation

The server collects updates from clients and performs weighted averaging:

$$
\mathbf{w}^{t+1} = \sum_{i=1}^N \frac{n_i}{n}\,\mathbf{w}_i^{t+1}
$$

### 4. Extensions with Regularization and Noise

In practical deep learning, it is common to add **L2 regularization** (weight decay) and **noise perturbation** (for differential privacy protection or improved generalization).

#### 4.1 Local Objective with Regularization

$$
F_i^\lambda(\mathbf{w}) = F_i(\mathbf{w}) + \frac{\lambda}{2}\|\mathbf{w}\|^2
$$

#### 4.2 Update Rule with Noise

$$
\mathbf{w}_i^{t+1} = \mathbf{w}_i^t - \eta\big(\nabla \tilde{F}_i(\mathbf{w}_i^t;\xi_i^t) + \lambda \mathbf{w}_i^t + \mathbf{z}_i^t\big)
$$

Where \$\mathbf{z}\_i^t \sim \mathcal{N}(0,\sigma^2 I)\$ is noise.

### 5. Summary

The mathematical essence of federated learning is **distributed weighted empirical risk minimization**:

\$\min\_{\mathbf{w}} \sum\_{i=1}^N \frac{n\_i}{n} F\_i(\mathbf{w})\$

The typical training process is:

1. The server initializes the model;
2. Clients perform multiple SGD updates on local data;
3. The server performs weighted averaging of client models (FedAvg);
4. Repeat iterations until convergence.

---


## 📖 Code
Simple Code Example (Horizontal Federated Learning with PyTorch)
Below is a simple example of horizontal federated learning, simulating multiple clients collaboratively training a classification model.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Simulate client local training
def client_update(model, data_loader, epochs=1, lr=0.01):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        for data, target in data_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    return model.state_dict()

# Server model aggregation
def server_aggregate(global_model, client_models, client_weights):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([client_models[i][key].float() * client_weights[i] for i in range(len(client_models))]).sum(0)
    global_model.load_state_dict(global_dict)
    return global_model

# Federated learning training
def federated_learning(n_clients=10, n_rounds=10, local_epochs=1):
    # Load MNIST dataset and distribute to clients
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Simulate data partitioning (non-IID can be achieved via sorting or grouping)
    indices = np.random.permutation(len(trainset))
    client_data_size = len(trainset) // n_clients
    client_loaders = [
        DataLoader(Subset(trainset, indices[i * client_data_size:(i + 1) * client_data_size]), batch_size=32, shuffle=True)
        for i in range(n_clients)
    ]
    
    # Initialize global model
    global_model = SimpleNet()
    
    # Federated learning main loop
    for round in range(n_rounds):
        client_models = []
        client_weights = [1.0 / n_clients] * n_clients  # Assume equal weights
        
        # Client local training
        for client_id in range(n_clients):
            client_model = SimpleNet()
            client_model.load_state_dict(global_model.state_dict())
            client_model = client_update(client_model, client_loaders[client_id], epochs=local_epochs)
            client_models.append(client_model)
        
        # Server aggregation
        global_model = server_aggregate(global_model, client_models, client_weights)
        
        print(f"Round {round + 1} completed")
    
    return global_model

# Test the model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return correct / total

# Main program
if __name__ == "__main__":
    # Run federated learning
    print("Starting Federated Learning...")
    global_model = federated_learning(n_clients=10, n_rounds=10, local_epochs=1)
    
    # Test global model
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    accuracy = test_model(global_model, testloader)
    print(f"Test Accuracy: {accuracy:.4f}")
```


## 📖 Code Explanation
1. **Task**: Perform handwritten digit classification on the MNIST dataset, simulating federated learning with 10 clients.
2. **Model**: `SimpleNet` is a two-layer fully connected network used for classifying 10 digits.
3. **Federated Learning Process**:
   - Each client trains the model on a local MNIST subset, generating parameter updates.
   - The server aggregates client parameters using a weighted average to update the global model.
4. **Testing**: Evaluate the global model's accuracy on the MNIST test set.
5. **Data Allocation**: For simplicity, data is randomly allocated to clients; in real scenarios, non-IID distributions can be simulated.

### Execution Requirements
- **Hardware**: Compatible with CPU or GPU; GPU can accelerate training.
- **Data**: The code automatically downloads the MNIST dataset.

### Output Example
Upon running, the program will output something like:  

Starting Federated Learning...  
Round 1 completed  
Round 2 completed  

Test Accuracy: 0.8923  

This indicates the global model's classification accuracy on the test set.

## 📖 Extensions
- **Non-IID Data**: Simulate non-IID distributions by sorting or grouping data.
- **Privacy Enhancement**: Add differential privacy or cryptographic techniques (e.g., secure multi-party computation).
- **Complex Models**: Replace with more complex models (e.g., CNN) or use real-world datasets.
