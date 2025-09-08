# Advanced: Model pruning and quantization
## 📖 Introduction
Model Pruning and Quantization are two key techniques for optimizing deep learning models, aimed at reducing computational complexity and resource demands while maintaining performance as much as possible. These techniques are particularly important for deploying large models on resource-constrained devices (e.g., mobile devices, edge devices). Below is an overview, methods, applications, and a code example for both.


## 📖 Model Pruning

#### Introduction
Model pruning reduces model size and computational load by removing unimportant or redundant parts of a neural network (e.g., weights, neurons, or channels). The goal is to create a lighter model while retaining performance close to the original model.
<div align="center">
<img width="601" height="191" alt="image" src="https://github.com/user-attachments/assets/7935d9e0-e5b0-4d8e-b070-133892c1a078" />
</div>

<div align="center">
(This picture was obtained from the Internet.)
</div>


### Main Methods
1. **Weight Pruning**:
   - **Description**: Sets weights below a certain threshold in the weight matrix to zero, creating a sparse model.
   - **Advantages**: High flexibility, applicable to various network architectures.
   - **Disadvantages**: Sparse matrix computations require specialized hardware support.
2. **Structured Pruning**:
   - **Description**: Removes entire neurons, channels, or layers (e.g., convolutional kernels), maintaining the model’s structural integrity.
   - **Advantages**: Directly reduces computational load, easy to accelerate on hardware.
   - **Disadvantages**: May lead to more noticeable performance degradation.
3. **Iterative Pruning**:
   - **Description**: Gradually prunes and fine-tunes the model, repeating multiple times to balance compression rate and performance.
   - **Applicable Scenarios**: When high compression ratios are needed.
4. **Importance-Based Pruning**:
   - **Description**: Determines pruning targets based on the importance of weights or channels (e.g., L1 norm, gradients).
   - **Example**: Pruning low-importance weights using the L1 norm.

### Application Scenarios
- Reduces model storage requirements, facilitating deployment on edge devices.
- Lowers inference latency, suitable for real-time applications (e.g., autonomous driving).
- Reduces energy consumption, ideal for mobile devices or IoT devices.

### Challenges
- **Performance Loss**: Excessive pruning may significantly reduce model accuracy.
- **Hyperparameter Tuning**: Requires selecting appropriate pruning ratios and thresholds.
- **Hardware Adaptation**: Sparse models need specialized optimizations to achieve acceleration.



## 📖 Model Quantization

#### Introduction
Model quantization reduces model size and inference time by lowering the precision of parameters and computations (e.g., from 32-bit floating-point to 8-bit integers) while maintaining performance. Quantization is commonly used to adapt large models to resource-constrained devices.
<div align="center">
<img width="490" height="210" alt="image" src="https://github.com/user-attachments/assets/cefc6549-7e36-4aae-8f56-cfe40b4efc73" />
</div>

<div align="center">
The first quantization strategy (reduced precision)
</div>

---

<div align="center">
<img width="490" height="255" alt="image" src="https://github.com/user-attachments/assets/90beb32f-7a5b-4c26-8ad3-4b0625d635f3" />
</div>

<div align="center">   
The second quantization strategy (more lightweight model)
</div>

<div align="center">
(The above two pictures were obtained from the Internet.)
</div>

---

#### Main Methods
1. **Post-Training Quantization (PTQ)**:
   - **Description**: Applies quantization directly to a pre-trained model without retraining.
   - **Types**:
     - **Static Quantization**: Determines quantization ranges in advance (e.g., based on validation set statistics).
     - **Dynamic Quantization**: Computes quantization ranges for activations dynamically during runtime.
   - **Advantages**: Simple and fast, suitable for rapid deployment.
   - **Disadvantages**: May lead to accuracy degradation.
2. **Quantization-Aware Training (QAT)**:
   - **Description**: Simulates quantization operations (e.g., fake quantization) during training to adapt the model to low-precision computation.
   - **Advantages**: Minimizes accuracy loss, offering better performance.
   - **Disadvantages**: Higher training cost.
3. **Mixed Precision Training**:
   - **Description**: Combines high-precision (e.g., FP32) and low-precision (e.g., FP16) computations to balance performance and efficiency.
   - **Applicable Scenarios**: Scenarios with GPU/TPU support.
4. **Integer vs. Floating-Point Quantization**:
   - **Integer Quantization**: Quantizes parameters to INT8 or INT4, common for edge devices.
   - **Floating-Point Quantization**: Uses formats like FP16, suitable for high-performance hardware.

#### Application Scenarios
- Deployment on edge devices (e.g., smartphones, smart cameras).
- Accelerating real-time inference (e.g., speech recognition, recommendation systems).
- Reducing power consumption and storage requirements.

#### Challenges
- **Accuracy Loss**: Lower precision may degrade model performance.
- **Hardware Compatibility**: Requires hardware support for low-precision computation (e.g., INT8).
- **Quantization Range Selection**: Inappropriate ranges may lead to information loss.

---

## 📖 Comparison of Pruning and Quantization
- **Objective**:
  - Pruning: Reduces the number of model parameters or structural complexity.
  - Quantization: Lowers the precision of parameters and computations.
- **Applicable Scenarios**:
  - Pruning: Suitable for scenarios requiring significant model size reduction.
  - Quantization: Ideal for scenarios needing faster inference and lower power consumption.
- **Combined Use**: Pruning and quantization are often used together, with pruning reducing model size followed by quantization to lower computational precision.

---

## 📖 Simple Code Example (Pruning and Quantization with PyTorch)

Below is an example combining weight pruning and post-training quantization, optimizing a simple convolutional neural network using PyTorch on the MNIST dataset.
## Code
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define a simple convolutional neural network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc(x)
        return x

# Train the model
def train_model(model, train_loader, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Apply weight pruning
def apply_pruning(model, prune_ratio=0.5):
    # Apply L1 unstructured pruning to conv1 and conv2 layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=prune_ratio)
    return model

# Apply post-training quantization
def apply_quantization(model):
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    return model

# Test the model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Main program
if __name__ == "__main__":
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    # Initialize model
    model = ConvNet()

    # Train original model
    print("Training original model...")
    train_model(model, trainloader)

    # Test original model
    accuracy = test_model(model, testloader)
    print(f"Original Model Accuracy: {accuracy:.4f}")

    # Apply pruning
    print("Applying pruning...")
    model = apply_pruning(model, prune_ratio=0.5)

    # Fine-tune pruned model
    print("Fine-tuning pruned model...")
    train_model(model, trainloader, epochs=1)

    # Test pruned model
    accuracy_pruned = test_model(model, testloader)
    print(f"Pruned Model Accuracy: {accuracy_pruned:.4f}")

    # Apply quantization
    print("Applying quantization...")
    model = apply_quantization(model)

    # Test quantized model
    accuracy_quantized = test_model(model, testloader)
    print(f"Quantized Model Accuracy: {accuracy_quantized:.4f}")

```

## 📖 Code Explanation
1. **Task**: Perform handwritten digit classification on the MNIST dataset using a simple convolutional neural network.
2. **Model**: `ConvNet` consists of two convolutional layers and one fully connected layer.
3. **Pruning**: Use PyTorch's `prune.l1_unstructured` to apply L1 unstructured pruning to convolutional layer weights, removing 50% of the weights.
4. **Quantization**: Apply post-training quantization (PTQ) to convert the model to INT8 precision.
5. **Training and Testing**:
   - First, train the original model.
   - Apply pruning and fine-tune to recover accuracy.
   - Finally, apply quantization and test performance.

### Execution Requirements
- **Dependencies**: Install with `pip install torch torchvision`
- **Hardware**: Compatible with CPU or GPU; quantization requires hardware supporting the `fbgemm` backend.
- **Data**: The code automatically downloads the MNIST dataset.

### Output Example
Upon running, the program may output:
```
Training original model...
Epoch 1, Loss: 0.3456
...
Original Model Accuracy: 0.9820
Applying pruning...
Fine-tuning pruned model...
Epoch 1, Loss: 0.1234
Pruned Model Accuracy: 0.9750
Applying quantization...
Quantized Model Accuracy: 0.9700
```



## 📖 Summary of Advantages and Challenges
- **Advantages**:
  - **Pruning**: Significantly reduces model size and computational load, suitable for resource-constrained scenarios.
  - **Quantization**: Accelerates inference and reduces power consumption, ideal for edge devices.
- **Challenges**:
  - **Pruning**: Requires balancing compression rate and accuracy; structured pruning may necessitate network redesign.
  - **Quantization**: Lower precision may lead to performance degradation and requires hardware support.

## 📖 Relationship with Other Techniques
- **Integration with Fine-Tuning**: Pruning and quantization are often applied after fine-tuning to further optimize the model.
- **Integration with Federated Learning**: In federated learning, pruning and quantization can reduce communication and computational costs for client models.
- **Unrelated to Meta-Learning**: Meta-learning focuses on rapid adaptation to new tasks, while pruning and quantization focus on model compression and efficiency.

