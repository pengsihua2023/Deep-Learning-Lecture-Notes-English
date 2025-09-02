## CNN

Convolutional Neural Network (CNN)

* Importance: CNN is the cornerstone of computer vision, widely used in image recognition, autonomous driving, etc. It is suitable for demonstrating the practical power of deep learning.
* Core concepts:
  CNN uses the “convolution” operation, like a “magnifying glass” scanning the image to extract features (such as edges, shapes).
  The pooling layer reduces the size of data, retains important information, and decreases computation.
  Finally, the fully connected layer is used for classification or prediction.
* Applications: Image classification (cat vs. dog recognition), face recognition, medical image analysis.
<div align="center">
  <img width="708" height="353" alt="image" src="https://github.com/user-attachments/assets/c404062e-9dc5-4c41-bf8d-93cf080c6181" />
</div>

<div align="center">
(This picture was obtained from the Internet.)
</div>


## Mathematical Description of Convolutional Neural Network (CNN)

The core of CNN consists of the following basic operations: **Convolutional Layer**, **Activation Function**, **Pooling Layer**, and finally the **Fully Connected Layer**. We describe them one by one.



### 1. Convolutional Layer

Let the input feature map be

$$
\mathbf{X} \in \mathbb{R}^{H \times W \times C_{in}}
$$

where \$H\$ is the height, \$W\$ is the width, and \$C\_{in}\$ is the number of input channels.

The convolution kernel (filter) is

$$
\mathbf{K} \in \mathbb{R}^{k_h \times k_w \times C_{in} \times C_{out}}
$$

where \$k\_h, k\_w\$ are the kernel size, and \$C\_{out}\$ is the number of output channels.

The convolution operation is defined as:

$$
Y_{i,j,c_{out}} = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} \sum_{c_{in}=0}^{C_{in}-1} 
X_{i+m, j+n, c_{in}} \cdot K_{m,n,c_{in},c_{out}} + b_{c_{out}}
$$

where \$b\_{c\_{out}}\$ is the bias term. The output feature map is

$$
\mathbf{Y} \in \mathbb{R}^{H' \times W' \times C_{out}}
$$

The specific dimensions depend on stride and padding.



### 2. Activation Function

A commonly used activation function is ReLU (Rectified Linear Unit):

$$
f(z) = \max(0, z)
$$

Applied to the convolution output:

$$
Z_{i,j,c} = f(Y_{i,j,c})
$$



### 3. Pooling Layer

The pooling operation is used to reduce the size of the feature map.
For example, in Max Pooling:

$$
P_{i,j,c} = \max_{0 \leq m < p_h,  0 \leq n < p_w} Z_{i \cdot s + m,  j \cdot s + n,  c}
$$

where \$p\_h, p\_w\$ are the pooling window sizes, and \$s\$ is the stride.



### 4. Fully Connected Layer

After several layers of convolution and pooling, we obtain a flattened feature vector:

$$
\mathbf{x} \in \mathbb{R}^d
$$

The fully connected layer output is:

$$
\mathbf{y} = W \mathbf{x} + \mathbf{b}
$$

where \$W \in \mathbb{R}^{k \times d}\$, \$\mathbf{b} \in \mathbb{R}^k\$.



### 5. Classification Layer (Softmax)

In classification tasks, the final output is a probability distribution through Softmax:

![Softmax formula](https://latex.codecogs.com/png.latex?\hat{y}_i%20=%20\frac{\exp\(y_i\)}{\sum_{j=1}^{k}%20\exp\(y_j\)})

<img width="158" height="70" alt="image" src="https://github.com/user-attachments/assets/320009e3-033a-435b-99dc-7ddd3e375e15" />

---



### Code
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Set random seed to ensure reproducible results
torch.manual_seed(42)

# 1. Data Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR10 dataset
trainset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True, 
    transform=transform
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True, 
    transform=transform
)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# 2. Define Convolutional Neural Network Model
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        # Convolutional layer part
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Input 3 channels, output 32 channels
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 max pooling
        # Fully connected layer part
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # CIFAR10 images become 8x8 after two pooling operations
        self.fc2 = nn.Linear(512, 10)  # 10 classes
        self.dropout = nn.Dropout(0.5)  # Dropout to prevent overfitting
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 3. Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIFAR10CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train the model
def train_model(num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

# 5. Test the model
def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# 6. Execute training and testing
if __name__ == "__main__":
    print("Training started...")
    train_model(num_epochs=10)
    print("\nTesting started...")
    test_model()
```

### Training Results
Epoch [10/10], Step [600], Loss: 0.4115  
Epoch [10/10], Step [700], Loss: 0.4196  

Testing started...  
Test Accuracy: 74.14% 
