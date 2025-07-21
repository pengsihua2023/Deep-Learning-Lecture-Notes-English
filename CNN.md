## CNN
**Convolutional Neural Network (CNN)**

**Importance**: CNNs are the cornerstone of computer vision, widely used in image recognition, autonomous driving, and more, ideal for demonstrating the practical power of deep learning.

**Core Concepts**:  
- CNNs use "convolution" operations, like a "magnifying glass" scanning images to extract features (e.g., edges, shapes).  
- Pooling layers reduce data size, retain key information, and decrease computational load.  
- Fully connected layers are used at the end for classification or prediction.  

**Applications**: Image classification (e.g., cat vs. dog), facial recognition, medical image analysis.


<img width="708" height="353" alt="image" src="https://github.com/user-attachments/assets/c404062e-9dc5-4c41-bf8d-93cf080c6181" />  

### Code
```
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
