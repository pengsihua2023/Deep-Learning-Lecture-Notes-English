## Famous Network: AlexNet (2012)
**Proposed by**: Alex Krizhevsky et al.  
**Features**: Introduced ReLU activation function, Dropout regularization, data augmentation, and GPU acceleration, significantly improving performance in the ImageNet competition.  
**Applications**: Image classification, feature extraction, foundation for transfer learning.  
**Key Points to Master**: Deep CNN design, overfitting control.  
<div align="center">
<img width="700" height="380" alt="image" src="https://github.com/user-attachments/assets/5bd0deb5-051a-43ba-95f7-931fcd671b32" />
<img width="700" height="400" alt="image" src="https://github.com/user-attachments/assets/e27296cb-2aee-4389-a119-7c2ac8120d4d" />  
<img width="700" height="250" alt="image" src="https://github.com/user-attachments/assets/7b145c0e-205a-4c61-ad7f-b477203e8db6" />    
</div>

### Code
This code implements a **simplified AlexNet convolutional neural network** for the image classification task on the **CIFAR-10 dataset**. The main functionalities are as follows:

1. **Model Definition**: Implements an AlexNet model adapted for CIFAR-10, consisting of 5 convolutional layers (`features`) and 3 fully connected layers (`classifier`), using ReLU activation, max pooling, and Dropout regularization, outputting 10-class classification results.

2. **Data Preprocessing**: Loads the CIFAR-10 dataset (training and test sets), applies transformations (resizing to 32x32, normalization), and uses DataLoader for batch processing (batch_size=64).

3. **Training Process**: Uses the SGD optimizer (learning rate 0.001, momentum 0.9) and cross-entropy loss function to train the model for 30 epochs. Records average loss and accuracy every 200 batches and prints the loss.

4. **Testing Process**: Evaluates the model on the test set, calculating and outputting the classification accuracy.

5. **Visualization**: Plots the training loss curve, saved as `alexnet_training_curve.png`, with support for Chinese display (using SimHei font).

The code runs on CPU or GPU, outputs the test set accuracy upon completion, and generates a loss curve plot to analyze the training performance.
## Code
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib

# Configure Matplotlib to use fonts supporting Chinese
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue

# Define AlexNet (adapted for CIFAR-10)
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 3 * 3, 4096),  # Adapted for 3x3x256=2304
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Store training metrics
train_losses = []
train_accuracies = []

# Training function
def train_model(epochs=30):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i % 200 == 199:
                avg_loss = running_loss / 200
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {avg_loss:.3f}')
                train_losses.append(avg_loss)
                train_accuracies.append(100 * correct / total)
                running_loss = 0.0
                correct = 0
                total = 0
    print('Training completed!')

# Testing function
def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test set accuracy: {accuracy:.2f}%')
    return accuracy

# Plot training loss curve
def plot_training_curve():
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Training Batch (every 200 batches)')
    plt.ylabel('Loss')
    plt.title('Simplified AlexNet Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('alexnet_training_curve.png', dpi=300, bbox_inches='tight')
    print('Training loss curve saved as: alexnet_training_curve.png')
    plt.close()

# Execute training, testing, and plotting
if __name__ == "__main__":
    train_model(epochs=30)
    test_model()
    plot_training_curve()
```

### Training Results
[Epoch 29, Batch 600] Loss: 0.421  
[Epoch 30, Batch 200] Loss: 0.378  
[Epoch 30, Batch 400] Loss: 0.396  
[Epoch 30, Batch 600] Loss: 0.409  
Training completed!  
Test set accuracy: 77.61%  
Training loss curve saved as: alexnet_training_curve.png   
<img width="1239" height="609" alt="image" src="https://github.com/user-attachments/assets/f102aae9-d87d-4f43-bbb1-b3fac9d373b7" />  
