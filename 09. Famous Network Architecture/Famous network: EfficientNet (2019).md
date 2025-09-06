# Famous Network: EfficientNet (2019)
**Proposed by**: Google  

<div align="center">  
<img width="220" height="213" alt="image" src="https://github.com/user-attachments/assets/5b194e36-40bf-47ca-a03a-fd24faf436ed" />  
  
First author：Mingxing Tan   
</div>

**Features**: Balances performance and efficiency through compound scaling (depth, width, resolution), suitable for resource-constrained scenarios.  
**Applications**: Efficient image classification, embedded devices.  
**Key Points to Master**: Model scaling strategies, lightweight design.  
<div align="center">
<img width="700" height="370" alt="image" src="https://github.com/user-attachments/assets/f88060f1-8ac4-4b2c-8cc7-6186af74255c" />  
</div>

<div align="center">
(This picture was obtained from the Internet.)
</div>


## Code description
This code implements a **simplified EfficientNet model** for the image classification task on the **CIFAR-10 dataset**. The main functionalities are as follows:

1. **Model Definition**:
   - Implements an `MBConv` module based on EfficientNet, including expansion convolution, depthwise separable convolution, and squeeze convolution, with support for residual connections.
   - Defines `SimpleEfficientNet`, comprising an initial convolutional layer, multiple MBConv blocks, and global average pooling, outputting 10-class classification results.

2. **Data Preprocessing**:
   - Loads the CIFAR-10 dataset (32x32 images) and applies normalization transformations.
   - Uses DataLoader for batch processing (batch_size=64).

3. **Training Process**:
   - Uses the Adam optimizer (learning rate 0.001) and cross-entropy loss function to train the model for 50 epochs.
   - Records and prints the average loss every 200 batches.

4. **Testing Process**:
   - Evaluates the model on the test set, calculating and outputting the classification accuracy.

5. **Visualization**:
   - Plots the training loss curve, saved as `efficientnet_training_curve.png`.
   - Selects 8 images from the test set, displays predicted and true labels, and saves the visualization as `efficientnet_predictions.png`, with support for Chinese display (using SimHei font).

The code runs on CPU or GPU, outputs the test set accuracy upon completion, and generates visualizations of the loss curve and prediction results to analyze model performance and classification effectiveness.
## Code
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib

# Configure Matplotlib to support Chinese display
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue

# Define MBConv block (core component of EfficientNet)
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConv, self).__init__()
        mid_channels = in_channels * expand_ratio
        self.use_residual = in_channels == out_channels and stride == 1

        layers = []
        # Expansion convolution (1x1)
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(mid_channels))
            layers.append(nn.ReLU6(inplace=True))
        
        # Depthwise separable convolution
        layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, 
                               stride=stride, padding=kernel_size//2, groups=mid_channels, bias=False))
        layers.append(nn.BatchNorm2d(mid_channels))
        layers.append(nn.ReLU6(inplace=True))
        
        # Squeeze convolution (1x1)
        layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out = out + x
        return out

# Define simplified EfficientNet model
class SimpleEfficientNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleEfficientNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            MBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1),  # 32x32
            MBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),  # 16x16
            MBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),  # Ang 8x8
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
        )
        self.classifier = nn.Sequential(
            nn.Linear(40, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 40)
        x = self.classifier(x)
        return x

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleEfficientNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Store training metrics
train_losses = []

# Training function
def train_model(epochs=50):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:
                avg_loss = running_loss / 200
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {avg_loss:.3f}')
                train_losses.append(avg_loss)
                running_loss = 0.0
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

# Visualize prediction results
def visualize_predictions():
    model.eval()
    images, labels = next(iter(testloader))  # Get a batch of test data
    images, labels = images[:8].to(device), labels[:8].to(device)  # Take 8 samples
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Denormalize images for display
    images = images.cpu() * 0.5 + 0.5  # Restore to [0,1]
    classes = ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    fig, axes = plt.subplots(1, 8, figsize=(12, 2))
    for i in range(8):
        axes[i].imshow(images[i].permute(1, 2, 0))
        axes[i].set_title(f'Predicted: {classes[predicted[i]]}\nTrue: {classes[labels[i]]}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('efficientnet_predictions.png', dpi=300, bbox_inches='tight')
    print('Prediction results saved as: efficientnet_predictions.png')
    plt.close()

# Plot training loss curve
def plot_training_curve():
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Training Batch (every 200 batches)')
    plt.ylabel('Loss')
    plt.title('Simplified EfficientNet Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('efficientnet_training_curve.png', dpi=300, bbox_inches='tight')
    print('Training loss curve saved as: efficientnet_training_curve.png')
    plt.close()

# Execute training, testing, and visualization
if __name__ == "__main__":
    train_model(epochs=50)
    test_model()
    plot_training_curve()
    visualize_predictions()
```


## Training Results

<img width="1251" height="616" alt="image" src="https://github.com/user-attachments/assets/119877a7-6a80-447d-9326-416810006a9d" />  

Figure 2: Loss Curve  

<img width="1359" height="221" alt="image" src="https://github.com/user-attachments/assets/ca55c897-aaa3-457b-9682-2cfb47dba08f" />  

Figure 3: Prediction Results  

