# Famous network: Inception (GoogleNet, 2014)
## 📖 **Proposed by**: 
Google  
<div align="center">
<img width="200" height="250" alt="image" src="https://github.com/user-attachments/assets/c4120069-66c0-4625-b257-fc28c310bce6" />   
  
  First author：Christian Szegedy  
</div>

## 📖 **Features**: 
Inception module processes multi-scale convolutions in parallel, optimizes computational efficiency, and introduces 1x1 convolutions for dimensionality reduction.  
## 📖 **Applications**: 
Image classification, feature extraction.  
## 📖 **Key Points to Master**: 
Multi-scale feature extraction, computational efficiency optimization.  
<div align="center">
<img width="800" height="190" alt="image" src="https://github.com/user-attachments/assets/7944d7cd-6b8f-4753-a453-38146ed9b160" />  
</div>

<div align="center">
(This picture was obtained from the Internet.)
</div>


## 📖 Code Description
This code implements a **simplified Inception model** (based on the GoogLeNet Inception architecture) for the image classification task on the **CIFAR-10 dataset**. The main functionalities are as follows:

1. **Model Definition**:
   - Implements `InceptionModule`, consisting of four parallel branches: 1x1 convolution, 3x3 convolution, 5x5 convolution, and pooling followed by 1x1 convolution, with outputs concatenated to capture multi-scale features.
   - Defines `SimpleInception`, including an initial convolutional layer, two Inception modules, a max pooling layer, and a fully connected classifier, outputting 10-class classification results.

2. **Data Preprocessing**:
   - Loads the CIFAR-10 dataset (32x32 images) and applies normalization transformations.
   - Uses DataLoader for batch processing (batch_size=64).

3. **Training Process**:
   - Uses the SGD optimizer (learning rate 0.001, momentum 0.9) and cross-entropy loss function to train the model for 50 epochs.
   - Records and prints the average loss every 200 batches.

4. **Testing Process**:
   - Evaluates the model on the test set, calculating and outputting the classification accuracy.

5. **Visualization**:
   - Plots the training loss curve, saved as `inception_training_curve.png`.
   - Selects 8 images from the test set, displays predicted and true labels, and saves the visualization as `inception_predictions.png`, with support for Chinese display (using SimHei font).

The code runs on CPU or GPU, outputs the test set accuracy upon completion, and generates visualizations of the loss curve and prediction results to analyze model performance and classification effectiveness.
### Code

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

# Define simplified Inception module
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3, out_5x5, out_pool):
        super(InceptionModule, self).__init__()
        # 1x1 convolution branch
        self.branch1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        # 3x3 convolution branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # 5x5 convolution branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        # Pooling + 1x1 convolution branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)

# Define simplified Inception model
class SimpleInception(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleInception, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            InceptionModule(64, 16, 32, 16, 16),  # Output: 16+32+16+16=80
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            InceptionModule(80, 32, 64, 32, 32),  # Output: 32+64+32+32=160
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Linear(160 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 160 * 4 * 4)
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
model = SimpleInception(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

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
    plt.savefig('inception_predictions.png', dpi=300, bbox_inches='tight')
    print('Prediction results saved as: inception_predictions.png')
    plt.close()

# Plot training loss curve
def plot_training_curve():
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Training Batch (every 200 batches)')
    plt.ylabel('Loss')
    plt.title('Simplified Inception Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('inception_training_curve.png', dpi=300, bbox_inches='tight')
    print('Training loss curve saved as: inception_training_curve.png')
    plt.close()

# Execute training, testing, and visualization
if __name__ == "__main__":
    train_model(epochs=50)
    test_model()
    plot_training_curve()
    visualize_predictions()

```


## 📖 Training Results
<img width="1253" height="612" alt="image" src="https://github.com/user-attachments/assets/1edf2e07-fda4-4094-a3b7-3786cc7dc393" />  

Figure 2: Loss Curve  

<img width="1354" height="221" alt="image" src="https://github.com/user-attachments/assets/eb3a6692-2abd-448f-a01f-1428945e6a62" />  
Figure 3: Prediction Results

