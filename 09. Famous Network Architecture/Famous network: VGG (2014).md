## Famous network: VGG (2014)
**Proposed by**: Oxford University VGG Team  
**Features**: Uses multiple stacked 3x3 convolutional kernels to build deep networks (16 or 19 layers), with a large number of parameters but a regular structure.  
**Applications**: Image classification, pre-trained models for object detection and segmentation.  
**Key Points to Master**: Advantages and disadvantages of deep networks, applications of pre-trained models.  
<img width="1043" height="746" alt="image" src="https://github.com/user-attachments/assets/50363707-9ec8-4de3-989f-f57b77d63465" />  

## Code description

This code implements a **simplified VGG-11 model** for the image classification task on the **CIFAR-10 dataset**. The main functionalities are as follows:

1. **Model Definition**:
   - Implements a simplified VGG-11 model with 8 convolutional layers (with ReLU activation and max pooling) and 3 fully connected layers, outputting 10-class classification results.
   - Convolutional layers progressively increase the number of channels (64→128→256→512), with max pooling for downsampling (32x32→2x2).

2. **Data Preprocessing**:
   - Loads the CIFAR-10 dataset (32x32 color images) and applies normalization transformations.
   - Uses DataLoader for batch processing (batch_size=64).

3. **Training Process**:
   - Uses the SGD optimizer (learning rate 0.001, momentum 0.9) and cross-entropy loss function to train the model for 50 epochs.
   - Records and prints the average loss every 200 batches.

4. **Testing Process**:
   - Evaluates the model on the test set, calculating and outputting the classification accuracy.

5. **Visualization**:
   - Plots the training loss curve, saved as `vgg_training_curve.png`.
   - Selects 8 images from the test set, displays predicted and true labels, and saves the visualization as `vgg_predictions.png`, with support for Chinese display (using SimHei font).

The code runs on CPU or GPU, outputs the test set accuracy upon completion, and generates visualizations of the loss curve and prediction results to analyze model performance and classification effectiveness.

## Code 
```
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

# Define simplified VGG-11 model (adapted for CIFAR-10)
class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4 -> 2x2
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 2 * 2)
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
model = VGG(num_classes=10).to(device)
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
    plt.savefig('vgg_predictions.png', dpi=300, bbox_inches='tight')
    print('Prediction results saved as: vgg_predictions.png')
    plt.close()

# Plot training loss curve
def plot_training_curve():
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Training Batch (every 200 batches)')
    plt.ylabel('Loss')
    plt.title('Simplified VGG Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('vgg_training_curve.png', dpi=300, bbox_inches='tight')
    print('Training loss curve saved as: vgg_training_curve.png')
    plt.close()

# Execute training, testing, and visualization
if __name__ == "__main__":
    train_model(epochs=50)
    test_model()
    plot_training_curve()
    visualize_predictions()
```

## Training Results
[Epoch 49, Batch 600] Loss: 0.168  
[Epoch 50, Batch 200] Loss: 0.131  
[Epoch 50, Batch 400] Loss: 0.142  
[Epoch 50, Batch 600] Loss: 0.143  
Training completed!  
Test set accuracy: 77.41%  
Training loss curve saved as: vgg_training_curve.png  
Prediction results saved as: vgg_predictions.png  

<img width="1248" height="612" alt="image" src="https://github.com/user-attachments/assets/6692f924-d6cc-4a2c-831b-7d175493e080" />  
Figure 2: Loss Curve  

<img width="1265" height="236" alt="image" src="https://github.com/user-attachments/assets/087d6d04-676d-49c0-b067-bc16b0b8778c" />  

Figure 3: Prediction Results
