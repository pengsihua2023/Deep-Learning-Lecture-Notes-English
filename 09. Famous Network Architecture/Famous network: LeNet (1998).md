## Famous network: LeNet (1998)
## Famous Network Architecture: LeNet (1998)
**Proposed by**: Yann LeCun  
<div align="center">
<img width="270" height="270" alt="image" src="https://github.com/user-attachments/assets/79c9b12d-ec4d-4f94-b441-ce9439c70eeb" />    
</div>

**Features**: One of the earliest convolutional neural networks (CNNs), consisting of convolutional layers, pooling layers, and fully connected layers, designed for handwritten digit recognition. Simple structure, foundational to CNNs.  
**Applications**: Simple image classification (e.g., MNIST dataset).  
**Key Points to Master**: Understanding convolutional operations and pooling mechanisms.  
<div align="center">
<img width="700" height="260" alt="image" src="https://github.com/user-attachments/assets/f2ccce70-ad11-40d2-bf71-3651aa4fd10b" />  
</div>

## Code description

This code implements a **LeNet model** for the handwritten digit classification task on the **MNIST dataset**. The main functionalities are as follows:

1. **Model Definition**:
   - Implements the classic LeNet convolutional neural network, consisting of 2 convolutional layers (with ReLU and max pooling) and 3 fully connected layers, outputting 10-class classification results (corresponding to MNIST digits 0-9).
   - Input is single-channel 28x28 grayscale images, and output is classification probabilities.

2. **Data Preprocessing**:
   - Loads the MNIST dataset (28x28 grayscale images) and applies normalization transformations.
   - Uses DataLoader for batch processing (batch_size=64).

3. **Training Process**:
   - Uses the SGD optimizer (learning rate 0.01, momentum 0.9) and cross-entropy loss function to train the model for 5 epochs.
   - Records and prints the average loss every 200 batches.

4. **Testing Process**:
   - Evaluates the model on the test set, calculating and outputting the classification accuracy.

5. **Visualization**:
   - Plots the training loss curve, saved as `lenet_training_curve.png`.
   - Selects 8 images from the test set, displays predicted and true labels, and saves the visualization as `lenet_predictions.png`, with support for Chinese display (using SimHei font), showing images in grayscale.

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

# Define LeNet model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),  # Input 1 channel, output 6 channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),  # Size after flattening
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),  # Output 10 classes (MNIST)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Store training metrics
train_losses = []

# Training function
def train_model(epochs=5):
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
    fig, axes = plt.subplots(1, 8, figsize=(12, 2))
    for i in range(8):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f'Predicted: {predicted[i].item()}\nTrue: {labels[i].item()}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('lenet_predictions.png', dpi=300, bbox_inches='tight')
    print('Prediction results saved as: lenet_predictions.png')
    plt.close()

# Plot training loss curve
def plot_training_curve():
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Training Batch (every 200 batches)')
    plt.ylabel('Loss')
    plt.title('LeNet Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lenet_training_curve.png', dpi=300, bbox_inches='tight')
    print('Training loss curve saved as: lenet_training_curve.png')
    plt.close()

# Execute training, testing, and visualization
if __name__ == "__main__":
    train_model(epochs=5)
    test_model()
    plot_training_curve()
    visualize_predictions()

```
### Training Results
[Epoch 5, Batch 200] Loss: 0.026  
[Epoch 5, Batch 400] Loss: 0.036  
[Epoch 5, Batch 600] Loss: 0.030  
[Epoch 5, Batch 800] Loss: 0.033  
Training completed!  
Test set accuracy: 98.84%  
Training loss curve saved as: lenet_training_curve.png  
Prediction results saved as: lenet_predictions.png  

<img width="1247" height="613" alt="image" src="https://github.com/user-attachments/assets/8c73bc50-ef8d-4d43-9e4c-e8bc89443317" />   
Figure 2: Loss Curve  

<img width="2528" height="379" alt="image" src="https://github.com/user-attachments/assets/48c22003-4e35-4df9-a438-e82280ea2be7" />  

Figure 3: Prediction Results
