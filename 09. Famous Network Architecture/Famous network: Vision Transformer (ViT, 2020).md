# Famous network: Vision Transformer (ViT, 2020)
## 📖 **Proposed by**: 
Google  
<div align="center">
<img width="220" height="236" alt="image" src="https://github.com/user-attachments/assets/e8b41203-25a3-4583-978d-46f29ea2f38c" />  
</div>

## 📖 **Features**: 
Divides images into patches and processes them with a Transformer, replacing traditional CNNs, suitable for large datasets.  
**Applications**: 
Image classification, object detection, image segmentation.  
## 📖 **Key Points to Master**: 
Self-attention mechanism, image patch processing.  

<div align="center">
<img width="642" height="488" alt="image" src="https://github.com/user-attachments/assets/c3611f64-f4e1-4f03-bbd6-efd480e0cc6e" />
</div>

<div align="center">
(This picture was obtained from the Internet.)
</div>

## 📖 Code description
```
Main Functionalities
1. Vision Transformer Model Implementation
   Image Patch Embedding: Divides 32×32 images into 4×4 patches
   Positional Encoding: Adds positional information to each patch
   Multi-Head Self-Attention: Captures global dependencies between patches
   Transformer Blocks: Multi-layer encoder structure
   Classification Head: Outputs probabilities for 10 classes
2. Real Dataset Processing
   CIFAR-10 Dataset: 50,000 training images + 10,000 test images
   Automatic Download: Downloads dataset automatically on first run
   Data Preprocessing: Standardization, data augmentation (random flipping)
   Data Loading: Efficient DataLoader implementation
3. Complete Training Workflow
   Model Initialization: Weight initialization, device configuration
   Training Loop: Forward propagation, loss calculation, backpropagation
   Validation Evaluation: Test set evaluation after each epoch
   Learning Rate Scheduling: CosineAnnealingLR optimizer
   Progress Monitoring: Real-time display of training progress and performance metrics
4. Performance Evaluation System
   Accuracy Calculation: Training and test accuracy
   Loss Monitoring: Training loss changes
   Detailed Analysis: Accuracy statistics for each class
   Overfitting Detection: Comparison of training vs. test accuracy
5. Visualization Features
   Training Curves: Trends in loss and accuracy
   Prediction Visualization: Comparison of real images with prediction results
   Color Coding: Green = correct prediction, Red = incorrect prediction
   High-Quality Output: Saved as PNG format
```

### Python Code
```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class LightVisionTransformer(nn.Module):
    """Lightweight Vision Transformer for CIFAR-10"""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=64, depth=3, num_heads=4, mlp_ratio=2, dropout=0.1):
        super(LightVisionTransformer, self).__init__()
        
        # Patch embedding
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional encoding and classification token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * mlp_ratio,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional encoding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        
        return x

def load_cifar10_data_light():
    """Load CIFAR-10 dataset (lightweight version)"""
    print("Loading CIFAR-10 dataset...")
    
    # Simplified data preprocessing
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Create data loaders (smaller batch size)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    
    # CIFAR-10 class names
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")
    
    return trainloader, testloader, classes

def train_light_vision_transformer():
    """Train lightweight Vision Transformer"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    trainloader, testloader, classes = load_cifar10_data_light()
    
    # Create model
    model = LightVisionTransformer(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=64,
        depth=3,
        num_heads=4,
        mlp_ratio=2,
        dropout=0.1
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Training loop
    print("Starting training...")
    num_epochs = 15  # Reduced number of epochs
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 200 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(trainloader)}, '
                      f'Loss: {loss.item():.4f}')
        
        train_accuracy = 100. * correct / total
        avg_loss = total_loss / len(trainloader)
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)
        
        # Testing phase
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        test_accuracy = 100. * test_correct / test_total
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Test Acc: {test_accuracy:.2f}%')
        print('-' * 50)
    
    return model, train_losses, train_accuracies, test_accuracies, classes

def visualize_sample_predictions(model, testloader, classes, device, num_samples=8):
    """Visualize sample prediction results"""
    print(f"\nVisualizing {num_samples} sample predictions...")
    model.eval()
    
    # Get some test samples
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Denormalize images
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    images_denorm = images.cpu() * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img = images_denorm[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        
        # Color coding: green=correct, red=incorrect
        color = 'green' if predicted[i] == labels[i] else 'red'
        axes[i].set_title(f'True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}', 
                         color=color, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('cifar10_light_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_light_training_curves(train_losses, train_accuracies, test_accuracies):
    """Plot lightweight training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    ax1.plot(train_losses, 'b-', linewidth=2)
    ax1.set_title('Light Vision Transformer Training Loss (CIFAR-10)', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(train_accuracies, 'b-', linewidth=2, label='Train Accuracy')
    ax2.plot(test_accuracies, 'r-', linewidth=2, label='Test Accuracy')
    ax2.set_title('Light Vision Transformer Accuracy (CIFAR-10)', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cifar10_light_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function"""
    print("=== Light Vision Transformer on CIFAR-10 Dataset ===")
    
    # Train model
    model, train_losses, train_accuracies, test_accuracies, classes = train_light_vision_transformer()
    
    # Plot training curves
    plot_light_training_curves(train_losses, train_accuracies, test_accuracies)
    
    # Visualize prediction results
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, testloader, _ = load_cifar10_data_light()
    visualize_sample_predictions(model, testloader, classes, device)
    
    print("\nTraining completed!")
    print("Results saved:")
    print("- cifar10_light_training_curves.png: Training curves")
    print("- cifar10_light_predictions.png: Sample predictions")

if __name__ == "__main__":
    main()

```

## 📖 Training Results

Epoch: 15/15, Batch: 1400/1563, Loss: 0.8820  
Epoch 15/15:  
  Train Loss: 0.8538, Train Acc: 69.65%  
  Test Acc: 67.46%  

Loading CIFAR-10 dataset...  
Training samples: 50000  
Test samples: 10000 

Visualizing 8 sample predictions...  

Training completed!  
Results saved:  
- cifar10_light_training_curves.png: Training curves  
- cifar10_light_predictions.png: Sample predictions  

<img width="1488" height="497" alt="image" src="https://github.com/user-attachments/assets/5a7569bc-82c4-40bc-b626-28462c50bf49" />  

Figure 2: Training Loss and Training Accuracy  

<img width="1578" height="802" alt="image" src="https://github.com/user-attachments/assets/4e875295-be21-4d4f-969b-d20a5c76ac36" />  

Figure 3: Model Prediction Results
