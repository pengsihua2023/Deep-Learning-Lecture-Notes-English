## Advanced: Self-Supervised Learning
Self-Supervised Learning (SSL) is a machine learning paradigm that automatically generates pseudo-labels (pretext tasks) from unlabeled data for training, thereby learning useful representations without requiring manually annotated labels. Unlike traditional unsupervised learning, which typically discovers patterns directly from data (e.g., clustering), self-supervised learning is akin to "supervising" tasks it generates itself. Compared to supervised learning, it can better utilize vast amounts of unlabeled data and is widely applied in fields like computer vision and natural language processing.
<div align="center">
<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/b83a577d-dfaa-42be-b580-00ea417fb3bd" />
</div>

Contrastive Learning is a prominent type of self-supervised learning. It learns representations by pulling similar samples (positive pairs) closer together and pushing dissimilar samples (negative pairs) farther apart, as seen in algorithms like SimCLR or MoCo.

Based on mainstream classifications, self-supervised learning can be divided into several categories. These classifications are based on task design and learning mechanisms, may overlap, but help in understanding the concept.

| **Type**         | **Content**                          | **Examples/Subcategories**                  | **Applications/Example Links**         |
|-------------------|-----------------------------------|---------------------------------------------------------|---------------------------------------------------------|
| **Generative**    | Generate by reconstructing input data <br> Generate new samples to learn representations <br> Can involve encoder-decoder <br> network structures, with the goal of minimizing <br> reconstruction error. | Autoencoders, Variational <br> Autoencoders (VAEs), Masked <br> Autoencoder. | Image denoising, feature extraction, data compression; in <br> NLP such as BERT's Masked Language Modeling <br>  |
| **Contrastive**   | Learn similarity and differences through positive and negative sample pairs <br> Typically using InfoNCE loss function, <br> preventing representation collapse. | SimCLR, MoCo, CLIP (Contrastive <br> Language-Image Pretraining). | Image classification, cross-modal retrieval (e.g., text-image <br> matching); similarity learning in audio or video. <br>  |
| **Non-Contrastive** | Similar to contrastive but does not <br> rely on negative samples; uses positive <br> samples or momentum encoders to prevent representation <br> collapse. | BYOL (Bootstrap Your Own <br> Latent), SimSiam, DINO (Self-distillation <br> Unsupervised). | Visual representation learning, especially effective on small datasets; <br> avoids computational overhead from negative samples. <br>  |
| **Predictive**    | Generate representations by predicting certain <br> properties or future states of data, <br> often involving context or transformation prediction. | Image rotation prediction, colorization <br> (Colorization), motion prediction <br> (Motion Prediction); next token prediction in sequential data <br> (e.g., GPT's prediction task). | Video frame prediction, anomaly detection; future value prediction <br> in time series data. <br>  |
| **Generative-Contrastive** | Combine generative and contrastive <br> mechanisms, reconstructing data while <br> contrasting representations. | Some of the latest methods, such as generative contrastive <br> loss. | Multimodal learning (e.g., image-text pairs); complex tasks like <br> medical image analysis. <br>  |

These types are not strictly mutually exclusive, and many modern methods (e.g., in vision or NLP) combine multiple mechanisms. For instance, in research from 2024–2025, self-supervised learning is increasingly applied to multimodal settings (e.g., extensions of CLIP) and federated learning scenarios.

## Self-Supervised Learning Example
**Rotation Prediction Example**  
- **Core Idea**: Predict the rotation angle of an image.  
- **Loss Function**: Cross-entropy loss.  
- **Data Augmentation**: Fixed-angle rotations (0°, 90°, 180°, 270°).  
- **Characteristics**: Simple, intuitive, and easy to understand.

**Core Concepts of Self-Supervised Learning**  
- **Auto-Generated Labels**: Generate supervisory signals from the data itself.  
- **Pretext Tasks**: Design tasks to learn useful representations.  
- **Feature Learning**: Learn general-purpose feature representations.  
- **Downstream Tasks**: Apply learned representations to practical tasks.

## Code
```python
import os
import sys
# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['KMP_WARNINGS'] = 'off'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
# Set matplotlib backend
plt.switch_backend('Agg')

class RotationEncoder(nn.Module):
    """Rotation Prediction Encoder"""
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=4):
        super(RotationEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        # Encode and classify
        return self.encoder(x)

def create_rotation_dataset(images, labels):
    """
    Create rotation prediction dataset
    Args:
        images: Original images
        labels: Original labels
    Returns:
        rotated_images: Rotated images
        rotation_labels: Rotation angle labels (0, 90, 180, 270 degrees)
    """
    rotated_images = []
    rotation_labels = []
    
    for image in images:
        # Create 4 rotated versions
        for rotation in range(4):
            angle = rotation * 90
            # Use torch rotation function
            if angle == 0:
                rotated_img = image
            elif angle == 90:
                rotated_img = torch.rot90(image, k=1, dims=[1, 2])
            elif angle == 180:
                rotated_img = torch.rot90(image, k=2, dims=[1, 2])
            elif angle == 270:
                rotated_img = torch.rot90(image, k=3, dims=[1, 2])
            
            rotated_images.append(rotated_img)
            rotation_labels.append(rotation)
    
    return torch.stack(rotated_images), torch.tensor(rotation_labels)

def train_rotation_prediction(model, train_loader, num_epochs=10, device='cpu'):
    """Train rotation prediction model"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    
    train_losses = []
    train_accuracies = []
    
    print("Starting rotation prediction training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            
            # Create rotation dataset
            rotated_images, rotation_labels = create_rotation_dataset(images, torch.zeros(len(images)))
            rotated_images = rotated_images.to(device)
            rotation_labels = rotation_labels.to(device)
            
            # Forward pass
            outputs = model(rotated_images)
            
            # Compute loss
            loss = criterion(outputs, rotation_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == rotation_labels).sum().item()
            total_samples += rotation_labels.size(0)
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                accuracy = 100.0 * total_correct / total_samples
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {accuracy:.2f}%')
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        avg_accuracy = 100.0 * total_correct / total_samples
        train_losses.append(avg_loss)
        train_accuracies.append(avg_accuracy)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}, Average Acc: {avg_accuracy:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}')
    
    return train_losses, train_accuracies

def evaluate_rotation_representation_quality(model, test_loader, device='cpu'):
    """Evaluate the quality of learned representations from rotation prediction"""
    model.eval()
    
    # Extract features (excluding the final classification layer)
    features = []
    labels = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            # Extract features (excluding the final classification layer)
            x = images.view(images.size(0), -1)
            x = model.encoder[0:6](x)  # Use only the encoder part, excluding the final classification layer
            features.append(x.cpu())
            labels.append(targets)
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # Train linear classifier
    linear_classifier = nn.Sequential(
        nn.Linear(features.size(1), 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 10)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(linear_classifier.parameters(), lr=0.01, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Split into training and validation sets
    split_idx = int(0.8 * len(features))
    train_features = features[:split_idx].to(device)
    train_labels = labels[:split_idx].to(device)
    val_features = features[split_idx:].to(device)
    val_labels = labels[split_idx:].to(device)
    
    best_accuracy = 0.0
    
    # Train linear classifier
    for epoch in range(20):
        linear_classifier.train()
        optimizer.zero_grad()
        outputs = linear_classifier(train_features)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 5 == 0:
            # Validation
            linear_classifier.eval()
            with torch.no_grad():
                val_outputs = linear_classifier(val_features)
                val_loss = criterion(val_outputs, val_labels)
                _, predicted = torch.max(val_outputs, 1)
                accuracy = (predicted == val_labels).float().mean()
                best_accuracy = max(best_accuracy, accuracy.item())
                print(f'Linear Classifier Epoch {epoch}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.4f}')
    
    return best_accuracy

def plot_rotation_training_curves(losses, accuracies):
    """Plot rotation prediction training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    ax1.plot(losses, 'b-o', linewidth=2, markersize=6)
    ax1.set_title('Rotation Prediction Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(accuracies, 'r-o', linewidth=2, markersize=6)
    ax2.set_title('Rotation Prediction Training Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('rotation_prediction_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)  # Reduce batch size
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Create rotation prediction model
    model = RotationEncoder(input_dim=784, hidden_dim=256, num_classes=4)
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train rotation prediction model
    print("Starting rotation prediction self-supervised learning training...")
    train_losses, train_accuracies = train_rotation_prediction(model, train_loader, num_epochs=10, device=device)
    
    # Plot training curves
    plot_rotation_training_curves(train_losses, train_accuracies)
    
    # Evaluate representation quality
    print("Evaluating learned representation quality...")
    accuracy = evaluate_rotation_representation_quality(model, test_loader, device=device)
    print(f"Rotation prediction learned representation - Linear classifier accuracy: {accuracy:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'rotation_prediction_model.pth')
    print("Rotation prediction model saved to: rotation_prediction_model.pth")
    
    print("Rotation prediction self-supervised learning completed!")
    
    # Clean up memory
    del model, train_loader, test_loader
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    main()
```

## Training Results
Epoch 10/10, Batch 850, Loss: 0.0013, Acc: 99.67%  
Epoch 10/10, Batch 900, Loss: 0.0102, Acc: 99.67%  
Epoch 10/10, Average Loss: 0.0091, Average Acc: 99.68%, LR: 0.000490  
Evaluating learned representation quality...  
Linear Classifier Epoch 0, Val Loss: 4.5848, Val Acc: 0.1250  
Linear Classifier Epoch 5, Val Loss: 2.3340, Val Acc: 0.1815  
Linear Classifier Epoch 10, Val Loss: 2.1722, Val Acc: 0.1700  
Linear Classifier Epoch 15, Val Loss: 2.1008, Val Acc: 0.2825  
Rotation prediction learned representation - Linear classifier accuracy: 0.2825  
Rotation prediction model saved to: rotation_prediction_model.pth  
Rotation prediction self-supervised learning completed!
