## Famous Network: U-Net (2015)
**Proposed by**: Olaf Ronneberger et al.  
**Features**: Symmetric encoder-decoder structure with skip connections to preserve detailed information, designed specifically for image segmentation.  
**Applications**: Medical image segmentation, semantic segmentation.  
**Key Points to Master**: Encoder-decoder architecture, feature fusion.  
<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/00b53895-4271-48df-abc5-f39e671ec419" />

## Code description
The code implements a simplified but effective U-Net image segmentation model, specifically designed for the semantic segmentation task on the CIFAR-10 dataset.  
### Core Functionalities
- 1. **Model Architecture (SimpleEffectiveUNet)**  
   U-Net Structure: Classic encoder-decoder architecture with skip connections  
   Simplified Design: 32-512 channels (smaller than standard U-Net)  
   4-Layer Depth: 4 encoder layers + bottleneck layer + 4 decoder layers  
   Output: Single-channel segmentation mask (0-1 probability values)  
- 2. **Data Processing (SimpleCIFAR10Dataset)**  
   Data Source: CIFAR-10 dataset (10-class object images)  
   Mask Generation: Generates fixed-shape segmentation masks based on class labels  
   - Airplane, Cat, Horse → Elliptical shape  
   - Car, Deer, Frog, Ship, Truck → Rectangular shape  
   - Bird, Dog → Circular shape  
   Data Augmentation: Standardization processing  
- 3. **Loss Function**  
   Combined Loss: Binary Cross-Entropy (BCE) + Dice Loss (50% weight each)  
   Objective: Balance pixel-level accuracy and region-level overlap  
- 4. **Training Strategy**  
   Optimizer: Adam (learning rate 0.001)  
   Scheduler: StepLR (halves learning rate every 15 epochs)  
   Data Volume: 10,000 training samples, 2,000 test samples  
   Batch Size: 16  
   Early Stopping: Stops if no improvement after 8 epochs  
- 5. **Visualization Features**  
   Training Process: Displays prediction results every 5 epochs  
   Final Testing: Comprehensive prediction display for 12 samples  
   Training Curves: Plots of loss and Dice score changes
## Code
```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

class SimpleEffectiveUNet(nn.Module):
    """Simplified but effective U-Net model"""
    def __init__(self, in_channels=3, out_channels=1):
        super(SimpleEffectiveUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 32)
        
        # Output layer
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        """Simple convolutional block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder path
        dec4 = self.up4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.up3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output
        output = self.final(dec1)
        return torch.sigmoid(output)

class SimpleCIFAR10Dataset(Dataset):
    """Simplified CIFAR-10 segmentation dataset"""
    def __init__(self, root='./data', train=True, transform=None, max_samples=None):
        self.cifar10 = datasets.CIFAR10(root=root, train=train, download=True, transform=None)
        self.transform = transform
        self.max_samples = max_samples
        
    def __len__(self):
        if self.max_samples:
            return min(self.max_samples, len(self.cifar10))
        return len(self.cifar10)
    
    def __getitem__(self, idx):
        img, label = self.cifar10[idx]
        
        # Convert PIL image to numpy array
        img_np = np.array(img)
        
        # Generate simple segmentation mask
        mask = self.generate_simple_mask(label)
        
        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        
        # Apply transform (if any)
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, mask_tensor
    
    def generate_simple_mask(self, label):
        """Generate simple segmentation mask - based on label category"""
        mask = np.zeros((32, 32), dtype=np.float32)
        
        # Generate different shaped masks based on CIFAR-10 labels
        if label == 0:  # Airplane - Elliptical shape
            y, x = np.ogrid[:32, :32]
            mask = ((x - 16)**2 / 8**2 + (y - 16)**2 / 6**2) <= 1
        elif label == 1:  # Car - Rectangular shape
            mask[12:20, 8:24] = 1.0
        elif label == 2:  # Bird - Circular shape
            y, x = np.ogrid[:32, :32]
            mask = ((x - 16)**2 + (y - 16)**2) <= 7**2
        elif label == 3:  # Cat - Elliptical shape
            y, x = np.ogrid[:32, :32]
            mask = ((x - 16)**2 / 7**2 + (y - 16)**2 / 8**2) <= 1
        elif label == 4:  # Deer - Rectangular shape
            mask[10:22, 10:22] = 1.0
        elif label == 5:  # Dog - Circular shape
            y, x = np.ogrid[:32, :32]
            mask = ((x - 16)**2 + (y - 16)**2) <= 8**2
        elif label == 6:  # Frog - Rectangular shape
            mask[11:21, 9:23] = 1.0
        elif label == 7:  # Horse - Elliptical shape
            y, x = np.ogrid[:32, :32]
            mask = ((x - 16)**2 / 9**2 + (y - 16)**2 / 7**2) <= 1
        elif label == 8:  # Ship - Rectangular shape
            mask[10:22, 8:24] = 1.0
        elif label == 9:  # Truck - Rectangular shape
            mask[12:20, 10:22] = 1.0
        
        return mask.astype(np.float32)

def simple_bce_loss(pred, target):
    """Simple binary cross-entropy loss"""
    return F.binary_cross_entropy(pred, target)

def dice_loss_simple(pred, target, smooth=1e-6):
    """Simplified Dice loss"""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return 1 - dice

def combined_loss_simple(pred, target, alpha=0.5):
    """Simplified combined loss: BCE + Dice"""
    bce = simple_bce_loss(pred, target)
    dice = dice_loss_simple(pred, target)
    
    return alpha * bce + (1 - alpha) * dice

def dice_score_simple(pred, target, smooth=1e-6):
    """Calculate Dice coefficient"""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return dice.item()

def train_simple_unet():
    """Train the simplified U-Net model"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transformation
    transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create dataset - use more training data
    train_dataset = SimpleCIFAR10Dataset(root='./data', train=True, transform=transform, max_samples=10000)
    test_dataset = SimpleCIFAR10Dataset(root='./data', train=False, transform=transform, max_samples=2000)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    
    # Create model
    model = SimpleEffectiveUNet(in_channels=3, out_channels=1)
    model = model.to(device)
    
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer - use simpler settings
    criterion = combined_loss_simple
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Higher learning rate
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)  # Simple step scheduler
    
    # Training loop
    print("Starting training...")
    num_epochs = 30
    train_losses = []
    test_dice_scores = []
    best_dice = 0
    patience = 8
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Testing phase
        model.eval()
        total_dice = 0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                dice = dice_score_simple(outputs, targets)
                total_dice += dice
                num_batches += 1
        
        avg_dice = total_dice / num_batches
        test_dice_scores.append(avg_dice)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), 'best_simple_unet_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Test Dice: {avg_dice:.4f}')
        
        # Display some prediction results every 5 epochs
        if (epoch + 1) % 5 == 0:
            visualize_predictions_simple(model, test_dataset, device, epoch + 1)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return model, train_losses, test_dice_scores

def visualize_predictions_simple(model, test_dataset, device, epoch, num_samples=8):
    """Visualize simplified prediction results"""
    model.eval()
    
    with torch.no_grad():
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        
        for i in range(num_samples):
            input_img, target_mask = test_dataset[i]
            input_img = input_img.unsqueeze(0).to(device)
            target_mask = target_mask.unsqueeze(0).to(device)
            
            # Predict
            pred_mask = model(input_img)
            
            # Convert to numpy arrays for display
            input_np = input_img[0].cpu().numpy().transpose(1, 2, 0)
            # Denormalize
            input_np = (input_np * 0.5 + 0.5).clip(0, 1)
            target_np = target_mask[0, 0].cpu().numpy()
            pred_np = pred_mask[0, 0].cpu().numpy()
            
            # Display results
            axes[i, 0].imshow(input_np)
            axes[i, 0].set_title(f'Input Image {i+1}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(target_np, cmap='gray')
            axes[i, 1].set_title(f'Ground Truth {i+1}')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_np, cmap='gray')
            axes[i, 2].set_title(f'Prediction {i+1} (Epoch {epoch})')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'unet_simple_predictions_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.show()

def test_simple_model_comprehensive(model, num_samples=12):
    """Comprehensive testing of simplified model"""
    print("\nComprehensive testing of simplified model...")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = SimpleCIFAR10Dataset(root='./data', train=False, transform=transform, max_samples=num_samples)
    
    with torch.no_grad():
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        
        for i in range(num_samples):
            input_img, target_mask = test_dataset[i]
            input_img = input_img.unsqueeze(0).to(device)
            target_mask = target_mask.unsqueeze(0).to(device)
            
            # Predict
            pred_mask = model(input_img)
            
            # Convert to numpy arrays for display
            input_np = input_img[0].cpu().numpy().transpose(1, 2, 0)
            # Denormalize
            input_np = (input_np * 0.5 + 0.5).clip(0, 1)
            target_np = target_mask[0, 0].cpu().numpy()
            pred_np = pred_mask[0, 0].cpu().numpy()
            
            # Display results
            axes[i, 0].imshow(input_np)
            axes[i, 0].set_title(f'CIFAR-10 Image {i+1}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(target_np, cmap='gray')
            axes[i, 1].set_title(f'Ground Truth {i+1}')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_np, cmap='gray')
            axes[i, 2].set_title(f'Prediction {i+1}')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('unet_simple_comprehensive_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_training_curves_simple(train_losses, test_dice_scores):
    """Plot simplified training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training loss
    ax1.plot(train_losses, 'b-', linewidth=2)
    ax1.set_title('Simple U-Net Training Loss Curve', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Combined Loss (BCE + Dice)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Test Dice score
    ax2.plot(test_dice_scores, 'r-', linewidth=2)
    ax2.set_title('Simple U-Net Test Dice Score Curve', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Dice Score', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('unet_simple_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function"""
    print("=== Simplified Effective U-Net CIFAR-10 Implementation ===")
    
    # Train model
    model, train_losses, test_dice_scores = train_simple_unet()
    
    # Comprehensive testing of simplified model
    test_simple_model_comprehensive(model)
    
    # Plot training curves
    plot_training_curves_simple(train_losses, test_dice_scores)
    
    print("\nTraining completed!")
    print("Visualization results saved:")
    print("- unet_simple_comprehensive_predictions.png: Comprehensive prediction results")
    print("- unet_simple_training_curves.png: Training curves")
    print("- unet_simple_predictions_epoch_*.png: Prediction results during training")

if __name__ == "__main__":
    main()
```


## Training Results

Epoch: 25/30, Batch: 600/625, Loss: 0.0011  
Epoch 25/30, Train Loss: 0.0029, Test Dice: 0.9626  
Early stopping at epoch 25  

Comprehensive testing of the simplified model...  

Training completed!  
Visualization results saved:  
- unet_simple_comprehensive_predictions.png: Comprehensive prediction results  
- unet_simple_training_curves.png: Training curves  
- unet_simple_predictions_epoch_*.png: Prediction results during training  

<img width="1160" height="478" alt="image" src="https://github.com/user-attachments/assets/1a5972da-8c94-4e1d-815c-9277f41070f6" />  

<img width="1486" height="597" alt="image" src="https://github.com/user-attachments/assets/85549d73-4890-4caa-a850-42566e62fc32" />
