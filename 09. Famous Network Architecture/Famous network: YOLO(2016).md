# Famous Network: YOLO (2016)
**Proposed by**: Joseph Redmon et al. (YOLOv1)  
<div align="center">
<img width="220" height="220" alt="image" src="https://github.com/user-attachments/assets/4a0d197f-c041-496d-b835-620c0a0f6504" />
  
</div>

**Features**: Single-stage object detection architecture, fast speed, with versions like YOLOv5/v8 optimizing accuracy and efficiency.  
**Applications**: Real-time object detection (e.g., autonomous driving, security).  
**Key Points to Master**: Single-stage detection, anchor box mechanism.  
<div align="center">
<img width="600" height="182" alt="image" src="https://github.com/user-attachments/assets/0a0b8862-4b86-418e-b20a-4f6269e012d7" />  
</div>

<div align="center">
(This picture was obtained from the Internet.)
</div>

## Code description 

```
This simple_yolo_fixed.py file implements a minimal YOLO object detection model, designed specifically for educational demonstration and the object detection task on the CIFAR-10 dataset.
🎯 Core Functionalities
1. SimpleYOLO Model Class
   Network Architecture: 5-layer CNN feature extraction + fully connected layer output
   Grid System: 7×7 grid division
   Prediction Output: Each grid cell predicts 30 values
   - Bounding Box 1: x, y, w, h, confidence (5 values)
   - Bounding Box 2: x, y, w, h, confidence (5 values)
   - Class Probabilities: 10 CIFAR-10 classes (10 values)
   Fixed Focus: Uses a fully connected layer to transform 1×1 feature map to 7×7 grid output
2. SimpleYOLODataset Class
   Data Source: CIFAR-10 dataset
   Bounding Box Generation: Generates different-sized bounding boxes based on class
   - Airplane, Car, Ship, Truck → Larger boxes
   - Other classes → Smaller boxes
   Label Format: YOLO format (x_center, y_center, width, height)
   Grid Mapping: Automatically maps bounding boxes to the 7×7 grid system
3. Loss Function (yolo_loss)
   Bounding Box Loss: MSE loss for coordinates and confidence
   Class Loss: MSE loss for class probabilities
   Combined Loss: Weighted combination of bounding box and class losses
4. Training Function (train_simple_yolo)
   Training Parameters: 2000 training samples, 500 test samples
   Batch Size: 8
   Learning Rate: 0.001 (Adam optimizer)
   Training Epochs: 20 epochs
   Learning Rate Scheduling: Halves every 10 epochs
5. Visualization Features
   Training Process Visualization: Displays prediction results every 5 epochs
   Bounding Box Drawing: Red rectangular boxes + class labels + confidence scores
   Comprehensive Testing: Detection results for 8 samples
   Training Curves: Plot of loss function changes
```
## Code

```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import random

class SimpleYOLO(nn.Module):
    """Simplified YOLO model implementation - fixed version"""
    def __init__(self, num_classes=10, grid_size=7, num_boxes=2):
        super(SimpleYOLO, self).__init__()
        
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        
        # Dimension of predictions per grid cell:
        # 5 (x, y, w, h, confidence) * num_boxes + num_classes
        self.output_dim = (5 * num_boxes + num_classes)
        
        # Feature extraction network (fixed version - ensures 7x7 output)
        self.features = nn.Sequential(
            # First convolutional block - 32x32 -> 16x16
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second convolutional block - 16x16 -> 8x8
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third convolutional block - 8x8 -> 4x4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth convolutional block - 4x4 -> 2x2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fifth convolutional block - 2x2 -> 1x1
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Fully connected layer to transform 1x1 feature map to 7x7 grid
        self.fc = nn.Linear(256, grid_size * grid_size * self.output_dim)
        
    def forward(self, x):
        # Feature extraction
        features = self.features(x)  # Output: (batch_size, 256, 1, 1)
        
        # Flatten features
        features = features.view(features.size(0), -1)  # (batch_size, 256)
        
        # Fully connected layer
        output = self.fc(features)  # (batch_size, 7*7*30)
        
        # Reshape output to (batch_size, grid_size, grid_size, output_dim)
        output = output.view(output.size(0), self.grid_size, self.grid_size, self.output_dim)
        
        return output

class SimpleYOLODataset(Dataset):
    """Simplified YOLO dataset"""
    def __init__(self, root='./data', train=True, transform=None, max_samples=None):
        self.cifar10 = datasets.CIFAR10(root=root, train=train, download=True, transform=None)
        self.transform = transform
        self.max_samples = max_samples
        self.grid_size = 7
        self.num_classes = 10
        self.num_boxes = 2
        
        # CIFAR-10 class names
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
    def __len__(self):
        if self.max_samples:
            return min(self.max_samples, len(self.cifar10))
        return len(self.cifar10)
    
    def __getitem__(self, idx):
        img, label = self.cifar10[idx]
        
        # Convert PIL image to numpy array
        img_np = np.array(img)
        
        # Generate simplified bounding box annotation
        bbox = self.generate_simple_bbox(img_np, label)
        
        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
        
        # Apply transform (if any)
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        # Generate YOLO format label
        yolo_label = self.create_yolo_label(bbox, label)
        
        return img_tensor, yolo_label
    
    def generate_simple_bbox(self, img, label):
        """Generate simplified bounding box"""
        h, w = img.shape[:2]
        
        # Generate different sized bounding boxes based on class
        if label in [0, 1, 8, 9]:  # Airplane, Car, Ship, Truck - Larger boxes
            box_w = random.randint(w//3, w//2)
            box_h = random.randint(h//3, h//2)
        else:  # Other classes - Smaller boxes
            box_w = random.randint(w//4, w//3)
            box_h = random.randint(h//4, h//3)
        
        # Random position, ensuring box stays within image
        x_center = random.randint(box_w//2, w - box_w//2)
        y_center = random.randint(box_h//2, h - box_h//2)
        
        return [x_center, y_center, box_w, box_h]
    
    def create_yolo_label(self, bbox, class_id):
        """Create YOLO format label tensor"""
        # Create label tensor (grid_size, grid_size, 5*num_boxes + num_classes)
        label = torch.zeros(self.grid_size, self.grid_size, 
                           self.num_boxes * 5 + self.num_classes)
        
        x_center, y_center, box_w, box_h = bbox
        
        # Calculate grid coordinates
        grid_x = int(x_center * self.grid_size / 32)
        grid_y = int(y_center * self.grid_size / 32)
        
        # Ensure grid coordinates are within valid range
        grid_x = min(max(grid_x, 0), self.grid_size - 1)
        grid_y = min(max(grid_y, 0), self.grid_size - 1)
        
        # Calculate coordinates relative to grid
        x_offset = (x_center * self.grid_size / 32) - grid_x
        y_offset = (y_center * self.grid_size / 32) - grid_y
        
        # Normalize width and height
        w_norm = box_w / 32.0
        h_norm = box_h / 32.0
        
        # Set prediction for the first bounding box
        box_idx = 0
        start_idx = box_idx * 5
        
        label[grid_y, grid_x, start_idx:start_idx+4] = torch.tensor([x_offset, y_offset, w_norm, h_norm])
        label[grid_y, grid_x, start_idx+4] = 1.0  # confidence
        
        # Set class probability
        class_start_idx = self.num_boxes * 5
        label[grid_y, grid_x, class_start_idx + class_id] = 1.0
        
        return label

def yolo_loss(predictions, targets, lambda_coord=5.0, lambda_noobj=0.5):
    """Simplified YOLO loss function"""
    batch_size = predictions.size(0)
    grid_size = predictions.size(1)
    
    # Separate prediction components
    pred_boxes = predictions[:, :, :, :10]  # First 10 are bounding boxes (x, y, w, h, conf) * 2
    pred_classes = predictions[:, :, :, 10:]  # Last 10 are class probabilities
    
    # Separate target components
    target_boxes = targets[:, :, :, :10]
    target_classes = targets[:, :, :, 10:]
    
    # Calculate bounding box loss
    box_loss = F.mse_loss(pred_boxes, target_boxes, reduction='sum')
    
    # Calculate class loss
    class_loss = F.mse_loss(pred_classes, target_classes, reduction='sum')
    
    # Total loss
    total_loss = lambda_coord * box_loss + class_loss
    
    return total_loss / batch_size

def train_simple_yolo():
    """Train simplified YOLO model"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transformation
    transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create dataset
    train_dataset = SimpleYOLODataset(root='./data', train=True, transform=transform, max_samples=2000)
    test_dataset = SimpleYOLODataset(root='./data', train=False, transform=transform, max_samples=500)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    
    # Create model
    model = SimpleYOLO(num_classes=10, grid_size=7, num_boxes=2)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = yolo_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    print("Starting training...")
    num_epochs = 20
    train_losses = []
    
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
        
        # Learning rate scheduling
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}')
        
        # Display some prediction results every 5 epochs
        if (epoch + 1) % 5 == 0:
            visualize_predictions(model, test_dataset, device, epoch + 1)
    
    return model, train_losses

def visualize_predictions(model, test_dataset, device, epoch, num_samples=6):
    """Visualize YOLO prediction results"""
    model.eval()
    
    with torch.no_grad():
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
        
        for i in range(num_samples):
            input_img, target_label = test_dataset[i]
            input_img = input_img.unsqueeze(0).to(device)
            
            # Predict
            pred_output = model(input_img)
            
            # Convert to numpy arrays for display
            input_np = input_img[0].cpu().numpy().transpose(1, 2, 0)
            # Denormalize
            input_np = (input_np * 0.5 + 0.5).clip(0, 1)
            
            # Display original image
            axes[i, 0].imshow(input_np)
            axes[i, 0].set_title(f'Input Image {i+1}')
            axes[i, 0].axis('off')
            
            # Display prediction results
            axes[i, 1].imshow(input_np)
            
            # Draw predicted bounding boxes
            pred_boxes = pred_output[0, :, :, :10].cpu().numpy()  # First 10 are bounding boxes
            pred_classes = pred_output[0, :, :, 10:].cpu().numpy()  # Last 10 are classes
            
            grid_size = 7
            cell_size = 32 / grid_size
            
            for grid_y in range(grid_size):
                for grid_x in range(grid_size):
                    # Check confidence of the first bounding box
                    conf = pred_boxes[grid_y, grid_x, 4]
                    if conf > 0.3:  # Confidence threshold
                        # Get bounding box coordinates
                        x_offset = pred_boxes[grid_y, grid_x, 0]
                        y_offset = pred_boxes[grid_y, grid_x, 1]
                        w_norm = pred_boxes[grid_y, grid_x, 2]
                        h_norm = pred_boxes[grid_y, grid_x, 3]
                        
                        # Convert to pixel coordinates
                        x_center = (grid_x + x_offset) * cell_size
                        y_center = (grid_y + y_offset) * cell_size
                        width = w_norm * 32
                        height = h_norm * 32
                        
                        # Get predicted class
                        class_probs = pred_classes[grid_y, grid_x, :]
                        pred_class = np.argmax(class_probs)
                        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                                      'dog', 'frog', 'horse', 'ship', 'truck']
                        
                        # Draw bounding box
                        rect = patches.Rectangle(
                            (x_center - width/2, y_center - height/2), 
                            width, height, 
                            linewidth=2, 
                            edgecolor='red', 
                            facecolor='none'
                        )
                        axes[i, 1].add_patch(rect)
                        
                        # Add class label
                        axes[i, 1].text(
                            x_center - width/2, y_center - height/2 - 5,
                            f'{class_names[pred_class]} ({conf:.2f})',
                            color='red', fontsize=8, weight='bold'
                        )
            
            axes[i, 1].set_title(f'Prediction {i+1} (Epoch {epoch})')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'yolo_predictions_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.show()

def test_yolo_model_comprehensive(model, num_samples=8):
    """Comprehensive testing of YOLO model"""
    print("\nComprehensive testing of YOLO model...")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = SimpleYOLODataset(root='./data', train=False, transform=transform, max_samples=num_samples)
    
    with torch.no_grad():
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4*num_samples))
        
        for i in range(num_samples):
            input_img, target_label = test_dataset[i]
            input_img = input_img.unsqueeze(0).to(device)
            
            # Predict
            pred_output = model(input_img)
            
            # Convert to numpy arrays for display
            input_np = input_img[0].cpu().numpy().transpose(1, 2, 0)
            # Denormalize
            input_np = (input_np * 0.5 + 0.5).clip(0, 1)
            
            # Display image and prediction results
            axes[i].imshow(input_np)
            
            # Draw predicted bounding boxes
            pred_boxes = pred_output[0, :, :, :10].cpu().numpy()
            pred_classes = pred_output[0, :, :, 10:].cpu().numpy()
            
            grid_size = 7
            cell_size = 32 / grid_size
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                          'dog', 'frog', 'horse', 'ship', 'truck']
            
            for grid_y in range(grid_size):
                for grid_x in range(grid_size):
                    # Check confidence of the first bounding box
                    conf = pred_boxes[grid_y, grid_x, 4]
                    if conf > 0.2:  # Lower confidence threshold to show more detections
                        # Get bounding box coordinates
                        x_offset = pred_boxes[grid_y, grid_x, 0]
                        y_offset = pred_boxes[grid_y, grid_x, 1]
                        w_norm = pred_boxes[grid_y, grid_x, 2]
                        h_norm = pred_boxes[grid_y, grid_x, 3]
                        
                        # Convert to pixel coordinates
                        x_center = (grid_x + x_offset) * cell_size
                        y_center = (grid_y + y_offset) * cell_size
                        width = w_norm * 32
                        height = h_norm * 32
                        
                        # Get predicted class
                        class_probs = pred_classes[grid_y, grid_x, :]
                        pred_class = np.argmax(class_probs)
                        
                        # Draw bounding box
                        rect = patches.Rectangle(
                            (x_center - width/2, y_center - height/2), 
                            width, height, 
                            linewidth=2, 
                            edgecolor='red', 
                            facecolor='none'
                        )
                        axes[i].add_patch(rect)
                        
                        # Add class label
                        axes[i].text(
                            x_center - width/2, y_center - height/2 - 5,
                            f'{class_names[pred_class]} ({conf:.2f})',
                            color='red', fontsize=8, weight='bold'
                        )
            
            axes[i].set_title(f'YOLO Detection {i+1}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('yolo_comprehensive_detections.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_training_curves(train_losses):
    """Plot training curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.title('Simple YOLO Training Loss Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('YOLO Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('yolo_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function"""
    print("=== Simplified YOLO Model Implementation (Fixed Version) ===")
    
    # Train model
    model, train_losses = train_simple_yolo()
    
    # Comprehensive testing of model
    test_yolo_model_comprehensive(model)
    
    # Plot training curves
    plot_training_curves(train_losses)
    
    print("\nTraining completed!")
    print("Visualization results saved:")
    print("- yolo_comprehensive_detections.png: Comprehensive detection results")
    print("- yolo_training_curves.png: Training curves")
    print("- yolo_predictions_epoch_*.png: Prediction results during training")

if __name__ == "__main__":
    main()
```


## Training Results

Epoch: 20/20, Batch: 0/250, Loss: 11.6010  
Epoch: 20/20, Batch: 100/250, Loss: 10.1258  
Epoch: 20/20, Batch: 200/250, Loss: 10.2170  
Epoch 20/20, Train Loss: 10.7619  

Comprehensive testing of the YOLO model...  

Training completed!  
Visualization results saved:  
- yolo_comprehensive_detections.png: Comprehensive detection results  
- yolo_training_curves.png: Training curves  
- yolo_predictions_epoch_*.png: Prediction results during training
  
<img width="840" height="950" alt="image" src="https://github.com/user-attachments/assets/980e6e4b-f042-4193-bdec-485665ef05f2" />   

<img width="212" height="833" alt="image" src="https://github.com/user-attachments/assets/aaf3e0bb-8053-4afb-96fb-fa5dd36a82cb" />  


<img width="992" height="602" alt="image" src="https://github.com/user-attachments/assets/41c604fb-ee9a-45fa-b1d4-195cc45593f9" />   
