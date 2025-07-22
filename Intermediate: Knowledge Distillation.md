## Intermediate: Knowledge Distillation

<img width="2048" height="850" alt="image" src="https://github.com/user-attachments/assets/2a79095f-2443-4a08-a7fa-a1a091bba957" />

Write a minimal PyTorch-based Knowledge Distillation example using a real dataset (MNIST handwritten digit dataset) to implement knowledge distillation from a larger teacher model (CNN) to a smaller student model (MLP). The task is digit classification, with knowledge distillation guiding the student model’s learning via the teacher model’s soft labels. Results will be demonstrated by evaluating classification accuracy and visualizing the student model’s prediction confusion matrix.  

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import torch.nn.functional as F

# Teacher model (larger CNN)
class TeacherCNN(nn.Module):
    def __init__(self):
        super(TeacherCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Student model (smaller MLP)
class StudentMLP(nn.Module):
    def __init__(self):
        super(StudentMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.fc(x)

# Knowledge distillation loss
class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        distillation_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        ce_loss = self.ce_loss(student_logits, labels)
        return self.alpha * distillation_loss + (1 - self.alpha) * ce_loss

# Load data
def get_data_loaders():
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    return train_loader, test_loader

# Train teacher model
def train_teacher(model, train_loader, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Teacher Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')

# Train student model (knowledge distillation)
def train_student(teacher, student, train_loader, epochs=5):
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    criterion = DistillationLoss(temperature=2.0, alpha=0.5)
    
    teacher.eval()
    student.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)
            loss = criterion(student_logits, teacher_logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Student Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')

# Evaluate and visualize
def evaluate_and_visualize(model, test_loader):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Compute accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f'Test Accuracy: {accuracy:.4f}')
    
    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix of Student Model on MNIST')
    plt.savefig('mnist_confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved as 'mnist_confusion_matrix.png'")
    
    # Print predictions for the first few samples
    print("\nSample Predictions (First 5):")
    for i in range(5):
        print(f"Sample {i+1}: True Label={true_labels[i]}, Predicted Label={predictions[i]}")

def main():
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    teacher = TeacherCNN().to(device)
    student = StudentMLP().to(device)
    
    # Load data
    train_loader, test_loader = get_data_loaders()
    
    # Train teacher model
    print("Training Teacher Model...")
    train_teacher(teacher, train_loader, epochs=5)
    
    # Train student model (knowledge distillation)
    print("\nTraining Student Model with Knowledge Distillation...")
    train_student(teacher, student, train_loader, epochs=5)
    
    # Evaluate student model
    print("\nEvaluating Student Model...")
    evaluate_and_visualize(student, test_loader)

if __name__ == "__main__":
    main()

```

### Code Description:
1. **Dataset**:
   - Uses the MNIST handwritten digit dataset (60,000 training samples, 10,000 test samples, 28x28 grayscale images, 10 classes).
   - Data is loaded via `torchvision` with standard normalization (`ToTensor`).

2. **Model Architecture**:
   - **Teacher Model (CNN)**: Two convolutional layers (16 and 32 channels, with ReLU and MaxPool), followed by a fully connected layer, outputting 10-dimensional classification scores.
   - **Student Model (MLP)**: Two fully connected layers (784→128→10, with ReLU), with significantly fewer parameters than the teacher model.
   - The teacher model provides soft labels, and the student model learns from both hard labels (true labels) and soft labels (teacher outputs).

3. **Knowledge Distillation Loss**:
   - Uses `DistillationLoss`, combining cross-entropy loss (hard labels) and KL divergence loss (soft labels).
   - Hyperparameters: temperature `T=2.0` (to soften probability distributions), `alpha=0.5` (to balance hard and soft label losses).

4. **Training**:
   - **Teacher Model**: Trained with cross-entropy loss for 5 epochs, using Adam optimizer with a learning rate of 0.001.
   - **Student Model**: Trained with knowledge distillation loss for 5 epochs, with the teacher model fixed and the student model learning.
   - Prints average loss per epoch.

5. **Evaluation and Visualization**:
   - **Evaluation**: Computes the classification accuracy of the student model on the test set.
   - **Visualization**: Plots a confusion matrix (10x10, showing the distribution of predicted and true labels) for the student model, saved as `mnist_confusion_matrix.png`.
   - **Prediction**: Prints the true and predicted labels for the first 5 test samples.
   - Higher values on the confusion matrix diagonal indicate better classification accuracy.

6. **Dependencies**:
   - Requires `torch`, `torchvision`, `sklearn`, `matplotlib`, and `seaborn` (`pip install torch torchvision scikit-learn matplotlib seaborn`).
   - The MNIST dataset is automatically downloaded to the `./data` directory.

### Results:
- Outputs training loss for both teacher and student models (per epoch).
- Outputs the student model's classification accuracy on the test set.
- Generates `mnist_confusion_matrix.png`, showcasing the student model's classification performance (confusion matrix).
- Prints true and predicted labels for the first 5 test samples.

### Notes:
- The confusion matrix is saved in the working directory and can be viewed with an image viewer; darker blue shades indicate higher prediction counts.
- The models are simple (teacher is a small CNN, student is a small MLP), suitable for demonstrating knowledge distillation concepts; for practical applications, consider increasing model complexity or tuning hyperparame

System: ters (e.g., temperature, alpha).
