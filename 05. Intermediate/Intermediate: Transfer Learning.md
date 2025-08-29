# Transfer Learning Tutorial

## Table of Contents

1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Main Approaches](#main-approaches)
4. [Application Scenarios](#application-scenarios)
5. [Advantages and Challenges](#advantages-and-challenges)
6. [Difference from Meta-Learning](#difference-from-meta-learning)
7. [PyTorch Code Example](#pytorch-code-example)

   * [Code Explanation](#code-explanation)
   * [Requirements](#requirements)
   * [Example Output](#example-output)

---

## Introduction

**Transfer Learning** is a machine learning technique where knowledge or model parameters learned from one task/domain (*source*) are applied to another related but different task/domain (*target*).

It helps improve learning efficiency and performance, especially when the target task has limited data. By leveraging pretrained models, transfer learning reduces training time and the demand for labeled datasets.
<div align="center">
<img width="700" height="350" alt="image" src="https://github.com/user-attachments/assets/148a52f4-c855-4f68-bbd0-e024055009f0" />
</div>
<div align="center">
(This picture was obtained from the Internet.)
</div>

## Core Concepts

* **Source Task and Target Task**:

  * Source task usually has large datasets and pretrained models (e.g., ImageNet pretrained models).
  * Target task has fewer data samples.
  * Transfer learning reuses source knowledge to accelerate target learning.

* **Feature Reuse**:

  * Low-level features (edges, textures) from pretrained models are general and transferable.

* **Fine-Tuning**:

  * Adjusting pretrained model parameters on the target task with slight modifications to adapt to the new domain.

---

## Main Approaches

1. **Feature Extraction**

   * Use a pretrained model as a fixed feature extractor by freezing weights.
   * Train only the new classification head for the target task.
   * **When to use**: Extremely small target datasets.

2. **Fine-Tuning**

   * Initialize the model with pretrained weights and update part or all layers.
   * **When to use**: Target task has more data and requires specific feature adaptation.

3. **Domain Adaptation**

   * Adjust the model to minimize domain gaps when source and target distributions differ (e.g., adversarial training).
   * **When to use**: Cross-domain tasks such as natural → medical images.

---

## Application Scenarios

* **Computer Vision**: Image classification, object detection using pretrained models (ResNet, VGG).
* **Natural Language Processing**: Text classification, machine translation with pretrained models (BERT, LLaMA).
* **Other Fields**: Speech recognition (acoustic models), robotics control, etc.

---

## Advantages and Challenges

### Advantages

* Reduces training time and dataset requirements.
* Improves performance on small datasets.
* Strong generalization ability by leveraging universal features.

### Challenges

* **Negative Transfer**: Large differences between tasks may reduce performance.
* **Overfitting**: Fine-tuning with limited data can cause overfitting.
* **Domain Gap**: Requires techniques to handle distribution differences between domains.

---

## Difference from Meta-Learning

* **Transfer Learning**: Focuses on reusing pretrained model knowledge; usually unidirectional (source → target).
* **Meta-Learning**: Focuses on “learning to learn,” enabling models to quickly adapt to new tasks through multi-task training.

---

## PyTorch Code Example

Here is a simple example using **ResNet18 pretrained on ImageNet** for transfer learning on the CIFAR10 dataset:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# Freeze convolutional layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer (CIFAR10 has 10 classes)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# CIFAR10 dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

# Training loop (only trains the fully connected layer)
def train_model(model, trainloader, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")

# Run training
if __name__ == "__main__":
    print("Training with Transfer Learning...")
    train_model(model, trainloader)
```

---

### Code Explanation

1. **Task**: CIFAR10 image classification using pretrained ResNet18.
2. **Model**: Freeze ResNet18 convolutional layers; replace and train the final fully connected layer.
3. **Training**: Use SGD optimizer; update only the fully connected layer’s weights.
4. **Data**: CIFAR10 dataset resized to `224x224` to match ResNet input.

---

### Requirements

* **Hardware**: GPU recommended for faster training.
* **Data**: CIFAR10 dataset is automatically downloaded.

---

### Example Output

```
Training with Transfer Learning...
Epoch 1, Loss: 1.2345
Epoch 2, Loss: 0.9876
...


