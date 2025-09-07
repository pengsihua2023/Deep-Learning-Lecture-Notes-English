# Contrastive Learning
## 📖 Introduction
**Contrastive Learning** is a type of **Self-Supervised Learning** method. It constructs contrastive tasks between samples, enabling the model to learn more discriminative feature representations.

The core idea is:

* Pull “similar” samples closer (smaller embedding distance),
* Push “dissimilar” samples apart (larger embedding distance).

This learning approach usually does not require manual labels. Instead, it automatically generates **positive pairs** and **negative pairs** through data augmentation or contextual information.  
<div align="center">  
<img width="560" height="390" alt="image" src="https://github.com/user-attachments/assets/4f14b5b3-0951-4417-b7f1-4a00f3b39683" />
</div>

## 📖 Formal Definition

Given a sample \$x\$, two views \$x\_i, x\_j\$ are obtained via data augmentation (such as image rotation, cropping, noise injection, etc.). They are regarded as **positive pairs**; views from other samples (such as \$x\_k\$) are **negative samples**.

The objective is to learn an encoder \$f(\cdot)\$ that maps samples to the feature space, such that:

* **Positive pairs** are as close as possible in the feature space:

  \$\text{sim}(f(x\_i), f(x\_j)) \ \text{maximize}\$

* **Negative pairs** are as far apart as possible in the feature space:

  \$\text{sim}(f(x\_i), f(x\_k)) \ \text{minimize}\$

Here, \$\text{sim}(\cdot,\cdot)\$ is usually **cosine similarity** or **inner product**.

## 📖 Typical Methods

* **SimCLR**: Constructs positive and negative pairs through large-scale data augmentation and trains with the InfoNCE loss.
* **MoCo** (Momentum Contrast): Introduces a momentum update mechanism to maintain a large dynamic negative sample queue.
* **BYOL** (Bootstrap Your Own Latent): Does not explicitly use negative samples, but learns through the interaction of two networks (online network & target network).

## 📖 Summary

The essence of contrastive learning is:

* It does not rely on large amounts of manual labels,
* It learns feature representations by “bringing similar samples closer, separating dissimilar samples,”
* It has wide applications in computer vision, natural language processing, speech, etc.

<div align="center">
<img width="420" height="250" alt="image" src="https://github.com/user-attachments/assets/5d389da9-c6c7-46d5-a1c5-096422a5328b" />
</div>

<div align="center">
(This picture was obtained from the Internet.)
</div>


* Importance:
  Contrastive learning is a self-supervised learning method that extracts high-quality feature representations by teaching models to distinguish between “similar” and “dissimilar” data pairs.
  It is the core of modern unsupervised learning, driving successes such as SimCLR, MoCo (computer vision), and CLIP (multimodal learning).
  In scenarios with scarce labeled data (e.g., medical imaging, low-resource languages), contrastive learning can significantly reduce reliance on manual labeling.

* Core Concept:
  The goal of contrastive learning is to bring similar data pairs (positive pairs) closer together in feature space, while pushing dissimilar data pairs (negative pairs) farther apart.
  It optimizes feature representations using contrastive loss functions (e.g., InfoNCE loss).

* Analogy: Like a “find your friends game,” the model learns to cluster “friends” (similar images/texts) together while separating “strangers” (dissimilar data).

* Applications:
  Image classification (SimCLR, MoCo): Achieve high-accuracy classification with limited labeled data.
  Multimodal learning (CLIP): Image-text retrieval, image generation (e.g., DALL·E).


## 📖 Mathematical Description of Contrastive Learning
The mathematical description of contrastive learning usually starts from the objective of "representation learning." The core idea is: **bring semantically similar samples closer and push semantically dissimilar samples apart**. Below is a more systematic mathematical formalization:

### 1. Representation Function

Suppose we have a sample set $\mathcal{X} = \{x_1, x_2, \dots, x_N\}$  

We use an encoder (e.g., a neural network) $f_\theta: \mathcal{X} \to \mathbb {R}^d$  

to map samples into a feature space: $z_i = f_\theta(x_i), \quad z_i \in \mathbb{R}^d$  

Normalization constraint $\|z_i\|_2 = 1$ is usually applied so that the representations lie on the unit hypersphere.  

### 2. Positive and Negative Samples

* **Positive pair**: comes from the same semantic category or different augmented (data augmentation) versions of the same sample, e.g. $(x_i, x_j^+)$.
* **Negative pair**: comes from different semantic categories, e.g. $(x_i, x_k^-)$.

### 3. Similarity Measure

A common choice is cosine similarity:

$\text{sim}(z_i, z_j) = \frac{z_i^\top z_j}{\|z_i\|\|z_j\|}$

If normalized, this simplifies to $\text{sim}(z_i, z_j) = z_i^\top z_j$.

### 4. Loss Function (InfoNCE Example)

The commonly used objective in contrastive learning is the **InfoNCE loss**.  
Let the positive sample for the $i$-th instance be $z_j^+$ and the others be negative samples $\{z_k^-\}$, then the loss is:

<img width="258" height="71" alt="image" src="https://github.com/user-attachments/assets/c8502645-6c4c-4ae2-b370-07e2d4d0e73d" />

where:

* $\tau > 0$ is the **temperature parameter**, controlling the smoothness of the distribution;
* The numerator corresponds to the **positive pair**;
* The denominator includes all candidates (positive + negative), usually with **softmax normalization**.

### 5. Overall Loss

For a batch of samples, take the average:

$\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \mathcal{L}_i$

### 6. Summary

* The core mathematical formulation of contrastive learning is the **normalized softmax log-likelihood objective (InfoNCE)**;
* Optimization goal: maximize similarity of positive pairs, minimize similarity of negative pairs;
* Extensions: **NT-Xent loss** (SimCLR), **Triplet Loss**, **Margin Loss**, etc.

---

## 📖 Code
Here is a minimal PyTorch-based Contrastive Learning example using a real dataset (MNIST handwritten digit dataset) to implement contrastive learning for learning image feature embeddings. The model will use a SimCLR-style contrastive loss (NT-Xent), aiming to make embeddings of images of the same digit closer and embeddings of different digits farther apart. Results will be demonstrated by visualizing the embedding space (using t-SNE dimensionality reduction) and evaluating k-NN classification accuracy.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Define a simple convolutional network
class SimpleCNN(nn.Module):
    def __init__(self, embed_dim=128):
        super(SimpleCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, embed_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)

# Contrastive loss (NT-Xent)
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, z1, z2, batch_size):
        # Normalize embeddings
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.transpose(0, 1)) / self.temperature
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(z.device)
        
        # Mask to remove self-similarity
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)
        
        # Compute loss
        loss = self.criterion(sim_matrix, labels.argmax(dim=1))
        return loss

# Data augmentation
def get_data_loaders():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True,
                                 transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Original data for evaluation (no augmentation)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True,
                                transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    return train_loader, test_loader

# Train the model
def train_model(model, train_loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = NTXentLoss(temperature=0.5)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            # Create two augmented views
            aug1 = transforms.RandomResizedCrop(28, scale=(0.8, 1.0))(images)
            aug2 = transforms.RandomRotation(10)(images)
            
            optimizer.zero_grad()
            z1 = model(aug1)
            z2 = model(aug2)
            loss = criterion(z1, z2, images.size(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')

# Evaluate and visualize
def evaluate_and_visualize(model, test_loader):
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            emb = model(images)
            embeddings.append(emb.cpu().numpy())
            labels.append(targets.numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings[:1000])  # Use first 1000 samples
    
    # Visualize embeddings
    plt.figure(figsize=(10, 8))
    for i in range(10):
        mask = labels[:1000] == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=f'Digit {i}', alpha=0.5)
    plt.legend()
    plt.title('t-SNE Visualization of MNIST Embeddings')
    plt.savefig('mnist_embeddings.png')
    plt.close()
    print("t-SNE visualization saved as 'mnist_embeddings.png'")
    
    # k-NN evaluation
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(embeddings[:8000], labels[:8000])
    pred = knn.predict(embeddings[8000:])
    accuracy = accuracy_score(labels[8000:], pred)
    print(f'k-NN Classification Accuracy: {accuracy:.4f}')

def main():
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(embed_dim=128).to(device)
    train_loader, test_loader = get_data_loaders()
    train_model(model, train_loader, epochs=10)
    evaluate_and_visualize(model, test_loader)

if __name__ == "__main__":
    main()
```

## 📖 Code Description:
1. **Dataset**:
   - Uses the MNIST handwritten digit dataset (60,000 training samples, 10,000 test samples).
   - Applies data augmentation (random cropping and rotation) during training to generate two views for contrastive learning.
   - Uses original images for testing to evaluate embedding quality.

2. **Model Architecture**:
   - Simple CNN encoder: two convolutional layers (with ReLU and MaxPool), followed by flattening and a fully connected layer, outputting 128-dimensional embeddings.
   - Contrastive loss (NT-Xent): encourages embeddings of augmented views of the same sample to be close, while embeddings of different samples are pushed apart; temperature parameter set to 0.5.

3. **Training**:
   - Uses Adam optimizer with a learning rate of 0.001, trained for 10 epochs.
   - Computes contrastive loss for each batch on two sets of augmented views to optimize the embedding space.

4. **Evaluation and Visualization**:
   - **Visualization**: Applies t-SNE dimensionality reduction on embeddings of the first 1,000 test samples, plots a 2D scatter plot colored by digit class, and saves it as `mnist_embeddings.png`.
   - **Evaluation**: Uses a k-NN classifier (k=5) in the embedding space to evaluate classification accuracy, with the first 8,000 samples as the training set and the remaining as the test set.
   - Outputs k-NN classification accuracy and the visualization file path.

5. **Dependencies**:
   - Requires `torch`, `torchvision`, `sklearn`, and `matplotlib` (`pip install torch torchvision scikit-learn matplotlib`).
   - MNIST dataset is automatically downloaded to the `./data` directory.

## 📖 Results:
- Outputs the average loss per training epoch.
- Generates `mnist_embeddings.png`, showing the distribution of different digit classes in the embedding space (ideally, same-class digits cluster together, and different classes are separated).
- Outputs k-NN classification accuracy, reflecting the quality of the embedding space.
- Runtime is long (due to t-SNE and k-NN computation); can run on CPU but faster on GPU.

## 📖 Notes:
- The scatter plot is saved in the working directory and can be viewed with an image viewer; colors represent different digit classes.
- The model is simple, suitable for demonstrating contrastive learning concepts; for practical applications, consider increasing network depth or tuning hyperparameters.
