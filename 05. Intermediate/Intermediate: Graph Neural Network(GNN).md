## Intermediate: Graph Neural Network(GNN)
<div align="center">
<img width="500" height="263" alt="image" src="https://github.com/user-attachments/assets/47f67caf-be26-42b4-928e-b8db05f1afab" />  
</div>

- **Importance**:  
  GNNs are specialized for handling graph-structured data (e.g., social networks, molecular structures) and are widely applied in recommendation systems, chemical modeling, and knowledge graphs.  
  They are key to extending deep learning to non-Euclidean data (e.g., graphs, networks), representing a cutting-edge direction in modern AI.  
- **Core Concept**:  
  A graph consists of nodes (vertices) and edges (connections). GNNs use "message passing" to allow nodes to aggregate information from their neighbors, learning both the graph's structure and features.  
  **Analogy**: Like "information spreading in a social circle," each node (person) updates its state based on information from friends.  
- **Applications**: Recommendation systems (e.g., Netflix recommendations), molecular design (drug discovery), traffic network analysis.
  



Write a minimal Graph Neural Network (GNN) example based on PyTorch and PyTorch Geometric, using a real dataset (Cora dataset, a common benchmark for graph classification). The model will implement a simple Graph Convolutional Network (GCN) for node classification. Results will be demonstrated by visualizing node embeddings (using t-SNE dimensionality reduction) and evaluating classification accuracy.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Define a simple GCN model
class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Visualize node embeddings
def visualize_embeddings(embeddings, labels, num_classes, title="t-SNE Visualization of Node Embeddings"):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        mask = labels == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=f'Class {i}', alpha=0.5)
    plt.legend()
    plt.title(title)
    plt.savefig('cora_embeddings.png')
    plt.close()
    print("t-SNE visualization saved as 'cora_embeddings.png'")

# Train and evaluate
def train_and_evaluate(model, data, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                pred = out.argmax(dim=1)
                acc = accuracy_score(data.y[data.val_mask].cpu(), pred[data.val_mask].cpu())
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Validation Accuracy: {acc:.4f}')
            model.train()
    
    # Test set evaluation
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        test_acc = accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
        print(f'\nTest Accuracy: {test_acc:.4f}')
        
        # Obtain embeddings (final layer output)
        embeddings = out.cpu().numpy()
        labels = data.y.cpu().numpy()
        visualize_embeddings(embeddings, labels, num_classes=data.num_classes)

def main():
    # Load the Cora dataset
    dataset = Planetoid(root='./data', name='Cora')
    data = dataset[0]
    data = data.to(device)
    
    # Initialize the model
    model = SimpleGCN(in_channels=dataset.num_features, hidden_channels=16, out_channels=dataset.num_classes).to(device)
    
    # Train and evaluate
    train_and_evaluate(model, data)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()

```

### Code Description:
1. **Dataset**:
   - Uses the Cora dataset (2,708 nodes, 7 classes, 1,433-dimensional features, representing academic papers and their citation relationships).
   - Each node is a paper, features are bag-of-words representations, edges are citation relationships, and the task is to predict paper categories.
   - Data is loaded via `torch_geometric`'s `Planetoid`, including training, validation, and test masks.

2. **Model Architecture**:
   - Simple GCN: Two GCNConv (graph convolutional) layers; the first layer maps 1,433-dimensional features to 16 dimensions, and the second layer maps to 7 dimensions (number of classes).
   - Uses ReLU activation and Dropout (p=0.5) to prevent overfitting.

3. **Training**:
   - Uses Adam optimizer with a learning rate of 0.01, weight decay of 5e-4, and trains for 200 epochs.
   - Loss function is cross-entropy, computed only on nodes with the training mask.
   - Prints training loss and validation accuracy every 50 epochs.

4. **Evaluation and Visualization**:
   - **Evaluation**: Computes node classification accuracy on the test set.
   - **Visualization**: Applies t-SNE to reduce node embeddings (final layer output) to 2D, plots a scatter plot colored by class, and saves it as `cora_embeddings.png`.
   - Ideally, nodes of the same class should cluster together in the embedding space.

5. **Dependencies**:
   - Requires `torch`, `torch_geometric`, `sklearn`, and `matplotlib` (`pip install torch torch-geometric scikit-learn matplotlib`).
   - The Cora dataset is automatically downloaded to the `./data` directory.

### Results:
- Outputs training loss and validation accuracy every 50 epochs.
- Outputs final classification accuracy on the test set.
- Generates `cora_embeddings.png`, showing the 2D distribution of node embeddings, with colors representing different classes.
- The scatter plot reflects whether the GNN learns meaningful embeddings (same-class nodes should be close, different-class nodes should be separated).

### Notes:
- The scatter plot is saved in the working directory and can be viewed with an image viewer.
- The model is simple (two-layer GCN), suitable for demonstrating GNN concepts; for practical applications, consider adding more layers or using advanced GNN variants (e.g., GAT).
