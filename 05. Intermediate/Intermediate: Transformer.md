
## Transformer

**Transformer** is a deep learning model architecture based on the **Attention Mechanism**, proposed by Vaswani et al. in the 2017 paper *“Attention is All You Need”*. It was originally designed for natural language processing (NLP) tasks but has since been widely applied to computer vision (CV), speech processing, and multimodal learning.

<div align="center">  
<img width="300" height="400" alt="image" src="https://github.com/user-attachments/assets/8d064b02-6166-47ec-bfc6-fb031f94192c" />  
</div> 

<div align="center">
(This picture was obtained from the Internet.)
</div>


### Core Ideas

1. **Self-Attention**:
   Each element (e.g., a word vector) computes correlations with other elements in the sequence, dynamically capturing contextual information and long-range dependencies.

2. **Positional Encoding**:
   Since the Transformer does not use recurrence (RNN) or convolution (CNN), positional encoding is introduced explicitly to inject sequence order information.

3. **Multi-Head Attention**:
   Multiple attention mechanisms are computed in parallel, focusing on different feature subspaces, thereby enhancing the model’s representational power.

4. **Encoder-Decoder Architecture**:

   * **Encoder**: Maps the input sequence into contextual representations.
   * **Decoder**: Generates the target sequence step by step based on encoder representations and previously generated outputs.

### Common Applications

* **Natural Language Processing (NLP)**: Machine translation (e.g., Google Translate), text generation (GPT series), question answering, text summarization.
* **Computer Vision (CV)**: Vision Transformer (ViT) for image classification and object detection.
* **Speech Processing**: Speech recognition and speech synthesis.
* **Multimodal Learning**: CLIP, DALL·E, GPT-4V, and other cross-modal models.

---

## Mathematical description of Transformer
The Transformer architecture is a core model in NLP and deep learning, originally proposed by Vaswani et al. in the 2017 paper *"Attention is All You Need"*. Below is its mathematical description, covering the main components, including input representation, attention mechanism, positional encoding, feed-forward network, and layer normalization.    

### 1. Overall architecture  
The Transformer consists of an Encoder and a Decoder, each containing multiple stacked layers (usually $N$ layers). The encoder processes the input sequence, and the decoder generates the output sequence. The core innovation is the Self-Attention mechanism, replacing the sequential processing of traditional Recurrent Neural Networks (RNNs).  

Input representation  
The input sequence (e.g., words or tokens) is first converted into vector representations:
* **Word Embedding**: Each word is mapped into a fixed-dimensional vector $x_i \in \mathbb{R}^d$, usually implemented via an embedding matrix  
  $E \in \mathbb{R}^{|V|\times d}$, where $|V|$ is the vocabulary size and $d$ is the embedding dimension.

* **Positional Encoding**: Since Transformer does not have inherent sequential order information, positional encoding (Positional Encoding) is added to capture the position of words.  
  Positional encoding $PE$ can be generated using fixed formulas:  

$$
PE{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d}}}\right), 
\quad
PE{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d}}}\right)
$$

Where $pos$ is the position of the word in the sequence, and $i$ is the dimension index. The final input is:  

$$
x_i = E_i + PE(pos_i)
$$

Where:  

* $E_i$ represents the embedding vector of word $w_i$ (i.e., the result retrieved from embedding matrix $E$);  
* $PE(pos_i)$ represents the positional encoding vector;  
* The two are added element-wise as the input to the Transformer.  

- Encoder  
Each encoder layer contains two main sub-modules:  

Multi-Head Self-Attention  
Feed-Forward Neural Network (FFN)  

Each sub-module is followed by a Residual Connection and Layer Normalization.  

- Decoder  
The decoder is similar to the encoder but includes Masked Self-Attention (to prevent future information leakage) and Encoder-Decoder Attention.  



### 2. Multi-Head Self-Attention Mechanism 
Self-attention is the core of the Transformer, allowing the model to focus on other words in the sequence when processing each word.  

Scaled Dot-Product Attention  

For an input sequence $X \in \mathbb{R}^{n \times d}$ ($n$ is sequence length, $d$ is embedding dimension), the steps for calculating attention scores are as follows:  

**1. Generate Query, Key, and Value:**  

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

Where $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$ are learnable projection matrices, and $d_k$ is the dimension of the attention head.  

**2. Calculate Attention Weights:**  

$$
Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

* $QK^T \in \mathbb{R}^{n \times n}$ represents the dot product between query and key, measuring the correlation between words.  
* $\sqrt{d_k}$ is a scaling factor to prevent large dot products from causing softmax saturation.  
* The softmax operation normalizes each row, yielding attention weights, applied to the value vector $V$.  



### Multi-Head Mechanism  

Multi-head attention splits $Q, K, V$ into $h$ heads, each independently computing attention:  

$$
MultiHead(Q, K, V) = Concat(head_1, \ldots, head_h)W^O
$$

Where:  

$$
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
$$

$$
W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d \times d_k}, \quad W^O \in \mathbb{R}^{h \cdot d_k \times d}, \quad h \text{ is the number of heads}, d_k = \frac{d}{h}
$$



### Masked Self-Attention (Decoder Specific)  

In the decoder, to prevent the current word from attending to subsequent words, a mask matrix $M$ is introduced, making attention weights at future positions $-\infty$:  

$$
Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$



### 3. Feed-Forward Neural Network (FFN)  
Each encoder and decoder layer contains a position-wise feed-forward network, applied to each word vector:  

$$
\mathrm{FFN}(x) = \mathrm{ReLU}(x W_1 + b_1) W_2 + b_2
$$  

Where $W_1 \in \mathbb{R}^{d \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d}$, and $d_{ff}$ is usually much larger than $d$ (e.g., $d_{ff} = 4d$).  



### 4. Residual Connection and Layer Normalization  
Each sub-module (Self-Attention or FFN) is followed by a residual connection and layer normalization:  

$$
y = \mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))
$$  

Where Sublayer is either Attention or FFN, and LayerNorm is defined as:  

$$
\mathrm{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$  

$\mu$ and $\sigma^2$ are the mean and variance of the input vector, and $\gamma, \beta$ are learnable parameters.  



### 5. Encoder-Decoder Attention  
The additional attention layer in the decoder uses the encoder’s output $K, V$ and the decoder’s $Q$:  

$\mathrm{Attention}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}})$  

This allows the decoder to attend to the context of the input sequence.  



### 6. Output Layer  

The final decoder layer generates output probabilities via linear transformation and softmax:  

$$
P(y_i) = \mathrm{softmax}(z W_{\text{out}} + b_{\text{out}})
$$  

Where $z$ is the output of the final decoder layer, $W_{\text{out}} \in \mathbb{R}^{d \times |V|}$.  



### 7. Output Layer  
The final decoder layer generates output probabilities via linear transformation and softmax:  

$P(y_i) = \mathrm{softmax}(z W_{\text{out}} + b_{\text{out}})$  

Where $z$ is the output of the final decoder layer, $W_{\text{out}} \in \mathbb{R}^{d \times |V|}$.  



### 8. Loss Function  
Training usually uses cross-entropy loss, with the objective of maximizing the probability of the correct output sequence:  

$\mathcal{L} = -\sum_{i=1}^{T} \log P(y_i \mid y_{<i}, X)$  

Where $T$ is the output sequence length, and $y_{<i}$ is the already generated words.  



### 9. Summary  
The mathematical core of Transformer lies in:  

Self-Attention: Capturing relationships within the sequence via Q, K, V.  
Multi-Head Mechanism: Capturing multiple semantic relationships in parallel.  
Positional Encoding: Compensating for the lack of sequential order information.  
Residuals and Normalization: Stabilizing training and accelerating convergence.  

---

**Full Transformer vs Encoder Transformer**  

### Comparison of Two Transformers  

| Component | Full Transformer | Encoder Transformer |
| ---- | ------------- | -------------- |
| Encoder  | ✅             | ✅              |
| Decoder  | ✅             | ❌              |
| Suitable tasks | Translation, Summarization         | Classification, Sentiment Analysis        |
| Input/Output | Sequence→Sequence         | Sequence→Class          |



## Transformer with Only Encoder  
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math

# Set matplotlib font to support Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

class PositionalEncoding(nn.Module):
    """Positional encoding layer, adds position information to the sequence"""
    def __init__(self, d_model, max_len=5000):
        """
        Initialize positional encoding
        
        Args:
            d_model (int): Model dimension
            max_len (int): Maximum sequence length (default 5000)
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix, shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # Generate position indices, shape (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute divisor term for sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Compute sine and cosine positional encodings, fill even and odd columns respectively
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add batch dimension and transpose, shape becomes (max_len, 1, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register positional encoding as buffer (not updated by gradient)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Forward pass, add positional encoding
        
        Args:
            x (torch.Tensor): Input tensor, shape (seq_len, batch_size, d_model)
        
        Returns:
            torch.Tensor: Tensor with positional encoding added, same shape
        """
        return x + self.pe[:x.size(0), :]

class SimpleTransformer(nn.Module):
    """Simple Transformer model for sequence classification"""
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, num_classes=2, max_len=100):
        """
        Initialize SimpleTransformer model
        
        Args:
            vocab_size (int): Vocabulary size
            d_model (int): Model dimension (default 64)
            nhead (int): Number of attention heads (default 4)
            num_layers (int): Number of Transformer encoder layers (default 2)
            num_classes (int): Number of classification categories (default 2)
            max_len (int): Maximum sequence length (default 100)
        """
        super(SimpleTransformer, self).__init__()
        
        self.d_model = d_model  # Save model dimension for scaling later
        # Embedding layer, convert token indices to d_model-dimensional vectors
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional encoding layer, add sequence position information
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Define single Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # Model dimension
            nhead=nhead,  # Number of attention heads
            dim_feedforward=d_model * 4,  # Feedforward hidden dimension (typically 4 * d_model)
            dropout=0.1,  # Dropout rate to prevent overfitting
            batch_first=True  # Input format: (batch_size, seq_len, d_model)
        )
        # Stack multiple Transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head, map Transformer output to classification result
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # Linear layer, reduce dimension
            nn.ReLU(),  # ReLU activation for non-linearity
            nn.Dropout(0.1),  # Dropout layer to prevent overfitting
            nn.Linear(d_model // 2, num_classes)  # Output classification result
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len)
        
        Returns:
            torch.Tensor: Classification output, shape (batch_size, num_classes)
        """
        batch_size, seq_len = x.shape
        
        # Embedding and scaling (multiply by sqrt(d_model) for stability)
        x = self.embedding(x) * math.sqrt(self.d_model)  # Shape: (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # Convert to (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)  # Add positional encoding
        x = x.transpose(0, 1)  # Convert back to (batch_size, seq_len, d_model)
        
        # Pass through Transformer encoder
        x = self.transformer_encoder(x)  # Shape: (batch_size, seq_len, d_model)
        
        # Global average pooling to fixed length representation
        x = x.mean(dim=1)  # Shape: (batch_size, d_model)
        
        # Classification head
        x = self.classifier(x)  # Shape: (batch_size, num_classes)
        return x

class SimpleDataset(Dataset):
    """Simple dataset class for sequence data"""
    def __init__(self, sequences, labels, vocab_size=1000, max_len=50):
        """
        Initialize dataset
        
        Args:
            sequences (list): Input sequence list (string or number list)
            labels (list): Label list
            vocab_size (int): Vocabulary size (default 1000)
            max_len (int): Maximum sequence length (default 50)
        """
        self.sequences = sequences
        self.labels = labels
        self.vocab_size = vocab_size
        self.max_len = max_len
    
    def __len__(self):
        """Return dataset size"""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get single sample
        
        Args:
            idx (int): Sample index
        
        Returns:
            tuple: (sequence tensor, label tensor)
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Convert string sequence to numeric sequence
        if isinstance(sequence, str):
            tokens = [ord(c) % self.vocab_size for c in sequence[:self.max_len]]
        else:
            tokens = list(sequence[:self.max_len])
        
        # Pad or truncate to fixed length
        if len(tokens) < self.max_len:
            tokens = tokens + [0] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def generate_synthetic_data(num_samples=1000, seq_len=30, vocab_size=1000, num_classes=2):
    """Generate synthetic data for demo"""
    np.random.seed(42)  # Set random seed for reproducibility
    
    sequences = []
    labels = []
    
    for i in range(num_samples):
        # Generate random sequence
        seq = np.random.randint(1, vocab_size, seq_len)
        sequences.append(seq)
        
        # Compute sequence features for label generation
        freq_1 = np.sum(seq == 1) / seq_len  # Frequency of number 1
        freq_2 = np.sum(seq == 2) / seq_len  # Frequency of number 2
        freq_3 = np.sum(seq == 3) / seq_len  # Frequency of number 3
        variance = np.var(seq)  # Sequence variance
        # Compute max consecutive identical numbers
        max_consecutive = 1
        current_consecutive = 1
        for j in range(1, len(seq)):
            if seq[j] == seq[j-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        # Generate label using combined features
        score = (freq_1 * 0.3 + freq_2 * 0.2 + freq_3 * 0.1 + 
                (variance / 1000) * 0.2 + (max_consecutive / seq_len) * 0.2)
        label = 1 if score > 0.5 else 0
        labels.append(label)
    
    return sequences, labels

def train_model(model, train_loader, val_loader, num_epochs=15, device='cpu'):
    """Train the model"""
    model.to(device)  # Move model to device
    criterion = nn.CrossEntropyLoss()  # Cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Adam optimizer, lr=0.0005
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)  # LR scheduler
    
    # Initialize record lists
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print("Start training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)  # Move data to device
            optimizer.zero_grad()  # Clear gradients
            output = model(data)  # Forward pass
            loss = criterion(output, target)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters
            
            train_loss += loss.item()
            _, predicted = output.max(1)  # Predicted class
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        train_loss /= len(train_loader)  # Average loss
        train_accuracy = 100. * train_correct / train_total  # Accuracy
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():  # Disable gradients
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100. * val_correct / val_total
        
        scheduler.step()  # Update LR
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print()
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training and validation loss and accuracy curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss', color='blue', linewidth=2, marker='o', markersize=4)
    ax1.plot(val_losses, label='Val Loss', color='red', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Accuracy curves
    ax2.plot(train_accuracies, label='Train Acc', color='blue', linewidth=2, marker='o', markersize=4)
    ax2.plot(val_accuracies, label='Val Acc', color='red', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')  # Save image
    plt.show()

def predict(model, sequences, device='cpu'):
    """Use trained model to predict"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for sequence in sequences:
            # Process single sequence
            if isinstance(sequence, str):
                tokens = [ord(c) % 1000 for c in sequence[:50]]  # String to numeric sequence
            else:
                tokens = list(sequence[:50])  # Ensure list format
            
            # Pad or truncate
            if len(tokens) < 50:
                tokens = tokens + [0] * (50 - len(tokens))
            else:
                tokens = tokens[:50]
            
            # Convert to tensor
            x = torch.tensor([tokens], dtype=torch.long).to(device)
            
            # Predict
            output = model(x)
            prob = F.softmax(output, dim=1)  # Class probabilities
            pred_class = output.argmax(dim=1).item()  # Predicted class
            confidence = prob.max().item()  # Max probability
            
            predictions.append({
                'sequence': sequence,
                'predicted_class': pred_class,
                'confidence': confidence,
                'probabilities': prob.cpu().numpy()[0]
            })
    
    return predictions

def main():
    """Main function, execute data generation, training and prediction"""
    # Set device (prefer GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    sequences, labels = generate_synthetic_data(num_samples=2000, seq_len=30)
    
    # Split train and validation sets (80% train, 20% validation)
    split_idx = int(0.8 * len(sequences))
    train_sequences = sequences[:split_idx]
    train_labels = labels[:split_idx]
    val_sequences = sequences[split_idx:]
    val_labels = labels[split_idx:]
    
    # Create datasets and dataloaders
    train_dataset = SimpleDataset(train_sequences, train_labels)
    val_dataset = SimpleDataset(val_sequences, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = SimpleTransformer(
        vocab_size=1000,
        d_model=64,
        nhead=4,
        num_layers=2,
        num_classes=2,
        max_len=50
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=15, device=device
    )
    
    # Plot curves
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Save model
    torch.save(model.state_dict(), 'simple_transformer_model.pth')
    print("Model saved to: simple_transformer_model.pth")
    
    # Test prediction
    print("\nTesting prediction...")
    test_sequences = [
        "This is a test sequence",
        "Another test sequence",
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ]
    
    predictions = predict(model, test_sequences, device=device)
    
    for pred in predictions:
        print(f"Sequence: {pred['sequence']}")
        print(f"Predicted class: {pred['predicted_class']}")
        print(f"Confidence: {pred['confidence']:.4f}")
        print(f"Probabilities: {pred['probabilities']}")
        print()

if __name__ == "__main__":
    main()

```

### Overall Code Description:

1. **Functionality**:

   * This is a complete PyTorch implementation for a Transformer-based sequence classification task.
   * It includes positional encoding, a Transformer model, dataset processing, synthetic data generation, model training, result visualization, and prediction functions.

2. **Main Components**:

   * **PositionalEncoding**: Implements the classic sine/cosine positional encoding to add position information to sequences.
   * **SimpleTransformer**: A simple Transformer model with embeddings, positional encoding, Transformer encoder, and classification head.
   * **SimpleDataset**: Custom dataset class supporting both string and numeric sequences, with padding and truncation.
   * **generate\_synthetic\_data**: Generates synthetic data and labels based on sequence features (frequency, variance, continuity).
   * **train\_model**: Trains the model, records loss and accuracy, using Adam optimizer and a learning rate scheduler.
   * **plot\_training\_curves**: Plots training and validation loss and accuracy curves.
   * **predict**: Makes predictions on new sequences, returning class, confidence, and probabilities.
   * **main**: The entry point, coordinating data generation, training, plotting, and prediction.

3. **Use Cases**:

   * Suitable for rapid prototyping of sequence classification tasks.
   * Synthetic data is for demonstration; in practice, it can be replaced with real datasets.
   * Performance can be optimized by adjusting model parameters such as `d_model` and `nhead`.

---

## Comments

### 1. Data Preparation

```python
# Generate 2000 samples
sequences, labels = generate_synthetic_data(num_samples=2000, seq_len=30)

# 80% training, 20% validation
split_idx = int(0.8 * len(sequences))
train_sequences = sequences[:split_idx]  # 1600 samples
val_sequences = sequences[split_idx:]    # 400 samples
```

### 2. Training Loop

```python
for epoch in range(15):
    # Training phase
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(data)  # Forward pass
        loss = criterion(output, target)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            output = model(data)
            # Compute validation loss and accuracy
```

### 3. Loss Function

```python
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Adam optimizer
```

### Summary

* **Data**: Synthetic data generated by the program, classified based on statistical features.
* **Model**: A Transformer encoder-only architecture, used for sequence classification.
* **Decoder**: Not included, since sequence output generation is unnecessary.
* **Pipeline**: Data generation → Preprocessing → Embedding → Positional Encoding → Self-Attention → Pooling → Classification.

This model is well-suited for learning the **fundamentals of Transformer architectures**, especially **self-attention** and **positional encoding**!
