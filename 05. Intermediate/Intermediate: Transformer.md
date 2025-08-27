## Intermediate: Transformer
<div align="center">
<img width="300" height="400" alt="image" src="https://github.com/user-attachments/assets/8d064b02-6166-47ec-bfc6-fb031f94192c" />  
</div>

- **Importance**: Transformers are the core of modern natural language processing (NLP), powering large models like ChatGPT and representing the forefront of deep learning.
- **Core Concept**:  
  Transformers use an "attention mechanism" to focus on the most important parts of the input (e.g., key words in a sentence).  
  They are more efficient than RNNs and well-suited for handling long sequences.  
- **Applications**: Chatbots (e.g., Grok), machine translation, text generation.  
  **Why Teach**: Transformers represent the latest advancements in AI.

 

## A Transformer with only an encoder

```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math

# Set matplotlib font to ensure proper display of Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue

class PositionalEncoding(nn.Module):
    """Positional encoding layer to add positional information to sequences"""
    def __init__(self, d_model, max_len=5000):
        """
        Initialize positional encoding
        
        Parameters:
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
        
        # Compute sine and cosine positional encodings, filling even and odd columns
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add batch dimension and transpose, shape becomes (max_len, 1, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register positional encoding as a buffer (not updated during gradient descent)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Forward pass to add positional encoding
        
        Parameters:
            x (torch.Tensor): Input tensor, shape (seq_len, batch_size, d_model)
        
        Returns:
            torch.Tensor: Tensor with positional encoding added, same shape
        """
        return x + self.pe[:x.size(0), :]

class SimpleTransformer(nn.Module):
    """Simple Transformer model for sequence classification tasks"""
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, num_classes=2, max_len=100):
        """
        Initialize SimpleTransformer model
        
        Parameters:
            vocab_size (int): Vocabulary size
            d_model (int): Model dimension (default 64)
            nhead (int): Number of attention heads (default 4)
            num_layers (int): Number of Transformer encoder layers (default 2)
            num_classes (int): Number of classification classes (default 2)
            max_len (int): Maximum sequence length (default 100)
        """
        super(SimpleTransformer, self).__init__()
        
        self.d_model = d_model  # Store model dimension for later scaling
        # Word embedding layer to convert word indices to d_model-dimensional vectors
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional encoding layer to add sequence positional information
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Define a single Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # Model dimension
            nhead=nhead,  # Number of attention heads
            dim_feedforward=d_model * 4,  # Feedforward network hidden layer dimension (typically 4x d_model)
            dropout=0.1,  # Dropout probability to prevent overfitting
            batch_first=True  # Input format is (batch_size, seq_len, d_model)
        )
        # Stack multiple Transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head to map Transformer output to classification results
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # Linear layer for dimensionality reduction
            nn.ReLU(),  # ReLU activation function for non-linearity
            nn.Dropout(0.1),  # Dropout layer to prevent overfitting
            nn.Linear(d_model // 2, num_classes)  # Output classification results
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len)
        
        Returns:
            torch.Tensor: Classification output, shape (batch_size, num_classes)
        """
        batch_size, seq_len = x.shape
        
        # Word embedding and scaling (multiply by sqrt(d_model) to stabilize training)
        x = self.embedding(x) * math.sqrt(self.d_model)  # Shape: (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # Convert to (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)  # Add positional encoding
        x = x.transpose(0, 1)  # Convert back to (batch_size, seq_len, d_model)
        
        # Process sequence through Transformer encoder
        x = self.transformer_encoder(x)  # Shape: (batch_size, seq_len, d_model)
        
        # Global average pooling to obtain fixed-length representation
        x = x.mean(dim=1)  # Shape: (batch_size, d_model)
        
        # Output classification results through classification head
        x = self.classifier(x)  # Shape: (batch_size, num_classes)
        return x

class SimpleDataset(Dataset):
    """Simple dataset class for handling sequence data"""
    def __init__(self, sequences, labels, vocab_size=1000, max_len=50):
        """
        Initialize dataset
        
        Parameters:
            sequences (list): List of input sequences (can be strings or numerical lists)
            labels (list): List of labels
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
        Get a single sample
        
        Parameters:
            idx (int): Sample index
        
        Returns:
            tuple: (sequence tensor, label tensor)
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Convert string sequence to numerical sequence
        if isinstance(sequence, str):
            tokens = [ord(c) % self.vocab_size for c in sequence[:self.max_len]]
        else:
            tokens = list(sequence[:self.max_len])
        
        # Pad or truncate sequence to fixed length
        if len(tokens) < self.max_len:
            tokens = tokens + [0] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def generate_synthetic_data(num_samples=1000, seq_len=30, vocab_size=1000, num_classes=2):
    """Generate synthetic data for demonstration"""
    np.random.seed(42)  # Set random seed for reproducibility
    
    sequences = []
    labels = []
    
    for i in range=num_samples):
        # Generate random sequence
        seq = np.random.randint(1, vocab_size, seq_len)
        sequences.append(seq)
        
        # Compute sequence features for label generation
        freq_1 = np.sum(seq == 1) / seq_len  # Frequency of digit 1
        freq_2 = np.sum(seq == 2) / seq_len  # Frequency of digit 2
        freq_3 = np.sum(seq == 3) / seq_len  # Frequency of digit 3
        variance = np.var(seq)  # Sequence variance
        # Compute maximum consecutive identical digit length
        max_consecutive = 1
        current_consecutive = 1
        for j in range(1, len(seq)):
            if seq[j] == seq[j-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        # Generate label based on combined features
        score = (freq_1 * 0.3 + freq_2 * 0.2 + freq_3 * 0.1 + 
                (variance / 1000) * 0.2 + (max_consecutive / seq_len) * 0.2)
        label = 1 if score > 0.5 else 0
        labels.append(label)
    
    return sequences, labels

def train_model(model, train_loader, val_loader, num_epochs=15, device='cpu'):
    """Train the model"""
    model.to(device)  # Move model to specified device
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss function
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Adam optimizer with learning rate 0.0005
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)  # Learning rate scheduler, reduce rate every 5 epochs
    
    # Initialize lists for tracking metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print("Starting training...")
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
            _, predicted = output.max(1)  # Get predicted class
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        train_loss /= len(train_loader)  # Compute average loss
        train_accuracy = 100. * train_correct / train_total  # Compute accuracy
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():  # Disable gradient computation
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
        
        scheduler.step()  # Update learning rate
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # Print training progress
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
        print(f'  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
        print()
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training and validation loss and accuracy curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss curves
    ax1.plot(train_losses, label='Training Loss', color='blue', linewidth=2, marker='o', markersize=4)
    ax1.plot(val_losses, label='Validation Loss', color='red', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Plot accuracy curves
    ax2.plot(train_accuracies, label='Training Accuracy', color='blue', linewidth=2, marker='o', markersize=4)
    ax2.plot(val_accuracies, label='Validation Accuracy', color='red', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')  # Save image
    plt.show()

def predict(model, sequences, device='cpu'):
    """Make predictions using the trained model"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for sequence in sequences:
            # Process a single sequence
            if isinstance(sequence, str):
                tokens = [ord(c) % 1000 for c in sequence[:50]]  # Convert string to numerical sequence
            else:
                tokens = list(sequence[:50])  # Ensure list format
            
            # Pad or truncate sequence
            if len(tokens) < 50:
                tokens = tokens + [0] * (50 - len(tokens))
            else:
                tokens = tokens[:50]
            
            # Convert to tensor
            x = torch.tensor([tokens], dtype=torch.long).to(device)
            
            # Make prediction
            output = model(x)
            prob = F.softmax(output, dim=1)  # Compute class probabilities
            pred_class = output.argmax(dim=1).item()  # Get predicted class
            confidence = prob.max().item()  # Get maximum probability
            
            predictions.append({
                'sequence': sequence,
                'predicted_class': pred_class,
                'confidence': confidence,
                'probabilities': prob.cpu().numpy()[0]
            })
    
    return predictions

def main():
    """Main function to execute data generation, model training, and prediction"""
    # Set device (prefer GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    sequences, labels = generate_synthetic_data(num_samples=2000, seq_len=30)
    
    # Split into training and validation sets (80% training, 20% validation)
    split_idx = int(0.8 * len(sequences))
    train_sequences = sequences[:split_idx]
    train_labels = labels[:split_idx]
    val_sequences = sequences[split_idx:]
    val_labels = labels[split_idx:]
    
    # Create datasets and data loaders
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
    
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=15, device=device
    )
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Save model
    torch.save(model.state_dict(), 'simple_transformer_model.pth')
    print("Model saved to: simple_transformer_model.pth")
    
    # Test prediction
    print("\nTesting prediction functionality...")
    test_sequences = [
        "This is a test sequence",
        "Another test sequence",
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ]
    
    predictions = predict(model, test_sequences, device=device)
    
    for pred in predictions:
        print(f"Sequence: {pred['sequence']}")
        print(f"Predicted Class: {pred['predicted_class']}")
        print(f"Confidence: {pred['confidence']:.4f}")
        print(f"Class Probabilities: {pred['probabilities']}")
        print()

if __name__ == "__main__":
    main()
```

```
## 以下是对完整代码的详细中文注释，涵盖了每个类和函数的功能、参数及实现逻辑：

```python

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math

# 设置matplotlib中文字体，确保中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class PositionalEncoding(nn.Module):
    """位置编码层，为序列添加位置信息"""
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码
        
        参数:
            d_model (int): 模型维度
            max_len (int): 最大序列长度（默认5000）
        """
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵，形状为(max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # 生成位置索引，形状为(max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算除数项，用于正弦和余弦函数
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 计算正弦和余弦位置编码，分别填充到偶数和奇数列
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 添加批次维度并转置，形状变为(max_len, 1, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # 将位置编码注册为buffer（不参与梯度更新）
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        前向传播，添加位置编码
        
        参数:
            x (torch.Tensor): 输入张量，形状为(seq_len, batch_size, d_model)
        
        返回:
            torch.Tensor: 添加位置编码后的张量，形状不变
        """
        return x + self.pe[:x.size(0), :]

class SimpleTransformer(nn.Module):
    """简单的Transformer模型，用于序列分类任务"""
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, num_classes=2, max_len=100):
        """
        初始化SimpleTransformer模型
        
        参数:
            vocab_size (int): 词汇表大小
            d_model (int): 模型维度（默认64）
            nhead (int): 多头注意力机制的头数（默认4）
            num_layers (int): Transformer编码器层数（默认2）
            num_classes (int): 分类类别数（默认2）
            max_len (int): 最大序列长度（默认100）
        """
        super(SimpleTransformer, self).__init__()
        
        self.d_model = d_model  # 保存模型维度，用于后续缩放
        # 词嵌入层，将词索引转换为d_model维向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码层，添加序列位置信息
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # 定义单个Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # 模型维度
            nhead=nhead,  # 注意力头数
            dim_feedforward=d_model * 4,  # 前馈网络隐藏层维度（通常为d_model的4倍）
            dropout=0.1,  # dropout概率，防止过拟合
            batch_first=True  # 输入格式为(batch_size, seq_len, d_model)
        )
        # 堆叠多个Transformer编码器层
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头，将Transformer输出映射到分类结果
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # 线性层，降维
            nn.ReLU(),  # ReLU激活函数，增加非线性
            nn.Dropout(0.1),  # dropout层，防止过拟合
            nn.Linear(d_model // 2, num_classes)  # 输出分类结果
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状为(batch_size, seq_len)
        
        返回:
            torch.Tensor: 分类输出，形状为(batch_size, num_classes)
        """
        batch_size, seq_len = x.shape
        
        # 词嵌入并缩放（乘以sqrt(d_model)以稳定训练）
        x = self.embedding(x) * math.sqrt(self.d_model)  # 形状: (batch_size, seq_len, d_model)
        
        # 添加位置编码
        x = x.transpose(0, 1)  # 转换为(seq_len, batch_size, d_model)
        x = self.pos_encoding(x)  # 添加位置编码
        x = x.transpose(0, 1)  # 转换回(batch_size, seq_len, d_model)
        
        # 通过Transformer编码器处理序列
        x = self.transformer_encoder(x)  # 形状: (batch_size, seq_len, d_model)
        
        # 全局平均池化，得到固定长度表示
        x = x.mean(dim=1)  # 形状: (batch_size, d_model)
        
        # 通过分类头输出分类结果
        x = self.classifier(x)  # 形状: (batch_size, num_classes)
        return x

class SimpleDataset(Dataset):
    """简单的数据集类，用于处理序列数据"""
    def __init__(self, sequences, labels, vocab_size=1000, max_len=50):
        """
        初始化数据集
        
        参数:
            sequences (list): 输入序列列表（可以是字符串或数字列表）
            labels (list): 标签列表
            vocab_size (int): 词汇表大小（默认1000）
            max_len (int): 最大序列长度（默认50）
        """
        self.sequences = sequences
        self.labels = labels
        self.vocab_size = vocab_size
        self.max_len = max_len
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        参数:
            idx (int): 样本索引
        
        返回:
            tuple: (序列张量, 标签张量)
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # 将字符串序列转换为数字序列
        if isinstance(sequence, str):
            tokens = [ord(c) % self.vocab_size for c in sequence[:self.max_len]]
        else:
            tokens = list(sequence[:self.max_len])
        
        # 填充或截断序列到固定长度
        if len(tokens) < self.max_len:
            tokens = tokens + [0] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def generate_synthetic_data(num_samples=1000, seq_len=30, vocab_size=1000, num_classes=2):
    """生成合成数据用于演示"""
    np.random.seed(42)  # 设置随机种子以确保可重复性
    
    sequences = []
    labels = []
    
    for i in range(num_samples):
        # 生成随机序列
        seq = np.random.randint(1, vocab_size, seq_len)
        sequences.append(seq)
        
        # 计算序列特征用于生成标签
        freq_1 = np.sum(seq == 1) / seq_len  # 数字1的频率
        freq_2 = np.sum(seq == 2) / seq_len  # 数字2的频率
        freq_3 = np.sum(seq == 3) / seq_len  # 数字3的频率
        variance = np.var(seq)  # 序列方差
        # 计算最大连续相同数字长度
        max_consecutive = 1
        current_consecutive = 1
        for j in range(1, len(seq)):
            if seq[j] == seq[j-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        # 综合特征生成标签
        score = (freq_1 * 0.3 + freq_2 * 0.2 + freq_3 * 0.1 + 
                (variance / 1000) * 0.2 + (max_consecutive / seq_len) * 0.2)
        label = 1 if score > 0.5 else 0
        labels.append(label)
    
    return sequences, labels

def train_model(model, train_loader, val_loader, num_epochs=15, device='cpu'):
    """训练模型"""
    model.to(device)  # 将模型移动到指定设备
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Adam优化器，学习率0.0005
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)  # 学习率调度器，每5个epoch降低学习率
    
    # 初始化记录列表
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print("开始训练...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)  # 移动数据到设备
            optimizer.zero_grad()  # 清空梯度
            output = model(data)  # 前向传播
            loss = criterion(output, target)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            
            train_loss += loss.item()
            _, predicted = output.max(1)  # 获取预测类别
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        train_loss /= len(train_loader)  # 计算平均损失
        train_accuracy = 100. * train_correct / train_total  # 计算准确率
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():  # 禁用梯度计算
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
        
        scheduler.step()  # 更新学习率
        
        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # 打印训练进度
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%')
        print(f'  验证损失: {val_loss:.4f}, 验证准确率: {val_accuracy:.2f}%')
        print()
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """绘制训练和验证的损失及准确率曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    ax1.plot(train_losses, label='训练损失', color='blue', linewidth=2, marker='o', markersize=4)
    ax1.plot(val_losses, label='验证损失', color='red', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('轮次', fontsize=12)
    ax1.set_ylabel('损失', fontsize=12)
    ax1.set_title('训练和验证损失曲线', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # 绘制准确率曲线
    ax2.plot(train_accuracies, label='训练准确率', color='blue', linewidth=2, marker='o', markersize=4)
    ax2.plot(val_accuracies, label='验证准确率', color='red', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('轮次', fontsize=12)
    ax2.set_ylabel('准确率 (%)', fontsize=12)
    ax2.set_title('训练和验证准确率曲线', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')  # 保存图像
    plt.show()

def predict(model, sequences, device='cpu'):
    """使用训练好的模型进行预测"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for sequence in sequences:
            # 处理单个序列
            if isinstance(sequence, str):
                tokens = [ord(c) % 1000 for c in sequence[:50]]  # 字符串转换为数字序列
            else:
                tokens = list(sequence[:50])  # 确保为列表格式
            
            # 填充或截断序列
            if len(tokens) < 50:
                tokens = tokens + [0] * (50 - len(tokens))
            else:
                tokens = tokens[:50]
            
            # 转换为张量
            x = torch.tensor([tokens], dtype=torch.long).to(device)
            
            # 进行预测
            output = model(x)
            prob = F.softmax(output, dim=1)  # 计算类别概率
            pred_class = output.argmax(dim=1).item()  # 获取预测类别
            confidence = prob.max().item()  # 获取最大概率
            
            predictions.append({
                'sequence': sequence,
                'predicted_class': pred_class,
                'confidence': confidence,
                'probabilities': prob.cpu().numpy()[0]
            })
    
    return predictions

def main():
    """主函数，执行数据生成、模型训练和预测"""
    # 设置设备（优先使用GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 生成合成数据
    print("生成合成数据...")
    sequences, labels = generate_synthetic_data(num_samples=2000, seq_len=30)
    
    # 划分训练集和验证集（80%训练，20%验证）
    split_idx = int(0.8 * len(sequences))
    train_sequences = sequences[:split_idx]
    train_labels = labels[:split_idx]
    val_sequences = sequences[split_idx:]
    val_labels = labels[split_idx:]
    
    # 创建数据集和数据加载器
    train_dataset = SimpleDataset(train_sequences, train_labels)
    val_dataset = SimpleDataset(val_sequences, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    model = SimpleTransformer(
        vocab_size=1000,
        d_model=64,
        nhead=4,
        num_layers=2,
        num_classes=2,
        max_len=50
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=15, device=device
    )
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # 保存模型
    torch.save(model.state_dict(), 'simple_transformer_model.pth')
    print("模型已保存到: simple_transformer_model.pth")
    
    # 测试预测
    print("\n测试预测功能...")
    test_sequences = [
        "这是一个测试序列",
        "另一个测试序列",
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ]
    
    predictions = predict(model, test_sequences, device=device)
    
    for pred in predictions:
        print(f"序列: {pred['sequence']}")
        print(f"预测类别: {pred['predicted_class']}")
        print(f"置信度: {pred['confidence']:.4f}")
        print(f"类别概率: {pred['probabilities']}")
        print()

if __name__ == "__main__":
    main()

```

### Overall Code Description:
1. **Code Functionality**:
   - A complete PyTorch implementation for a Transformer-based sequence classification task.
   - Includes positional encoding, a Transformer model, dataset processing, data generation, model training, result visualization, and prediction functionality.

2. **Main Components**:
   - **PositionalEncoding**: Implements classic sine/cosine positional encoding to add positional information to sequences.
   - **SimpleTransformer**: A simple Transformer model consisting of word embeddings, positional encoding, a Transformer encoder, and a classification head.
   - **SimpleDataset**: A custom dataset class supporting string and numerical sequences, handling padding and truncation.
   - **generate_synthetic_data**: Generates synthetic data with labels based on sequence features (e.g., frequency, variance, continuity).
   - **train_model**: Trains the model, tracks loss and accuracy, using the Adam optimizer and learning rate scheduling.
   - **plot_training_curves**: Plots training and validation loss and accuracy curves.
   - **predict**: Predicts the class, confidence, and probability for new sequences.
   - **main**: The main function that coordinates data generation, training, plotting, and prediction.

3. **Use Cases**:
   - Suitable for rapid prototyping of sequence classification tasks.
   - Synthetic data is used for demonstration; real datasets can be substituted for practical applications.
   - Performance can be optimized by adjusting model parameters (e.g., `d_model`, `nhead`).

## Comments
### 1. Data Preparation
```
# Generate 2000 samples
sequences, labels = generate_synthetic_data(num_samples=2000, seq_len=30)

# 80% training, 20% validation
split_idx = int(0.8 * len(sequences))
train_sequences = sequences[:split_idx]  # 1600 samples
val_sequences = sequences[split_idx:]    # 400 samples
```

### 2. Training Loop
```
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
```
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Adam optimizer
```

### Summary
**Data**: Synthetically generated data, classified based on statistical features.  
**Model**: Transformer with only an encoder, used for sequence classification.  
**Decoder**: Not used, as sequence generation is not required.  
**Workflow**: Data generation → Preprocessing → Embedding → Positional encoding → Self-attention → Pooling → Classification.  
This model is ideal for learning the basics of Transformers, especially the self-attention mechanism and positional encoding!
