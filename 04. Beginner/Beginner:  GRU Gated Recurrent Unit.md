## Beginner: GRU (Gated Recurrent Unit)
<img width="552" height="277" alt="image" src="https://github.com/user-attachments/assets/2c7f3eef-f4be-471c-b7df-cd62b479df28" />
<img width="884" height="266" alt="image" src="https://github.com/user-attachments/assets/75186129-08a6-478c-b91e-82a65e0a601f" />

The Gated Recurrent Unit (GRU) is a variant of Recurrent Neural Networks (RNNs) commonly used for processing sequential data, proposed by Kyunghyun Cho et al. in 2014. GRU aims to address the vanishing or exploding gradient problems encountered by traditional RNNs in long sequence processing while simplifying the structure of Long Short-Term Memory (LSTM) networks, offering lower computational complexity and fewer parameters.

### Core Concept of GRU
GRU controls the flow and forgetting of information through an update gate and a reset gate, effectively capturing long-term dependencies in sequences. Compared to LSTM, GRU combines the forget gate and input gate into a single update gate, simplifying the structure while retaining strong modeling capabilities.

### GRU Working Mechanism
 
A GRU unit at each time step receives the current input $x_t$ and the hidden state of the previous time step $h_{t-1}$, and outputs a new hidden state $h_t$. Its core formulas are as follows:


### 1. Update Gate ($z_t$):
  
$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
$$
  
The update gate determines how much of the hidden state information from the previous time step is retained, and how much new information is accepted. $\sigma$ is the sigmoid activation function, with outputs in the range $[0, 1]$.


### 2. Reset Gate ($r_t$):

$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
$$

The reset gate controls the degree of combination between the current input and the previous hidden state, and is used to decide how much past information to forget.


### 3. Candidate Hidden State ($\tilde{h}_t$):
<div align="center">
<img width="280" height="35" alt="image" src="https://github.com/user-attachments/assets/5debf918-f924-4f59-be30-f3094a401580" /> 
</div
The candidate hidden state is obtained by adjusting the historical information with the reset gate and combining it with the current input. $\odot$ denotes element-wise multiplication, and $\tanh$ is the activation function.


### 4. Final Hidden State ($h_t$):

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

The final hidden state is obtained by weighting and combining the previous hidden state and the candidate hidden state through the update gate.

### GRU Characteristics
- **Simplified Structure**: Compared to LSTM, GRU has only two gates (update gate and reset gate), fewer parameters, and higher computational efficiency.
- **Long-Term Dependencies**: Through its gating mechanism, GRU effectively captures dependencies in long sequences, mitigating the vanishing gradient problem.
- **Flexibility**: GRU is suitable for various sequence modeling tasks, such as natural language processing (NLP) and time series forecasting.

### Comparison of GRU and LSTM
- **Similarities**: Both use gating mechanisms to address RNN gradient issues and are suitable for long-sequence tasks.
- **Differences**:
  - GRU has a simpler structure, fewer parameters, and faster training speed.
  - LSTM has a separate memory cell, suitable for more complex tasks but with higher computational cost.
  - In practice, the performance of GRU and LSTM varies by task, and the choice depends on the specific scenario.

### Application Scenarios
GRU is widely used in:
- **Natural Language Processing**: Machine translation, text generation, sentiment analysis.
- **Time Series Analysis**: Stock price prediction, weather forecasting.
- **Speech Processing**: Speech recognition, speech synthesis.

### Summary
GRU is an efficient, simplified variant of RNNs that achieves selective information transfer and forgetting through update and reset gates. While maintaining strong sequence modeling capabilities, it reduces computational complexity, making it an ideal choice for many sequence tasks.

## Example
Below is a simple GRU example using Python, PyTorch, and a real dataset (sine wave sequence) to demonstrate its principles, with visualization of prediction results using Matplotlib. The example uses sine wave data for sequence prediction, where the GRU learns the sequence pattern and predicts subsequent values. The code includes data preparation, GRU model definition, training, and visualization.

### Description
**Dataset**: Uses a sine wave (sin(t)) as real data, generating 1000 points. Each sample contains 10 consecutive points as input to predict the next point.
- **GRU Model**:
  - Input size: 1 (univariate time series).
  - Hidden layer size: 16 (simple network, sufficient to capture sine wave patterns).
  - Output size: 1 (predicts the next value).
  - The GRU layer processes the sequence, and a linear layer (fc) maps the output of the last time step to the predicted value.
- **Training**: Uses the Adam optimizer and Mean Squared Error (MSE) loss function, trained for 100 epochs.
- **Visualization**: Uses Matplotlib to plot true values (solid blue line) and predicted values (dashed red line), demonstrating GRU’s ability to fit the sine wave pattern.

## Code
```python
# Fix OpenMP error - must be before all other imports
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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
# Set matplotlib backend and font
plt.switch_backend('Agg')
# Support for Chinese fonts
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def generate_sine_wave_data():
    """Generate sine wave data"""
    t = np.linspace(0, 20, 1000)  # Time axis
    data = np.sin(t)  # Sine wave
    sequence_length = 10  # Sequence length
    X, y = [], []
    # Prepare input-output pairs: use 10 previous points to predict the next point
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    
    X = np.array(X).reshape(-1, sequence_length, 1)  # [samples, sequence length, features]
    y = np.array(y).reshape(-1, 1)  # [samples, 1]
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    return X, y, t, data

class GRUModel(nn.Module):
    """GRU model for time series prediction"""
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1, dropout=0.1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # GRU forward pass
        out, _ = self.gru(x, h0)
        
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

def train_model(model, X, y, epochs=100, lr=0.01):
    """Train the model"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses = []
    
    print("Starting GRU model training...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = model(X)
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step(loss)
        
        train_losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return train_losses

def evaluate_model(model, X, y):
    """Evaluate the model"""
    model.eval()
    with torch.no_grad():
        predictions = model(X).numpy()
        true_values = y.numpy()
        
        # Calculate evaluation metrics
        mse = np.mean((predictions - true_values) ** 2)
        mae = np.mean(np.abs(predictions - true_values))
        rmse = np.sqrt(mse)
        
        print(f"Model evaluation results:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        
        return predictions, mse, mae, rmse

def visualize_results(t, data, y, predictions, sequence_length, train_losses):
    """Visualize results"""
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Prediction vs True Values
    axes[0, 0].plot(t[sequence_length:], y.numpy(), label='True Values', color='blue', linewidth=2)
    axes[0, 0].plot(t[sequence_length:], predictions, label='Predicted Values', color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_title('GRU Sine Wave Prediction Results', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Training Loss Curve
    axes[0, 1].plot(train_losses, 'b-', linewidth=2)
    axes[0, 1].set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # 3. Prediction Error
    error = y.numpy() - predictions
    axes[1, 0].plot(t[sequence_length:], error, color='green', linewidth=1)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 0].set_title('Prediction Error', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Error')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Error Distribution Histogram
    axes[1, 1].hist(error, bins=30, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Error Distribution Histogram', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gru_sine_wave_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_model_architecture():
    """Visualize model architecture"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Define colors
    colors = {
        'input': '#E8F4FD',
        'gru': '#FFE6E6',
        'fc': '#E6FFE6',
        'output': '#F0E6FF'
    }
    
    # Define positions
    positions = {
        'input': (2, 6),
        'gru1': (4, 6),
        'gru2': (6, 6),
        'fc': (8, 6),
        'output': (10, 6)
    }
    
    # Draw network layers
    for name, pos in positions.items():
        if 'gru' in name:
            color = colors['gru']
            width, height = 2.0, 1.2
        elif 'fc' in name:
            color = colors['fc']
            width, height = 2.0, 1.2
        elif 'input' in name:
            color = colors['input']
            width, height = 1.8, 1.0
        elif 'output' in name:
            color = colors['output']
            width, height = 1.8, 1.0
        else:
            color = colors['input']
            width, height = 1.8, 1.0
        
        # Draw box
        box = FancyBboxPatch(
            (pos[0] - width/2, pos[1] - height/2),
            width, height,
            boxstyle="round,pad=0.15",
            facecolor=color,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(box)
        
        # Add text
        if 'gru' in name:
            text = f'{name.upper()}\nHidden Size: 32'
        elif 'fc' in name:
            text = f'{name.upper()}\n32→1'
        elif 'input' in name:
            text = 'Input\n10×1'
        elif 'output' in name:
            text = 'Output\n1'
        else:
            text = name
        
        ax.text(pos[0], pos[1], text, ha='center', va='center',
                fontsize=10, fontweight='bold')
    
    # Draw connection lines
    main_flow = ['input', 'gru1', 'gru2', 'fc', 'output']
    for i in range(len(main_flow) - 1):
        start_pos = positions[main_flow[i]]
        end_pos = positions[main_flow[i + 1]]
        ax.arrow(start_pos[0] + 0.9, start_pos[1],
                 end_pos[0] - start_pos[0] - 1.8, 0,
                 head_width=0.15, head_length=0.15, fc='blue', ec='blue', linewidth=3)
    
    # Set axes
    ax.set_xlim(0, 12)
    ax.set_ylim(4, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.title('GRU Model Architecture', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('gru_model_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function"""
    print("Starting GRU sine wave prediction experiment...")
    
    # 1. Generate data
    print("Generating sine wave data...")
    X, y, t, data = generate_sine_wave_data()
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # 2. Create model
    print("Creating GRU model...")
    model = GRUModel(input_size=1, hidden_size=32, num_layers=2, output_size=1, dropout=0.1)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Generate model architecture diagram
    print("Generating model architecture diagram...")
    visualize_model_architecture()
    
    # 4. Train model
    train_losses = train_model(model, X, y, epochs=100, lr=0.01)
    
    # 5. Evaluate model
    predictions, mse, mae, rmse = evaluate_model(model, X, y)
    
    # 6. Visualize results
    print("Generating visualization results...")
    visualize_results(t, data, y, predictions, 10, train_losses)
    
    # 7. Save model
    torch.save(model.state_dict(), 'gru_sine_wave_model.pth')
    print("Model saved to: gru_sine_wave_model.pth")
    
    print("GRU sine wave prediction experiment completed!")
    print("Generated files:")
    print("- gru_model_architecture.png: Model architecture diagram")
    print("- gru_sine_wave_results.png: Prediction results visualization")
    print("- gru_sine_wave_model.pth: Trained model")
    
    # Clean up memory
    del model, X, y
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    main()
```

### Training Results
GRU learns the periodic pattern of the sine wave through its update and reset gates.
<img width="1124" height="404" alt="image" src="https://github.com/user-attachments/assets/7edf7421-1de7-49db-a6ad-59f663023739" />
<img width="1941" height="1283" alt="image" src="https://github.com/user-attachments/assets/539fc09d-fab6-4b79-ab8a-5d50d92fe6ec" />
