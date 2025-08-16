
## Beginner: RNN
Recurrent Neural Network (RNN)  
- Importance: RNN is suitable for processing sequential data (e.g., text, speech) and serves as a foundation for natural language processing and time series prediction.  
- Core Concept:  
RNN has a "memory" that retains previous inputs, making it ideal for handling ordered data (e.g., sentences).  
Variants like LSTM (Long Short-Term Memory) can remember longer sequences.  
- Applications: Speech recognition (e.g., Siri), machine translation, stock prediction.  
- Why Teach: RNN demonstrates how deep learning handles dynamic data, relating to applications like voice assistants.

<img width="956" height="304" alt="image" src="https://github.com/user-attachments/assets/ecdeb7fe-d4e1-4ef1-b7b9-9e766e6bf9bd" />  

RNN: The basic recurrent neural network unit processes the input x(t) and the previous time step’s hidden state h(t-1) through a tanh activation function to generate the current hidden state h(t). It is simple but prone to the vanishing gradient problem, which limits its ability to handle long sequences.  
RNN (Recurrent Neural Network) is considered to have “memory” because it passes information between time steps through the hidden state h(t). The current hidden state depends not only on the current input x(t) but also on the previous hidden state h(t-1), allowing it to “remember” some information from prior parts of the sequence. This structure makes it suitable for processing sequential data, such as time series or natural language. However, the standard RNN’s memory capacity is limited, and it is affected by the vanishing gradient problem, making it difficult to capture long-range dependencies.  

## Code（Pytorch）
```
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Display Chinese
matplotlib.rcParams['axes.unicode_minus'] = False    # Display negative sign

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Set random seed to ensure reproducible results
torch.manual_seed(42)
np.random.seed(42)

# 1. Generate simple sequence data
def generate_sequence_data(num_samples, seq_length):
    data = []
    labels = []
    for _ in range(num_samples):
        # Randomly generate sequence (0 or 1)
        seq = np.random.randint(0, 2, size=(seq_length,))
        # Label: Whether the number of 1s in the sequence exceeds half
        label = 1 if np.sum(seq) > seq_length // 2 else 0
        data.append(seq)
        labels.append(label)
    return torch.FloatTensor(data).unsqueeze(-1), torch.LongTensor(labels)

# Data parameters
num_samples = 1000
seq_length = 10
input_size = 1
hidden_size = 16
num_classes = 2

# Generate training and test data
train_data, train_labels = generate_sequence_data(num_samples, seq_length)
test_data, test_labels = generate_sequence_data(num_samples // 5, seq_length)

# 2. Define RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        # RNN forward propagation
        out, _ = self.rnn(x, h0)
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

# 3. Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleRNN(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Train the model
def train_model(num_epochs=20):
    model.train()
    loss_list = []  # Record loss for each epoch
    for epoch in range(num_epochs):
        inputs, labels = train_data.to(device), train_labels.to(device)
        
        # Forward propagation
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward propagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())  # Save loss
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    return loss_list  # Return loss list

# 5. Test the model
def test_model():
    model.eval()
    with torch.no_grad():
        inputs, labels = test_data.to(device), test_labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        return predicted.cpu().numpy(), labels.cpu().numpy()  # Return predicted and true labels

# 6. Visualization functions
def plot_loss_curve(loss_list):
    """Plot training loss curve"""
    plt.figure(figsize=(8, 4))
    plt.plot(loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(pred, true):
    """Plot confusion matrix"""
    cm = confusion_matrix(true, pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

# 7. Execute training and testing
if __name__ == "__main__":
    print("Training started...")
    loss_list = train_model(num_epochs=20)
    print("\nTesting started...")
    pred, true = test_model()
    # Visualization
    plot_loss_curve(loss_list)
    plot_confusion_matrix(pred, true)
```

## Training results
Training started...
Epoch [5/20], Loss: 0.6591  
Epoch [10/20], Loss: 0.6286  
Epoch [15/20], Loss: 0.5319  
Epoch [20/20], Loss: 0.3664  

Testing started...  
Test Accuracy: 79.00%  

<img width="791" height="385" alt="image" src="https://github.com/user-attachments/assets/ea2bbd16-e1b1-4334-a856-d1e91223a8a7" /> 
Figure 2 Training loss curve  

<img width="396" height="295" alt="image" src="https://github.com/user-attachments/assets/4c5c3a4d-986f-4b42-8612-9a05ed4e3f87" />  

Figure 3 Confusion matrix  

## Code Functionality Overview
The functionality of this code is briefly described as follows:

- 1. **Function Overview**  
The code implements a simple RNN sequence binary classification model based on PyTorch, including visualization of the training loss curve and the confusion matrix for the test set. The main workflow includes data generation, model definition, training, testing, and visualization.

- 2. **Main Steps Explanation**  
   - (1) **Data Generation**  
     Randomly generates binary sequences (0 or 1), each with a length of 10.  
     Labeling rule: If the number of 1s in the sequence exceeds half, the label is 1; otherwise, it is 0.  
     Generates 1000 training samples and 200 test samples.  
   - (2) **Model Definition**  
     Constructs a simple RNN model (SimpleRNN) that takes sequence data as input and outputs a probability distribution for binary classification.  
     The RNN outputs the hidden state of the last time step, which is mapped to two classes via a fully connected layer.  
   - (3) **Training Process**  
     Uses cross-entropy loss function and Adam optimizer.  
     Trains for 20 epochs, records the loss for each epoch, and prints the loss every 5 epochs.  
   - (4) **Testing and Evaluation**  
     Evaluates the model on the test set and outputs the accuracy.  
     Returns predicted labels and true labels.  
   - (5) **Visualization**  
     Plots the training loss curve to visually demonstrate the model’s convergence process.  
     Plots the confusion matrix to show the model’s classification performance on the test set (number of correct/incorrect classifications).  

- 3. **Applicable Scenarios**  
Suitable for introductory demonstrations of sequence binary classification tasks, such as simple time-series signals, text, or event streams.  
Can serve as a template for understanding RNN model structure, training, and evaluation workflows.  

- 4. **Summary**  
This code implements an end-to-end RNN sequence binary classification task, including data generation, model training, testing, evaluation, and visualization. It is suitable for deep learning beginners to understand the basic usage of RNNs and the classification process.
