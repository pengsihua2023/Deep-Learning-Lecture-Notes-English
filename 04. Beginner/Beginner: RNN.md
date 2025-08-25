## Beginner: RNN
### Recurrent Neural Network
* Importance: RNNs are suitable for processing sequential data (such as text and speech) and are fundamental in natural language processing and time series prediction.
* Core Concepts:
  RNNs have "memory," which allows them to remember previous inputs, making them suitable for handling ordered data (such as sentences).
  Variants such as LSTM (Long Short-Term Memory) can remember longer sequences.
* Applications: Speech recognition (e.g., Siri), machine translation, stock prediction.
* Why teach: RNNs demonstrate how deep learning processes dynamic data, relevant to applications such as voice assistants.

<img width="956" height="304" alt="image" src="https://github.com/user-attachments/assets/ecdeb7fe-d4e1-4ef1-b7b9-9e766e6bf9bd" />  

RNN: The basic recurrent neural network unit processes the input \$X\_t\$ and the hidden state \$h(t-1)\$ from the previous time step through a tanh activation function, generating the current hidden state \$h(t)\$. It is simple but prone to the vanishing gradient problem, which limits its ability to handle long sequences.
RNNs are considered to have "memory" because they transmit information across time steps through the hidden state \$h(t)\$. The hidden state at the current time step depends not only on the current input \$x(t)\$ but also on the hidden state \$h(t-1)\$ of the previous step, thereby "remembering" parts of the previous sequence. This makes them suitable for handling sequential data, such as time series or natural language. However, the memory ability of standard RNNs is limited, and they are easily affected by the vanishing gradient problem, making it difficult to capture long-term dependencies.

---

### 1. Basic Structure of RNN

RNNs are neural networks specialized in processing sequential data by introducing hidden states to capture temporal dependencies in sequences.
The core idea: the output at the current time depends not only on the current input but also on the previous hidden state.

### Basic Formula

For time step \$t\$, the RNN computation is as follows:
Hidden state update



$$
h_t = \sigma(W_{xh}x_t + W_{hh}h_{t-1} + b_h)
$$

* \$x\_t\$: input vector at time step \$t\$
* \$h\_t\$: hidden state at time step \$t\$
* \$h\_{t-1}\$: hidden state at the previous time step
* \$W\_{xh}\$: weight matrix from input to hidden layer
* \$W\_{hh}\$: weight matrix from hidden layer to hidden layer
* \$b\_h\$: bias of the hidden layer
* \$\sigma\$: activation function (commonly \$\tanh\$ or ReLU)

**Output:**

$$
y_t = W_{hy}h_t + b_y
$$

* \$y\_t\$: output at time step \$t\$
* \$W\_{hy}\$: weight matrix from hidden layer to output layer
* \$b\_y\$: bias of the output layer

If a nonlinear output is needed (e.g., classification tasks), apply an activation function (such as softmax) to \$y\_t\$:

$$
o_t = \text{softmax}(y_t)
$$

---

### 2. Forward Propagation

RNNs iteratively compute hidden states and outputs through time steps. For a sequence \$x\_1, x\_2, \ldots, x\_T\$, the forward propagation process is:

1. Initialize \$h\_0\$ (commonly as a zero vector or random initialization).
2. For each time step \$t = 1, 2, \ldots, T\$:

   * Compute \$h\_t = \sigma(W\_{xh}x\_t + W\_{hh}h\_{t-1} + b\_h)\$
   * Compute \$y\_t = W\_{hy}h\_t + b\_y\$
3. Depending on the task, either collect all \$y\_t\$ or use only the final output \$y\_T\$.

---

### 3. Loss Function

RNNs typically use a loss function to measure the gap between predicted outputs and true labels. For sequence prediction tasks, cross-entropy loss (classification) or mean squared error (regression) is commonly used. The total loss is the sum of losses over all time steps:

$$
L = \sum_{t=1}^{T} L_t(\hat{y}_t, y_t)
$$

where \$L\_t\$ is the loss at time step \$t\$, \$\hat{y}\_t\$ is the predicted output, and \$y\_t\$ is the true label.

---

### 4. Backpropagation Through Time (BPTT)

RNN training unfolds along time steps through backpropagation, known as **BPTT**. The goal is to minimize the loss function \$L\$ by updating weights \$W\_{xh}, W\_{hh}, W\_{hy}\$ and biases \$b\_h, b\_y\$ via gradient descent.

### Gradient Calculation

* **For each time step \$t\$, compute the gradient of loss with respect to the hidden state:**

$$
\frac{\partial L}{\partial h_t} = \frac{\partial L_t}{\partial h_t} + \frac{\partial L_{t+1}}{\partial h_t} + \cdots + \frac{\partial L_T}{\partial h_t}
$$

where \$\frac{\partial L\_{t+k}}{\partial h\_t}\$ is recursively computed via the chain rule across time steps.

* **Weight gradients:**

$$
\frac{\partial L}{\partial W_{xh}} = \sum_{t=1}^{T} \frac{\partial L}{\partial h_t} \cdot \frac{\partial h_t}{\partial W_{xh}}
$$

Gradients for \$W\_{hh}, W\_{hy}, b\_h, b\_y\$ are calculated similarly.

---

### 5. Vanishing / Exploding Gradient Problem

Since \$h\_t\$ depends on \$h\_{t-1}\$, gradients propagate through repeated multiplications with the matrix \$W\_{hh}\$, which can lead to:

* **Vanishing gradients**: gradients become too small, making it hard to propagate influence from earlier steps.
* **Exploding gradients**: gradients become too large, leading to unstable training.


### Solutions

* **Gradient clipping** (limit gradient magnitude);
* **More advanced architectures**: such as LSTM and GRU.

---

## Code (Pytorch)

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
## Training Results

Training started...
Epoch \[5/20], Loss: 0.6591
Epoch \[10/20], Loss: 0.6286
Epoch \[15/20], Loss: 0.5319
Epoch \[20/20], Loss: 0.3664

Testing started...
Test Accuracy: 79.00%

<img width="791" height="385" alt="image" src="https://github.com/user-attachments/assets/ea2bbd16-e1b1-4334-a856-d1e91223a8a7" />  
Figure 2 Training Loss Curve  

<img width="396" height="295" alt="image" src="https://github.com/user-attachments/assets/4c5c3a4d-986f-4b42-8612-9a05ed4e3f87" />  
Figure 3 Confusion Matrix  

## Code Function Overview

This code implements the following functions:

* **1. Function Overview**
  The code implements a simple PyTorch-based RNN for binary sequence classification, with visualization of training loss curve and test set confusion matrix. The main workflow includes data generation, model definition, training, testing, and visualization.

* **2. Main Steps**

  * (1) Data Generation
    Randomly generate binary sequences (0 or 1), each of length 10.
    Label rule: if the number of 1s in the sequence exceeds half, the label is 1; otherwise, it is 0.
    Generate 1000 training samples and 200 test samples.

  * (2) Model Definition
    A simple RNN model (SimpleRNN) is constructed, taking sequence data as input and outputting the probability distribution of two classes.
    The RNN outputs the hidden state of the last time step, which is mapped to 2 classes through a fully connected layer.

  * (3) Training Process
    Uses cross-entropy loss function and Adam optimizer.
    Trains for 20 epochs, records the loss of each epoch, and prints the loss every 5 epochs.

  * (4) Testing and Evaluation
    Evaluates the model on the test set and outputs accuracy.
    Returns predicted labels and true labels.

  * (5) Visualization
    Plots the training loss curve to intuitively show the model convergence process.
    Plots the confusion matrix to show classification performance on the test set (correct/incorrect counts).

* **3. Applicable Scenarios**
  Suitable for introductory demonstrations of sequence binary classification tasks, such as simple time series signals, text, or event streams.
  Can serve as a template for RNN model structure, training, and evaluation workflow.

* **4. Summary**
  This code implements an end-to-end RNN binary sequence classification task, including data generation, model training, testing, evaluation, and visualization. It is suitable for beginners in deep learning to understand the basic usage of RNNs and classification workflow.

---
