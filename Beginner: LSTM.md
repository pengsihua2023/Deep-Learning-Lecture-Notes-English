## Beginner: LSTM

LSTM (Long Short-Term Memory)  
- Importance:  
LSTM is a variant of RNN (Recurrent Neural Network) that addresses the vanishing gradient problem of standard RNNs, enabling it to retain information over longer sequences.  
It is widely used in tasks such as speech recognition, time series prediction, and text generation, making it a classic model for sequence modeling.  
- Core Concept:  
LSTM uses "gate mechanisms" (input gate, forget gate, output gate) to control the retention and forgetting of information, making it suitable for processing long sequence data.  
- Applications: Voice assistants (e.g., Siri), stock price prediction, machine translation.

<img width="969" height="304" alt="image" src="https://github.com/user-attachments/assets/1d0be4d9-f07c-4428-9aaa-cfe0ecbfd411" />  

- LSTM: Introduces a gating mechanism to better handle long-term dependencies. It includes:  
   - Forget gate: Determines how much information from the previous time step to discard.  
   - Input gate: Controls how much of the new input x(t) is incorporated.  
   - Output gate: Determines the output of the current hidden state h(t).  
   - LSTM manages information flow through these gates, mitigating the vanishing gradient problem. 

## Code （Pytorch）

Here's the code with Chinese comments translated to English, keeping the code unchanged:

```
import os
import urllib.request
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# Download and extract UCI HAR dataset
def download_and_extract_har():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    zip_path = "UCI_HAR_Dataset.zip"
    data_dir = "UCI HAR Dataset"
    if not os.path.exists(data_dir):
        print("Downloading UCI HAR dataset...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        print("Dataset is ready.")
    else:
        print("Dataset already exists.")
    return data_dir

# Load HAR dataset (binary classification: WALKING vs. others)
def load_har_binary(data_dir):
    X_train = np.loadtxt(os.path.join(data_dir, "train", "X_train.txt"))
    y_train = np.loadtxt(os.path.join(data_dir, "train", "y_train.txt"))
    X_test = np.loadtxt(os.path.join(data_dir, "test", "X_test.txt"))
    y_test = np.loadtxt(os.path.join(data_dir, "test", "y_test.txt"))
    y_train = (y_train == 1).astype(np.float32)
    y_test = (y_test == 1).astype(np.float32)
    def reshape_X(X):
        X = X[:, :558]  # 9*62=558
        return X.reshape(X.shape[0], 9, 62)
    X_train = reshape_X(X_train)
    X_test = reshape_X(X_test)
    return X_train, y_train, X_test, y_test

# LSTM model definition (same as before)
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

# Training, testing, and visualization functions (same as before)
def train_model(train_data, train_labels, model, criterion, optimizer, batch_size=64, epochs=10):
    model.train()
    loss_list = []
    for epoch in range(epochs):
        permutation = torch.randperm(train_data.size(0))
        epoch_loss = 0
        for i in range(0, train_data.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = train_data[indices].to(device), train_labels[indices].to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        avg_loss = epoch_loss / train_data.size(0)
        loss_list.append(avg_loss)
        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    return loss_list

def batch_test(data, labels, model):
    model.eval()
    with torch.no_grad():
        inputs, targets = data.to(device), labels.to(device)
        outputs = model(inputs)
        preds = (outputs > 0.5).float()
        accuracy = (preds == targets).float().mean().item()
        print(f'Batch test accuracy: {accuracy*100:.2f}%')
        return preds.cpu().numpy(), targets.cpu().numpy()

def plot_loss_curve(loss_list):
    plt.figure(figsize=(8,4))
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main program
if __name__ == "__main__":
    # Download and load data
    data_dir = download_and_extract_har()
    X_train, y_train, X_test, y_test = load_har_binary(data_dir)
    print(f"Training set samples: {X_train.shape[0]}, Test set samples: {X_test.shape[0]}")
    # Convert to torch tensors
    train_data = torch.tensor(X_train, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    test_data = torch.tensor(X_test, dtype=torch.float32)
    test_labels = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    # Parameters
    input_size = train_data.shape[2]
    hidden_size = 32
    output_size = 1
    seq_length = train_data.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleLSTM(input_size, hidden_size, output_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # Training
    print("Starting training...")
    loss_list = train_model(train_data, train_labels, model, criterion, optimizer, batch_size=128, epochs=10)
    print("Training finished, starting batch testing...")
    batch_test(test_data, test_labels, model)
    plot_loss_curve(loss_list)
```

## Results
Begin to training...  
Epoch [2/10], Loss: 0.0628  
Epoch [4/10], Loss: 0.0118  
Epoch [6/10], Loss: 0.0154  
Epoch [8/10], Loss: 0.0014  
Epoch [10/10], Loss: 0.0003  
Training is over, batch testing begins...  
Batch test accuracy: 98.71%  

<img width="785" height="388" alt="image" src="https://github.com/user-attachments/assets/983ef728-8e77-4cb7-9553-43494e967bda" /> 
**Figure 2: Training Loss Curve**

## Code Functionality Explanation

The code implements the training and evaluation of an LSTM binary classification model based on the UCI HAR (Human Activity Recognition) real dataset. Its main functionalities are as follows:  
- 1. **Automatic Download and Extraction of Real Data**  
  Automatically downloads the "Human Activity Recognition (HAR)" dataset from the UCI website and extracts it, eliminating the need for manual data preparation.  
- 2. **Data Preprocessing**  
  Reads the feature data (X_train, X_test) and labels (y_train, y_test) from the training and test sets.  
  Trims the original 561-dimensional features of each sample to 558 dimensions (9×62) and reshapes them into a 9-step sequence with 62 dimensions per step, suitable for LSTM input.  
  Binarizes the labels: distinguishes only between WALKING (label 1) and non-WALKING (label 0) for binary classification.  
- 3. **LSTM Model Definition**  
  Constructs a simple LSTM neural network that takes sequence data (9 steps, 62 dimensions per step) as input and outputs a binary classification probability.  
  The LSTM outputs the hidden state of the last time step, which is passed through a fully connected layer and a Sigmoid activation to obtain the classification probability.  
- 4. **Training and Testing Process**  
  Uses BCELoss (binary cross-entropy loss) and the Adam optimizer for training.  
  Supports batch training and batch testing, automatically calculating and outputting training loss and test accuracy.  
  Records the loss during training for subsequent visualization.  
- 5. **Visualization**  
  Plots the training loss curve to help observe the model's convergence behavior.  

### Overall Functionality Summary  
This code implements an end-to-end LSTM sequence binary classification pipeline, including:  
- Automatic downloading and processing of a real public dataset  
- Data preprocessing and serialization  
- LSTM model construction  
- Training, testing, and accuracy evaluation  
- Loss curve visualization  
**Applicable Scenarios**: Any task requiring LSTM for binary classification of real sequential data, with the ability to be directly adapted to other datasets with similar structures.
