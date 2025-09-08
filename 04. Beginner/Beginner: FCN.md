# Beginner: FCN
#### **Fully Connected Neural Network**  
## 📖 Definition

A Fully Connected Neural Network (FCNN), also known as a Feedforward Neural Network, is one of the most basic structures in artificial neural networks. In this type of network, **every neuron in one layer is connected to every neuron in the next layer**. That means the output of one layer serves as the weighted input for all neurons in the following layer.

<div align="center">
<img width="700" height="210" alt="image" src="https://github.com/user-attachments/assets/4f07aa2a-dd72-4e95-8543-7f71810d8023" />  
</div>


<div align="center">
(This picture was obtained from the Internet.)
</div>

## 📖 Structural Features

1. **Input Layer**: Receives the raw data (such as feature vectors).
2. **Hidden Layers**: Consist of multiple neurons that perform weighted summations followed by nonlinear transformations via activation functions, extracting and combining features.
3. **Output Layer**: Produces the final prediction or classification result.
4. **Weights & Biases**: Each connection has a weight parameter, and each neuron usually has a bias term.
5. **Activation Function**: Introduces nonlinearity in hidden or output layers, allowing the network to approximate complex functions.

## 📖 Characteristics and Applications

* **Advantages**: Simple structure, highly general, and capable of approximating any continuous function (Universal Approximation Theorem).
* **Disadvantages**: Large number of parameters (especially with high-dimensional inputs), prone to overfitting, and relatively inefficient to train.
* **Applications**: Commonly used in early machine learning tasks such as classification and regression, e.g., handwritten digit recognition (MNIST), or structured/tabular data prediction.


## 📖 Mathematical Description

### 1. Network Structure

A typical fully connected neural network consists of several **layers**:

* Input layer  
* One or more hidden layers  
* Output layer  

In a fully connected structure, **each neuron in a given layer is connected to all neurons in the previous layer**.



### 2. Mathematical Notation

Let:

* Input vector:

$$
\mathbf{x} \in \mathbb{R}^{d}
$$

* The $l$-th layer has $n_l$ neurons, with output:

$$
\mathbf{h}^{(l)} \in \mathbb{R}^{n_l}
$$

* Weight matrix and bias:

$$
\mathbf{W}^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}, \quad \mathbf{b}^{(l)} \in \mathbb{R}^{n_l}
$$

* Activation function:

$$
\sigma(\cdot)
$$



### 3. Forward Propagation

The input layer is defined as:

$$
\mathbf{h}^{(0)} = \mathbf{x}
$$

For the $l$-th layer ($l=1,2,\dots,L$):

1. **Linear transformation:**

$$
\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}
$$

2. **Nonlinear activation:**

$$
\mathbf{h}^{(l)} = \sigma\left(\mathbf{z}^{(l)}\right)
$$

Finally, the output layer result is:

$$
\hat{\mathbf{y}} = \mathbf{h}^{(L)}
$$



### 4. Loss Function

During training, given target output $\mathbf{y}$, common loss functions include:

* **Regression (MSE):**

$$
\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}) = \frac{1}{N}\sum_{i=1}^N \|\hat{\mathbf{y}}^{(i)} - \mathbf{y}^{(i)}\|^2
$$

* **Classification (Cross-Entropy):**

$$
\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}) = - \sum_{k=1}^K y_k \log \hat{y}_k
$$



### 5. Parameter Update (Backpropagation + Gradient Descent)

By backpropagation, we compute the gradients of the loss function with respect to the parameters:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}}
$$

Then update them using gradient descent or its variants (e.g., Adam, SGD, RMSProp):

$$
\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}
$$

$$
\mathbf{b}^{(l)} \leftarrow \mathbf{b}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}}
$$

where $\eta$ is the learning rate.



### 6. Summary

In summary, a fully connected neural network can be abstracted as:

$$
\hat{\mathbf{y}} = f(\mathbf{x}; \Theta) = \sigma^{(L)}\Big(\mathbf{W}^{(L)} \sigma^{(L-1)}(\cdots \sigma^{(1)}(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}) \cdots ) + \mathbf{b}^{(L)}\Big)
$$

where $\Theta = \{\mathbf{W}^{(l)}, \mathbf{b}^{(l)} \mid l=1,\dots,L\}$ represents the set of model parameters.


## code（pytorch）
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Set the random seed to ensure reproducible results
torch.manual_seed(42)

# 1. Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Loading the Fashion MNIST dataset
trainset = torchvision.datasets.FashionMNIST(
    root='./data', 
    train=True,
    download=True, 
    transform=transform
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.FashionMNIST(
    root='./data', 
    train=False,
    download=True, 
    transform=transform
)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# 2. Defining the Neural Network Model
class FashionMNISTClassifier(nn.Module):
    def __init__(self):
        super(FashionMNISTClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer to hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)      # Hidden layer to output layer (10 categories)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 3. Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionMNISTClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training model
def train_model(num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward Propagation
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backpropagation and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

# 5. Testing model
def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# 6. Perform training and testing
if __name__ == "__main__":
    print("Training started...")
    train_model(num_epochs=5)
    print("\nTesting started...")
    test_model()

```
### Training Results
Epoch [5/5], Step [800], Loss: 0.3124   
Epoch [5/5], Step [900], Loss: 0.2941   

Testing started...   
Test Accuracy: 87.24%   
  
$\begin{pmatrix} a & b \\ c & d \end{pmatrix}$   
