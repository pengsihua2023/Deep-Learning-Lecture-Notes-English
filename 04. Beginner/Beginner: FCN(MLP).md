## Beginner: FCN(MLP)
### **Fully Connected Neural Network (MLP)**  

### **Importance**: 
This is a foundational model in deep learning, and understanding it helps grasp core neural network concepts (e.g., neurons, weights, activation functions, gradient descent).  

### **Core Concepts**:  
- A neural network consists of an input layer, hidden layers, and an output layer, functioning like neurons in the brain.  
- Each neuron receives inputs, computes a weighted sum, and passes it through an activation function (e.g., sigmoid or ReLU) to produce an output.  
- Training involves adjusting weights through "trial and error" to make predictions closer to actual values (using gradient descent).  
- Applications: House price prediction, classification tasks (e.g., determining if an image is a cat or a dog).  

<img width="850" height="253" alt="image" src="https://github.com/user-attachments/assets/4f07aa2a-dd72-4e95-8543-7f71810d8023" />  
  
### Mathematical Description

A **fully connected network** (dense network) consists of multiple layers of neurons.  
Each neuron in one layer is connected to **all** neurons in the next layer.



### (1) Input

$$
\mathbf{x} \in \mathbb{R}^{d}
$$

The input vector with dimension $d$.  



### (2) Linear Transformation

$$
\mathbf{z}^{(l)} = W^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, 
\quad \mathbf{a}^{(0)} = \mathbf{x}
$$

*Computes the pre-activation for layer $l$.*



### (3) Activation

$$
\mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})
$$

*Applies the activation function elementwise.*



### (4) Output

$$
\mathbf{y} = \mathbf{a}^{(L)}
$$

*Final output of the network.*  

- For **regression**: $\sigma$ may be the identity function.  
- For **binary classification**: $\sigma$ is often sigmoid.  
- For **multi-class classification**: $\sigma$ is typically softmax.  

## code（pytorch）
```
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
## Training Results
Epoch [5/5], Step [800], Loss: 0.3124   
Epoch [5/5], Step [900], Loss: 0.2941   

Testing started...   
Test Accuracy: 87.24%   
  
$\begin{pmatrix} a & b \\ c & d \end{pmatrix}$   
