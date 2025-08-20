## L1 Norm Regularization (L1 Regularization)
### Mathematical Definition of L1 Norm Regularization

L1 Norm Regularization, also known as Lasso regularization, is a technique used in machine learning and deep learning to prevent overfitting by adding a penalty term to the loss function. The penalty is based on the L1 norm (absolute value) of the model's weights. The L1 regularization term is defined as:

<img width="1066" height="492" alt="image" src="https://github.com/user-attachments/assets/85bbf436-023a-4cfc-96ed-1cbbbf270d7f" />


Where:
- Data Loss is the original loss (e.g., mean squared error for regression or cross-entropy for classification).
- The L1 term encourages sparsity, often driving some weights to exactly zero, which can lead to feature selection.

### Code Implementation in Python

Below is a simple implementation of L1 regularization in the context of a deep learning model. This example shows how to compute the L1 regularization term manually and how to use it in popular frameworks like TensorFlow and PyTorch.

#### Manual Implementation with NumPy
This computes the L1 regularization term for a set of weights.

```python
import numpy as np

def l1_regularization(weights, lambda_reg=0.01):
    """
    Compute L1 regularization term.
    
    Parameters:
    - weights: Array of model weights (numpy array or list of arrays)
    - lambda_reg: Regularization strength (default: 0.01)
    
    Returns:
    - L1 regularization term (scalar)
    """
    l1_term = 0
    if isinstance(weights, list):
        for w in weights:
            l1_term += np.sum(np.abs(w))
    else:
        l1_term = np.sum(np.abs(weights))
    return lambda_reg * l1_term

# Example usage
weights = [np.array([[1.0, -2.0], [0.5, 0.0]]), np.array([-1.0, 3.0])]
l1_loss = l1_regularization(weights, lambda_reg=0.01)
print(l1_loss)
# Output: 0.075 (sum of absolute values = 7.5, scaled by 0.01)
```

### Implementation in Deep Learning Frameworks

#### TensorFlow/Keras
In TensorFlow/Keras, L1 regularization can be applied directly to layers using the `kernel_regularizer` argument.

```python
import tensorflow as tf
from tensorflow.keras import regularizers

# Define a model with L1 regularization
model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        64,
        input_shape=(10,),
        kernel_regularizer=regularizers.l1(l1=0.01),  # L1 regularization with lambda=0.01
        activation='relu'
    ),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model (loss includes L1 regularization automatically)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Example input and dummy data
x = tf.random.normal((100, 10))
y = tf.random.uniform((100,), maxval=3, dtype=tf.int32)
y = tf.keras.utils.to_categorical(y, num_classes=3)

# Train the model
# model.fit(x, y, epochs=5)
```

#### PyTorch
In PyTorch, L1 regularization is not built into the layers but can be added to the loss function manually.

```python
import torch
import torch.nn as nn

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss, and optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lambda_reg = 0.01

# Example training loop with L1 regularization
def train_step(model, inputs, targets):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Add L1 regularization
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.norm(param, p=1)  # L1 norm
    loss += lambda_reg * l1_loss
    
    loss.backward()
    optimizer.step()
    return loss.item()

# Example data
inputs = torch.randn(100, 10)
targets = torch.randint(0, 3, (100,))
loss = train_step(model, inputs, targets)
print(loss)
```

### Notes
- **Sparsity**: L1 regularization promotes sparsity, often setting some weights to zero, which can simplify the model and act as a form of feature selection.
- **Hyperparameter Tuning**: The regularization strength \( \lambda \) (e.g., 0.01) is critical and typically tuned via cross-validation.
- **Use Case**: L1 regularization is useful when you want a sparse model or when dealing with high-dimensional data.
- **Frameworks**: TensorFlow/Keras integrates L1 regularization directly into layers, while PyTorch requires manual addition to the loss function.
- **Numerical Stability**: Frameworks handle numerical stability internally, but in custom implementations, ensure weights are properly scaled to avoid numerical issues.
