## L2 Norm Regularization
### Mathematical Definition of L2 Norm Regularization

L2 Norm Regularization, also known as Ridge regularization or weight decay, is a technique used in machine learning and deep learning to prevent overfitting by adding a penalty term to the loss function. The penalty is based on the L2 norm (Euclidean norm) of the model's weights. The L2 regularization term is defined as:

<img width="1150" height="747" alt="image" src="https://github.com/user-attachments/assets/b6dd8435-1a07-47e5-af16-c30a4f283879" />
  

### Code Implementation in Python

Below is a simple implementation of L2 regularization in the context of a deep learning model. This example shows how to compute the L2 regularization term manually and how to use it in popular frameworks like TensorFlow and PyTorch.

#### Manual Implementation with NumPy
This computes the L2 regularization term for a set of weights.

```python
import numpy as np

def l2_regularization(weights, lambda_reg=0.01):
    """
    Compute L2 regularization term.
    
    Parameters:
    - weights: Array of model weights (numpy array or list of arrays)
    - lambda_reg: Regularization strength (default: 0.01)
    
    Returns:
    - L2 regularization term (scalar)
    """
    l2_term = 0
    if isinstance(weights, list):
        for w in weights:
            l2_term += np.sum(np.square(w))
    else:
        l2_term = np.sum(np.square(weights))
    return lambda_reg * l2_term

# Example usage
weights = [np.array([[1.0, -2.0], [0.5, 0.0]]), np.array([-1.0, 3.0])]
l2_loss = l2_regularization(weights, lambda_reg=0.01)
print(l2_loss)
# Output: 0.155 (sum of squared values = 15.5, scaled by 0.01)
```

### Implementation in Deep Learning Frameworks

#### TensorFlow/Keras
In TensorFlow/Keras, L2 regularization can be applied directly to layers using the `kernel_regularizer` argument.

```python
import tensorflow as tf
from tensorflow.keras import regularizers

# Define a model with L2 regularization
model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        64,
        input_shape=(10,),
        kernel_regularizer=regularizers.l2(l2=0.01),  # L2 regularization with lambda=0.01
        activation='relu'
    ),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model (loss includes L2 regularization automatically)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Example input and dummy data
x = tf.random.normal((100, 10))
y = tf.random.uniform((100,), maxval=3, dtype=tf.int32)
y = tf.keras.utils.to_categorical(y, num_classes=3)

# Train the model
# model.fit(x, y, epochs=5)
```

#### PyTorch
In PyTorch, L2 regularization is often implemented via weight decay in the optimizer, but it can also be added manually to the loss function.

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # L2 via weight_decay

# Alternative: Manual L2 regularization in training loop
def train_step(model, inputs, targets, lambda_reg=0.01):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Add L2 regularization
    l2_loss = 0
    for param in model.parameters():
        l2_loss += torch.norm(param, p=2) ** 2  # L2 norm (squared)
    loss += lambda_reg * l2_loss
    
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
- **Weight Decay vs. L2 Regularization**: In PyTorch, `weight_decay` in optimizers (e.g., Adam, SGD) implements L2 regularization by adding the squared norm of weights to the loss, scaled by the weight decay factor. This is equivalent to L2 regularization in most cases.
- **Effect**: L2 regularization penalizes large weights, leading to smoother models but does not promote sparsity (unlike L1 regularization).
- **Hyperparameter Tuning**: The regularization strength lambda (e.g., 0.01) is critical and typically tuned via cross-validation.
- **Use Case**: L2 regularization is widely used in neural networks to improve generalization, especially when overfitting is a concern.
- **Frameworks**: TensorFlow/Keras integrates L2 regularization into layers, while PyTorch typically uses weight decay in optimizers or manual addition to the loss.
- **Numerical Stability**: Frameworks handle numerical stability internally, but in custom implementations, ensure weights are properly scaled to avoid numerical issues.
