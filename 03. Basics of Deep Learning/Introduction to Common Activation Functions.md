## Basics of Deep Learning: Introduction to Common Activation Functions
## 1. Sigmoid Activation Function
<img width="1200" height="800" alt="image" src="https://github.com/user-attachments/assets/ffd061a2-165c-486a-83d0-e93fc62d9979" />

The Sigmoid activation function is a commonly used non-linear activation function, widely applied in neural networks, particularly in binary classification problems. Below is an introduction:

#### **Definition**
The Sigmoid function maps any real number to the range (0, 1), making it suitable for tasks requiring probabilistic outputs (e.g., predicting probabilities in binary classification). It is defined as:


$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$


Where:
-  x: Input value (can be a scalar, vector, or matrix).
- e : Base of the natural logarithm (approximately 2.718).
- Output: A value between 0 and 1.

#### **Implementation in Python**
Here’s a simple implementation of the Sigmoid function using `NumPy`:

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Example
x = np.array([-2, -1, 0, 1, 2])
print(sigmoid(x))  # Output: [0.11920292, 0.26894142, 0.5, 0.73105858, 0.88079708]
```

#### **Characteristics**
- **Output Range**: (0, 1), making it ideal for interpreting outputs as probabilities.
- **Non-linearity**: Introduces non-linearity to neural networks, enabling them to model complex relationships.
- **Smoothness**: The function is differentiable, which is crucial for backpropagation in neural networks.
- **Vanishing Gradient**: For large positive or negative inputs, the gradient of the Sigmoid function becomes very small, potentially causing slow convergence or vanishing gradient problems during training.

#### **Applications**
- **Binary Classification**: Commonly used in the output layer of neural networks for binary classification tasks, where the output represents the probability of belonging to a class.
- **Logistic Regression**: The Sigmoid function is the core of logistic regression, transforming linear combinations into probabilities.
- **Gating in LSTMs**: Used in Long Short-Term Memory (LSTM) networks to control information flow (e.g., forget gate, input gate).

#### **Example in Deep Learning**
Using the Sigmoid function in a neural network output layer with `PyTorch`:

```python
import torch
import torch.nn as nn

# Input tensor
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# Apply Sigmoid
sigmoid = nn.Sigmoid()
output = sigmoid(x)
print(output)  # Output: tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])
```

#### **Limitations**
- **Vanishing Gradient**: For inputs far from zero, gradients approach zero, slowing down learning in deep networks.
- **Non-Zero-Centered**: Outputs are always positive, which can complicate gradient updates in some cases.
- **Computationally Expensive**: The exponential operation in the Sigmoid function is computationally heavier compared to alternatives like ReLU.

#### **Alternatives**
Due to its limitations, other activation functions like **ReLU** (Rectified Linear Unit) or **tanh** are often preferred in hidden layers of deep neural networks, while Sigmoid remains popular in output layers for binary classification.

#### **Summary**
The Sigmoid activation function is a fundamental tool in neural networks, particularly for binary classification tasks. Its ability to map inputs to $(0, 1)$ makes it ideal for probabilistic outputs, but its limitations, such as vanishing gradients, should be considered when designing deep learning models.  

## 2. ReLU Activation Function  

The **ReLU (Rectified Linear Unit)** activation function is a widely used non-linear activation function in neural networks, particularly in deep learning models like convolutional neural networks (CNNs) and deep neural networks, due to its simplicity and effectiveness.
<img width="1387" height="793" alt="image" src="https://github.com/user-attachments/assets/5d1e30fd-7cc9-4b58-a157-04060102b558" />


#### **Definition**
The ReLU function outputs the input directly if it is positive; otherwise, it outputs zero. It is defined as:


f(x) = max(0, x)


Where:
-  x : Input value (can be a scalar, vector, or matrix).
- Output:  x if  x > 0, otherwise  0).

#### **Implementation in Python**
Here’s a simple implementation of the ReLU function using `NumPy`:

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

# Example
x = np.array([-2, -1, 0, 1, 2])
print(relu(x))  # Output: [0, 0, 0, 1, 2]
```

#### **Characteristics**
- **Output Range**: (0, infty), producing non-negative outputs.
- **Non-linearity**: Introduces non-linearity, enabling neural networks to model complex patterns.
- **Sparsity**: Outputs zero for negative inputs, leading to sparse activations, which can improve computational efficiency and reduce overfitting.
- **Differentiability**: ReLU is differentiable everywhere except at \( x = 0 \), where the derivative is typically defined as 0 or 1 for practicality in backpropagation (subgradient).
  - Derivative:  f'(x) = 1 if  x > 0, otherwise  0 .

#### **Applications**- **Hidden Layers**: ReLU is the default activation function for hidden layers in most deep learning models (e.g., CNNs, fully connected layers) due to its fast convergence and computational efficiency.
- **Deep Learning Frameworks**: Widely supported in frameworks like `TensorFlow` and `PyTorch` for efficient training.
- Example in `PyTorch`:
  ```python
  import torch
  import torch.nn as nn

  # Input tensor
  x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

  # Apply ReLU
  relu = nn.ReLU()
  output = relu(x)
  print(output)  # Output: tensor([0., 0., 0., 1., 2.])
  ```

#### **Advantages**
- **Fast Convergence**: Unlike Sigmoid or tanh, ReLU avoids vanishing gradient issues for positive inputs, as its gradient is either 0 or 1, leading to faster training.
- **Computational Efficiency**: The max operation is simpler than the exponential operations in Sigmoid or tanh.
- **Sparsity**: Zero outputs for negative inputs create sparse representations, which can enhance model efficiency and generalization.

#### **Limitations**
- **Dying ReLU Problem**: Neurons with negative inputs may output zero consistently, causing them to "die" (stop learning) if weights are not updated properly, especially with high learning rates.
- **Non-Zero-Centered**: Outputs are always non-negative, which can lead to biased gradient updates in some cases.
- **Not Differentiable at Zero**: While this is rarely a practical issue, the undefined derivative at \( x = 0 \) requires careful handling in optimization.

#### **Variants of ReLU**
To address ReLU’s limitations, several variants have been developed:
- **Leaky ReLU**:  f(x) = max(alpha x, x), wherealpha (e.g., 0.01) allows small negative outputs to prevent dying neurons.
- **Parametric ReLU (PReLU)**: Similar to Leaky ReLU, but alpha is a learnable parameter.
- **ELU (Exponential Linear Unit)**: Smooths negative inputs with an exponential function for better robustness.
- Example of Leaky ReLU in `PyTorch`:
  ```python
  leaky_relu = nn.LeakyReLU(negative_slope=0.01)
  output = leaky_relu(x)
  print(output)  # Small negative values for negative inputs
  ```

#### **Summary**
The ReLU activation function is a cornerstone of modern deep learning due to its simplicity, efficiency, and ability to mitigate vanishing gradient issues. It is primarily used in hidden layers of neural networks, transforming negative inputs to zero and preserving positive inputs. While it has limitations like the dying ReLU problem, its variants (e.g., Leaky ReLU) and careful model design can address these issues, making ReLU a preferred choice for most deep learning architectures.damental tool in neural networks, particularly for binary classification tasks. Its ability to map inputs to \((0, 1)\) makes it ideal for probabilistic outputs, but its limitations, such as vanishing gradients, should be considered when designing deep learning models.

## 3. Leaky ReLU activation function 
     
<img width="853" height="482" alt="image" src="https://github.com/user-attachments/assets/79ad6100-1bb7-47de-95df-f546be3edd0b" />    
### Mathematical Definition of Leaky ReLU

The Leaky ReLU (Rectified Linear Unit) is an activation function used in neural networks to introduce non-linearity. Unlike the standard ReLU, which outputs zero for negative inputs, Leaky ReLU allows a small, non-zero gradient for negative inputs to prevent the "dying ReLU" problem. It is defined as:

<img width="1201" height="400" alt="image" src="https://github.com/user-attachments/assets/84a96d64-3c49-4273-aec9-802153b7bd05" />
  

### Code Implementation in Python

Below is a simple implementation of the Leaky ReLU activation function using Python with NumPy, suitable for deep learning applications. This can be used in frameworks like TensorFlow or PyTorch or as a standalone function.

```python
import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU activation function.
    
    Parameters:
    - x: Input array (numpy array or scalar)
    - alpha: Slope for negative inputs (default: 0.01)
    
    Returns:
    - Output array after applying Leaky ReLU
    """
    return np.where(x > 0, x, alpha * x)
```

### Example Usage

```python
# Example inputs
x = np.array([-2, -1, 0, 1, 2])

# Apply Leaky ReLU
output = leaky_relu(x, alpha=0.01)
print(output)
# Output: [-0.02 -0.01  0.    1.    2.  ]
```

### Implementation in Deep Learning Frameworks

#### TensorFlow/Keras
In TensorFlow, Leaky ReLU is available as a built-in layer or function:

```python
import tensorflow as tf

# Define Leaky ReLU layer
leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)

# Example usage in a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(10,)),
    tf.keras.layers.LeakyReLU(alpha=0.01)
])
```

#### PyTorch
In PyTorch, Leaky ReLU is available as a module or functional API:

```python
import torch
import torch.nn as nn

# Define Leaky ReLU
leaky_relu = nn.LeakyReLU(negative_slope=0.01)

# Example input
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = leaky_relu(x)
print(output)
# Output: tensor([-0.0200, -0.0100,  0.0000,  1.0000,  2.0000])
```

### Notes
- The default \( \alpha \) (e.g., 0.01) can be adjusted based on the task, but small values are typically used.
- Leaky ReLU is computationally efficient and helps mitigate issues with vanishing gradients compared to standard ReLU.
- In practice, frameworks like TensorFlow and PyTorch optimize these operations for GPU acceleration.
  
## 4. Softmax activation function
<img width="384" height="300" alt="image" src="https://github.com/user-attachments/assets/f290108e-edcd-4408-b99e-4a0a35be16a3" />     
  ### Mathematical Definition of Softmax

<img width="1210" height="649" alt="image" src="https://github.com/user-attachments/assets/d9c5ce72-0b56-4982-b282-34e81ab8547a" />
  

### Code Implementation in Python

Below is a simple implementation of the Softmax function using NumPy, suitable for deep learning applications.

```python
import numpy as np

def softmax(x):
    """
    Softmax activation function.
    
    Parameters:
    - x: Input array (numpy array, typically 1D or 2D for batched inputs)
    
    Returns:
    - Output array with probabilities (same shape as input)
    """
    # Subtract the max for numerical stability
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
```

### Example Usage

```python
# Example input (logits)
x = np.array([2.0, 1.0, 0.1])

# Apply Softmax
output = softmax(x)
print(output)
# Output: [0.65900114 0.24243297 0.09856589]
print(np.sum(output))  # Should be ~1.0
# Output: 1.0
```

For batched inputs (e.g., 2D array where each row is a sample):

```python
x_batch = np.array([[2.0, 1.0, 0.1], [1.0, 3.0, 0.5]])
output_batch = softmax(x_batch)
print(output_batch)
# Output: [[0.65900114 0.24243297 0.09856589]
#          [0.19167327 0.70238467 0.10594206]]
```

### Implementation in Deep Learning Frameworks

#### TensorFlow/Keras
In TensorFlow, Softmax is available as a function or layer:

```python
import tensorflow as tf

# Softmax function
x = tf.constant([2.0, 1.0, 0.1])
output = tf.nn.softmax(x)
print(output)
# Output: [0.65900114 0.24243297 0.09856589]

# As a layer in a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, input_shape=(10,)),
    tf.keras.layers.Softmax()
])
```

#### PyTorch
In PyTorch, Softmax is available in the functional API or as a module:

```python
import torch
import torch.nn.functional as F

# Softmax function
x = torch.tensor([2.0, 1.0, 0.1])
output = F.softmax(x, dim=-1)
print(output)
# Output: tensor([0.6590, 0.2424, 0.0986])

# For batched inputs
x_batch = torch.tensor([[2.0, 1.0, 0.1], [1.0, 3.0, 0.5]])
output_batch = F.softmax(x_batch, dim=-1)
print(output_batch)
# Output: tensor([[0.6590, 0.2424, 0.0986],
#                 [0.1917, 0.7024, 0.1059]])
```

### Notes
- **Numerical Stability**: Subtracting the maximum value from the inputs before exponentiation (as shown in the NumPy implementation) prevents overflow issues with large logits.
- **Use Case**: Softmax is typically used in the output layer of a neural network for multi-class classification, where the output represents class probabilities.
- **Gradient**: Softmax is differentiable, making it suitable for backpropagation in neural networks.
- Frameworks like TensorFlow and PyTorch optimize Softmax for GPU acceleration and handle batched inputs efficiently.  
## 5. Tanh activation function
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/39f4bd79-59bc-437d-8e55-242656ca9ce2" />  
  
### Mathematical Definition of Tanh

The Tanh (Hyperbolic Tangent) activation function is a non-linear function commonly used in neural networks to introduce non-linearity. It maps input values to the range \((-1, 1)\). The Tanh function is defined as:

<img width="1158" height="653" alt="image" src="https://github.com/user-attachments/assets/81212078-62b3-4d41-b7c4-dab6459be247" />  
  

### Code Implementation in Python

Below is a simple implementation of the Tanh function using NumPy, suitable for deep learning applications.

```python
import numpy as np

def tanh(x):
    """
    Tanh activation function.
    
    Parameters:
    - x: Input array (numpy array or scalar)
    
    Returns:
    - Output array after applying Tanh
    """
    return np.tanh(x)
```

### Example Usage

```python
# Example inputs
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

# Apply Tanh
output = tanh(x)
print(output)
# Output: [-0.96402758 -0.76159416  0.          0.76159416  0.96402758]
```

### Implementation in Deep Learning Frameworks

#### TensorFlow/Keras
In TensorFlow, Tanh is available as a built-in function or layer:

```python
import tensorflow as tf

# Tanh function
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
output = tf.nn.tanh(x)
print(output)
# Output: [-0.9640276 -0.7615942  0.         0.7615942  0.9640276]

# As a layer in a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(10,)),
    tf.keras.layers.Activation('tanh')
])
```

#### PyTorch
In PyTorch, Tanh is available as a module or functional API:

```python
import torch
import torch.nn as nn

# Tanh function
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = torch.tanh(x)
print(output)
# Output: tensor([-0.9640, -0.7616,  0.0000,  0.7616,  0.9640])

# As a module
tanh = nn.Tanh()
output = tanh(x)
print(output)
# Output: tensor([-0.9640, -0.7616,  0.0000,  0.7616,  0.9640])
```

### Notes
- **Characteristics**: Tanh is zero-centered, which can make optimization easier compared to Sigmoid, but it may still suffer from vanishing gradients for large \( |x| \).
- **Use Case**: Tanh is often used in hidden layers of neural networks, particularly in recurrent neural networks (RNNs) like LSTMs, though ReLU variants are more common in modern feedforward networks.
- **Numerical Stability**: The NumPy and framework implementations handle numerical stability internally, but care should be taken with very large inputs to avoid overflow in custom implementations.
- Frameworks like TensorFlow and PyTorch optimize Tanh for GPU acceleration and support batched inputs efficiently.  


