## Basics of Deep Learning: Introduction to Common Activation Functions

<img width="2054" height="896" alt="image" src="https://github.com/user-attachments/assets/c8e02fb5-bbe2-4a85-8e8c-727f86244505" />

### Sigmoid Activation Function
The Sigmoid activation function is a commonly used non-linear activation function, widely applied in neural networks, particularly in binary classification problems. Below is an introduction:

#### **Definition**
The Sigmoid function maps any real number to the range \((0, 1)\), making it suitable for tasks requiring probabilistic outputs (e.g., predicting probabilities in binary classification). It is defined as:

\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

Where:
- \( x \): Input value (can be a scalar, vector, or matrix).
- \( e \): Base of the natural logarithm (approximately 2.718).
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
- **Output Range**: \((0, 1)\), making it ideal for interpreting outputs as probabilities.
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
The Sigmoid activation function is a fundamental tool in neural networks, particularly for binary classification tasks. Its ability to map inputs to \((0, 1)\) makes it ideal for probabilistic outputs, but its limitations, such as vanishing gradients, should be considered when designing deep learning models.
