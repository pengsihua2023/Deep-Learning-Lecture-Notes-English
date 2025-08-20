## Basics of Deep Learning: Introduction to Common Activation Functions

<img width="2054" height="896" alt="image" src="https://github.com/user-attachments/assets/c8e02fb5-bbe2-4a85-8e8c-727f86244505" />

### Sigmoid Activation Function
<img width="1200" height="800" alt="image" src="https://github.com/user-attachments/assets/ffd061a2-165c-486a-83d0-e93fc62d9979" />

The Sigmoid activation function is a commonly used non-linear activation function, widely applied in neural networks, particularly in binary classification problems. Below is an introduction:

#### **Definition**
The Sigmoid function maps any real number to the range (0, 1), making it suitable for tasks requiring probabilistic outputs (e.g., predicting probabilities in binary classification). It is defined as:

<img width="217" height="93" alt="image" src="https://github.com/user-attachments/assets/b7990223-86af-4e09-aeaa-734fb3e577b9" />


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

## ReLU Activation Function

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
## 4. Softmax activation function
<img width="384" height="300" alt="image" src="https://github.com/user-attachments/assets/f290108e-edcd-4408-b99e-4a0a35be16a3" />   
## 5. Tanh activation function
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/39f4bd79-59bc-437d-8e55-242656ca9ce2" /> 



