# SGD Optimizer (Stochastic Gradient Descent)


## 📖 **Principle**

SGD (Stochastic Gradient Descent) is a classic optimization algorithm widely used in training deep learning models. Its core idea is to compute the gradient of the loss function with respect to model parameters and update the parameters in the opposite direction of the gradient to minimize the loss. The "stochastic" nature of SGD comes from the fact that each update uses only one sample or a mini-batch of data, rather than the entire dataset, which accelerates computation.

1. **Core Formula**:

* **Parameter Update Rule**:

$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L(\theta_t, x_i, y_i)
$$

Where:

* \$\theta\_t\$: Current parameter.

* \$\eta\$: Learning rate, controls the step size.

* \$\nabla\_\theta L\$: Gradient of the loss function \$L\$ with respect to the parameters, based on sample \$(x\_i, y\_i)\$ or mini-batch data.

* **If Momentum is used, the update rule becomes**:

$$
v_t = \gamma v_{t-1} + \eta \cdot \nabla_\theta L(\theta_t)
$$

$$
\theta_{t+1} = \theta_t - v_t
$$

Where \$\gamma\$ is the momentum coefficient (typically 0.9), and \$v\_t\$ is the velocity (accumulated historical gradients).

2. **Characteristics**:

   * **Advantages**:

     * Computationally efficient, suitable for large-scale datasets.
     * Randomness helps escape local minima.
   * **Disadvantages**:

     * Gradient noise is high, which may lead to unstable convergence.
     * Requires manual tuning of learning rate \$\eta\$ and momentum \$\gamma\$.
   * **Use Cases**: Image classification, text classification, regression, and other deep learning tasks. Suitable for simple optimization or as a baseline algorithm.

3. **Comparison with Adadelta**:

   * SGD requires manually setting the learning rate, while Adadelta adjusts it adaptively.
   * SGD is simple and straightforward, while Adadelta introduces exponential moving averages of squared gradients and squared updates, making it more complex but more robust.

---

## 📖 **PyTorch Usage**

PyTorch provides a built-in `torch.optim.SGD` optimizer, supporting both vanilla SGD and the momentum variant. Below is a minimal code example showing how to use SGD to train a simple CNN for an image classification task.

##### **Code Example**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# 1. Define model (use pretrained ResNet18)
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # Assume 10 classes

# 2. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # SGD optimizer

# 3. Prepare data (example data)
inputs = torch.rand(16, 3, 224, 224)  # Batch of images (batch, channels, height, width)
labels = torch.randint(0, 10, (16,))   # Random labels

# 4. Training step
model.train()
optimizer.zero_grad()  # Clear gradients
outputs = model(inputs)  # Forward pass
loss = criterion(outputs, labels)  # Compute loss
loss.backward()  # Backpropagation
optimizer.step()  # Update parameters

print(f"Loss: {loss.item()}")
```

## 📖 **Code Explanation**

* **Model**: Pretrained ResNet18 with the last layer replaced to fit a 10-class classification task.
* **Optimizer**: `optim.SGD` initialization includes:

  * `model.parameters()`: Learnable parameters of the model.
  * `lr`: Learning rate, set to 0.01.
  * `momentum`: Momentum coefficient, set to 0.9 for improved convergence stability.
* **Training**: Standard forward pass, loss computation, backpropagation, and parameter update.
* **Data**: Example uses random images and labels. In practice, replace with real datasets (e.g., CIFAR-10).



## 📖 **Notes**

1. **Hyperparameters**:

   * `lr`: Learning rate must be tuned for the task. Common range: 0.001–0.1.
   * `momentum`: Typically set to 0.9; can be set to 0 if not using momentum.
2. **Data Preprocessing**:

   * Images should be normalized (e.g., mean \[0.485, 0.456, 0.406], std \[0.229, 0.224, 0.225]).
   * Use `torchvision.transforms` for preprocessing.
3. **Device**:

   * If using GPU, move model and data to GPU: `model.cuda()`, `inputs.cuda()`, `labels.cuda()`.
4. **Practical Usage**:

   * Replace example data with real datasets (e.g., `torchvision.datasets.CIFAR10`).
   * Add DataLoader and multi-epoch training loop.



#### 📖 **Summary**

SGD is a simple and efficient optimizer that updates parameters using stochastic gradients, making it suitable for many deep learning tasks. PyTorch’s `optim.SGD` is easy to use, requiring only learning rate and momentum as inputs. Compared with Adadelta, SGD needs manual hyperparameter tuning but is computationally lighter, making it useful for quick experiments.


