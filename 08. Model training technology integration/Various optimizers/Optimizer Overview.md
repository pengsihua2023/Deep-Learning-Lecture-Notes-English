# Optimizer Overview

In deep learning, besides the **Adam optimizer**, there are many other optimizers that are widely used. Each optimizer has its own unique characteristics and applicable scenarios, suitable for different types of problems and models. Below is a list of common optimizers in deep learning with brief descriptions:

## 📖 1. **SGD (Stochastic Gradient Descent)**

* **Description**: The most basic optimizer, updating parameters in the opposite direction of the gradient. Optionally, momentum can be added to accelerate convergence.
* **Formula**:

$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta J(\theta)
$$

With momentum:

$$
v_t = \beta v_{t-1} + (1 - \beta)\nabla_\theta J(\theta), \quad \theta_{t+1} = \theta_t - \eta v_t
$$

* **Characteristics**:

  * Simple and straightforward, suitable for convex optimization problems.
  * Converges slowly, prone to local minima or saddle points.
  * Requires manual adjustment of learning rate, may need a scheduler.
* **Use Case**: Works well with large-scale data, SGD+momentum is often effective, especially in image classification tasks.
* **PyTorch Example**:

  ```python
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  ```

## 📖 2. **RMSProp (Root Mean Square Propagation)**

* **Description**: Adjusts the learning rate using the exponential moving average of squared gradients, suitable for non-stationary objectives.
* **Formula**:

<img width="518" height="60" alt="image" src="https://github.com/user-attachments/assets/931d190c-0fba-4a8b-a7f3-dd21cc9a576b" />

* **Characteristics**:

  * Adapts learning rate, good for non-convex problems.
  * Reduces update magnitude for large gradients, avoiding oscillations.
  * Hyperparameters (e.g., \$\rho\$ and \$\epsilon\$) require careful tuning.


* **Use Case**: Commonly used in RNNs and sequence models.

  * **PyTorch Example**:

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

### 📖 3. **Adagrad (Adaptive Gradient Algorithm)**

* **Description**: Adjusts the learning rate adaptively based on the historical sum of squared gradients, suitable for sparse data.
* **Formula**:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\sum_{i=1}^t g_i^2} + \epsilon} g_t
$$

* **Characteristics**:

  * Learning rate decreases over time for frequently updated parameters, good for sparse features.
  * Learning rate decreases monotonically, may stop learning prematurely.
* **Use Case**: Sparse data (e.g., word embeddings in NLP).
* **PyTorch Example**:

  ```python
  optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
  ```

## 📖 4. **Adadelta**

* **Description**: An improvement of Adagrad, using a moving window of squared gradients to avoid infinite decay of the learning rate.

* **Formula**:

  <img width="536" height="66" alt="image" src="https://github.com/user-attachments/assets/f57312e5-ad0f-4107-9760-3b2fa0ca1a04" />

* **Characteristics**:

  * No need to manually set learning rate.
  * More suitable for long training, alleviates Adagrad’s aggressive decay.

* **Use Case**: When robustness and stability are required.

* **PyTorch Example**:

  ```python
  optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9)
  ```

## 📖 5. **AdamW (Adam with Weight Decay)**

* **Description**: A variant of Adam, with weight decay (an improved form of L2 regularization), better for regularized models.
* **Characteristics**:

  * Applies weight decay directly to parameters, not gradients.
  * More suitable for complex models requiring regularization.
* **Use Case**: Widely used in deep learning tasks, especially with regularization needs.
* **PyTorch Example**:

  ```python
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
  ```

## 📖 6. **AMSGrad**

* **Description**: A variant of Adam, fixes convergence issues by keeping the maximum of past squared gradients.
* **Characteristics**:

  * Improves Adam’s second-moment calculation, avoids overly aggressive learning rate decay.
  * More stable in some non-convex problems.
* **Use Case**: Try when Adam does not converge well.
* **PyTorch Example**:

  ```python
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
  ```

## 📖 7. **Nadam**

* **Description**: Combines Adam with Nesterov momentum, “anticipating” gradient updates.
* **Characteristics**:

  * Uses momentum more precisely than Adam, suitable for complex problems.
  * Slightly higher computational cost.
* **Use Case**: When faster convergence is needed.
* **PyTorch Example**:

  ```python
  optimizer = torch.optim.NAdam(model.parameters(), lr=0.002)
  ```

## 📖 8. **L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)**

* **Description**: A second-order optimization algorithm, based on quasi-Newton methods, suitable for small-batch data.
* **Characteristics**:

  * Approximates the Hessian matrix with limited memory, good for small datasets.
  * Higher computational cost, less efficient with large batches.
* **Use Case**: Small datasets or tasks requiring high precision optimization.
* **PyTorch Example**:

  ```python
  optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01)
  ```

## 📖 9. **Rprop (Resilient Backpropagation)**

* **Description**: Updates parameters using only the sign of the gradient (not magnitude), suitable for noisy data.
* **Characteristics**:

  * Insensitive to gradient magnitude, robust.
  * Not suitable for deep networks, less efficient.
* **Use Case**: Simple networks or noisy optimization problems.
* **PyTorch Example**:

  ```python
  optimizer = torch.optim.Rprop(model.parameters(), lr=0.01)
  ```

## 📖 10. **SparseAdam**

* **Description**: A variant of Adam, designed for sparse tensors.
* **Characteristics**:

  * Updates only non-zero parameter momenta, saving memory.
  * Suitable for sparse data (e.g., embeddings).
* **Use Case**: Sparse feature optimization in NLP.
* **PyTorch Example**:

  ```python
  optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.001)
  ```

## 📖 11. **ASGD (Averaged Stochastic Gradient Descent)**

* **Description**: A variant of SGD, improves stability by averaging historical parameters.
* **Characteristics**:

  * Suitable for large-scale distributed training.
  * Converges slower but more stable.
* **Use Case**: Distributed training or when stable convergence is required.
* **PyTorch Example**:

  ```python
  optimizer = torch.optim.ASGD(model.parameters(), lr=0.01)
  ```

## 📖 Tips for Choosing an Optimizer

* **Default choice**: Adam or AdamW are the most commonly used, as their adaptiveness and robustness suit most tasks.
* **Need regularization**: Prefer AdamW.
* **Sparse data**: Consider Adagrad or SparseAdam.
* **Simple tasks or large batch data**: SGD+momentum may be more effective.
* **Complex non-convex problems**: Try Nadam or AMSGrad.
* **Small datasets**: L-BFGS may be more suitable.

## 📖 Optimizer Choice

Adam is a common choice for binary classification tasks because it is insensitive to the initial learning rate and converges quickly. If convergence issues arise in binary classification tasks, consider:

* **AdamW**: Adds weight decay, enhancing regularization.
* **SGD+momentum**: If the model is simple or the dataset is large.
* **RMSProp**: If the data is noisy or the model is sequence-based (e.g., RNN).

## 📖 Summary

There are many types of optimizers in deep learning, mainly including SGD, Adam, RMSProp, Adagrad, Adadelta, AdamW, Nadam, etc. Each optimizer differs in learning rate adjustment, convergence speed, and applicable scenarios. The choice of optimizer should be based on task type, dataset size, and model complexity.

