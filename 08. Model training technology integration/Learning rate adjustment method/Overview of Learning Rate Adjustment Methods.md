

## Overview of Learning Rate Adjustment Methods

In deep learning, the **learning rate** is a key hyperparameter that controls the step size when the optimizer updates model parameters. A learning rate that is too high may cause the model to diverge or oscillate, while a learning rate that is too low may result in slow convergence or getting stuck in local optima. Therefore, learning rate adjustment methods are crucial in the optimization process, either dynamically or pre-defined, to improve training efficiency and model performance. Below is an overview of learning rate adjustment methods in deep learning, covering **non-adaptive** and **adaptive** approaches, as well as related strategies and implementations.

### 1. **Non-Adaptive Learning Rate Adjustment Methods**

Non-adaptive methods adjust the learning rate based on predefined rules or schedules, without relying on historical gradient information. These methods are often used with simple optimizers (such as SGD).

#### (1) **Fixed Learning Rate**

* **Description**: Uses a constant learning rate throughout training.
* **Advantages**: Simple and easy to implement, suitable for simple tasks.
* **Disadvantages**: Cannot adapt to complex optimization problems, prone to oscillations or slow convergence.
* **Applicable Scenarios**: Small datasets or tasks with simple model structures.
* **PyTorch Example**:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Fixed learning rate
```

#### (2) **Learning Rate Scheduler**

Adjusts the learning rate over time using predefined rules. Common schedulers include:

* **Step Decay**:

  * Learning rate decays by a fixed ratio after certain steps (epochs or iterations).

- Formula:

$$
\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}
$$

where \$\eta\_0\$ is the initial learning rate, \$\gamma\$ is the decay factor, and \$s\$ is the step size.

* PyTorch Example:

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```

* **Exponential Decay**:

  * Learning rate decays exponentially over time:

$$
\eta_t = \eta_0 \cdot e^{-kt}
$$

* Suitable for quickly decreasing the learning rate.
* PyTorch Example:

```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
```

* **Cosine Annealing**:

  * Learning rate decreases following a cosine function, gradually approaching a minimum value.
  * Suitable for exploring global optima and fine-tuning.
  * PyTorch Example:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

* **ReduceLROnPlateau**:

  * Reduces the learning rate when a validation metric (e.g., loss or accuracy) stops improving.
  * Dynamically adapts to training progress.
  * PyTorch Example:

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
```

* **Advantages**: Flexible, adjusts learning rate according to training progress; suitable for various tasks.
* **Disadvantages**: Requires manual tuning of scheduler parameters (e.g., step size, decay factor), often needs experimentation.
* **Applicable Scenarios**: Used with SGD or other non-adaptive optimizers, widely applied in image classification, object detection, etc.

#### (3) **Warm-up**

* **Description**: Gradually increases the learning rate during the early phase of training, starting from a small value to avoid initial oscillations.
* **Implementation**:

```
<img width="579" height="52" alt="image" src="https://github.com/user-attachments/assets/5a5065a6-9176-4415-925f-65b339a61c48" />  
```

```
 - Commonly used in large models or complex tasks (e.g., Transformers).  
```

* **PyTorch Example** (custom implementation):

```python
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min(epoch / 10.0, 1.0))
```

* **Advantages**: Stabilizes early training, suitable for large models.
* **Disadvantages**: Requires additional configuration of warm-up steps.



### 2. **Adaptive Learning Rate Adjustment Methods**

Adaptive methods compute learning rates dynamically for each parameter by analyzing gradient history (e.g., mean, variance). These methods are usually embedded in optimizers. Popular optimizers like Adam already include adaptive mechanisms. Below are common adaptive optimizers and their learning rate principles:

#### (1) **Adagrad**

* **Principle**: Adjusts learning rate based on the cumulative sum of squared gradients.

   <img width="348" height="89" alt="image" src="https://github.com/user-attachments/assets/0bbd0119-128c-4562-8d73-b0ff7e691116" />  

* **Features**: Learning rate decreases for frequently updated parameters; well-suited for sparse data.
* **Disadvantages**: Monotonic decrease in learning rate may cause premature stopping.
* **Applicable Scenarios**: Sparse feature optimization such as word embeddings in NLP.

#### (2) **RMSProp**

* **Principle**: Uses exponential moving average of squared gradients.

   <img width="639" height="89" alt="image" src="https://github.com/user-attachments/assets/820e3e07-39b7-4b2a-b8cc-aebf5953617f" />  

* **Features**: Prevents overly fast decay of learning rate in Adagrad; suitable for non-stationary problems.
* **Applicable Scenarios**: Sequence models such as RNNs.

#### (3) **Adam**

* **Principle**: Combines first-order momentum (mean of gradients) and second-order momentum (variance of gradients).

   <img width="599" height="132" alt="image" src="https://github.com/user-attachments/assets/02552104-9b48-4218-9298-41cd3605ab6d" />  

* **Features**: Highly adaptive, default parameters often work well, fast convergence.
* **Applicable Scenarios**: Almost all deep learning tasks (e.g., CNNs, Transformers).

#### (4) **AdamW**

* **Principle**: An improved version of Adam, introducing weight decay (a variant of L2 regularization).
* **Features**: Better suited for models requiring regularization, prevents overfitting.
* **Applicable Scenarios**: Large-scale models or tasks requiring regularization.

#### (5) **Adadelta**

* **Principle**: Improves Adagrad by using a sliding window for squared gradients, eliminating the need for manually setting a learning rate.
* **Features**: Avoids overly fast decay, more stable computation.
* **Applicable Scenarios**: Long training or complex models.

#### (6) **Nadam**

* **Principle**: Combines Adam with Nesterov momentum, anticipating gradient directions.
* **Features**: More precise convergence, suitable for complex optimization problems.

#### (7) **AMSGrad**

* **Principle**: Improves Adam by retaining the maximum second-order moment, avoiding convergence issues.
* **Features**: More stable in certain non-convex problems.



### 3. **Combining Schedulers with Adaptive Optimizers**

Although adaptive optimizers (e.g., Adam) already adjust learning rates dynamically, they can still be combined with schedulers for further optimization. Examples:

* **Cosine Annealing + Adam**: Commonly used in Transformer models, adjusting Adam’s initial learning rate with cosine annealing.
* **Warm-up + Adam**: Used in pretraining large models like BERT, starting with warm-up followed by a scheduler (e.g., linear decay).

**PyTorch Example** (Adam + Cosine Annealing):

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    scheduler.step()  # Update learning rate
```



### 4. **Recommendations for Choosing Learning Rate Adjustment Methods**

* **Simple tasks or large-scale data**: Use SGD + a scheduler (e.g., StepLR or ReduceLROnPlateau).
* **Complex non-convex problems**: Prefer Adam or AdamW for strong adaptability and fast convergence.
* **Sparse data**: Adagrad or SparseAdam are more suitable.
* **Large model training**: Combine Warm-up and Cosine Annealing (e.g., Transformer).
* **Experimental adjustments**: Try Nadam or AMSGrad to address Adam’s convergence issues.

In binary classification tasks, Adam is often the default choice due to its robustness to learning rate selection and fast convergence. If training is unstable, alternatives include:

* **AdamW**: Adds regularization, suitable for complex models.
* **SGD + StepLR**: Useful for large-scale data, where schedulers provide fine-grained control.
* **Cosine Annealing**: Enhances convergence quality when combined with Adam.



### 5. **Notes**

* **Initial Learning Rate**: Adaptive optimizers (e.g., Adam) are less sensitive to initial learning rate (commonly 0.001 or 0.0001), while non-adaptive methods (e.g., SGD) require careful tuning.
* **Hyperparameters**: Both schedulers and adaptive optimizers require tuning (e.g., decay rate, step size), best optimized experimentally.
* **Computation Overhead**: Adaptive optimizers (e.g., Adam) require storing momentum terms, increasing memory usage, while SGD is lighter.
* **Validation Monitoring**: When using ReduceLROnPlateau, ensure validation loss or metrics are monitored.



### Summary

Learning rate adjustment methods in deep learning can be divided into:

* **Non-adaptive**: Fixed learning rates or schedulers (e.g., Step Decay, Cosine Annealing, Warm-up), based on predefined rules.
* **Adaptive**: Dynamically adjusted learning rates using gradient history (e.g., Adam, RMSProp, Adagrad).

The two can also be combined (e.g., Adam + Cosine Annealing). Method selection depends on task complexity, data scale, and model type. Adam and AdamW are currently the most widely used methods, suitable for most deep learning tasks, while schedulers add flexibility for non-adaptive optimizers (like SGD).



