
# L-BFGS Optimizer

## Introduction

L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) is a Quasi-Newton optimization method. It accelerates gradient descent by approximating the inverse of the Hessian matrix, remaining efficient even in high-dimensional problems, while requiring only limited memory. Therefore, it is suitable for large-scale unconstrained optimization problems.

---

## Mathematical Description

1. **Optimization Objective**
   L-BFGS is used to solve unconstrained optimization problems:

$$
\min_{x \in \mathbb{R}^n} f(x)
$$

2. **Iterative Update**
   At the \$k\$-th iteration, the update rule is:

$$
x_{k+1} = x_k + \alpha_k p_k
$$

where \$\alpha\_k\$ is the step size (determined by line search), and \$p\_k\$ is the search direction.

3. **Search Direction**
   The search direction is determined by the approximate inverse Hessian:

$$
p_k = - H_k \nabla f(x_k)
$$

4. **Difference Vectors**
   Define increments of variables and gradients:

$$
s_k = x_{k+1} - x_k, \quad y_k = \nabla f(x_{k+1}) - \nabla f(x_k)
$$

5. **BFGS Update Formula**
   The classical BFGS inverse Hessian update formula is:

$$
H_{k+1} = \left(I - \frac{s_k y_k^T}{y_k^T s_k}\right) H_k \left(I - \frac{y_k s_k^T}{y_k^T s_k}\right) + \frac{s_k s_k^T}{y_k^T s_k}
$$

6. **Limited-memory Idea of L-BFGS**
   A full BFGS requires storing the entire matrix, which is too costly. L-BFGS only stores the most recent \$m\$ pairs of \$(s\_i, y\_i)\$ vectors, and uses **two-loop recursion** to efficiently compute the search direction without explicitly storing the inverse Hessian.

   **Two-loop recursion:**

   * Initialization: \$q = \nabla f(x\_k)\$
   * Backward loop: \$\alpha\_i = \frac{s\_i^T q}{y\_i^T s\_i}, \quad q \leftarrow q - \alpha\_i y\_i\$
   * Set initial matrix: \$H\_0^k = \frac{s\_{k-1}^T y\_{k-1}}{y\_{k-1}^T y\_{k-1}} I\$
   * Forward loop: \$\beta = \frac{y\_i^T r}{y\_i^T s\_i}, \quad r \leftarrow r + s\_i(\alpha\_i - \beta)\$
   * Final result: \$p\_k = -r\$



## Core Idea

1. **Basic Principle**

   * Newton's method uses the Hessian to provide second-order information, but the cost of storage and computation is too high in high-dimensional problems.
   * L-BFGS approximates the inverse Hessian using limited historical information, achieving efficient "limited-memory" updates.

2. **Algorithm Core**

   * Use the most recent \$m\$ pairs of \$(s\_i, y\_i)\$ to construct the approximate matrix.
   * Use this matrix to compute the descent direction, then combine with line search to obtain the next iterate.



## Features and Advantages

* **Memory-friendly**: Only stores limited historical information, suitable for high-dimensional optimization problems.
* **Fast convergence**: Utilizes second-order information, with convergence speed close to Newton’s method.
* **Wide application**: Applied in logistic regression, conditional random fields (CRF), word embedding training (e.g., word2vec), and more.

---

## Example Application

Using L-BFGS in PyTorch:

```python
import torch
from torch.optim import LBFGS

model = ...  # define the model
optimizer = LBFGS(model.parameters(), lr=0.1)

def closure():
    optimizer.zero_grad()
    loss = loss_fn(model(input), target)
    loss.backward()
    return loss

for i in range(20):
    optimizer.step(closure)
```

> Note: The `closure` function must be defined, because L-BFGS evaluates the objective function and gradient multiple times per iteration.

---

## Applicable Scenarios and Limitations

* **Applicable**: Optimization problems with small to medium-scale dense data.
* **Limitations**:

  * May get stuck in local optima for non-convex optimization problems.
  * Limited use in large-scale deep neural networks, as computation and storage are still relatively expensive; SGD and Adam are more common.

