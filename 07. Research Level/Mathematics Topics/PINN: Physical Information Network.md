## PINN: Physics-Informed Neural Networks

### Principles and Usage

#### **Introduction**

Physics-Informed Neural Networks (PINNs) are a neural network framework that combines deep learning with physical laws to solve partial differential equations (PDEs) or simulate physical systems. PINNs embed physical equations (such as governing equations, initial conditions, and boundary conditions) into the loss function of the neural network, and approximate the solution of PDEs by optimizing network parameters. Compared with traditional numerical methods (such as finite difference or finite element), PINNs do not require discretization of the grid and are suitable for high-dimensional or complex geometry problems.

#### **Principles**

1. **Core Idea**:

* PINNs use a neural network \$u(x,t;\theta)\$ (with parameters \$\theta\$) to approximate the solution \$u(x,t)\$ of a PDE.
* By including PDE residuals, initial conditions, and boundary conditions in the loss function, the network output is constrained to satisfy physical laws.
* The optimization goal is to minimize the loss function so that the network output approaches the true solution.

2. **PDE Form**: Consider a general PDE:

$$
\mathcal{N}[u(x,t)] = f(x,t), \quad (x,t) \in \Omega
$$

with initial and boundary conditions:

$$
u(x,0) = u_{0}(x), \quad u(x,t) \in \partial \Omega = g(x,t)
$$

Where:

* \$\mathcal{N}\$: the PDE operator (e.g. \$\frac{\partial u}{\partial t} + \frac{\partial^2 u}{\partial x^2} = 0\$).
* \$\Omega\$: the domain.
* \$u\_0, g\$: initial and boundary conditions.

3. **Loss Function**:
   The loss function of PINNs consists of three parts:

* **PDE Residual Loss:**

$$
L_{\text{PDE}} = \frac{1}{N_f} \sum_{i=1}^{N_f} \left| \mathcal{N}[u(x_i, t_i; \theta)] - f(x_i, t_i) \right|^2
$$

Calculated at sampled points \${x\_i, t\_i}\$.

* **Initial Condition Loss:**

$$
L_{\text{init}} = \frac{1}{N_i} \sum_{i=1}^{N_i} \left| u(x_i, 0; \theta) - u_0(x_i) \right|^2
$$

* **Boundary Condition Loss:**

$$
L_{\text{bc}} = \frac{1}{N_b} \sum_{i=1}^{N_b} \left| u(x_i, t_i; \theta) - g(x_i, t_i) \right|^2
$$

* **Total Loss:**

$$
L = \lambda_1 L_{\text{PDE}} + \lambda_2 L_{\text{init}} + \lambda_3 L_{\text{bc}}
$$

where \$\lambda\_1, \lambda\_2, \lambda\_3\$ are weights to balance the parts.

4. **Automatic Differentiation**:

* PINNs use neural network automatic differentiation (autograd) to compute derivatives in PDEs (such as \$\frac{\partial u}{\partial t}, \frac{\partial^2 u}{\partial x^2}\$) without explicit discretization.
* The network input is space-time coordinates \$(x,t)\$, and the output is the solution \$u(x,t)\$.

5. **Advantages**:

   * No need for mesh discretization, suitable for high-dimensional or complex geometries.
   * Flexible and easy to incorporate physical constraints.
   * Can combine small amounts of observational data with physical laws.

6. **Disadvantages**:

* Training may be slow, optimization of complex PDEs can be unstable.
* Loss weights \$\lambda\_i\$ require tuning.
* Accuracy for complex nonlinear PDEs may not match traditional numerical methods.

7. **Application Scenarios**:

   * Solving PDEs (e.g. heat conduction, wave equations, fluid dynamics).
   * Data-driven physical modeling (e.g. with experimental data).
   * Inverse problems (e.g. parameter estimation).

---

#### **PyTorch Usage**

The following is a minimal PyTorch code example showing how to use PINNs to solve the 1D Burgers equation (a nonlinear PDE), along with explanations.

##### **Problem Description**

**Burgers Equation:**

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}, 
\quad x \in [-1,1], t \in [0,1]
$$

**Initial Condition:**

$$
u(x,0) = -\sin(\pi x)
$$

**Boundary Condition:**

$$
u(-1,t) = u(1,t) = 0
$$

where \$\nu = \frac{0.01}{\pi}\$ is the viscosity coefficient.

---

##### **Code Example**

```python
# (same as original, unchanged)
```

##### **Code Explanation**

* **Model**:

  * Define a simple fully connected neural network with input \$(x,t)\$ and output \$u(x,t)\$.
  * Use \$\tanh\$ activation for smooth PDE solutions.

* **Loss Function**:

  * `compute_loss` computes three parts:

    * **PDE residual**: compute \$u\_t, u\_x, u\_{xx}\$ with `torch.autograd.grad` to form Burgers equation residual.
    * **Initial condition**: enforce

$$
u(x,0) \approx -\sin(\pi x).
$$

```
* **Boundary condition**: enforce  
```

$$
u(-1,t) = u(1,t) = 0.
$$

* The total loss is the weighted sum of all parts (weights set to 1 here, can be tuned).

* **Data**:

  * Randomly sample PDE points \$(x\_f,t\_f)\$, initial points \$(x\_i,t\_i)\$, and boundary points \$(x\_b,t\_b)\$.
  * Initial condition \$u\_i = -\sin(\pi x)\$, boundary condition \$u\_b = 0\$.

* **Training**:

  * Use Adam optimizer (well-suited for PINNs, SGD/Adagrad also possible).
  * Train 1000 iterations, print losses.

* **Testing**:

  * Predict the solution at \$t=0.5, x \in \[-1,1]\$, verify output shape.

---

#### **Notes**

1. **Hyperparameters**:

   * Number of sampled points \$(N\_f, N\_i, N\_b)\$: more points improve accuracy but increase computation cost.
   * Network structure: depth/width depends on PDE complexity.
   * Loss weights: may need tuning for complex PDEs.

2. **Automatic Differentiation**:

   * Use `requires_grad_(True)` to compute higher-order derivatives.
   * Ensure correct tensor shapes `[N,1]`.

3. **Optimizer**:

   * Adam is better than SGD/Adagrad for PINNs due to adaptivity.
   * L-BFGS can be tried for faster convergence.

4. **Applications**:

   * Replace Burgers equation with other PDEs (heat, wave).
   * Add observational data for inverse problems.
   * Use GPU for acceleration: `model.cuda(), x_f.cuda(), ...`.

5. **Visualization**:

   * Use matplotlib to plot predicted \$u(x,t)\$.

---

#### **Summary**

PINNs are a powerful method for solving PDEs or modeling physical systems by embedding physical equations into neural network loss functions. PyTorch implementation is straightforward, using autograd for PDE residuals and optimization with initial and boundary conditions. The example demonstrates basic application to Burgers equation.

---

## Multi-Dimensional PDEs, Inverse Problems, and Visualization

### More Detailed PINN Code Examples

Based on the previous introduction and 1D Burgers equation, here are more detailed examples covering:

* **Multi-dimensional PDEs**: e.g. 2D heat conduction equation (variant of 2D Laplace equation).
* **Inverse problems**: estimating unknown PDE parameters (e.g. diffusion coefficient) with observational data.
* **Visualization**: use Matplotlib for heatmaps and animations.

These examples are based on PyTorch, with full code including data generation, training loop, loss logging, and visualization. Assumes you have basic PyTorch and Matplotlib environment.

#### 1. **Multi-Dimensional PDE Example: 2D Heat Conduction**

2D heat conduction equation (simplified Laplace):

$$
\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0, 
\quad (x,y) \in [0,1] \times [0,1]
$$

**Boundary Conditions:**

* \$u(0,y) = 0, \quad u(1,y) = 0\$
* \$u(x,0) = 0, \quad u(x,1) = \sin(\pi x)\$

This is a steady-state problem (no time dimension), solved by minimizing residuals with PINN.

##### **Code Example**

```python
# (same as original, unchanged)
```

##### **Code Explanation**

* **Network**: 3 fully connected layers, Tanh activations for smoothness.
* **Loss**: PDE residual (second derivatives) and boundary condition loss, boundary weighted by 10.
* **Data**: Randomly sample interior and boundary points uniformly.
* **Training**: 5000 iterations, record loss curve.
* **Visualization**: Loss curve + 2D heatmap with `imshow`.

---

#### 2. **Inverse Problem Example: Parameter Estimation**

In Burgers equation, assume viscosity coefficient \$\nu\$ is unknown. Use observational data to estimate \$\nu\$. Add observation loss and treat \$\nu\$ as learnable.

##### **Code Example**

```python
# (same as original, unchanged)
```

##### **Code Explanation**

* **Network**: Add `self.nu` as learnable parameter (initial 0.1).
* **Loss**: Add observation loss `loss_obs` using simulated data (true solution + noise).
* **Training**: Record `nu_history`, estimate \$\nu\$.
* **Visualization**: Loss curve, \$\nu\$ estimation curve + animation of \$u(x,t)\$ (saved as GIF).

---

#### 3. **Notes and Extensions**

* **Multi-dimensional PDEs**: For 3D or higher, adjust input layer (e.g. `nn.Linear(3, ...)`) and ensure uniform sampling.
* **Inverse problems**: Observational data may come from experiments/simulations; for complex parameters (e.g. functional), use additional networks.
* **Visualization**: Use `FuncAnimation` for dynamic PDEs, heatmaps for steady-state.
* **Optimization**: If convergence is slow, try L-BFGS or more sample points.
* **Running**: Code requires PyTorch, NumPy, Matplotlib. Animations require imagemagick for GIF output.

