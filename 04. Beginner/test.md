# Fully Connected Neural Network (Mathematical Description)

## 1. Network Structure

A typical fully connected neural network consists of several **layers**:

* Input layer  
* One or more hidden layers  
* Output layer  

In a fully connected structure, **each neuron in a given layer is connected to all neurons in the previous layer**.

---

## 2. Mathematical Notation

Let:

* Input vector:

$$
\mathbf{x} \in \mathbb{R}^{d}
$$

* The $l$-th layer has $n_l$ neurons, with output:

$$
\mathbf{h}^{(l)} \in \mathbb{R}^{n_l}
$$

* Weight matrix and bias:

$$
\mathbf{W}^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}, \quad \mathbf{b}^{(l)} \in \mathbb{R}^{n_l}
$$

* Activation function:

$$
\sigma(\cdot)
$$

---

## 3. Forward Propagation

The input layer is defined as:

$$
\mathbf{h}^{(0)} = \mathbf{x}
$$

For the $l$-th layer ($l=1,2,\dots,L$):

1. **Linear transformation:**

$$
\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}
$$

2. **Nonlinear activation:**

$$
\mathbf{h}^{(l)} = \sigma\left(\mathbf{z}^{(l)}\right)
$$

Finally, the output layer result is:

$$
\hat{\mathbf{y}} = \mathbf{h}^{(L)}
$$

---

## 4. Loss Function

During training, given target output $\mathbf{y}$, common loss functions include:

* **Regression (MSE):**

$$
\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}) = \frac{1}{N}\sum_{i=1}^N \|\hat{\mathbf{y}}^{(i)} - \mathbf{y}^{(i)}\|^2
$$

* **Classification (Cross-Entropy):**

$$
\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}) = - \sum_{k=1}^K y_k \log \hat{y}_k
$$

---

## 5. Parameter Update (Backpropagation + Gradient Descent)

By backpropagation, we compute the gradients of the loss function with respect to the parameters:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}}
$$

Then update them using gradient descent or its variants (e.g., Adam, SGD, RMSProp):

$$
\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}
$$

$$
\mathbf{b}^{(l)} \leftarrow \mathbf{b}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}}
$$

where $\eta$ is the learning rate.

---

## Summary

In summary, a fully connected neural network can be abstracted as:

$$
\hat{\mathbf{y}} = f(\mathbf{x}; \Theta) = \sigma^{(L)}\Big(\mathbf{W}^{(L)} \sigma^{(L-1)}(\cdots \sigma^{(1)}(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}) \cdots ) + \mathbf{b}^{(L)}\Big)
$$

where $\Theta = \{\mathbf{W}^{(l)}, \mathbf{b}^{(l)} \mid l=1,\dots,L\}$ represents the set of model parameters.

 
