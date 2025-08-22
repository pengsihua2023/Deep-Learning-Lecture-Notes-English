Here’s a standard **mathematical description of a fully connected neural network (FCN)** in English:

---

### Structure

A fully connected network (also called a **dense network**) consists of multiple layers of neurons. Each neuron in one layer is connected to **all** neurons in the next layer.

---

### Notation

* Input vector:

  $$
  \mathbf{x} \in \mathbb{R}^{d}
  $$

  where $d$ is the input dimension.

* Layers:
  Suppose the network has $L$ layers (excluding the input).
  The $l$-th layer ($l = 1,2,\dots,L$) has $n_l$ neurons.

* Weights and biases:

  * Weight matrix for layer $l$:

    $$
    W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}
    $$
  * Bias vector for layer $l$:

    $$
    \mathbf{b}^{(l)} \in \mathbb{R}^{n_l}
    $$

* Activation function (elementwise):

  $$
  \sigma(\cdot)
  $$

---

### Forward Propagation

The output of each layer is computed as follows:

1. **Pre-activation (linear transformation):**

   $$
   \mathbf{z}^{(l)} = W^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}
   $$

   where $\mathbf{a}^{(0)} = \mathbf{x}$ is the input.

2. **Activation:**

   $$
   \mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})
   $$

---

### Final Output

For the last layer $L$, the network output is:

$$
\mathbf{y} = \mathbf{a}^{(L)}
$$

Depending on the task:

* For **regression**, $\sigma$ at the last layer might be the identity function.
* For **binary classification**, often a sigmoid function.
* For **multi-class classification**, typically a softmax function.

---

👉 Do you want me to also include the **compact matrix form** of the whole network function, like

$$
f(\mathbf{x}; \theta) = \mathbf{a}^{(L)}
$$

with all parameters $\theta = \{W^{(l)}, \mathbf{b}^{(l)}\}_{l=1}^L$?
