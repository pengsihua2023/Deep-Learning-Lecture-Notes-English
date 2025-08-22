# Fully Connected Neural Network (Mathematical Description)

A **fully connected network** (also called a **dense network**) consists of multiple layers of neurons.  
Each neuron in one layer is connected to **all** neurons in the next layer.

---

## Notation

- **Input vector**  
  $$
  \mathbf{x} \in \mathbb{R}^{d}
  $$
  where $d$ is the input dimension.

- **Layers**  
  Suppose the network has $L$ layers (excluding the input).  
  The $l$-th layer ($l = 1,2,\dots,L$) has $n_l$ neurons.

- **Parameters**  
  - Weight matrix for layer $l$:  
    $$
    W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}
    $$
  - Bias vector for layer $l$:  
    $$
    \mathbf{b}^{(l)} \in \mathbb{R}^{n_l}
    $$

- **Activation function (elementwise)**  
  $$
  \sigma(\cdot)
  $$

---

## Forward Propagation

1. **Pre-activation (linear transformation):**
   $$
   \mathbf{z}^{(l)} = W^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, 
   \quad \mathbf{a}^{(0)} = \mathbf{x}
   $$

2. **Activation:**
   $$
   \mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})
   $$

---

## Output

For the last layer $L$, the network output is:
$$
\mathbf{y} = \mathbf{a}^{(L)}
$$

- For **regression**: the last activation may be the identity function.  
- For **binary classification**: a sigmoid activation is often used.  
- For **multi-class classification**: a softmax activation is typically applied.

