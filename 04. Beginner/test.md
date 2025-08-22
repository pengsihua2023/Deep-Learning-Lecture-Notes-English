## Mathematical Description

### 1. Network Structure

A typical fully connected neural network consists of several **layers**:

* Input layer  
* One or more hidden layers  
* Output layer  

In a fully connected structure, **each neuron in a given layer is connected to all neurons in the previous layer**.



### 2. Mathematical Notation

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



### 3. Forward Propagation

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



### 4. Loss Function

During training, given target output $\mathbf{y}$, common loss functions include:

* **Regression (MSE):**

$$
\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}) = \frac{1}{N}\sum_{i=1}^N \|\hat{\mathbf{y}}^{(i)} - \mathbf{y}^{(i)}\|^2
$$

* **Classification (Cross-Entropy):**

$$
\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}) = - \sum_{k=1}^K y_k \log \hat{y}_k
$$



### 5. Parameter Update (Backpropagation + Gradient Descent)

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



### Summary

In summary, a fully connected neural network can be abstracted as:

$$
\hat{\mathbf{y}} = f(\mathbf{x}; \Theta) = \sigma^{(L)}\Big(\mathbf{W}^{(L)} \sigma^{(L-1)}(\cdots \sigma^{(1)}(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}) \cdots ) + \mathbf{b}^{(L)}\Big)
$$

where $\Theta = \{\mathbf{W}^{(l)}, \mathbf{b}^{(l)} \mid l=1,\dots,L\}$ represents the set of model parameters.

=========================
Here’s a compact **mathematical description of a Convolutional Neural Network (CNN)**, with formulas expressed in **LaTeX** so you can paste them directly into a GitHub markdown file (enclosed in `$$ ... $$` for display math mode).

---

## **Convolution Layer**

Given an input tensor $X \in \mathbb{R}^{H \times W \times C_{in}}$ (height, width, input channels), a convolutional kernel
$K \in \mathbb{R}^{k_h \times k_w \times C_{in} \times C_{out}}$, and bias $b \in \mathbb{R}^{C_{out}}$, the convolution operation is:

$$
Y_{i,j,c} = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} \sum_{d=0}^{C_{in}-1} 
X_{i+m,\, j+n,\, d} \, K_{m,n,d,c} \; + \; b_c
$$

where

* $(i,j)$ are spatial positions in the output,
* $c$ is the output channel index,
* $k_h, k_w$ are kernel height and width.

---

## **Activation Function**

After convolution, a nonlinear activation (e.g., ReLU) is applied:

$$
Z_{i,j,c} = \sigma \big( Y_{i,j,c} \big), \quad 
\sigma(x) = \max(0, x)
$$

---

## **Pooling Layer**

For max-pooling with window size $p \times p$:

$$
P_{i,j,c} = \max_{\substack{0 \leq m < p \\ 0 \leq n < p}}
Z_{\,i \cdot p + m,\, j \cdot p + n,\, c}
$$

---

## **Fully Connected Layer**

Flatten pooled features into a vector $\mathbf{p} \in \mathbb{R}^N$.
With weight matrix $W \in \mathbb{R}^{M \times N}$ and bias $\mathbf{b} \in \mathbb{R}^M$:

$$
\mathbf{f} = W \mathbf{p} + \mathbf{b}
$$

---

## **Output (Softmax for Classification)**

For $M$ classes, the probability of class $j$ is:

$$
\hat{y}_j 
= \frac{\exp(f_j)}{\sum_{k=1}^{M} \exp(f_k)}
$$  
  
$$
\hat{y_j} = \frac{\exp(f_j)}{\sum_{k=1}^{M} \exp(f_k)}
$$


---

✅ You can copy this block directly into a **GitHub README.md** and the math will render properly if GitHub’s math rendering is enabled (via KaTeX/MathJax).

Would you like me to also provide a **full end-to-end CNN pipeline equation** (input → conv → activation → pooling → fully connected → softmax) in a single compact formula, or keep it modular like above?




