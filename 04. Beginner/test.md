好的 ✅，我已经把内容中的中文部分翻译成了英文，并且保留了 LaTeX 公式不变：

```markdown
Here is the mathematical description of a Fully Connected Neural Network (FCNN or MLP):

---

## 1. Network Structure

A typical fully connected neural network consists of several **layers**:

* Input layer
* One or more hidden layers
* Output layer

In a fully connected structure, **each neuron in one layer is connected to all neurons in the previous layer**.

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

The input layer is denoted as:

$$
\mathbf{h}^{(0)} = \mathbf{x}
$$

For the $l$-th layer ($l=1,2,\dots,L$):

1. **Linear transformation:**

$$
\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}
$$

2. **Non-linear activation:**

$$
\mathbf{h}^{(l)} = \sigma\left(\mathbf{z}^{(l)}\right)
$$

Finally, the output layer result is:

$$
\hat{\mathbf{y}} = \mathbf{h}^{(L)}
$$

---

## 4. Loss Function

During training, given the target output $\mathbf{y}$, common loss functions include:

* **Regression problem:** Mean Squared Error (MSE)

$$
\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}) = \frac{1}{N}\sum_{i=1}^N \|\hat{\mathbf{y}}^{(i)} - \mathbf{y}^{(i)}\|^2
$$

* **Classification problem:** Cross-Entropy Loss

$$
\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}) = - \sum_{k=1}^K y_k \log \hat{y}_k
$$

---

## 5. Parameter Update (Backpropagation + Gradient Descent)

Using backpropagation to compute gradients of the loss function with respect to parameters:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}}
$$

Then update parameters using gradient descent or its variants (such as Adam, SGD, RMSProp):

$$
\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}
$$

$$
\mathbf{b}^{(l)} \leftarrow \mathbf{b}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}}
$$

where $\eta$ is the learning rate.

---

✅ In summary, a fully connected neural network can be abstracted as:

$$
\hat{\mathbf{y}} = f(\mathbf{x}; \Theta) = \sigma^{(L)}\Big(\mathbf{W}^{(L)} \sigma^{(L-1)}(\cdots \sigma^{(1)}(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}) \cdots ) + \mathbf{b}^{(L)}\Big)
$$

where $\Theta = \{\mathbf{W}^{(l)}, \mathbf{b}^{(l)} \mid l=1,\dots,L\}$ are the model parameters.

---

Would you like me to also create a **diagram of the MLP structure** (including input layer, hidden layers, output layer, and connections) in Markdown/TikZ style?
```

要不要我直接帮你改成 **GitHub README 友好格式（带公式转图片保证显示）**？


